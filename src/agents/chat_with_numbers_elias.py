import asyncio
import difflib
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from logging import INFO, Formatter, StreamHandler, getLogger
from typing import Any, Dict, List, Optional, Tuple

# import asyncpg
import duckdb
import pandas as pd
import pendulum
from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from src.general.chat.models import (
    AssistantMessage,
    ComponentLLMUsage,
    LLMUsage,
    MessageHistory,
    MessageHistoryForLLM,
    MessageForLLM,
)
from src.numbers.config import Config
from src.numbers.graphs.api_caller import run_api_caller
from src.numbers.graphs.optimisation import run_optimisation_graph
from src.numbers.graphs.set_points import set_points
from src.numbers.models.query.intent import QueryIntentOutputElias
from src.numbers.models.query.query_type import QueryParams, QueryParamsDefault
from src.numbers.state import Result, State
from src.numbers.utils.api import historical, keys
from src.numbers.utils.api.keys import get_object_metadata, load_fallback_object_data
from src.numbers.utils.prompting.load_prompt import render_prompt

# Configure logging
logger = getLogger(__name__)

# Remove any existing handlers to prevent duplicates
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

logger.setLevel(INFO)

# Create console handler with a higher log level
console_handler = StreamHandler()
console_handler.setLevel(INFO)
console_formatter = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)

# Add the handler to the logger
logger.addHandler(console_handler)

# Prevent duplicate logging by disabling propagation to parent loggers
# logger.propagate = False

# Setup database connection
cnx = sqlite3.connect("src/petex_mocked_data.sqlite")

# Setup model
model = OpenAIModel(model_name="gpt-4o-mini", provider=OpenAIProvider())


class TimeSpan(BaseModel):
    """Model representing a time span for queries."""

    time_started: datetime = Field(..., description="Start date of the query time span")
    time_ended: datetime = Field(..., description="End date of the query time span")
    time_started_str: Optional[str] = Field(None, description="Formatted start date")
    time_ended_str: Optional[str] = Field(None, description="Formatted end date")

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def format_dates(self):
        """Format datetime objects as strings using pendulum's LLLL format."""
        if self.time_started and not self.time_started_str:
            self.time_started_str = pendulum.instance(self.time_started).format("LLLL")
        if self.time_ended and not self.time_ended_str:
            self.time_ended_str = pendulum.instance(self.time_ended).format("LLLL")
        return self


class AggregationType(str, Enum):
    """Enum representing the type of aggregation to be performed."""

    SUM = "Sum"
    MEAN = "Avg"
    MAX = "Max"
    MIN = "Min"
    COUNT = "Count"


class Aggregation(BaseModel):
    """Model representing an aggregation for queries."""

    aggregation_type: Optional[AggregationType] = Field(
        None, description="The type of aggregation to be performed"
    )
    aggregation_period: Optional[str] = Field(
        None,
        description="The period of the aggregation. Can be one of day, month, year",
    )
    aggregation_interval: Optional[str] = Field(
        None, description="The value of the aggregation period. e.g 1 for 1 day"
    )


class ObjectDataset(BaseModel):
    """Model representing a dataset for an object."""

    object_type_property: str = Field(
        ..., description="The property of the object type that is being queried"
    )
    datasource: str = Field(
        "Preferred Rate",
        description="The datasource of the object type that is being queried",
    )
    sub_datasource: Optional[str] = Field(
        None, description="The sub-datasource of the object type that is being queried"
    )

    @model_validator(mode="after")
    def format_datasource(self):
        """Format the datasource to be None by default."""
        if self.datasource:
            self.datasource = "Preferred Rate"
        return self

    @model_validator(mode="after")
    def format_sub_datasource(self):
        """Format the sub-datasource to be None by default."""
        if self.sub_datasource:
            self.sub_datasource = None
        return self


class QueryInterpretation(BaseModel):
    """Base model for query interpretation."""

    object_datasets: List[ObjectDataset] = Field(
        ..., description="The datasets that are being queried"
    )
    time_span: TimeSpan = Field(..., description="The time span of the query")

    aggregation: Optional[Aggregation | Any] = Field(
        default_factory=Aggregation,
        description="The aggregation that is being performed",
    )


class QueryInterpretationHistorical(QueryInterpretation):
    """Model for historical query interpretation."""

    aggregation: Aggregation = Field(
        default_factory=Aggregation,
        description="The aggregation that is being performed",
    )


class QueryInterpretationEventOccurrence(QueryInterpretation):
    """Model for event occurrence query interpretation."""

    aggregation: None = None


class SQLQuery(BaseModel):
    """Model representing a generated SQL query."""

    query: str = Field(..., description="The SQL query string to execute")
    description: str = Field(..., description="Description of what this query does")
    params: Optional[dict] = Field(
        default_factory=dict, description="Parameters for the query"
    )


class SQLQueryResult(BaseModel):
    """Model representing the result of the SQL query generation."""

    queries: List[SQLQuery] = Field(
        default_factory=list, description="List of SQL queries generated"
    )


@dataclass
class PrepareMessageHistoryNode(BaseNode[State]):
    """Node for preparing message history from the state."""

    async def run(
        self, ctx: GraphRunContext[State]
    ) -> "QueryIntentNode | AppendAssistantResponseNode":
        try:
            logger.info("Starting PrepareMessageHistoryNode execution")

            message_history = ctx.state.message_history
            messages = message_history.messages
            if not messages:
                logger.error("Message history is empty")
                return AppendAssistantResponseNode(fallback=True)

            last_message = messages[-1]
            if last_message.role != "user":
                logger.error("Last message in history is not from user")
                return AppendAssistantResponseNode(fallback=True)

            # Get context from the last few messages
            n_message_context_limit = ctx.state.config.n_message_context_limit
            ctx.state.message_history_for_llm = (
                MessageHistoryForLLM.from_message_history(
                    message_history=message_history,
                    n_message_context_limit=n_message_context_limit,
                )
            )

            ctx.state.time_of_request = last_message.datetime_dt

            logger.info(
                f"Extracted query: {ctx.state.message_history_for_llm.model_dump_json()}",
            )

            return QueryIntentNode()

        except Exception as e:
            logger.error(
                f"Unexpected error in PrepareMessageHistoryNode: {str(e)}",
                exc_info=True,
            )
            return AppendAssistantResponseNode(fallback=True)


@dataclass
class QueryIntentNode(BaseNode[State, None, None]):
    """Node for parsing query parameters."""

    async def run(
        self, ctx: GraphRunContext[State]
    ) -> "IntentRouterNode | AppendAssistantResponseNode":
        model_query_intent = OpenAIModel(model_name="gpt-4o", provider=OpenAIProvider())

        try:
            logger.info("Starting QueryIntentNode execution")
            logger.info(f"Processing query: {ctx.state.message_history_for_llm.model_dump_json()}")

            # Create an agent to parse query parameters
            logger.debug("Initializing query intent agent")

            agent_query_intent = Agent(
                model_query_intent,
                deps_type=QueryIntentOutputElias,
                result_type=QueryIntentOutputElias,
                instrument=True,
                system_prompt=ctx.state.config.elias_query_intent_dev_prompt,
                # system_prompt=(
                #         """
                #         Analyse this user query and return the objects that are being queried.
                #         Only classify as production type data request if the query explicitly asks for PRODUCTION change of gas rate, oil rate, water rate of a field or a well.
                #         """
                # ),
            )

            logger.info("Running query intent analysis")
            response_query_intent = await agent_query_intent.run(ctx.state.message_history_for_llm.model_dump_json())
            query_intent = response_query_intent.output
            logger.info(f"Query intent analysis completed: {query_intent}")

            # Store the parsed query parameters in the state
            ctx.state.query_intent = query_intent
            ctx.state.query_intent_agent_metadata = response_query_intent.all_messages()
            logger.info("Successfully stored query intent results in state")

            return IntentRouterNode()
        except Exception as e:
            logger.error(
                f"Unexpected error in QueryIntentNode: {str(e)}", exc_info=True
            )
            return AppendAssistantResponseNode()


@dataclass
class IntentRouterNode(BaseNode[State]):
    """Node for routing to appropriate handlers based on query intent."""

    async def run(
        self, ctx: GraphRunContext[State]
    ) -> "QueryParamsNode | SmallTalkNode | GeneralInfoNode | AppendAssistantResponseNode":
        try:
            logger.info("Starting IntentRouterNode execution")

            # Get the query intent from state
            query_intent = ctx.state.query_intent
            if not query_intent:
                logger.error("Missing query intent data in IntentRouterNode")
                return AppendAssistantResponseNode(fallback=True)

            logger.debug(f"Routing based on query intent: {query_intent}")

            # Check for small talk
            if query_intent.small_talk:
                logger.info("Detected small talk request, routing to SmallTalkNode")
                return SmallTalkNode()

            if query_intent.data_request:
                logger.info("Detected data request, routing to QueryParamsNode")
                return QueryParamsNode()

            # Check for general info
            if query_intent.general_info:
                logger.info("Detected general info request, routing to SmallTalkNode")
                return SmallTalkNode()

            # # Default to normal query flow
            # logger.info("Proceeding with data query flow")
            # return QueryParamsNode()

            return AppendAssistantResponseNode(fallback=True)

        except Exception as e:
            logger.error(
                f"Unexpected error in IntentRouterNode: {str(e)}", exc_info=True
            )
            return AppendAssistantResponseNode(fallback=True)


@dataclass
class SmallTalkNode(BaseNode[State]):
    """Node for handling small talk conversations."""

    async def run(self, ctx: GraphRunContext[State]) -> "AppendAssistantResponseNode":
        try:
            logger.info("Starting SmallTalkNode execution")

            # Get the original query
            query = ctx.state.message_history_for_llm.model_dump_json()

            # Create a system prompt for generating small talk responses
            # system_prompt = """
            # You are a friendly assistant responding to casual conversation or small talk.
            # Provide a warm, engaging response that continues the conversation naturally.
            # Keep responses concise but friendly.
            # """
            system_prompt = ctx.state.config.elias_general_info_dev_prompt

            # Define a simple response structure for the small talk agent
            class SmallTalkResponse(BaseModel):
                response: str = Field(
                    ..., description="Friendly response to small talk"
                )

            # Create an agent for generating small talk responses
            small_talk_agent = Agent(
                model=OpenAIModel(model_name="gpt-4o-mini", provider=OpenAIProvider()),
                deps_type=SmallTalkResponse,
                result_type=SmallTalkResponse,
                system_prompt=system_prompt,
                instrument=True,
            )

            # Generate the small talk response
            logger.info("Generating small talk response")
            response = await small_talk_agent.run(query)

            # Store the response in state
            ctx.state.generated_response = response.output.response
            logger.info("Successfully generated small talk response")

            # Proceed to append the response to the message history
            return AppendAssistantResponseNode()

        except Exception as e:
            logger.error(f"Unexpected error in SmallTalkNode: {str(e)}", exc_info=True)
            return AppendAssistantResponseNode()


# Note: This node is currently not being used anywhere.
@dataclass
class GeneralInfoNode(BaseNode[State]):
    """Node for handling general information requests."""

    async def run(self, ctx: GraphRunContext[State]) -> "AppendAssistantResponseNode":
        try:
            logger.info("Starting GeneralInfoNode execution")

            # Create a system prompt for generating general information responses
            system_prompt = """
            You are an assistant specializing in general information about oil and gas operations.
            Provide clear, informative responses to general questions about:
            - Oil and gas industry concepts
            - Well operations
            - Field management
            - Production metrics
            - Basic petroleum engineering concepts
            
            Keep responses educational but concise.
            """

            # Define a response structure for the general info agent
            class GeneralInfoResponse(BaseModel):
                response: str = Field(
                    ..., description="Informative response to general question"
                )

            # Create an agent for generating general information responses
            general_info_agent = Agent(
                model=OpenAIModel(model_name="gpt-4o", provider=OpenAIProvider()),
                deps_type=GeneralInfoResponse,
                result_type=GeneralInfoResponse,
                system_prompt=system_prompt,
                instrument=True,
            )

            # Generate the general information response
            logger.info("Generating general information response")
            response = await general_info_agent.run(ctx.state.message_history_for_llm.model_dump_json())

            # Store the response in state
            ctx.state.generated_response = response.data.response

            logger.info("Successfully generated general information response")

            # Proceed to append the response to the message history
            return AppendAssistantResponseNode()

        except Exception as e:
            logger.error(
                f"Unexpected error in GeneralInfoNode: {str(e)}", exc_info=True
            )
            return AppendAssistantResponseNode(fallback=True)


@dataclass
class QueryParamsNode(BaseNode[State, None, None]):
    """Node for parsing query parameters."""

    async def run(
        self, ctx: GraphRunContext[State]
    ) -> "QueryInterpretationNode | AppendAssistantResponseNode":
        model_query_params = OpenAIModel(model_name="gpt-4o", provider=OpenAIProvider())

        try:
            logger.info("Starting QueryParamsNode execution")

            ### Fetching object types and properties from API
            (
                object_types,
                object_type_properties_dict,
                object_instances_dict,
            ) = await get_object_metadata(use_local_data=True)

            system_prompt = (
                (
                    "Analyse this conversation and answer the final user query. Return the objects that are being queried. "
                    f"Available object types and their instances: {object_instances_dict}. "
                    "Only include objects that are explicitly mentioned."
                ),
            )

            ### Commenting out agent call for now ###
            logger.info("Initializing query parameters agent")
            agent_query_params = Agent(
                model_query_params,
                deps_type=QueryParams,
                result_type=QueryParams,
                system_prompt=system_prompt,
                instrument=True,
            )

            logger.info("Running query parameters analysis")
            response_query_params = await agent_query_params.run(
                ctx.state.message_history_for_llm.model_dump_json(),
                # message_history=ctx.state.query_params_agent_metadata
                # if ctx.state.query_params_agent_metadata
                # else None,
            )
            
            query_params = response_query_params.output
            ctx.state.system_prompt_query_params = response_query_params.all_messages()

            # query_params = QueryParamsDefault()
            logger.info(f"Query parameters analysis completed: {query_params}")

            # Store the parsed query parameters in the state
            ctx.state.query_params = query_params
            ctx.state.object_type_properties_dict = object_type_properties_dict
            logger.info("Successfully stored query parameters in state")

            return QueryInterpretationNode()
        except Exception as e:
            logger.error(
                f"Unexpected error in QueryParamsNode: {str(e)}", exc_info=True
            )
            return AppendAssistantResponseNode(fallback=True)


@dataclass
class ObjectPropertyFuzzyMatchNode(BaseNode[State, None, None]):
    """Node for fuzzy matching object type properties and instances."""

    threshold: float = 0.5  # Configurable threshold for fuzzy matching

    def extract_potential_terms(self, query: str) -> List[str]:
        """
        Extract potential terms from the query text.
        Splits the query into words and considers word combinations.

        Args:
            query: The query text

        Returns:
            List of potential terms to match against
        """
        words = query.split()
        potential_terms = []

        # Consider single words
        potential_terms.extend(words)

        # Consider pairs of words
        for i in range(len(words) - 1):
            potential_terms.append(f"{words[i]} {words[i+1]}")

        # Consider triplets of words
        for i in range(len(words) - 2):
            potential_terms.append(f"{words[i]} {words[i+1]} {words[i+2]}")

        return potential_terms

    def get_best_match_with_score(
        self, query_term: str, available_terms: List[str]
    ) -> tuple[Optional[str], float]:
        """
        Find the best fuzzy match and its score for a term among available terms.

        Args:
            query_term: The term to match
            available_terms: List of available terms

        Returns:
            Tuple of (best matching term or None, match score)
        """
        matches_with_scores = []
        for term in available_terms:
            score = difflib.SequenceMatcher(
                None, query_term.lower(), term.lower()
            ).ratio()
            if score >= self.threshold:
                matches_with_scores.append((term, score))

        # Return the best match if any
        if matches_with_scores:
            return max(matches_with_scores, key=lambda x: x[1])
        return None, 0.0

    async def run(
        self, ctx: GraphRunContext[State]
    ) -> "QueryInterpretationNode | AppendAssistantResponseNode":
        try:
            logger.info("Starting ObjectPropertyFuzzyMatchNode execution")

            # Get the required data from state
            object_type_properties_dict = ctx.state.object_type_properties_dict
            query_params = ctx.state.query_params
            query_params_object_types = [o.object_type for o in query_params.objects] if query_params else []
            last_message = ctx.state.message_history_for_llm.last_message
            query = last_message.content if last_message and last_message.content else ""

            if not object_type_properties_dict:
                logger.error("Missing required state data for fuzzy matching")
                return AppendAssistantResponseNode()

            # Extract potential terms from query
            potential_terms = self.extract_potential_terms(query)
            logger.debug(f"Extracted potential terms from query: {potential_terms}")

            # Dictionaries to store matches and their scores
            object_type_properties_match = {}
            object_type_instances_match = {}

            # For each object type
            for obj_type, properties in object_type_properties_dict.items():
                if obj_type in query_params_object_types:
                    # Match properties
                    property_matches = {}
                    for term in potential_terms:
                        matched_property, score = self.get_best_match_with_score(
                            term, properties
                        )
                        if matched_property:
                            if matched_property in property_matches:
                                property_matches[matched_property] = max(
                                    property_matches[matched_property], score
                                )
                            else:
                                property_matches[matched_property] = score
                            logger.debug(
                                f"Matched property '{term}' to '{matched_property}' for {obj_type} with score {score:.2f}"
                            )

                    # Match instances for this object type
                    instance_matches = {}
                    # Get instances for this object type from query_params
                    for obj in query_params.objects:
                        if obj.object_type == obj_type:
                            available_instances = obj.object_instances
                            for term in potential_terms:
                                matched_instance, score = (
                                    self.get_best_match_with_score(
                                        term, available_instances
                                    )
                                )
                                if matched_instance:
                                    if matched_instance in instance_matches:
                                        instance_matches[matched_instance] = max(
                                            instance_matches[matched_instance], score
                                        )
                                    else:
                                        instance_matches[matched_instance] = score
                                    logger.debug(
                                        f"Matched instance '{term}' to '{matched_instance}' for {obj_type} with score {score:.2f}"
                                    )

                    # Only include object types that have matches
                    if property_matches:
                        # Sort matches by score in descending order
                        sorted_property_matches = dict(
                            sorted(
                                property_matches.items(),
                                key=lambda x: x[1],
                                reverse=True,
                            )
                        )
                        object_type_properties_match[obj_type] = sorted_property_matches

                    if instance_matches:
                        # Sort matches by score in descending order
                        sorted_instance_matches = dict(
                            sorted(
                                instance_matches.items(),
                                key=lambda x: x[1],
                                reverse=True,
                            )
                        )
                        object_type_instances_match[obj_type] = sorted_instance_matches

            # Store the matched properties and instances in state
            ctx.state.object_type_properties_match = object_type_properties_match
            ctx.state.object_type_instances_match = object_type_instances_match
            logger.info(
                f"Successfully completed fuzzy matching of properties: {object_type_properties_match}"
            )
            logger.info(
                f"Successfully completed fuzzy matching of instances: {object_type_instances_match}"
            )

            return QueryInterpretationNode()

        except Exception as e:
            logger.error(
                f"Unexpected error in ObjectPropertyFuzzyMatchNode: {str(e)}",
                exc_info=True,
            )
            return AppendAssistantResponseNode()


@dataclass
class QueryInterpretationNode(BaseNode[State, None, None]):
    """Node for interpreting queries."""

    async def run(
        self, ctx: GraphRunContext[State]
    ) -> "ObjectImputationNode | AppendAssistantResponseNode":
        try:
            logger.info("Starting QueryInterpretationNode execution")
            query_params = ctx.state.query_params
            logger.debug(f"Processing query parameters: {query_params}")

             # Ensure we have object properties - get from state or fetch if missing
            object_properties_available = {}

            if (
                hasattr(ctx.state, "object_type_properties_dict")
                and ctx.state.object_type_properties_dict
            ):
                logger.debug("Using object properties from state")
                object_properties_available = ctx.state.object_type_properties_dict
            else:
                logger.warning(
                    "object_type_properties_dict not found in state, fetching metadata"
                )
                try:
                    _, object_properties_available, _ = await get_object_metadata(
                        use_local_data=True
                    )
                    logger.info(
                        "Successfully fetched object properties from API or fallback"
                    )
                except Exception as e:
                    logger.error(f"Error fetching object metadata: {str(e)}")
                    _, object_properties_available, _ = load_fallback_object_data()
                    logger.warning("Using fallback data for object properties")

            # Hard code properties for UGM demo as a final fallback
            if not object_properties_available:
                logger.warning("Using hardcoded properties as fallback")
                object_properties = [
                    "Water Rate",
                    "Oil Rate",
                    "Gas Rate",
                    "Pump Frequency",
                    "Choke 1 Size",
                    "Wellhead Pressure",
                    "Gas Lift Gas rate",
                ]
                object_properties_available = {
                    object_type: object_properties for object_type in ["WELL", "FIELD"]
                }

            system_prompt = f"""
                Analyse this conversation and answer only the final user query. Return a structured query interpretation. 
                Object types and their available properties available are: {object_properties_available}
                The date today is {pendulum.parse('2025-04-08').format("LLLL")}
                If no time range is specified, use the previous 3 days.
                """

            structured_response = QueryInterpretation
            logger.debug("Using QueryInterpretation as structured response type")

            logger.info("Initializing interpretation agent")
            agent = Agent(
                model=OpenAIModel(model_name="gpt-4o-mini", provider=OpenAIProvider()),
                deps_type=structured_response,
                result_type=structured_response,
                system_prompt=system_prompt,
                instrument=True,
            )

            logger.info("Running query interpretation")
            # print(f"Interpretation query: {ctx.state.query}")
            response = await agent.run(
                ctx.state.message_history_for_llm.model_dump_json(),
                # message_history=ctx.state.interpretation_agent_metadata
                # if ctx.state.interpretation_agent_metadata
                # else None,
            )
            interpretation = response.output
            logger.info(f"Query interpretation completed: {interpretation}")

            # Store the interpretation in the state
            ctx.state.interpretation = interpretation
            # ctx.state.interpretation_agent_metadata = response.all_messages()
            logger.info("Successfully stored interpretation in state")

            return ObjectImputationNode()
        except Exception as e:
            logger.error(
                f"Unexpected error in QueryInterpretationNode: {str(e)}", exc_info=True
            )
            return AppendAssistantResponseNode()


@dataclass
class ObjectImputationNode(BaseNode[State, None, None]):
    """Node for interpreting queries."""

    async def run(
        self, ctx: GraphRunContext[State]
    ) -> "QueryHandlerNode | AppendAssistantResponseNode":
        try:
            logger.info("Starting ObjectImputationNode execution")
            query_params = ctx.state.query_params
            logger.debug(f"Processing query parameters: {query_params}")

            # Fetch object instances from get_object_metadata function
            try:
                logger.info("Fetching object instances from API with fallback")
                _, _, object_instance_dict = await get_object_metadata(
                    use_local_data=True
                )
                logger.debug(
                    f"Retrieved instances for all object types: {object_instance_dict}"
                )
            except Exception as e:
                logger.error(f"Error fetching object metadata: {str(e)}")
                # Use fallback data directly if even the metadata fetch fails
                _, _, object_instance_dict = load_fallback_object_data()
                logger.info(f"Using complete fallback data for all object types")

            # Update object instances in query_params
            for o in query_params.objects:
                if o.object_type in object_instance_dict:
                    o.object_instances = object_instance_dict[o.object_type]
                    logger.debug(
                        f"Updated instances for {o.object_type}: {o.object_instances}"
                    )

            logger.info("Object imputation completed successfully")
            return QueryHandlerNode()
        except Exception as e:
            logger.error(
                f"Unexpected error in ObjectImputationNode: {str(e)}", exc_info=True
            )
            return AppendAssistantResponseNode()


@dataclass
class CollectAPIResultsNodeAsync(BaseNode[State, None, None]):
    """Async node for collecting API results using parallel execution."""

    async def run(
        self, ctx: GraphRunContext[State]
    ) -> "GenerateMetadataNode | AppendAssistantResponseNode":
        try:
            logger.info("Starting CollectAPIResultsNodeAsync execution")

            # # If table_name is specified, skip API calls and read directly using DuckDB
            # if ctx.state.table_name:
            #     logger.info(
            #         f"Table name '{ctx.state.table_name}' specified, reading directly with DuckDB"
            #     )
            #     try:
            #         logger.debug("Attempting to read data using DuckDB")
            #         # Use DuckDB to read from the table
            #         # Properly quote the table name to handle special characters like hyphens
            #         quoted_table_name = f'"{ctx.state.table_name}"'
            #         df = duckdb.query(f"SELECT * FROM {quoted_table_name}").to_df()
            #         logger.info(
            #             f"Successfully read {len(df)} rows from table {ctx.state.table_name}"
            #         )
            #         ctx.state.dataframe = df
            #     except Exception as e:
            #         logger.error(
            #             f"Failed to access table {ctx.state.table_name}: {str(e)}"
            #         )
            #         return AppendAssistantResponseNode()

            #     logger.info("Proceeding to GenerateMetadataNode with direct table data")
            #     return GenerateMetadataNode()

            # Get parameters from state
            query_params = ctx.state.query_params
            interpretation = ctx.state.interpretation
            logger.debug(f"Processing query parameters: {query_params}")
            logger.debug(f"Processing interpretation: {interpretation}")

            # Prepare API payloads
            payloads = []
            for object in query_params.objects:
                logger.debug(f"Processing object type: {object.object_type}")
                for instance in object.object_instances:
                    logger.debug(f"Processing instance: {instance}")
                    for payload in interpretation.object_datasets:
                        logger.debug(f"Processing payload: {payload}")

                        # Extract parameters for API call
                        api_payload = {
                            "object_type": object.object_type,
                            "object_instance": instance,
                            "object_type_property": payload.object_type_property,
                            "data_source": payload.datasource,
                            "sub_datasource": payload.sub_datasource,
                            "start_time": pendulum.instance(
                                interpretation.time_span.time_started
                            ).int_timestamp
                            * 1000,
                            "end_time": pendulum.instance(
                                interpretation.time_span.time_ended
                            ).int_timestamp
                            * 1000,
                            "aggregate": None,
                            "aggregate_period": None,
                            "aggregate_interval": None,
                        }
                        logger.debug(f"Created API payload: {api_payload}")
                        payloads.append(api_payload)

            # Store payloads in state for reference
            ctx.state.api_payloads = payloads
            logger.info(f"Created {len(payloads)} payloads for API calls")

            if not payloads:
                logger.warning("No payloads created for API calls")
                return AppendAssistantResponseNode()

            # Use the optimized api_caller to execute the API calls
            table_name = "query_data"
            logger.info(
                f"Running API calls using api_caller, will save to table: {table_name}"
            )

            updated_state = await run_api_caller(
                payloads=payloads, table_name=table_name, existing_state=ctx.state
            )

            # Copy relevant attributes from updated_state back to ctx.state
            for attr_name in dir(updated_state):
                if not attr_name.startswith("_"):  # Skip private attributes
                    setattr(ctx.state, attr_name, getattr(updated_state, attr_name))

            if len(ctx.state.dataframe) == 0:
                logger.warning("No API results returned")
                return AppendAssistantResponseNode()

            logger.info("Proceeding to GenerateMetadataNode")
            return GenerateMetadataNode()
        except Exception as e:
            logger.error(
                f"Unexpected error in CollectAPIResultsNodeAsync: {str(e)}",
                exc_info=True,
            )
            return AppendAssistantResponseNode()


@dataclass
class QueryHandlerNode(BaseNode[State]):
    """Node for handling different query types based on query intent."""

    def _update_state_from_graph(
        self, ctx: GraphRunContext[State], updated_state: State, source_graph: str
    ) -> None:
        """Helper method to update state attributes from another graph's state.

        Args:
            ctx: Current graph context
            updated_state: State returned from another graph
            source_graph: Name of the source graph for logging
        """
        # List of attributes to transfer if they exist and have values
        transferable_attrs = [
            "table_name",
            "interpretation",
            "dataframe",
            "metadata",
            "api_payloads",
            "result",
            "pivot_tables",
            "generated_response",
        ]

        for attr in transferable_attrs:
            if hasattr(updated_state, attr):
                value = getattr(updated_state, attr)
                if value is not None:
                    setattr(ctx.state, attr, value)
                    logger.info(f"Transferred {attr} from {source_graph} graph")

    async def run(
        self, ctx: GraphRunContext[State]
    ) -> "CollectAPIResultsNodeAsync | ExecuteSQLNode | ResponseNode | GenerateMetadataNode | AppendAssistantResponseNode":
        try:
            logger.info("Starting QueryHandlerNode execution")

            # Get the query intent from state
            query_intent = ctx.state.query_intent
            if not query_intent:
                logger.error("Missing query intent data")
                return AppendAssistantResponseNode()

            # Check that interpretation exists
            if not ctx.state.interpretation:
                logger.error(
                    "Missing interpretation data needed for specialized handlers"
                )
                return ObjectImputationNode()

            logger.debug(f"Processing query intent: {query_intent}")

            # Check if this is a data request
            if query_intent.data_request:
                # This is janky logic right now but will do for now until we refactor the prompting
                data_req = query_intent.data_request
                if data_req.production:
                    ctx.state.prompt_response_style = "production_summary"
                elif data_req.well_shut_ins:
                    ctx.state.prompt_response_style = "shut_in_list"
                elif data_req.set_points:
                    ctx.state.prompt_response_style = "set_point_changes"
                elif data_req.model_updates:
                    ctx.state.prompt_response_style = "model_updates"
                elif data_req.optimisation:
                    ctx.state.prompt_response_style = "optimisation_summary"
                elif data_req.ranking:
                    ctx.state.prompt_response_style = "ranking_summary"
                else:
                    ctx.state.prompt_response_style = "default"

                # Determine which specific data requests are present
                data_req = query_intent.data_request
                data_handlers = []

                # Check for production rates request
                if data_req.production:
                    logger.info("Detected production rates request")
                    from src.numbers.graphs.production import run_production_graph

                    try:
                        logger.info("BRANCHING TO PRODUCTION GRAPH")
                        updated_state = await run_production_graph(ctx.state)
                        self._update_state_from_graph(ctx, updated_state, "production")
                        data_handlers.append("production")

                        # Check for direct result to skip SQL generation
                        if ctx.state.results:
                            return ResponseNode()

                    except Exception as e:
                        logger.error(f"Error running production graph: {str(e)}")

                # Check for well shut-ins request
                if data_req.well_shut_ins:
                    logger.info("Detected well shut-ins request")
                    from src.numbers.graphs.shut_in import run_shut_in_graph

                    try:
                        logger.info("BRANCHING TO SHUT-IN GRAPH")
                        updated_state = await run_shut_in_graph(ctx.state)
                        self._update_state_from_graph(ctx, updated_state, "shut-in")
                        data_handlers.append("shut_in")

                        # Check for direct result to skip SQL generation
                        if ctx.state.results:
                            return ResponseNode()

                    except Exception as e:
                        logger.error(f"Error running shut-in graph: {str(e)}")

                if data_req.set_points:
                    logger.info("BRANCH OFF TO SET POINTS GRAPH")
                    updated_state = await set_points(
                        change_type="set_points", existing_state=ctx.state
                    )
                    self._update_state_from_graph(ctx, updated_state, "set-points")
                    return ResponseNode()

                if data_req.optimisation:
                    logger.info("BRANCH OFF TO OPTIMISATION GRAPH")
                    updated_state = await run_optimisation_graph(
                        existing_state=ctx.state
                    )
                    self._update_state_from_graph(ctx, updated_state, "optimisation")
                    return ResponseNode()

                if data_req.model_updates:
                    logger.info("BRANCH OFF TO MODEL UPDATES GRAPH")
                    updated_state = await set_points(
                        change_type="model_updates", existing_state=ctx.state
                    )
                    self._update_state_from_graph(ctx, updated_state, "model-updates")
                    return ResponseNode()

                # If we've handled at least one data request type, skip to SQL generation
                if data_handlers:
                    logger.info(f"Handled data requests: {', '.join(data_handlers)}")

                    # Generate metadata if needed before going to SQL generator
                    if not ctx.state.metadata and ctx.state.dataframe is not None:
                        logger.info(
                            "Generating metadata before proceeding to SQL generation"
                        )
                        return GenerateMetadataNode()

                    # Skip to SQL generation since we now have a table ready for querying
                    if ctx.state.pivot_tables:
                        return ResponseNode()
                    else:
                        return GenerateMetadataNode()

            # If not a data request or no handlers were triggered, proceed to normal interpretation
            logger.info("Proceeding with standard object imputation flow")
            return CollectAPIResultsNodeAsync()

        except Exception as e:
            logger.error(
                f"Unexpected error in QueryHandlerNode: {str(e)}", exc_info=True
            )
            return AppendAssistantResponseNode()


@dataclass
class GenerateMetadataNode(BaseNode[State, None, None]):
    """Node for generating metadata from the dataset."""

    async def run(
        self, ctx: GraphRunContext[State]
    ) -> "SQLGeneratorNode | AppendAssistantResponseNode":
        try:
            logger.info("Starting GenerateMetadataNode execution")
            # Generate metadata from the dataframe similar to generate_metadata function
            df = ctx.state.dataframe
            logger.debug(
                f"Processing dataframe with {len(df)} rows and {len(df.columns)} columns"
            )
            metadata = {}

            for col in df.columns:
                logger.debug(f"Processing column: {col}")
                series = df[col]
                col_meta = {"original_dtype": str(series.dtype)}

                # Check if numeric
                try:
                    logger.debug(f"Attempting numeric conversion for column: {col}")
                    numeric_series = pd.to_numeric(series, errors="coerce")
                    if numeric_series.notnull().sum() > 0:
                        col_meta["inferred_type"] = "numeric"
                        col_meta.update(numeric_series.describe().to_dict())
                        metadata[col] = col_meta
                        logger.debug(f"Column {col} identified as numeric")
                        continue
                except Exception as e:
                    logger.debug(f"Column {col} is not numeric: {str(e)}")

                # Check if date
                try:
                    logger.debug(f"Attempting date conversion for column: {col}")
                    date_series = pd.to_datetime(series, errors="coerce")
                    if (
                        date_series.notnull().sum() / len(series) >= 0.8
                    ):  # At least 80% parse as dates
                        col_meta["inferred_type"] = "datetime"
                        col_meta["min"] = date_series.min()
                        col_meta["max"] = date_series.max()
                        metadata[col] = col_meta
                        logger.debug(f"Column {col} identified as datetime")
                        continue
                except Exception as e:
                    logger.debug(f"Column {col} is not datetime: {str(e)}")

                # Else categorical
                logger.debug(f"Column {col} identified as categorical")
                col_meta["inferred_type"] = "categorical"
                col_meta["unique_values"] = series.dropna().unique().tolist()
                col_meta["num_unique"] = len(col_meta["unique_values"])
                metadata[col] = col_meta

            ctx.state.metadata = metadata
            logger.info(f"Metadata generation completed for {len(metadata)} columns")
            return SQLGeneratorNode()
        except Exception as e:
            logger.error(
                f"Unexpected error in GenerateMetadataNode: {str(e)}", exc_info=True
            )
            return AppendAssistantResponseNode()


@dataclass
class SQLGeneratorNode(BaseNode[State, None, None]):
    """Node for generating SQL queries."""

    async def run(
        self, ctx: GraphRunContext[State]
    ) -> "ExecuteSQLNode | AppendAssistantResponseNode":
        try:
            logger.info("Starting SQLGeneratorNode execution")
            # Use an agent to generate SQL queries
            original_query = ctx.state.message_history_for_llm.model_dump_json()
            metadata = ctx.state.metadata
            table_name = ctx.state.table_name
            if not table_name:
                table_name = "query_data"
                logger.warning(f"No table name in state, using default: {table_name}")

            sql_generator_system_prompt = f"""
                You are an expert SQL query generator specializing in system engineering and data integrity analysis.
                Your task is to generate SQL queries.
                Analyse this conversation and generate a sql query to answer the final user query.

                Table Metadata: {metadata}

                When generating SQL:
                - Use only the '{table_name}' table
                - For date fields, group by appropriate system time intervals
                - The datetime field format is: 2024-11-08 00:00:00
                - Always include object instance, object type property and unit fields.
                - Use datetime fields when it makes sense to do so.
                - When using date fields, always use CAST function to cast them to date first
                - Prioritize clarity and compatibility with DuckDB syntax

                The date today is: {pendulum.parse('2025-04-08').format("LLLL")}
            """

            logger.info("Initializing SQL generator agent")
            model = OpenAIModel(model_name="gpt-4o", provider=OpenAIProvider())
            agent_sql_generator = Agent(
                model,
                deps_type=SQLQueryResult,
                result_type=SQLQueryResult,
                system_prompt=sql_generator_system_prompt,
                instrument=True,
            )

            class InvalidRequest(BaseModel):
                """Response the user input didn't include enough information to generate SQL."""

                error_message: str

            @agent_sql_generator.result_validator
            async def validate_output(ctx: GraphRunContext[State], output: SQLQueryResult) -> SQLQueryResult:
                if isinstance(output, InvalidRequest):
                    return output

                # model often adds extraneous backslashes to SQL
                for query in output.queries:
                    query.query = query.query.replace("\\", "")
                    if not query.query.upper().startswith("SELECT"):
                        raise ModelRetry("Please create a SELECT query")

                try:
                    # Check if any tables are registered in DuckDB
                    registered_tables = duckdb.sql("SHOW TABLES").fetchall()
                    if not registered_tables:
                        logger.warning(
                            "No tables registered in DuckDB, skipping SQL validation"
                        )
                        return output

                    table_name = (
                        ctx.state.table_name
                        if hasattr(ctx.state, "table_name")
                        else "query_data"
                    )
                    # Check if the table exists
                    table_exists = (
                        duckdb.sql(
                            f"SELECT * FROM information_schema.tables WHERE table_name = '{table_name}'"
                        ).fetchone()
                        is not None
                    )

                    if not table_exists:
                        logger.warning(
                            f"Table '{table_name}' not found, skipping SQL validation"
                        )
                        return output

                    # Validate syntax with DuckDB
                    for query in output.queries:
                        duckdb.sql(f"EXPLAIN {query.query}")
                except Exception as e:
                    logger.warning(f"Query validation error: {e}")
                    # Don't fail with ModelRetry if validation fails due to missing tables
                    # Just return the output and let execution handle any issues
                    return output
                else:
                    return output

            # Run the agent to generate SQL queries
            logger.info("Generating SQL queries")
            response_query = await agent_sql_generator.run(
                original_query,
                # message_history=ctx.state.sql_query_agent_metadata
                # if ctx.state.sql_query_agent_metadata
                # else None,
            )
            sql_query = response_query.output
            logger.debug(f"Generated SQL queries: {sql_query}")

            # Store the SQL query result in the state
            ctx.state.sql_query = sql_query
            ctx.state.original_sql_query = original_query
            ctx.state.sql_metadata = metadata
            # ctx.state.sql_query_agent_metadata = response_query.all_messages()
            logger.info("Successfully stored SQL query results in state")
            return ExecuteSQLNode()
        except Exception as e:
            logger.error(
                f"Unexpected error in SQLGeneratorNode: {str(e)}", exc_info=True
            )
            return AppendAssistantResponseNode()


@dataclass
class ExecuteSQLNode(BaseNode[State]):
    """Node for executing SQL queries."""

    async def run(self, ctx: GraphRunContext[State]) -> "ResponseNode":
        query_data = ctx.state.dataframe

        try:
            logger.info("Starting ExecuteSQLNode execution")
            # Execute each query using the existing database connection
            results: list[pd.DataFrame] = []
            logger.debug(f"Executing {len(ctx.state.sql_query.queries)} SQL queries")

            # Get the table name from state
            table_name = ctx.state.table_name
            if not table_name:
                table_name = "query_data"
                logger.warning(f"No table name in state, using default: {table_name}")

            # Quote the table name to handle special characters like hyphens
            quoted_table_name = f'"{table_name}"'
            logger.info(f"quoted_table_name: {quoted_table_name}")
            logger.debug(f"Using quoted table name: {quoted_table_name}")

            for i, query in enumerate(ctx.state.sql_query.queries, 1):
                try:
                    logger.info(
                        f"Executing query {i}/{len(ctx.state.sql_query.queries)}"
                    )
                    logger.debug(f"Query: {query.query}")
                    logger.info(f"query: {query.query}")
                    # Make sure query uses the correct table name and quote it properly
                    # This is a simple approach - in a real implementation you might want to use a SQL parser
                    modified_query = query.query.replace(table_name, quoted_table_name)
                    logger.debug(
                        f"Using quoted table name '{quoted_table_name}', modified query: {modified_query}"
                    )
                    logger.info(f"modified_query: {modified_query}")

                    # Execute the SQL query
                    result_df = duckdb.query(
                        modified_query
                    ).to_df()  # returns a result dataframe
                    results.append(result_df)
                    logger.info(
                        f"Query {i} executed successfully, returned {len(result_df)} rows"
                    )
                except Exception as e:
                    logger.error(f"Error executing SQL query {i}: {str(e)}")
                    logger.error(f"Failed query: {query.query}")

            # loop over results and add title plus id
            for i, r in enumerate(results):
                last_message = ctx.state.message_history_for_llm.last_message
                title = last_message.content if last_message and last_message.content else ""
                ctx.state.results.append(Result(title=title, data=r))

            logger.info(f"All queries executed. Total results: {len(results)}")
            return ResponseNode()
        except Exception as e:
            logger.error(f"Unexpected error in ExecuteSQLNode: {str(e)}", exc_info=True)
            return ResponseNode()


@dataclass
class ResponseNode(BaseNode[State]):
    """Node for generating a response based on the last n messages and query results."""

    async def run(self, ctx: GraphRunContext[State]) -> "AppendAssistantResponseNode":
        try:
            n_messages = ctx.state.config.n_message_context_limit
            logger.info(
                f"Starting ResponseNode execution using last {n_messages} messages"
            )

            # Get the original query and results
            message_history_for_llm = ctx.state.message_history_for_llm
            results = ctx.state.results

            if not message_history_for_llm.last_message:
                logger.warning("No message available to generate response")
                return AppendAssistantResponseNode(fallback=True)

            if not results:
                logger.warning("No results available to generate response")
                return AppendAssistantResponseNode(fallback=True)

            style = ctx.state.prompt_response_style or "default"
            style_filename = f"{style}.md"
            style_instructions = render_prompt(style_filename, context={})

            response_system_prompt = render_prompt(
                template_filename="base_system_prompt.md",
                context={"style_instructions": style_instructions},
            )

            # Define a simple input structure for the agent
            class ResponseInput(BaseModel):
                relevant_data: list[Result]
                message_history: list[MessageForLLM]
                last_message: MessageForLLM


            # Define a simple response structure for the agent
            class ResponseOutput(BaseModel):
                response: str = Field(
                    ..., description="The generated response to the user's query"
                )
                tables_used: list[str] = Field(
                    default_factory=list,
                    description="The tables names used to answer the user's query",
                )

            logger.info("Initializing response agent")
            model = OpenAIModel(model_name="gpt-4o", provider=OpenAIProvider())
            agent_retrieval = Agent(
                model,
                deps_type=ResponseOutput,
                result_type=ResponseOutput,
                system_prompt=response_system_prompt,
                instrument=True,
            )

            logger.info("Generating response")
            response_node_input = ResponseInput(
                relevant_data=results,
                message_history=message_history_for_llm.message_history,
                last_message=message_history_for_llm.last_message,
            )
    
            start_timestamp = time.time()
            agent_response = await agent_retrieval.run(response_node_input.model_dump_json())
            end_timestamp = time.time()

            # Correctly access the response data from the agent result
            generated_response, generated_response_sources = (
                agent_response.output.response,
                agent_response.output.tables_used,
            )

            # Store the response in state for use by AppendAssistantResponseNode
            ctx.state.generated_response = generated_response
            ctx.state.generated_response_sources = generated_response_sources
            ctx.state.llm_usage = ComponentLLMUsage.create(
                model="gpt-4o",
                prompt_tokens=agent_response.usage().request_tokens,
                completion_tokens=agent_response.usage().response_tokens,
                completion_time_secs=end_timestamp - start_timestamp,
            )

            if generated_response_sources:
                ctx.state.generated_response += (
                    f"\n\nData Table Sources: {', '.join(generated_response_sources)}"
                )

            logger.info("Successfully generated response")
            return AppendAssistantResponseNode()

        except Exception as e:
            logger.error(f"Unexpected error in ResponseNode: {str(e)}", exc_info=True)
            ctx.state.generated_response = f"I found some data but encountered an error while generating a response: {str(e)}"
            return AppendAssistantResponseNode()


@dataclass
class AppendAssistantResponseNode(BaseNode[State, None, None]):
    """Node for appending assistant response to message history."""

    fallback: bool = field(default=False)
    debug_msg: str = field(default="")

    async def run(self, ctx: GraphRunContext[State]) -> End:
        try:
            logger.info("Starting AppendAssistantResponseNode execution")

            # Get the message list for easier access
            messages = ctx.state.message_history.messages

            if self.debug_msg:
                messages.append(AssistantMessage(
                    content=self.debug_msg,
                ))
                return End(None)

            if self.fallback:
                self._add_fallback_message(messages=messages)
                return End(None)

            # First, remove any previous data table messages
            # Keep track of indices to remove
            indices_to_remove = []
            for i, msg in enumerate(messages):
                # Check if this is a message containing a table
                if msg.role == "assistant" and hasattr(msg, "context") and msg.context:
                    try:
                        # Try to parse the context as JSON to check for dataframe_table
                        context_data = (
                            json.loads(msg.context)
                            if isinstance(msg.context, str)
                            else msg.context
                        )
                        if (
                            isinstance(context_data, dict)
                            and context_data.get("tool_name") == "dataframe_table"
                        ):
                            indices_to_remove.append(i)
                    except (json.JSONDecodeError, AttributeError):
                        # If we can't parse the context, just continue
                        pass

            # Remove the identified messages in reverse order to avoid index shifting
            for i in sorted(indices_to_remove, reverse=True):
                logger.info(f"Removing previous data table message at index {i}")
                messages.pop(i)

            # Create context information about data sources
            context_info = self._create_context_info(ctx)

            # Add the text response if it exists
            if ctx.state.generated_response:
                self._add_text_response(ctx, messages, context_info)
            # If no text response but results/pivot tables exist, process them separately
            elif ctx.state.results:
                self._process_results(ctx, messages, context_info)
            # Or process pivot tables if available
            elif ctx.state.pivot_tables:
                self._process_pivot_tables(ctx, messages, context_info)
            # Add fallback message if no results or response
            else:
                self._add_fallback_message(messages, context_info)

            # Return End() with None value to satisfy the constructor requirement
            return End(None)
        except Exception as e:
            logger.error(
                f"Unexpected error in AppendAssistantResponseNode: {str(e)}",
                exc_info=True,
            )
            self._add_error_message(ctx.state.message_history.messages, str(e))
            return End(None)

    def _create_context_info(self, ctx: GraphRunContext[State]) -> list:
        """Create context information about data sources from API payloads."""
        context_info = []
        if ctx.state.api_payloads:
            logger.info("Processing API payloads for context information")
            for payload in ctx.state.api_payloads:
                context_info.append(
                    {
                        "object_type": payload["object_type"],
                        "object_instance": payload["object_instance"],
                        "object_type_property": payload["object_type_property"],
                        "data_source": payload["data_source"],
                        "time_range": f"{pendulum.from_timestamp(payload['start_time']/1000).format('YYYY-MM-DD')} to {pendulum.from_timestamp(payload['end_time']/1000).format('YYYY-MM-DD')}",
                    }
                )
            # Store data provenance in state
            ctx.state.data_provenance = context_info
            logger.debug(f"Created context info from {len(context_info)} API payloads")
        return context_info

    def _add_text_response(
        self, ctx: GraphRunContext[State], messages: list, context_info: list
    ):
        """Add the text response with context to the message history."""
        # Create context information including SQL queries if available
        context_data = {}

        # Add SQL query information if available
        if ctx.state.sql_query and ctx.state.sql_query.queries:
            context_data["sql_queries"] = [
                {"query": query.query, "description": query.description}
                for query in ctx.state.sql_query.queries
            ]

        # Add provenance data if available
        if context_info:
            context_data["provenance"] = context_info

        # Add data tables information for collapsible "Data Table(s)" section in UI
        data_tables = []

        # Add results if available
        if ctx.state.results:
            for i, result in enumerate(ctx.state.results):
                if hasattr(result, "data") and not result.data.empty:
                    table_data = {
                        "title": result.title,
                        "data": result.data.to_json(
                            orient="records", date_format="iso"
                        ),
                    }
                    # Add description if available
                    if hasattr(result, "description") and result.description:
                        table_data["description"] = result.description
                    # Add SQL query info if available
                    if ctx.state.sql_query and i < len(ctx.state.sql_query.queries):
                        table_data["sql_query"] = {
                            "query": ctx.state.sql_query.queries[i].query,
                            "description": ctx.state.sql_query.queries[i].description,
                        }
                    data_tables.append(table_data)

        # Add pivot tables if available
        elif ctx.state.pivot_tables:
            for i, pivot_table in enumerate(ctx.state.pivot_tables):
                if not pivot_table.data.empty:
                    table_data = {
                        "title": pivot_table.title or f"Table {i+1}",
                        "data": pivot_table.data.to_json(
                            orient="records", date_format="iso"
                        ),
                    }
                    # Add description if available
                    if hasattr(pivot_table, "description") and pivot_table.description:
                        table_data["description"] = pivot_table.description
                    # Add SQL query info if available
                    if ctx.state.sql_query and i < len(ctx.state.sql_query.queries):
                        table_data["sql_query"] = {
                            "query": ctx.state.sql_query.queries[i].query,
                            "description": ctx.state.sql_query.queries[i].description,
                        }
                    data_tables.append(table_data)

        # Add data tables to context if any exist
        if data_tables:
            context_data["data_tables"] = data_tables

        messages.append(
            AssistantMessage(
                content=ctx.state.generated_response,
                context=json.dumps(context_data, indent=2),
                llm_usage=LLMUsage(
                    component="_add_text_response",
                    component_llm_usage=ctx.state.llm_usage,
                ),
            )
        )
        logger.info("Appended generated text response with context")

    def _get_sql_query_info(
        self, ctx: GraphRunContext[State], index: int
    ) -> Optional[dict]:
        """Get SQL query information for a result at the given index."""
        if ctx.state.sql_query and index < len(ctx.state.sql_query.queries):
            return {
                "query": ctx.state.sql_query.queries[index].query,
                "description": ctx.state.sql_query.queries[index].description,
            }
        return None

    def _add_dataframe_message(
        self,
        messages: list,
        title: str,
        dataframe: pd.DataFrame,
        context_info: list,
        sql_query_info: Optional[dict],
        description: str = "",
        index: int = 0,
    ):
        """Add a message for a DataFrame table with appropriate context."""
        if dataframe.empty:
            return

        # Convert DataFrame to JSON records format
        json_data = dataframe.to_json(orient="records", date_format="iso")

        # Construct context dictionary
        context = {
            "tool_name": "dataframe_table",
            "title": title,
            "data": json_data,
            "provenance": context_info,
        }

        # Add SQL query info if available
        if sql_query_info:
            context["sql_query"] = sql_query_info

        # Add description if available
        if description:
            context["description"] = description

        # Add the message
        messages.append(
            AssistantMessage(
                content=f"Data for: {title}",
                context=context,
            )
        )
        logger.info(f"Added table {index+1} with title '{title}' for rendering")

    def _process_results(
        self, ctx: GraphRunContext[State], messages: list, context_info: list
    ):
        """Process results and add messages for each result."""
        logger.info(f"Processing {len(ctx.state.results)} results for table rendering")

        for i, result_item in enumerate(ctx.state.results):
            sql_query_info = self._get_sql_query_info(ctx, i)

            # Handle DataFrame result
            if isinstance(result_item, pd.DataFrame):
                query_desc = ""
                if sql_query_info:
                    query_desc = sql_query_info.get("description", "")

                table_title = query_desc if query_desc else f"Query Result {i+1}"
                self._add_dataframe_message(
                    messages,
                    table_title,
                    result_item,
                    context_info,
                    sql_query_info,
                    index=i,
                )

            # Handle dictionary with title and data
            elif (
                isinstance(result_item, dict)
                and "title" in result_item
                and "data" in result_item
            ):
                result_df = result_item["data"]
                if isinstance(result_df, pd.DataFrame):
                    table_title = result_item["title"]
                    self._add_dataframe_message(
                        messages,
                        table_title,
                        result_df,
                        context_info,
                        sql_query_info,
                        index=i,
                    )

        logger.info("Successfully added assistant messages for result tables.")

    def _process_pivot_tables(
        self, ctx: GraphRunContext[State], messages: list, context_info: list
    ):
        """Process pivot tables and add messages for each table."""
        logger.info(
            f"Processing {len(ctx.state.pivot_tables)} pivot tables for rendering"
        )

        for i, pivot_table in enumerate(ctx.state.pivot_tables):
            if not pivot_table.data.empty:
                table_title = pivot_table.title or f"Table {i+1}"
                description = pivot_table.description
                sql_query_info = self._get_sql_query_info(ctx, i)

                self._add_dataframe_message(
                    messages,
                    table_title,
                    pivot_table.data,
                    context_info,
                    sql_query_info,
                    description,
                    i,
                )

        logger.info("Successfully added assistant messages for pivot tables.")

    def _add_fallback_message(self, messages: list, context_info: list = []):
        """Add a fallback message when no results or response is available."""
        logger.warning("No results or generated response to append.")

        # Create context data with provenance if available
        context_data = {}
        if context_info:
            context_data["provenance"] = context_info

        messages.append(
            AssistantMessage(
                content="I couldn't find any data matching your query or generate a text response. Please try a different query or check your parameters.",
                context=json.dumps(context_data, indent=2),
            )
        )

    def _add_error_message(self, messages: list, error_message: str):
        """Add an error message to the message history."""
        messages.append(
            AssistantMessage(
                content=f"I encountered an error processing your query: {error_message}. Please try again with a different query.",
                context="{}",
            )
        )


# Define the main function for processing queries
async def process_query(
    message_history: MessageHistory,
    config: Config = Config(),
    table_name: Optional[str] = None,
):
    """Process a natural language query about numbers data.

    Args:
        message_history: The message history containing the conversation
        table_name: Name of the table to use directly (skips API calls if specified)
    """

    # If a table_name wasn't provided, generate a session-specific one
    if not table_name:
        # Create a session-specific table name
        session_specific_table = f"query_data_{message_history.session_id}"
        logger.info(f"Generated session-specific table name: {session_specific_table}")
        table_name = session_specific_table

    logger.info(f"Processing query using table name: {table_name}")

    # Prepare state
    state = State(
        message_history=message_history,
        config=config,
        table_name=table_name,
    )

    # Create the graph with the correct return type
    graph = Graph[State, None](
        nodes=[
            PrepareMessageHistoryNode,
            QueryIntentNode,
            IntentRouterNode,
            SmallTalkNode,
            GeneralInfoNode,
            QueryParamsNode,
            ObjectPropertyFuzzyMatchNode,
            QueryInterpretationNode,
            QueryHandlerNode,
            ObjectImputationNode,
            CollectAPIResultsNodeAsync,
            GenerateMetadataNode,
            SQLGeneratorNode,
            ExecuteSQLNode,
            ResponseNode,
            AppendAssistantResponseNode,
        ]
    )

    # Run the graph
    try:
        await graph.run(start_node=PrepareMessageHistoryNode(), state=state)
        logger.info("Graph execution completed successfully")

        # The state now contains the updated message history
        return message_history, state

    except Exception as e:
        logger.error(f"Error executing graph: {str(e)}", exc_info=True)
        # Add error message to the message history
        message_history.messages.append(
            AssistantMessage(
                content=f"I encountered an error processing your query: {str(e)}. Please try again with a different query.",
                context={},
            )
        )
        return message_history, state


# Main function for compatibility with the API
async def chat_with_numbers_elias(message_history: MessageHistory):
    """Process a chat message and return the updated message history.

    Args:
        message_history: The message history containing the conversation
    """
    # Just return the result directly - no need to extract from GraphRunResult
    return await process_query(message_history=message_history)


# Example usage (commented out)
# async def main():
#     # Example message history
#     messages = [
#         AssistantMessage(content="Hello! How can I help you with your data?"),
#         UserMessage(content="What was the daily average Oil Rate of wells LE-01 and LE-02 before Jan 17?"),
#     ]
#     message_history = MessageHistory(session_id="test-session", messages=messages)
#
#     # Process the query
#     result = await chat_with_numbers_elias(message_history)
#     print(result)
#
# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())
