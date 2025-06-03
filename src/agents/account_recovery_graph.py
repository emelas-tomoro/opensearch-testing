import asyncio
import json
from dataclasses import dataclass, field
from enum import Enum
from logging import INFO, Formatter, StreamHandler, getLogger
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from pydantic import BaseModel, Field
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from opensearchpy import OpenSearch
from src.config.base_models import AccountParams, SearchResult, State
from src.utils.search_utils import GameAccountSearcher, QueryType

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

# Setup OpenAI client
client = OpenAI()

@dataclass
class PrepareMessageHistoryNode(BaseNode[State]):
    """Node for preparing message history from the state."""
    
    async def run(self, ctx: GraphRunContext[State]) -> "ExtractParamsNode":
        try:
            logger.info("Starting PrepareMessageHistoryNode execution")
            
            # Initialize searcher if not already done
            if not ctx.state.searcher:
                opensearch_client = OpenSearch(
                    hosts=[{'host': 'localhost', 'port': 9200, 'scheme': 'http'}],
                    http_auth=('admin', 'admin'),
                    use_ssl=False,
                    verify_certs=False,
                    ssl_show_warn=False,
                )
                ctx.state.searcher = GameAccountSearcher(opensearch_client)
            
            return ExtractParamsNode()
            
        except Exception as e:
            logger.error(f"Unexpected error in PrepareMessageHistoryNode: {str(e)}", exc_info=True)
            return End(None)

@dataclass
class ExtractParamsNode(BaseNode[State]):
    """Node for extracting parameters from user input."""
    
    async def run(self, ctx: GraphRunContext[State]) -> "SearchNode | End":
        try:
            logger.info("Starting ExtractParamsNode execution")
            
            # Get system prompt based on current state
            system_prompt = self._get_system_prompt(ctx.state)
            
            # Build conversation context
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(ctx.state.conversation_history)
            
            # Extract parameters using OpenAI
            response = client.responses.parse(
                model="gpt-4.1-mini",
                input=messages,
                text_format=AccountParams, 
            )
            
            # Parse the response
            try:
                params_dict = json.loads(response.output_text)
                new_params = AccountParams(**params_dict)
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Error parsing parameters: {str(e)}")
                return End(None)
            
            # Merge with existing parameters
            self._merge_params(ctx.state, new_params)
            
            logger.info(f"Extracted parameters: {new_params.model_dump(exclude_none=True)}")
            
            return SearchNode()
            
        except Exception as e:
            logger.error(f"Unexpected error in ExtractParamsNode: {str(e)}", exc_info=True)
            return End(None)
    
    def _get_system_prompt(self, state: State) -> str:
        """Generate system prompt based on current state."""
        trend_info = ""
        if state.search_history:
            last_hits = state.search_history[-1]['hits']
            trend = state.search_history[-1]['trend']
            trend_info = f"\nLast search returned {last_hits} results. Results are {trend}."
        
        return f"""
You are a game account search assistant helping a user find their missing account.
Your goal is to gather information systematically based on field importance rankings.

Field Rankings (1=most important, 5=least important):
{json.dumps(state.field_rankings, indent=2)}

Current iteration: {state.current_iteration + 1}/{state.max_iterations}
{trend_info}

Instructions:
1. Extract any account information from the user's message.
2. If one of the fields seem vague or generic, don't be afraid to prompt again for the same information.
3. If results are decreasing significantly, ask more specific clarifying questions
4. Focus on the highest-ranked fields first (name is most important)
5. Ask one focused question at a time to refine the search
6. Be conversational and helpful

Current known information:
{self._format_current_params(state)}

Please respond with a JSON object containing only the extracted parameters. Use null for any fields where no information was found.
"""
    
    def _format_current_params(self, state: State) -> str:
        """Format current parameters for display."""
        params_dict = state.current_params.model_dump(exclude_none=True)
        if not params_dict:
            return "None collected yet"
        return json.dumps(params_dict, indent=2)
    
    def _merge_params(self, state: State, new_params: AccountParams) -> None:
        """Merge new parameters with existing ones."""
        current_dict = state.current_params.model_dump()
        new_dict = new_params.model_dump(exclude_none=True)
        
        for key, value in new_dict.items():
            if value is not None:
                current_dict[key] = value
        
        state.current_params = AccountParams(**current_dict)

@dataclass
class SearchNode(BaseNode[State]):
    """Node for performing the search."""
    
    async def run(self, ctx: GraphRunContext[State]) -> "ResponseNode | End":
        try:
            logger.info("Starting SearchNode execution")
            
            # Build search query
            query_builder = self._build_search_query(ctx.state)
            
            # Perform search
            results = query_builder.search()
            hits = results['no_of_hits']
            
            # Calculate trend
            trend = self._calculate_trend(ctx.state, hits)
            
            # Store results
            ctx.state.current_results = SearchResult(
                hits=hits,
                results=results['hits']['hits'],
                trend=trend
            )
            
            # Update search history
            ctx.state.search_history.append({
                'iteration': ctx.state.current_iteration + 1,
                'params': ctx.state.current_params.model_dump(exclude_none=True),
                'hits': hits,
                'trend': trend
            })
            
            logger.info(f"Search completed with {hits} hits ({trend})")
            
            # Check if we should stop
            if hits == 1 or hits <= ctx.state.min_threshold:
                return ResponseNode()
            
            # Increment iteration counter
            ctx.state.current_iteration += 1
            
            # Check if we've reached max iterations
            if ctx.state.current_iteration >= ctx.state.max_iterations:
                return ResponseNode()
            
            return ResponseNode()
            
        except Exception as e:
            logger.error(f"Unexpected error in SearchNode: {str(e)}", exc_info=True)
            return End(None)
    
    def _build_search_query(self, state: State) -> GameAccountSearcher:
        """Build search query based on current parameters."""
        query_builder = GameAccountSearcher(state.searcher.client)
        
        params_dict = state.current_params.model_dump(exclude_none=True)
        
        # Sort by field rankings
        sorted_params = sorted(
            params_dict.items(), 
            key=lambda x: state.field_rankings.get(x[0], 999)
        )
        
        for field, value in sorted_params:
            if field == 'name':
                query_builder = query_builder.add_query(field, value, QueryType.MATCH)
            elif field == 'alliance_name':
                query_builder = query_builder.add_query(field, value, QueryType.MATCH, fuzziness="AUTO")
            else:
                query_builder = query_builder.add_query(field, value, QueryType.MATCH)
        
        return query_builder
    
    def _calculate_trend(self, state: State, current_hits: int) -> str:
        """Calculate if results are improving or declining."""
        if not state.search_history:
            return "initial"
        
        previous_hits = state.search_history[-1]['hits']
        
        if current_hits < previous_hits * 0.5:  # 50% decrease
            return "decreasing significantly"
        elif current_hits < previous_hits:
            return "decreasing slightly"
        elif current_hits > previous_hits * 1.5:  # 50% increase
            return "improving significantly"
        elif current_hits > previous_hits:
            return "improving slightly"
        else:
            return "stable"

@dataclass
class ResponseNode(BaseNode[State]):
    """Node for generating the response."""
    
    async def run(self, ctx: GraphRunContext[State]) -> "End":
        try:
            logger.info("Starting ResponseNode execution")
            
            # Generate response based on current state
            response = self._generate_response(ctx.state)
            
            # Store response in state
            ctx.state.generated_response = response
            
            logger.info("Response generated successfully")
            return End(None)
            
        except Exception as e:
            logger.error(f"Unexpected error in ResponseNode: {str(e)}", exc_info=True)
            return End(None)
    
    def _generate_response(self, state: State) -> str:
        """Generate response based on current state."""
        if not state.current_results:
            return "I couldn't find any results matching your criteria."
        
        hits = state.current_results.hits
        
        if hits == 1:
            return "I found your account! Here are the details:\n" + \
                   json.dumps(state.current_results.results[0]['_source'], indent=2)
        
        if hits <= state.min_threshold:
            return f"I found {hits} potential matches. Here are the results:\n" + \
                   json.dumps([r['_source'] for r in state.current_results.results], indent=2)
        
        if state.current_iteration >= state.max_iterations:
            return f"I've reached the maximum number of iterations. Here are the {hits} results I found:\n" + \
                   json.dumps([r['_source'] for r in state.current_results.results], indent=2)
        
        # Generate next question based on current results
        trend = state.current_results.trend
        if trend == "decreasing significantly":
            return f"⚠️ Results dropped to {hits}. Could you provide more specific information about your account?"
        
        return f"I found {hits} results. Could you provide more information to help narrow down the search?"

async def process_account_recovery(
    initial_query: str,
    min_threshold: int = 100,
    max_iterations: int = 3
) -> Tuple[State, str]:
    """Process an account recovery request.
    
    Args:
        initial_query: The initial query from the user
        min_threshold: Minimum number of results to consider successful
        max_iterations: Maximum number of search iterations
        
    Returns:
        Tuple of (State, response)
    """
    # Initialize state
    state = State(
        conversation_history=[{"role": "user", "content": initial_query}],
        min_threshold=min_threshold,
        max_iterations=max_iterations
    )
    
    # Create the graph
    graph = Graph[State, None](
        nodes=[
            PrepareMessageHistoryNode,
            ExtractParamsNode,
            SearchNode,
            ResponseNode
        ]
    )
    
    # Run the graph
    try:
        await graph.run(start_node=PrepareMessageHistoryNode(), state=state)
        logger.info("Graph execution completed successfully")
        return state, state.generated_response or "No response generated"
        
    except Exception as e:
        logger.error(f"Error executing graph: {str(e)}", exc_info=True)
        return state, f"Error processing request: {str(e)}"

# Example usage
async def main():
    """Example usage of the account recovery graph."""
    print("\n🔍 Account Recovery Assistant")
    print("=" * 50)
    print("Please provide information about the account you're looking for.")
    print("You can include details like:")
    print("- Account name")
    print("- Alliance name")
    print("- Last updated date")
    print("- Country")
    print("- Experience level")
    print("=" * 50)
    
    # Get initial query from user
    initial_query = input("\nEnter your query: ").strip()
    while not initial_query:
        print("Please enter a valid query.")
        initial_query = input("Enter your query: ").strip()
    
    print("\nProcessing your request...")
    
    # Initialize state with first query
    state = State(
        conversation_history=[{"role": "user", "content": initial_query}],
        min_threshold=5,
        max_iterations=3
    )
    
    # Create the graph
    graph = Graph[State, None](
        nodes=[
            PrepareMessageHistoryNode,
            ExtractParamsNode,
            SearchNode,
            ResponseNode
        ]
    )
    graph.mermaid_save("account_recovery_graph.png")

    # Run the graph
    try:
        await graph.run(start_node=PrepareMessageHistoryNode(), state=state)
        
        # Print initial response and parameters
        print("\n" + "="*50)
        print("ASSISTANT RESPONSE")
        print("="*50)
        print(state.generated_response)
        print("\nCurrent Parameters:")
        print(json.dumps(state.current_params.model_dump(exclude_none=True), indent=2))
        
        # Continue conversation until max iterations or success
        while state.current_iteration < state.max_iterations:
            if state.current_results and (state.current_results.hits == 1 or state.current_results.hits <= state.min_threshold):
                break
                
            # Get user input for next iteration
            user_input = input("\nYour response: ").strip()
            if not user_input:
                print("Empty input, stopping session.")
                break
                
            # Add user input to conversation history
            state.conversation_history.append({"role": "user", "content": user_input})
            
            # Run the graph again
            await graph.run(start_node=PrepareMessageHistoryNode(), state=state)
            
            # Print response and updated parameters
            print("\n" + "="*50)
            print("ASSISTANT RESPONSE")
            print("="*50)
            print(state.generated_response)
            print("\nCurrent Parameters:")
            print(json.dumps(state.current_params.model_dump(exclude_none=True), indent=2))
        
        # Print final summary
        print("\n" + "="*50)
        print("SESSION SUMMARY")
        print("="*50)
        print(f"Final Response: {state.generated_response}")
        print(f"Total Iterations: {state.current_iteration}")
        print(f"Final Parameters: {json.dumps(state.current_params.model_dump(exclude_none=True), indent=2)}")
        
        print("\nSearch History:")
        for search in state.search_history:
            print(f"  Iteration {search['iteration']}: {search['hits']} hits ({search['trend']})")
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        return

if __name__ == "__main__":
    asyncio.run(main())
