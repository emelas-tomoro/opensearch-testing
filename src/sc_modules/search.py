import time
from typing import Any, Dict

from fastapi import APIRouter

from app.clients.pandora_search_client import PandoraSearchClient
from app.core.models.endpoint_models import (
    GameCode,
    LostAccountSearchResultCode,
    Message,
    Role,
    SearchRequest,
    SearchResponse,
)
from app.core.models.pandora_search_client_models import OpenSearchResponse
from app.core.models.query_models import OpenSearchQuery
from app.dependencies import opensearch_dependency
from app.utils.logging import (
    basic_logger,
    endpoint_context,
    game_context,
)

router = APIRouter()


@router.post("/search/info")
async def get_info(
    opensearch_client: opensearch_dependency,
) -> Dict[str, Any]:
    """
    Endpoint to test connection via opensearch client.
    """
    with endpoint_context("SEARCH"):
        start_time = time.time()

        basic_logger.info("Request received")

        response = await opensearch_client.info()

        round_trip_ms = int((time.time() - start_time) * 1000)
        basic_logger.info(f"Response generated in {round_trip_ms}ms: {response}")

        return response


@router.post("/search/account")
async def get_account(
    account_id: int,
    game_code: GameCode,
    opensearch_client: opensearch_dependency,
) -> OpenSearchResponse:
    """
    Endpoint to test the account search via opensearch client.
    """
    with endpoint_context("SEARCH"), game_context(game_code.to_game()):
        start_time = time.time()

        basic_logger.info("Request received")
        basic_logger.info(f"Account ID: {account_id}")

        query = {"query": {"bool": {"must": [{"term": {"account_id": account_id}}]}}}

        pandora_client = PandoraSearchClient(opensearch_client)
        response = await pandora_client.search(game=game_code.to_game(), query=query)

        round_trip_ms = int((time.time() - start_time) * 1000)
        basic_logger.info(f"Response generated in {round_trip_ms}ms: {response}")

        return response


@router.post("/search/chat/mock", response_model=SearchResponse)
async def mock_search_lost_account(request: SearchRequest) -> SearchResponse:
    """
    Mock endpoint to simulate a lost account search.

    It includes the same parameters as the original search endpoint (i.e. form data)

    The intention is to have a very similar interface to the existing endpoint.

    This endpoint is to be called after the user has attempted to fill out the
    form three times and has not been able to find their account.

    Note: The conversation_history and query_history are intended to be a single
    string that represents the conversation and query history.
    This keeps our API stateless.
    """

    with endpoint_context("MOCK"), game_context(request.game_code.to_game()):
        basic_logger.info(f"Request: {request}")

        # parse request with Strucutred outputs containing useful information

        # take exisitng queries and strucutre output and then call elasticsearch

        # check output and return either a ask for more information or account found

        # create a mock SearchResponse using the SearchRequest data

        user_msg = Message(role=Role.USER, content=request.input_text)
        assistant_msg = Message(
            role=Role.ASSISTANT,
            content=request.output_text + " Mock response for testing.",
        )

        mock_query = [OpenSearchQuery(query={"query": "test query"})]

        mock_search_response = SearchResponse(
            output_text="Mock response for testing.",
            conversation_history=request.conversation_history
            + [user_msg, assistant_msg],
            query_history=request.query_history + mock_query,
            resultCode=LostAccountSearchResultCode.ASK_ANOTHER_QUESTION,
            resultDescription=LostAccountSearchResultCode.ASK_ANOTHER_QUESTION.to_response(),
            lostAccountId="123456789",
            lostAccountTier=1,
        )

        return mock_search_response
