import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from opensearchpy import OpenSearch
from pydantic_graph import End

from src.agents.account_recovery_graph import (
    State,
    AccountParams,
    SearchResult,
    PrepareMessageHistoryNode,
    ExtractParamsNode,
    SearchNode,
    ResponseNode,
    process_account_recovery
)
from src.utils.search_utils import GameAccountSearcher

# Mock data
MOCK_SEARCH_RESULTS = {
    'no_of_hits': 2,
    'hits': {
        'hits': [
            {
                '_source': {
                    'name': 'test_account',
                    'alliance_name': 'test_alliance',
                    'updated': '2024-01-01',
                    'create_country': 'US',
                    'exp_level': '10'
                }
            },
            {
                '_source': {
                    'name': 'test_account2',
                    'alliance_name': 'test_alliance2',
                    'updated': '2024-01-02',
                    'create_country': 'UK',
                    'exp_level': '20'
                }
            }
        ]
    }
}

@pytest.fixture
def mock_opensearch_client():
    """Fixture for mock OpenSearch client."""
    return Mock(spec=OpenSearch)

@pytest.fixture
def mock_searcher(mock_opensearch_client):
    """Fixture for mock GameAccountSearcher."""
    searcher = Mock(spec=GameAccountSearcher)
    searcher.client = mock_opensearch_client
    searcher.search.return_value = MOCK_SEARCH_RESULTS
    searcher.add_query = Mock(return_value=searcher)
    return searcher

@pytest.fixture
def mock_openai_response():
    """Fixture for mock OpenAI response."""
    return Mock(
        output_text=json.dumps({
            'name': 'test_account',
            'alliance_name': 'test_alliance',
            'updated': None,
            'create_country': None,
            'exp_level': None
        })
    )

@pytest.fixture
def initial_state():
    """Fixture for initial state."""
    return State(
        conversation_history=[
            {"role": "user", "content": "I want to find an account called test_account"}
        ],
        min_threshold=5,
        max_iterations=3
    )

@pytest.mark.asyncio
async def test_prepare_message_history_node(initial_state, mock_searcher):
    """Test PrepareMessageHistoryNode."""
    # Create node
    node = PrepareMessageHistoryNode()
    
    # Create mock context
    ctx = Mock()
    ctx.state = initial_state
    
    # Mock GameAccountSearcher initialization
    with patch('src.agents.account_recovery_graph.GameAccountSearcher', return_value=mock_searcher):
        # Run node
        next_node = await node.run(ctx)
        
        # Verify searcher was initialized
        assert ctx.state.searcher is not None
        assert isinstance(next_node, ExtractParamsNode)

@pytest.mark.asyncio
async def test_extract_params_node(initial_state, mock_openai_response):
    """Test ExtractParamsNode."""
    # Create node
    node = ExtractParamsNode()
    
    # Create mock context
    ctx = Mock()
    ctx.state = initial_state
    
    # Mock OpenAI response
    with patch('src.agents.account_recovery_graph.client.responses.parse', return_value=mock_openai_response):
        # Run node
        next_node = await node.run(ctx)
        
        # Verify parameters were extracted and merged
        assert ctx.state.current_params.name == 'test_account'
        assert ctx.state.current_params.alliance_name == 'test_alliance'
        assert isinstance(next_node, SearchNode)

@pytest.mark.asyncio
async def test_search_node(initial_state, mock_searcher):
    """Test SearchNode."""
    # Create node
    node = SearchNode()
    
    # Create mock context
    ctx = Mock()
    ctx.state = initial_state
    ctx.state.searcher = mock_searcher
    
    # Mock the add_query chain
    mock_searcher.add_query.return_value = mock_searcher
    
    # Run node
    next_node = await node.run(ctx)
    
    # Verify search was performed
    assert ctx.state.current_results is not None
    assert ctx.state.current_results.hits == MOCK_SEARCH_RESULTS['no_of_hits']  # Use the mock data's hit count
    assert len(ctx.state.search_history) == 1
    assert isinstance(next_node, ResponseNode)
    
    # Verify add_query was called
    assert mock_searcher.add_query.called

@pytest.mark.asyncio
async def test_response_node(initial_state):
    """Test ResponseNode."""
    # Create node
    node = ResponseNode()
    
    # Create mock context with search results
    ctx = Mock()
    ctx.state = initial_state
    ctx.state.current_results = SearchResult(
        hits=2,
        results=MOCK_SEARCH_RESULTS['hits']['hits'],
        trend='initial'
    )
    
    # Run node
    next_node = await node.run(ctx)
    
    # Verify response was generated
    assert ctx.state.generated_response is not None
    assert 'test_account' in ctx.state.generated_response
    assert isinstance(next_node, End)

@pytest.mark.asyncio
async def test_process_account_recovery_success(initial_state, mock_searcher, mock_openai_response):
    """Test successful account recovery process."""
    # Mock OpenAI response
    with patch('src.agents.account_recovery_graph.client.responses.parse', return_value=mock_openai_response):
        # Mock searcher
        with patch('src.agents.account_recovery_graph.GameAccountSearcher', return_value=mock_searcher):
            # Run process
            state, response = await process_account_recovery(
                initial_query="I want to find an account called test_account",
                min_threshold=5,
                max_iterations=3
            )
            
            # Verify results
            assert state.current_params.name == 'test_account'
            assert state.current_results is not None
            assert state.current_results.hits == MOCK_SEARCH_RESULTS['no_of_hits']  # Use the mock data's hit count
            assert response is not None

@pytest.mark.asyncio
async def test_process_account_recovery_error(initial_state):
    """Test account recovery process with error."""
    # Mock OpenAI to raise an error
    with patch('src.agents.account_recovery_graph.client.responses.parse', side_effect=Exception("API Error")):
        # Run process
        state, response = await process_account_recovery(
            initial_query="I want to find an account called test_account",
            min_threshold=5,
            max_iterations=3
        )
        
        # Verify error handling
        assert "Error" in response or "No response generated" in response  # Accept either error message

@pytest.mark.asyncio
async def test_search_trend_calculation():
    """Test search trend calculation."""
    # Create node
    node = SearchNode()
    
    # Create mock context
    ctx = Mock()
    ctx.state = State()
    
    # Test initial trend
    trend = node._calculate_trend(ctx.state, 10)
    assert trend == "initial"
    
    # Add search history
    ctx.state.search_history = [{'hits': 20}]
    
    # Test decreasing trend
    trend = node._calculate_trend(ctx.state, 5)
    assert trend == "decreasing significantly"
    
    # Test improving trend
    trend = node._calculate_trend(ctx.state, 30)
    assert trend == "improving significantly"  # This matches the implementation's threshold of 1.5x

@pytest.mark.asyncio
async def test_parameter_merging():
    """Test parameter merging functionality."""
    # Create node
    node = ExtractParamsNode()
    
    # Create mock context
    ctx = Mock()
    ctx.state = State()
    ctx.state.current_params = AccountParams(
        name="old_name",
        alliance_name="old_alliance"
    )
    
    # Create new parameters
    new_params = AccountParams(
        name="new_name",
        exp_level="10"
    )
    
    # Merge parameters
    node._merge_params(ctx.state, new_params)
    
    # Verify merge
    assert ctx.state.current_params.name == "new_name"
    assert ctx.state.current_params.alliance_name == "old_alliance"
    assert ctx.state.current_params.exp_level == "10" 