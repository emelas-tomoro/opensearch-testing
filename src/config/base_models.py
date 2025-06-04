from typing import Any, Dict, List, Optional, TYPE_CHECKING
import os
from pydantic import BaseModel, Field


if TYPE_CHECKING:
    from src.utils.search_utils import GameAccountSearcher

class AccountParams(BaseModel):
    """Model representing account search parameters."""
    name: Optional[str] = Field(default=None, description="The name of the account")
    alliance_name: Optional[str] = Field(default=None, description="The name of the alliance")
    updated: Optional[str] = Field(default=None, description="The last updated date of the account")
    create_country: Optional[str] = Field(default=None, description="The country of the account")
    exp_level: Optional[str] = Field(default=None, description="The experience level of the account")

class SearchResult(BaseModel):
    """Model representing search results."""
    hits: int = Field(..., description="Number of hits")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Search results")
    trend: str = Field(..., description="Trend of results (improving/decreasing)")

class State(BaseModel):
    """State model for the account recovery graph."""
    test: bool
    index: str = Field(default=os.getenv('INDEX_NAME', 'game_accounts'))
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    search_history: List[Dict[str, Any]] = Field(default_factory=list)
    current_params: AccountParams = Field(default_factory=AccountParams)
    current_results: Optional[SearchResult] = None
    generated_response: Optional[str] = None
    min_threshold: int = Field(default=100)
    max_iterations: int = Field(default=3)
    current_iteration: int = Field(default=0)
    field_rankings: Dict[str, int] = Field(
        default={
            'name': 1,
            'alliance_name': 2,
            'updated': 3,
            'create_country': 4,
            'exp_level': 5
        }
    )
    searcher: Optional[Any] = None

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow"
    }
