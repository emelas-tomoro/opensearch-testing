#!/usr/bin/env python3
"""
Search utilities for OpenSearch game accounts index.
Provides a flexible GameAccountSearcher class for advanced search operations.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from opensearchpy import OpenSearch
from enum import Enum


class QueryType(Enum):
    """Enum for different types of queries supported."""
    MATCH = "match"
    TERM = "term"
    PREFIX = "prefix"
    WILDCARD = "wildcard"
    RANGE = "range"
    MULTI_MATCH = "multi_match"


class GameAccountSearcher:
    """
    Searcher class for game accounts that supports various query types
    and combinations.

    You can add queries to the searcher in a variety of ways.
    You can add a query to the searcher by calling the add_query method.
    You can add a range query to the searcher by calling the add_range_query method.
    You can clear the searcher by calling the clear method.
    You can search the searcher by calling the search method.

    The searcher will return a dictionary of results.

    You can build a query by chaining the methods together.
    For example:
    searcher.add_query("player_tag", "ABC12", QueryType.TERM).add_range_query("last_login", gte="now-30d/d").search()

    You can also build a query by calling the methods separately.
    For example:
    searcher.add_query("player_tag", "ABC12", QueryType.TERM)
    searcher.add_range_query("last_login", gte="now-30d/d")
    searcher.search()

    """
    
    def __init__(self, client: OpenSearch, index: str = "game_accounts"):
        self.client = self._get_client(client)
        self.index = index
        self._query_parts = {
            "must": [],
            "should": [],
            "filter": [],
            "must_not": []
        }
        self._boost_factors = {}
    
    def _get_client(self, client: OpenSearch) -> OpenSearch:
        """Get the client."""

        client = OpenSearch(
            hosts=[{'host': 'localhost', 'port': 9200, 'scheme': 'http'}],  # Explicitly use HTTP
            http_auth=('admin', 'admin'),  # Default credentials for GitHub Actions OpenSearch
            use_ssl=False,
            verify_certs=False,
            ssl_show_warn=False,
        )

        return client
    
    def add_query(
        self,
        field: str,
        value: Any,
        query_type: QueryType = QueryType.MATCH,
        boost: float = 1.0,
        fuzziness: Optional[str] = None,
        context: str = "must",
        fields: Optional[List[str]] = None
    ) -> 'GameAccountSearcher':
        """
        Add a query condition to the search.
        
        Args:
            field: The field to search in
            value: The value to search for
            query_type: Type of query to use (match, term, etc.)
            boost: Boost factor for this query
            fuzziness: Fuzziness level for match queries (e.g., "AUTO")
            context: Where to add the query (must, should, filter, must_not)
            fields: For multi_match queries, list of fields to search in
        """
        if context not in self._query_parts:
            raise ValueError(f"Invalid context: {context}")
            
        query = self._build_query(field, value, query_type, boost, fuzziness, fields)
        self._query_parts[context].append(query)
        return self
    
    def _build_query(
        self,
        field: str,
        value: Any,
        query_type: QueryType,
        boost: float,
        fuzziness: Optional[str],
        fields: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Build the appropriate query based on type and parameters."""
        if query_type == QueryType.MULTI_MATCH:
            if not fields:
                raise ValueError("fields must be provided for multi_match queries")
            return {
                "multi_match": {
                    "query": value,
                    "fields": fields,
                    "boost": boost
                }
            }
            
        # For term queries, we don't use the query parameter
        if query_type == QueryType.TERM:
            return {query_type.value: {field: value}}
            
        # For other query types (match, prefix, wildcard)
        query_params = {"query": value}
        if boost != 1.0:
            query_params["boost"] = boost
        if fuzziness and query_type == QueryType.MATCH:
            query_params["fuzziness"] = fuzziness
            
        return {query_type.value: {field: query_params}}
    
    def add_range_query(
        self,
        field: str,
        gte: Optional[Any] = None,
        lte: Optional[Any] = None,
        gt: Optional[Any] = None,
        lt: Optional[Any] = None,
        context: str = "must"
    ) -> 'GameAccountSearcher':
        """
        Add a range query condition.

        Args:
            field: The field to search in
            gte: Greater than or equal to
            lte: Less than or equal to
            gt: Greater than
            lt: Less than
            context: Where to add the query (must, should, filter, must_not)

        Returns:
            The searcher instance

        Example usage:
        searcher.add_range_query("last_login", gte="now-30d/d") # last login in the last 30 days
        searcher.add_range_query("last_login", gte="2024-01-01", lte="2024-12-31") # last login between 2024-01-01 and 2024-12-31
        searcher.add_range_query("last_login", gt="2024-01-01", lt="2024-12-31") # last login after 2024-01-01 and before 2024-12-31
        searcher.add_range_query("last_login", gte="2024-01-01", lte="2024-12-31", context="filter") # last login between 2024-01-01 and 2024-12-31
        searcher.add_range_query("last_login", gte="2024-01-01", lte="2024-12-31", context="filter", boost=2.0) # last login between 2024-01-01 and 2024-12-31 with a boost of 2.0
        
        """
        range_params = {}
        for param, value in [("gte", gte), ("lte", lte), ("gt", gt), ("lt", lt)]:
            if value is not None:
                range_params[param] = value
                
        if not range_params:
            raise ValueError("At least one range parameter must be provided")
            
        self._query_parts[context].append({
            "range": {
                field: range_params
            }
        })
        return self
    
    def clear(self) -> 'GameAccountSearcher':
        """Clear all query conditions."""
        self._query_parts = {
            "must": [],
            "should": [],
            "filter": [],
            "must_not": []
        }
        self._boost_factors = {}
        return self
    
    def search(
        self,
        size: int = 10,
        from_: int = 0,
        sort: Optional[List[Dict[str, str]]] = None,
        min_should_match: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute the search with the current query conditions.
        
        Args:
            size: Number of results to return
            from_: Starting offset for pagination
            sort: List of sort conditions
            min_should_match: Minimum number of should clauses that must match
        """
        query = {"bool": {}}
        
        # Add all query parts that have conditions
        for context, conditions in self._query_parts.items():
            if conditions:
                query["bool"][context] = conditions
                
        if min_should_match is not None and self._query_parts["should"]:
            query["bool"]["minimum_should_match"] = min_should_match
            
        body = {"query": query}
        
        if size:
            body["size"] = size
        if from_:
            body["from"] = from_
        if sort:
            body["sort"] = sort

        # print(f'query: {body}')
            
        return self.client.search(index=self.index, body=body)

    # def close(self) -> None:
    #     """Close the OpenSearch client connection."""
    #     if self.client:
    #         self.client.close()


# Example usage functions that demonstrate the new searcher
def search_by_player_tag(client: OpenSearch, player_tag: str, index: str = "game_accounts") -> Dict[str, Any]:
    """Search for a player by their exact tag."""
    searcher = GameAccountSearcher(client, index)
    return searcher.add_query(
        "player_tag",
        player_tag.lower(),
        query_type=QueryType.TERM
    ).search()


def search_by_alliance(client: OpenSearch, alliance_name: str, index: str = "game_accounts") -> Dict[str, Any]:
    """Search for players in an alliance using fuzzy matching."""
    searcher = GameAccountSearcher(client, index)
    return searcher.add_query(
        "alliance_name",
        alliance_name,
        query_type=QueryType.MATCH,
        fuzziness="AUTO"
    ).search()


def search_active_premium_players(
    client: OpenSearch,
    min_registration_date: Optional[str] = None,
    index: str = "game_accounts"
) -> Dict[str, Any]:
    """Search for active premium players."""
    searcher = GameAccountSearcher(client, index)
    searcher.add_query("subscription_status", "premium", query_type=QueryType.TERM)
    searcher.add_query("account_status", "active", query_type=QueryType.TERM)
    
    if min_registration_date:
        searcher.add_range_query(
            "registration_date",
            gte=min_registration_date
        )
        
    return searcher.search()


def search_by_country_and_language(
    client: OpenSearch,
    country: str,
    language: str,
    index: str = "game_accounts"
) -> Dict[str, Any]:
    """Search for players from a specific country who speak a specific language."""
    searcher = GameAccountSearcher(client, index)
    searcher.add_query("country", country, query_type=QueryType.TERM, context="filter")
    searcher.add_query("preferred_language", language, query_type=QueryType.TERM, context="filter")
    return searcher.search()


def search_recent_players(
    client: OpenSearch,
    days: int = 30,
    index: str = "game_accounts"
) -> Dict[str, Any]:
    """Search for players who have logged in within the last N days."""
    searcher = GameAccountSearcher(client, index)
    searcher.add_range_query(
        "last_login",
        gte=f"now-{days}d/d"
    )
    return searcher.search()


def search_by_device_id(
    client: OpenSearch,
    device_id: str,
    index: str = "game_accounts"
) -> Dict[str, Any]:
    """Search for accounts associated with a specific device ID."""
    searcher = GameAccountSearcher(client, index)
    return searcher.add_query(
        "device_id",
        device_id,
        query_type=QueryType.TERM
    ).search()


def search_by_email_or_phone(
    client: OpenSearch,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    index: str = "game_accounts"
) -> Dict[str, Any]:
    """Search for accounts by email and/or phone number."""
    if not email and not phone:
        raise ValueError("Either email or phone must be provided")
        
    searcher = GameAccountSearcher(client, index)
    if email:
        searcher.add_query("email", email, query_type=QueryType.TERM, context="should")
    if phone:
        searcher.add_query("phone_number", phone, query_type=QueryType.TERM, context="should")
        
    return searcher.search(min_should_match=1)


def search_by_age_range(
    client: OpenSearch,
    min_age: int,
    max_age: int,
    index: str = "game_accounts"
) -> Dict[str, Any]:
    """Search for players within a specific age range."""
    now = datetime.now()
    max_date = (now.replace(year=now.year - min_age)).strftime("%Y-%m-%d")
    min_date = (now.replace(year=now.year - max_age)).strftime("%Y-%m-%d")
    
    searcher = GameAccountSearcher(client, index)
    searcher.add_range_query(
        "date_of_birth",
        gte=min_date,
        lte=max_date
    )
    return searcher.search()


def search_with_aggregations(
    client: OpenSearch,
    index: str = "game_accounts"
) -> Dict[str, Any]:
    """Get aggregated statistics about the player base."""
    searcher = GameAccountSearcher(client, index)
    body = {
        "size": 0,
        "aggs": {
            "subscription_stats": {
                "terms": {"field": "subscription_status"}
            },
            "account_status_stats": {
                "terms": {"field": "account_status"}
            },
            "language_stats": {
                "terms": {"field": "preferred_language"}
            },
            "avg_account_age": {
                "avg": {
                    "script": {
                        "source": "ChronoUnit.DAYS.between(doc['registration_date'].value, doc['last_login'].value)"
                    }
                }
            }
        }
    }
    return client.search(index=index, body=body)

### Example usage ###

# if __name__ == "__main__":
#     client = OpenSearch(
#         hosts=[{'host': 'localhost', 'port': 9200, 'scheme': 'http'}],  # Explicitly use HTTP
#         http_auth=('admin', ' admin'),  # Default credentials for GitHub Actions OpenSearch
#         use_ssl=False,
#         verify_certs=False,
#         ssl_show_warn=False,
#     )

#     # Basic search
#     searcher = GameAccountSearcher(client)
#     results = searcher.add_query("player_tag", "ABC12", QueryType.TERM).search()

#     # Complex search with multiple conditions
#     searcher = GameAccountSearcher(client)
#     results_basic = (
#         searcher
#         .add_query("alliance_name", "dragon", QueryType.MATCH, fuzziness="AUTO")
#         .add_query("subscription_status", "premium", QueryType.TERM, context="filter")
#         .add_range_query("last_login", gte="now-30d/d")
#         .search(size=20, from_=0)
#     )

#     # Multi-field search
#     searcher = GameAccountSearcher(client)
#     results_multi = searcher.add_query(
#         "search_term",
#         "dragon",
#         QueryType.MULTI_MATCH,
#         fields=["alliance_name", "avatar"]
#     ).search()

