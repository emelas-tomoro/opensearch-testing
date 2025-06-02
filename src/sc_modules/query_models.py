from pydantic import BaseModel


class OpenSearchQuery(BaseModel):
    """OpenSearch placeholder query"""

    query: dict  # placeholder for actual query structure
