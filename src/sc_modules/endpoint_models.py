import json
from enum import IntEnum, StrEnum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, Json, field_serializer

from app.core.models.game_models import GameCode
from app.core.models.query_models import OpenSearchQuery


class SourceEnum(StrEnum):
    """Enum for the source of the request."""

    HELPSHIFT = "HELPSHIFT"


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """Data structure for a message in the chat"""

    role: Role
    content: str


class SearchModel(BaseModel):
    """
    Base model for search requests and responses
    """

    output_text: str = Field(
        "",
        description="Free text content from the chatbot. This gets updated to have "
        "a dynamic response",
    )
    conversation_history: Json[List[Message]] | List[Message] = Field(
        default_factory=list,  # type: ignore[arg-type]
        description="A JSON string of message history of the chat that convert "
        "to List[Message]",
    )
    query_history: Json[List[OpenSearchQuery]] | List[OpenSearchQuery] = Field(
        default_factory=list,  # type: ignore[arg-type]
        description="A JSON string of OpenSearch queries used "
        "in the chat that convert to List[OpenSearchQuery]",
    )


class SearchRequest(SearchModel):
    """Payload sent when a user reports a lost account"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    externalId: str = Field(..., description="Unique issue / ticket identifier")
    source: SourceEnum = Field(
        ...,
        description="Source of the request. "
        "Likely to be HELPSHIFT, but this attribute is kept "
        "to be consistent with the past approach.",
    )
    game_code: GameCode = Field(..., description="Game identifier", alias="game")

    contactAccountId: str = Field(
        ..., description="The player’s current account ID (contact)"
    )
    sessionId: Optional[str] = Field(None, description="Session ID of the current chat")
    deviceId: Optional[str] = Field(None, description="Device ID of the current chat")

    targetAccountTag: str = Field(
        ..., description="FORM DATA: Tag of the account the player is trying to recover"
    )
    targetAccountAvatar: str = Field(
        ..., description="FORM DATA: Display name / avatar for the lost account"
    )
    targetAccountAlliance: str = Field(
        ..., description="FORM DATA: Alliance / clan name for the lost account"
    )
    targetAccountLevel: int = Field(
        ..., description="FORM DATA: Level of the lost account"
    )
    input_text: str = Field(..., description="Free text content from the user")


class LostAccountSearchResultCode(IntEnum):
    FOUND_MATCHING_ACCOUNT = 1000
    CANDIDATES_FOUND = 1001
    NOTHING_FOUND = 1002
    MAX_ATTEMPTS_EXCEEDED = 1003
    CONTACT_IS_SAME_AS_TARGET = 1004
    ASK_ANOTHER_QUESTION = 1005  # new code for raw text

    def to_response(self) -> str:
        """Convert the result code to a human-readable response."""
        _look_up = {
            self.FOUND_MATCHING_ACCOUNT: "Found a matching account",
            self.CANDIDATES_FOUND: "Candidates found, refine search",
            self.NOTHING_FOUND: "No accounts found matching your criteria",
            self.MAX_ATTEMPTS_EXCEEDED: "Maximum attempts exceeded, "
            "please try again later",
            self.CONTACT_IS_SAME_AS_TARGET: "Your contact account is the same "
            "as the target account",
            self.ASK_ANOTHER_QUESTION: "Ask another question",
        }
        return _look_up.get(self, "Unknown result code")


class SearchResponse(SearchModel):
    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)
    resultCode: LostAccountSearchResultCode = Field(
        ..., description="Result code indicating search outcome"
    )
    resultDescription: str = Field(
        ..., description="Human-readable description of the search result"
    )
    lostAccountId: Optional[str] = Field(
        None, description="ID of the found lost account, if any"
    )
    lostAccountTier: Optional[int] = Field(
        None, description="Tier/level of the found lost account, if any"
    )

    # When SearchResponse (which inherits from SearchModel) serializes the
    # conversation_history field (which internally holds a List[Message]),
    # Pydantic matches the List[Message] part of the union and serializes
    # it directly into a JSON array in the response.

    # The Json[List[Message]] part, which would serialize to a JSON string,
    # is not chosen in this scenario for serialization when the data is already a list.

    # The below ensures is serialised in a manner helpshift expects.

    # They are always serialized as JSON strings, regardless of how they
    # are used in the response.

    # This means we can pass around query history and conversation history
    # to and from Helpshift

    @field_serializer("conversation_history", when_used="json")
    def serialize_conversation_history_to_string(self, v: List[Message]) -> str:
        """Serializes the list of Message objects to a JSON string."""
        return json.dumps([item.model_dump() for item in v])

    @field_serializer("query_history", when_used="json")
    def serialize_query_history_to_string(self, v: List[OpenSearchQuery]) -> str:
        """Serializes the list of OpenSearchQuery objects to a JSON string."""
        return json.dumps([item.model_dump() for item in v])


class TargetAccount(BaseModel):
    """
    Model for the target account information in a lost account search request.
    This is used to capture the details of the account the user is trying to recover.
    """

    targetAccountTag: str = Field(
        ..., description="FORM DATA: Tag of the account the player is trying to recover"
    )
    targetAccountAvatar: str = Field(
        ..., description="FORM DATA: Display name / avatar for the lost account"
    )
    targetAccountAlliance: str = Field(
        ..., description="FORM DATA: Alliance / clan name for the lost account"
    )
    targetAccountLevel: int = Field(
        ..., description="FORM DATA: Level of the lost account"
    )
