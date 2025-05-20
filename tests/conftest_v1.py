"""
Pytest configuration for setting up and tearing down a test Postgres database
(with pgvector) and providing test fixtures for ChatRecord creation.
"""

import os
from typing import List, AsyncGenerator, Generator
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncSession
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from testcontainers.postgres import PostgresContainer
from alembic.config import Config
from alembic import command
from unittest.mock import patch

from app.core.models.router_models import (
    UpfrontClassification,
    DangerSchema,
    DangerInputClassification,
    SmallTalkSchema,
    SmallTalkClassification,
    FunctionInputClassification,
    FunctionClassifiedSchema,
    IsUpsetSchema,
    IsUpsetClassification,
    IsSensitiveSchema,
    IsSensitiveClassification,
    IsEnglishSchema,
    IsNonsenseSchema,
    IsNonsenseClassification,
    IsEnglishClassification,
)
from app.main import app
from app.core.models.interface_models import (
    ChatRecord,
    ChatRequest,
    ChatResponse,
    QueryMetadata,
    GameContext,
    QueryMessageCategory,
    ResponseMessageCategory,
)
from app.core.models.messages.message_models import Message, Role
from constants import ALEMBIC_CONFIG_PATH, ALEMBIC_FOLDER_PATH
from db.utils import get_async_session


def _make_messages(user_query: str) -> List[Message]:
    """
    A helper function that creates a list containing a single user Message object.
    """
    return [Message(role=Role.USER, content=user_query)]


@pytest.fixture
def messages() -> List[Message]:
    """
    Example fixture for returning a minimal list of user messages.
    """
    return _make_messages("A test message")


@pytest.fixture
def game_context() -> GameContext:
    """
    Provides a minimal, valid GameContext object for testing.
    """
    return GameContext(
        game_version="test-game-version",
        os_version="test-os-version",
        device="test-device",
        platform="iOS",
        quality=10,
        region="test-region",
        country="test-country",
        playerName="TestPlayer",
        playerTag="ABC123",
        playerLevel=5,
        sessionId="session-xyz",
        sessionCount=1,
        type=1,
        ageGateStatus="above",
    )


@pytest.fixture
def upfront_classification() -> UpfrontClassification:
    return UpfrontClassification(
        danger_schema=DangerSchema(
            classification=DangerInputClassification.DO_NOT_ESCALATE,
            cot_reasoning="",
        ),
        small_talk_schema=SmallTalkSchema(classification=SmallTalkClassification.OTHER),
        route_schema=FunctionClassifiedSchema(
            classification=FunctionInputClassification.UNRELATED_STOP
        ),
        upset_schema=IsUpsetSchema(classification=IsUpsetClassification.NOT_UPSET),
        sensitive_schema=IsSensitiveSchema(
            classification=IsSensitiveClassification.NOT_SENSITIVE
        ),
        english_schema=IsEnglishSchema(classification=IsEnglishClassification.ENGLISH),
        nonsense_schema=IsNonsenseSchema(
            classification=IsNonsenseClassification.NOT_NONSENSE
        ),
        cached_rag_response=None,
    )


@pytest.fixture
def query_metadata() -> QueryMetadata:
    """
    Provides a minimal, valid QueryMetadata object for testing.
    """
    return QueryMetadata(
        query_category=QueryMessageCategory.BUG_REPORT,  # type: ignore # or any enum value
        response_category=ResponseMessageCategory.placeholder,
        round_trip_ms=123,
    )


@pytest.fixture
def chat_request(game_context: GameContext) -> ChatRequest:
    """
    Builds a ChatRequest containing a single user message and the provided GameContext.
    """
    user_message = Message(role=Role.USER, content="Hello, I'm reporting a bug.")
    return ChatRequest(chat_history=[user_message], context=game_context, rt_mode=True)


@pytest.fixture
def chat_response() -> ChatResponse:
    """
    Builds a ChatResponse containing a single assistant message
    with classification recognized by QueryMessageCategory.
    """
    assistant_message = Message(
        role=Role.ASSISTANT,
        content="Thanks for reporting this issue. We'll look into it!",
    )
    return ChatResponse(
        message=assistant_message,
        classification=FunctionInputClassification.BUG_REPORT,
        bug_found=None,
        bug_list=None,
        high_risk_found=None,
    )


@pytest.fixture
def chat_record(
    chat_request: ChatRequest,
    chat_response: ChatResponse,
    query_metadata: QueryMetadata,
    game_context: GameContext,
) -> ChatRecord:
    """
    Assembles a ChatRecord using the above fixtures for request, response,
    metadata, and context.
    """
    return ChatRecord(
        chat_length=len(chat_request.chat_history),
        query_message=chat_request,
        response_message=chat_response,
        query_metadata=query_metadata,
        context=game_context,
    )


def _create_alembic_config(url: str) -> Config:
    """Helper function to create Alembic config with the correct settings"""
    alembic_cfg = Config(ALEMBIC_CONFIG_PATH)
    alembic_cfg.set_main_option("sqlalchemy.url", url)
    alembic_cfg.set_main_option("script_location", str(ALEMBIC_FOLDER_PATH))

    # Patch environment variables during migration to prevent reading from .env
    with patch.dict(
        os.environ,
        {"DB_USER": "", "DB_PASSWORD": "", "DB_HOST": "", "DB_PORT": "", "DB_NAME": ""},
    ):
        return alembic_cfg


@pytest.fixture(scope="session")
def migrated_test_db() -> Generator[str, None, None]:
    """
    Spins up either:
      - a Postgres + pgvector container using testcontainers (LOCAL)
      - OR uses the Postgres service container from GitHub Actions (CI).

    Installs the pgvector extension and runs Alembic migrations.
    Yields a database URL that can be used for async engine creation.
    """
    # If we're on GitHub Actions, rely on the service container:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        # Adjust these credentials/host/DB if you changed them in the GitHub Actions file
        url = "postgresql+psycopg://test:test@localhost:5432/test_db"

        # Create pgvector extension & run migrations
        sync_engine = create_engine(url, future=True)
        with sync_engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()

        # Create Alembic config with the test URL
        alembic_cfg = _create_alembic_config(url)

        # Ensure env variables are patched during migration
        with patch.dict(
            os.environ,
            {
                "DB_USER": "test",
                "DB_PASSWORD": "test",
                "DB_HOST": "localhost",
                "DB_PORT": "5432",
                "DB_NAME": "test_db",
            },
        ):
            command.upgrade(alembic_cfg, "head")

        yield url

    else:
        # LOCAL usage: spin up an ephemeral container using Testcontainers
        with PostgresContainer("ankane/pgvector:latest") as container:
            container.start()

            # Convert from "postgresql+psycopg2://" to "postgresql+psycopg://"
            # (required because testcontainers uses psycopg2)
            url = container.get_connection_url().replace("psycopg2", "psycopg")

            # Create pgvector extension
            sync_engine = create_engine(url, future=True)
            with sync_engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                conn.commit()

            # Extract connection params for env patching during migration
            # Example: postgresql+psycopg://test:test@localhost:12345/test
            params = url.split("://")[1].split("@")
            creds = params[0].split(":")
            host_port_db = params[1].split("/")
            host_port = host_port_db[0].split(":")
            db_name = host_port_db[1]

            # Create Alembic config with the test URL
            alembic_cfg = _create_alembic_config(url)

            # Ensure env variables are patched during migration
            with patch.dict(
                os.environ,
                {
                    "DB_USER": creds[0],
                    "DB_PASSWORD": creds[1],
                    "DB_HOST": host_port[0],
                    "DB_PORT": host_port[1],
                    "DB_NAME": db_name,
                },
            ):
                command.upgrade(alembic_cfg, "head")

            yield url


@pytest_asyncio.fixture(scope="session")
async def async_engine(migrated_test_db: str) -> AsyncGenerator[AsyncEngine, None]:
    """
    Creates a single AsyncEngine for the entire test session, pointing to the
    ephemeral test DB returned by migrated_test_db. Disposes when tests complete.
    """
    engine = create_async_engine(migrated_test_db, echo=False, future=True)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def db_session(async_engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """
    Yields a fresh AsyncSession for each test function.
    After the test, truncates all the relevant tables so each test starts clean,
    even if the tested code called session.commit().
    """
    SessionLocal = sessionmaker(
        bind=async_engine, class_=AsyncSession, expire_on_commit=False
    )

    async with SessionLocal() as session:
        yield session

        # After each test, truncate the tables you want to clear.
        # Add or remove table names depending on your schema.
        tables = [
            "message_history",
            "conversation_feedback",
            "chat_session",
            "contextual_rag",
        ]
        for table_name in tables:
            await session.execute(text(f"TRUNCATE TABLE {table_name} CASCADE;"))

        # We do a final commit so the truncation is persisted.
        await session.commit()


@pytest.fixture
def override_get_async_session(db_session: AsyncSession):
    """
    Temporarily overrides the FastAPI dependency get_async_session so that
    any route dependent on it will use our test session instead of production.
    """

    def _override_get_async_session():
        yield db_session

    app.dependency_overrides[get_async_session] = _override_get_async_session

    # Provide the test session to the test function
    yield

    # Cleanup: remove the override after the test
    app.dependency_overrides.pop(get_async_session, None)


@pytest_asyncio.fixture
async def override_retriever_pool(async_engine: AsyncEngine):
    """
    Overrides the retriever's connection pool to use the existing AsyncEngine's pool.
    """
    from app.retrieval import retriever  # Adjust the import path if necessary

    with patch.object(retriever, "get_pool", return_value=async_engine):
        yield
