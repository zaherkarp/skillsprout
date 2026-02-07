"""Pytest configuration and shared fixtures."""
import pytest
import asyncio
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

from app.db.session import Base
from app.core.config import settings

# Import all models to register them with Base.metadata
from app.models.models import (
    Occupation, Skill, OccupationSkill, UserProfile,
    UserCurrentOccupation, UserSkillRating, RecommendationEvent,
    RecommendedOccupation, UserFeedback, ModelRegistry
)

# Test database URL
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
async def test_db() -> AsyncGenerator[AsyncSession, None]:
    """Create a fresh test database for each test function."""
    # Explicitly import models to ensure they're registered before table creation
    # (importing at module level may not guarantee execution order)
    from app.models.models import (  # noqa: F401
        Occupation, Skill, OccupationSkill, UserProfile,
        UserCurrentOccupation, UserSkillRating, RecommendationEvent,
        RecommendedOccupation, UserFeedback, ModelRegistry
    )

    # Create async engine for testing
    # IMPORTANT: Use poolclass=StaticPool for in-memory SQLite to ensure same connection
    from sqlalchemy.pool import StaticPool
    engine = create_async_engine(
        TEST_DATABASE_URL,
        poolclass=StaticPool,  # Use StaticPool instead of NullPool for in-memory DB
        echo=False,
        connect_args={"check_same_thread": False},  # Allow sharing across async contexts
    )

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(lambda sync_conn: Base.metadata.create_all(bind=sync_conn))

    # Create session
    TestSessionLocal = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with TestSessionLocal() as session:
        yield session

    # Drop all tables
    async with engine.begin() as conn:
        await conn.run_sync(lambda sync_conn: Base.metadata.drop_all(bind=sync_conn))

    await engine.dispose()
