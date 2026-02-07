"""Pytest configuration and shared fixtures."""
import pytest
import asyncio
import os
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool, StaticPool
from sqlalchemy import text

from app.db.session import Base
from app.core.config import settings

# Import all models to register them with Base.metadata
from app.models.models import (
    Occupation, Skill, OccupationSkill, UserProfile,
    UserCurrentOccupation, UserSkillRating, RecommendationEvent,
    RecommendedOccupation, UserFeedback, ModelRegistry
)

# Test database URL - use env var if set (for Docker/CI), otherwise SQLite for local testing
TEST_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")


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
    # IMPORTANT: Use StaticPool for in-memory SQLite to ensure same connection
    # For PostgreSQL (Docker/CI), use NullPool for better test isolation
    is_sqlite = TEST_DATABASE_URL.startswith("sqlite")

    engine_kwargs = {
        "echo": False,
    }

    if is_sqlite:
        # SQLite in-memory needs StaticPool to maintain single connection
        engine_kwargs["poolclass"] = StaticPool
        engine_kwargs["connect_args"] = {"check_same_thread": False}
    else:
        # PostgreSQL can use NullPool for better isolation
        engine_kwargs["poolclass"] = NullPool

    engine = create_async_engine(TEST_DATABASE_URL, **engine_kwargs)

    # Create all tables (only for SQLite - PostgreSQL uses Alembic migrations)
    if is_sqlite:
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
        # Rollback any uncommitted changes
        await session.rollback()

    # Cleanup tables between tests
    if is_sqlite:
        # SQLite: Drop and recreate tables for each test
        async with engine.begin() as conn:
            await conn.run_sync(lambda sync_conn: Base.metadata.drop_all(bind=sync_conn))
    else:
        # PostgreSQL: Truncate all tables for test isolation (faster than drop/create)
        async with engine.begin() as conn:
            await conn.execute(text("TRUNCATE TABLE user_feedback, recommended_occupation, user_skill_rating, user_current_occupation, recommendation_event, occupation_skill, user_profile, skill, occupation, model_registry CASCADE"))

    await engine.dispose()
