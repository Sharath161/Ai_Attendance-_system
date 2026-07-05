from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from core.config import get_settings


settings = get_settings()

engine_options = {}
if settings.database_url.startswith("postgresql"):
    engine_options = {
        "pool_size": settings.api_db_pool_size,
        "max_overflow": settings.api_db_max_overflow,
        "pool_timeout": settings.db_pool_timeout_seconds,
        "pool_recycle": settings.db_pool_recycle_seconds,
    }

engine: AsyncEngine = create_async_engine(settings.database_url, **engine_options)

SessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session
