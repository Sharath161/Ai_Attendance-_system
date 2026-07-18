from core.config import get_settings
from core.database import engine
from core.models import Base


async def create_tables_for_local_dev() -> None:
    settings = get_settings()
    if not settings.auto_create_tables:
        return

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
