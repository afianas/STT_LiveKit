import os
import asyncpg


async def create_pool() -> asyncpg.Pool:
    """
    Creates and returns a database connection pool.
    Min 1 connection always open, max 5 at once.
    The pool is reused across all sessions in the same worker process.
    """
    return await asyncpg.create_pool(
        dsn=os.getenv("DATABASE_URL"),
        min_size=1,
        max_size=5,
    )