import asyncpg


async def ensure_schema(pool: asyncpg.Pool):
    """
    Creates the meetings and transcript_segments tables if they don't exist.
    Safe to call every time the agent starts — won't overwrite existing data.
    """
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS meetings (
                id          TEXT PRIMARY KEY,
                started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                ended_at    TIMESTAMPTZ
            );

            CREATE TABLE IF NOT EXISTS transcript_segments (
                id                   BIGSERIAL    PRIMARY KEY,
                meeting_id           TEXT         NOT NULL REFERENCES meetings(id),
                participant_identity TEXT         NOT NULL,
                spoken_at            TIMESTAMPTZ  NOT NULL,
                text                 TEXT         NOT NULL,
                created_at           TIMESTAMPTZ  NOT NULL DEFAULT now()
            );

            CREATE INDEX IF NOT EXISTS idx_segments_meeting
                ON transcript_segments(meeting_id, spoken_at);
        """)