from datetime import datetime

import asyncpg


async def upsert_meeting(pool: asyncpg.Pool, meeting_id: str):
    """
    Inserts a new meeting row when transcription starts.
    ON CONFLICT DO NOTHING means if the room already exists, it skips silently.
    """
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO meetings (id)
            VALUES ($1)
            ON CONFLICT (id) DO NOTHING
        """, meeting_id)


async def save_segment(
    pool: asyncpg.Pool,
    meeting_id: str,
    participant_identity: str,
    spoken_at: datetime,
    text: str,
):
    """
    Saves one spoken turn to the database.
    Called every time a participant finishes a sentence or thought.
    """
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO transcript_segments
                (meeting_id, participant_identity, spoken_at, text)
            VALUES ($1, $2, $3, $4)
        """, meeting_id, participant_identity, spoken_at, text)


async def close_meeting(pool: asyncpg.Pool, meeting_id: str):
    """
    Stamps ended_at on the meeting when the room shuts down.
    Lets you know the transcript is complete.
    """
    async with pool.acquire() as conn:
        await conn.execute("""
            UPDATE meetings
            SET ended_at = now()
            WHERE id = $1
        """, meeting_id)