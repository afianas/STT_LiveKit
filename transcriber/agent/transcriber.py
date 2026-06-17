import logging
import asyncio
from datetime import datetime, timezone

import asyncpg
from livekit.agents import Agent, StopResponse, llm, stt

from transcriber.db import save_segment

logger = logging.getLogger("transcriber.agent")


class Transcriber(Agent):
    """
    One instance of this agent is created per participant in the meeting.

    Lifecycle:
      1. Participant joins the room
      2. MultiUserTranscriber creates a Transcriber for them
      3. LiveKit streams their audio here
      4. Whisper converts audio to text
      5. on_user_turn_completed fires with the transcribed text
      6. We save it to the DB and stop (no LLM reply needed)
    """

    def __init__(
        self,
        *,
        participant_identity: str,
        meeting_id: str,
        db_pool: asyncpg.Pool,
        stt: stt.STT,
    ):
        super().__init__(
            instructions="not-needed",        # no LLM system prompt — we only do STT
            stt=stt,                          # speech-to-text engine
        )
        self.participant_identity = participant_identity
        self.meeting_id = meeting_id
        self.db_pool = db_pool

    async def on_user_turn_completed(
        self,
        chat_ctx: llm.ChatContext,
        new_message: llm.ChatMessage,
    ):
        """
        Fires every time a participant finishes speaking a turn.
        A "turn" is detected by VAD (voice activity detection) — 
        silence after speech signals the turn is over.
        """
        text = new_message.text_content
        spoken_at = datetime.now(timezone.utc)

        logger.info(
            f"[{spoken_at.strftime('%H:%M:%S')}] "
            f"{self.participant_identity} -> {text}"
        )

        # Try to save to DB with retry logic
        success = False
        for attempt in range(3):
            try:
                await save_segment(
                    pool=self.db_pool,
                    meeting_id=self.meeting_id,
                    participant_identity=self.participant_identity,
                    spoken_at=spoken_at,
                    text=text,
                )
                success = True
                break
            except Exception as e:
                logger.warning(
                    f"Failed to save transcript segment to database (attempt {attempt + 1}/3): {e}"
                )
                if attempt < 2:
                    await asyncio.sleep(1.0)

        if not success:
            logger.critical(
                f"DATABASE SAVE FAILED: Could not save transcript to database! "
                f"Fallback transcript dump: "
                f"meeting_id={self.meeting_id} "
                f"participant={self.participant_identity} "
                f"time={spoken_at.isoformat()} "
                f"text={text!r}"
            )

        # StopResponse tells LiveKit: don't generate an LLM reply,
        # just transcribe and stop. We're not a chatbot.
        raise StopResponse()