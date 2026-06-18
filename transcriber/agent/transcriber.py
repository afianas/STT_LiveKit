from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from livekit.agents import Agent, StopResponse, llm, stt

if TYPE_CHECKING:
    from transcriber.agent.session_manager import SessionManager

logger = logging.getLogger("transcriber.agent")


class Transcriber(Agent):
    """
    One instance of this agent is created per participant in the meeting.
    """

    def __init__(
        self,
        *,
        participant_identity: str,
        session_manager: SessionManager,
        stt: stt.STT,
    ):
        super().__init__(
            instructions="not-needed",
            stt=stt,
        )

        self.participant_identity = participant_identity
        self.session_manager = session_manager

        # Created only during graceful shutdown
        self._turn_completed_event: asyncio.Event | None = None

    async def on_user_turn_completed(
        self,
        chat_ctx: llm.ChatContext,
        new_message: llm.ChatMessage,
    ):
        text = new_message.text_content

        started_speaking_at = new_message.metrics.get(
            "started_speaking_at"
        )
        stopped_speaking_at = new_message.metrics.get(
            "stopped_speaking_at"
        )

        if started_speaking_at is not None:
            started_at = datetime.fromtimestamp(
                started_speaking_at,
                timezone.utc,
            )
        elif stopped_speaking_at is not None:
            started_at = datetime.fromtimestamp(
                stopped_speaking_at,
                timezone.utc,
            )
        else:
            started_at = datetime.now(timezone.utc)

        if stopped_speaking_at is not None:
            ended_at = datetime.fromtimestamp(
                stopped_speaking_at,
                timezone.utc,
            )
        else:
            ended_at = started_at

        logger.info(
            "[%s] %s -> %s",
            started_at.strftime("%H:%M:%S"),
            self.participant_identity,
            text,
        )

        self.session_manager.add_segment(
            participant=self.participant_identity,
            started_at=started_at,
            ended_at=ended_at,
            text=text,
        )

        #
        # Signal graceful shutdown if it is waiting
        #
        if self._turn_completed_event is not None:
            self._turn_completed_event.set()

        raise StopResponse()