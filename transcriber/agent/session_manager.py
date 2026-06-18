from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from livekit import rtc
from livekit.agents import AgentSession, JobContext, room_io, utils

from transcriber.agent.transcriber import Transcriber

if TYPE_CHECKING:
    pass  # reserved for future forward-reference annotations

logger = logging.getLogger("transcriber.session_manager")

# Resolved once at import time so the output directory is stable regardless of
# the process CWD.  Override with the TRANSCRIPTS_DIR environment variable.
_TRANSCRIPTS_DIR: Path = Path(
    os.getenv(
        "TRANSCRIPTS_DIR",
        str(Path(__file__).parent.parent.parent / "transcripts"),
    )
).resolve()

# Matches any character that is not a word character or hyphen.
# Used to sanitise the meeting_id before using it as a filename.
_UNSAFE_FILENAME_RE = re.compile(r"[^\w\-]")


def _safe_filename(name: str) -> str:
    """Replace characters that are illegal in filenames with underscores."""
    return _UNSAFE_FILENAME_RE.sub("_", name)


class SessionManager:
    """
    Manages one AgentSession per participant in the room.

    Why one session per participant?
    Each participant's audio is a separate stream in LiveKit.
    We need a dedicated Transcriber listening to each stream
    so we can attribute text to the right speaker.

    Responsibilities:
    - Watch for participants joining / leaving
    - Create a Transcriber session when someone joins
    - Clean up their session when they leave
    - Collect transcript segments in memory
    - Write a JSON transcript file when the meeting ends
    """

    def __init__(
        self,
        ctx: JobContext,
        meeting_id: str,
    ):
        self.ctx = ctx
        self.meeting_id = meeting_id

        # In-memory transcript storage — all segments for this meeting
        self.transcript_segments: list[dict] = []

        # Maps participant identity -> their active AgentSession
        self._sessions: dict[str, AgentSession] = {}

        # Maps participant identity -> starting Task 
        self._starting_sessions: dict[str, asyncio.Task] = {}

        # Tracks in-flight session starting tasks (safe to cancel on shutdown)
        self._start_tasks: set[asyncio.Task] = set()

        # Tracks in-flight session closing tasks (MUST NOT cancel on shutdown, wait for them)
        self._close_tasks: set[asyncio.Task] = set()

    def start(self):
        """Register event listeners for the room."""
        self.ctx.room.on("participant_connected", self.on_participant_connected)
        self.ctx.room.on("participant_disconnected", self.on_participant_disconnected)

    def add_segment(
        self,
        participant: str,
        started_at: datetime,
        ended_at: datetime,
        text: str,
    ) -> None:
        """
        Append a transcript segment to in-memory storage.
        Insertion order is preserved (list is ordered by arrival).
        """
        self.transcript_segments.append({
            "participant": participant,
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "text": text,
        })

    async def _write_transcript_json(self) -> None:
        """
        Sort all collected segments by started_at and write a JSON file to
        _TRANSCRIPTS_DIR/{meeting_id}.json.

        The write is atomic: content is flushed to a temp file in the same
        directory and then renamed, so a crash mid-write never produces a
        truncated or corrupt transcript file.
        """
        # Skip the write entirely if nothing was recorded
        if not self.transcript_segments:
            logger.info(
                "No segments recorded for meeting %s; skipping transcript file.",
                self.meeting_id,
            )
            return

        # Snapshot before sorting so late-arriving appends from another
        # coroutine do not mutate the iterable while we process it.
        segments_snapshot = list(self.transcript_segments)
        sorted_segments = sorted(segments_snapshot, key=lambda s: s["started_at"])

        payload = {
            "meeting_id": self.meeting_id,
            "segments": sorted_segments,
        }

        # _TRANSCRIPTS_DIR is an absolute path resolved at import time.
        _TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

        # Sanitise the meeting_id so characters like '/', ':', '\' (which
        # are valid in LiveKit room names but illegal in filenames) don't cause
        # an OSError.
        safe_id = _safe_filename(self.meeting_id)
        output_path = _TRANSCRIPTS_DIR / f"{safe_id}.json"

        # Atomic write — write to a sibling temp file, then rename.
        content = json.dumps(payload, ensure_ascii=False, indent=2)
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=_TRANSCRIPTS_DIR,
            delete=False,
            suffix=".tmp",
            encoding="utf-8",
        ) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        tmp_path.replace(output_path)
        logger.info("Transcript saved to %s", output_path)

    async def aclose(self) -> None:
        """
        Graceful shutdown:
        1. Unregister room event listeners immediately to avoid race conditions
        2. Cancel any pending session-start tasks
        3. Wait for all in-flight session-close tasks
        4. Cleanly close all remaining active sessions
        5. Write the transcript JSON file (with fallback logging on failure)
        """
        self.ctx.room.off("participant_connected", self.on_participant_connected)
        self.ctx.room.off("participant_disconnected", self.on_participant_disconnected)

        # Cancel any startup in progress
        await utils.aio.cancel_and_wait(*self._start_tasks)

        # Wait for all existing close/disconnect tasks to finish
        if self._close_tasks:
            await asyncio.gather(*self._close_tasks, return_exceptions=True)

        # Cleanly close all remaining active sessions
        if self._sessions:
            await asyncio.gather(
                *[self._close_session(identity, s) for identity, s in self._sessions.items()],
                return_exceptions=True,
            )

        # if the write fails for any reason (disk full, bad permissions,
        # etc.), dump every segment to the log so the data is not lost entirely.
        try:
            await self._write_transcript_json()
        except Exception:
            logger.critical(
                "Failed to write transcript JSON for meeting %s — "
                "dumping %d segments to log as fallback:",
                self.meeting_id,
                len(self.transcript_segments),
                exc_info=True,
            )
            for seg in self.transcript_segments:
                logger.critical("SEGMENT: %s", seg)

        logger.info("Meeting %s ended.", self.meeting_id)

    def on_participant_connected(self, participant: rtc.RemoteParticipant):
        """
        Fires when someone joins the LiveKit room.
        Starts a transcription session for them asynchronously.
        """
        if participant.identity in self._sessions or participant.identity in self._starting_sessions:
            return  # already transcribing or starting them, skip

        logger.info(f"Participant joined: {participant.identity}")
        task = asyncio.create_task(self._start_session(participant))
        self._starting_sessions[participant.identity] = task
        self._start_tasks.add(task)

        def on_done(t: asyncio.Task):
            try:
                if not t.cancelled():
                    # Store the completed session so we can close it later
                    self._sessions[participant.identity] = t.result()
            except Exception as e:
                logger.error(
                    f"Error starting session for {participant.identity}: {e}",
                    exc_info=True,
                )
            finally:
                self._starting_sessions.pop(participant.identity, None)
                self._start_tasks.discard(t)

        task.add_done_callback(on_done)

    def on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        """
        Fires when someone leaves the LiveKit room.
        Closes their transcription session.
        """
        # Cancel in-flight startup if it exists
        starting_task = self._starting_sessions.pop(participant.identity, None)
        if starting_task is not None:
            logger.info(f"Cancelling in-flight session startup for participant: {participant.identity}")
            starting_task.cancel()

        session = self._sessions.pop(participant.identity, None)
        if session is None:
            return

        logger.info(f"Participant left: {participant.identity}")
        task = asyncio.create_task(self._close_session(participant.identity, session))
        self._close_tasks.add(task)

        def on_done(t: asyncio.Task):
            try:
                if not t.cancelled():
                    t.result()
            except Exception as e:
                logger.error(
                    f"Error closing session for {participant.identity}: {e}",
                    exc_info=True,
                )
            finally:
                self._close_tasks.discard(t)

        task.add_done_callback(on_done)

    async def _start_session(self, participant: rtc.RemoteParticipant) -> AgentSession:
        """
        Creates and starts an AgentSession for one participant.
        audio_input=True  → listen to their mic
        audio_output=False → don't speak back to them
        text_input=False  → don't accept chat messages (not needed)
        """
        if participant.identity in self._sessions:
            return self._sessions[participant.identity]

        session = AgentSession(vad=self.ctx.proc.userdata["vad"],min_endpointing_delay=1.5,)

        await session.start(
            agent=Transcriber(
                participant_identity=participant.identity,
                session_manager=self,
                stt=self.ctx.proc.userdata["stt"],
            ),
            room=self.ctx.room,
            room_options=room_io.RoomOptions(
                audio_input=True,
                text_output=True,
                audio_output=False,
                participant_identity=participant.identity,
                close_on_disconnect=False,
                text_input=False,
            ),
        )
        return session

    async def _close_session(
        self,
        participant_identity: str,
        session: AgentSession,
    ) -> None:
        logger.info(
            "Closing session for participant=%s",
            participant_identity,
        )

        #
        # Force any buffered speech to become a completed turn.
        #
        try:
            logger.debug(
                "Committing final user turn "
                "for participant=%s",
                participant_identity,
            )

            await session.commit_user_turn(
                skip_reply=True,
            )

            logger.debug(
                "Final user turn committed "
                "for participant=%s",
                participant_identity,
            )

        except Exception:
            logger.exception(
                "Failed to commit final user turn "
                "for participant=%s",
                participant_identity,
            )

        #
        # Give pending STT finals and callbacks
        # a brief chance to complete.
        #
        try:
            await asyncio.sleep(0.75)
        except Exception:
            pass

        #
        # Drain remaining tasks.
        #
        try:
            logger.debug(
                "Draining session "
                "for participant=%s",
                participant_identity,
            )

            await session.drain()

            logger.debug(
                "Session drained "
                "for participant=%s",
                participant_identity,
            )

        except Exception:
            logger.exception(
                "Failed to drain session "
                "for participant=%s",
                participant_identity,
            )

        #
        # Close the AgentSession.
        #
        try:
            logger.debug(
                "Closing AgentSession "
                "for participant=%s",
                participant_identity,
            )

            await session.aclose()

            logger.debug(
                "AgentSession closed "
                "for participant=%s",
                participant_identity,
            )

        except Exception:
            logger.exception(
                "Failed to close session "
                "for participant=%s",
                participant_identity,
            )