from transcriber.agent.transcriber import Transcriber
from transcriber.db import close_meeting
import asyncio
import logging

import asyncpg
from livekit import rtc
from livekit.agents import AgentSession, JobContext, room_io, utils


logger = logging.getLogger("transcriber.session_manager")


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
    - Close the meeting in DB when the room shuts down
    """

    def __init__(
        self,
        ctx: JobContext,
        db_pool: asyncpg.Pool,
        meeting_id: str,
    ):
        self.ctx = ctx
        self.db_pool = db_pool
        self.meeting_id = meeting_id

        # Maps participant identity -> their active AgentSession
        self._sessions: dict[str, AgentSession] = {}

        # Tracks in-flight async tasks so we can cancel them on shutdown
        self._tasks: set[asyncio.Task] = set()

    def start(self):
        """Register event listeners for the room."""
        self.ctx.room.on("participant_connected", self.on_participant_connected)
        self.ctx.room.on("participant_disconnected", self.on_participant_disconnected)

    async def aclose(self):
        """
        Graceful shutdown:
        1. Cancel any pending session-start tasks
        2. Drain and close all active sessions
        3. Unregister room event listeners
        4. Mark meeting as ended in DB
        """
        await utils.aio.cancel_and_wait(*self._tasks)

        await asyncio.gather(
            *[self._close_session(s) for s in self._sessions.values()]
        )

        self.ctx.room.off("participant_connected", self.on_participant_connected)
        self.ctx.room.off("participant_disconnected", self.on_participant_disconnected)

        await close_meeting(self.db_pool, self.meeting_id)
        logger.info(f"Meeting {self.meeting_id} ended.")

    def on_participant_connected(self, participant: rtc.RemoteParticipant):
        """
        Fires when someone joins the LiveKit room.
        Starts a transcription session for them asynchronously.
        """
        if participant.identity in self._sessions:
            return  # already transcribing them, skip

        logger.info(f"Participant joined: {participant.identity}")
        task = asyncio.create_task(self._start_session(participant))
        self._tasks.add(task)

        def on_done(t: asyncio.Task):
            try:
                if not t.cancelled():
                    # Store the completed session so we can close it later
                    self._sessions[participant.identity] = t.result()
            except Exception as e:
                logger.error(f"Error starting session for {participant.identity}: {e}", exc_info=True)
            finally:
                self._tasks.discard(t)

        task.add_done_callback(on_done)

    def on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        """
        Fires when someone leaves the LiveKit room.
        Closes their transcription session.
        """
        session = self._sessions.pop(participant.identity, None)
        if session is None:
            return

        logger.info(f"Participant left: {participant.identity}")
        task = asyncio.create_task(self._close_session(session))
        self._tasks.add(task)

        def on_done(t: asyncio.Task):
            try:
                if not t.cancelled():
                    t.result()
            except Exception as e:
                logger.error(f"Error closing session for {participant.identity}: {e}", exc_info=True)
            finally:
                self._tasks.discard(t)

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

        session = AgentSession(vad=self.ctx.proc.userdata["vad"])

        await session.start(
            agent=Transcriber(
                participant_identity=participant.identity,
                meeting_id=self.meeting_id,
                db_pool=self.db_pool,
                stt=self.ctx.proc.userdata["stt"],
            ),
            room=self.ctx.room,
            room_options=room_io.RoomOptions(
                audio_input=True,
                text_output=True,
                audio_output=False,
                participant_identity=participant.identity,
                text_input=False,
            ),
        )
        return session

    async def _close_session(self, session: AgentSession) -> None:
        """
        drain() waits for any in-progress transcription to finish.
        aclose() then releases all resources.
        """
        await session.drain()
        await session.aclose()