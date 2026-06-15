import asyncio
import logging

from dotenv import load_dotenv
from livekit.agents import AgentServer, AutoSubscribe, JobContext, JobProcess, cli
from livekit.plugins import silero

from transcriber.agent import SessionManager
from transcriber.db import create_pool, ensure_schema, upsert_meeting

load_dotenv()

logger = logging.getLogger("transcriber.server")

server = AgentServer()


@server.rtc_session(agent_name="transcriber")
async def entrypoint(ctx: JobContext):
    """
    Called once per LiveKit room the agent is dispatched into.

    Flow:
    1. Get/Create the DB pool inside entrypoint (on the correct event loop)
    2. Register this room as a meeting in the DB
    3. Start the session manager (watches for participants)
    4. Connect to the room (audio only — no video needed)
    5. Handle anyone already in the room before we joined
    6. Register cleanup to run when the room closes
    """
    db_pool = ctx.proc.userdata.get("db_pool")
    if db_pool is None:
        db_pool = await create_pool()
        await ensure_schema(db_pool)
        ctx.proc.userdata["db_pool"] = db_pool

    meeting_id = ctx.room.name  # LiveKit room name = our meeting ID

    await upsert_meeting(db_pool, meeting_id)
    logger.info(f"Transcription started for meeting: {meeting_id}")

    session_manager = SessionManager(
        ctx=ctx,
        db_pool=db_pool,
        meeting_id=meeting_id,
    )
    session_manager.start()

    # AUDIO_ONLY — we don't need video, saves bandwidth
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Handle participants already in the room when agent joins
    for participant in ctx.room.remote_participants.values():
        session_manager.on_participant_connected(participant)

    ctx.add_shutdown_callback(session_manager.aclose)


def prewarm(proc: JobProcess):
    """
    Runs once when a worker process starts up — before any rooms are joined.
    Good place to load heavy resources so they're ready instantly.

    We load:
    - VAD (voice activity detection): detects when someone starts/stops speaking
    """
    logger.info("Prewarming worker...")

    # Silero VAD — lightweight ML model that detects speech vs silence
    proc.userdata["vad"] = silero.VAD.load()

    logger.info("Worker ready.")


server.setup_fnc = prewarm

if __name__ == "__main__":
    cli.run_app(server)