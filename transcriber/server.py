import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from transcriber.agent import SessionManager, FasterWhisperSTT
from transcriber.db import create_pool, ensure_schema, upsert_meeting
import asyncio
import logging

from dotenv import load_dotenv
from livekit.agents import AgentServer, AutoSubscribe, JobContext, JobProcess, cli
from livekit.plugins import silero

load_dotenv()

logger = logging.getLogger("transcriber.server")

server = AgentServer(
    num_idle_processes=1,
    initialize_process_timeout=60.0,
)


@server.rtc_session(agent_name="transcriber")
async def entrypoint(ctx: JobContext):
    # Create DB pool here instead of prewarm — same event loop guaranteed.
    # Cache it in proc.userdata so that it is reused across meetings in the same process.
    db_pool = ctx.proc.userdata.get("db_pool")
    if db_pool is None:
        db_pool = await create_pool()
        await ensure_schema(db_pool)
        ctx.proc.userdata["db_pool"] = db_pool

    meeting_id = ctx.room.name
    await upsert_meeting(db_pool, meeting_id)
    logger.info(f"Transcription started for meeting: {meeting_id}")

    session_manager = SessionManager(
        ctx=ctx,
        db_pool=db_pool,
        meeting_id=meeting_id,
    )
    session_manager.start()

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    for participant in ctx.room.remote_participants.values():
        session_manager.on_participant_connected(participant)

    ctx.add_shutdown_callback(session_manager.aclose)


def prewarm(proc: JobProcess):
    # Load VAD and STT here to avoid event loop issues and speed up session startup
    logger.info("Prewarming worker...")
    proc.userdata["vad"] = silero.VAD.load()

    model_size = os.getenv("WHISPER_MODEL_SIZE", "base")
    device = os.getenv("WHISPER_DEVICE", "cuda")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
    local_files_only = os.getenv("WHISPER_LOCAL_FILES_ONLY", "true").lower() == "true"

    logger.info(f"Initializing local FasterWhisperSTT (model={model_size}, device={device}, compute_type={compute_type}, local_files_only={local_files_only})...")
    try:
        stt = FasterWhisperSTT(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            local_files_only=local_files_only,
        )
    except Exception as e:
        logger.warning(f"Failed to initialize FasterWhisperSTT on {device} ({e}). Falling back to CPU/int8...")
        stt = FasterWhisperSTT(
            model_size=model_size,
            device="cpu",
            compute_type="int8",
            local_files_only=local_files_only,
        )
        device = "cpu"
        compute_type = "int8"

    logger.info(
        f"Whisper loaded successfully: "
        f"{model_size=} {device=} {compute_type=}"
    )
    proc.userdata["stt"] = stt
    logger.info("Worker ready.")


server.setup_fnc = prewarm

if __name__ == "__main__":
    cli.run_app(server)