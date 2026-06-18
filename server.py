import os
import logging

from dotenv import load_dotenv
from livekit.agents import AgentServer, AutoSubscribe, JobContext, JobProcess, cli
from livekit.plugins import silero

from transcriber.agent import SessionManager, FasterWhisperSTT

load_dotenv()

logger = logging.getLogger("transcriber.server")

server = AgentServer(
    num_idle_processes=1,
    initialize_process_timeout=60.0,
)


@server.rtc_session(agent_name="transcriber")
async def entrypoint(ctx: JobContext):
    meeting_id = ctx.room.name
    logger.info(f"Transcription started for meeting: {meeting_id}")

    session_manager = SessionManager(
        ctx=ctx,
        meeting_id=meeting_id,
    )
    # Connect first so the participant list is fully populated before we
    # register event listeners. This prevents participant_connected from firing
    # for a participant who is also present in remote_participants, which would
    # cause _start_session to run twice for the same identity.
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    session_manager.start()  # register event handlers only after connect

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