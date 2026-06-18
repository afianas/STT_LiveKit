import asyncio
import os
from dotenv import load_dotenv
from livekit import api

load_dotenv()

async def dispatch():
    lk = api.LiveKitAPI(
        url=os.getenv("LIVEKIT_URL"),
        api_key=os.getenv("LIVEKIT_API_KEY"),
        api_secret=os.getenv("LIVEKIT_API_SECRET"),
    )
    await lk.agent_dispatch.create_dispatch(
        api.CreateAgentDispatchRequest(
            agent_name="transcriber",
            room="test-meeting-1",
        )
    )
    print("Agent dispatched!")
    await lk.aclose()

asyncio.run(dispatch())