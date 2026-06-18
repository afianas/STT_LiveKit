# create_room.py
import asyncio
from livekit import api
import os
from dotenv import load_dotenv

load_dotenv()

async def create_room(room_name: str):
    lk = api.LiveKitAPI(
        url=os.getenv("LIVEKIT_URL"),
        api_key=os.getenv("LIVEKIT_API_KEY"),
        api_secret=os.getenv("LIVEKIT_API_SECRET"),
    )
    room = await lk.room.create_room(
        api.CreateRoomRequest(name=room_name)
    )
    print(f"Room created: {room.name}")
    await lk.aclose()

asyncio.run(create_room("test-meeting-1"))