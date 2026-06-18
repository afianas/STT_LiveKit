import os
from dotenv import load_dotenv
from livekit import api

load_dotenv()

ROOM_NAME = "test-meeting-1"

users = ["Afia", "John", "Sarah", "Mike"]

for user in users:
    token = (
        api.AccessToken(
            api_key=os.getenv("LIVEKIT_API_KEY"),
            api_secret=os.getenv("LIVEKIT_API_SECRET"),
        )
        .with_identity(user.lower())
        .with_name(user)
        .with_grants(api.VideoGrants(
            room_join=True,
            room=ROOM_NAME,
            can_publish=True,
            can_subscribe=True,
        ))
        .to_jwt()
    )
    print(f"\n{user}:")
    print(token)