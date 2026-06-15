import os
from dotenv import load_dotenv
from livekit import api

load_dotenv()

token = (
    api.AccessToken(
        api_key=os.getenv("LIVEKIT_API_KEY"),
        api_secret=os.getenv("LIVEKIT_API_SECRET"),
    )
    .with_identity("afia")
    .with_name("Afia")
    .with_grants(api.VideoGrants(room_join=True, room="test-meeting-1"))
    .to_jwt()
)

print("Your token:")
print(token)