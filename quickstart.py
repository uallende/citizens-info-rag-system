import weaviate
import weaviate.classes as wvc
import os
import requests
import json
from dotenv import load_dotenv


load_dotenv()
openai_api_key = os.getenv("WCS_API_KEY")
wcs_api_key = os.getenv("WCS_API_KEY")

client = weaviate.connect_to_wcs(
    cluster_url=wcs_api_key,
    auth_credentials=weaviate.auth.AuthApiKey(wcs_api_key),
    headers={
        "X-OpenAI-Api-Key": openai_api_key  # Replace with your inference API key
    }
)

try:
    pass # Replace with your code. Close client gracefully in the finally block.

finally:
    client.close()  # Close client gracefully