import weaviate
import os
from dotenv import load_dotenv


load_dotenv()
openai_api_key = os.getenv("OPENAI_KEY")
wcs_url = os.getenv("WCS_URL")

client = weaviate.connect_to_wcs(
    cluster_url = wcs_url,
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WCS_API_KEY")),
    headers={
        "X-OpenAI-Api-Key": os.environ["OPENAI_APIKEY"]  # Replace with your inference API key
    }
)

try:
    pass # Replace with your code. Close client gracefully in the finally block.

finally:
    client.close()  # Close client gracefully