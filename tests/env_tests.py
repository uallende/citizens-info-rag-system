from dotenv import load_dotenv
import os

load_dotenv()  # Load the .env file
hf_token = os.getenv("HUGGINGFACE_TOKEN")
print(hf_token)