import os
from dotenv import load_dotenv

load_dotenv()

chatgpt_api_key = os.getenv('API_KEY')

print(chatgpt_api_key)