import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential

load_dotenv('.env')

# AZURE AI SEARCH CREDENTIALS
searchservice = os.environ.get('searchservice')
index_name = os.environ.get('index')
searchkey = os.environ.get('searchkey')

# OPENAI CONFIGURATION
openai_api_key = os.environ.get('openai_api_key')

# DATA CONFIGURATION
filepath = os.environ.get('filepath')

# set credentials
search_creds = AzureKeyCredential(searchkey)