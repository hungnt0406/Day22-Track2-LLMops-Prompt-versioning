import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration variables
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "default_project")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")

def verify_config():
    """Verifies that the configuration is loaded correctly and prints details."""
    missing_vars = []
    if not LANGCHAIN_API_KEY:
        missing_vars.append("LANGCHAIN_API_KEY")
    if not OPENAI_API_KEY:
        missing_vars.append("OPENAI_API_KEY")
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please ensure they are set in your .env file.")
        return False

    print("✅ Config loaded successfully")
    print(f"   LangSmith project : {LANGCHAIN_PROJECT}")
    print(f"   OpenAI endpoint   : {OPENAI_BASE_URL}")
    print(f"   Default LLM model : {OPENAI_MODEL_NAME}")
    print(f"   Embedding model   : {EMBEDDING_MODEL_NAME}")
    return True

if __name__ == "__main__":
    verify_config()
