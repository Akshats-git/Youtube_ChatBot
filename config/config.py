import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the YouTube ChatBot application."""
    
    # OpenAI API Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # Model Configuration
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    TEMPERATURE = 0.7
    QUERY_TEMPERATURE = float(os.getenv("QUERY_TEMPERATURE", 0.0))
    
    # Text Processing Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

    # Retrieval Configuration
    RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", 6))
    BM25_K = int(os.getenv("BM25_K", 6))
    FINAL_K = int(os.getenv("FINAL_K", 6))
    MULTIQUERY_COUNT = int(os.getenv("MULTIQUERY_COUNT", 3))
    MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", 8000))
    ENABLE_QUERY_REWRITE = os.getenv("ENABLE_QUERY_REWRITE", "true").lower() == "true"
    ENABLE_MULTIQUERY = os.getenv("ENABLE_MULTIQUERY", "true").lower() == "true"
    ENABLE_COMPRESSION = os.getenv("ENABLE_COMPRESSION", "true").lower() == "true"

    # Persistence
    INDEX_DIR = os.getenv("INDEX_DIR", "data/indexes")

    # Summarization
    SUMMARY_MAX_CHUNKS = int(os.getenv("SUMMARY_MAX_CHUNKS", 6))
    SUMMARY_MAX_TOKENS = int(os.getenv("SUMMARY_MAX_TOKENS", 300))
    
    # Application Settings
    MAX_TOKENS = 500
    
    @staticmethod
    def validate():
        """Validate that required configuration is present."""
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set. Please set it in your .env file.")
        return True
