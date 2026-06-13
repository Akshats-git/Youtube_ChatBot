import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Centralised configuration loaded from environment variables."""

    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Models
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", 0.7))
    QUERY_TEMPERATURE: float = float(os.getenv("QUERY_TEMPERATURE", 0.0))
    SUMMARY_TEMPERATURE: float = float(os.getenv("SUMMARY_TEMPERATURE", 0.2))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", 500))
    SUMMARY_MAX_TOKENS: int = int(os.getenv("SUMMARY_MAX_TOKENS", 300))

    # Text processing
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 200))
    SNIPPET_MAX_CHARS: int = int(os.getenv("SNIPPET_MAX_CHARS", 1200))

    # Retrieval
    RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", 6))
    BM25_K: int = int(os.getenv("BM25_K", 6))
    FINAL_K: int = int(os.getenv("FINAL_K", 6))
    MULTIQUERY_COUNT: int = int(os.getenv("MULTIQUERY_COUNT", 3))
    MAX_CONTEXT_CHARS: int = int(os.getenv("MAX_CONTEXT_CHARS", 8000))
    ENABLE_QUERY_REWRITE: bool = os.getenv("ENABLE_QUERY_REWRITE", "true").lower() == "true"
    ENABLE_MULTIQUERY: bool = os.getenv("ENABLE_MULTIQUERY", "true").lower() == "true"
    ENABLE_COMPRESSION: bool = os.getenv("ENABLE_COMPRESSION", "true").lower() == "true"

    # Persistence
    INDEX_DIR: str = os.getenv("INDEX_DIR", "data/indexes")

    # Summarisation
    SUMMARY_MAX_CHUNKS: int = int(os.getenv("SUMMARY_MAX_CHUNKS", 6))

    @staticmethod
    def validate() -> bool:
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set. Please add it to your .env file.")
        return True
