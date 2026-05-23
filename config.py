"""Configuration management using environment variables with sensible defaults."""
import os
from dataclasses import dataclass
from functools import cached_property

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    """Application config loaded from environment variables."""

    # LLM
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o")

    # Search
    search_api_key: str = os.getenv("SEARCH_API_KEY", "")
    search_engine: str = os.getenv("SEARCH_ENGINE", "serpapi")

    # Email
    smtp_host: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_user: str = os.getenv("SMTP_USER", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")

    # Memory
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
    sqlite_path: str = os.getenv("SQLITE_PATH", "./data/research.db")
    embedding_model_path: str = os.getenv(
        "EMBEDDING_MODEL_PATH",
        "shibing624/text2vec-base-chinese",  # 默认从 HuggingFace 在线加载
    )

    # Redis (optional, falls back to in-memory)
    redis_url: str = os.getenv("REDIS_URL", "")
    session_ttl: int = int(os.getenv("SESSION_TTL", "3600"))

    # Limits
    max_search_results: int = int(os.getenv("MAX_SEARCH_RESULTS", "10"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))

    @cached_property
    def redis_enabled(self) -> bool:
        return bool(self.redis_url)


config = Config()
