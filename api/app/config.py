"""Runtime configuration, read once from environment variables.

On the cluster these variables come from your ConfigMap and Secret (section 4.1).
Nothing here is baked into the image, so one image runs in every environment.
Add your own keys (relevance floor, cache settings, ...) the same way.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _load_env_file() -> None:
    for env_path in [".env", "api/.env", "../api/.env"]:
        if os.path.exists(env_path):
            try:
                with open(env_path, encoding="utf-8") as f:
                    for line in f:
                        stripped = line.strip()
                        if stripped and not stripped.startswith("#") and "=" in stripped:
                            k, v = stripped.split("=", 1)
                            os.environ.setdefault(k.strip(), v.strip().strip("'\""))
                break
            except OSError:
                continue


_load_env_file()


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    # Which implementations to load. "stub" needs no cluster, no model, no network.
    retriever: str = "stub"          # RETRIEVER: stub | chroma
    llm_provider: str = "stub"       # LLM_PROVIDER: stub | ollama-host | ollama-incluster | hosted

    # The embedding pin (section 4.4). The indexer reads the same ConfigMap key.
    embed_model: str = "nomic-embed-text"

    # Vector store
    chroma_host: str = "chroma"
    chroma_port: int = 8000
    collection: str = "rahbar"

    # Retrieval and generation
    top_k: int = 5
    temperature: float = 0.2
    llm_model: str = "llama3.2:3b"
    ollama_url: str = "http://host.minikube.internal:11434"
    hosted_base_url: str = ""        # e.g. an OpenAI-compatible endpoint
    llm_api_key: str = field(default="", repr=False)   # from the Secret; never log it
    llm_timeout: float = 30.0
    relevance_floor: float = 0.5     # RELEVANCE_FLOOR: min score to avoid abstention

    # Semantic cache (section 3.3)
    redis_host: str = "redis"
    redis_port: int = 6379
    cache_threshold: float = 0.90    # Cosine similarity threshold for semantic cache hit
    cache_ttl_seconds: int = 86400   # 24 hours TTL
    enable_cache: bool = True

    # Vector dimensions for pin check
    embed_dim: int = 768

    log_level: str = "INFO"


def get_settings() -> Settings:
    return Settings(
        retriever=os.environ.get("RETRIEVER", "stub"),
        llm_provider=os.environ.get("LLM_PROVIDER", "stub"),
        embed_model=os.environ.get("EMBED_MODEL", "nomic-embed-text"),
        chroma_host=os.environ.get("CHROMA_HOST", "chroma"),
        chroma_port=_int("CHROMA_PORT", 8000),
        collection=os.environ.get("CHROMA_COLLECTION", "rahbar"),
        top_k=_int("TOP_K", 5),
        temperature=_float("TEMPERATURE", 0.2),
        llm_model=os.environ.get("LLM_MODEL", "llama3.2:3b"),
        ollama_url=os.environ.get("OLLAMA_URL", "http://host.minikube.internal:11434"),
        hosted_base_url=os.environ.get("HOSTED_BASE_URL", ""),
        llm_api_key=os.environ.get("LLM_API_KEY", ""),
        llm_timeout=_float("LLM_TIMEOUT", 30.0),
        relevance_floor=_float("RELEVANCE_FLOOR", 0.5),
        redis_host=os.environ.get("REDIS_HOST", "redis"),
        redis_port=_int("REDIS_PORT", 6379),
        cache_threshold=_float("CACHE_THRESHOLD", 0.90),
        cache_ttl_seconds=_int("CACHE_TTL_SECONDS", 86400),
        enable_cache=os.environ.get("ENABLE_CACHE", "true").lower() in ("1", "true", "yes"),
        embed_dim=_int("EMBED_DIM", 768),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
    )
