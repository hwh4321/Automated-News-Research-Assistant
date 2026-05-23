"""ChromaDB-backed vector store for semantic search over research history."""
import os
from pathlib import Path
from typing import Any

from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from config import config

os.environ.setdefault("HF_HUB_OFFLINE", "1")


class VectorStore:
    """Wraps ChromaDB for storing and querying research embeddings."""

    def __init__(self) -> None:
        Path(config.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = PersistentClient(path=config.chroma_persist_dir)
        self._ef = SentenceTransformerEmbeddingFunction(
            model_name=config.embedding_model_path,
        )
        self._collection = self._client.get_or_create_collection(
            name="research_docs",
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, docs: list[str], metadatas: list[dict[str, Any]] | None = None,
            ids: list[str] | None = None) -> None:
        self._collection.add(documents=docs, metadatas=metadatas, ids=ids)

    def query(self, query: str, *, n_results: int = 5) -> list[dict[str, Any]]:
        results = self._collection.query(query_texts=[query], n_results=n_results)
        metas = results["metadatas"][0] or [{}] * len(results["ids"][0])
        return [
            {"id": did, "text": doc, "metadata": meta, "score": 1 - dist}
            for did, doc, meta, dist in zip(
                results["ids"][0], results["documents"][0], metas, results["distances"][0],
            )
        ]

    def count(self) -> int:
        return self._collection.count()


vector_store = VectorStore()
