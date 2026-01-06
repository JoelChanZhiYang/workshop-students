"""
Shared utilities for the RAG workshop.
"""

from utils.cosine_similarity import cosine_similarity_batch
from utils.llm_utils import generate_completion, get_embedding
from utils.types import Chunks, GenerationResult, QueryResult, RetrievalResult

__all__ = [
    "get_embedding",
    "generate_completion",
    "cosine_similarity_batch",
    "Chunks",
    "RetrievalResult",
    "GenerationResult",
    "QueryResult",
]
