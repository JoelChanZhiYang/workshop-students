# Run tests: uv run pytest step1_retrieval/exercise_2/test_retrieval.py

import asyncio

from tests.test_runner import TestChunk
from utils import cosine_similarity_batch, get_embedding
from utils.types import Chunks, RetrievalResult


async def preprocess(test_chunks: list[TestChunk]) -> Chunks:
    """Preprocess chunks by computing embeddings for each.

    For each chunk, get its embedding and store it along with the original content.

    Args:
        test_chunks: List of TestChunk objects

    Returns:
        Chunks dictionary where each entry has:
        - Key: chunk_id (str)
        - Value: dict with "content" and "embedding" keys

    API:
        embedding = await get_embedding("some text")  # returns list[float]
    """
    # TODO: Embed each chunk and store in dictionary
    raise NotImplementedError("Students need to implement preprocess()")


async def retrieve(question: str, chunks: Chunks, top_k: int = 3) -> RetrievalResult:
    """Retrieve most relevant chunks using embedding similarity.

    Embed the question, then find chunks with the most similar embeddings.

    Args:
        question: The search query
        chunks: Preprocessed chunks dictionary (with embeddings)
        top_k: Number of top results to return

    Returns:
        RetrievalResult with sources (list of chunk_ids for top_k chunks)

    API:
        embedding = await get_embedding("some text")  # returns list[float]
        scores = cosine_similarity_batch(query_emb, [emb1, emb2, ...])  # returns list[float]
    """
    # TODO: Embed query, compute similarities, return top_k chunks
    raise NotImplementedError("Students need to implement retrieve()")
