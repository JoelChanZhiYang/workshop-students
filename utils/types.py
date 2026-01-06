from pydantic import BaseModel

# Type alias for preprocessed chunks
# chunk_id -> {"content": str, "embedding": list[float] | None, ...}
Chunks = dict[str, dict]


class RetrievalResult(BaseModel):
    """Result from a retrieval function."""

    sources: list[str] = []  # chunk_ids that were retrieved
    metadata: dict = {}  # scores, timing, etc.


class GenerationResult(BaseModel):
    """Result from a generation function."""

    answer: str
    metadata: dict = {}  # model, agent_type, etc.


class QueryResult(BaseModel):
    """Deprecated: Use RetrievalResult or GenerationResult instead."""

    answer: str = ""
    sources: list[str] = []  # chunk_ids that were retrieved/used
    metadata: dict = {}  # scores, reasoning, timing, etc.
