import logging
import os
from typing import Any, Sequence

from cashews import cache
from dotenv import load_dotenv
from leakybucket.bucket import AsyncLeakyBucket
from leakybucket.persistence.memory import InMemoryLeakyBucketStorage
from llama_index.core.base.llms.types import ChatResponseAsyncGen
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import ChatMessage, ChatResponse, CompletionResponse
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

logger = logging.getLogger(__name__)

load_dotenv()

# Setup filesystem cache
cache.setup("disk://?directory=.cache")

# Create rate limiters first (needed before LLM initialization)
# Embeddings: 500 requests per minute (OpenAI's tier 1 limit)
embedding_storage = InMemoryLeakyBucketStorage(max_rate=500, time_period=60)
embedding_rate_limiter = AsyncLeakyBucket(embedding_storage)

# Chat completions: 30 requests per minute (gpt-4o-mini tier 1 limit)
chat_storage = InMemoryLeakyBucketStorage(max_rate=30, time_period=60)
chat_rate_limiter = AsyncLeakyBucket(chat_storage)


class RateLimitedLLM(OpenAI):
    """OpenAI LLM wrapper with rate limiting and caching."""

    def __init__(self, rate_limiter: AsyncLeakyBucket, **kwargs):
        super().__init__(**kwargs)
        self._rate_limiter = rate_limiter

    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        preview = (messages[-1].content or "")[:20] if messages else ""
        logger.debug(f"Calling achat: {preview}...")
        async with self._rate_limiter:
            return await super().achat(messages, **kwargs)

    async def acomplete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse:
        logger.debug(f"Calling acomplete: {prompt[:20]}...")
        async with self._rate_limiter:
            return await super().acomplete(prompt, formatted, **kwargs)


class RateLimitedEmbedding(OpenAIEmbedding):
    """OpenAI Embedding wrapper with rate limiting and caching."""

    def __init__(self, rate_limiter: AsyncLeakyBucket, **kwargs):
        super().__init__(**kwargs)
        self._rate_limiter = rate_limiter

    @cache(ttl="999d", key="embedding:{text}")
    async def aget_text_embedding(self, text: str) -> list[float]:
        async with self._rate_limiter:
            logger.debug(f"Cache MISS - Calling aget_text_embedding: {text[:20]}...")
            result = await super().aget_text_embedding(text)
            return result


# Initialize rate-limited LLM
llm = RateLimitedLLM(
    rate_limiter=chat_rate_limiter,
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
    timeout=30.0,
    temperature=0.0,
)

# Initialize rate-limited embedding model (with built-in caching)
embed_model = RateLimitedEmbedding(
    rate_limiter=embedding_rate_limiter,
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
    timeout=30.0,
)


async def get_embedding(text: str) -> list[float]:
    logger.debug(f"Getting embedding: {text[:20]}...")
    return await embed_model.aget_text_embedding(text)


@cache(ttl="999d", key="completion:{prompt}")
async def generate_completion(prompt: str, system_message: str | None = None) -> str:
    logger.debug(f"Generating completion: {prompt[:20]}...")
    messages = []

    if system_message:
        messages.append(ChatMessage(role="system", content=system_message))

    messages.append(ChatMessage(role="user", content=prompt))

    # Use LlamaIndex's async chat method
    response = await llm.achat(messages)

    return response.message.content or ""
