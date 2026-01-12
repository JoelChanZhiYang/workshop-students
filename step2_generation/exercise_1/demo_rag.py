"""Demo: Compare generation with and without retrieval."""

import asyncio

from step2_generation.exercise_1.simple_rag_exercise import generate, preprocess
from tests.test_runner import get_chunks, get_test_cases
from utils import generate_completion

# Queries designed to show RAG value:
# - Notice: LLM might know these facts, but RAG provides:
#   1. Grounded answers with source citations
#   2. Concise responses (no verbose explanations)
#   3. Verifiable information from specific chunks
DEMO_QUERIES = [
    # Specific number - compare precision
    "What is the exact per-stream payment range for Spotify artists?",
    # 2023 event - test recency
    "How many users did Threads gain in its first five days?",
    # Technical detail - LLM might give generic answer
    "What operating system does the Steam Deck run?",
    # Specific feature name - test exact terminology
    "What feature did Twitter add in 2022 to combat misinformation?",
    # Cross-platform comparison - requires grounding
    "How do Minecraft and Roblox differ in how they let players create content?",
]


async def generate_without_retrieval(question: str) -> str:
    """Ask the LLM directly without any context."""
    prompt = f"Question: {question}\n\nAnswer:"
    return await generate_completion(prompt)


async def generate_with_retrieval(question: str, chunks: dict) -> str:
    """Use RAG: retrieve relevant context, then generate."""
    result = await generate(question, chunks)
    return result


async def main():
    print("=" * 80)
    print("RAG Demo: Generation With vs Without Retrieval")
    print("=" * 80)

    # Preprocess chunks once
    chunks_data = get_chunks("student_tech")
    chunks = await preprocess(chunks_data)

    for i, query in enumerate(DEMO_QUERIES, 1):
        print(f"\n{'=' * 80}")
        print(f"Query {i}: {query}")
        print("=" * 80)

        # Without retrieval
        answer_no_rag = await generate_without_retrieval(query)
        print(f"\n[WITHOUT RETRIEVAL]")
        print(f"{answer_no_rag}")

        # With retrieval
        answer_rag = await generate_with_retrieval(query, chunks)
        print(f"\n[WITH RETRIEVAL]")
        print(f"{answer_rag}")


if __name__ == "__main__":
    asyncio.run(main())
