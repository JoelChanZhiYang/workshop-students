"""
Exercise 2: Agentic RAG with LlamaIndex

This module demonstrates building an agentic RAG system that can autonomously
decide when and how to use retrieval tools to answer complex questions.
"""

from step2_generation.exercise_2.agentic_rag_exercise import (
    generate,
    preprocess,
    retrieve,
)

__all__ = [
    "preprocess",
    "retrieve",
    "generate",
]
