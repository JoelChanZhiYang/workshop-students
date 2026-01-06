"""
Exercise 1: Simple RAG

A basic Retrieval-Augmented Generation system that combines
embedding-based retrieval with LLM answer generation.
"""

from step2_generation.exercise_1.simple_rag_exercise import generate, preprocess, retrieve

__all__ = ["preprocess", "retrieve", "generate"]
