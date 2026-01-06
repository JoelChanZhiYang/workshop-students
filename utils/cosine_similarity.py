import numpy as np


def cosine_similarity_batch(
    query_embedding: list[float], chunk_embeddings: list[list[float]]
) -> list[float]:
    # Convert to NumPy arrays for vectorized operations
    query_vec = np.array(query_embedding)
    chunk_vecs = np.array(chunk_embeddings)

    # For normalized embeddings, cosine similarity = dot product
    # This performs matrix-vector multiplication in optimized C code
    similarity_scores = chunk_vecs @ query_vec

    return similarity_scores.tolist()
