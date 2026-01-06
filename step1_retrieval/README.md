# Step 1: Retrieval

Build retrieval systems that find relevant chunks from a knowledge base.

## Exercise 1: Word-Overlap Retrieval

**File:** `exercise_1/simple_retrieval_exercise.py`

**Implement:**
- `preprocess()` - Convert chunks to a searchable dictionary format
- `retrieve()` - Score chunks by counting word overlaps with the query

**Hints:**
- Use a dictionary comprehension for `preprocess()`
- `defaultdict` is useful for counting scores
- Don't forget to handle case sensitivity
- Check the docstrings for expected input/output formats

**Test:**
```bash
uv run pytest step1_retrieval/exercise_1/test_retrieval.py -vs
```

---

## Exercise 2: Embedding Retrieval

**File:** `exercise_2/embedding_retrieval_exercise.py`

**Implement:**
- `preprocess()` - Compute and store embeddings for each chunk
- `retrieve()` - Find chunks with most similar embeddings to the query

**Hints:**
- Store both content and embedding for each chunk
- Higher cosine similarity = more relevant
- Remember to sort and return the top_k
- Check the docstrings for expected input/output formats

**API:**
```python
from utils import get_embedding, cosine_similarity_batch

embedding = await get_embedding("some text")  # returns list[float]
scores = cosine_similarity_batch(query_emb, [emb1, emb2, ...])  # returns list[float]
```

**Test:**
```bash
uv run pytest step1_retrieval/exercise_2/test_retrieval.py -vs
```
