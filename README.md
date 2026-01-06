# RAG Workshop

A hands-on workshop for building Retrieval-Augmented Generation (RAG) systems, progressing from simple word-overlap retrieval to agentic RAG with ReAct agents.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) package manager (handles Python installation automatically)
- OpenAI API key (or compatible endpoint)

## Installation

1. **Install uv** (if not already installed):

   Follow the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/) for your operating system.

   Verify the installation:
   ```bash
   uv --version
   ```

2. **Clone the repository:**
   ```bash
   git clone https://github.com/JoelChanZhiYang/workshop-students.git
   cd workshop-students
   ```

3. **Install dependencies:**
   ```bash
   uv sync
   ```
   This will automatically download Python 3.14 and install all dependencies.

4. **Configure environment:**
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your-api-key-here
   # Optional: use a custom endpoint
   # OPENAI_BASE_URL=https://your-endpoint.com/v1
   ```

5. **Verify setup:**
   ```bash
   uv run python -c "from utils import get_embedding, generate_completion; print('Setup OK')"
   ```
   If this fails, check that your `.env` file exists and contains a valid `OPENAI_API_KEY`.

## Workshop Structure

> **Note:** Exercises build on each other. Complete them in order (Step 1 → Step 2).

### Step 1: Retrieval

| Exercise | Description | Command |
|----------|-------------|---------|
| Exercise 1 | Simple word-overlap retrieval | `uv run pytest step1_retrieval/exercise_1/test_retrieval.py` |
| Exercise 2 | Embedding-based retrieval | `uv run pytest step1_retrieval/exercise_2/test_retrieval.py` |

### Step 2: Generation

| Exercise | Description | Command |
|----------|-------------|---------|
| Exercise 1 | Simple RAG (retrieve + generate) | `uv run python -m step2_generation.exercise_1.demo_rag` |
| Exercise 2 | Agentic RAG with ReAct | `uv run pytest step2_generation/exercise_2/test_retrieval.py` |

## Running Tests

Run all tests:
```bash
uv run pytest
```

Run a specific exercise:
```bash
uv run pytest step1_retrieval/exercise_1/test_retrieval.py -v
```

Use `-s` to see console output (print statements, logs, etc.):
```bash
uv run pytest step1_retrieval/exercise_1/test_retrieval.py -vs
```

## API Reference

The `utils` module provides helper functions for exercises:

```python
from utils import get_embedding, generate_completion, cosine_similarity_batch

# Get embeddings
embedding = await get_embedding("some text")  # returns list[float]

# Generate completions
answer = await generate_completion("your prompt here")

# Compute similarity scores
scores = cosine_similarity_batch(query_emb, [emb1, emb2, ...])  # returns list[float]
```
