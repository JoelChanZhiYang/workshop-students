# Step 2: Generation

Build RAG systems that retrieve context and generate answers.

> **Prerequisite:** Complete Step 1 Exercise 2 first. These exercises use your embedding-based retrieval.

## Exercise 1: Simple RAG

**File:** `exercise_1/simple_rag_exercise.py`

**Implement:**
- `generate()` - Retrieve relevant chunks, build a prompt, and call the LLM

**Hints:**
- `retrieve()` returns a `RetrievalResult` with a `sources` list of chunk IDs
- Look up the actual content from the chunks dictionary using those IDs
- Build a prompt that gives the LLM context to answer the question
- Check the docstring for the expected return type

**API:**
```python
from utils import generate_completion

answer = await generate_completion("your prompt here")
```

**Run:**
```bash
uv run python -m step2_generation.exercise_1.demo_rag
```

---

## Exercise 2: Agentic RAG

**File:** `exercise_2/agentic_rag_exercise.py`

**Implement:**
- `generate()` - Create a ReAct agent that autonomously decides when to search

**Hints:**
- The agent needs a tool to search the knowledge base
- Think about what the tool description should say
- The agent handles the reasoning loop - you just need to set it up and call it
- Check the docstring for the expected return type

**API:**
```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from utils.llm_utils import llm
```

**Test:**
```bash
uv run pytest step2_generation/exercise_2/test_retrieval.py -vs
```
