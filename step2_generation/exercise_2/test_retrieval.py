import pytest
import pytest_asyncio

import step2_generation.exercise_2.agentic_rag_exercise as agentic_rag
from step2_generation.exercise_2.agentic_rag_exercise import generate, preprocess
from tests.test_runner import get_chunks, get_test_cases

# Only test agentic difficulty tests for this exercise
_all_test_cases = get_test_cases("student_tech")
_test_cases = [tc for tc in _all_test_cases if tc.difficulty == "agentic"]


@pytest_asyncio.fixture(scope="module")
async def chunks():
    """Preprocess chunks once for all tests."""
    chunks_data = get_chunks("student_tech")
    return await preprocess(chunks_data)


@pytest.mark.asyncio
@pytest.mark.parametrize("test_case", _test_cases, ids=lambda tc: tc.query[:50])
async def test_retrieval(test_case, chunks, mocker):
    """Test that the agentic RAG retrieves the expected chunks for each query."""
    # Spy on retrieve where it's defined in agentic_rag module
    spy = mocker.spy(agentic_rag, "retrieve")

    # Run the agentic generate (retrieval happens inside the agent)
    await generate(test_case.query, chunks)

    # Collect all sources from all retrieve calls the agent made
    all_sources = {src for r in spy.spy_return_list for src in r.sources}
    assert set(test_case.expected_chunk_ids) <= all_sources, (
        f"Expected: {test_case.expected_chunk_ids}\nGot: {list(all_sources)}"
    )
