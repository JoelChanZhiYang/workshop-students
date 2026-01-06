import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Awaitable, Callable, Literal

from pydantic import BaseModel

from utils.types import Chunks, QueryResult

# Configure logging from environment variable
log_level = os.getenv("LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


class TestChunk(BaseModel):
    chunk_id: str
    chunk_content: str
    labels: list[str] | None = None


class TestCase(BaseModel):
    query: str
    expected_chunk_ids: list[str]
    expected_chunks: list[str]
    difficulty: str | None = None
    labels: list[str] | None = None
    note: str | None = None


def get_chunks(dataset_name: str) -> list[TestChunk]:
    """Load chunks for a specific dataset."""
    datasets_dir = Path(__file__).parent / "data"

    data_file_path = datasets_dir / f"{dataset_name}__chunks.json"

    with data_file_path.open("r") as fp:
        data = json.load(fp)
        return [TestChunk.model_validate(chunk) for chunk in data]


def get_test_cases(dataset_name: str) -> list[TestCase]:
    """Load test cases for a specific dataset."""
    datasets_dir = Path(__file__).parent / "data"
    tests_file_path = datasets_dir / f"{dataset_name}__tests.json"

    if not tests_file_path.exists():
        raise FileNotFoundError(f"Test file not found for dataset: {dataset_name}")

    with tests_file_path.open("r") as fp:
        test_data = json.load(fp)
        return [TestCase.model_validate(case) for case in test_data]
