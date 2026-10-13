"""Structural validation for the expanded golden test set.

Guards the config/learning_golden_test_set.json contract without touching
the live vector store: schema, non-empty ground truth, unique queries, and
the deterministic chunk-ID format ({doc_hash8}_{index:04d}_{hash8}).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

GOLDEN_SET = Path("config/learning_golden_test_set.json")
CHUNK_ID_PATTERN = re.compile(r"^[a-f0-9]{8}_\d{4}_[a-f0-9]{8}$")


@pytest.fixture
def golden_set(project_root: Path) -> dict:
    """Load the golden test set JSON."""
    path = project_root / GOLDEN_SET
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class TestGoldenSetStructure:
    def test_file_has_expected_version(self, golden_set: dict) -> None:
        assert golden_set["version"] == "2.0"

    def test_expanded_to_at_least_30_cases(self, golden_set: dict) -> None:
        assert len(golden_set["test_cases"]) >= 30

    def test_every_case_has_required_fields(self, golden_set: dict) -> None:
        for case in golden_set["test_cases"]:
            assert case["query"].strip(), "query must be non-empty"
            assert case["expected_chunk_ids"], (
                f"expected_chunk_ids must be non-empty for: {case['query'][:40]}"
            )
            assert isinstance(case["expected_sources"], list)
            assert case.get("reference_answer"), (
                f"reference_answer required for: {case['query'][:40]}"
            )

    def test_queries_are_unique(self, golden_set: dict) -> None:
        queries = [case["query"] for case in golden_set["test_cases"]]
        assert len(queries) == len(set(queries)), "duplicate query in golden set"

    def test_chunk_ids_match_deterministic_format(self, golden_set: dict) -> None:
        for case in golden_set["test_cases"]:
            for chunk_id in case["expected_chunk_ids"]:
                assert CHUNK_ID_PATTERN.match(chunk_id), (
                    f"malformed chunk id: {chunk_id}"
                )

    def test_all_chunks_belong_to_the_manual_doc(self, golden_set: dict) -> None:
        for case in golden_set["test_cases"]:
            for chunk_id in case["expected_chunk_ids"]:
                assert chunk_id.startswith("c2bca2af_"), (
                    f"foreign chunk id: {chunk_id}"
                )
