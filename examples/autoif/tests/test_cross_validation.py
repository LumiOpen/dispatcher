"""Tests for cross-validation output format consistency.

The two CV paths (inline via verifiers_task.py and standalone via
verifiers_cross_validation.py) must produce output records with an identical
set of fields so that downstream steps can consume them interchangeably.
"""

import json
import os
import sys
import subprocess
import tempfile
from dataclasses import fields as dc_fields
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.utils.cross_validator import CrossValidationResult, run_cross_validation
from src.utils.function_executor import FunctionExecutor

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

GOOD_FUNC = """\
def evaluate(text):
    return len(text) > 0
"""

BAD_FUNC = """\
def evaluate(text):
    return False
"""

UNSAFE_FUNC = """\
import os
def evaluate(text):
    os.system('rm -rf /')
    return True
"""

GOOD_CASES = [
    {"input": {"text": "hello"}, "output": True},
    {"input": {"text": "world"}, "output": True},
]

FAILING_CASES = [
    {"input": {"text": ""}, "output": True},
]

CV_RESULT_FIELDS = {f.name for f in dc_fields(CrossValidationResult)}

BASE_ITEM_FIELDS = {
    "instruction_id",
    "instruction",
    "instruction_category",
    "placeholders",
    "eval_funcs",
    "cases",
}

EXPECTED_OUTPUT_FIELDS = BASE_ITEM_FIELDS | CV_RESULT_FIELDS


def _make_input_item(instruction_id="test-001", functions=None, cases=None, **extra):
    item = {
        "instruction_id": instruction_id,
        "instruction": "Write a greeting.",
        "instruction_category": "generation",
        "placeholders": {},
        "eval_funcs": functions or [GOOD_FUNC],
        "cases": cases or GOOD_CASES,
    }
    item.update(extra)
    return item


def _write_jsonl(path, items):
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _run_standalone_script(input_items, env_overrides=None):
    """Run verifiers_cross_validation.py as a subprocess and return output records."""
    script = os.path.join(
        os.path.dirname(__file__), "..", "src", "verifiers_cross_validation.py"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "input.jsonl")
        out_path = os.path.join(tmpdir, "output.jsonl")
        _write_jsonl(in_path, input_items)

        autoif_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        src_dir = os.path.join(autoif_dir, "src")
        repo_root = os.path.abspath(os.path.join(autoif_dir, "..", ".."))
        pythonpath = os.pathsep.join([src_dir, autoif_dir, repo_root])

        env = {
            **os.environ,
            "PYTHONPATH": pythonpath,
            "MIN_FUNCTIONS": "1",
            "MIN_TEST_CASES": "1",
            "FUNCTION_PASS_RATE": "0.5",
            "MIN_CASE_PASSES": "1",
        }
        if env_overrides:
            env.update(env_overrides)

        result = subprocess.run(
            [sys.executable, script,
             "--input-file", in_path,
             "--output-file", out_path],
            capture_output=True, text=True, env=env, timeout=120,
        )
        assert result.returncode == 0, (
            f"Script failed:\nstdout={result.stdout}\nstderr={result.stderr}"
        )
        return _read_jsonl(out_path)


# ---------------------------------------------------------------------------
# CrossValidationResult unit tests
# ---------------------------------------------------------------------------

class TestCrossValidationResult:
    def test_to_dict_contains_all_fields(self):
        cv = CrossValidationResult(
            success=True,
            passing_functions=[GOOD_FUNC],
            passing_cases=GOOD_CASES,
            best_accuracy=1.0,
            total_functions=1,
            total_cases=2,
        )
        d = cv.to_dict()
        assert set(d.keys()) == CV_RESULT_FIELDS

    def test_to_dict_roundtrips_through_json(self):
        cv = CrossValidationResult(
            success=False,
            passing_functions=[],
            passing_cases=[],
            best_accuracy=0.0,
            total_functions=3,
            total_cases=5,
        )
        roundtripped = json.loads(json.dumps(cv.to_dict()))
        assert roundtripped["success"] is False
        assert roundtripped["best_accuracy"] == 0.0
        assert isinstance(roundtripped["passing_functions"], list)

    def test_success_field_is_bool(self):
        for val in (True, False):
            cv = CrossValidationResult(
                success=val,
                passing_functions=[], passing_cases=[],
                best_accuracy=0.0, total_functions=0, total_cases=0,
            )
            d = cv.to_dict()
            assert isinstance(d["success"], bool)
            assert d["success"] is val


# ---------------------------------------------------------------------------
# run_cross_validation unit tests
# ---------------------------------------------------------------------------

class TestRunCrossValidation:
    def test_all_passing(self):
        cv = run_cross_validation(
            [GOOD_FUNC], GOOD_CASES,
            min_functions=1, min_test_cases=1,
            function_pass_rate=0.5, min_case_passes=1,
        )
        assert cv.success is True
        assert len(cv.passing_functions) >= 1
        assert len(cv.passing_cases) >= 1
        assert cv.best_accuracy > 0

    def test_all_failing(self):
        cv = run_cross_validation(
            [BAD_FUNC], GOOD_CASES,
            min_functions=1, min_test_cases=1,
            function_pass_rate=0.5, min_case_passes=1,
        )
        assert cv.success is False
        assert cv.best_accuracy == 0.0

    def test_no_safe_functions(self):
        cv = run_cross_validation(
            [UNSAFE_FUNC], GOOD_CASES,
            min_functions=1, min_test_cases=1,
        )
        assert cv.success is False
        assert cv.passing_functions == []
        assert cv.total_functions > 0

    def test_empty_functions(self):
        cv = run_cross_validation(
            [], GOOD_CASES,
            min_functions=1, min_test_cases=1,
        )
        assert cv.success is False

    def test_empty_cases(self):
        cv = run_cross_validation(
            [GOOD_FUNC], [],
            min_functions=1, min_test_cases=1,
        )
        assert cv.success is False

    def test_result_has_all_fields(self):
        cv = run_cross_validation(
            [GOOD_FUNC], GOOD_CASES,
            min_functions=1, min_test_cases=1,
            function_pass_rate=0.5, min_case_passes=1,
        )
        assert set(cv.to_dict().keys()) == CV_RESULT_FIELDS


# ---------------------------------------------------------------------------
# Standalone script tests (verifiers_cross_validation.py)
# ---------------------------------------------------------------------------

class TestStandaloneScript:
    def test_basic_output_fields(self):
        items = [_make_input_item()]
        records = _run_standalone_script(items)
        assert len(records) == 1
        assert EXPECTED_OUTPUT_FIELDS.issubset(set(records[0].keys()))

    def test_success_is_bool(self):
        items = [_make_input_item()]
        records = _run_standalone_script(items)
        assert isinstance(records[0]["success"], bool)

    def test_passing_item(self):
        items = [_make_input_item(functions=[GOOD_FUNC], cases=GOOD_CASES)]
        records = _run_standalone_script(items)
        assert records[0]["success"] is True
        assert len(records[0]["passing_functions"]) >= 1
        assert len(records[0]["passing_cases"]) >= 1

    def test_failing_item(self):
        items = [_make_input_item(functions=[BAD_FUNC], cases=GOOD_CASES)]
        records = _run_standalone_script(items)
        assert records[0]["success"] is False

    def test_error_records_skipped(self):
        items = [
            _make_input_item(instruction_id="good-001"),
            {"__ERROR__": {
                "error": "no_valid_functions",
                "message": "No valid functions",
                "task_data": {"instruction_id": "bad-001"},
            }},
            _make_input_item(instruction_id="good-002"),
        ]
        records = _run_standalone_script(items)
        assert len(records) == 2
        ids = {r["instruction_id"] for r in records}
        assert ids == {"good-001", "good-002"}

    def test_all_error_records_produces_empty_output(self):
        items = [
            {"__ERROR__": {"error": "e1", "message": "m1", "task_data": {}}},
            {"__ERROR__": {"error": "e2", "message": "m2", "task_data": {}}},
        ]
        records = _run_standalone_script(items)
        assert records == []

    def test_original_fields_preserved(self):
        item = _make_input_item(extra_field="should_survive")
        records = _run_standalone_script([item])
        assert records[0]["extra_field"] == "should_survive"
        assert records[0]["instruction"] == item["instruction"]

    def test_mixed_pass_fail(self):
        items = [
            _make_input_item(instruction_id="pass", functions=[GOOD_FUNC], cases=GOOD_CASES),
            _make_input_item(instruction_id="fail", functions=[BAD_FUNC], cases=GOOD_CASES),
        ]
        records = _run_standalone_script(items)
        assert len(records) == 2
        by_id = {r["instruction_id"]: r for r in records}
        assert by_id["pass"]["success"] is True
        assert by_id["fail"]["success"] is False
        for r in records:
            assert EXPECTED_OUTPUT_FIELDS.issubset(set(r.keys()))

    def test_eval_func_alias(self):
        """Items using 'eval_func' (singular) should also be handled."""
        item = {
            "instruction_id": "alias-001",
            "instruction": "test",
            "instruction_category": "test",
            "placeholders": {},
            "eval_func": [GOOD_FUNC],
            "cases": GOOD_CASES,
        }
        records = _run_standalone_script([item])
        assert len(records) == 1
        assert "success" in records[0]


# ---------------------------------------------------------------------------
# Output format consistency between inline and standalone paths
# ---------------------------------------------------------------------------

class TestOutputFormatConsistency:
    """Both paths must produce records with the same CV-related fields."""

    def _simulate_inline_cv(self, functions, cases, **cv_kwargs):
        """Simulate what verifiers_task.py returns for inline CV."""
        cv_result = run_cross_validation(functions, cases, **cv_kwargs)
        base = {
            "instruction_id": "inline-001",
            "instruction": "Write a greeting.",
            "instruction_category": "generation",
            "placeholders": {},
            "eval_funcs": functions,
            "cases": cases,
        }
        return {**base, **cv_result.to_dict()}

    def _get_standalone_record(self, functions, cases, **env_overrides):
        """Run the standalone script for a single item."""
        item = _make_input_item(
            instruction_id="standalone-001", functions=functions, cases=cases,
        )
        records = _run_standalone_script([item], env_overrides=env_overrides)
        assert len(records) == 1
        return records[0]

    def test_same_cv_fields_on_success(self):
        inline = self._simulate_inline_cv(
            [GOOD_FUNC], GOOD_CASES,
            min_functions=1, min_test_cases=1,
            function_pass_rate=0.5, min_case_passes=1,
        )
        standalone = self._get_standalone_record([GOOD_FUNC], GOOD_CASES)

        inline_cv_keys = set(inline.keys()) & CV_RESULT_FIELDS
        standalone_cv_keys = set(standalone.keys()) & CV_RESULT_FIELDS
        assert inline_cv_keys == CV_RESULT_FIELDS
        assert standalone_cv_keys == CV_RESULT_FIELDS

    def test_same_cv_fields_on_failure(self):
        inline = self._simulate_inline_cv(
            [BAD_FUNC], GOOD_CASES,
            min_functions=1, min_test_cases=1,
            function_pass_rate=0.5, min_case_passes=1,
        )
        standalone = self._get_standalone_record([BAD_FUNC], GOOD_CASES)

        assert inline["success"] is False
        assert standalone["success"] is False

        inline_cv_keys = set(inline.keys()) & CV_RESULT_FIELDS
        standalone_cv_keys = set(standalone.keys()) & CV_RESULT_FIELDS
        assert inline_cv_keys == standalone_cv_keys == CV_RESULT_FIELDS

    def test_field_types_match(self):
        inline = self._simulate_inline_cv(
            [GOOD_FUNC], GOOD_CASES,
            min_functions=1, min_test_cases=1,
            function_pass_rate=0.5, min_case_passes=1,
        )
        standalone = self._get_standalone_record([GOOD_FUNC], GOOD_CASES)

        for field in CV_RESULT_FIELDS:
            assert type(inline[field]) == type(standalone[field]), (
                f"Type mismatch for '{field}': "
                f"inline={type(inline[field]).__name__}, "
                f"standalone={type(standalone[field]).__name__}"
            )

    def test_both_paths_preserve_base_fields(self):
        inline = self._simulate_inline_cv(
            [GOOD_FUNC], GOOD_CASES,
            min_functions=1, min_test_cases=1,
            function_pass_rate=0.5, min_case_passes=1,
        )
        standalone = self._get_standalone_record([GOOD_FUNC], GOOD_CASES)

        for field in BASE_ITEM_FIELDS:
            assert field in inline, f"Inline missing base field '{field}'"
            assert field in standalone, f"Standalone missing base field '{field}'"

    def test_success_true_has_nonempty_passing_lists(self):
        inline = self._simulate_inline_cv(
            [GOOD_FUNC], GOOD_CASES,
            min_functions=1, min_test_cases=1,
            function_pass_rate=0.5, min_case_passes=1,
        )
        standalone = self._get_standalone_record([GOOD_FUNC], GOOD_CASES)

        for label, record in [("inline", inline), ("standalone", standalone)]:
            if record["success"]:
                assert len(record["passing_functions"]) > 0, f"{label}: success=true but no passing_functions"
                assert len(record["passing_cases"]) > 0, f"{label}: success=true but no passing_cases"

    def test_success_false_allows_empty_passing_lists(self):
        inline = self._simulate_inline_cv(
            [BAD_FUNC], GOOD_CASES,
            min_functions=1, min_test_cases=1,
            function_pass_rate=0.5, min_case_passes=1,
        )
        standalone = self._get_standalone_record([BAD_FUNC], GOOD_CASES)

        for label, record in [("inline", inline), ("standalone", standalone)]:
            assert record["success"] is False
            assert isinstance(record["passing_functions"], list)
            assert isinstance(record["passing_cases"], list)

    def test_best_accuracy_range(self):
        inline = self._simulate_inline_cv(
            [GOOD_FUNC], GOOD_CASES,
            min_functions=1, min_test_cases=1,
            function_pass_rate=0.5, min_case_passes=1,
        )
        standalone = self._get_standalone_record([GOOD_FUNC], GOOD_CASES)

        for record in [inline, standalone]:
            assert 0.0 <= record["best_accuracy"] <= 1.0

    def test_total_counts_consistent(self):
        funcs = [GOOD_FUNC, BAD_FUNC]
        cases = GOOD_CASES

        inline = self._simulate_inline_cv(
            funcs, cases,
            min_functions=1, min_test_cases=1,
            function_pass_rate=0.5, min_case_passes=1,
        )
        standalone = self._get_standalone_record(funcs, cases)

        assert inline["total_cases"] == standalone["total_cases"] == len(cases)
        for record in [inline, standalone]:
            assert record["total_functions"] <= len(funcs)
            assert len(record["passing_functions"]) <= record["total_functions"]
            assert len(record["passing_cases"]) <= record["total_cases"]


# ---------------------------------------------------------------------------
# Downstream filtering contract
# ---------------------------------------------------------------------------

class TestDownstreamContract:
    """Verify the contract that downstream steps rely on: filter by 'success'."""

    def test_filter_success_true(self):
        items = [
            _make_input_item(instruction_id="p1", functions=[GOOD_FUNC]),
            _make_input_item(instruction_id="f1", functions=[BAD_FUNC]),
            _make_input_item(instruction_id="p2", functions=[GOOD_FUNC]),
        ]
        records = _run_standalone_script(items)
        passing = [r for r in records if r["success"] is True]
        failing = [r for r in records if r["success"] is False]
        assert len(passing) + len(failing) == len(records)
        assert all(r["success"] is True for r in passing)
        assert all(r["success"] is False for r in failing)

    def test_error_records_absent_from_output(self):
        items = [
            _make_input_item(instruction_id="good"),
            {"__ERROR__": {"error": "test", "message": "x", "task_data": {}}},
        ]
        records = _run_standalone_script(items)
        for r in records:
            assert "__ERROR__" not in r
