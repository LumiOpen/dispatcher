"""Reusable cross-validation logic for verifier functions.

Used both inline during GPU generation (when SKIP_INLINE_CROSSVAL is false)
and as a standalone CPU job (via src/verifiers_cross_validation.py).
"""

from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import logging

from .function_executor import FunctionExecutor

logger = logging.getLogger(__name__)


@dataclass
class CrossValidationResult:
    """Cross-validation results for a single instruction."""
    success: bool
    passing_functions: List[str]
    passing_cases: List[Dict[str, Any]]
    best_accuracy: float
    total_functions: int
    total_cases: int
    error_counts: Dict[str, int] = field(default_factory=dict)
    first_error_details: Optional[str] = None
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_cross_validation(
    functions: List[str],
    test_cases: List[Dict[str, Any]],
    executor: Optional[FunctionExecutor] = None,
    min_functions: int = 1,
    min_test_cases: int = 1,
    function_pass_rate: float = 0.8,
    min_case_passes: int = 1,
    log_file: Optional[str] = None,
) -> CrossValidationResult:
    """Two-pass cross-validation of functions against test cases.

    Phase 1 — filter test cases: a case passes if at least min_case_passes
    functions produce the expected output (default 1 = any-agree).

    Phase 2 — score functions against the filtered cases only: a function
    passes if its accuracy on filtered cases >= function_pass_rate.

    Success requires >= min_functions passing functions and >= min_test_cases
    passing cases.

    Args:
        functions: List of function code strings (each containing an evaluate()).
        test_cases: List of dicts with 'input' and 'output' keys.
        executor: FunctionExecutor instance (created if not provided).
        min_functions: Minimum number of passing functions required.
        min_test_cases: Minimum number of passing test cases required.
        function_pass_rate: Minimum accuracy on filtered cases for a function to pass.
        min_case_passes: Minimum number of functions that must agree for a
            test case to survive Phase 1 (default 1 = any function agrees).
        log_file: Optional path to per-instruction log file. When provided,
            function code, subprocess stderr, and per-case results are written
            to the file for post-mortem debugging.

    Returns:
        CrossValidationResult with passing functions, cases, and stats.
    """
    if executor is None:
        executor = FunctionExecutor()

    unsafe_reasons: Counter = Counter()
    safe_functions = []
    for f in functions:
        ok, reason = executor.is_safe_function(f)
        if ok:
            safe_functions.append(f)
        elif reason:
            unsafe_reasons[reason] += 1

    if not safe_functions:
        error_counts = dict(unsafe_reasons) if unsafe_reasons else {"all_functions_unsafe": len(functions)}
        return CrossValidationResult(
            success=False,
            passing_functions=[],
            passing_cases=[],
            best_accuracy=0.0,
            total_functions=len(functions),
            total_cases=len(test_cases),
            error_counts=error_counts,
            failure_reason="all_functions_unsafe",
        )

    num_funcs = len(safe_functions)
    num_cases = len(test_cases)

    # Build pass/fail matrix via batch execution: one subprocess per function
    # loads the model once and evaluates all test cases in sequence.
    results = [[False] * num_cases for _ in range(num_funcs)]
    error_reasons: Counter = Counter()
    error_reasons.update(unsafe_reasons)
    first_error: Optional[str] = None

    for func_idx, func in enumerate(safe_functions):
        batch = executor.test_function_batch(
            func, test_cases, log_file=log_file, func_idx=func_idx,
        )
        for case_idx, (passed, reason) in enumerate(batch):
            results[func_idx][case_idx] = passed
            if not passed and reason:
                bucket = reason.split(":")[0].strip()
                error_reasons[bucket] += 1
                if first_error is None:
                    first_error = reason[:300]

    # Phase 1: filter test cases — keep those where >= min_case_passes functions agree
    case_pass_counts = [
        sum(results[fi][ci] for fi in range(num_funcs))
        for ci in range(num_cases)
    ]
    passing_case_indices = [
        ci for ci in range(num_cases)
        if case_pass_counts[ci] >= min_case_passes
    ]
    passing_cases = [test_cases[ci] for ci in passing_case_indices]

    # Phase 2: score functions against filtered cases only
    passing_functions: List[str] = []
    best_accuracy = 0.0

    if passing_case_indices:
        for func_idx, func in enumerate(safe_functions):
            correct = sum(results[func_idx][ci] for ci in passing_case_indices)
            accuracy = correct / len(passing_case_indices)
            best_accuracy = max(best_accuracy, accuracy)
            if accuracy >= function_pass_rate:
                passing_functions.append(func)

    success = len(passing_functions) >= min_functions and len(passing_cases) >= min_test_cases

    failure_reason = None
    if not success:
        reasons = []
        if len(passing_cases) < min_test_cases:
            reasons.append(f"insufficient_cases({len(passing_cases)}/{min_test_cases})")
        if len(passing_functions) < min_functions:
            reasons.append(f"insufficient_functions({len(passing_functions)}/{min_functions})")
            if passing_case_indices and best_accuracy < function_pass_rate:
                reasons.append(f"low_accuracy({best_accuracy:.0%}<{function_pass_rate:.0%})")
        failure_reason = "+".join(reasons) if reasons else "unknown"

    return CrossValidationResult(
        success=success,
        passing_functions=passing_functions,
        passing_cases=passing_cases,
        best_accuracy=best_accuracy,
        total_functions=len(safe_functions),
        total_cases=len(test_cases),
        error_counts=dict(error_reasons) if error_reasons else {},
        first_error_details=first_error,
        failure_reason=failure_reason,
    )
