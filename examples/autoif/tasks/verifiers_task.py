"""Verifiers task - Generate evaluation functions and test cases with refinement loop.

This task implements a circular improvement system:
1. Generate N evaluation functions and M test cases
2. Cross-validate with detailed error tracking
3. If cross-validation fails, analyze errors with LLM
4. Generate constraint-specific prompt improvements
5. Regenerate with improved prompts
6. Repeat until success or max attempts
"""
from typing import Any, Dict, Generator, Union, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import os
import json
import random

import logging
logger = logging.getLogger(__name__)

import jinja2

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task.base import GeneratorTask, TaskFailed

from src.utils.constants import (
    REASON_NO_VALID_FUNCTIONS, REASON_NO_VALID_TEST_CASES,
    REASON_INSUFFICIENT_FUNCTIONS, REASON_INSUFFICIENT_TEST_CASES,
    REASON_NO_PASSING_TEST_CASES, REASON_NO_FUNCTIONS_MEET_ACCURACY,
    REASON_FUNCTION_PARSE_FAILURE, REASON_CASES_PARSE_ERROR
)
from src.utils.response_parser import ResponseParser
from src.utils.function_executor import FunctionExecutor
from src.utils.lang_id import LANG_MAP

__all__ = ["GenerateVerifiersTask"]

NUM_FUNC_GENERATIONS = int(os.getenv("NUM_FUNC_GENERATIONS", 3))
NUM_TEST_CASES = int(os.getenv("NUM_TEST_CASES", 3))
MIN_FUNCTIONS = int(os.getenv("MIN_FUNCTIONS", 1))
MIN_TEST_CASES = int(os.getenv("MIN_TEST_CASES", 1))
FUNCTION_PASS_RATE = float(os.getenv("FUNCTION_PASS_RATE", 0.8))

# Refinement loop settings
MAX_REFINEMENT_ATTEMPTS = int(os.getenv("MAX_REFINEMENT_ATTEMPTS", 3))
REFINEMENT_ENABLED = os.getenv("REFINEMENT_ENABLED", "true").lower() == "true"

# Prompt templates
FUNC_PROMPT_PATH = "model_prompts/create_eval_function_prompt.j2"
TEST_CASES_PROMPT_PATH = "model_prompts/create_test_cases_prompt.j2"
ANALYZE_ERRORS_PROMPT_PATH = "model_prompts/analyze_errors_prompt.j2"


@dataclass
class TestCaseFailure:
    """Details of a single test case failure."""
    case_idx: int
    input_data: Dict[str, Any]
    input_preview: str  # Truncated for LLM context
    expected: bool
    actual: Optional[bool]
    error: Optional[str] = None


@dataclass
class FunctionResult:
    """Cross-validation results for a single function."""
    func_idx: int
    code: str
    accuracy: float
    correct_count: int
    total_cases: int
    passes_threshold: bool
    failures: List[TestCaseFailure] = field(default_factory=list)


@dataclass
class CrossValidationResult:
    """Complete cross-validation results with detailed error tracking."""
    success: bool
    function_results: List[FunctionResult]
    passing_functions: List[str]
    passing_cases: List[Dict[str, Any]]
    best_accuracy: float
    total_functions: int
    total_cases: int
    error_summary: Optional[str] = None


class GenerateVerifiersTask(GeneratorTask):
    """
    Generate evaluation functions and test cases with refinement loop.

    Flow:
    1. Generate NUM_FUNC_GENERATIONS evaluation functions
    2. Sample NUM_TEST_CASES random user queries from the provided dataset
    3. Generate 1 positive and 1 negative test case per query
    4. Cross-validate functions against test cases with detailed error tracking
    5. If validation fails and REFINEMENT_ENABLED:
       a. Analyze errors using LLM
       b. Generate constraint-specific prompt improvements
       c. Regenerate functions with improved prompt
       d. Repeat until success or MAX_REFINEMENT_ATTEMPTS
    6. Return results or raise TaskFailed
    """

    FUNC_GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 8192,
        "n": NUM_FUNC_GENERATIONS,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }

    TEST_CASES_GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 16384,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }

    ANALYSIS_GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.3,  # Lower temperature for more focused analysis
        "top_p": 0.95,
        "max_tokens": 4096,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }

    def __init__(self, *args, **kwargs):
        self.parser = ResponseParser()
        self.executor = FunctionExecutor()
        self._jinja_env = None
        super().__init__(*args, **kwargs)

    def _get_jinja_env(self) -> jinja2.Environment:
        """Get or create Jinja2 environment."""
        if self._jinja_env is None:
            self._jinja_env = jinja2.Environment(
                loader=jinja2.FileSystemLoader("."),
                undefined=jinja2.Undefined
            )
        return self._jinja_env

    def _render_template(self, template_path: str, **kwargs) -> str:
        """Render a Jinja2 template."""
        env = self._get_jinja_env()
        template = env.get_template(template_path)
        return template.render(**kwargs)

    def _build_placeholder_info(self, placeholders: Dict) -> List[Dict]:
        """Build placeholder info list for template."""
        info = []
        for name, meta in placeholders.items():
            entry = {"name": name, "type": meta.get("type", "unknown")}
            if meta.get("type") == "numeric":
                entry["description"] = f"integer between {meta.get('min', 1)} and {meta.get('max', 10)}"
            elif meta.get("type") == "static":
                entry["description"] = f"one of: {meta.get('values', [])}"
            elif meta.get("type") == "dynamic":
                entry["description"] = "dynamically generated value"
            info.append(entry)
        return info

    def _sample_placeholder_values(self, placeholders: Dict) -> Dict[str, Any]:
        """Sample concrete values for placeholders."""
        values = {}
        for name, meta in placeholders.items():
            ptype = meta.get("type", "unknown")
            if ptype == "numeric":
                values[name] = random.randint(meta.get("min", 1), meta.get("max", 5))
            elif ptype == "static":
                vals = meta.get("values", ["default"])
                values[name] = random.choice(vals) if vals else "default"
            elif ptype == "dynamic":
                values[name] = meta.get("sample", "example_value")
            else:
                values[name] = "unknown"
        return values

    @staticmethod
    def _get_language_name() -> str:
        """Get the full language name from LANGUAGE env variable."""
        lang_code = os.environ.get('LANGUAGE', 'en').lower().strip()
        return LANG_MAP.get(lang_code, 'English')

    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        """Generate evaluation functions and test cases with refinement loop."""
        instruction_id = self.data.get("instruction_id", "unknown")
        instruction = self.data.get("instruction", "")
        instruction_category = self.data.get("instruction_category", "")
        placeholders = self.data.get("placeholders", {})
        user_queries = self.data.get("user_queries", [])

        if not instruction:
            raise TaskFailed(message="Missing 'instruction' field", error_type="missing_instruction")

        if not user_queries:
            raise TaskFailed(message="No user queries provided", error_type="missing_user_queries")

        # Build placeholder info for function prompt
        placeholder_info = self._build_placeholder_info(placeholders)

        language = self._get_language_name()

        # =====================================================================
        # Phase 1: Generate test cases (only once - these remain constant)
        # =====================================================================
        num_queries_to_sample = min(NUM_TEST_CASES, len(user_queries))
        sampled_queries = random.sample(user_queries, num_queries_to_sample)

        test_cases_prompt = self._render_template(
            TEST_CASES_PROMPT_PATH,
            instruction=instruction,
            placeholders=placeholder_info if placeholders else None,
            user_queries=sampled_queries,
            language=language
        )
        logger.info(f"[GenerateVerifiersTask] IID:{instruction_id} test cases generation prompt: {test_cases_prompt}")

        response: Response = yield Request({
            "messages": [{"role": "user", "content": test_cases_prompt}],
            **self.TEST_CASES_GEN_PARAMS
        })
        logger.info(f"[GenerateVerifiersTask] IID:{instruction_id} test cases generation response: {response.get_text()}")

        test_cases_data, error = self.parser.parse_test_cases(response.get_text(), instruction_id)
        self.data["test_cases_data"] = test_cases_data

        if not test_cases_data:
            raise TaskFailed(
                message="Failed to parse test cases",
                error_type=REASON_NO_VALID_TEST_CASES
            )

        # Convert test cases to cross-validation format
        all_cases = self._prepare_test_cases(test_cases_data)

        if len(all_cases) < MIN_TEST_CASES:
            raise TaskFailed(
                message=f"Only {len(all_cases)} test cases, need {MIN_TEST_CASES}",
                error_type=REASON_INSUFFICIENT_TEST_CASES
            )

        # =====================================================================
        # Phase 2: Generate functions with refinement loop
        # =====================================================================
        prompt_improvements: List[str] = []  # Accumulated improvements
        attempt = 0
        best_result: Optional[CrossValidationResult] = None

        while attempt < MAX_REFINEMENT_ATTEMPTS:
            attempt += 1
            logger.info(f"[GenerateVerifiersTask] IID:{instruction_id} Attempt {attempt}/{MAX_REFINEMENT_ATTEMPTS}")

            # Build function generation prompt with accumulated improvements
            func_prompt = self._build_function_prompt(
                instruction=instruction,
                placeholder_info=placeholder_info,
                prompt_improvements=prompt_improvements,
            )
            logger.info(f"[GenerateVerifiersTask] IID:{instruction_id} function generation prompt (attempt {attempt}): {func_prompt}")

            # Generate functions
            response: Response = yield Request({
                "messages": [{"role": "user", "content": func_prompt}],
                **self.FUNC_GEN_PARAMS
            })

            # Parse functions
            all_functions: List[str] = []
            response_texts = response.get_text(n=NUM_FUNC_GENERATIONS)
            if response_texts:
                for text in response_texts:
                    logger.info(f"[GenerateVerifiersTask] IID:{instruction_id} function response: {text[:500]}...")
                    func_str, error = self.parser.parse_function(text, instruction_id)
                    if func_str:
                        all_functions.append(func_str)

            self.data["eval_funcs"] = all_functions
            self.data["refinement_attempt"] = attempt

            if not all_functions:
                logger.warning(f"[GenerateVerifiersTask] IID:{instruction_id} No functions parsed on attempt {attempt}")
                if not REFINEMENT_ENABLED or attempt >= MAX_REFINEMENT_ATTEMPTS:
                    raise TaskFailed(
                        message=f"No valid functions parsed after {attempt} attempts",
                        error_type=REASON_NO_VALID_FUNCTIONS
                    )
                # Add a generic improvement and retry
                prompt_improvements.append(
                    "CRITICAL: Ensure your response contains a valid Python function definition "
                    "starting with 'def evaluate(response: str, **kwargs) -> bool:' inside a code block."
                )
                continue

            # Cross-validate with detailed error tracking
            cv_result = self._run_cross_validation_detailed(all_functions, all_cases)

            # Track best result
            if best_result is None or cv_result.best_accuracy > best_result.best_accuracy:
                best_result = cv_result

            logger.info(
                f"[GenerateVerifiersTask] IID:{instruction_id} Attempt {attempt} result: "
                f"accuracy={cv_result.best_accuracy:.1%}, passing_funcs={len(cv_result.passing_functions)}"
            )

            if cv_result.success:
                # Success! Return results
                self.data["refinement_attempts"] = attempt
                self.data["prompt_improvements"] = prompt_improvements

                return {
                    "instruction_id": instruction_id,
                    "instruction": instruction,
                    "instruction_category": instruction_category,
                    "placeholders": placeholders,
                    "eval_func": cv_result.passing_functions,
                    "cases": cv_result.passing_cases,
                    "refinement_attempts": attempt,
                    "best_accuracy": cv_result.best_accuracy,
                }

            # =====================================================================
            # Phase 3: Analyze errors and generate improvements
            # =====================================================================
            if not REFINEMENT_ENABLED or attempt >= MAX_REFINEMENT_ATTEMPTS:
                break

            logger.info(f"[GenerateVerifiersTask] IID:{instruction_id} Analyzing errors for refinement...")

            analysis_prompt = self._build_analysis_prompt(
                instruction=instruction,
                placeholder_info=placeholder_info,
                cv_result=cv_result,
            )

            response: Response = yield Request({
                "messages": [{"role": "user", "content": analysis_prompt}],
                **self.ANALYSIS_GEN_PARAMS
            })

            analysis_text = response.get_text()
            logger.info(f"[GenerateVerifiersTask] IID:{instruction_id} Error analysis: {analysis_text[:1000]}...")

            # Extract prompt improvement from analysis
            improvement = self._extract_prompt_improvement(analysis_text)
            if improvement:
                prompt_improvements.append(improvement)
                logger.info(f"[GenerateVerifiersTask] IID:{instruction_id} Added improvement: {improvement[:200]}...")

        # =====================================================================
        # Failed after all attempts
        # =====================================================================
        # Store detailed failure info for debugging
        self.data["refinement_attempts"] = attempt
        self.data["prompt_improvements"] = prompt_improvements
        if best_result:
            self.data["best_accuracy"] = best_result.best_accuracy
            self.data["error_summary"] = best_result.error_summary

        if best_result and best_result.passing_cases:
            raise TaskFailed(
                message=f"No functions meet {FUNCTION_PASS_RATE*100:.0f}% accuracy after {attempt} attempts (best: {best_result.best_accuracy:.1%})",
                error_type=REASON_NO_FUNCTIONS_MEET_ACCURACY
            )
        else:
            raise TaskFailed(
                message=f"No test cases pass any function after {attempt} attempts",
                error_type=REASON_NO_PASSING_TEST_CASES
            )

    def _build_function_prompt(
        self,
        instruction: str,
        placeholder_info: List[Dict],
        prompt_improvements: List[str],
    ) -> str:
        """Build function generation prompt with accumulated improvements."""
        # Render base prompt
        base_prompt = self._render_template(
            FUNC_PROMPT_PATH,
            instruction=instruction,
            placeholders=placeholder_info,
            language=language
        )

        if not prompt_improvements:
            return base_prompt

        # Append constraint-specific improvements
        improvements_section = "\n\n# IMPORTANT: Constraint-Specific Guidance\n\n"
        improvements_section += "Based on previous attempts, pay special attention to:\n\n"
        for i, improvement in enumerate(prompt_improvements, 1):
            improvements_section += f"{i}. {improvement}\n\n"

        return base_prompt + improvements_section

    def _build_analysis_prompt(
        self,
        instruction: str,
        placeholder_info: List[Dict],
        cv_result: CrossValidationResult,
    ) -> str:
        """Build error analysis prompt."""
        # Prepare function results for template
        function_results = []
        for fr in cv_result.function_results[:3]:  # Limit to top 3 for context
            function_results.append({
                "code": fr.code[:2000],  # Truncate long functions
                "accuracy": round(fr.accuracy * 100, 1),
                "failures": [
                    {
                        "case_idx": f.case_idx,
                        "input_preview": f.input_preview,
                        "expected": f.expected,
                        "actual": f.actual,
                        "error": f.error,
                    }
                    for f in fr.failures[:5]  # Limit failures per function
                ],
            })

        return self._render_template(
            ANALYZE_ERRORS_PROMPT_PATH,
            instruction=instruction,
            placeholders=placeholder_info,
            num_functions=cv_result.total_functions,
            num_test_cases=cv_result.total_cases,
            function_results=function_results,
        )

    def _extract_prompt_improvement(self, analysis_text: str) -> Optional[str]:
        """Extract the prompt improvement from analysis response."""
        if "PROMPT_IMPROVEMENT:" not in analysis_text:
            return None

        parts = analysis_text.split("PROMPT_IMPROVEMENT:")
        if len(parts) < 2:
            return None

        improvement = parts[1].strip()
        # Clean up - take until end or next section marker
        if "\n\n\n" in improvement:
            improvement = improvement.split("\n\n\n")[0]

        # Limit length
        if len(improvement) > 1000:
            improvement = improvement[:1000] + "..."

        return improvement.strip() if improvement.strip() else None

    def _prepare_test_cases(self, test_cases_data: Dict) -> List[Dict[str, Any]]:
        """Prepare test cases for cross-validation."""
        all_cases = []

        positive_cases = test_cases_data.get("positive", [])
        negative_cases = test_cases_data.get("negative", [])

        for case in positive_cases:
            test_input = {"response": case["response"]}
            for key, val in case.items():
                if key != "response":
                    test_input[key] = val
            all_cases.append({"input": test_input, "output": True})

        for case in negative_cases:
            test_input = {"response": case["response"]}
            for key, val in case.items():
                if key != "response":
                    test_input[key] = val
            all_cases.append({"input": test_input, "output": False})

        # Deduplicate
        return self._deduplicate_cases(all_cases)

    def _deduplicate_cases(self, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate test cases."""
        seen = set()
        result = []
        for case in cases:
            case_json = json.dumps(case, sort_keys=True)
            if case_json not in seen:
                seen.add(case_json)
                result.append(case)
        return result

    def _run_cross_validation_detailed(
        self,
        functions: List[str],
        test_cases: List[Dict[str, Any]]
    ) -> CrossValidationResult:
        """Cross-validate with detailed error tracking."""

        # Filter safe functions first
        safe_functions = []
        for func in functions:
            is_safe, _ = self.executor.is_safe_function(func)
            if is_safe:
                safe_functions.append(func)

        if not safe_functions:
            return CrossValidationResult(
                success=False,
                function_results=[],
                passing_functions=[],
                passing_cases=[],
                best_accuracy=0.0,
                total_functions=len(functions),
                total_cases=len(test_cases),
                error_summary="No safe functions after safety filtering",
            )

        case_passes = [False] * len(test_cases)
        function_results: List[FunctionResult] = []
        passing_functions: List[str] = []
        best_accuracy = 0.0

        for func_idx, func in enumerate(safe_functions):
            correct_count = 0
            failures: List[TestCaseFailure] = []

            for case_idx, test_case in enumerate(test_cases):
                passed, error_msg = self.executor.test_function(func, test_case, log_errors=False)
                expected = test_case["output"]

                if passed:
                    case_passes[case_idx] = True
                    correct_count += 1
                    actual = expected  # If passed, actual matches expected
                else:
                    # Determine actual result
                    try:
                        actual_result = self.executor.execute_function(
                            func, test_case["input"], log_errors=False
                        )
                        actual = actual_result if isinstance(actual_result, bool) else None
                    except Exception:
                        actual = None

                    # Record failure details
                    input_data = test_case["input"]
                    response_preview = str(input_data.get("response", ""))[:200]

                    failures.append(TestCaseFailure(
                        case_idx=case_idx,
                        input_data=input_data,
                        input_preview=response_preview,
                        expected=expected,
                        actual=actual,
                        error=error_msg[:200] if error_msg else None,
                    ))

            accuracy = correct_count / len(test_cases) if test_cases else 0
            best_accuracy = max(best_accuracy, accuracy)
            passes_threshold = accuracy >= FUNCTION_PASS_RATE

            func_result = FunctionResult(
                func_idx=func_idx,
                code=func,
                accuracy=accuracy,
                correct_count=correct_count,
                total_cases=len(test_cases),
                passes_threshold=passes_threshold,
                failures=failures,
            )
            function_results.append(func_result)

            if passes_threshold:
                passing_functions.append(func)

        # Collect passing cases
        passing_cases = [
            case for idx, case in enumerate(test_cases)
            if case_passes[idx]
        ]

        # Generate error summary
        error_summary = None
        if not passing_functions:
            total_failures = sum(len(fr.failures) for fr in function_results)
            error_summary = (
                f"Best accuracy: {best_accuracy:.1%}. "
                f"Total failures across {len(function_results)} functions: {total_failures}. "
                f"Threshold: {FUNCTION_PASS_RATE:.0%}"
            )

        success = len(passing_functions) > 0 and len(passing_cases) > 0

        return CrossValidationResult(
            success=success,
            function_results=function_results,
            passing_functions=passing_functions,
            passing_cases=passing_cases,
            best_accuracy=best_accuracy,
            total_functions=len(safe_functions),
            total_cases=len(test_cases),
            error_summary=error_summary,
        )
