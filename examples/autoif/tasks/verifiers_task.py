"""Verifiers task - Generate evaluation functions and test cases.

This task:
1. Generates N evaluation functions (each with test cases) via a single LLM call
2. Parses functions and test cases from each generation
3. Cross-validates functions against all collected test cases
4. Returns passing functions and cases, or raises TaskFailed
"""
from typing import Any, Dict, Generator, Union, List, Optional, Tuple
from dataclasses import dataclass, field
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
)
from src.utils.response_parser import ResponseParser
from src.utils.function_executor import FunctionExecutor
from src.utils.lang_id import LANG_MAP

__all__ = ["GenerateVerifiersTask"]

NUM_FUNC_GENERATIONS = int(os.getenv("NUM_FUNC_GENERATIONS", 3))
NUM_TEST_CASES_PER_FUNC = int(os.getenv("NUM_TEST_CASES_PER_FUNC", 3))
MIN_FUNCTIONS = int(os.getenv("MIN_FUNCTIONS", 1))
MIN_TEST_CASES = int(os.getenv("MIN_TEST_CASES", 1))
FUNCTION_PASS_RATE = float(os.getenv("FUNCTION_PASS_RATE", 0.8))
TEST_PASS_RATE = float(os.getenv("TEST_PASS_RATE", 0.5))

PROMPT_PATH = "model_prompts/create_verifiers_prompt.j2"


@dataclass
class CrossValidationResult:
    """Cross-validation results."""
    success: bool
    passing_functions: List[str]
    passing_cases: List[Dict[str, Any]]
    best_accuracy: float
    total_functions: int
    total_cases: int


class GenerateVerifiersTask(GeneratorTask):
    """
    Generate evaluation functions and test cases.

    Flow:
    1. Generate NUM_FUNC_GENERATIONS evaluation functions (each with test cases)
       via a single LLM call with n=NUM_FUNC_GENERATIONS
    2. Parse functions and test cases from each generation
    3. Cross-validate functions against all collected test cases
    4. Return passing functions and cases, or raise TaskFailed
    """

    GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 8192,
        "n": NUM_FUNC_GENERATIONS,
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
                entry["description"] = f"Dynamically generated reasonable integer value"
            elif meta.get("type") == "static":
                entry["description"] = f"One of: {meta.get('values', [])}"
            elif meta.get("type") == "dynamic":
                entry["description"] = "Dynamically generated value based on context"
            info.append(entry)
        return info

    @staticmethod
    def _get_language_name() -> str:
        """Get the full language name from LANGUAGE env variable."""
        lang_code = os.environ.get('LANGUAGE', 'en').lower().strip()
        return LANG_MAP.get(lang_code, 'English')

    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        """Generate evaluation functions and test cases."""
        instruction_id = self.data.get("instruction_id", "unknown")
        instruction = self.data.get("instruction", "")
        instruction_category = self.data.get("instruction_category", "")
        placeholders = self.data.get("placeholders", {})

        if not instruction:
            raise TaskFailed(message="Missing 'instruction' field", error_type="missing_instruction")

        # Build placeholder info
        placeholder_info = self._build_placeholder_info(placeholders)
        language = self._get_language_name()

        # Render prompt
        prompt = self._render_template(
            PROMPT_PATH,
            instruction=instruction,
            placeholders=placeholder_info if placeholders else None,
            num_test_cases=NUM_TEST_CASES_PER_FUNC,
            language=language,
        )
        logger.info(f"[GenerateVerifiersTask] IID:{instruction_id} prompt: {prompt}")

        # Single LLM call with n=NUM_FUNC_GENERATIONS
        response: Response = yield Request({
            "messages": [{"role": "user", "content": prompt}],
            **self.GEN_PARAMS
        })

        # Parse all generations
        all_functions: List[str] = []
        all_cases: List[Dict[str, Any]] = []

        response_texts = response.get_text(n=NUM_FUNC_GENERATIONS)
        if response_texts:
            for text in response_texts:
                logger.info(f"[GenerateVerifiersTask] IID:{instruction_id} generation response: {text[:500]}...")

                # Parse function
                func_str, error = self.parser.parse_function(text, instruction_id)
                if func_str:
                    all_functions.append(func_str)

                # Parse test cases
                cases, error = self.parser.parse_test_cases(text, instruction_id)
                if cases:
                    all_cases.extend(cases)

        # Deduplicate
        all_cases = self._deduplicate_cases(all_cases)

        self.data["eval_funcs"] = all_functions
        self.data["cases"] = all_cases

        # Validate we have enough functions
        if not all_functions:
            raise TaskFailed(
                message="No valid functions parsed from any generation",
                error_type=REASON_NO_VALID_FUNCTIONS
            )

        if len(all_functions) < MIN_FUNCTIONS:
            raise TaskFailed(
                message=f"Only {len(all_functions)} functions parsed, need {MIN_FUNCTIONS}",
                error_type=REASON_INSUFFICIENT_FUNCTIONS
            )

        # Validate we have enough test cases
        if not all_cases:
            raise TaskFailed(
                message="No valid test cases parsed from any generation",
                error_type=REASON_NO_VALID_TEST_CASES
            )

        if len(all_cases) < MIN_TEST_CASES:
            raise TaskFailed(
                message=f"Only {len(all_cases)} test cases, need {MIN_TEST_CASES}",
                error_type=REASON_INSUFFICIENT_TEST_CASES
            )

        # Cross-validate
        cv_result = self._run_cross_validation(all_functions, all_cases)

        logger.info(
            f"[GenerateVerifiersTask] IID:{instruction_id} Cross-validation: "
            f"accuracy={cv_result.best_accuracy:.1%}, "
            f"passing_funcs={len(cv_result.passing_functions)}, "
            f"passing_cases={len(cv_result.passing_cases)}"
        )

        if not cv_result.success:
            if not cv_result.passing_functions:
                raise TaskFailed(
                    message=f"No functions meet {FUNCTION_PASS_RATE*100:.0f}% accuracy (best: {cv_result.best_accuracy:.1%})",
                    error_type=REASON_NO_FUNCTIONS_MEET_ACCURACY
                )
            raise TaskFailed(
                message="No test cases pass any function",
                error_type=REASON_NO_PASSING_TEST_CASES
            )

        return {
            "instruction_id": instruction_id,
            "instruction": instruction,
            "instruction_category": instruction_category,
            "placeholders": placeholders,
            "eval_func": cv_result.passing_functions,
            "cases": cv_result.passing_cases,
            "best_accuracy": cv_result.best_accuracy,
        }

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

    def _run_cross_validation(
        self,
        functions: List[str],
        test_cases: List[Dict[str, Any]]
    ) -> CrossValidationResult:
        """Cross-validate functions against test cases.

        A function passes if its accuracy >= FUNCTION_PASS_RATE.
        A test case passes if its pass rate across safe functions >= TEST_PASS_RATE.
        Success requires >= MIN_FUNCTIONS passing functions and >= MIN_TEST_CASES passing cases.
        """
        # Filter safe functions
        safe_functions = []
        for func in functions:
            is_safe, _ = self.executor.is_safe_function(func)
            if is_safe:
                safe_functions.append(func)

        if not safe_functions:
            return CrossValidationResult(
                success=False,
                passing_functions=[],
                passing_cases=[],
                best_accuracy=0.0,
                total_functions=len(functions),
                total_cases=len(test_cases),
            )

        case_pass_counts = [0] * len(test_cases)
        passing_functions: List[str] = []
        best_accuracy = 0.0

        for func in safe_functions:
            correct_count = 0

            for case_idx, test_case in enumerate(test_cases):
                passed, _ = self.executor.test_function(func, test_case, log_errors=False)
                if passed:
                    case_pass_counts[case_idx] += 1
                    correct_count += 1

            accuracy = correct_count / len(test_cases) if test_cases else 0
            best_accuracy = max(best_accuracy, accuracy)

            if accuracy >= FUNCTION_PASS_RATE:
                passing_functions.append(func)

        passing_cases = [
            case for idx, case in enumerate(test_cases)
            if case_pass_counts[idx] / len(safe_functions) >= TEST_PASS_RATE
        ]

        success = len(passing_functions) >= MIN_FUNCTIONS and len(passing_cases) >= MIN_TEST_CASES

        return CrossValidationResult(
            success=success,
            passing_functions=passing_functions,
            passing_cases=passing_cases,
            best_accuracy=best_accuracy,
            total_functions=len(safe_functions),
            total_cases=len(test_cases),
        )
