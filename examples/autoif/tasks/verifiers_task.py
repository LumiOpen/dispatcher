"""Verifiers task - Generate evaluation functions and test cases separately, then cross-validate"""
from typing import Any, Dict, Generator, Union, List, Optional, Tuple
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

__all__ = ["GenerateVerifiersTask"]

NUM_FUNC_GENERATIONS = int(os.getenv("NUM_FUNC_GENERATIONS", 3))
NUM_TEST_CASES = int(os.getenv("NUM_TEST_CASES", 3))
MIN_FUNCTIONS = int(os.getenv('MIN_FUNCTIONS', 1))
MIN_TEST_CASES = int(os.getenv('MIN_TEST_CASES', 1))
FUNCTION_PASS_RATE = float(os.getenv('FUNCTION_PASS_RATE', 0.8))
LANGUAGE = os.getenv('LANGUAGE', 'en')

# Prompt templates
FUNC_PROMPT_PATH = "model_prompts/create_eval_function_prompt.j2"
TEST_CASES_PROMPT_PATH = "model_prompts/create_test_cases_prompt.j2"


class GenerateVerifiersTask(GeneratorTask):
    """
    Generate evaluation functions and test cases separately, then cross-validate.

    Flow:
    1. Generate NUM_FUNC_GENERATIONS evaluation functions
    2. Sample NUM_TEST_CASES random user queries from the provided dataset
    3. Generate 1 positive and 1 negative test case per query (all in one LLM call)
    4. Cross-validate functions against test cases
    5. Raise TaskFailed for verifiers that don't pass validation

    Input data format:
    {
        'instruction_id': str,
        'instruction': str,
        'instruction_category': str,
        'placeholders': dict,
        'user_queries': List[str]  # Dataset of user queries
    }
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

    def __init__(self, *args, **kwargs):
        self.parser = ResponseParser()
        self.executor = FunctionExecutor()
        self._jinja_env = None
        super().__init__(*args, **kwargs)

    def _get_jinja_env(self) -> jinja2.Environment:
        """Get or create Jinja2 environment."""
        if self._jinja_env is None:
            self._jinja_env = jinja2.Environment(
                loader=jinja2.FileSystemLoader('.'),
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
            entry = {'name': name, 'type': meta.get('type', 'unknown')}
            if meta.get('type') == 'numeric':
                entry['description'] = f"integer between {meta.get('min', 1)} and {meta.get('max', 10)}"
            elif meta.get('type') == 'static':
                entry['description'] = f"one of: {meta.get('values', [])}"
            elif meta.get('type') == 'dynamic':
                entry['description'] = "dynamically generated value"
            info.append(entry)
        return info

    def _sample_placeholder_values(self, placeholders: Dict) -> Dict[str, Any]:
        """Sample concrete values for placeholders."""
        values = {}
        for name, meta in placeholders.items():
            ptype = meta.get('type', 'unknown')
            if ptype == 'numeric':
                values[name] = random.randint(meta.get('min', 1), meta.get('max', 5))
            elif ptype == 'static':
                vals = meta.get('values', ['default'])
                values[name] = random.choice(vals) if vals else 'default'
            elif ptype == 'dynamic':
                # For dynamic placeholders, use a sample value
                values[name] = meta.get('sample', 'example_value')
            else:
                values[name] = 'unknown'
        return values

    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        """Generate evaluation functions and test cases, then cross-validate."""
        instruction_id = self.data.get('instruction_id', 'unknown')
        instruction = self.data.get('instruction', '')
        instruction_category = self.data.get('instruction_category', '')
        placeholders = self.data.get('placeholders', {})
        user_queries = self.data.get('user_queries', [])

        if not instruction:
            raise TaskFailed(message="Missing 'instruction' field", error_type="missing_instruction")

        if not user_queries:
            raise TaskFailed(message="No user queries provided", error_type="missing_user_queries")

        # Build placeholder info for function prompt
        placeholder_info = self._build_placeholder_info(placeholders)

        # =====================================================================
        # Phase 1: Generate evaluation functions (parallel via n parameter)
        # =====================================================================
        func_prompt = self._render_template(
            FUNC_PROMPT_PATH,
            instruction=instruction,
            placeholders=placeholder_info,
            language=LANGUAGE
        )
        logger.info(f"[GenerateVerifiersTask] IID:{instruction_id} function generation prompt: {func_prompt}")

        response: Response = yield Request({
            "messages": [{"role": "user", "content": func_prompt}],
            **self.FUNC_GEN_PARAMS
        })

        # Process multiple responses from n=NUM_FUNC_GENERATIONS
        all_functions: List[str] = []
        response_texts = response.get_text(n=NUM_FUNC_GENERATIONS)
        if response_texts:
            for text in response_texts:
                logger.info(f"[GenerateVerifiersTask] IID:{instruction_id} function generation response: {text}")
                func_str, error = self.parser.parse_function(text, instruction_id)
                logger.info(f"[GenerateVerifiersTask] IID:{instruction_id} function generation error: {error}")
                logger.info(f"[GenerateVerifiersTask] IID:{instruction_id} function generation func_str: {func_str}")
                if func_str:
                    all_functions.append(func_str)
        self.data['eval_funcs'] = all_functions

        # =====================================================================
        # Phase 2: Generate test cases - one positive and one negative per query
        # Sample NUM_TEST_CASES queries and generate all test cases in one LLM call
        # =====================================================================
        num_queries_to_sample = min(NUM_TEST_CASES, len(user_queries))
        sampled_queries = random.sample(user_queries, num_queries_to_sample)

        test_cases_prompt = self._render_template(
            TEST_CASES_PROMPT_PATH,
            instruction=instruction,
            placeholders=placeholder_info if placeholders else None,
            user_queries=sampled_queries,
            language=LANGUAGE
        )
        logger.info(f"[GenerateVerifiersTask] IID:{instruction_id} test cases generation prompt: {test_cases_prompt}")
        
        response: Response = yield Request({
            "messages": [{"role": "user", "content": test_cases_prompt}],
            **self.TEST_CASES_GEN_PARAMS
        })
        logger.info(f"[GenerateVerifiersTask] IID:{instruction_id} test cases generation response: {response.get_text()}")
        test_cases_data, error = self.parser.parse_test_cases(response.get_text(), instruction_id)
        logger.info(f"[GenerateVerifiersTask] IID:{instruction_id} test cases generation error: {error}")
        logger.info(f"[GenerateVerifiersTask] IID:{instruction_id} test cases generation test_cases_data: {test_cases_data}")
        self.data['test_cases_data'] = test_cases_data
        # =====================================================================
        # Phase 3: Cross-validate
        # =====================================================================
        result = self._cross_validate(
            instruction_id=instruction_id,
            instruction=instruction,
            instruction_category=instruction_category,
            placeholders=placeholders,
            functions=all_functions,
            test_cases_data=test_cases_data
        )

        return result

    def _cross_validate(
        self,
        instruction_id: str,
        instruction: str,
        instruction_category: str,
        placeholders: Dict[str, Any],
        functions: List[str],
        test_cases_data: Optional[Dict[str, List]]
    ) -> Dict[str, Any]:
        """Cross-validate functions against test cases."""
        
        # Check functions
        if not functions:
            raise TaskFailed(
                message=f"No valid functions parsed from {NUM_FUNC_GENERATIONS} generations",
                error_type=REASON_NO_VALID_FUNCTIONS
            )

        if len(functions) < MIN_FUNCTIONS:
            raise TaskFailed(
                message=f"Only {len(functions)} functions parsed, need {MIN_FUNCTIONS}",
                error_type=REASON_INSUFFICIENT_FUNCTIONS
            )

        # Check test cases
        if not test_cases_data:
            raise TaskFailed(
                message="Failed to parse test cases",
                error_type=REASON_NO_VALID_TEST_CASES
            )

        positive_cases = test_cases_data.get('positive', [])
        negative_cases = test_cases_data.get('negative', [])
        
        if len(positive_cases) + len(negative_cases) < MIN_TEST_CASES:
            raise TaskFailed(
                message=f"Only {len(positive_cases) + len(negative_cases)} test cases, need {MIN_TEST_CASES}",
                error_type=REASON_INSUFFICIENT_TEST_CASES
            )

        # Filter safe functions
        safe_functions = [f for f in functions if self.executor.is_safe_function(f)[0]]
        if not safe_functions:
            raise TaskFailed(
                message="No safe functions after safety filtering",
                error_type="no_safe_functions"
            )

        # Convert test cases to cross-validation format
        # Positive cases should return True, negative should return False
        all_cases = []
        for case in positive_cases:
            test_input = {'response': case['response']}
            # Add any placeholder values from the case
            for key, val in case.items():
                if key != 'response':
                    test_input[key] = val
            all_cases.append({'input': test_input, 'output': True})
        
        for case in negative_cases:
            test_input = {'response': case['response']}
            for key, val in case.items():
                if key != 'response':
                    test_input[key] = val
            all_cases.append({'input': test_input, 'output': False})

        # Deduplicate
        unique_cases = self._deduplicate_cases(all_cases)

        # Cross-validate
        final_functions, final_cases = self._run_cross_validation(safe_functions, unique_cases)

        if not final_cases:
            raise TaskFailed(
                message="No test cases pass any function",
                error_type=REASON_NO_PASSING_TEST_CASES
            )

        if not final_functions:
            raise TaskFailed(
                message=f"No functions meet {FUNCTION_PASS_RATE*100:.0f}% accuracy threshold",
                error_type=REASON_NO_FUNCTIONS_MEET_ACCURACY
            )

        return {
            'instruction_id': instruction_id,
            'instruction': instruction,
            'instruction_category': instruction_category,
            'placeholders': placeholders,
            'eval_func': final_functions,
            'cases': final_cases
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
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Cross-validate functions against test cases."""
        case_passes = [False] * len(test_cases)
        accurate_functions = []

        for func in functions:
            correct_count = 0

            for case_idx, test_case in enumerate(test_cases):
                passed, _ = self.executor.test_function(func, test_case, log_errors=False)
                if passed:
                    case_passes[case_idx] = True
                    correct_count += 1

            accuracy = correct_count / len(test_cases) if test_cases else 0
            if accuracy >= FUNCTION_PASS_RATE:
                accurate_functions.append(func)

        passing_cases = [
            case for idx, case in enumerate(test_cases)
            if case_passes[idx]
        ]

        return accurate_functions, passing_cases
