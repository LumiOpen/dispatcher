"""Handler class for verification functionality in autoif generator task."""

import os
import re
import json
from typing import List, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass
from .keyword_handler import KeywordHandler
from .lang_id import detect_language
from .function_executor import FunctionExecutor
from .error_utils import format_error_type_with_turn
from dispatcher.taskmanager.task.base import TaskFailed


@dataclass
class VerificationData:
    """Internal data class for verification data for a specific turn."""
    instruction_ids: List
    instructions: List
    eval_funcs: List
    cases: List
    instruction_categories: List
    kwargs: List


class VerificationHandler:
    """Handler class for performing response verification in autoif generator task.

    This class encapsulates all the data and logic needed for response verification,
    including instructions, evaluation functions, language checking, and keyword handling.
    """

    def __init__(self,
                 turn_idx: int,
                 instruction_ids: List[List],
                 instructions: List[List],
                 eval_funcs: List[List],
                 cases: List[List],
                 instruction_categories: List[List],
                 keyword_handler: Optional[KeywordHandler] = None):
        """Initialize VerificationHandler.

        Args:
            turn_idx: Current turn index
            instruction_ids: Instruction IDs per turn
            instructions: Instructions per turn
            eval_funcs: Evaluation functions per turn
            cases: Test cases per turn
            instruction_categories: Categories per turn
            keyword_handler: Optional keyword handler for keyword modifications
        """
        self.turn_idx = turn_idx
        self.keyword_handler = keyword_handler

        # Get target language from environment
        self.target_language = os.environ.get("LANGUAGE")

        # Prepare verification data for current turn
        self.data = VerificationData(
            instruction_ids=instruction_ids[turn_idx] if turn_idx < len(instruction_ids) else [],
            instructions=instructions[turn_idx] if turn_idx < len(instructions) else [],
            eval_funcs=eval_funcs[turn_idx] if turn_idx < len(eval_funcs) else [],
            cases=cases[turn_idx] if turn_idx < len(cases) else [],
            instruction_categories=instruction_categories[turn_idx] if turn_idx < len(instruction_categories) else [],
            kwargs=[]  # Will be populated by keyword handler if needed
        )

        # Add keyword generation data if available for this turn
        if self.keyword_handler and self.keyword_handler.has_keyword_instructions():
            self.data.kwargs = self.keyword_handler.get_execution_kwargs()

    def verify_response(self, response_text: str) -> None:
        """Verify the response using all validation checks.

        Args:
            response_text: The response to verify

        Raises:
            TaskFailed: If verification fails
        """
        # Step 1: Check for error response
        self._check_error(response_text, error_code='contradicting_constraints', turn=self.turn_idx)

        # Step 2: Check language
        self._check_language(response_text)

        # Step 3: Run evaluation functions
        self._run_evaluation_functions(response_text)

    def _check_error(response: str, error_code: Optional[str] = None, turn: Optional[int] = None):
        """
        Checks whether the response contains a JSON object with an error key.
        If error_code is provided, checks for that specific error.
        Handles LLM responses that may contain additional text and markdown formatting.

        Args:
            response: The response text to check for errors
            error_code: Optional specific error code to check for
            turn: Optional turn number (0-indexed) for error reporting

        Raises:
            TaskFailed: If an error is found in the response
        """
        def extract_json_from_response(text: str) -> Optional[Dict[str, Any]]:
            """Extract JSON object from LLM response that may contain markdown or extra text."""
            # First, try to find JSON within markdown code blocks
            json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
            match = re.search(json_pattern, text, re.DOTALL | re.IGNORECASE)

            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass

            # If no markdown blocks found, try to find JSON object in the text
            # Look for curly braces that might contain JSON
            brace_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(brace_pattern, text, re.DOTALL)

            for match in matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    continue

            # As a last resort, try to parse the entire response as JSON
            try:
                parsed = json.loads(text.strip())
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

            return None

        # Extract JSON from the response
        response_data = extract_json_from_response(response)

        if response_data and 'error' in response_data:
            error_value = response_data['error']

            # If a specific error_code is provided, check if it matches
            if error_code is not None:
                if error_value == error_code:
                    raise TaskFailed(
                        message=f"Expected error '{error_code}' found in response: {response}",
                        error_type=format_error_type_with_turn(error_code, turn)
                    )
            else:
                # If no specific error_code is provided, raise for any error
                raise TaskFailed(
                    message=f"Error found in response: {error_value}",
                    error_type=format_error_type_with_turn("error_in_response", turn)
                )

    def _check_language(self, response: str) -> None:
        """Check if the response is in the target language."""
        if not self.target_language:
            return  # Skip language check if no target language is set

        # Get language prediction
        try:
            lang_code1, lang_code2 = detect_language(response)
        except Exception as e:
            raise TaskFailed(
                message=f"Language detection error: {e} <response>{response}</response>",
                error_type=format_error_type_with_turn("language_detection", self.turn_idx)
            )

        valid_lang_ids = {lang_code1, lang_code2} if lang_code2 is not None else {lang_code1}
        if self.target_language not in valid_lang_ids:
            raise TaskFailed(
                message=f"The response is not in the expected language {self.target_language}, but got {valid_lang_ids} <response>{response}</response>",
                error_type=format_error_type_with_turn("invalid_language", self.turn_idx)
            )

    def _run_evaluation_functions(self, response: str) -> None:
        """Run evaluation functions and check accuracy."""
        # Check if evaluation functions exist
        if not self.data.eval_funcs:
            raise TaskFailed(
                message=f"No evaluation functions found",
                error_type=format_error_type_with_turn("no_eval_functions", self.turn_idx)
            )

        # Use FunctionExecutor for safe function execution
        executor = FunctionExecutor()

        # Use instruction_ids or create enumerated indices
        instruction_ids = self.data.instruction_ids or list(range(len(self.data.eval_funcs)))

        # Process each instruction group
        instruction_results = []
        accuracy_threshold = 0  # Threshold for instruction to pass

        for idx, instruction_funcs in enumerate(self.data.eval_funcs):
            instruction_id = instruction_ids[idx] if idx < len(instruction_ids) else idx

            if not instruction_funcs:
                raise TaskFailed(
                    message=f"No evaluation functions found for instruction {instruction_id}",
                    error_type=format_error_type_with_turn("no_eval_functions_for_instruction", self.turn_idx)
                )

            # Get kwargs for this specific instruction (if available)
            instruction_kwargs = {}
            if idx < len(self.data.kwargs):
                instruction_kwargs = self.data.kwargs[idx] if self.data.kwargs[idx] else {}

            # Run all functions for this instruction and collect results
            instruction_acc = []
            for func in instruction_funcs:
                try:
                    # Execute function with timeout protection
                    result = executor.execute_with_response(func, response, log_errors=True, **instruction_kwargs)
                    if result is not None:
                        instruction_acc.append(result)
                except Exception as e:
                    raise TaskFailed(
                        message=f"Error executing evaluation function {func} for instruction {instruction_id}: {e} <response>{response}</response>",
                        error_type=format_error_type_with_turn("function_execution_failed", self.turn_idx)
                    )

            # For this instruction, calculate accuracy
            instruction_accuracy = np.mean(instruction_acc) if instruction_acc else 0
            instruction_results.append(instruction_accuracy)

        # Check if ALL instructions pass the threshold (all instructions must be followed)
        failed_instructions = []
        for idx, accuracy in enumerate(instruction_results):
            instruction_id = instruction_ids[idx] if idx < len(instruction_ids) else idx
            if accuracy <= accuracy_threshold:
                failed_instructions.append((instruction_id, accuracy))

        if failed_instructions:
            # Format error message based on number of failed instructions
            if len(failed_instructions) == 1:
                instruction_id, accuracy = failed_instructions[0]
                raise TaskFailed(
                    message=f"The response did not pass verification for instruction {instruction_id} with accuracy {accuracy}. <response>{response}</response>",
                    error_type=format_error_type_with_turn("instruction_verification_failed", self.turn_idx)
                )
            else:
                failed_ids = [str(instr_id) for instr_id, _ in failed_instructions]
                raise TaskFailed(
                    message=f"The response did not pass verification for instructions {', '.join(failed_ids)}. <response>{response}</response>",
                    error_type=format_error_type_with_turn("multiple_instructions_verification_failed", self.turn_idx)
                )