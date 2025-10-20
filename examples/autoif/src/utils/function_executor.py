"""
Utility functions for parsing and validating functions and test cases.

This module contains reusable functions for:
- Parsing function and test case data from responses
- Validating function safety
- Testing function execution against test cases
- Deduplicating test cases
"""

import json
import signal
from typing import List, Dict, Any, Tuple, Optional
import os

from .constants import (
    UNSAFE_CODE_PATTERNS,
    REASON_FUNCTION_PARSE_FAILURE,
    REASON_CASES_PARSE_ERROR,
    REASON_PARSE_FAILURE,
    REASON_HARMFUL_CODE_DETECTED,
    REASON_FUNCTION_EXECUTION_ERROR,
    REASON_MISSING_EVALUATE_FUNCTION,
    REASON_FUNCTION_TIMEOUT,
    REASON_INVALID_TEST_CASE_FORMAT,
    REASON_MISSING_TEST_CASE_FIELDS
)
from .logging_utils import CrossValidationLogger
from dispatcher.taskmanager.task.base import TaskFailed

FUNCTION_TIMEOUT = int(os.getenv('FUNCTION_TIMEOUT', 10))  # seconds

class TimeoutError(Exception):
    """Custom timeout exception for function execution."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for function execution timeout."""
    raise TimeoutError("Function execution timed out")


class FunctionExecutor:
    """Handles validation and testing of functions."""
    
    def __init__(self, logger: Optional[CrossValidationLogger] = None):
        self.logger = logger
    
    def is_safe_function(self, func_str: str) -> Tuple[bool, Optional[str]]:
        """
        Check if function contains potentially harmful imports or operations.
        
        Args:
            func_str: Function code as string
            
        Returns:
            Tuple of (is_safe, error_reason)
        """
        for pattern in UNSAFE_CODE_PATTERNS:
            if pattern in func_str.lower():
                reason = f"{REASON_HARMFUL_CODE_DETECTED}_{pattern.replace(' ', '_')}"
                # Don't log here - let the caller decide if they want to log
                return False, reason
        
        return True, None

    def execute_with_response(self, func_str: str, response: str, **kwargs) -> int:
        """
        Execute an evaluation function with a response string.

        Args:
            func_str: Function code as string
            response: Response text to evaluate
            **kwargs: Additional keyword arguments to pass to the evaluation function

        Returns:
            Integer result

        Raises:
            TaskFailed: If execution fails for any reason
        """
        # Check function safety first
        is_safe, reason = self.is_safe_function(func_str)
        if not is_safe:
            raise TaskFailed(
                message=f"Unsafe function detected: {reason}",
                error_type=REASON_HARMFUL_CODE_DETECTED
            )
        
        try:
            # Create isolated namespace and execute function
            namespace = {}
            exec(func_str, namespace)
            
            # Get evaluate function
            evaluate_func = namespace.get('evaluate')
            if evaluate_func is None:
                raise TaskFailed(
                    message="No 'evaluate' function found in the provided code",
                    error_type=REASON_MISSING_EVALUATE_FUNCTION
                )
            
            # Set up timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(FUNCTION_TIMEOUT)
            
            try:
                # Pass response as first argument and any additional kwargs
                result = evaluate_func(response, **kwargs)
                if result is None:
                    raise TaskFailed(
                        message="Evaluation function returned None",
                        error_type=REASON_FUNCTION_EXECUTION_ERROR
                    )
                return int(result)
            finally:
                signal.alarm(0)  # Disable the alarm
                
        except TimeoutError:
            raise TaskFailed(
                message=f"Function timed out after {FUNCTION_TIMEOUT}s",
                error_type=REASON_FUNCTION_TIMEOUT
            )
        except Exception as e:
            raise TaskFailed(
                message=f"Execution error: {str(e)}",
                error_type=REASON_FUNCTION_EXECUTION_ERROR
            )
    
    def test_function(self, func_str: str, test_case: Dict[str, Any], 
                     func_idx: int = -1, case_idx: int = -1, instruction_id: str = "unknown",
                     log_errors: bool = True) -> Tuple[bool, Optional[str]]:
        """
        Test if a function passes a test case.
        
        Args:
            func_str: Function code as string
            test_case: Test case with 'input' and 'output' keys
            func_idx: Function index for logging
            case_idx: Test case index for logging
            instruction_id: Instruction ID for error logging
            log_errors: Whether to log individual errors (False during cross-validation)
            
        Returns:
            Tuple of (passed, error_reason)
        """
        id_str = f"func{func_idx}_case{case_idx}" if func_idx >= 0 and case_idx >= 0 else ""
        
        # Validate test case format
        if not isinstance(test_case, dict):
            if log_errors:
                error_msg = f"{id_str} Test case is not a dict: {type(test_case)}"
                self.logger.log_error(instruction_id, error_msg)
            return False, REASON_INVALID_TEST_CASE_FORMAT
        
        if 'input' not in test_case or 'output' not in test_case:
            if log_errors:
                error_msg = f"{id_str} Missing input/output in test case"
                self.logger.log_error(instruction_id, error_msg)
            return False, REASON_MISSING_TEST_CASE_FIELDS
        
        # Check function safety
        is_safe, reason = self.is_safe_function(func_str)
        if not is_safe:
            if log_errors:
                error_msg = f"{id_str} Unsafe function detected: {reason}"
                self.logger.log_error(instruction_id, error_msg)
            return False, reason
        
        # Execute function with timeout
        return self._execute_function_with_timeout(func_str, test_case, id_str, instruction_id, log_errors)
    
    def _execute_function_with_timeout(self, func_str: str, test_case: Dict[str, Any], 
                                     id_str: str, instruction_id: str, log_errors: bool = True) -> Tuple[bool, Optional[str]]:
        """Execute function with timeout protection."""
        try:
            # Create namespace and execute function
            namespace = {}
            exec(func_str, namespace)
            
            # Get evaluate function
            evaluate_func = namespace.get('evaluate')
            if evaluate_func is None:
                if log_errors:
                    self.logger.log_error(instruction_id, f"{id_str} No evaluate function found")
                return False, REASON_MISSING_EVALUATE_FUNCTION
            
            # Set up timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(FUNCTION_TIMEOUT)
            
            try:
                # Run the test case
                # Handle new format with kwargs (dict input) vs old format (single input)
                if isinstance(test_case['input'], dict):
                    # New format: pass all key-value pairs as keyword arguments
                    result = evaluate_func(**test_case['input'])
                else:
                    # Old format: pass input as single positional argument
                    result = evaluate_func(test_case['input'])
                
                # Normalize expected output
                expected = (test_case['output'] if isinstance(test_case['output'], bool) 
                          else test_case['output'].lower() == 'true')
                
                return result == expected, None
            finally:
                signal.alarm(0)  # Disable the alarm
                
        except TimeoutError:
            if log_errors:
                self.logger.log_error(instruction_id, f"{id_str} Function timed out after {FUNCTION_TIMEOUT}s")
            return False, REASON_FUNCTION_TIMEOUT
        except Exception as e:
            if log_errors:
                self.logger.log_error(instruction_id, f"{id_str} Execution error: {str(e)}")
            return False, f"{REASON_FUNCTION_EXECUTION_ERROR}: {str(e)}"
