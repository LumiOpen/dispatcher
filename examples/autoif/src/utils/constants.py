"""
Constants and reason codes for AutoIF cross-validation process.

This module defines all constants, thresholds, and error reason codes
used throughout the cross-validation process.
"""

# ---- Parser Error Reason Codes ----
REASON_FUNCTION_PARSE_FAILURE = "function_parse_failure"
REASON_CASES_PARSE_ERROR = "cases_parse_error"  
REASON_PARSE_FAILURE = "parse_failure"

# ---- Function Error Reason Codes ----
REASON_HARMFUL_CODE_DETECTED = "harmful_code_detected"
REASON_FUNCTION_EXECUTION_ERROR = "function_execution_error"
REASON_MISSING_EVALUATE_FUNCTION = "missing_evaluate_function"
REASON_FUNCTION_TIMEOUT = "function_timeout"

# ---- Test Case Error Reason Codes ----
REASON_NO_VALID_TEST_CASES = "no_valid_test_cases"
REASON_INVALID_TEST_CASE_FORMAT = "invalid_test_case_format"
REASON_MISSING_TEST_CASE_FIELDS = "missing_test_case_fields"

# ---- Cross-Validation Error Reason Codes ----
REASON_NO_VALID_FUNCTIONS = "no_valid_functions"
REASON_INSUFFICIENT_FUNCTIONS = "insufficient_functions"
REASON_INSUFFICIENT_TEST_CASES = "insufficient_test_cases"
REASON_NO_PASSING_TEST_CASES = "no_passing_test_cases"
REASON_LOW_FUNCTION_ACCURACY = "low_function_accuracy"
REASON_NO_FUNCTIONS_MEET_ACCURACY = "no_functions_meet_accuracy_threshold"

# ---- Safety Patterns ----
UNSAFE_CODE_PATTERNS = ['requests', 'subprocess', 'os.', 'sys.']
