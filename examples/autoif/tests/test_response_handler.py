#!/usr/bin/env python3
"""
Test script for response_handler functionality.

This script tests the check_error function from response_handler.py
with various response formats to ensure proper error detection.
"""

import sys
import os

# Add both the src directory and the parent directory to the path
current_dir = os.path.dirname(__file__)
src_dir = os.path.join(current_dir, '..', 'src')
autoif_dir = os.path.join(current_dir, '..')
dispatcher_root = os.path.join(current_dir, '..', '..', '..')

sys.path.insert(0, src_dir)
sys.path.insert(0, autoif_dir)
sys.path.insert(0, dispatcher_root)

from utils.response_handler import check_error, extract_score
from dispatcher.taskmanager.task.base import TaskFailed


class TestException(Exception):
    """Custom exception for test assertions."""
    pass


def assert_raises(exception_type, func, *args, **kwargs):
    """Helper function to assert that a function raises a specific exception."""
    try:
        func(*args, **kwargs)
        raise TestException(f"Expected {exception_type.__name__} but no exception was raised")
    except exception_type as e:
        return e
    except Exception as e:
        raise TestException(f"Expected {exception_type.__name__} but got {type(e).__name__}: {e}")


def test_check_error_with_plain_json():
    """Test check_error with plain JSON error response."""
    response = '{"error": "contradicting_constraints"}'
    
    # Should raise TaskFailed for any error
    exc = assert_raises(TaskFailed, check_error, response)
    
    assert exc.error_type == "error_in_response"
    assert "contradicting_constraints" in str(exc.message)


def test_check_error_with_specific_error_code():
    """Test check_error with specific error code matching."""
    response = '{"error": "contradicting_constraints"}'
    
    # Should raise TaskFailed when specific error matches
    exc = assert_raises(TaskFailed, check_error, response, "contradicting_constraints")
    
    assert exc.error_type == "contradicting_constraints"
    assert "Expected error 'contradicting_constraints' found" in str(exc.message)


def test_check_error_with_specific_error_code_no_match():
    """Test check_error with specific error code that doesn't match."""
    response = '{"error": "contradicting_constraints"}'
    
    # Should not raise TaskFailed when specific error doesn't match
    try:
        check_error(response, "different_error")
    except TaskFailed:
        raise TestException("check_error should not raise TaskFailed when error codes don't match")


def test_check_error_with_markdown_json():
    """Test check_error with JSON wrapped in markdown code blocks."""
    response = '''Here's the error information:

```json
{
  "error": "contradicting_constraints"
}
```

This indicates that there was a constraint violation.'''
    
    exc = assert_raises(TaskFailed, check_error, response)
    
    assert exc.error_type == "error_in_response"
    assert "contradicting_constraints" in str(exc.message)


def test_check_error_with_markdown_no_json_label():
    """Test check_error with JSON in markdown blocks without 'json' label."""
    response = '''The system encountered an issue:

```
{
  "error": "contradicting_constraints"
}
```

Please try again later.'''
    
    exc = assert_raises(TaskFailed, check_error, response)
    
    assert exc.error_type == "error_in_response"
    assert "contradicting_constraints" in str(exc.message)


def test_check_error_with_json_in_text():
    """Test check_error with JSON embedded in text without markdown."""
    response = 'The system encountered an issue: {"error": "network_error"} which needs to be addressed.'
    
    exc = assert_raises(TaskFailed, check_error, response)
    
    assert exc.error_type == "error_in_response"
    assert "network_error" in str(exc.message)


def test_check_error_with_complex_llm_response():
    """Test check_error with a complex LLM response containing explanation and error."""
    response = '''I understand you're asking me to perform this task, but I need to report an issue.

After analyzing the requirements, I found the following problem:

```json
{
  "error": "insufficient_data",
  "description": "Not enough training data available"
}
```

This means I cannot complete the requested operation at this time. Please provide more data and try again.'''
    
    exc = assert_raises(TaskFailed, check_error, response)
    
    assert exc.error_type == "error_in_response"
    assert "insufficient_data" in str(exc.message)


def test_check_error_case_sensitivity():
    """Test check_error with different case variations."""
    response = '{"Error": "case_sensitive_error"}'
    
    # Should not raise TaskFailed because key is "Error" not "error"
    try:
        check_error(response)
    except TaskFailed:
        raise TestException("check_error should be case-sensitive for the 'error' key")


def test_extract_score_basic():
    """Test extract_score with basic score patterns."""
    test_cases = [
        ("Score: 5", 5),
        ("Some text\nScore: 3", 3),
        ("Analysis complete\nScore: 1", 1),
        ("Score: 4  \n", 4),
        ("Score: 2\t", 2),
    ]
    
    for text, expected_score in test_cases:
        result = extract_score(text)
        if result != expected_score:
            raise TestException(f"Expected {expected_score}, got {result} for text: '{text[:50]}...'")


def test_extract_score_markdown_formatting():
    """Test extract_score with markdown formatted scores."""
    test_cases = [
        ("Analysis complete. **Score: 3**", 3),
        ("Final result: `Score: 2`", 2),
    ]
    
    for text, expected_score in test_cases:
        result = extract_score(text)
        if result != expected_score:
            raise TestException(f"Expected {expected_score}, got {result} for text: '{text[:50]}...'")


def test_extract_score_multiple_scores():
    """Test extract_score with multiple scores - should return the last valid one."""
    test_cases = [
        ("Multiple scores: Score: 3 and Score: 5", 5)
    ]
    
    for text, expected_score in test_cases:
        result = extract_score(text)
        if result != expected_score:
            raise TestException(f"Expected {expected_score}, got {result} for text: '{text}'")


def test_extract_score_error_cases():
    """Test extract_score with cases that should raise TaskFailed."""
    error_cases = [
        "No score here",
        "Score: 5 but more text after",
        "Score: not_a_number",
        "Some random text",
        "Score mentioned but no number",
        "Score: without number",
    ]
    
    for text in error_cases:
        try:
            result = extract_score(text)
            raise TestException(f"'{text}' should have raised TaskFailed but returned {result}")
        except TaskFailed as e:
            if e.error_type != "score_extraction_failed":
                raise TestException(f"'{text}' raised TaskFailed with wrong error_type: {e.error_type}")
        except Exception as e:
            raise TestException(f"'{text}' raised unexpected exception: {e}")


def run_all_tests():
    """Run all tests manually (for environments without pytest)."""
    test_functions = [
        test_check_error_with_plain_json,
        test_check_error_with_specific_error_code,
        test_check_error_with_specific_error_code_no_match,
        test_check_error_with_markdown_json,
        test_check_error_with_markdown_no_json_label,
        test_check_error_with_json_in_text,
        test_check_error_with_complex_llm_response,
        test_check_error_case_sensitivity,
        test_extract_score_basic,
        test_extract_score_markdown_formatting,
        test_extract_score_multiple_scores,
        test_extract_score_error_cases
    ]
    
    passed = 0
    failed = 0
    
    print("Running response_handler function tests...")
    print("=" * 60)
    
    for test_func in test_functions:
        try:
            test_func()
            print(f"✅ {test_func.__name__}")
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__}: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Summary: {passed} passed, {failed} failed")
    print(f"Success rate: {(passed/(passed+failed))*100:.1f}%")
    
    return failed == 0


if __name__ == "__main__":
    # Running directly
    success = run_all_tests()
    sys.exit(0 if success else 1)
