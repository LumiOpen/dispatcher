"""Tests for function_executor.py subprocess-based execution."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.utils.function_executor import (
    _run_in_subprocess,
    FunctionExecutor,
    extract_imports,
    FUNCTION_TIMEOUT,
)
from src.utils.constants import (
    REASON_HARMFUL_CODE_DETECTED,
    REASON_FUNCTION_EXECUTION_ERROR,
    REASON_MISSING_EVALUATE_FUNCTION,
    REASON_FUNCTION_TIMEOUT,
    REASON_INVALID_TEST_CASE_FORMAT,
    REASON_MISSING_TEST_CASE_FIELDS,
)
from dispatcher.taskmanager.task.base import TaskFailed


# ---------------------------------------------------------------------------
# Mock functions as code strings
# ---------------------------------------------------------------------------

FUNC_RETURNS_TRUE = """\
def evaluate(text):
    return True
"""

FUNC_RETURNS_FALSE = """\
def evaluate(text):
    return False
"""

FUNC_LENGTH_CHECK = """\
def evaluate(response, **kwargs):
    min_len = kwargs.get("min_length", 10)
    return len(response) >= min_len
"""

FUNC_WORD_COUNT = """\
def evaluate(response, **kwargs):
    threshold = kwargs.get("threshold", 3)
    return int(len(response.split()) >= threshold)
"""

FUNC_USES_RE = """\
import re
def evaluate(response, **kwargs):
    pattern = kwargs.get("pattern", r"\\d+")
    return bool(re.search(pattern, response))
"""

FUNC_DICT_INPUT = """\
def evaluate(response, min_words=5):
    return len(response.split()) >= min_words
"""

FUNC_NO_EVALUATE = """\
def check(text):
    return True
"""

FUNC_RAISES = """\
def evaluate(text):
    raise ValueError("intentional error")
"""

FUNC_RETURNS_NONE = """\
def evaluate(response, **kwargs):
    pass
"""

FUNC_SYNTAX_ERROR = """\
def evaluate(text)
    return True
"""

FUNC_SLOW = """\
import time
def evaluate(text):
    time.sleep(60)
    return True
"""

FUNC_MATH = """\
import math
def evaluate(response, **kwargs):
    return int(math.sqrt(len(response)))
"""

FUNC_UNSAFE_SUBPROCESS = """\
import subprocess
def evaluate(text):
    return True
"""

FUNC_UNSAFE_OS = """\
import os
def evaluate(text):
    os.listdir(".")
    return True
"""


# ===================================================================
# Tests for _run_in_subprocess
# ===================================================================

class TestRunInSubprocess:
    """Tests for the low-level subprocess runner."""

    def test_test_mode_returns_true(self):
        result = _run_in_subprocess(FUNC_RETURNS_TRUE, mode="test", timeout=5, test_input="hello")
        assert result == {"result": True}

    def test_test_mode_returns_false(self):
        result = _run_in_subprocess(FUNC_RETURNS_FALSE, mode="test", timeout=5, test_input="hello")
        assert result == {"result": False}

    def test_test_mode_with_dict_input(self):
        func = FUNC_DICT_INPUT
        result = _run_in_subprocess(func, mode="test", timeout=5, test_input={"response": "one two three four five"})
        assert result == {"result": True}

        result = _run_in_subprocess(func, mode="test", timeout=5, test_input={"response": "one two"})
        assert result == {"result": False}

    def test_execute_mode_returns_int(self):
        result = _run_in_subprocess(FUNC_MATH, mode="execute", timeout=5, response="hello world!!!")
        assert "result" in result
        assert isinstance(result["result"], int)

    def test_execute_mode_with_kwargs(self):
        result = _run_in_subprocess(
            FUNC_WORD_COUNT, mode="execute", timeout=5,
            response="one two three four", kwargs={"threshold": 3},
        )
        assert result == {"result": 1}

    def test_execute_mode_none_returns_error(self):
        result = _run_in_subprocess(FUNC_RETURNS_NONE, mode="execute", timeout=5, response="hello")
        assert "error" in result
        assert result["error_type"] == "returned_none"

    def test_no_evaluate_function(self):
        result = _run_in_subprocess(FUNC_NO_EVALUATE, mode="test", timeout=5, test_input="hello")
        assert "error" in result
        assert result["error_type"] == "no_evaluate_function"

    def test_function_raises_exception(self):
        result = _run_in_subprocess(FUNC_RAISES, mode="test", timeout=5, test_input="hello")
        assert "error" in result
        assert result["error_type"] == "execution_error"
        assert "intentional error" in result["error"]

    def test_syntax_error(self):
        result = _run_in_subprocess(FUNC_SYNTAX_ERROR, mode="test", timeout=5, test_input="hello")
        assert "error" in result

    def test_timeout(self):
        result = _run_in_subprocess(FUNC_SLOW, mode="test", timeout=2, test_input="hello")
        assert "error" in result
        assert result["error_type"] == "timeout"

    def test_stdlib_import(self):
        result = _run_in_subprocess(FUNC_USES_RE, mode="test", timeout=5, test_input="abc123")
        assert result == {"result": True}

        result = _run_in_subprocess(FUNC_USES_RE, mode="test", timeout=5, test_input="no digits")
        assert result == {"result": False}

    def test_empty_function_string(self):
        result = _run_in_subprocess("", mode="test", timeout=5, test_input="hello")
        assert "error" in result
        assert result["error_type"] == "no_evaluate_function"


# ===================================================================
# Tests for FunctionExecutor.test_function
# ===================================================================

class TestFunctionExecutorTestFunction:
    """Tests for FunctionExecutor.test_function via subprocess."""

    def setup_method(self):
        self.executor = FunctionExecutor()

    def test_passing_test_case_bool_true(self):
        case = {"input": "anything", "output": True}
        passed, reason = self.executor.test_function(FUNC_RETURNS_TRUE, case, log_errors=False)
        assert passed is True
        assert reason is None

    def test_passing_test_case_bool_false(self):
        case = {"input": "anything", "output": False}
        passed, reason = self.executor.test_function(FUNC_RETURNS_FALSE, case, log_errors=False)
        assert passed is True
        assert reason is None

    def test_failing_test_case(self):
        case = {"input": "anything", "output": False}
        passed, reason = self.executor.test_function(FUNC_RETURNS_TRUE, case, log_errors=False)
        assert passed is False
        assert reason is None

    def test_output_as_string_true(self):
        case = {"input": "anything", "output": "true"}
        passed, reason = self.executor.test_function(FUNC_RETURNS_TRUE, case, log_errors=False)
        assert passed is True

    def test_output_as_string_false(self):
        case = {"input": "anything", "output": "false"}
        passed, reason = self.executor.test_function(FUNC_RETURNS_FALSE, case, log_errors=False)
        assert passed is True

    def test_dict_input_kwargs(self):
        func = FUNC_DICT_INPUT
        case = {"input": {"response": "one two three four five six"}, "output": True}
        passed, reason = self.executor.test_function(func, case, log_errors=False)
        assert passed is True
        assert reason is None

    def test_dict_input_kwargs_failing(self):
        func = FUNC_DICT_INPUT
        case = {"input": {"response": "one", "min_words": 5}, "output": True}
        passed, reason = self.executor.test_function(func, case, log_errors=False)
        assert passed is False

    def test_invalid_test_case_not_dict(self):
        passed, reason = self.executor.test_function(FUNC_RETURNS_TRUE, "not a dict", log_errors=False)
        assert passed is False
        assert reason == REASON_INVALID_TEST_CASE_FORMAT

    def test_missing_input_key(self):
        case = {"output": True}
        passed, reason = self.executor.test_function(FUNC_RETURNS_TRUE, case, log_errors=False)
        assert passed is False
        assert reason == REASON_MISSING_TEST_CASE_FIELDS

    def test_missing_output_key(self):
        case = {"input": "hello"}
        passed, reason = self.executor.test_function(FUNC_RETURNS_TRUE, case, log_errors=False)
        assert passed is False
        assert reason == REASON_MISSING_TEST_CASE_FIELDS

    def test_no_evaluate_function(self):
        case = {"input": "hello", "output": True}
        passed, reason = self.executor.test_function(FUNC_NO_EVALUATE, case, log_errors=False)
        assert passed is False
        assert reason == REASON_MISSING_EVALUATE_FUNCTION

    def test_function_exception(self):
        case = {"input": "hello", "output": True}
        passed, reason = self.executor.test_function(FUNC_RAISES, case, log_errors=False)
        assert passed is False
        assert REASON_FUNCTION_EXECUTION_ERROR in reason

    def test_unsafe_function_rejected(self):
        case = {"input": "hello", "output": True}
        passed, reason = self.executor.test_function(FUNC_UNSAFE_SUBPROCESS, case, log_errors=False)
        assert passed is False
        assert REASON_HARMFUL_CODE_DETECTED in reason

    def test_timeout(self):
        import src.utils.function_executor as mod
        orig = mod.FUNCTION_TIMEOUT
        mod.FUNCTION_TIMEOUT = 2
        try:
            case = {"input": "hello", "output": True}
            passed, reason = self.executor.test_function(FUNC_SLOW, case, log_errors=False)
            assert passed is False
            assert reason == REASON_FUNCTION_TIMEOUT
        finally:
            mod.FUNCTION_TIMEOUT = orig


# ===================================================================
# Tests for FunctionExecutor.execute_with_response
# ===================================================================

class TestFunctionExecutorExecuteWithResponse:
    """Tests for FunctionExecutor.execute_with_response via subprocess."""

    def setup_method(self):
        self.executor = FunctionExecutor()

    def test_returns_int(self):
        result = self.executor.execute_with_response(FUNC_MATH, "hello world!!!")
        assert isinstance(result, int)

    def test_with_kwargs(self):
        result = self.executor.execute_with_response(
            FUNC_WORD_COUNT, "one two three four", threshold=3,
        )
        assert result == 1

    def test_returns_none_raises(self):
        with pytest.raises(TaskFailed) as exc_info:
            self.executor.execute_with_response(FUNC_RETURNS_NONE, "hello")
        assert exc_info.value.error_type == REASON_FUNCTION_EXECUTION_ERROR

    def test_no_evaluate_raises(self):
        with pytest.raises(TaskFailed) as exc_info:
            self.executor.execute_with_response(FUNC_NO_EVALUATE, "hello")
        assert exc_info.value.error_type == REASON_MISSING_EVALUATE_FUNCTION

    def test_unsafe_function_raises(self):
        with pytest.raises(TaskFailed) as exc_info:
            self.executor.execute_with_response(FUNC_UNSAFE_OS, "hello")
        assert exc_info.value.error_type == REASON_HARMFUL_CODE_DETECTED

    def test_exception_raises(self):
        with pytest.raises(TaskFailed) as exc_info:
            self.executor.execute_with_response(FUNC_RAISES, "hello")
        assert exc_info.value.error_type == REASON_FUNCTION_EXECUTION_ERROR

    def test_timeout_raises(self):
        import src.utils.function_executor as mod
        orig = mod.FUNCTION_TIMEOUT
        mod.FUNCTION_TIMEOUT = 2
        try:
            with pytest.raises(TaskFailed) as exc_info:
                self.executor.execute_with_response(FUNC_SLOW, "hello")
            assert exc_info.value.error_type == REASON_FUNCTION_TIMEOUT
        finally:
            mod.FUNCTION_TIMEOUT = orig

    def test_length_check_with_kwargs(self):
        result = self.executor.execute_with_response(
            FUNC_LENGTH_CHECK, "short", min_length=100,
        )
        assert result == 0

        result = self.executor.execute_with_response(
            FUNC_LENGTH_CHECK, "a long enough string for the test", min_length=5,
        )
        assert result == 1


# ===================================================================
# Tests for FunctionExecutor.is_safe_function
# ===================================================================

class TestIsSafeFunction:

    def setup_method(self):
        self.executor = FunctionExecutor()

    def test_safe_function(self):
        safe, reason = self.executor.is_safe_function(FUNC_RETURNS_TRUE)
        assert safe is True
        assert reason is None

    def test_safe_with_re(self):
        safe, reason = self.executor.is_safe_function(FUNC_USES_RE)
        assert safe is True

    def test_unsafe_subprocess(self):
        safe, reason = self.executor.is_safe_function(FUNC_UNSAFE_SUBPROCESS)
        assert safe is False
        assert REASON_HARMFUL_CODE_DETECTED in reason

    def test_unsafe_os(self):
        safe, reason = self.executor.is_safe_function(FUNC_UNSAFE_OS)
        assert safe is False
        assert REASON_HARMFUL_CODE_DETECTED in reason


# ===================================================================
# Tests for extract_imports
# ===================================================================

class TestExtractImports:

    def test_import_statement(self):
        code = "import re\ndef evaluate(x): return True"
        assert "re" in extract_imports(code)

    def test_from_import(self):
        code = "from collections import Counter\ndef evaluate(x): return True"
        assert "collections" in extract_imports(code)

    def test_nested_module(self):
        code = "import os.path\ndef evaluate(x): return True"
        assert "os" in extract_imports(code)

    def test_multiple_imports(self):
        code = "import re\nimport math\nfrom typing import List\ndef evaluate(x): return True"
        imports = extract_imports(code)
        assert imports == {"re", "math", "typing"}

    def test_no_imports(self):
        code = "def evaluate(x): return True"
        assert extract_imports(code) == set()

    def test_syntax_error_fallback(self):
        code = "import re\ndef evaluate(x)\n    return True"
        imports = extract_imports(code)
        assert "re" in imports


# ===================================================================
# Tests for subprocess memory isolation
# ===================================================================

class TestSubprocessIsolation:
    """Verify that each subprocess call is truly isolated."""

    def test_global_state_not_shared(self):
        """Globals set in one subprocess must not leak into the next."""
        func_set_global = """\
import json
GLOBAL_STATE = "leaked"
def evaluate(text):
    return True
"""
        func_check_global = """\
def evaluate(text):
    try:
        return GLOBAL_STATE == "leaked"
    except NameError:
        return False
"""
        r1 = _run_in_subprocess(func_set_global, mode="test", timeout=5, test_input="x")
        assert r1 == {"result": True}

        r2 = _run_in_subprocess(func_check_global, mode="test", timeout=5, test_input="x")
        assert r2 == {"result": False}

    def test_module_import_not_shared(self):
        """A heavy import in one call should not persist in a subsequent call."""
        func_import = """\
import json
def evaluate(text):
    return "json" in dir()
"""
        func_check = """\
def evaluate(text):
    try:
        json.dumps({})
        return True
    except NameError:
        return False
"""
        _run_in_subprocess(func_import, mode="test", timeout=5, test_input="x")
        r2 = _run_in_subprocess(func_check, mode="test", timeout=5, test_input="x")
        assert r2 == {"result": False}
