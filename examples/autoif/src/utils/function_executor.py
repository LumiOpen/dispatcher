"""
Utility functions for parsing and validating functions and test cases.

This module contains reusable functions for:
- Parsing function and test case data from responses
- Validating function safety
- Testing function execution against test cases
- Deduplicating test cases
- Auto-installing missing dependencies
"""

import ast
import importlib.util
import json
import signal
import subprocess
import sys
import re
from typing import List, Dict, Any, Tuple, Optional, Set
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

FUNCTION_TIMEOUT = int(os.getenv('FUNCTION_TIMEOUT', 10))  # seconds
INIT_TIMEOUT = int(os.getenv('INIT_TIMEOUT', 0)) or FUNCTION_TIMEOUT  # exec/import phase
CASE_TIMEOUT = int(os.getenv('CASE_TIMEOUT', 30))  # per-test-case timeout
INSTALL_TIMEOUT = int(os.getenv('INSTALL_TIMEOUT', 60))  # seconds for pip install
AUTO_INSTALL_PACKAGES = os.getenv('AUTO_INSTALL_PACKAGES', 'true').lower() == 'true'

# Module-level worker pool; set via set_worker_pool() before cross-validation.
_worker_pool = None


def set_worker_pool(pool) -> None:
    """Assign a :class:`WorkerPool` for function evaluation.

    When set, ``test_function_batch`` dispatches work to pool workers
    instead of spawning a fresh subprocess per function.
    """
    global _worker_pool
    _worker_pool = pool


def _run_in_worker_batch(func_str: str, test_inputs: list,
                         init_timeout: int = 0, case_timeout: int = 0,
                         log_file: Optional[str] = None) -> Dict[str, Any]:
    """Run one function against all test inputs via the worker pool."""
    _init_t = init_timeout or INIT_TIMEOUT
    _case_t = case_timeout or CASE_TIMEOUT

    result = _worker_pool.run_function(
        func_str, mode="test_batch",
        init_timeout=_init_t, case_timeout=_case_t,
        test_inputs=test_inputs,
    )

    if log_file:
        _append_to_log(log_file, "--- (executed via worker pool) ---\n")

    return result


def _run_in_worker(func_str: str, mode: str,
                   init_timeout: int = 0, case_timeout: int = 0,
                   **extras) -> Dict[str, Any]:
    """Run one function via the worker pool (single-case modes)."""
    _init_t = init_timeout or INIT_TIMEOUT
    _case_t = case_timeout or CASE_TIMEOUT
    return _worker_pool.run_function(
        func_str, mode=mode,
        init_timeout=_init_t, case_timeout=_case_t,
        **extras,
    )


# Packages that are known to have different import names vs pip names
PACKAGE_NAME_MAPPING = {
    'cv2': 'opencv-python',
    'PIL': 'Pillow',
    'sklearn': 'scikit-learn',
    'yaml': 'pyyaml',
    'bs4': 'beautifulsoup4',
    'dateutil': 'python-dateutil',
}

# Packages that should never be auto-installed (security/size concerns)
BLOCKED_PACKAGES = {
    'tensorflow', 'torch', 'pytorch', 'keras',  # Too large
    'paramiko', 'fabric', 'ansible',  # SSH/remote execution
    'boto3', 'google-cloud', 'azure',  # Cloud SDKs
}

# Cache of already installed packages to avoid repeated checks
_installed_packages_cache: Set[str] = set()

# Extra seconds added to subprocess.run timeout beyond the SIGALRM timeout
# inside the runner, to account for process startup overhead.
_SUBPROCESS_OVERHEAD = 10


# Script executed in an isolated subprocess for memory-safe function evaluation.
# Reads JSON payload from stdin, exec's the function, runs evaluate(), and
# writes a JSON result to stdout.  When the subprocess exits, all memory
# (loaded NLP models, etc.) is fully reclaimed by the OS.
_SUBPROCESS_RUNNER = r'''
import sys, os, io, json, signal

def _timeout_handler(signum, frame):
    raise TimeoutError("timeout")

try:
    data = json.loads(sys.stdin.read())
except Exception as e:
    json.dump({"error": f"payload_parse: {e}"}, sys.stdout)
    sys.exit(0)

func_str = data["func_str"]
mode = data["mode"]
init_timeout = data.get("init_timeout", data.get("timeout", 10))
case_timeout = data.get("case_timeout", data.get("timeout", 10))

_real_stdout = sys.stdout
sys.stdout = sys.stderr

signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm(init_timeout)

out = None
try:
    namespace = {}
    exec(func_str, namespace)
    evaluate_func = namespace.get("evaluate")
    if evaluate_func is None:
        out = {"error_type": "no_evaluate_function", "error": "No evaluate function found"}
    elif mode == "test":
        signal.alarm(case_timeout)
        test_input = data["test_input"]
        if isinstance(test_input, dict):
            result = evaluate_func(**test_input)
        else:
            result = evaluate_func(test_input)
        out = {"result": result}
    elif mode == "test_batch":
        test_inputs = data["test_inputs"]
        case_results = []
        for ti in test_inputs:
            signal.alarm(case_timeout)
            try:
                if isinstance(ti, dict):
                    result = evaluate_func(**ti)
                else:
                    result = evaluate_func(ti)
                case_results.append({"result": result})
            except TimeoutError:
                case_results.append({"error_type": "timeout", "error": f"Function timed out after {case_timeout}s"})
            except Exception as e:
                case_results.append({"error_type": "execution_error", "error": str(e)})
        out = {"results": case_results}
    elif mode == "execute":
        signal.alarm(case_timeout)
        response = data["response"]
        kwargs = data.get("kwargs", {})
        result = evaluate_func(response, **kwargs)
        if result is None:
            out = {"error_type": "returned_none", "error": "Evaluation function returned None"}
        else:
            out = {"result": int(result)}

except TimeoutError:
    out = {"error_type": "timeout", "error": f"Function timed out after {init_timeout}s (init phase)"}
except Exception as e:
    out = {"error_type": "execution_error", "error": str(e)}
finally:
    signal.alarm(0)
    sys.stdout = _real_stdout
    if out is not None:
        json.dump(out, sys.stdout)
'''


def _run_in_subprocess(func_str: str, mode: str, timeout: int = FUNCTION_TIMEOUT,
                       init_timeout: int = 0, case_timeout: int = 0,
                       **extra_payload) -> Dict[str, Any]:
    """
    Execute function code in an isolated subprocess.

    Returns a dict with either {"result": ...} or {"error": ..., "error_type": ...}.
    All memory used by the subprocess (imported modules, model objects, etc.)
    is fully reclaimed when it exits.
    """
    _init_t = init_timeout or INIT_TIMEOUT
    _case_t = case_timeout or CASE_TIMEOUT
    payload = {"func_str": func_str, "mode": mode,
               "init_timeout": _init_t, "case_timeout": _case_t,
               **extra_payload}
    wall_timeout = _init_t + _case_t + _SUBPROCESS_OVERHEAD

    try:
        proc = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS_RUNNER],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=wall_timeout,
        )
    except subprocess.TimeoutExpired:
        return {"error_type": "timeout", "error": f"Subprocess timed out after {wall_timeout}s"}

    if proc.returncode != 0:
        stderr_tail = proc.stderr[-500:] if proc.stderr else "(no stderr)"
        return {"error_type": "execution_error", "error": f"Subprocess exited with code {proc.returncode}: {stderr_tail}"}

    if not proc.stdout.strip():
        return {"error_type": "execution_error", "error": "Subprocess produced no output"}

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"error_type": "execution_error", "error": f"Bad JSON from subprocess: {proc.stdout[:200]}"}


def _append_to_log(log_file: str, text: str) -> None:
    """Append text to a log file (creates if needed)."""
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text)


def _run_in_subprocess_batch(func_str: str, test_inputs: list,
                             timeout: int = FUNCTION_TIMEOUT,
                             init_timeout: int = 0, case_timeout: int = 0,
                             log_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Execute one function against *all* test inputs in a single subprocess.

    The model / heavy imports load once during exec() (governed by
    *init_timeout*); each test case then gets its own SIGALRM-based
    *case_timeout* inside the subprocess.

    If *log_file* is given, ``proc.stderr`` is appended to it after the
    subprocess finishes (regardless of success or failure).

    Returns {"results": [per-case dict]} on success, or a top-level
    {"error": ...} if the function itself fails to load.
    """
    _init_t = init_timeout or INIT_TIMEOUT
    _case_t = case_timeout or CASE_TIMEOUT
    num_cases = len(test_inputs)
    total_timeout = _init_t + _case_t * num_cases + _SUBPROCESS_OVERHEAD
    payload = {
        "func_str": func_str,
        "mode": "test_batch",
        "init_timeout": _init_t,
        "case_timeout": _case_t,
        "test_inputs": test_inputs,
    }

    stderr_text = ""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS_RUNNER],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=total_timeout,
        )
        stderr_text = proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        stderr_text = (exc.stderr or b"").decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        if log_file:
            _append_to_log(log_file, f"--- stderr ---\n{stderr_text}\n")
        return {"error_type": "timeout",
                "error": f"Batch subprocess timed out after {total_timeout}s"}

    if log_file and stderr_text:
        _append_to_log(log_file, f"--- stderr ---\n{stderr_text}\n")

    if proc.returncode != 0:
        stderr_tail = stderr_text[-500:] if stderr_text else "(no stderr)"
        return {"error_type": "execution_error",
                "error": f"Subprocess exited with code {proc.returncode}: {stderr_tail}"}

    if not proc.stdout.strip():
        return {"error_type": "execution_error",
                "error": "Subprocess produced no output"}

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"error_type": "execution_error",
                "error": f"Bad JSON from subprocess: {proc.stdout[:200]}"}


class TimeoutError(Exception):
    """Custom timeout exception for function execution."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for function execution timeout."""
    raise TimeoutError("Function execution timed out")


def extract_imports(func_str: str) -> Set[str]:
    """
    Extract top-level module names from import statements in code.

    Args:
        func_str: Python code as string

    Returns:
        Set of module names that are imported
    """
    imports = set()

    try:
        tree = ast.parse(func_str)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Get the top-level module name (e.g., 'numpy' from 'numpy.random')
                    top_module = alias.name.split('.')[0]
                    imports.add(top_module)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top_module = node.module.split('.')[0]
                    imports.add(top_module)
    except SyntaxError:
        # If AST parsing fails, try regex as fallback
        import_pattern = r'^(?:from\s+(\w+)|import\s+(\w+))'
        for match in re.finditer(import_pattern, func_str, re.MULTILINE):
            module = match.group(1) or match.group(2)
            if module:
                imports.add(module)

    return imports


def get_pip_package_name(import_name: str) -> str:
    """
    Get the pip package name for an import.

    Args:
        import_name: The name used in the import statement

    Returns:
        The pip package name to install
    """
    return PACKAGE_NAME_MAPPING.get(import_name, import_name)


def is_module_available(module_name: str) -> bool:
    """
    Check if a module is available for import without actually loading it.

    Uses importlib.util.find_spec to probe the module search path instead of
    __import__, which would pull the module (and all its transitive deps like
    spacy/trankit models) into the parent process's memory permanently.
    """
    if module_name in _installed_packages_cache:
        return True

    try:
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            _installed_packages_cache.add(module_name)
            return True
        return False
    except (ModuleNotFoundError, ValueError):
        return False


def install_package(package_name: str, timeout: int = INSTALL_TIMEOUT) -> Tuple[bool, str]:
    """
    Install a package using pip.

    Args:
        package_name: The pip package name to install
        timeout: Timeout in seconds for the installation

    Returns:
        Tuple of (success, message)
    """
    if package_name.lower() in BLOCKED_PACKAGES:
        return False, f"Package '{package_name}' is blocked from auto-installation"

    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--quiet', package_name],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode == 0:
            return True, f"Successfully installed {package_name}"
        else:
            return False, f"Failed to install {package_name}: {result.stderr}"

    except subprocess.TimeoutExpired:
        return False, f"Installation of {package_name} timed out after {timeout}s"
    except Exception as e:
        return False, f"Error installing {package_name}: {str(e)}"


def ensure_imports_available(func_str: str, logger=None, instruction_id: str = "unknown") -> Tuple[bool, List[str]]:
    """
    Ensure all imports in a function are available, installing if necessary.

    Args:
        func_str: Python code as string
        logger: Optional logger for messages
        instruction_id: ID for logging

    Returns:
        Tuple of (all_available, list_of_installed_packages)
    """
    if not AUTO_INSTALL_PACKAGES:
        return True, []

    imports = extract_imports(func_str)
    installed = []

    # Filter out standard library modules
    stdlib_modules = {
        're', 'os', 'sys', 'json', 'math', 'random', 'string', 'collections',
        'itertools', 'functools', 'datetime', 'time', 'typing', 'copy',
        'hashlib', 'base64', 'urllib', 'html', 'xml', 'csv', 'io',
        'pathlib', 'glob', 'shutil', 'tempfile', 'logging', 'warnings',
        'unittest', 'doctest', 'pdb', 'traceback', 'inspect', 'dis',
        'abc', 'contextlib', 'decimal', 'fractions', 'numbers', 'cmath',
        'statistics', 'operator', 'heapq', 'bisect', 'array', 'weakref',
        'types', 'enum', 'graphlib', 'pprint', 'reprlib', 'difflib',
        'textwrap', 'unicodedata', 'stringprep', 'readline', 'rlcompleter',
        'struct', 'codecs', 'locale', 'gettext', 'secrets', 'hmac',
        'secrets', 'pickle', 'copyreg', 'shelve', 'marshal', 'dbm',
        'sqlite3', 'zlib', 'gzip', 'bz2', 'lzma', 'zipfile', 'tarfile',
        'configparser', 'netrc', 'xdrlib', 'plistlib', 'crypt', 'tty',
        'termios', 'pty', 'fcntl', 'pipes', 'resource', 'syslog',
        'socket', 'ssl', 'select', 'selectors', 'asyncio', 'signal',
        'mmap', 'email', 'mailbox', 'mimetypes', 'binascii', 'quopri',
        'uu', 'http', 'ftplib', 'poplib', 'imaplib', 'nntplib', 'smtplib',
        'smtpd', 'telnetlib', 'uuid', 'socketserver', 'xmlrpc', 'ipaddress',
        'audioop', 'aifc', 'sunau', 'wave', 'chunk', 'colorsys', 'imghdr',
        'sndhdr', 'ossaudiodev', 'getpass', 'curses', 'platform', 'errno',
        'ctypes', 'multiprocessing', 'concurrent', 'subprocess', 'sched',
        'queue', 'contextvars', 'thread', 'threading', 'dummy_threading',
        '_thread', 'dataclasses', 'builtins', 'keyword', 'symbol', 'token',
        'tokenize', 'tabnanny', 'compileall', 'py_compile', 'zipimport',
        'pkgutil', 'modulefinder', 'runpy', 'importlib', 'parser', 'ast',
        'symtable', 'gc', 'atexit', 'faulthandler',
    }

    for module in imports:
        if module in stdlib_modules:
            continue

        if is_module_available(module):
            continue

        # Try to install the package
        pip_name = get_pip_package_name(module)

        if logger:
            logger.log_info(instruction_id, f"Auto-installing missing package: {pip_name}")

        success, message = install_package(pip_name)

        if success:
            installed.append(pip_name)
            _installed_packages_cache.add(module)
            if logger:
                logger.log_info(instruction_id, message)
        else:
            if logger:
                logger.log_error(instruction_id, message)
            return False, installed

    return True, installed


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
                return False, reason

        return True, None

    def execute_with_response(self, func_str: str, response: str, **kwargs) -> int:
        """
        Execute an evaluation function with a response string.

        Runs the generated code in an isolated subprocess so that imported
        modules and model objects are fully freed when the subprocess exits.

        Args:
            func_str: Function code as string
            response: Response text to evaluate
            **kwargs: Additional keyword arguments to pass to the evaluation function

        Returns:
            Integer result

        Raises:
            TaskFailed: If execution fails for any reason
        """
        from dispatcher.taskmanager.task.base import TaskFailed

        is_safe, reason = self.is_safe_function(func_str)
        if not is_safe:
            raise TaskFailed(
                message=f"Unsafe function detected: {reason}",
                error_type=REASON_HARMFUL_CODE_DETECTED
            )

        imports_ok, installed = ensure_imports_available(
            func_str,
            logger=self.logger,
            instruction_id="execute_with_response"
        )
        if not imports_ok:
            raise TaskFailed(
                message="Failed to install required dependencies",
                error_type=REASON_FUNCTION_EXECUTION_ERROR
            )

        if _worker_pool is not None:
            result = _run_in_worker(
                func_str, mode="execute",
                init_timeout=INIT_TIMEOUT, case_timeout=CASE_TIMEOUT,
                response=response, kwargs=kwargs,
            )
        else:
            result = _run_in_subprocess(
                func_str, mode="execute",
                init_timeout=INIT_TIMEOUT, case_timeout=CASE_TIMEOUT,
                response=response, kwargs=kwargs,
            )

        if "error" in result:
            error_type = result.get("error_type", "execution_error")
            error_map = {
                "no_evaluate_function": REASON_MISSING_EVALUATE_FUNCTION,
                "timeout": REASON_FUNCTION_TIMEOUT,
                "returned_none": REASON_FUNCTION_EXECUTION_ERROR,
                "execution_error": REASON_FUNCTION_EXECUTION_ERROR,
            }
            raise TaskFailed(
                message=result["error"],
                error_type=error_map.get(error_type, REASON_FUNCTION_EXECUTION_ERROR)
            )

        return result["result"]

    def test_function_batch(
        self,
        func_str: str,
        test_cases: List[Dict[str, Any]],
        log_file: Optional[str] = None,
        func_idx: int = -1,
    ) -> List[Tuple[bool, Optional[str]]]:
        """
        Test one function against all test cases in a single subprocess.

        The function code (and any heavy model loading it triggers) executes
        once; each test case is then evaluated in sequence inside that same
        process.  This avoids reloading NLP models for every (func, case) pair.

        If *log_file* is given, function code, subprocess stderr and per-case
        results are appended to the file for post-mortem debugging.

        Returns a list of (passed, error_reason) tuples, one per test case,
        in the same order as *test_cases*.
        """
        num = len(test_cases)
        # Pre-validate test cases without spawning a subprocess
        valid_indices: List[int] = []
        out: List[Optional[Tuple[bool, Optional[str]]]] = [None] * num

        for i, tc in enumerate(test_cases):
            if not isinstance(tc, dict):
                out[i] = (False, REASON_INVALID_TEST_CASE_FORMAT)
            elif 'input' not in tc or 'output' not in tc:
                out[i] = (False, REASON_MISSING_TEST_CASE_FIELDS)
            else:
                valid_indices.append(i)

        if not valid_indices:
            if log_file:
                _append_to_log(log_file,
                    f"\n--- Function {func_idx} ---\n"
                    f"SKIPPED: no valid test cases\n")
            return out  # type: ignore[return-value]

        # Write function header to log before running
        if log_file:
            code_lines = func_str.strip().splitlines()
            code_preview = "\n".join(code_lines[:30])
            if len(code_lines) > 30:
                code_preview += f"\n... ({len(code_lines) - 30} more lines)"
            _append_to_log(log_file,
                f"\n--- Function {func_idx} ({len(code_lines)} lines) ---\n"
                f"{code_preview}\n")

        # Safety + import checks (once per function, not per case)
        is_safe, reason = self.is_safe_function(func_str)
        if not is_safe:
            for i in valid_indices:
                out[i] = (False, reason)
            if log_file:
                _append_to_log(log_file, f"REJECTED: unsafe ({reason})\n")
            return out  # type: ignore[return-value]

        imports_ok, _installed = ensure_imports_available(func_str)
        if not imports_ok:
            err = f"{REASON_FUNCTION_EXECUTION_ERROR}: missing_dependencies"
            for i in valid_indices:
                out[i] = (False, err)
            if log_file:
                _append_to_log(log_file, f"REJECTED: missing dependencies\n")
            return out  # type: ignore[return-value]

        test_inputs = [test_cases[i]['input'] for i in valid_indices]
        if _worker_pool is not None:
            batch = _run_in_worker_batch(
                func_str, test_inputs,
                init_timeout=INIT_TIMEOUT, case_timeout=CASE_TIMEOUT,
                log_file=log_file,
            )
        else:
            batch = _run_in_subprocess_batch(
                func_str, test_inputs,
                init_timeout=INIT_TIMEOUT, case_timeout=CASE_TIMEOUT,
                log_file=log_file,
            )

        if "error" in batch:
            # Whole batch failed (model couldn't load, etc.)
            etype = batch.get("error_type", "execution_error")
            if etype == "no_evaluate_function":
                err = REASON_MISSING_EVALUATE_FUNCTION
            elif etype == "timeout":
                err = REASON_FUNCTION_TIMEOUT
            else:
                err = f"{REASON_FUNCTION_EXECUTION_ERROR}: {batch['error']}"
            for i in valid_indices:
                out[i] = (False, err)
            if log_file:
                _append_to_log(log_file,
                    f"--- results: BATCH FAILED ({etype}: {batch['error'][:200]}) ---\n")
            return out  # type: ignore[return-value]

        # Map per-case subprocess results back to the original indices
        case_results = batch.get("results", [])
        for pos, ci in enumerate(valid_indices):
            if pos >= len(case_results):
                out[ci] = (False, f"{REASON_FUNCTION_EXECUTION_ERROR}: missing result")
                continue

            cr = case_results[pos]
            if "error" in cr:
                etype = cr.get("error_type", "execution_error")
                if etype == "timeout":
                    out[ci] = (False, REASON_FUNCTION_TIMEOUT)
                else:
                    out[ci] = (False, f"{REASON_FUNCTION_EXECUTION_ERROR}: {cr['error']}")
            else:
                tc = test_cases[ci]
                expected = (tc['output'] if isinstance(tc['output'], bool)
                            else tc['output'].lower() == 'true')
                out[ci] = (cr["result"] == expected, None)

        # Write per-case results to log
        if log_file:
            passed_count = sum(1 for r in out if r is not None and r[0])
            error_count = sum(1 for r in out if r is not None and not r[0] and r[1])
            lines = [f"--- results: {passed_count}/{num} passed, {error_count} errors ---\n"]
            for i, r in enumerate(out):
                if r is None:
                    lines.append(f"  Case {i:3d}: SKIP\n")
                elif r[0]:
                    lines.append(f"  Case {i:3d}: PASS\n")
                else:
                    reason_str = r[1][:120] if r[1] else "unknown"
                    lines.append(f"  Case {i:3d}: FAIL ({reason_str})\n")
            _append_to_log(log_file, "".join(lines))

        return out  # type: ignore[return-value]

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

        # Ensure all imports are available (auto-install if needed)
        imports_ok, installed = ensure_imports_available(
            func_str,
            logger=self.logger if log_errors else None,
            instruction_id=instruction_id
        )
        if not imports_ok:
            if log_errors and self.logger:
                self.logger.log_error(instruction_id, f"{id_str} Failed to install required dependencies")
            return False, f"{REASON_FUNCTION_EXECUTION_ERROR}: missing_dependencies"

        # Execute function with timeout
        return self._execute_function_with_timeout(func_str, test_case, id_str, instruction_id, log_errors)

    def _execute_function_with_timeout(self, func_str: str, test_case: Dict[str, Any],
                                     id_str: str, instruction_id: str, log_errors: bool = True) -> Tuple[bool, Optional[str]]:
        """Execute function with timeout protection (worker pool or subprocess)."""
        if _worker_pool is not None:
            result = _run_in_worker(
                func_str, mode="test",
                init_timeout=INIT_TIMEOUT, case_timeout=CASE_TIMEOUT,
                test_input=test_case['input'],
            )
        else:
            result = _run_in_subprocess(
                func_str, mode="test",
                init_timeout=INIT_TIMEOUT, case_timeout=CASE_TIMEOUT,
                test_input=test_case['input'],
            )

        if "error" in result:
            error_type = result.get("error_type", "execution_error")

            if error_type == "no_evaluate_function":
                if log_errors:
                    self.logger.log_error(instruction_id, f"{id_str} No evaluate function found")
                return False, REASON_MISSING_EVALUATE_FUNCTION

            if error_type == "timeout":
                if log_errors:
                    self.logger.log_error(instruction_id, f"{id_str} Function timed out after {FUNCTION_TIMEOUT}s")
                return False, REASON_FUNCTION_TIMEOUT

            if log_errors:
                self.logger.log_error(instruction_id, f"{id_str} Execution error: {result['error']}")
            return False, f"{REASON_FUNCTION_EXECUTION_ERROR}: {result['error']}"

        # Normalize expected output
        expected = (test_case['output'] if isinstance(test_case['output'], bool)
                  else test_case['output'].lower() == 'true')

        return result["result"] == expected, None
