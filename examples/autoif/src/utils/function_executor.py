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
from dispatcher.taskmanager.task.base import TaskFailed

FUNCTION_TIMEOUT = int(os.getenv('FUNCTION_TIMEOUT', 10))  # seconds
INSTALL_TIMEOUT = int(os.getenv('INSTALL_TIMEOUT', 60))  # seconds for pip install
AUTO_INSTALL_PACKAGES = os.getenv('AUTO_INSTALL_PACKAGES', 'true').lower() == 'true'

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
    Check if a module is available for import.

    Args:
        module_name: The module name to check

    Returns:
        True if the module can be imported
    """
    if module_name in _installed_packages_cache:
        return True

    try:
        __import__(module_name)
        _installed_packages_cache.add(module_name)
        return True
    except ImportError:
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

        # Ensure all imports are available (auto-install if needed)
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
