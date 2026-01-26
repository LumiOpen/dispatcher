import json
import re
from typing import List, Dict, Any, Tuple, Optional

from .constants import (
    REASON_FUNCTION_PARSE_FAILURE,
    REASON_CASES_PARSE_ERROR,
)
from .logging_utils import CrossValidationLogger


class ResponseParser:
    """Handles parsing of function and test case data from LLM responses."""
    
    def __init__(self, logger: CrossValidationLogger = None):
        self.logger = logger

    def parse_function(self, response: str, instruction_id: str = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse the evaluation function from a response.

        Args:
            response: Raw LLM response string
            instruction_id: Optional instruction ID for logging

        Returns:
            Tuple of (function_string, error_reason)
        """
        # Handle None response (e.g., from API connection errors)
        if response is None:
            self._log_error(instruction_id, "Response is None (API error)")
            return None, REASON_FUNCTION_PARSE_FAILURE

        # Try to extract Python code block
        func_str = self._extract_code_block(response, 'python')
        
        if func_str:
            func_str = func_str.strip()
            if 'def evaluate' in func_str:
                return func_str, None
            else:
                self._log_error(instruction_id, "Function missing 'def evaluate'")
                return None, REASON_FUNCTION_PARSE_FAILURE
        
        # No code block found - try to find function definition directly
        func_match = re.search(
            r'((?:import\s+\w+.*?\n)*\s*def\s+evaluate\s*\(.*?\n(?:.*?\n)*?(?=\n\n|\Z))', 
            response, re.DOTALL
        )
        if func_match:
            return func_match.group(1).strip(), None
        
        self._log_error(instruction_id, "Could not find Python function in response")
        return None, REASON_FUNCTION_PARSE_FAILURE

    def parse_test_cases(self, response: str, instruction_id: str = None) -> Tuple[Optional[Dict[str, List]], Optional[str]]:
        """
        Parse positive and negative test cases from a response.

        Expected format:
        {
            "positive": [{"response": "...", ...kwargs}, ...],
            "negative": [{"response": "...", ...kwargs}, ...]
        }

        Args:
            response: Raw LLM response string
            instruction_id: Optional instruction ID for logging

        Returns:
            Tuple of ({"positive": [...], "negative": [...]}, error_reason)
        """
        # Handle None response (e.g., from API connection errors)
        if response is None:
            self._log_error(instruction_id, "Response is None (API error)")
            return None, REASON_CASES_PARSE_ERROR

        # Try JSON code block first
        json_content = self._extract_code_block(response, 'json')
        if json_content:
            try:
                data = json.loads(json_content)
                result, error = self._validate_test_cases_structure(data)
                if result:
                    return result, None
            except json.JSONDecodeError:
                pass
        
        # Try parsing the entire response as JSON
        try:
            data = json.loads(response)
            result, error = self._validate_test_cases_structure(data)
            if result:
                return result, None
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON object in the response
        json_match = re.search(r'\{[\s\S]*"positive"[\s\S]*"negative"[\s\S]*\}', response)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                result, error = self._validate_test_cases_structure(data)
                if result:
                    return result, None
            except json.JSONDecodeError:
                pass
        
        self._log_error(instruction_id, "Could not parse test cases JSON")
        return None, REASON_CASES_PARSE_ERROR

    def _validate_test_cases_structure(self, data: Any) -> Tuple[Optional[Dict], Optional[str]]:
        """Validate that parsed data has the expected test cases structure."""
        if not isinstance(data, dict):
            return None, "Not a dictionary"
        
        if 'positive' not in data or 'negative' not in data:
            return None, "Missing positive/negative keys"
        
        if not isinstance(data['positive'], list) or not isinstance(data['negative'], list):
            return None, "positive/negative not lists"
        
        # Validate each test case has 'response' field
        for cases, label in [(data['positive'], 'positive'), (data['negative'], 'negative')]:
            for i, case in enumerate(cases):
                if not isinstance(case, dict) or 'response' not in case:
                    return None, f"{label}[{i}] missing 'response'"
        
        return data, None

    def _extract_code_block(self, response: str, language: str) -> Optional[str]:
        """Extract code block content for a given language."""
        start_pattern = f'```{language}'
        start_idx = response.find(start_pattern)
        if start_idx == -1:
            return None
        
        content_start = start_idx + len(start_pattern)
        if content_start < len(response) and response[content_start] == '\n':
            content_start += 1
        
        # Find closing ```
        search_start = content_start
        while True:
            end_idx = response.find('```', search_start)
            if end_idx == -1:
                return None
            
            # Check if this is a valid closing marker
            line_start = response.rfind('\n', content_start, end_idx)
            if line_start == -1:
                prefix = response[content_start:end_idx]
            else:
                prefix = response[line_start + 1:end_idx]
            
            if prefix.strip() == '':
                after_backticks = end_idx + 3
                if (after_backticks >= len(response) or 
                    response[after_backticks] in '\n\r \t'):
                    return response[content_start:end_idx]
            
            search_start = end_idx + 1

    def _log_error(self, instruction_id: str, message: str):
        """Log an error if logger is available."""
        if self.logger:
            self.logger.log_error(instruction_id or "unknown", message)
