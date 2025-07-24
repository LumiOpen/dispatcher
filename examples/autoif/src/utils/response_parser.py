import json
import ast
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
    
    def parse_function_and_cases(self, response: str, instruction_id: str = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Parse function and test cases from response string.
        
        Tries multiple parsing strategies:
        1. Separate Python and JSON code blocks
        2. Complete JSON object
        3. JSON from markdown block  
        4. Regex extraction
        
        Args:
            response: Raw LLM response string
            instruction_id: Optional instruction ID for logging filtered instructions
            
        Returns:
            Tuple of (parsed_data, error_reason)
                parsed_data: Dictionary with 'func' and 'cases' keys if successful
                    'func' is the function code string
                    'cases' is a list of test case dictionaries with keys 'input' and 'output'
        """
        function_errors = []
        cases_errors = []
        
        # Strategy 1: Parse separate Python and JSON blocks
        parsed_data = self._try_separate_blocks(response, function_errors, cases_errors)
        if parsed_data:
            return parsed_data, None
        
        # Strategy 2: Try as complete JSON object
        parsed_data = self._try_complete_json(response, function_errors, cases_errors)
        if parsed_data:
            return parsed_data, None
        
        # Strategy 3: Parse JSON from markdown block
        parsed_data = self._try_markdown_json(response, function_errors, cases_errors)
        if parsed_data:
            return parsed_data, None
        
        # Strategy 4: Parse JSON with escaped content from markdown block  
        parsed_data = self._try_escaped_json_block(response, function_errors, cases_errors)
        if parsed_data:
            return parsed_data, None
        
        # Strategy 5: Try regex extraction
        parsed_data = self._try_regex_extraction(response, function_errors, cases_errors)
        if parsed_data:
            return parsed_data, None
        
        # All parsing methods failed - determine error type
        return self._handle_parsing_failure(function_errors, cases_errors, instruction_id)
    
    def _try_separate_blocks(self, response: str, function_errors: list, cases_errors: list) -> Optional[Dict[str, Any]]:
        """
        Try parsing separate Python and JSON code blocks. 
        This format is expected.
        """
        # Use more robust regex patterns that handle triple backticks within code
        func_match = self._extract_code_block(response, 'python')
        json_match = self._extract_code_block(response, 'json')
        
        if func_match and json_match:
            try:
                func_str = func_match.strip()
                cases_json = json_match.strip()
                cases_data = json.loads(cases_json)
                
                if 'cases' not in cases_data:
                    cases_errors.append("Missing 'cases' key in JSON block")
                    return None
                
                return {"func": func_str, "cases": cases_data['cases']}
            except (IndexError, json.JSONDecodeError) as e:
                cases_errors.append(f"Failed to parse JSON code block: {str(e)}")
        elif func_match and not json_match:
            function_errors.append("Found Python block but missing JSON block")
        elif not func_match and json_match:
            function_errors.append("Missing Python block")
        
        return None
    
    def _extract_code_block(self, response: str, language: str) -> Optional[str]:
        """
        Extract code block content, handling cases where the code itself contains triple backticks.
        
        Args:
            response: The full response text
            language: The language identifier (e.g., 'python', 'json')
            
        Returns:
            The extracted code content or None if not found
        """
        # Find the start of the code block
        start_pattern = f'```{language}'
        start_idx = response.find(start_pattern)
        if start_idx == -1:
            return None
        
        # Move past the opening marker and any newlines
        content_start = start_idx + len(start_pattern)
        if content_start < len(response) and response[content_start] == '\n':
            content_start += 1
        
        # For the end marker, we need to be more careful about finding the closing ```
        # We look for ``` that appears at the start of a line (possibly with whitespace before it)
        search_start = content_start
        while True:
            end_idx = response.find('```', search_start)
            if end_idx == -1:
                return None
            
            # Check if this ``` is at the start of a line (preceded only by whitespace)
            line_start = response.rfind('\n', content_start, end_idx)
            if line_start == -1:
                # No newline found before this ```, check if it's at the very beginning
                prefix = response[content_start:end_idx]
            else:
                # Check the content from the start of the line to the ```
                prefix = response[line_start + 1:end_idx]
            
            # If the prefix contains only whitespace, this is likely a closing marker
            if prefix.strip() == '':
                # Additional check: make sure what follows ``` is either end of string,
                # newline, or whitespace (not part of a longer sequence)
                after_backticks = end_idx + 3
                if (after_backticks >= len(response) or 
                    response[after_backticks] in '\n\r \t' or
                    response[after_backticks:after_backticks+1] == ''):
                    # This looks like a valid end marker
                    return response[content_start:end_idx]
            
            # Not a valid end marker, continue searching
            search_start = end_idx + 1
    
    def _try_complete_json(self, response: str, function_errors: list, cases_errors: list) -> Optional[Dict[str, Any]]:
        """Try parsing as complete JSON object."""
        try:
            data = json.loads(response)
            
            if 'func' in data and 'cases' in data:
                return data
            elif 'func' not in data:
                function_errors.append("Missing 'func' key in JSON")
            elif 'cases' not in data:
                cases_errors.append("Missing 'cases' key in JSON")
        except json.JSONDecodeError as e:
            function_errors.append(f"Failed to parse response as JSON: {str(e)}")
        
        return None
    
    def _try_markdown_json(self, response: str, function_errors: list, cases_errors: list) -> Optional[Dict[str, Any]]:
        """Try parsing JSON from markdown block."""
        try:
            json_content = self._extract_code_block(response, 'json')
            if json_content:
                data = json.loads(json_content)
                if 'func' in data and 'cases' in data:
                    return data
                elif 'func' not in data:
                    function_errors.append("Missing 'func' key in markdown JSON block")
                elif 'cases' not in data:
                    cases_errors.append("Missing 'cases' key in markdown JSON block")
        except json.JSONDecodeError as e:
            function_errors.append(f"Failed to parse markdown JSON: {str(e)}")
        
        return None
    
    def _try_escaped_json_block(self, response: str, function_errors: list, cases_errors: list) -> Optional[Dict[str, Any]]:
        """Try parsing JSON with escaped content from markdown block."""
        try:
            json_content = self._extract_code_block(response, 'json')
            if json_content:
                # Try to handle escaped JSON content
                # First, try to parse as-is
                try:
                    data = json.loads(json_content)
                    if 'func' in data and 'cases' in data:
                        return data
                    elif 'func' not in data:
                        function_errors.append("Missing 'func' key in escaped JSON block")
                    elif 'cases' not in data:
                        cases_errors.append("Missing 'cases' key in escaped JSON block")
                except json.JSONDecodeError:
                    # If that fails, try manual parsing for the specific pattern
                    return self._parse_escaped_json_manually(json_content, function_errors, cases_errors)
        except Exception as e:
            function_errors.append(f"Failed to parse escaped JSON block: {str(e)}")
        
        return None
    
    def _parse_escaped_json_manually(self, json_content: str, function_errors: list, cases_errors: list) -> Optional[Dict[str, Any]]:
        """Manually parse JSON with escaped strings that json.loads can't handle."""
        try:
            # Use regex to extract the func and cases more carefully
            # Look for "func": "..." pattern, handling escaped quotes and newlines
            func_pattern = r'"func"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,'
            func_match = re.search(func_pattern, json_content, re.DOTALL)
            
            if not func_match:
                function_errors.append("Could not find func field in escaped JSON")
                return None
            
            # Extract and unescape the function string
            func_raw = func_match.group(1)
            func_str = self._unescape_json_string(func_raw)
            
            # Look for cases array
            cases_pattern = r'"cases"\s*:\s*(\[.*?\])\s*(?:,|\})'
            cases_match = re.search(cases_pattern, json_content, re.DOTALL)
            
            if not cases_match:
                cases_errors.append("Could not find cases field in escaped JSON")
                return None
            
            # Parse the cases array
            cases_str = cases_match.group(1)
            cases_data = json.loads(cases_str)
            
            return {"func": func_str, "cases": cases_data}
            
        except Exception as e:
            function_errors.append(f"Failed manual parsing of escaped JSON: {str(e)}")
            return None
    
    def _unescape_json_string(self, escaped_str: str) -> str:
        """Unescape a JSON string value."""
        # Handle common JSON escape sequences
        result = escaped_str.replace('\\n', '\n')
        result = result.replace('\\"', '"')
        result = result.replace('\\\\', '\\')
        result = result.replace('\\t', '\t')
        result = result.replace('\\r', '\r')
        return result
    
    def _try_regex_extraction(self, response: str, function_errors: list, cases_errors: list) -> Optional[Dict[str, Any]]:
        """Try regex extraction as fallback."""
        func_match = re.search(r'"func"\s*:\s*"(.*?)"(?=\s*,|\s*})', response, re.DOTALL)
        cases_match = re.search(r'"cases"\s*:\s*(\[.*?\])(?=\s*,|\s*})', response, re.DOTALL)
        
        if func_match and cases_match:
            try:
                func = func_match.group(1)
                func = func.replace('\\n', '\n').replace('\\"', '"')
                
                cases_str = cases_match.group(1)
                cases = ast.literal_eval(cases_str)
                return {"func": func, "cases": cases}
            except Exception as e:
                function_errors.append(f"Failed to extract function with regex: {str(e)}")
                cases_errors.append(f"Failed to evaluate cases with ast: {str(e)}")
        elif func_match and not cases_match:
            function_errors.append("Found function but couldn't extract cases with regex")
        elif not func_match and cases_match:
            function_errors.append("Found cases but couldn't extract function with regex")
        
        return None
    
    def _handle_parsing_failure(self, function_errors: list, cases_errors: list, instruction_id: str = None) -> Tuple[None, str]:
        """Handle parsing failure by logging appropriate error."""
        if function_errors:
            combined_error = " | ".join(function_errors)
            if self.logger is None:
                print(f"Function parse failure: {combined_error}")
            else:
                self.logger.log_error(instruction_id or "unknown", f"Function parse failure: {combined_error}")
            return None, REASON_FUNCTION_PARSE_FAILURE
        
        if cases_errors:
            combined_error = " | ".join(cases_errors)
            if self.logger is None:
                print(f"Cases parse failure: {combined_error}")
            else:
                self.logger.log_error(instruction_id or "unknown", f"Cases parse failure: {combined_error}")
            return None, REASON_CASES_PARSE_ERROR
        
        # Fallback to function parse failure
        if self.logger is None:
            print("Unknown parsing failure")
        else:
            self.logger.log_error(instruction_id or "unknown", "Unknown parsing failure")
        return None, REASON_FUNCTION_PARSE_FAILURE
