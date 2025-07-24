"""
Simplified logging utilities for AutoIF cross-validation process.

This module provides minimal logging functionality following the new requirements:
- ERROR {'instruction_id': <id>, 'error': <error_text>} for parsing errors
- CROSS_VALIDATION_ERROR {'instruction_id': <id>, ...} for cross-validation errors
- FILTERED {'instruction_id': <id>, 'reason': <reason>} for filtered verifiers
- Final summary with total/filtered/passed counts only
"""

import logging
import os
import json
from typing import Optional


class CrossValidationLogger:
    """Simplified logger for cross-validation process."""
    
    def __init__(self, log_file: str):
        """Initialize the logger with a specific log file."""
        self.log_file = log_file
        
        # Simple counters for final summary
        self.total_verifiers = 0
        self.filtered_verifiers = 0
        self.passed_verifiers = 0
        
        # Set up the logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self._setup_file_handler()
    
    def _setup_file_handler(self):
        """Set up file handler for logging."""
        # Clear existing handlers
        self.logger.handlers = []
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # Clear the log file
        with open(self.log_file, 'w') as f:
            pass
        
        # Configure file logging - simpler format without timestamp/level for cleaner output
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(file_handler)
    
    def log_error(self, instruction_id: str, error_message: str):
        """
        Log a parsing/execution error with exact error message.
        
        Args:
            instruction_id: The instruction ID where the error occurred
            error_message: The exact error message encountered
        """
        error_data = {
            'instruction_id': instruction_id,
            'error': error_message
        }
        self.logger.info(f"ERROR {json.dumps(error_data)}")
    
    def log_cross_validation_error(self, instruction_id: str, num_functions_tested: int, 
                                 num_test_cases_verified: int, num_failing_tests: int):
        """
        Log a cross-validation error with summary statistics.
        
        Args:
            instruction_id: The instruction ID being cross-validated
            num_functions_tested: Number of functions tested
            num_test_cases_verified: Number of test cases verified
            num_failing_tests: Number of failing tests
        """
        error_data = {
            'instruction_id': instruction_id,
            'num_of_functions_tested': num_functions_tested,
            'num_of_test_cases_verified': num_test_cases_verified,
            'num_of_failing_tests': num_failing_tests
        }
        self.logger.info(f"CROSS_VALIDATION_ERROR {json.dumps(error_data)}")
        
    def log_filtered(self, instruction_id: str, reason: str):
        """
        Log that a verifier was filtered out.
        
        Args:
            instruction_id: The instruction ID that was filtered
            reason: The reason why it was filtered
        """
        filtered_data = {
            'instruction_id': instruction_id,
            'reason': reason
        }
        self.logger.info(f"FILTERED {json.dumps(filtered_data)}")
        self.filtered_verifiers += 1
    
    def increment_total_verifiers(self):
        """Increment the total verifier count."""
        self.total_verifiers += 1
    
    def increment_passed_verifiers(self):
        """Increment the passed verifier count."""
        self.passed_verifiers += 1
    
    def log_final_summary(self):
        """Log the final summary with total/filtered/passed counts only."""
        summary = {
            'total_verifiers': self.total_verifiers,
            'filtered_verifiers': self.filtered_verifiers,
            'passed_verifiers': self.passed_verifiers
        }
        self.logger.info(f"FINAL_SUMMARY {json.dumps(summary)}")
    
    def log_info(self, message: str):
        """Log an info message."""
        self.logger.info(message)
