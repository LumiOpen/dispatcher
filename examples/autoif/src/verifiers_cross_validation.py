"""
AutoIF Cross-Validation Script

This script performs cross-validation of verification functions and test cases.
It validates that functions can correctly evaluate test cases with high accuracy.

Cross-validation process:
1. Parse functions and test cases from LLM responses
2. Validate function safety (no harmful code patterns)
3. Deduplicate test cases
4. Filter responses with less than MIN_FUNCTIONS parsable functions and less than MIN_TEST_CASES parsable test cases
5. Filter out test cases that don't pass any function
6. Keep only functions that meet FUNCTION_PASS_RATE (how many test cases they pass)
7. Output results for further processing

Author: AutoIF Team
"""

import os
import argparse
import json
import re
from typing import List, Dict, Any, Optional

from src.utils.constants import (
    REASON_NO_VALID_FUNCTIONS, REASON_NO_VALID_TEST_CASES,
    REASON_INSUFFICIENT_FUNCTIONS, REASON_INSUFFICIENT_TEST_CASES,
    REASON_NO_PASSING_TEST_CASES, REASON_NO_FUNCTIONS_MEET_ACCURACY,
    REASON_PARSE_FAILURE
)
from src.utils.logging_utils import CrossValidationLogger
from src.utils.response_parser import ResponseParser
from src.utils.function_executor import FunctionExecutor
from src.utils.lang_id import detect_language

MIN_FUNCTIONS = int(os.getenv('MIN_FUNCTIONS', 1))  # Minimum number of parsable functions required
MIN_TEST_CASES = int(os.getenv('MIN_TEST_CASES', 1))  # Minimum number of parsable test cases required
FUNCTION_PASS_RATE = float(os.getenv('FUNCTION_PASS_RATE', 0.8))  # Proportion of generated test cases a function must pass
LANGUAGE = os.getenv('LANGUAGE', 'eng')  # Language code for filtering test cases
OUT_DIR = os.getenv('OUT_DIR', 'exp1') # sub-directory for the output files in data/

class CrossValidationResult:
    """Container for cross-validation results of a single instruction."""
    
    def __init__(self, instruction_id: str, instruction: str, instruction_category: Optional[str] = None):
        self.instruction_id = instruction_id
        self.instruction = instruction
        self.instruction_category = instruction_category
        self.filtered = True
        self.reason = None
        # Raw data (all parsed functions and cases before any filtering)
        self.raw_eval_func = []
        self.raw_cases = []
        # Filtered data (functions and cases after cross-validation)
        self.eval_func = []
        self.cases = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization with raw data."""
        return {
            'instruction_id': self.instruction_id,
            'instruction': self.instruction,
            'instruction_category': self.instruction_category,
            'filtered': self.filtered,
            'reason': self.reason,
            'eval_func': self.raw_eval_func,  # Use raw data for "all" output
            'cases': self.raw_cases
        }
    
    def to_filtered_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for filtered output with only validated data."""
        return {
            'instruction_id': self.instruction_id,
            'instruction': self.instruction,
            'instruction_category': self.instruction_category,
            'eval_func': self.eval_func,
            'cases': self.cases
        }
    
    def set_raw_data(self, functions: List[str], test_cases: List[Dict[str, Any]]):
        """Set the raw parsed functions and test cases."""
        self.raw_eval_func = functions
        self.raw_cases = test_cases
    
    def mark_passed(self, functions: List[str], test_cases: List[Dict[str, Any]]):
        """Mark the result as passed with valid functions and test cases."""
        self.filtered = False
        self.reason = None
        self.eval_func = functions
        self.cases = test_cases


class InstructionProcessor:
    """Processes a single instruction through the cross-validation pipeline."""
    
    def __init__(self, logger: CrossValidationLogger):
        self.logger = logger
        self.parser = ResponseParser(logger)
        self.executor = FunctionExecutor(logger)
    
    def process_instruction(self, data: Dict[str, Any]) -> CrossValidationResult:
        """
        Process a single instruction through the complete cross-validation pipeline.
        
        Steps:
        1. Extract instruction data
        2. Parse all responses to get functions and test cases
        3. Check minimum requirements and store raw parsed data
        4. Validate function safety
        5. Filter test cases by language
        6. Deduplicate test cases
        7. Cross-validate functions against test cases
        8. Filter by accuracy threshold
        
        Note: Raw parsed data (before filtering) is always preserved in the result
        for the "all" output file, while filtered data is used for validation.
        
        Args:
            data: Instruction data with responses
            
        Returns:
            CrossValidationResult with validation outcome and both raw and filtered data
        """
        try:
            # Step 1: Extract instruction information
            original = data['original']
            instruction = original['instruction']
            instruction_id = original['instruction_id']
            instruction_category = original['instruction_category']
            
            result = CrossValidationResult(instruction_id, instruction, instruction_category)
            
            # Step 2: Parse responses to extract functions and test cases
            # In this step parsing errors can occur, which will be logged
            functions, cases = self._parse_responses(data.get('responses', []), instruction_id)
            
            # Step 3: Check minimum requirements (also stores raw data)
            # empty (or too few) functions or test cases will result in an early exit. "Too few" is controlled by MIN_FUNCTIONS and MIN_TEST_CASES.
            validation_result = self._validate_minimum_requirements(
                functions, cases, result
            )
            if validation_result:
                # Log the filtering with specific reason
                self.logger.log_filtered(instruction_id, validation_result.reason)
                return validation_result
            
            # Step 4: Check function safety
            safe_functions = self._filter_safe_fn(functions, instruction_id)
            if not safe_functions:
                result.reason = "no_safe_functions"
                self.logger.log_filtered(instruction_id, "no_safe_functions")
                return result

            # We are leaving out language identification for test cases - we do not need our test cases to be in the target language
            
            # Step 5: Deduplicate test cases
            unique_cases = self._deduplicate_test_cases(cases)
            if not unique_cases:
                result.reason = "no_test_cases_after_deduplication"
                self.logger.log_filtered(instruction_id, "no_test_cases_after_deduplication")
                return result
            
            # Step 6: Cross-validate functions against test cases
            final_functions, final_cases = self._cross_validate_functions_and_cases(
                safe_functions, unique_cases, instruction_id
            )
            
            # Step 7: Check if any functions meet accuracy threshold
            if not final_functions:
                result.reason = REASON_NO_FUNCTIONS_MEET_ACCURACY
                self.logger.log_filtered(instruction_id, REASON_NO_FUNCTIONS_MEET_ACCURACY)
                return result
            
            # Success: mark as passed
            result.mark_passed(final_functions, final_cases)
            return result
            
        except Exception as e:
            self.logger.log_error(instruction_id or "unknown", f"unexpected_error: {str(e)}")
            result = CrossValidationResult("unknown", "unknown")
            result.reason = f"unexpected_error: {str(e)}"
            # For unexpected errors, try to get instruction_id if available
            try:
                instruction_id = data['original']['instruction_id']
                self.logger.log_filtered(instruction_id, "unexpected_error")
            except:
                pass
            return result
    
    def _parse_responses(self, responses: List[str], instruction_id: str) -> tuple[List[str], List[Dict[str, Any]]]:
        """
        Parse all responses to extract functions and cases.
        Returns:
            Tuple of (functions, cases)
        """
        functions = []
        cases = []
        
        for response in responses:
            # Parse function and test cases from response
            parsed_data, parse_reason = self.parser.parse_function_and_cases(response, instruction_id)
            if parse_reason or not parsed_data:
                # Error already logged in parser
                continue
            if 'func' in parsed_data:
                functions.append(parsed_data['func'])
            if 'cases' in parsed_data:
                cases.extend(parsed_data['cases'])
            
        return functions, cases

    def _filter_safe_fn(self, functions: List[str], instruction_id: str) -> List[str]:
        """
        Check the safety of each function in the list.
        
        Args:
            functions: List of function strings to validate
            instruction_id: ID of the instruction for logging
        
        Returns:
            List of safe functions
        """
        safe_functions = []
        for fn_str in functions:
            # Validate function safety
            is_safe, safety_reason = self.executor.is_safe_function(fn_str)
            if not is_safe:
                self.logger.log_error(instruction_id, f"Unsafe function detected: {safety_reason}")
                continue
            safe_functions.append(fn_str)
        
        return safe_functions

    def _filter_cases_by_language(self, cases: List[Dict[str, Any]], instruction_id: str) -> List[Dict[str, Any]]:
        """
        Filter test cases by the specified language.
        
        Args:
            cases: List of test case dictionaries
            instruction_id: ID of the instruction for logging
        
        Returns:
            List of test cases that match the specified language
        """
        # Validate if the test cases are in the expected language
        valid_test_cases = []
        for case in cases:
            try:
                lang_code1, lang_code2 = detect_language(case['input'])
                # lang_cde1 is the three-letter code, lang_code2 is the two-letter code
                if lang_code1 == LANGUAGE or (lang_code2 is not None and lang_code2 == LANGUAGE):
                    valid_test_cases.append(case)
            except Exception as e:
                self.logger.log_error(instruction_id, f"Language detection error: {str(e)}")
        
        return valid_test_cases
        
    
    def _validate_minimum_requirements(self, functions: List[str], test_cases: List[Dict[str, Any]], 
                                     result: CrossValidationResult) -> Optional[CrossValidationResult]:
        """Check if minimum requirements for functions and test cases are met."""
        # Always store raw data, even if validation fails
        result.set_raw_data(functions, test_cases)
        
        if not functions:
            result.reason = REASON_NO_VALID_FUNCTIONS
            return result
        
        if not test_cases:
            result.reason = REASON_NO_VALID_TEST_CASES
            return result
        
        if len(functions) < MIN_FUNCTIONS:
            result.reason = f"{REASON_INSUFFICIENT_FUNCTIONS}_{len(functions)}_of_{MIN_FUNCTIONS}"
            return result
        
        if len(test_cases) < MIN_TEST_CASES:
            result.reason = f"{REASON_INSUFFICIENT_TEST_CASES}_{len(test_cases)}_of_{MIN_TEST_CASES}"
            return result
        
        return None
    
    def _cross_validate_functions_and_cases(self, functions: List[str], test_cases: List[Dict[str, Any]], instruction_id: str) -> tuple[List[str], List[Dict[str, Any]]]:
        """
        Cross-validate functions against test cases in a single pass.
        
        Performs both filtering operations simultaneously:
        1. Filter test cases that pass at least one function
        2. Filter functions that meet accuracy threshold
        
        Returns:
            Tuple of (final_functions, final_test_cases)
        """
        # Track which test cases pass at least one function
        case_passes = [False] * len(test_cases)
        # Track functions that meet accuracy threshold
        accurate_functions = []
        
        # Single pass through functions
        for func_idx, function in enumerate(functions):
            correct_count = 0
            
            # Test this function against all test cases
            for case_idx, test_case in enumerate(test_cases):
                passed, _ = self.executor.test_function(function, test_case, func_idx, case_idx, log_errors=False)
                if passed:
                    case_passes[case_idx] = True
                    correct_count += 1
            
            # Check if this function meets accuracy threshold
            accuracy = correct_count / len(test_cases) if test_cases else 0
            if accuracy >= FUNCTION_PASS_RATE:
                accurate_functions.append(function)
        
        # Filter test cases that passed at least one function
        passing_test_cases = [
            test_case for case_idx, test_case in enumerate(test_cases)
            if case_passes[case_idx]
        ]
        
        if not passing_test_cases:
            num_functions_tested = len(functions)
            num_test_cases_verified = len(test_cases)
            num_failing_tests = len(test_cases)  # All tests failed
            self.logger.log_cross_validation_error(
                instruction_id, num_functions_tested, num_test_cases_verified, num_failing_tests
            )
            return [], []
        
        return accurate_functions, passing_test_cases

    def _deduplicate_test_cases(self, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate test cases using JSON serialization.
        
        Args:
            cases: List of test case dictionaries
            
        Returns:
            List of unique test cases
        """
        seen = set()
        result = []
        for case in cases:
            case_json = json.dumps(case, sort_keys=True)
            if case_json not in seen:
                seen.add(case_json)
                result.append(case)
        return result


def load_verifier_data(verifiers_file: str, logger: CrossValidationLogger) -> List[Dict[str, Any]]:
    """Load verifier data from input file."""
    verifier_data = []
    line_number = 0
    try:
        with open(verifiers_file, 'r') as f:
            for line in f:
                line_number += 1
                try:
                    data = json.loads(line.strip())
                    verifier_data.append(data)
                except json.JSONDecodeError as e:
                    # Try to extract instruction_id for logging if possible
                    instruction_id = f"line_{line_number}"
                    try:
                        # Attempt partial parsing to get instruction_id
                        if '"instruction_id"' in line:
                            match = re.search(r'"instruction_id"\s*:\s*"([^"]+)"', line)
                            if match:
                                instruction_id = match.group(1)
                    except:
                        pass
                    
                    logger.log_error(instruction_id, f"Failed to parse input line {line_number}: {str(e)}")
    except FileNotFoundError:
        logger.log_error("unknown", f"Verifiers file {verifiers_file} not found")
        raise
    
    return verifier_data


def write_results(all_results: List[CrossValidationResult], filtered_results: List[CrossValidationResult],
                 output_all_file: str, output_filtered_file: str, logger: CrossValidationLogger):
    """Write results to output files."""
    # Write all results with raw parsed data (no filtering applied to eval_func and cases)
    with open(output_all_file, 'w') as f:
        for result in all_results:
            f.write(json.dumps(result.to_dict()) + '\n')
    
    # Write only passed verifiers with filtered/validated data
    with open(output_filtered_file, 'w') as f:
        for result in filtered_results:
            f.write(json.dumps(result.to_filtered_dict()) + '\n')


def cross_validate_verifiers(verifiers_file: str, output_all_file: str, output_filtered_file: str):
    """
    Main cross-validation function.
    
    This function orchestrates the complete cross-validation process:
    1. Load verifier data from input file
    2. Process each instruction through the validation pipeline
    3. Collect results and statistics
    4. Write results to output files
    5. Log comprehensive summary
    """
    # Set up logging
    logfile = f"{OUT_DIR}/{os.path.basename(output_all_file)}.log"
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    logger = CrossValidationLogger(logfile)
    
    # Load input data
    verifier_data = load_verifier_data(verifiers_file, logger)
    
    # Initialize processor
    processor = InstructionProcessor(logger)
    
    # Process each instruction
    all_results = []
    passed_results = []
    
    for data in verifier_data:
        logger.increment_total_verifiers()
        result = processor.process_instruction(data)
        all_results.append(result)
        
        if not result.filtered:
            logger.increment_passed_verifiers()
            passed_results.append(result)
    
    # Write results
    write_results(all_results, passed_results, output_all_file, output_filtered_file, logger)
    
    # Log final summary only
    logger.log_final_summary()
    
    total_count = len(all_results)
    filtered_count = sum(1 for r in all_results if r.filtered)
    passed_count = len(passed_results)
    
    print(f"Total verifiers: {total_count}")
    print(f"Filtered verifiers: {filtered_count} ({filtered_count/total_count*100:.1f}%)")
    print(f"Passed verifiers: {passed_count} ({passed_count/total_count*100:.1f}%)")
    print(f"Full logfile available at: {logfile}")


def main():
    """Main entry point for the cross-validation script."""
    parser = argparse.ArgumentParser(description='Cross-validate verifiers and concatenate with queries')

    parser.add_argument('--verifiers-file', type=str, required=True,
                        help='Input file with verification functions')
    parser.add_argument('--output-all-file', type=str, required=True,
                        help='Output file for all verifiers with filter status')
    parser.add_argument('--output-filtered-file', type=str, required=True,
                        help='Output file for filtered verifiers')

    args = parser.parse_args()

    # Run cross-validation
    cross_validate_verifiers(
        args.verifiers_file,
        args.output_all_file,
        args.output_filtered_file,
    )

if __name__ == "__main__":
    main()
