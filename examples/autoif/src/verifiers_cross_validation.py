import os
import argparse
import json
import ast
import re
import random
import signal
import logging
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---- Constants ----
# Thresholds
FUNCTION_TIMEOUT = 5  # seconds
MIN_FUNCTIONS = 1 # original autoif 3
MIN_TEST_CASES = 1 # original autoif 10
ACCURACY_THRESHOLD = 0.8

# ---- Filter reason codes ----
# Parser errors
REASON_FUNCTION_PARSE_FAILURE = "function_parse_failure"  # Failed to parse function
REASON_CASES_PARSE_ERROR = "cases_parse_error"  # Failed to parse test cases
REASON_PARSE_FAILURE = "parse_failure"  # Failed to parse response for other reason

# Function errors
REASON_HARMFUL_CODE_DETECTED = "harmful_code_detected"  # Function contains potentially harmful imports/code
REASON_FUNCTION_EXECUTION_ERROR = "function_execution_error"  # Error when executing function definition
REASON_MISSING_EVALUATE_FUNCTION = "missing_evaluate_function"  # Function doesn't contain evaluate() function
REASON_FUNCTION_TIMEOUT = "function_timeout"  # Function execution exceeded time limit

# Test case errors
REASON_NO_VALID_TEST_CASES = "no_valid_test_cases"  # No valid test cases found
REASON_INVALID_TEST_CASE_FORMAT = "invalid_test_case_format"  # Test case has invalid format
REASON_MISSING_TEST_CASE_FIELDS = "missing_test_case_fields"  # Test case missing input or output fields
REASON_NON_FINNISH_TEST_CASES = "non_finnish_test_cases"  # Test cases aren't in Finnish

# Cross-validation errors
REASON_NO_VALID_FUNCTIONS = "no_valid_functions"  # No valid functions found at all
REASON_INSUFFICIENT_FUNCTIONS = "insufficient_functions"  # Not enough valid functions
REASON_INSUFFICIENT_TEST_CASES = "insufficient_test_cases"  # Not enough valid test cases
REASON_NO_PASSING_TEST_CASES = "no_passing_test_cases"  # No test cases pass any function
REASON_LOW_FUNCTION_ACCURACY = "low_function_accuracy"  # Function accuracy below threshold
REASON_NO_FUNCTIONS_MEET_ACCURACY = "no_functions_meet_accuracy_threshold"  # No functions meet accuracy threshold

# Error tracking stats
error_stats = {
    "parsing_errors": {
        "function": Counter(),
        "test_case": Counter(),
    },
    "function_errors": Counter(),
    "test_case_errors": Counter(),
    "cross_validation_errors": Counter(),
    "details": {},  # Store detailed error messages by category
    "counts": {
        "total_functions_attempted": 0,
        "total_testcases_attempted": 0,
        "total_function_executions": 0,
        "total_testcase_validations": 0
    }
}

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    """Handler for timeout signal."""
    raise TimeoutError("Function execution timed out")

def log_error(category: str, reason: str, detail: str = None, subcategory: str = None):
    """Log an error to both the error stats counter and the detailed log."""
    if subcategory:
        if category not in error_stats or subcategory not in error_stats[category]:
            error_stats[category][subcategory] = Counter()
        error_stats[category][subcategory][reason] += 1
        log_prefix = f"{category}:{subcategory}"
    else:
        if category not in error_stats:
            error_stats[category] = Counter()
        error_stats[category][reason] += 1
        log_prefix = category
    
    if detail:
        if reason not in error_stats["details"]:
            error_stats["details"][reason] = []
        # Limit number of detailed messages stored to avoid memory issues
        if len(error_stats["details"][reason]) < 20:  # Store up to 20 examples per error type
            error_stats["details"][reason].append(detail)
    
    # Log to the error log file with clearer category/subcategory
    if detail:
        logger.error(f"{log_prefix} - {reason}: {detail}")
    else:
        logger.error(f"{log_prefix} - {reason}")

def parse_function_and_cases(response: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Parse function and test cases from response string. Returns (data, reason)."""
    # Track errors for function and cases separately
    function_errors = []
    cases_errors = []
    
    # ---------- Version 2: Parse separate Python and JSON blocks ----------
    # Extract function from Python code block
    func_match = re.findall(r'```python(.*?)```', response, re.DOTALL)
    # Extract test cases from JSON code block
    json_match = re.findall(r'```json(.*?)```', response, re.DOTALL)
    
    # If we have both blocks, we can process them separately
    if func_match and json_match:
        func_str = None
        cases_data = None
        
        # Parse function
        try:
            func_str = func_match[0].strip()
        except IndexError as e:
            function_errors.append(f"Failed to parse Python code block: {str(e)}")
        
        # Parse cases
        if func_str:  # Only try to parse cases if function parsing was successful
            try:
                cases_json = json_match[0].strip()
                cases_data = json.loads(cases_json)
                
                if 'cases' not in cases_data:
                    cases_errors.append("Missing 'cases' key in JSON block")
                else:
                    # Both function and cases parsed successfully
                    return {"func": func_str, "cases": cases_data['cases']}, None
            except (IndexError, json.JSONDecodeError) as e:
                cases_errors.append(f"Failed to parse JSON code block: {str(e)}")
    elif func_match and not json_match:
        function_errors.append("Found Python block but missing JSON block")
    elif not func_match and json_match:
        function_errors.append("Missing Python block")
    
    # ---------- Version 1: Try as a complete JSON object ----------
    try:
        data = json.loads(response)
        
        if 'func' in data and 'cases' in data:
            return data, None
        elif 'func' not in data:
            function_errors.append("Missing 'func' key in JSON")
        elif 'cases' not in data:
            cases_errors.append("Missing 'cases' key in JSON")
    except json.JSONDecodeError as e:
        function_errors.append(f"Failed to parse response as JSON: {str(e)}")
        
    # ---------- Version 1: Parse JSON from markdown block ----------
    try:
        json_dict = re.findall(r'```json(.*?)```', response, re.DOTALL)
        if json_dict:
            data = json.loads(json_dict[0].strip())
            if 'func' in data and 'cases' in data:
                return data, None
            elif 'func' not in data:
                function_errors.append("Missing 'func' key in markdown JSON block")
            elif 'cases' not in data:
                cases_errors.append("Missing 'cases' key in markdown JSON block")
    except (IndexError, json.JSONDecodeError) as e:
        function_errors.append(f"Failed to parse markdown JSON: {str(e)}")
        
    # ---------- Version 1: Try regex extraction ----------
    func_match = re.search(r'"func"\s*:\s*"(.*?)"(?=\s*,|\s*})', response, re.DOTALL)
    cases_match = re.search(r'"cases"\s*:\s*(\[.*?\])(?=\s*,|\s*})', response, re.DOTALL)
    
    if func_match and cases_match:
        try:
            func = func_match.group(1)
            func = func.replace('\\n', '\n').replace('\\"', '"')
            
            cases_str = cases_match.group(1)
            cases = ast.literal_eval(cases_str)
            return {"func": func, "cases": cases}, None
        except Exception as e:
            function_errors.append(f"Failed to extract function with regex: {str(e)}")
            cases_errors.append(f"Failed to evaluate cases with ast: {str(e)}")
    elif func_match and not cases_match:
        function_errors.append("Found function but couldn't extract cases with regex")
    elif not func_match and cases_match:
        function_errors.append("Found cases but couldn't extract function with regex")
    
    # If we get here, all parsing methods have failed
    # Log function errors first if we have any
    if function_errors:
        combined_error = " | ".join(function_errors)
        log_error("parsing_errors", REASON_FUNCTION_PARSE_FAILURE, f"Function parse failure: {combined_error}", "function")
        return None, REASON_FUNCTION_PARSE_FAILURE
    
    # Otherwise log cases errors if we have those
    if cases_errors:
        combined_error = " | ".join(cases_errors)
        log_error("parsing_errors", REASON_CASES_PARSE_ERROR, f"Cases parse failure: {combined_error}", "test_case")
        return None, REASON_CASES_PARSE_ERROR
    
    # If we somehow get here with no specific errors (unlikely), use function parse failure as default
    log_error("parsing_errors", REASON_FUNCTION_PARSE_FAILURE, "Unknown parsing failure", "function")
    return None, REASON_FUNCTION_PARSE_FAILURE

def is_safe_function(func_str: str) -> Tuple[bool, Optional[str]]:
    """Check if function contains potentially harmful imports or operations."""
    unsafe_patterns = ['import', 'download', 'requests', 'subprocess', 'os.', 'sys.']
    
    # Check for unsafe patterns
    for pattern in unsafe_patterns:
        if pattern in func_str.lower():
            log_error("function_errors", REASON_HARMFUL_CODE_DETECTED, f"Found unsafe pattern '{pattern}' in function")
            return False, REASON_HARMFUL_CODE_DETECTED
            
    return True, None

def test_function(func_str: str, test_case: Dict[str, Any], func_idx: int = -1, case_idx: int = -1) -> Tuple[bool, Optional[str]]:
    """Test if a function passes a test case. Returns (passed, reason)."""
    id_str = f"func{func_idx}_case{case_idx}" if func_idx >= 0 and case_idx >= 0 else ""
    
    error_stats["counts"]["total_testcase_validations"] += 1
    
    if not isinstance(test_case, dict):
        log_error("test_case_errors", REASON_INVALID_TEST_CASE_FORMAT, 
                 f"{id_str} Test case is not a dict: {type(test_case)}")
        return False, REASON_INVALID_TEST_CASE_FORMAT
        
    if 'input' not in test_case or 'output' not in test_case:
        log_error("test_case_errors", REASON_MISSING_TEST_CASE_FIELDS, 
                 f"{id_str} Missing input/output in test case")
        return False, REASON_MISSING_TEST_CASE_FIELDS
    
    # Check if function is safe
    is_safe, reason = is_safe_function(func_str)
    if not is_safe:
        return False, reason
        
    try:
        error_stats["counts"]["total_function_executions"] += 1
        
        # Create a namespace for the function
        namespace = {}
        
        # Execute the function definition
        exec(func_str, namespace)
        
        # Get the evaluate function from the namespace
        evaluate_func = namespace.get('evaluate')
        if evaluate_func is None:
            log_error("function_errors", REASON_MISSING_EVALUATE_FUNCTION, 
                     f"{id_str} No evaluate function found")
            return False, REASON_MISSING_EVALUATE_FUNCTION
        
        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(FUNCTION_TIMEOUT)
        
        try:
            # Run the test case
            result = evaluate_func(test_case['input'])
            # Normalize expected output
            expected = test_case['output'] if isinstance(test_case['output'], bool) else test_case['output'].lower() == 'true'
            
            return result == expected, None
        finally:
            signal.alarm(0)  # Disable the alarm
            
    except TimeoutError:
        log_error("function_errors", REASON_FUNCTION_TIMEOUT, 
                 f"{id_str} Function timed out after {FUNCTION_TIMEOUT}s")
        return False, REASON_FUNCTION_TIMEOUT
    except Exception as e:
        log_error("function_errors", REASON_FUNCTION_EXECUTION_ERROR, 
                 f"{id_str} Execution error: {str(e)}")
        return False, f"{REASON_FUNCTION_EXECUTION_ERROR}: {str(e)}"

def deduplicate_test_cases(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate test cases using JSON serialization."""
    seen = set()
    result = []
    for case in cases:
        case_json = json.dumps(case, sort_keys=True)
        if case_json not in seen:
            seen.add(case_json)
            result.append(case)
    return result

def cross_validate_verifiers(verifiers_file: str, all_output_file: str, filtered_output_file: str) -> List[Dict]:
    """Cross-validate verification functions and test cases."""
    # Initialize language identifier
    # glot_client = InferenceClient('cis-lmu/glotlid')
    
    # Load verifier data
    verifier_data = []
    try:
        with open(verifiers_file, 'r') as f:
            for line in f:
                try:
                    verifier_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    log_error("parsing_errors", REASON_PARSE_FAILURE, f"Failed to parse input line: {str(e)}", "function")
    except FileNotFoundError:
        logger.error(f"Error: Verifiers file {verifiers_file} not found")
        exit(1)
    
    # Process each verifier
    all_results = []
    filtered_verifiers = []
    
    total_count = len(verifier_data)
    filtered_count = 0
    
    logger.info(f"Processing {total_count} verifier entries")
    
    for data_idx, data in enumerate(verifier_data):
        logger.info(f"Processing item {data_idx}/{total_count}")
            
        instruction = None
        try:
            original = data['original']
            instruction = original['instruction']
            
            # Prepare result object with filtered=True by default
            result = {
                'instruction': instruction,
                'filtered': True,
                'reason': None,
                'eval_func': [],
                'cases': []
            }
            
            # Track reasons for this specific instruction
            instruction_reasons = {
                "parsing": [],
                "function": [],
                "test_case": [],
                "cross_validation": []
            }
            
            valid_funcs = []
            all_test_cases = []
            
            # Process each response for this instruction
            response_idx = 0
            for response in data.get('responses', []):
                # Count the attempted function parsing
                error_stats["counts"]["total_functions_attempted"] += 1
                
                parsed, parse_reason = parse_function_and_cases(response)
                
                # Record parsing failures
                if parse_reason:
                    instruction_reasons["parsing"].append(parse_reason)
                    continue
                    
                if not parsed:
                    instruction_reasons["parsing"].append(REASON_PARSE_FAILURE)
                    continue
                
                try:   
                    func_str = parsed['func']
                    test_cases = parsed['cases']
                
                    # Skip unsafe functions
                    is_safe, safety_reason = is_safe_function(func_str)
                    if not is_safe:
                        instruction_reasons["function"].append(safety_reason)
                        continue
                
                    # Count the attempted test cases
                    error_stats["counts"]["total_testcases_attempted"] += len(test_cases)
                    
                    # Check if test cases are in Finnish
                    valid_cases = []
                    for case_idx, case in enumerate(test_cases):
                        # if case.get('input') and is_finnish(case['input'], glot_client):
                        #     valid_cases.append(case)
                        # else:
                        #     instruction_reasons["test_case"].append(REASON_NON_FINNISH_TEST_CASES)
                        #     continue
                        valid_cases.append(case)
                    
                    if not valid_cases:
                        instruction_reasons["test_case"].append(REASON_NO_VALID_TEST_CASES)
                        continue
                        
                    # Add to collection
                    valid_funcs.append(func_str)
                    all_test_cases.extend(valid_cases)
                    
                    response_idx += 1
                    
                except (KeyError, TypeError) as e:
                    # Handle the case when the parsed data structure is unexpected
                    error_msg = f"invalid_data_structure: {str(e)}"
                    instruction_reasons["parsing"].append(error_msg)
                    log_error("parsing_errors", "invalid_data_structure", str(e), "function")
                    continue
            
            # Deduplicate test cases
            all_test_cases = deduplicate_test_cases(all_test_cases)
            
            # Cross validate functions against test cases
            if not valid_funcs:
                # Use collected reasons if available
                if instruction_reasons["function"]:
                    result['reason'] = instruction_reasons["function"][0]
                    log_error("cross_validation_errors", REASON_NO_VALID_FUNCTIONS, f"No valid functions due to function errors")
                elif instruction_reasons["parsing"]:
                    result['reason'] = instruction_reasons["parsing"][0]
                    log_error("cross_validation_errors", REASON_NO_VALID_FUNCTIONS, f"No valid functions due to parsing errors")
                else:
                    result['reason'] = REASON_NO_VALID_FUNCTIONS
                    log_error("cross_validation_errors", REASON_NO_VALID_FUNCTIONS, f"No valid functions found")
                filtered_count += 1
                all_results.append(result)
                continue
                
            if not all_test_cases:
                result['reason'] = REASON_NO_VALID_TEST_CASES
                log_error("cross_validation_errors", REASON_NO_VALID_TEST_CASES, f"No valid test cases")
                filtered_count += 1
                all_results.append(result)
                continue
                
            if len(valid_funcs) < MIN_FUNCTIONS:
                reason = f"{REASON_INSUFFICIENT_FUNCTIONS}_{len(valid_funcs)}_of_{MIN_FUNCTIONS}"
                result['reason'] = reason
                log_error("cross_validation_errors", REASON_INSUFFICIENT_FUNCTIONS, f"Only {len(valid_funcs)} of {MIN_FUNCTIONS} required functions")
                filtered_count += 1
                all_results.append(result)
                continue
                
            if len(all_test_cases) < MIN_TEST_CASES:
                reason = f"{REASON_INSUFFICIENT_TEST_CASES}_{len(all_test_cases)}_of_{MIN_TEST_CASES}"
                result['reason'] = reason
                log_error("cross_validation_errors", REASON_INSUFFICIENT_TEST_CASES, f"Only {len(all_test_cases)} of {MIN_TEST_CASES} required test cases")
                filtered_count += 1
                all_results.append(result)
                continue
                
            final_funcs = []
            final_cases = []
            
            # Keep only test cases that pass at least one function
            # Use function indices in logging
            for case_idx, case in enumerate(all_test_cases):
                case_passes = False
                for func_idx, func in enumerate(valid_funcs):
                    passed, _ = test_function(func, case, func_idx, case_idx)
                    if passed:
                        case_passes = True
                        break
                
                if case_passes:
                    final_cases.append(case)
            
            if not final_cases:
                result['reason'] = REASON_NO_PASSING_TEST_CASES
                log_error("cross_validation_errors", REASON_NO_PASSING_TEST_CASES, f"No test cases pass any function")
                filtered_count += 1
                all_results.append(result)
                continue
            
            # Keep only functions with accuracy > threshold
            valid_func_found = False
            for func_idx, func in enumerate(valid_funcs):
                if not final_cases:
                    continue
                    
                correct = 0
                for case_idx, case in enumerate(final_cases):
                    passed, _ = test_function(func, case, func_idx, case_idx)
                    if passed:
                        correct += 1
                        
                accuracy = correct / len(final_cases)
                logger.info(f"Function {func_idx}: {correct}/{len(final_cases)} = {accuracy:.2f} accuracy")
                
                if accuracy >= ACCURACY_THRESHOLD:
                    final_funcs.append(func)
                    valid_func_found = True
            
            if not valid_func_found:
                result['reason'] = REASON_NO_FUNCTIONS_MEET_ACCURACY
                log_error("cross_validation_errors", REASON_NO_FUNCTIONS_MEET_ACCURACY, f"No functions meet accuracy threshold")
                filtered_count += 1
                all_results.append(result)
                continue
            
            # This verifier passes all checks
            result['filtered'] = False
            result['reason'] = None
            result['eval_func'] = final_funcs
            result['cases'] = final_cases
            
            all_results.append(result)
            
            # Add to filtered list
            filtered_entry = {
                'instruction': instruction,
                'eval_func': final_funcs,
                'cases': final_cases
            }
            filtered_verifiers.append(filtered_entry)
        except Exception as e:
            # Catch any unexpected errors during processing
            error_msg = f"Unexpected error processing entry {data_idx}: {str(e)}"
            logger.error(error_msg)
            if instruction:
                result = {
                    'instruction': instruction,
                    'filtered': True,
                    'reason': f"unexpected_error: {str(e)}",
                    'eval_func': [],
                    'cases': []
                }
                all_results.append(result)
                filtered_count += 1
    
    # Write all results (including filtered ones with reasons)
    with open(all_output_file, 'w') as f:
        for result in all_results:
            f.write(json.dumps(result) + '\n')
    
    # Write filtered verifiers to file (only those that passed)
    with open(filtered_output_file, 'w') as f:
        for verifier in filtered_verifiers:
            f.write(json.dumps(verifier) + '\n')
    
    # Print summary statistics
    log_error_summary()
    
    logger.info(f"Total verifiers: {len(all_results)}")
    print(f"Total verifiers: {len(all_results)}")
    logger.info(f"Filtered verifiers: {filtered_count} ({filtered_count/len(all_results)*100:.1f}%)")
    print(f"Filtered verifiers: {filtered_count} ({filtered_count/len(all_results)*100:.1f}%)")
    logger.info(f"Passed verifiers: {len(filtered_verifiers)} ({len(filtered_verifiers)/len(all_results)*100:.1f}%)")
    print(f"Passed verifiers: {len(filtered_verifiers)} ({len(filtered_verifiers)/len(all_results)*100:.1f}%)")

def log_error_summary():
    """Log a summary of all errors encountered during processing."""
    total_functions = error_stats["counts"]["total_functions_attempted"]
    total_test_cases = error_stats["counts"]["total_testcases_attempted"]
    total_executions = error_stats["counts"]["total_function_executions"]
    total_validations = error_stats["counts"]["total_testcase_validations"]
    
    summary = ["\n===== ERROR SUMMARY =====\n"]
    
    # Parsing errors summary
    summary.append("PARSING ERRORS:")
    
    # Function parsing errors
    func_parsing_errors = sum(error_stats["parsing_errors"]["function"].values())
    summary.append(f"  Function parsing errors: {func_parsing_errors}/{total_functions} ({func_parsing_errors/total_functions*100:.1f}% failed)")
    
    if error_stats["parsing_errors"]["function"]:
        for reason, count in error_stats["parsing_errors"]["function"].most_common():
            summary.append(f"    {reason}: {count} ({count/total_functions*100:.1f}%)")
            # Add examples
            if reason in error_stats["details"] and error_stats["details"][reason]:
                summary.append(f"      Examples:")
                for i, example in enumerate(error_stats["details"][reason][:2]):
                    summary.append(f"      {i+1}. {example[:100]}...")
                    
    # Test case parsing errors
    test_parsing_errors = sum(error_stats["parsing_errors"]["test_case"].values())
    summary.append(f"\n  Test case parsing errors: {test_parsing_errors}/{total_test_cases} ({test_parsing_errors/max(1,total_test_cases)*100:.1f}% failed)")
    
    if error_stats["parsing_errors"]["test_case"]:
        for reason, count in error_stats["parsing_errors"]["test_case"].most_common():
            summary.append(f"    {reason}: {count} ({count/max(1,total_test_cases)*100:.1f}%)")
            # Add examples
            if reason in error_stats["details"] and error_stats["details"][reason]:
                summary.append(f"      Examples:")
                for i, example in enumerate(error_stats["details"][reason][:2]):
                    summary.append(f"      {i+1}. {example[:100]}...")
    
    # Function errors summary
    summary.append("\nFUNCTION ERRORS:")
    func_errors = sum(error_stats["function_errors"].values())
    if error_stats["function_errors"]:
        summary.append(f"  Total: {func_errors}/{total_executions} function executions failed ({func_errors/max(1,total_executions)*100:.1f}%)")
        for reason, count in error_stats["function_errors"].most_common():
            summary.append(f"    {reason}: {count} ({count/max(1,total_executions)*100:.1f}%)")
            # Add examples
            if reason in error_stats["details"] and error_stats["details"][reason]:
                summary.append(f"      Examples:")
                for i, example in enumerate(error_stats["details"][reason][:2]):
                    summary.append(f"      {i+1}. {example[:100]}...")
    else:
        summary.append("  None")
        
    # Test case errors summary
    summary.append("\nTEST CASE ERRORS:")
    test_case_errors = sum(error_stats["test_case_errors"].values())
    if error_stats["test_case_errors"]:
        summary.append(f"  Total: {test_case_errors}/{total_validations} test case validations failed ({test_case_errors/max(1,total_validations)*100:.1f}%)")
        for reason, count in error_stats["test_case_errors"].most_common():
            summary.append(f"    {reason}: {count} ({count/max(1,total_validations)*100:.1f}%)")
    else:
        summary.append("  None")
        
    # Cross validation errors summary
    summary.append("\nCROSS-VALIDATION ERRORS:")
    if error_stats["cross_validation_errors"]:
        for reason, count in error_stats["cross_validation_errors"].most_common():
            summary.append(f"  {reason}: {count}")
    else:
        summary.append("  None")

    summary.append("\n============================\n")
    
    # Log the summary
    logger.info("\n".join(summary))

def main():
    parser = argparse.ArgumentParser(description='Cross-validate verifiers and concatenate with queries')
    
    parser.add_argument('--verifiers_file', type=str, required=True,
                        help='Input file with verification functions')
    parser.add_argument('--all_results_file', type=str, required=True,
                        help='Output file for all verifiers with filter status')
    parser.add_argument('--filtered_file', type=str, required=True,
                        help='Output file for filtered verifiers')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Final output file with query-instruction pairs')
    
    args = parser.parse_args()

    logfile = f"logs/{os.path.basename(args.all_results_file)}.log"

    # Empty the log file before starting new processing
    with open(logfile, 'w') as f:
        # Just open in write mode to clear the file
        pass
    
    # Configure file logging
    logger.handlers = []  # Clear any existing handlers
    file_handler = logging.FileHandler(logfile)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Starting cross-validation with MIN_FUNCTIONS={MIN_FUNCTIONS}, MIN_TEST_CASES={MIN_TEST_CASES}")
    
    # Cross-validate verifiers
    cross_validate_verifiers(
        args.verifiers_file, 
        args.all_results_file, 
        args.filtered_file,
    )
    print(f"Full logfile available at: {logfile}")

if __name__ == "__main__":
    main()