import signal
from typing import Dict, List, Optional, Any
import re
import numpy as np
import os
from src.utils.lang_id import detect_language
from src.utils.function_executor import FunctionExecutor

from dispatcher.taskmanager.task.base import TaskFailed

LANGUAGE=os.environ.get("LANGUAGE")

def response_verify(response: str, data: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    """
    Verifies the query response using all provided evaluation functions and checks
    that the response is in the target language.
    
    The evaluation functions can be in two formats:
    1. Flat list: All functions are applied and overall accuracy is calculated
    2. List of lists: Each sub-list contains functions for one instruction. 
       The response must pass at least one function from each sub-list to ensure
       it follows all instructions.
    
    Args:
        response: The response to verify
        data: Dictionary containing instruction, query, and eval_funcs (or eval_func for backward compatibility)
              eval_funcs can be:
              - List of functions (old format)
              - List of lists, where each sub-list contains functions for one instruction (new format)
        
    Returns:
        List of messages for scoring or None if verification failed
    """
    # First check if the response is in the desired language
    # Get language prediction
    # lang_code1 is the three-letter code, lang_code2 is the two-letter code
    try:
        lang_code1, lang_code2 = detect_language(response)
    except Exception as e:
        raise TaskFailed(
            message=f"Language detection error: {e} <response>{response}</response>",
            error_type="language_detection"
        )

    valid_lang_ids = {lang_code1, lang_code2} if lang_code2 is not None else {lang_code1}
    if LANGUAGE not in valid_lang_ids:
        raise TaskFailed(
            message=f"The response is not in the expected language {LANGUAGE}, but got {valid_lang_ids} <response>{response}</response>",
            error_type="invalid_language"
        )
    
    # Parse the evaluation functions
    eval_func_data = data.get('eval_funcs', data.get('eval_func', []))
    if not eval_func_data:
        raise TaskFailed(
            message=f"No evaluation functions found",
            error_type="no_eval_functions"
        )
    
    # Use FunctionExecutor for safe function execution
    executor = FunctionExecutor()
    
    # Normalize eval_func_data to always be a list of lists
    # If it's a flat list (old format), wrap it in a single list (treat as one instruction)
    # If it's already a list of lists (new format), use as-is
    normalized_eval_funcs = eval_func_data if eval_func_data and isinstance(eval_func_data[0], list) else [eval_func_data]
    
    # Get instruction IDs from data, or use enumerated indices as fallback
    instruction_ids = data.get('instruction_ids', list(data.get('instruction_id', range(len(normalized_eval_funcs)))))
    
    # Process each instruction group
    instruction_results = []
    accuracy_threshold = 0  # Threshold for instruction to pass
    
    for idx, instruction_funcs in enumerate(normalized_eval_funcs):
        instruction_id = instruction_ids[idx] if idx < len(instruction_ids) else idx
        
        if not instruction_funcs:
            raise TaskFailed(
                message=f"No evaluation functions found for instruction {instruction_id}",
                error_type="no_eval_functions_for_instruction"
            )
        
        # Run all functions for this instruction and collect results
        instruction_acc = []
        for func in instruction_funcs:
            try:
                # Execute function with timeout protection using the new method
                result = executor.execute_with_response(func, response, log_errors=True)
                if result is not None:
                    instruction_acc.append(result)
            except Exception as e:
                raise TaskFailed(
                    message=f"Error executing evaluation function {func} for instruction {instruction_id}: {e} <response>{response}</response>",
                    error_type="function_execution_failed"
                )
        
        # For this instruction, calculate accuracy
        instruction_accuracy = np.mean(instruction_acc) if instruction_acc else 0
        instruction_results.append(instruction_accuracy)
    
    # Check if ALL instructions pass the threshold (all instructions must be followed)
    failed_instructions = []
    for idx, accuracy in enumerate(instruction_results):
        instruction_id = instruction_ids[idx] if idx < len(instruction_ids) else idx
        if accuracy <= accuracy_threshold:
            failed_instructions.append((instruction_id, accuracy))
    
    if failed_instructions:
        # Format error message based on number of failed instructions
        if len(failed_instructions) == 1:
            instruction_id, accuracy = failed_instructions[0]
            error_context = f" for instruction {instruction_id}" if len(normalized_eval_funcs) > 1 else ""
            raise TaskFailed(
                message=f"The response did not pass verification{error_context} with accuracy {accuracy}. <response>{response}</response>",
                error_type="instruction_verification_failed"
            )
        else:
            failed_ids = [str(instr_id) for instr_id, _ in failed_instructions]
            raise TaskFailed(
                message=f"The response did not pass verification for instructions {', '.join(failed_ids)}. <response>{response}</response>",
                error_type="multiple_instructions_verification_failed"
            )

def construct_scoring_messages(response: str, data: Dict[str, Any]) -> List[Dict[str, str]]:
    """ Constructs the scoring prompt based on the response and data. """
    
    # If passed verification, construct the scoring prompt
    scoring_prompt = open("model_prompts/scoring_prompt.txt").read().strip()
    scoring_prompt = scoring_prompt.format(
        instructions=data.get('instructions', list(data.get('instruction', ''))),
        query=data.get('query', ''),
        response=response
    )
    
    scoring_messages = [
        {"role": "user", "content": scoring_prompt}
    ]

    return scoring_messages

def extract_score(scored_text: str) -> Optional[int]:
    """
    Extracts the score from the scored text
    
    Args:
        scored_text: The text containing the score
        
    Returns:
        The extracted score
    """
    # Extract the score using regex
    score_match = re.search(r'Score: (\d+)$', scored_text)
    
    if not score_match:
        raise TaskFailed(
            message=f"Score not found in the scoring response: {scored_text}",
            error_type="score_extraction_failed"
        )
    
    try:
        score = int(score_match.group(1))
    except Exception as e:
        raise TaskFailed(
            message=f"Error converting score to integer: {e}",
            error_type="score_conversion_failed"
        )
    return score