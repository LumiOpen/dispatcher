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
    
    Args:
        response: The response to verify
        data: Dictionary containing instruction, query, and eval_func
        
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
            message=f"Language detection error: {e}",
            error_type="language_detection"
        )

    valid_lang_ids = {lang_code1, lang_code2} if lang_code2 is not None else {lang_code1}
    if LANGUAGE not in valid_lang_ids:
        raise TaskFailed(
            message=f"The response is not in the expected language {LANGUAGE}, but got {valid_lang_ids} <response>{response}</response>",
            error_type="invalid_language"
        )
    
    # Parse the evaluation functions
    eval_func_data = data.get('eval_func', [])
    if not eval_func_data:
        raise TaskFailed(
            message=f"No evaluation functions found",
            error_type="no_eval_functions"
        )
    
    # Use FunctionExecutor for safe function execution
    executor = FunctionExecutor()
    
    # Run each evaluation function and collect results
    acc = []
    for func in eval_func_data:
        try:
            # Execute function with timeout protection using the new method
            result = executor.execute_with_response(func, response, log_errors=True)
            if result is not None:
                acc.append(result)
        except Exception as e:
            raise TaskFailed(
                message=f"Error executing evaluation function {func}: {e} <response>{response}</response>",
                error_type="function_execution_failed"
            )
    
    # Calculate accuracy as in the original code
    acc_value = np.mean(acc) if acc else 0
    
    # Filter out responses with acc <= 0
    if acc_value <= 0:
        raise TaskFailed(
            message=f"The response did not pass the verification with accuracy {acc_value}. <response>{response}</response>",
            error_type="response_verification_failed"
        )

def construct_scoring_messages(response: str, data: Dict[str, Any]) -> List[Dict[str, str]]:
    """ Constructs the scoring prompt based on the response and data. """
    
    # If passed verification, construct the scoring prompt
    scoring_prompt = open("model_prompts/scoring_prompt.txt").read().strip()
    scoring_prompt = scoring_prompt.format(
        instruction=data.get('instruction', ''),
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