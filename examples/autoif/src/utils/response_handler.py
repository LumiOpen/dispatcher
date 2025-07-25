import signal
from typing import Dict, List, Optional, Any
import re
import numpy as np
import os
from src.utils.lang_id import detect_language
from src.utils.function_executor import FunctionExecutor

LANGUAGE=os.environ.get("LANGUAGE")

def response_verify(response: str, data: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    """
    Verifies the query response using all provided evaluation functions and checks
    that the response is in Finnish language.
    
    Args:
        response: The response to verify
        data: Dictionary containing instruction, query, and eval_func
        
    Returns:
        List of messages for scoring or None if verification failed
    """
    # First check if the response is in the desired language
    try:

        # Get language prediction
        # lang_code1 is the three-letter code, lang_code2 is the two-letter code
        lang_code1, lang_code2 = detect_language(response)
        if lang_code1 != LANGUAGE:
            print(f"Response language is {lang_code1} ({lang_code2}). Expected {LANGUAGE}.")
            return None
            
    except Exception as e:
        print(f"Language identification error: {e}")
        # If language check fails, continue with other verifications
        # Alternative: return None to fail this verification
    
    # Parse the evaluation functions
    eval_func_data = data.get('eval_func', [])
    if not eval_func_data:
        print("No evaluation functions found")
        return None
    
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
            print(f"Error executing evaluation function: {e}")
            continue
    
    # Calculate accuracy as in the original code
    print(f"Accuracy scores: {acc}")
    acc_value = np.mean(acc) if acc else 0
    
    # Filter out responses with acc <= 0
    if acc_value <= 0:
        return None
    
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
        return None
    
    try:
        score = int(score_match.group(1))
    except Exception as e:
        print(f"Error extracting score: {e}")
        return None
    return score