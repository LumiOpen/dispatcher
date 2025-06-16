import signal
from typing import Dict, List, Optional, Any
import re
import numpy as np

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
    # First check if the response is in Finnish using GLOT
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        # Load GLOT model for language identification
        model_name = "cis-lmu/glotlid"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Get language prediction
        inputs = tokenizer(response, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        predicted_label = outputs.logits.argmax(-1).item()
        lang_id = model.config.id2label[predicted_label]
        
        # Check if the language is Finnish (ISO code 'fin')
        if lang_id != 'fin':
            print(f"Response language is {lang_id}, not Finnish (fin)")
            return None
            
    except Exception as e:
        print(f"Language identification error: {e}")
        # If language check fails, continue with other verifications
        # Alternative: return None to fail this verification
    
    # Set up timeout handler
    def timeout_handler(signum, frame):
        raise TimeoutError("Function execution timed out")
    
    # Parse the evaluation functions
    eval_func_data = data.get('eval_func', [])
    if not eval_func_data:
        print("No evaluation functions found")
        return None
    
    # Process eval_func list similar to the original code
    eval_funcs = []
    
    # Handle eval_func in format from original code: list of (func, score) tuples
    for func, score in eval_func_data:
        local_vars = {}
        try:
            exec(func, globals(), local_vars)
            if 'evaluate' in local_vars:
                eval_funcs.append(local_vars['evaluate'])
        except Exception as e:
            print(f"Error parsing evaluation function: {e}")
    
    if not eval_funcs:
        print("No valid evaluation functions found")
        return None
    
    # Run each evaluation function and collect results
    acc = []
    for eval_func in eval_funcs:
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(5)  # 5 second timeout
            res = eval_func(response)
        except Exception as e:
            print(f"Evaluation error: {e}")
            res = None
        finally:
            signal.alarm(0)  # Disable the alarm
        
        if res is not None:
            try:
                acc.append(int(res))
            except:
                continue
    
    # Calculate accuracy as in the original code
    acc_value = np.mean(acc) if acc else 0
    
    # Filter out responses with acc <= 0
    if acc_value <= 0:
        return None
    
    # If passed verification, construct the scoring prompt
    prompt_template = """You are an expert that is good at judging whether a response is following the instruction and query.
[Instruction] {instruction}
[Query] {query}
[Response] {response}
Please notice that the response may not be helpful as it needs to strictly follow the requirements in the Instruction.
You need to judge whether the response answers the query. Please first provide a detailed analysis and then give a score ranking from 0 to 10 at the last line.
Scoring 0 means the response is totally unrelated to the query, while scoring 10 means the response is helpful and highly related to the query.
Please only provide a score in the format `Score: {{score}}` without any other contents at the last line."""
    
    scoring_prompt = prompt_template.format(
        instruction=data.get('instruction', ''),
        query=data.get('query', ''),
        response=response
    )
    
    scoring_messages = [
        {"role": "user", "content": scoring_prompt}
    ]
    
    return scoring_messages


    """
    Verifies the query response using all provided evaluation functions.
    If the response passes verification (acc > 0), constructs a scoring prompt in messages format.
    
    Args:
        response: The response to verify
        data: Dictionary containing instruction, query, and eval_func
        
    Returns:
        List of messages for scoring or None if verification failed
    """
    # Set up timeout handler
    def timeout_handler(signum, frame):
        raise TimeoutError("Function execution timed out")
    
    # Parse the evaluation functions
    eval_func_data = data.get('eval_func', [])
    if not eval_func_data:
        print("No evaluation functions found")
        return None
    
    # Process eval_func list similar to the original code
    eval_funcs = []
    
    # Handle eval_func in format from original code: list of (func, score) tuples
    for func, score in eval_func_data:
        local_vars = {}
        try:
            exec(func, globals(), local_vars)
            if 'evaluate' in local_vars:
                eval_funcs.append(local_vars['evaluate'])
        except Exception as e:
            print(f"Error parsing evaluation function: {e}")
    
    if not eval_funcs:
        print("No valid evaluation functions found")
        return None
    
    # Run each evaluation function and collect results
    acc = []
    for eval_func in eval_funcs:
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(5)  # 5 second timeout
            res = eval_func(response)
        except Exception as e:
            print(f"Evaluation error: {e}")
            res = None
        finally:
            signal.alarm(0)  # Disable the alarm
        
        if res is not None:
            try:
                acc.append(int(res))
            except:
                continue
    
    # Calculate accuracy as in the original code
    acc_value = np.mean(acc) if acc else 0
    
    # Filter out responses with acc <= 0
    if acc_value <= 0:
        return None
    
    # If passed verification, construct the scoring prompt
    prompt_template = """You are an expert that is good at judging whether a response is following the instruction and query.
[Instruction] {instruction}
[Query] {query}
[Response] {response}
Please notice that the response may not be helpful as it needs to strictly follow the requirements in the Instruction.
You need to judge whether the response answers the query. Please first provide a detailed analysis and then give a score ranking from 0 to 10 at the last line.
Scoring 0 means the response is totally unrelated to the query, while scoring 10 means the response is helpful and highly related to the query.
Please only provide a score in the format `Score: {{score}}` without any other contents at the last line."""
    
    scoring_prompt = prompt_template.format(
        instruction=data.get('instruction', ''),
        query=data.get('query', ''),
        response=response
    )
    
    scoring_messages = [
        {"role": "user", "content": scoring_prompt}
    ]
    
    return scoring_messages
def response_score_filter(scored_text: str) -> Optional[str]:
    """
    Extracts the score from the scored text and returns the response if score > 8.
    
    Args:
        scored_text: The text containing the score
        
    Returns:
        The response if score > 8, otherwise None
    """
    # Extract the score using regex
    score_match = re.search(r'Score: (\d+)$', scored_text)
    
    if not score_match:
        return None
    
    try:
        score = int(score_match.group(1))
        if score > 8:  # Quality score threshold
            return scored_text
        return None
    except:
        return None