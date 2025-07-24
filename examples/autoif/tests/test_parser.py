#!/usr/bin/env python3
"""
Test script for ResponseParser functionality.

This script tests the ResponseParser class from response_parser.py
with various response formats to ensure proper parsing.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.response_parser import ResponseParser


def test_response_parser():
    """Test the ResponseParser with sample responses."""
    
    responses = [
        "```python\nimport re \n\ndef evaluate(response: str) -> bool:\n    # Count the number of code snippets in the response\n    code_snippets = re.findall(r'```.*?```', response, re.DOTALL)\n    \n    # Check if there are at least 2 code snippets\n    if len(code_snippets) >= 2:\n        return True\n    else:\n        return False\n```\n\n```json\n{\n    \"cases\": [\n        { \"input\": \"This is a test with ```code snippet 1``` and ```code snippet 2```\", \"output\": true },\n        { \"input\": \"This is a test with only ```one code snippet```\", \"output\": false },\n        { \"input\": \"This is a test with ```code snippet 1```, ```code snippet 2```, and ```code snippet 3```\", \"output\": true }\n    ]\n}\n```", 
        "```python\nimport re \n\ndef evaluate(response: str) -> bool:\n    # Count the number of code snippets in the response\n    code_snippets = re.findall(r'```.*?```', response, re.DOTALL)\n    \n    # Check if there are at least 2 code snippets\n    if len(code_snippets) >= 2:\n        return True\n    else:\n        return False\n```\n\n```json\n{\n    \"cases\": [\n        { \"input\": \"This is a response with ```one code snippet``` and ```another code snippet```\", \"output\": true },\n        { \"input\": \"This is a response with only ```one code snippet```\", \"output\": false },\n        { \"input\": \"This is a response with ```first code snippet```, ```second code snippet```, and ```third code snippet```\", \"output\": true }\n    ]\n}\n```", 
        "```python\nimport re \n\ndef evaluate(response: str) -> bool:\n    # Count the number of code snippets in the response\n    code_snippets = re.findall(r'```.*?```', response, re.DOTALL)\n    \n    # Check if there are at least 2 code snippets\n    if len(code_snippets) >= 2:\n        return True\n    else:\n        return False\n```\n\n```json\n{\n    \"cases\": [\n        { \"input\": \"This is a response with ```python code``` and another ```python code```\", \"output\": true },\n        { \"input\": \"This is a response with only one ```python code```\", \"output\": false },\n        { \"input\": \"This is a response with no code snippets\", \"output\": false }\n    ]\n}\n```", 
        "```python\nimport re \n\ndef evaluate(response: str) -> bool:\n    # Count the number of code snippets in the response\n    code_snippets = re.findall(r'```.*?```', response, re.DOTALL)\n    \n    # Check if there are at least 2 code snippets\n    if len(code_snippets) >= 2:\n        return True\n    else:\n        return False\n```\n\n```json\n{\n    \"cases\": [\n        { \"input\": \"This is a response with ```code snippet 1``` and ```code snippet 2```\", \"output\": true },\n        { \"input\": \"This is a response with ```code snippet 1``` but no second code snippet\", \"output\": false },\n        { \"input\": \"This is a response with ```code snippet 1``` and ```code snippet 2``` and ```code snippet 3```\", \"output\": true }\n    ]\n}\n```", 
        "```python\nimport re \n\ndef evaluate(response: str) -> bool:\n    # Count the number of code snippets in the response\n    code_snippets = re.findall(r'```.*?```', response, re.DOTALL)\n    \n    # Check if there are at least 2 code snippets\n    if len(code_snippets) >= 2:\n        return True\n    else:\n        return False\n```\n\n```json\n{\n    \"cases\": [\n        { \"input\": \"This is a response with a code snippet: ```print('Hello World')``` and another one: ```print('Hello Again')```\", \"output\": true },\n        { \"input\": \"This is a response with a single code snippet: ```print('Hello World')```\", \"output\": false },\n        { \"input\": \"This is a response without any code snippets\", \"output\": false }\n    ]\n}\n```", 
        "```python\nimport re\n\ndef evaluate(response: str) -> bool:\n    \"\"\"\n    Evaluates whether a response contains at least 2 code snippets.\n    \n    A code snippet is defined as a block of text enclosed in triple backticks (```) \n    or a block of text enclosed in single backticks (``) with a programming language \n    keyword (e.g., python, java, c++).\n\n    Args:\n        response (str): The input string to be evaluated.\n\n    Returns:\n        bool: True if the response contains at least 2 code snippets, False otherwise.\n    \"\"\"\n    # Count the number of triple backticks\n    triple_backticks = len(re.findall(r'```', response))\n    \n    # Count the number of single backticks with a programming language keyword\n    single_backticks = len(re.findall(r'`([^`]+)`', response))\n    programming_languages = ['python', 'java', 'c++', 'javascript', 'c#']\n    single_backticks_with_language = sum(1 for match in re.findall(r'`([^`]+)`', response) if any(language in match.lower() for language in programming_languages))\n    \n    # A code snippet is defined as a block of text enclosed in triple backticks or \n    # a block of text enclosed in single backticks with a programming language keyword\n    code_snippets = triple_backticks // 2 + single_backticks_with_language\n    \n    # Return True if the response contains at least 2 code snippets, False otherwise\n    return code_snippets >= 2\n```\n\n```json\n{\n    \"cases\": [\n        { \"input\": \"This is an example with ```python code``` and another ```java code```\", \"output\": true },\n        { \"input\": \"This is an example with `python code` and another `java code`\", \"output\": true },\n        { \"input\": \"This is an example with only one `python code`\", \"output\": false }\n    ]\n}\n```", 
        "```python\nimport re \n\ndef evaluate(response: str) -> bool:\n    # Count the number of code snippets in the response\n    code_snippets = re.findall(r'```.*?```', response, re.DOTALL)\n    \n    # Check if there are at least 2 code snippets\n    if len(code_snippets) >= 2:\n        return True\n    else:\n        return False\n```\n\n```json\n{\n    \"cases\": [\n        { \"input\": \"Here is a code snippet: ```print('Hello World')```. And here is another one: ```print('Hello again')```\", \"output\": true },\n        { \"input\": \"This response only has one code snippet: ```print('Hello World')```\", \"output\": false },\n        { \"input\": \"No code snippets here\", \"output\": false }\n    ]\n}\n```", 
        "```python\nimport re \n\ndef evaluate(response: str) -> bool:\n    # Count the number of code snippets in the response\n    code_snippets = re.findall(r'```.*?```', response, re.DOTALL)\n    \n    # Check if there are at least 2 code snippets\n    if len(code_snippets) >= 2:\n        return True\n    else:\n        return False\n```\n\n```json\n{\n    \"cases\": [\n        { \"input\": \"This is a response with ```code snippet 1``` and ```code snippet 2```\", \"output\": true },\n        { \"input\": \"This is a response with ```only one code snippet```\", \"output\": false },\n        { \"input\": \"This is a response with ```code snippet 1```, ```code snippet 2```, and ```code snippet 3```\", \"output\": true }\n    ]\n}\n```", 
        "```python\nimport re\n\ndef evaluate(response: str) -> bool:\n    # Count the number of code snippets in the response\n    code_snippets = re.findall(r'```.*?```', response, flags=re.DOTALL)\n    \n    # Check if there are at least 2 code snippets\n    if len(code_snippets) >= 2:\n        return True\n    else:\n        return False\n```\n\n```json\n{\n    \"cases\": [\n        { \"input\": \"This is a response with ```one code snippet``` and ```another code snippet```\", \"output\": true },\n        { \"input\": \"This is a response with only ```one code snippet```\", \"output\": false },\n        { \"input\": \"This is a response with ```first code snippet```, ```second code snippet```, and ```third code snippet```\", \"output\": true }\n    ]\n}\n```", 
        "```python\nimport re \n\ndef evaluate(response: str) -> bool:\n    # Count the number of code snippets in the response\n    code_snippets = re.findall(r'```.*?```', response, re.DOTALL)\n    \n    # Check if there are at least 2 code snippets\n    if len(code_snippets) >= 2:\n        return True\n    else:\n        return False\n```\n\n```json\n{\n    \"cases\": [\n        { \"input\": \"This is a test with ```python code``` and another ```java code```\", \"output\": true },\n        { \"input\": \"This is a test with ```python code```\", \"output\": false },\n        { \"input\": \"This is a test with ```python code``` and another ```java code``` and a third ```c++ code```\", \"output\": true }\n    ]\n}\n```"
    ]
    
    # Initialize the parser
    parser = ResponseParser()
    
    print("Testing ResponseParser with sample responses...")
    print("=" * 60)
    
    successful_parses = 0
    total_responses = len(responses)
    
    for i, response in enumerate(responses):
        print(f"\nTesting Response #{i + 1}:")
        print("-" * 30)
        
        # Parse the response
        parsed_data, error_reason = parser.parse_function_and_cases(response, f"test_{i}")
        
        if parsed_data is not None:
            successful_parses += 1
            print(f"✅ Successfully parsed!")
            print(f"   Function length: {len(parsed_data['func'])} characters")
            print(f"   Number of test cases: {len(parsed_data['cases'])}")
            
            # Show a snippet of the function
            func_lines = parsed_data['func'].split('\n')
            first_line = func_lines[0] if func_lines else ""
            print(f"   Function starts with: {first_line[:50]}...")
            
            # Show test case inputs
            for j, case in enumerate(parsed_data['cases'][:2]):  # Show first 2 cases
                input_preview = case['input'][:40] + "..." if len(case['input']) > 40 else case['input']
                print(f"   Case {j+1}: {input_preview} -> {case['output']}")
                
        else:
            print(f"❌ Failed to parse: {error_reason}")
    
    print("\n" + "=" * 60)
    print(f"Summary: {successful_parses}/{total_responses} responses parsed successfully")
    print(f"Success rate: {(successful_parses/total_responses)*100:.1f}%")
    
    return successful_parses == total_responses


if __name__ == "__main__":
    test_response_parser()

