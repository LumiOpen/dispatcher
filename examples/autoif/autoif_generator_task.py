"""Example task – two responses + judge"""
from typing import Any, Dict, Generator, List, Union
import re

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task.base import GeneratorTask

from src.utils.response_handler import response_verify, extract_score, construct_scoring_messages, format_instructions_with_conjunctions

__all__ = ["GenerateQueryResponsesTask"]


class GenerateQueryResponsesTask(GeneratorTask):
    """Generate query responses with verifiers"""

    # Fixed generation hyper‑parameters for candidate answers
    GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 4096,
    }

    # --------------- generator ---------------
    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        # self.data is prepopulated with the data from the jsonl row being
        # processed
        # these are queries with verifiers of format
        # {
        #     'instruction_ids': instruction_ids, # backwards compatible with 'instruction_id'
        #     'instructions': instructions, # backwards compatible with 'instruction'
        #     'query': query,
        #     'query_response': query_response,
        #     'query_metadata': query_metadata,
        #     'eval_funcs': [["def evaluate():...",],], # backwards compatible with 'eval_func' ["def evaluate():...",]
        #     'cases': [[{'input', 'output'},],], # backwards compatible with 'cases' [{'input', 'output'},]
        #     'prompt': prompt
        # }
        # there is M lines with the same query with different instructions
        queries_messages = [
            {
                "role": "user",
                "content": (
                    self.data.get("prompt")
                ),
            },
        ]
        
        # Step 1 – get response for the query
        # print(f"queries_messages: {queries_messages}")
        queries_resp: Response = yield Request({"messages": queries_messages, **self.GEN_PARAMS})
        response_text = queries_resp.get_text()
        # this function performs response verification
        # based on by AutoIF/code/7_query_verification.py
        # if the response did not pass the verification, it will raise a TaskFailed exception
        response_verify(response_text, self.data)

        scoring_messages = construct_scoring_messages(response_text, self.data)

        # Step 2 - score query response
        scored_resp: Response = yield Request({"messages": scoring_messages, **self.GEN_PARAMS})
        scoring_text = scored_resp.get_text()
        # this function extracts the score from the response. 
        # If the score is not found, it will raise a TaskFailed exception
        score = extract_score(scoring_text)

        # Add fullstop if query doesn't end with punctuation
        query = self.data.get("query", "")
        if not re.search(r'[.!?]$', query):
            query += "."

        # Handle both "instructions" (list) and "instruction" (string - old version) cases
        instructions = self.data.get("instructions", self.data.get("instruction", ""))
        instructions_text = format_instructions_with_conjunctions(instructions)

        messages = [
            {
                "role": "user", 
                "content": f"{query} {instructions_text}"
            },
            {
                "role": "assistant", 
                "content": response_text
            },
        ]

        return {
            'instruction_ids': self.data.get("instruction_ids", self.data.get("instruction_id")),
            'instructions': self.data.get("instructions", self.data.get("instruction")),
            'query': self.data.get("query"),
            'query_response': self.data.get("query_response"),
            'query_metadata': self.data.get("query_metadata"),
            'response': response_text,
            'eval_funcs': self.data.get("eval_funcs", self.data.get("eval_func")),
            'cases': self.data.get("cases"),
            'prompt': self.data.get("prompt"),
            'messages': messages,
            'score': score,
            'scoring_response': scoring_text
        }
