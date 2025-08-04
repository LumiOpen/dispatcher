"""Example task – two responses + judge"""
from typing import Any, Dict, Generator, List, Union
import re

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task.base import GeneratorTask

from src.utils.response_handler import response_verify, extract_score, construct_scoring_messages

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
        #     'instruction_id': instruction_id,
        #     'instruction': instruction,
        #     'query': query,
        #     'query_response': query_response,
        #     'query_metadata': query_metadata,
        #     'eval_func': 'def evaluate():...',
        #     'cases': [{'input', 'output'}],
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
        
        messages = [
            {
                "role": "user", 
                "content": f"{query} {self.data.get('instruction')}"
            },
            {
                "role": "assistant", 
                "content": response_text
            },
        ]

        return {
            'instruction_id': self.data.get("instruction_id"),
            'instruction': self.data.get("instruction"),
            'query': self.data.get("query"),
            'query_response': self.data.get("query_response"),
            'query_metadata': self.data.get("query_metadata"),
            'response': response_text,
            'eval_func': self.data.get("eval_func"),
            'cases': self.data.get("cases"),
            'prompt': self.data.get("prompt"),
            'messages': messages,
            'score': score,
            'scoring_response': scoring_text
        }
