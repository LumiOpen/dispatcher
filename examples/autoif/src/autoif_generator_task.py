"""Example task – two responses + judge"""
from typing import Any, Dict, Generator, List, Union

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task.base import GeneratorTask

from dispatcher.examples.autoif.src.utils.response_handler import response_verify, response_score_filter

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
        #     'instruction': instruction,
        #     'query': query,
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
        queries_resp: Response = yield Request({"messages": queries_messages, **self.GEN_PARAMS})
        # this function performs response verification
        # and constructs scoring prompt
        # based on by AutoIF/code/7_query_verification.py
        # the output can be None if the response did not pass the verification
        query_scoring_msgs = response_verify(queries_resp.get_text(), self.data)

        if query_scoring_msgs is None:
            self.data['response'] = None
            return self.data
        
        # Step 2 - score query response
        # TODO Should this be multiple identical generations?
        scored_resp: Response = yield Request({"messages": query_scoring_msgs, **self.GEN_PARAMS})

        # Filter responses based on scores
        # this function checks the scores and returns the response
        # or returns None if the response does not pass the filter
        filtered_response = response_score_filter(scored_resp.get_text())

        if filtered_response is None:
            self.data['response'] = None
            return self.data
        
        self.data['response'] = filtered_response
        return self.data
        # TODO sft format
