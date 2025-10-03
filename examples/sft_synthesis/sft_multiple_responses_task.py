"""Example task – two responses + judge"""
from typing import Any, Dict, Generator, List, Union

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task import GeneratorTask, TaskFailed
import logging
import re

__all__ = ["MultipleResponsesTask"]

JUDGE_PROMPT = """
Review the user’s request and the corresponding response using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the response is relevant and provides some information related to the user’s inquiry, even if it is incomplete or contains some irrelevant content. 
- Add a second point if the response addresses a substantial portion of the user’s request, but does not completely resolve the query or provide a direct answer. 
- Award a third point if the response answers the basic elements of the user’s request in a useful way, regardless of whether it seems to have been written by an AI Assistant or if it has elements typically found in blogs or search results. 
- Grant a fourth point if the response is clearly written from an AI Assistant’s perspective, addressing the user’s request directly and comprehensively, and is well-organized and helpful, even if there is slight room for improvement in clarity, conciseness or focus. 
- Bestow a fifth point for a response that is impeccably tailored to the user’s request by an AI Assistant, without extraneous information, reflecting expert knowledge, and demonstrating a high-quality, engaging, and insightful answer. 

Evaluation:
The user's request and the response given are immediately below in the <request></request> and <response></response> tags, respectively.

<request>{prompt}</request>
<response>{response}</response>

After examining the user’s instruction and the response.

- Briefly justify your total score, up to 100 words. 
- Conclude with the score using the format: “Score: X/5” 

Remember to assess from the AI Assistant perspective. To evaluate the response in alignment with this additive scoring model, systematically attribute points based on the outlined criteria.
"""

class MultipleResponsesTask(GeneratorTask):
    logger = logging.getLogger(__name__)
    # Fixed generation hyper‑parameters for candidate answers
    GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 16384,
        "n": 5,  # response per request
    }

    # Deterministic parameters for the judge
    JUDGE_PARAMS: Dict[str, Any] = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 512,
    }

    # --------------- generator ---------------
    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        # self.data is prepopulated with the data from the jsonl row being
        # processed
        messages = self.data.get("messages")
        # id = self.data.get("id")
        responses_array = []

        # Step 1 – get n responses for prompt
        responses: List[Response] = yield Request({"messages": messages, **self.GEN_PARAMS})
        resp_texts = responses.get_text(n=self.GEN_PARAMS["n"])
        self.logger.debug(f"\n\nRESPONSES:\n{resp_texts}\n")

        # Step 2 - score each response
        user_prompt = next((m.get("content") for m in messages if m.get("role") == "user"), "(unknown)")
        for resp_index, resp_text in enumerate(resp_texts):
            judge_messages = [
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(prompt=user_prompt, response=resp_text)
                },
            ]
            judge_resp: List[Response] = yield Request({"messages": judge_messages, **self.GEN_PARAMS})
            judge_text = judge_resp.get_text()
            if not judge_text:
                raise TaskFailed(
                    message="Judge model returned an empty empty or invalid response.",
                    error_type="judge_response_invalid"
                )
            score_match = re.search(r'Score:\s*(\d+)/\d+', judge_text)
            if not score_match:
                raise TaskFailed(
                    message="Could not find Score",
                    error_type="invalid score format"
                )
            score = int(score_match.group(1))
            responses_array.append({"response": resp_text, "score": score})

        return {
            # "id": id,
            "prompt": user_prompt,
            "responses": responses_array
        }
