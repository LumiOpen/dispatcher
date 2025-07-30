"""Example task – two responses + judge"""
from typing import Any, Dict, Generator, List, Union

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task.base import GeneratorTask, TaskFailed

__all__ = ["CompareTwoResponsesTask"]


class CompareTwoResponsesTask(GeneratorTask):
    """Generate two answers, have the model judge, and return preferred vs dispreferred."""

    # Fixed generation hyper‑parameters for candidate answers
    GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 4096,
    }

    # Deterministic parameters for the judge
    JUDGE_PARAMS: Dict[str, Any] = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 256,
    }

    # --------------- generator ---------------
    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        # self.data is prepopulated with the data from the jsonl row being
        # processed
        messages = self.data.get("messages")

        # Step 1 – two identical generation requests
        responses: List[Response] = yield [
            Request({"messages": messages, **self.GEN_PARAMS}),
            Request({"messages": messages, **self.GEN_PARAMS}),
        ]

        resp_a, resp_b = responses  # arrival order defines A and B
        text_a = resp_a.get_text()
        text_b = resp_b.get_text()

        # Step 2 – judge prompt
        user_prompt = next((m.get("content") for m in messages if m.get("role") == "user"), "(unknown)")
        judge_messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict judge. Reply with 'A' or 'B' to indicate which response is better."),
            },
            {
                "role": "user",
                "content": (
                    f"### User prompt\n{user_prompt}\n\n"
                    f"### Response A\n{text_a}\n\n"
                    f"### Response B\n{text_b}\n\n"
                    "Which response is better? Reply with just 'A' or 'B'."
                ),
            },
        ]
        judge_resp: Response = yield Request({"messages": judge_messages, **self.JUDGE_PARAMS})
        judge_text = judge_resp.get_text()

        if not judge_text:
            raise TaskFailed(
                message="Judge model returned an empty empty or invalid response.",
                error_type="judge_response_invalid"
            )

        judge_text = judge_text.strip().upper()
        
        winner_is_a = False
        if judge_text.startswith("A"):
            winner_is_a = True
        elif not judge_text.startswith("B"):
            raise TaskFailed(
                message=f"Judge model returned an unexpected response: '{judge_text}'",
                error_type="judge_response_invalid"
            )

        if winner_is_a:
            pref_resp, dis_resp = resp_a, resp_b
        else:
            pref_resp, dis_resp = resp_b, resp_a

        # return dict can contain anything you wish to record from this task.
        return {
            "messages": messages,
            "preferred_text": pref_resp.get_text(),
            "dispreferred_text": dis_resp.get_text(),
        }
