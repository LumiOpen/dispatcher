"""Example tasks – two responses + judge, and validated response with retry"""
from typing import Any, Dict, Generator, List, Union

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task import GeneratorTask, TaskFailed, TaskRetry

__all__ = ["CompareTwoResponsesTask", "ValidatedResponseTask"]


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
        judge_model = judge_resp.model_name

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
            "judge_model": judge_model,
        }


class ValidatedResponseTask(GeneratorTask):
    """Generate a response and validate it against post-checks.

    Demonstrates using TaskRetry to handle cases where an LLM response does
    not pass user-defined validation. Instead of implementing sequential retry
    attempts inside task_generator (which would consume the work timeout of a
    single work item for multiple inference calls), raising TaskRetry releases
    the work item back to the dispatcher server for immediate re-issue.

    This keeps each attempt independent so that the per-item work timeout
    covers only a single inference call, and the server's built-in retry
    tracking (max_retries) decides when to give up and tombstone the item.
    """

    GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 4096,
    }

    # --------------- generator ---------------
    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        messages = self.data.get("messages")

        # Step 1 - generate a response
        resp: Response = yield Request({"messages": messages, **self.GEN_PARAMS})
        text = resp.get_text()

        # Step 2 - run post-checks on the response
        # If validation fails, raise TaskRetry so the dispatcher server
        # re-issues this work item to a worker for another attempt.
        if not text or not text.strip():
            raise TaskRetry(message="Empty response from model")

        # Example: check that the response contains a required marker
        if "ANSWER:" not in text:
            raise TaskRetry(message="Response missing required ANSWER: marker")

        # Step 3 - validation passed, return the result
        return {
            "messages": messages,
            "response": text.strip(),
        }
