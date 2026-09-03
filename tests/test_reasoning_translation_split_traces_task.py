from __future__ import annotations

import importlib.util
import pathlib
import unittest
from types import SimpleNamespace


TASK_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "examples/backlog_inference/tasks/reasoning_translation_split_traces_task.py"
)
SPEC = importlib.util.spec_from_file_location(
    "reasoning_translation_split_traces_task", TASK_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

Task = MODULE.ReasoningTranslationSplitTracesTask
IssueType = MODULE.TranslationIssueType


class FakeBatchEncoding:
    def __init__(self, token_ids: list[int]):
        self.data = {
            "input_ids": token_ids,
            "attention_mask": [1] * len(token_ids),
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key]


class TransformersV5StyleTokenizer:
    """Return BatchEncoding-like metadata unless return_dict=False is used."""

    def __init__(self, rendered_lengths: dict[str, int]):
        self.rendered_lengths = rendered_lengths

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        return_dict=None,
    ):
        assert tokenize is True
        assert add_generation_prompt is True
        content = messages[0]["content"]
        length = next(
            (
                rendered_length
                for marker, rendered_length in self.rendered_lengths.items()
                if marker in content
            ),
            1,
        )
        token_ids = list(range(length))
        if return_dict is False:
            return token_ids
        return FakeBatchEncoding(token_ids)

    def encode(self, text):
        return text.split()


class TestReasoningTranslationSplitTracesTask(unittest.TestCase):
    def setUp(self):
        self.original_get_tokenizer = Task.__dict__["get_tokenizer"]
        self.original_max_model_len = MODULE.MAX_MODEL_LEN
        self.original_trace_params = Task.TRACES_TRANSLATION_GEN_PARAMS
        MODULE.MAX_MODEL_LEN = 10
        Task.TRACES_TRANSLATION_GEN_PARAMS = {
            **Task.TRACES_TRANSLATION_GEN_PARAMS,
            "max_tokens": 4,
        }

    def tearDown(self):
        Task.get_tokenizer = self.original_get_tokenizer
        MODULE.MAX_MODEL_LEN = self.original_max_model_len
        Task.TRACES_TRANSLATION_GEN_PARAMS = self.original_trace_params

    @staticmethod
    def data():
        return {
            "id": "sample-1",
            "messages": [
                {"role": "user", "content": "question-marker"},
                {
                    "role": "assistant",
                    "content": "<think>trace-marker</think>answer-marker",
                },
            ],
        }

    @staticmethod
    def context():
        return SimpleNamespace(retry_count=0, max_retries=3)

    def test_preflight_counts_token_ids_not_batch_encoding_fields(self):
        Task.get_tokenizer = staticmethod(
            lambda: TransformersV5StyleTokenizer({"trace-marker": 7})
        )

        task = Task(self.data(), self.context())

        self.assertTrue(task.is_done())
        self.assertFalse(task.should_retry())
        self.assertIsNone(task.get_next_request())
        result, _ = task.get_result()
        self.assertFalse(result["task_metadata"]["success"])
        self.assertEqual(
            result["task_metadata"]["error_type"],
            IssueType.CONTEXT_LENGTH_EXCEEDED,
        )
        params = result["translation_issues"][0]["params"]
        self.assertEqual(params["input_tokens"], 7)
        self.assertEqual(params["max_input_tokens"], 6)

    def test_preflight_allows_prompt_that_exactly_fits(self):
        Task.get_tokenizer = staticmethod(
            lambda: TransformersV5StyleTokenizer({"trace-marker": 6})
        )

        task = Task(self.data(), self.context())

        self.assertFalse(task.is_done())
        self.assertIsNotNone(task.get_next_request())


if __name__ == "__main__":
    unittest.main()
