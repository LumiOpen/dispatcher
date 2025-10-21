"""Augmentation task - Generate augmented instructions"""
from typing import Any, Dict, Generator, Union

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task.base import GeneratorTask

__all__ = ["AugmentInstructionsTask"]


class AugmentInstructionsTask(GeneratorTask):
    """
    Generate augmented instructions from seed prompts.

    This task handles only the LLM inference step. Pre-processing and
    post-processing are handled by separate scripts:
    - Pre-processing: create_instructions_input.py
    - Post-processing: process_instructions_output.py

    The task receives prompts and returns raw responses for the dispatcher
    to write as JSONL.
    """

    # Generation parameters for instruction augmentation
    GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.75,
        "top_p": 0.9,
        "min_p": 0.0,
        "max_tokens": 8192,
        "stop": ["\n\n"],
    }

    def task_generator(self) -> Generator[Union[Request, list[Request]], Any, Dict[str, Any]]:
        """
        Generate augmented instructions.

        Input data format (self.data):
        {
            'prompt': str,
            'category': str (optional)
        }

        This task:
        1. Generates instructions using LLM
        2. Returns original data + responses for dispatcher to write as JSONL

        Post-processing (language filtering, deduplication) is handled
        separately by process_instructions_output.py.
        """
        # Get the prompt and category from input
        prompt = self.data.get('prompt', '')
        category = self.data.get('category', None)

        if not prompt:
            raise ValueError("Input data must contain 'prompt' field")

        # Build chat messages
        messages = [{"role": "user", "content": prompt}]

        # Generate response
        response: Response = yield Request({"messages": messages, **self.GEN_PARAMS})
        response_text = response.get_text()

        # Return result for dispatcher's output file (raw responses)
        # Post-processing will be done separately
        return {
            'original': self.data,
            'responses': [response_text],
            'category': category
        }
