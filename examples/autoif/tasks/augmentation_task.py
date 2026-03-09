"""Augmentation task - Generate augmented instructions"""
import os
from typing import Any, Dict, Generator, Union

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task.base import GeneratorTask

__all__ = ["AugmentInstructionsTask"]

NUM_GENERATIONS = int(os.getenv("NUM_GENERATIONS", 2))

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
        "max_tokens": 8192,
        "n": NUM_GENERATIONS,
    }

    def task_generator(self) -> Generator[Union[Request, list[Request]], Any, Dict[str, Any]]:
        """
        Generate augmented instructions.

        Input data format (self.data):
        {
            'prompt': str,
            'category': str (optional)
        }

        """
        # Get the prompt and category from input
        prompt = self.data.get('prompt', '')
        category = self.data.get('category', None)

        if not prompt:
            raise ValueError("Input data must contain 'prompt' field")

        # Build chat messages
        messages = [{"role": "user", "content": prompt}]

        response: Response = yield Request({"messages": messages, **self.GEN_PARAMS})

        # Parse all N generations
        responses = []
        response_texts = response.get_text(n=NUM_GENERATIONS)
        if response_texts:
            responses = [text for text in response_texts if text]

        return {
            'original': self.data,
            'responses': responses,
            'category': category
        }
