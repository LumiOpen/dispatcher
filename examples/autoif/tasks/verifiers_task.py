"""Verifiers task - Generate verification functions"""
from typing import Any, Dict, Generator, Union, List
import os

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task.base import GeneratorTask

__all__ = ["GenerateVerifiersTask"]

NUM_GENERATIONS = int(os.getenv("NUM_GENERATIONS", 3))

class GenerateVerifiersTask(GeneratorTask):
    """
    Generate verification functions and test cases for instructions.

    This task handles only the LLM inference step. Pre-processing and
    post-processing (cross-validation) are handled by separate scripts:
    - Pre-processing: create_verifiers_input.py
    - Post-processing: verifiers_cross_validation.py

    The task receives prompts and returns raw responses for the dispatcher
    to write as JSONL.
    """

    # Generation parameters for verifier generation
    GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 8192,
        "n": NUM_GENERATIONS,
    }

    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        """
        Generate verification functions for an instruction.

        Input data format (self.data):
        {
            'instruction_id': str,
            'instruction': str,
            'instruction_category': str,
            'prompt': str
        }

        This task:
        1. Generates multiple verifier variations
        2. Returns original data + responses for dispatcher to write as JSONL

        Post-processing (cross-validation of functions against test cases)
        is handled separately by verifiers_cross_validation.py.
        """
        # Get data from input
        instruction_id = self.data.get('instruction_id')
        instruction = self.data.get('instruction')
        instruction_category = self.data.get('instruction_category', '')
        prompt = self.data.get('prompt', '')

        if not prompt:
            raise ValueError("Input data must contain 'prompt' field")

        # Build chat messages
        messages = [{"role": "user", "content": prompt}]

        # Generate multiple variations of verifiers

        response: Response = yield Request({"messages": messages, **self.GEN_PARAMS})
        responses = response.get_text(n=NUM_GENERATIONS)

        # Return result for dispatcher output (raw responses)
        return {
            'original': self.data,
            'responses': responses
        }
