"""Per-turn keyword handling functionality for instruction processing and generation."""
from typing import Any, Dict, Generator, List, Union, Optional
import re
import os
import json
from dataclasses import dataclass

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task.base import TaskFailed
from .utils.error_utils import format_error_type_with_turn


@dataclass
class KeywordData:
    """Internal data class for keyword data for a specific turn."""
    instruction_ids: List
    instructions: List
    instruction_categories: List[str]
    query: str
    original_instructions: List[str]  # Store originals before modification


class KeywordHandler:
    """Per-turn keyword handler for instruction processing and generation.

    This class handles keyword-related functionality for a single turn,
    maintaining no cross-turn state and not modifying external data.
    """

    # Constants for keyword handling
    KEYWORD_CATEGORY = 'keyword'
    # Structure: {key_name: (default_value, expected_type)}
    KWARGS_KEYS = {
        'keywords': ([], list),
        'N': (1, int)
    }

    def __init__(self,
                 turn_idx: int,
                 instruction_categories: List[str],
                 instructions: List[str],
                 instruction_ids: List,
                 query: str):
        """Initialize with the base path and data for a specific turn.

        Args:
            turn_idx: Current turn index
            instruction_categories: Categories for this turn
            instructions: Instructions for this turn
            instruction_ids: Instruction IDs for this turn
            query: Query for this turn
        """
        self.turn_idx = turn_idx

        # Store original data (immutable)
        self.data = KeywordData(
            instruction_ids=instruction_ids.copy(),
            instructions=instructions.copy(),
            instruction_categories=instruction_categories.copy(),
            query=query,
            original_instructions=instructions.copy()
        )

        # Find keyword instructions for this turn
        self.keyword_instruction_indices = self._find_keyword_instruction_indices()

        # Store generated keyword data (will be populated during generation)
        self.keyword_generation_data = {}

    def has_keyword_instructions(self) -> bool:
        """Check if there are any keyword instructions in this turn."""
        return len(self.keyword_instruction_indices) > 0

    def _find_keyword_instruction_indices(self) -> List[int]:
        """Find indices of keyword instructions in this turn."""
        keyword_indices = []
        for idx, category in enumerate(self.data.instruction_categories):
            if category == self.KEYWORD_CATEGORY:
                keyword_indices.append(idx)
        return keyword_indices

    def process_keyword_generation(self, gen_params: Dict[str, Any]) -> Generator[Union[Request, List[Request]], Any, None]:
        """Process keyword generation for all keyword instructions in this turn."""
        for instruction_idx in self.keyword_instruction_indices:
            # Get the original instruction that needs keyword generation
            original_instruction = self.data.original_instructions[instruction_idx]

            # Load keyword generation prompt and format it
            keyword_prompt_template = self._load_keyword_generation_prompt()
            keyword_generation_prompt = keyword_prompt_template.format(
                query=self.data.query,
                instruction=original_instruction
            )

            # Generate keywords
            keyword_resp: Response = yield Request({
                "messages": [{"role": "user", "content": keyword_generation_prompt}],
                **gen_params
            })
            keyword_response_text = keyword_resp.get_text()

            # Parse JSON response and store it
            try:
                keyword_data = self._parse_keyword_response(keyword_response_text)
                self._store_keyword_data(instruction_idx, keyword_data, original_instruction)

            except (json.JSONDecodeError, KeyError) as e:
                raise TaskFailed(
                    message=f"Failed to parse keyword generation response at turn {self.turn_idx}, instruction {instruction_idx}: {e}. Response was: {keyword_response_text}",
                    error_type=format_error_type_with_turn("keyword_generation_failed", self.turn_idx)
                )

    def apply_keyword_modifications_to_prompt(self, prompt: str) -> str:
        """Apply keyword-generated instruction modifications to the prompt.

        Args:
            prompt: Original prompt to modify

        Returns:
            Modified prompt with keyword replacements
        """
        modified_prompt = prompt

        # Replace all keyword instructions in this turn with the newly generated ones
        for instruction_idx, keyword_data in self.keyword_generation_data.items():
            original_instruction = self.data.original_instructions[instruction_idx]
            new_instruction = keyword_data['new_instruction']

            # Use regex to find and replace the original instruction in the prompt
            escaped_original = re.escape(original_instruction)
            modified_prompt = re.sub(escaped_original, new_instruction, modified_prompt)

        return modified_prompt

    def get_final_instructions(self) -> List[str]:
        """Get the final instructions for this turn with keyword replacements applied.

        Returns:
            List of final instructions with keyword modifications applied
        """
        final_instructions = self.data.original_instructions.copy()

        # Apply keyword replacements
        for instruction_idx, keyword_data in self.keyword_generation_data.items():
            if instruction_idx < len(final_instructions):
                final_instructions[instruction_idx] = keyword_data['new_instruction']

        return final_instructions

    def get_execution_kwargs(self) -> List[Dict[str, Any]]:
        """Get keyword generation kwargs for evaluation functions for this turn.

        Returns:
            List of kwargs dictionaries, one per instruction in the turn.
            Empty dict for non-keyword instructions.
        """
        # Initialize list of kwargs dictionaries for all instructions
        kwargs_data = []

        for instruction_idx in range(len(self.data.instructions)):
            if instruction_idx in self.keyword_generation_data:
                # This is a keyword instruction - include its kwargs
                keyword_data = self.keyword_generation_data[instruction_idx]
                instruction_kwargs = {}
                for key in self._get_kwargs_key_names():
                    instruction_kwargs[key] = keyword_data[key]
                kwargs_data.append(instruction_kwargs)
            else:
                # This is a non-keyword instruction - use empty dict
                kwargs_data.append({})

        return kwargs_data

    def _load_keyword_generation_prompt(self) -> str:
        """Load the keyword generation prompt from file."""
        with open("model_prompts/generate_keywords_prompt.txt", 'r', encoding='utf-8') as f:
            return f.read()

    def _parse_keyword_response(self, keyword_response_text: str) -> Dict[str, Any]:
        """Parse keyword generation JSON response."""
        try:
            # First try to extract JSON from triple backticks (expected format)
            backticks_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', keyword_response_text, re.DOTALL)
            if backticks_match:
                return json.loads(backticks_match.group(1))
            else:
                # Fallback: Extract JSON from response (handle cases where response might have extra text)
                json_match = re.search(r'\{.*\}', keyword_response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    return json.loads(keyword_response_text)
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            raise TaskFailed(
                message=f"Failed to parse keyword generation response as valid JSON: {e}. Response text: {keyword_response_text}",
                error_type=format_error_type_with_turn("keyword_response_parse_failed", None)
            )

    def _store_keyword_data(self, instruction_idx: int, keyword_data: Dict[str, Any], original_instruction: str) -> None:
        """Store keyword generation data for later use."""
        # Dynamically build the stored data based on KWARGS_KEYS
        stored_data = {}
        for key in self._get_kwargs_key_names():
            raw_value = keyword_data.get(key)
            # Convert to expected type with fallback to default
            converted_value = self._convert_value_to_expected_type(key, raw_value)
            stored_data[key] = converted_value

        # Always include the new instruction
        stored_data['new_instruction'] = keyword_data.get('instruction', original_instruction)
        self.keyword_generation_data[instruction_idx] = stored_data

    def _get_kwargs_key_names(self) -> List[str]:
        """Get list of all kwargs key names for iteration."""
        return list(self.KWARGS_KEYS.keys())

    def _get_default_value_for_key(self, key: str) -> Any:
        """Get the appropriate default value for a specific kwargs key."""
        if key in self.KWARGS_KEYS:
            return self.KWARGS_KEYS[key][0]  # Return the default value
        return None  # Default to None for unknown keys

    def _get_expected_type_for_key(self, key: str) -> type:
        """Get the expected type for a specific kwargs key."""
        if key in self.KWARGS_KEYS:
            return self.KWARGS_KEYS[key][1]  # Return the expected type
        return type(None)  # Default to None type for unknown keys

    def _convert_value_to_expected_type(self, key: str, value: Any) -> Any:
        """Convert a value to the expected type for a specific kwargs key."""
        if value is None:
            return self._get_default_value_for_key(key)

        expected_type = self._get_expected_type_for_key(key)

        # Handle special cases
        if expected_type == list and not isinstance(value, list):
            # If expecting a list but got something else, wrap it in a list
            return [value] if value is not None else []
        elif expected_type == int and not isinstance(value, int):
            # Try to convert to int
            try:
                if isinstance(value, str):
                    return int(value)
                elif isinstance(value, float):
                    return int(value)
                else:
                    return self._get_default_value_for_key(key)
            except (ValueError, TypeError):
                return self._get_default_value_for_key(key)
        elif expected_type == str and not isinstance(value, str):
            # Convert to string
            return str(value) if value is not None else self._get_default_value_for_key(key)

        # If it's already the expected type or no conversion needed
        return value