"""Handler class for scoring functionality in autoif generator task."""

import os
import re
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from dispatcher.taskmanager.task.base import TaskFailed
from .error_utils import format_error_type_with_turn
from .text_utils import format_instructions_with_conjunctions

@dataclass
class ScoringData:
    """Internal data class for scoring data for a specific turn."""
    is_no_followup: bool
    instruction_ids: List
    instructions: List
    queries: List[str]
    all_responses: List[str]


class ScoringHandler:
    """Handler class for performing response scoring in autoif generator task.

    This class encapsulates all the data and logic needed for response scoring.
    """

    def __init__(self,
                 turn_idx: int,
                 is_no_followup: bool,
                 instruction_ids: List[List],
                 instructions: List[List],
                 queries: List[str],
                 all_responses: List[str]):
        """Initialize ScoringHandler.

        Args:
            turn_idx: Current turn index
            is_no_followup: Whether this is a no_followup case
            instruction_ids: Instruction IDs per turn
            instructions: Instructions per turn
            queries: List of queries
            all_responses: All responses so far
        """
        self.turn_id = turn_idx

        # Store score threshold
        self.score_threshold = int(os.environ.get("SCORE_THRESHOLD", "4"))

        # Prepare scoring data for current turn
        self.data = ScoringData(
            is_no_followup=is_no_followup,
            instruction_ids=instruction_ids[turn_idx] if turn_idx < len(instruction_ids) else [],
            instructions=instructions[turn_idx] if turn_idx < len(instructions) else [],
            queries=queries,
            all_responses=all_responses
        )

    def construct_scoring_messages(self, response_text: str) -> List[Dict[str, str]]:
        """Construct scoring messages based on the case type.

        Args:
            response_text: The response to score

        Returns:
            List of messages for scoring
        """
        if self.data.is_no_followup and self.turn_id > 0:
            # For no_followup case after first turn, use rephrase scoring
            return self._construct_rephrase_scoring_messages(response_text)
        else:
            # Regular scoring case
            return self._construct_regular_scoring_messages(response_text)

    def extract_and_check_score(self, scoring_text: str) -> int:
        """Extract score from scoring text and check if it meets threshold.

        Args:
            scoring_text: The scoring response text

        Returns:
            The extracted score

        Raises:
            TaskFailed: If score is below threshold
        """
        score = self._extract_score(scoring_text)

        # Check if score meets threshold
        if score < self.score_threshold:
            raise TaskFailed(
                message=f"Score {score} at turn {self.turn_id + 1} is below threshold {self.score_threshold}. Scoring response: <response>{scoring_text}</response>",
                error_type=format_error_type_with_turn("score_below_threshold", self.turn_id)
            )

        return score

    def _construct_regular_scoring_messages(self, response_text: str) -> List[Dict[str, str]]:
        """Construct scoring messages for regular scoring case."""
        # Load the scoring prompt
        scoring_prompt = open("model_prompts/scoring_prompt.txt").read().strip()
        scoring_prompt = scoring_prompt.format(
            instructions=self.data.instructions,
            query=self.data.queries[self.turn_id] if self.turn_id < len(self.data.queries) else '',
            response=response_text
        )

        return [{"role": "user", "content": scoring_prompt}]

    def _construct_rephrase_scoring_messages(self, response_text: str) -> List[Dict[str, str]]:
        """Construct scoring messages for no_followup rephrase scoring."""
        # Load the rephrase scoring prompt
        with open("model_prompts/scoring_rephrase_prompt.txt", "r") as f:
            scoring_prompt = f.read().strip()

        # Accumulate all constraints up to current turn
        accumulated_constraints = []
        # Get all instructions from all turns up to current turn
        for i in range(self.turn_id + 1):
            # We need to reconstruct the full instructions data
            # Since we only have current turn data, we'll use what's available
            if i == self.turn_id:
                accumulated_constraints.extend(self.data.instructions)

        # Format the prompt
        scoring_prompt = scoring_prompt.format(
            query=self.data.queries[0],
            previous_turn_response=self.data.all_responses[0],
            current_response=response_text,
            instructions=self.format_instructions_with_conjunctions(accumulated_constraints)
        )

        return [{"role": "user", "content": scoring_prompt}]

    def _extract_score(self, scored_text: str) -> int:
        """Extract the score from the scored text.

        Args:
            scored_text: The text containing the score

        Returns:
            The extracted score

        Raises:
            TaskFailed: If score cannot be extracted
        """
        # Try multiple patterns to extract the score at the end of text, ordered by specificity
        patterns = [
            # Handle markdown bold formatting at end: **Score: 5**
            r'\*\*Score:\s*(\d+)\*\*\s*$',
            # Handle markdown code formatting at end: `Score: 5`
            r'`Score:\s*(\d+)`\s*$',
            # Handle parenthetical score at end: (Score 5) or (Score: 5)
            r'\(Score\s*:?\s*(\d+)\)\s*$',
            # Keep the original pattern for cases where it works
            r'Score:\s*(\d+)\s*$',
            # Handle Score followed by word boundary at end (fallback)
            r'Score\s*:?\s*(\d+)\s*$',
        ]

        for pattern in patterns:
            # Search case-insensitively
            match = re.search(pattern, scored_text, re.IGNORECASE)
            if match:
                try:
                    score = int(match.group(1))
                    return score
                except (ValueError, IndexError):
                    continue

        # If no pattern matched, raise TaskFailed
        raise TaskFailed(
            message=f"Score not found in the scoring response: {scored_text}",
            error_type=format_error_type_with_turn("score_extraction_failed", self.turn_id)
        )
