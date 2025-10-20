"""Text formatting utilities for autoif generator task."""

from typing import List, Union


def format_instructions_with_conjunctions(instructions: Union[str, List[str]]) -> str:
    """Format instructions with proper conjunctions and capitalization.

    Args:
        instructions: Single instruction string or list of instructions

    Returns:
        Formatted instruction text with conjunctions

    Examples:
        - Single: "Your response should..."
        - Two: "Your response should... and format your response..."
        - Three+: "Your response should..., you should... and include..."
    """
    if isinstance(instructions, str):
        return instructions

    if not instructions:
        return ""

    if len(instructions) == 1:
        return instructions[0]

    # Lowercase first letter of instructions from second onwards
    formatted = [instructions[0]] + [instr[0].lower() + instr[1:] if instr else instr
                                   for instr in instructions[1:]]

    if len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"

    return ", ".join(formatted[:-1]) + f" and {formatted[-1]}"