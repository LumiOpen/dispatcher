"""Error handling utilities for autoif generator task."""

from typing import Optional


def format_error_type_with_turn(error_type: str, turn: Optional[int] = None) -> str:
    """
    Formats error type with turn information if turn is provided.

    Args:
        error_type: Base error type
        turn: Optional turn number (0-indexed)

    Returns:
        Formatted error type with turn prefix if turn is provided
    """
    if turn is not None:
        return f"turn{turn + 1}_{error_type}"
    return error_type