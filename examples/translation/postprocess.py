"""Postprocessing utilities for translation task."""

import re
from typing import List


def reconstruct_translated_text(translations: List[str], structure_info: List[str]) -> str:
    """
    Reconstruct the full translated text from line translations and structure info.
    
    Args:
        translations: List of translated lines
        structure_info: List of structure information (newlines, spacing)
        
    Returns:
        Reconstructed full text with original structure preserved
    """
    if not translations:
        return ""
    
    if len(translations) != len(structure_info):
        # Fallback: just join with single newlines
        return '\n'.join(translations)
    
    result = []
    for i, (translation, structure) in enumerate(zip(translations, structure_info)):
        result.append(translation)
        if i < len(translations) - 1:  # Don't add structure after the last line
            result.append(structure)
        elif structure and not structure.startswith('\n'):
            # If the last structure doesn't start with newline, it means
            # the original text didn't end with newline
            pass
        else:
            # Add any trailing structure
            result.append(structure)
    
    return ''.join(result)