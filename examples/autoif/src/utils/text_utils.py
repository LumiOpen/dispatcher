"""Text formatting utilities for autoif generator task."""

import os
from typing import List, Union


# Language-specific conjunctions for "and"
LANG_CONJUNCTIONS = {
    'en': 'and',      # English
    'fi': 'ja',       # Finnish
    'fr': 'et',       # French
    'de': 'und',      # German
    'es': 'y',        # Spanish
    'it': 'e',        # Italian
    'pt': 'e',        # Portuguese
    'nl': 'en',       # Dutch
    'sv': 'och',      # Swedish
    'da': 'og',       # Danish
    'cs': 'a',        # Czech
    'pl': 'i',        # Polish
    'hu': 'és',       # Hungarian
    'ro': 'și',       # Romanian
    'sk': 'a',        # Slovak
    'sl': 'in',       # Slovenian
    'hr': 'i',        # Croatian
    'bg': 'и',        # Bulgarian
    'et': 'ja',       # Estonian
    'lv': 'un',       # Latvian
    'lt': 'ir',       # Lithuanian
    'mt': 'u',        # Maltese
    'ga': 'agus',     # Irish
    'el': 'και',      # Greek
}


def get_conjunction() -> str:
    """Get the conjunction for the current language from LANGUAGE env variable.
    
    Returns:
        The conjunction word (e.g., 'and' for English, 'ja' for Finnish).
        Defaults to 'and' if language not found or not set.
    """
    lang = os.environ.get('LANGUAGE', 'en').lower().strip()
    return LANG_CONJUNCTIONS.get(lang, 'and')


def format_constraints_with_conjunctions(constraints: Union[str, List[str]]) -> str:
    """Format constraints with proper conjunctions and capitalization.

    Uses language-specific conjunction based on LANGUAGE environment variable.
    Strips trailing punctuation from constraints to allow proper joining.

    Args:
        constraints: Single constraint string or list of constraints

    Returns:
        Formatted constraint text with conjunctions.
        Format: "{Constraint1}, {constraint2}, ... and {constraintN}."

    Examples (English):
        - Single: "Your response should be short."
        - Two: "Your response should be short and format your response as a list."
        - Three+: "Your response should be short, format your response as a list and include examples."

    Examples (Finnish):
        - Two: "Vastauksesi pitäisi olla lyhyt ja muotoile vastauksesi listaksi."
    """
    if isinstance(constraints, str):
        return constraints

    if not constraints:
        return ""

    if len(constraints) == 1:
        return constraints[0]

    conjunction = get_conjunction()

    # Strip trailing punctuation from all constraints
    stripped = [instr.rstrip('.!?;:') for instr in constraints]

    # Lowercase first letter of constraints from second onwards
    formatted = [stripped[0]] + [instr[0].lower() + instr[1:] if instr else instr
                                 for instr in stripped[1:]]

    if len(formatted) == 2:
        return f"{formatted[0]} {conjunction} {formatted[1]}."

    return ", ".join(formatted[:-1]) + f" {conjunction} {formatted[-1]}."