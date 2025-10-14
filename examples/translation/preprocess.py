"""Preprocessing utilities for translation tasks, including FLORES-200 data loading and few-shot prompt construction."""

import json
import os
import random
from typing import List, Tuple, Optional
import logging

from language_names import LANGUAGE_NAMES

logger = logging.getLogger(__name__)

# Base path for FLORES-200 dataset for building few-shot samples
FLORES_200_BASE_PATH = "/scratch/project_462000353/posttraining_data/FLORES-200"

def load_flores_data(lang_code: str, split: str = "dev") -> List[str]:
    """
    Load FLORES-200 data for a specific language.
    
    Args:
        lang_code: Language code (e.g., 'eng', 'deu', 'fra')
        split: Dataset split ('dev', 'devtest', etc.)
    
    Returns:
        List of text strings from the dataset
    """
    # lang_code should be a 3-letter code
    if len(lang_code) == 2:
        lang_code = LANGUAGE_NAMES.get(lang_code, ["", lang_code])[1]
    file_path = os.path.join(FLORES_200_BASE_PATH, f"{lang_code}-{split}.txt")
    
    if not os.path.exists(file_path):
        logger.warning(f"FLORES-200 file not found: {file_path}")
        return []
    
    texts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Check if it's a JSONL format
                    try:
                        data = json.loads(line)
                        if 'text' in data:
                            texts.append(data['text'])
                        else:
                            # If no 'text' field, treat the whole JSON as text
                            texts.append(str(data))
                    except json.JSONDecodeError:
                        # If not JSON, treat as plain text
                        texts.append(line)
        
        logger.info(f"Loaded {len(texts)} examples from {file_path}")
        return texts
        
    except Exception as e:
        logger.error(f"Error loading FLORES-200 data from {file_path}: {e}")
        return []

def get_few_shot_examples(source_lang: str, target_lang: str, n_shots: int = 5) -> List[Tuple[str, str]]:
    """
    Get few-shot examples from FLORES-200 dataset.
    
    Args:
        source_lang: Source language code (e.g., 'eng')
        target_lang: Target language code (e.g., 'deu', 'fra')
        n_shots: Number of few-shot examples to retrieve
    
    Returns:
        List of (source_text, target_text) tuples
    """
    source_texts = load_flores_data(source_lang)
    target_texts = load_flores_data(target_lang)
    
    if not source_texts or not target_texts:
        logger.warning(f"Could not load FLORES-200 data for {source_lang}->{target_lang}")
        return []
    
    # Ensure we have aligned data
    min_length = min(len(source_texts), len(target_texts))
    if min_length == 0:
        logger.warning(f"No aligned examples found for {source_lang}->{target_lang}")
        return []
    
    # Sample n_shots examples randomly
    available_indices = list(range(min_length))
    if len(available_indices) < n_shots:
        logger.warning(f"Only {len(available_indices)} examples available, requested {n_shots}")
        n_shots = len(available_indices)
    
    selected_indices = random.sample(available_indices, n_shots)
    
    examples = []
    for idx in selected_indices:
        source_text = source_texts[idx].strip()
        target_text = target_texts[idx].strip()
        if source_text and target_text:
            examples.append((source_text, target_text))
    
    logger.info(f"Selected {len(examples)} few-shot examples for {source_lang}->{target_lang}")
    return examples

def build_few_shot_prompt(
    target_lang_name: str,
    few_shot_examples: List[Tuple[str, str]],
    current_text: str
) -> str:
    """
    Build a few-shot prompt.
    
    This format is designed for base models (not chat models) and uses a simple
    pattern that the model can learn to continue. 

    General purpose template: "## {src_lang}: {src}\n## {trg_lang}: {trg}"
    
    Args:
        target_lang_name: Full name of target language (e.g., "German", "French")
        few_shot_examples: List of (source_text, target_text) tuples
        current_text: The text to be translated
    
    Returns:
        Complete few-shot prompt ready for the model
    """
    prompt_parts = []
    
    # Add few-shot examples
    for source_text, target_text in few_shot_examples:
        prompt_parts.append(f"## English: {source_text}")
        prompt_parts.append(f"## {target_lang_name}: {target_text}")
        prompt_parts.append("") # to have double new-lines between shots

    # Add the current text to translate
    prompt_parts.append(f"## English: {current_text}")
    prompt_parts.append(f"## {target_lang_name}: ")

    return "\n".join(prompt_parts)

def split_text_into_lines(text: str) -> Tuple[List[str], List[str]]:
    """
    Split text into lines while preserving the structure information.
    
    Args:
        text: Input text to split
        
    Returns:
        Tuple of (lines_content, structure_info) where:
        - lines_content: List of non-empty lines
        - structure_info: List indicating what comes after each line 
          (e.g., '\n', '\n\n', etc.)
    """
    lines = []
    structure = []
    
    # Split by lines but keep track of empty lines and their positions
    raw_lines = text.split('\n')
    
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i].strip()
        
        if line:  # Non-empty line
            lines.append(line)
            
            # Count consecutive newlines after this line
            newline_count = 1  # At least one newline
            j = i + 1
            while j < len(raw_lines) and not raw_lines[j].strip():
                newline_count += 1
                j += 1
            
            structure.append('\n' * newline_count)
            i = j
        else:
            i += 1
    
    # Handle case where text doesn't end with newline
    if structure and not text.endswith('\n'):
        structure[-1] = structure[-1].rstrip('\n')
    
    return lines, structure

def preprocess_for_few_shot_translation(
    text: str,
    source_lang_code: str,
    target_lang_code: str,
    target_lang_name: str,
    n_shots: int = 5
) -> Tuple[List[str], List[str], Optional[List[Tuple[str, str]]]]:
    """
    Preprocess text for line-by-line few-shot translation.
    
    Args:
        text: Full text to translate
        source_lang_code: Source language code (e.g., 'eng')
        target_lang_code: Target language code (e.g., 'deu')
        target_lang_name: Full name of target language (e.g., "German")
        n_shots: Number of few-shot examples (default: 5)
        
    Returns:
        Tuple of (line_prompts, structure_info, few_shot_examples) where:
        - line_prompts: List of prompts for each line
        - structure_info: List of structure information for reconstruction
        - few_shot_examples: The few-shot examples used (for debugging)
    """
    # Split text into lines and preserve structure
    lines, structure = split_text_into_lines(text)
    
    if not lines:
        logger.warning("No non-empty lines found in input text")
        return [], [], None
    
    # Get few-shot examples from FLORES-200
    examples = get_few_shot_examples(source_lang_code, target_lang_code, n_shots)
    
    if not examples:
        logger.warning(f"No few-shot examples available, falling back to zero-shot")
        # Fallback to simple zero-shot prompts for each line
        line_prompts = [f"English: {line}\n{target_lang_name}:" for line in lines]
        return line_prompts, structure, None
    
    # Build few-shot prompts for each line
    line_prompts = []
    for line in lines:
        prompt = build_few_shot_prompt(target_lang_name, examples, line)
        line_prompts.append(prompt)
    
    logger.info(f"Preprocessed {len(lines)} lines for few-shot translation")
    return line_prompts, structure

