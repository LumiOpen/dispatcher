#!/usr/bin/env python3
"""
Analyze JSONL file to create a dictionary of instruction IDs with their failure and success counts.
Handles dynamic error types and special parsing for verification failures.
"""

import json
import sys
import csv
import argparse
import re
import os
from collections import defaultdict
from typing import Dict, List, Set, Any


class InstructionAnalyzer:
    def __init__(self):
        # instruction_id -> {instruction: str, instruction_categories: list, error_counts: {error_type: count}, succeeded: count}
        self.instruction_data = defaultdict(lambda: {
            'instruction': '',
            'instruction_categories': [],
            'error_counts': defaultdict(int),
            'succeeded': 0
        })
        self.all_error_types = set()  # Track all encountered error types
    
    def extract_turn_from_error(self, error_type: str) -> int:
        """Extract turn number from error type string."""
        match = re.match(r'turn(\d+)_', error_type)
        if match:
            return int(match.group(1)) - 1  # Convert to 0-based index
        return -1  # Invalid turn
    
    def parse_verification_failed_ids(self, error_message: str) -> List[str]:
        """Parse instruction IDs from verification failed error messages."""
        # For single instruction: "The response did not pass verification for instruction 34 with accuracy..."
        single_match = re.search(r'for instruction (\d+) with accuracy', error_message)
        if single_match:
            return [single_match.group(1)]
        
        # For multiple instructions: "The response did not pass verification for instructions 12, 3. <response>..."
        multi_match = re.search(r'for instructions ([0-9, ]+)\.', error_message)
        if multi_match:
            ids_str = multi_match.group(1)
            return [id_str.strip() for id_str in ids_str.split(',')]
        
        return []
    
    def get_instruction_text(self, inst_id: str, instructions_lists: List[List[str]], 
                           instruction_ids_lists: List[List[str]], 
                           instruction_categories_lists: List[List[str]] = None) -> tuple:
        """Find instruction text and categories for a given instruction ID across all turns."""
        for turn_idx, (ids_list, instr_list) in enumerate(zip(instruction_ids_lists, instructions_lists)):
            if inst_id in ids_list:
                idx = ids_list.index(inst_id)
                if idx < len(instr_list):
                    instruction_text = instr_list[idx]
                    categories = []
                    if (instruction_categories_lists and 
                        turn_idx < len(instruction_categories_lists) and
                        idx < len(instruction_categories_lists[turn_idx])):
                        categories = instruction_categories_lists[turn_idx][idx]
                    return instruction_text, categories
        return '', []
    
    def process_error_record(self, record: Dict[str, Any]) -> None:
        """Process an error record (__ERROR__ key present)."""
        error_data = record.get('__ERROR__', {})
        task_data = error_data.get('task_data', {})
        error_type = error_data.get('error', '')
        error_message = error_data.get('message', '')  # Full error message for parsing
        
        instruction_ids = task_data.get('instruction_ids', [])
        instructions = task_data.get('instructions', [])
        instruction_categories = task_data.get('instruction_categories', [])
        
        # Track this error type
        self.all_error_types.add(error_type)
        
        # Handle special verification failed cases
        if 'instruction_verification_failed' in error_type or 'multiple_instructions_verification_failed' in error_type:
            failed_ids = self.parse_verification_failed_ids(error_message)
            for inst_id in failed_ids:
                instruction_text, categories = self.get_instruction_text(inst_id, instructions, instruction_ids, instruction_categories)
                self.instruction_data[inst_id]['instruction'] = instruction_text
                self.instruction_data[inst_id]['instruction_categories'] = categories
                self.instruction_data[inst_id]['error_counts'][error_type] += 1
            return
        
        # Handle regular turn-based errors
        turn_idx = self.extract_turn_from_error(error_type)
        if turn_idx >= 0 and turn_idx < len(instruction_ids) and turn_idx < len(instructions):
            if len(instruction_ids[turn_idx]) == len(instructions[turn_idx]):
                # Process each instruction in the specified turn
                for i, (inst_id, instruction) in enumerate(zip(instruction_ids[turn_idx], instructions[turn_idx])):
                    categories = []
                    if (turn_idx < len(instruction_categories) and 
                        i < len(instruction_categories[turn_idx])):
                        categories = instruction_categories[turn_idx][i]
                    
                    self.instruction_data[inst_id]['instruction'] = instruction
                    self.instruction_data[inst_id]['instruction_categories'] = categories
                    self.instruction_data[inst_id]['error_counts'][error_type] += 1
        else:
            original_content_str = error_data.get('original_content', '')
            if original_content_str:
                try:
                    original_content = json.loads(original_content_str)
                    inst_ids = original_content.get('instruction_ids', [])
                    instructions = original_content.get('instructions', [])
                    categories_lists = original_content.get('instruction_categories', [])
                    
                    # Apply to turn 0 instructions only
                    if (len(inst_ids) > 0 and len(instructions) > 0 and
                        len(inst_ids[0]) == len(instructions[0])):
                        
                        for i, (inst_id, instruction) in enumerate(zip(inst_ids[0], instructions[0])):
                            categories = []
                            if len(categories_lists) > 0 and i < len(categories_lists[0]):
                                categories = categories_lists[0][i]
                            
                            self.instruction_data[inst_id]['instruction'] = instruction
                            self.instruction_data[inst_id]['instruction_categories'] = categories
                            self.instruction_data[inst_id]['error_counts'][error_type] += 1
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse original_content JSON: {e}", file=sys.stderr)

    def process_success_record(self, record: Dict[str, Any]) -> None:
        """Process a success record (instruction_ids key present)."""
        instruction_ids = record.get('instruction_ids', [])
        instructions = record.get('instructions', [])
        instruction_categories = record.get('instruction_categories', [])
        
        # Process all turns - count each instruction occurrence across all sublists
        for turn_idx in range(len(instruction_ids)):
            if (turn_idx < len(instructions) and
                len(instruction_ids[turn_idx]) == len(instructions[turn_idx])):
                
                # Process each instruction in this turn
                for i, (inst_id, instruction) in enumerate(zip(instruction_ids[turn_idx], instructions[turn_idx])):
                    categories = []
                    if (turn_idx < len(instruction_categories) and 
                        i < len(instruction_categories[turn_idx])):
                        categories = instruction_categories[turn_idx][i]
                    
                    self.instruction_data[inst_id]['instruction'] = instruction
                    self.instruction_data[inst_id]['instruction_categories'] = categories
                    self.instruction_data[inst_id]['succeeded'] += 1
    
    def process_file(self, filepath: str) -> None:
        """Process the entire JSONL file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        record = json.loads(line)
                        
                        if '__ERROR__' in record:
                            self.process_error_record(record)
                        elif 'instruction_ids' in record:
                            self.process_success_record(record)
                        
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse line {line_num}: {e}", file=sys.stderr)
                        continue
                        
        except FileNotFoundError:
            print(f"Error: File '{filepath}' not found", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error reading file '{filepath}': {e}", file=sys.stderr)
            sys.exit(1)
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get the analysis results as a list of dictionaries."""
        results = []
        
        # Get all error types sorted for consistent column ordering
        sorted_error_types = sorted(self.all_error_types)
        
        for inst_id, data in self.instruction_data.items():
            result = {
                'instruction_id': inst_id,
                'instruction': data['instruction'],
                'succeeded': data['succeeded'],
                'instruction_categories': str(data['instruction_categories'])
            }
            
            # Add columns for each error type
            total_errors = 0
            for error_type in sorted_error_types:
                count = data['error_counts'][error_type]
                result[error_type] = count
                total_errors += count
            
            result['total_errors'] = total_errors
            result['total'] = total_errors + data['succeeded']
            
            # Calculate error percentage
            if result['total'] > 0:
                result['total_errors_perc'] = (total_errors / result['total']) * 100
            else:
                result['total_errors_perc'] = 0.0
            
            results.append(result)
        
        # Sort by error percentage descending, then by total errors descending, then by instruction_id for consistency
        results.sort(key=lambda x: (-x['total_errors_perc'], -x['total_errors'], x['instruction_id']))
        
        return results
    
    def write_csv(self, output_file: str) -> None:
        """Write results to CSV format."""
        results = self.get_results()
        
        if not results:
            print("No data to write.", file=sys.stderr)
            return
        
        # Dynamic fieldnames based on encountered error types
        sorted_error_types = sorted(self.all_error_types)
        fieldnames = ['instruction_id', 'instruction', 'instruction_categories', 'total_errors', 'total_errors_perc', 'succeeded', 'total'] + sorted_error_types
        
        with open(output_file, 'w', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    
    def print_summary(self) -> None:
        """Print a summary of the analysis."""
        results = self.get_results()
        
        if not results:
            print("No data found to analyze.", file=sys.stderr)
            return
        
        total_instructions = len(results)
        total_errors = sum(r['total_errors'] for r in results)
        total_successes = sum(r['succeeded'] for r in results)
        
        print(f"\n=== Analysis Summary ===", file=sys.stderr)
        print(f"Total unique instructions: {total_instructions}", file=sys.stderr)
        print(f"Total errors: {total_errors}", file=sys.stderr)
        print(f"Total successes: {total_successes}", file=sys.stderr)
        
        if total_errors + total_successes > 0:
            print(f"Overall error rate: {total_errors/(total_errors+total_successes)*100:.1f}%", file=sys.stderr)
        else:
            print(f"Overall error rate: N/A (no data)", file=sys.stderr)
        
        # # Show breakdown by error type
        # if self.all_error_types:
        #     print(f"\nError types encountered:", file=sys.stderr)
        #     error_totals = defaultdict(int)
        #     for result in results:
        #         for error_type in self.all_error_types:
        #             error_totals[error_type] += result.get(error_type, 0)
            
        #         for error_type in sorted(self.all_error_types):
        #             print(f"  {error_type}: {error_totals[error_type]}", file=sys.stderr)


class CategoryAnalyzer:
    def __init__(self, instruction_analyzer: InstructionAnalyzer):
        self.instruction_analyzer = instruction_analyzer
        # category -> {error_counts: {error_type: count}, succeeded: count}
        self.category_data = defaultdict(lambda: {
            'error_counts': defaultdict(int),
            'succeeded': 0
        })
    
    def analyze_categories(self) -> None:
        """Analyze error data by instruction categories."""
        for inst_id, data in self.instruction_analyzer.instruction_data.items():
            categories = data.get('instruction_categories', [])
            
            # Ensure categories is always a list
            if isinstance(categories, str):
                categories = [categories]
            elif not categories:
                categories = ['uncategorized']
            
            # Each instruction can have multiple categories
            for category in categories:
                # Add successes
                self.category_data[category]['succeeded'] += data['succeeded']
                
                # Add errors
                for error_type, count in data['error_counts'].items():
                    self.category_data[category]['error_counts'][error_type] += count
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get the category analysis results as a list of dictionaries."""
        self.analyze_categories()
        results = []
        
        # Get all error types sorted for consistent column ordering
        sorted_error_types = sorted(self.instruction_analyzer.all_error_types)
        
        for category, data in self.category_data.items():
            result = {
                'category': category,
                'succeeded': data['succeeded']
            }
            
            # Add columns for each error type
            total_errors = 0
            for error_type in sorted_error_types:
                count = data['error_counts'][error_type]
                result[error_type] = count
                total_errors += count
            
            result['total_errors'] = total_errors
            result['total'] = total_errors + data['succeeded']
            
            # Calculate error percentage
            if result['total'] > 0:
                result['total_errors_perc'] = (total_errors / result['total']) * 100
            else:
                result['total_errors_perc'] = 0.0
            
            results.append(result)
        
        # Sort by error percentage descending, then by total errors descending, then by category for consistency
        results.sort(key=lambda x: (-x['total_errors_perc'], -x['total_errors'], x['category']))
        
        return results
    
    def write_csv(self, output_file: str) -> None:
        """Write category analysis results to CSV format."""
        results = self.get_results()
        
        if not results:
            print("No category data to write.", file=sys.stderr)
            return
        
        # Dynamic fieldnames based on encountered error types
        sorted_error_types = sorted(self.instruction_analyzer.all_error_types)
        fieldnames = ['category', 'total_errors', 'total_errors_perc', 'succeeded', 'total'] + sorted_error_types
        
        with open(output_file, 'w', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    
    def print_summary(self) -> None:
        """Print a summary of the category analysis."""
        results = self.get_results()
        
        if not results:
            print("No category data found to analyze.", file=sys.stderr)
            return
        
        total_categories = len(results)
        total_errors = sum(r['total_errors'] for r in results)
        total_successes = sum(r['succeeded'] for r in results)
        
        print(f"\n=== Category Analysis Summary ===", file=sys.stderr)
        print(f"Total categories: {total_categories}", file=sys.stderr)
        print(f"Total errors: {total_errors}", file=sys.stderr)
        print(f"Total successes: {total_successes}", file=sys.stderr)
        
        if total_errors + total_successes > 0:
            print(f"Overall error rate: {total_errors/(total_errors+total_successes)*100:.1f}%", file=sys.stderr)
        else:
            print(f"Overall error rate: N/A (no data)", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze JSONL file to create instruction and category analysis files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.jsonl                    # Creates instruction_analysis.csv and categories_analysis.csv

The script dynamically creates columns for each error type encountered in the data.
Two output files will be created in the same directory as the input file:
- instruction_analysis.csv: Analysis by individual instructions
- categories_analysis.csv: Analysis by instruction categories
        """
    )
    
    parser.add_argument(
        'input_file',
        help='Input JSONL file to analyze'
    )
    
    args = parser.parse_args()
    
    # Get the directory of the input file
    input_dir = os.path.dirname(os.path.abspath(args.input_file))
    instruction_output_file = os.path.join(input_dir, 'instruction_analysis.csv')
    categories_output_file = os.path.join(input_dir, 'categories_analysis.csv')
    
    # Analyze instructions
    analyzer = InstructionAnalyzer()
    analyzer.process_file(args.input_file)
    analyzer.print_summary()
    analyzer.write_csv(instruction_output_file)
    print(f"Instruction analysis written to: {instruction_output_file}", file=sys.stderr)
    
    # Analyze categories
    category_analyzer = CategoryAnalyzer(analyzer)
    category_analyzer.print_summary()
    category_analyzer.write_csv(categories_output_file)
    print(f"Category analysis written to: {categories_output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
