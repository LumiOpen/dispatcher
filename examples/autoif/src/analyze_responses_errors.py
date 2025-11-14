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
        # Track instruction counts per sample for error vs success analysis
        self.error_samples_instruction_counts = []  # List of instruction counts for samples with errors
        self.success_samples_instruction_counts = []  # List of instruction counts for successful samples
    
    def count_instructions_in_sample(self, instructions: List[List[str]]) -> int:
        """Count the total number of instructions in a sample (sum of all elements in all sublists)."""
        total_count = 0
        for instruction_list in instructions:
            total_count += len(instruction_list)
        return total_count
    
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
        
        # Count instructions in this error sample
        instruction_count = self.count_instructions_in_sample(instructions)
        self.error_samples_instruction_counts.append(instruction_count)
        
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
        
        # Count instructions in this successful sample
        instruction_count = self.count_instructions_in_sample(instructions)
        self.success_samples_instruction_counts.append(instruction_count)
        
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
    
    def process_file(self, filepath: str, analyze_n: int = None) -> None:
        """Process the entire JSONL file or up to analyze_n samples."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                samples_processed = 0
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check if we've reached the limit
                    if analyze_n is not None and samples_processed >= analyze_n:
                        print(f"Reached limit of {analyze_n} samples, stopping processing.", file=sys.stderr)
                        break
                    
                    try:
                        record = json.loads(line)
                        
                        if '__ERROR__' in record:
                            self.process_error_record(record)
                            samples_processed += 1
                        elif 'instruction_ids' in record:
                            self.process_success_record(record)
                            samples_processed += 1
                        
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
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics needed for the summary."""
        results = self.get_results()
        
        total_instructions = len(results)
        total_errors = sum(r['total_errors'] for r in results)
        total_successes = sum(r['succeeded'] for r in results)
        
        # Calculate overall error rate
        overall_error_rate = None
        if total_errors + total_successes > 0:
            overall_error_rate = round((total_errors / (total_errors + total_successes)) * 100, 1)
        
        # Calculate instruction count statistics for error samples
        error_sample_count = len(self.error_samples_instruction_counts)
        avg_error_instructions = 0
        if error_sample_count > 0:
            avg_error_instructions = round(sum(self.error_samples_instruction_counts) / error_sample_count, 2)
        
        # Calculate instruction count statistics for successful samples
        success_sample_count = len(self.success_samples_instruction_counts)
        avg_success_instructions = 0
        if success_sample_count > 0:
            avg_success_instructions = round(sum(self.success_samples_instruction_counts) / success_sample_count, 2)
        
        # Calculate average instruction difference
        avg_instructions_difference = None
        if error_sample_count > 0 and success_sample_count > 0:
            avg_instructions_difference = round(avg_error_instructions - avg_success_instructions, 2)
        
        return {
            "overall_statistics": {
                "total_unique_instructions": total_instructions,
                "total_errors": total_errors,
                "total_successes": total_successes,
                "overall_error_rate_percent": overall_error_rate
            },
            "instruction_count_analysis": {
                "error_samples": {
                    "sample_count": error_sample_count,
                    "avg_instructions_per_sample": avg_error_instructions
                },
                "successful_samples": {
                    "sample_count": success_sample_count,
                    "avg_instructions_per_sample": avg_success_instructions
                },
                "comparison": {
                    "avg_instructions_difference": avg_instructions_difference
                }
            }
        }
    
    def generate_summary_text(self) -> str:
        """Generate summary text for both printing and file output."""
        statistics = self.get_statistics()
        
        if not statistics["overall_statistics"]["total_unique_instructions"]:
            return "No data found to analyze.\n"
        
        overall = statistics["overall_statistics"]
        instruction_analysis = statistics["instruction_count_analysis"]
        
        lines = []
        lines.append("=== Analysis Summary ===")
        lines.append(f"Total unique instructions: {overall['total_unique_instructions']}")
        lines.append(f"Total errors: {overall['total_errors']}")
        lines.append(f"Total successes: {overall['total_successes']}")
        
        if overall["overall_error_rate_percent"] is not None:
            lines.append(f"Overall error rate: {overall['overall_error_rate_percent']}%")
        else:
            lines.append(f"Overall error rate: N/A (no data)")
        
        # Instruction count analysis
        lines.append("")
        lines.append("=== Instruction Count Analysis ===")
        error_stats = instruction_analysis["error_samples"]
        success_stats = instruction_analysis["successful_samples"]
        
        if error_stats["sample_count"] > 0:
            lines.append(f"Error samples: {error_stats['sample_count']} samples, avg {error_stats['avg_instructions_per_sample']:.2f} instructions per sample")
        else:
            lines.append(f"Error samples: 0 samples")
        
        if success_stats["sample_count"] > 0:
            lines.append(f"Successful samples: {success_stats['sample_count']} samples, avg {success_stats['avg_instructions_per_sample']:.2f} instructions per sample")
        else:
            lines.append(f"Successful samples: 0 samples")
        
        # Comparison if both types have data
        comparison = instruction_analysis["comparison"]
        if comparison["avg_instructions_difference"] is not None:
            diff = comparison["avg_instructions_difference"]
            if diff > 0:
                lines.append(f"Error samples have on average {diff:.2f} more instructions than successful samples")
            elif diff < 0:
                lines.append(f"Error samples have on average {abs(diff):.2f} fewer instructions than successful samples")
            else:
                lines.append(f"Error and successful samples have the same average number of instructions")
        
        return "\n".join(lines) + "\n"
    
    def write_summary_txt(self, output_file: str) -> None:
        """Write summary statistics to text format."""
        summary_text = self.generate_summary_text()
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary_text)
    
    def print_summary(self) -> None:
        """Print a summary of the analysis."""
        summary_text = self.generate_summary_text()
        # Print to stderr with a leading newline for formatting
        print(f"\n{summary_text.rstrip()}", file=sys.stderr)
        
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
    
    def generate_summary_text(self) -> str:
        """Generate summary text for both printing and file output."""
        results = self.get_results()
        
        if not results:
            return "No category data found to analyze.\n"
        
        total_categories = len(results)
        total_errors = sum(r['total_errors'] for r in results)
        total_successes = sum(r['succeeded'] for r in results)
        
        lines = []
        lines.append("=== Category Analysis Summary ===")
        lines.append(f"Total categories: {total_categories}")
        lines.append(f"Total errors: {total_errors}")
        lines.append(f"Total successes: {total_successes}")
        
        if total_errors + total_successes > 0:
            lines.append(f"Overall error rate: {total_errors/(total_errors+total_successes)*100:.1f}%")
        else:
            lines.append(f"Overall error rate: N/A (no data)")
        
        return "\n".join(lines) + "\n"
    
    def print_summary(self) -> None:
        """Print a summary of the category analysis."""
        summary_text = self.generate_summary_text()
        print(f"\n{summary_text.rstrip()}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze JSONL file to create instruction and category analysis files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.jsonl                    # Creates instruction_analysis.csv, categories_analysis.csv, and summary.txt
  %(prog)s input.jsonl --analyze-n 1000   # Process only the first 1000 samples

The script dynamically creates columns for each error type encountered in the data.
Three output files will be created in the same directory as the input file:
- instruction_analysis.csv: Analysis by individual instructions
- categories_analysis.csv: Analysis by instruction categories
- summary.txt: Summary statistics including instruction count analysis
        """
    )
    
    parser.add_argument(
        'input_file',
        help='Input JSONL file to analyze'
    )
    
    parser.add_argument(
        '--analyze-n',
        type=int,
        default=None,
        help='Number of samples to process from the beginning of the file (default: process all samples)'
    )
    
    args = parser.parse_args()
    
    # Get the directory of the input file
    input_dir = os.path.dirname(os.path.abspath(args.input_file))
    instruction_output_file = os.path.join(input_dir, 'instruction_analysis.csv')
    categories_output_file = os.path.join(input_dir, 'categories_analysis.csv')
    summary_output_file = os.path.join(input_dir, 'summary.txt')
    
    # Analyze instructions
    analyzer = InstructionAnalyzer()
    analyzer.process_file(args.input_file, analyze_n=args.analyze_n)
    analyzer.print_summary()
    analyzer.write_csv(instruction_output_file)
    analyzer.write_summary_txt(summary_output_file)
    print(f"Instruction analysis written to: {instruction_output_file}", file=sys.stderr)
    
    # Analyze categories
    category_analyzer = CategoryAnalyzer(analyzer)
    category_analyzer.print_summary()
    category_analyzer.write_csv(categories_output_file)
    
    # Append category summary to summary.txt
    with open(summary_output_file, 'a', encoding='utf-8') as f:
        f.write("\n")
        f.write(category_analyzer.generate_summary_text())
    
    print(f"Summary written to: {summary_output_file}", file=sys.stderr)
    print(f"Category analysis written to: {categories_output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
