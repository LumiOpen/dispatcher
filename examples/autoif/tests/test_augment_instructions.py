import unittest
import json
import os
import sys
import tempfile
import shutil
import subprocess
from unittest.mock import patch, MagicMock

# Add src directory to path for importing modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Mock the huggingface_hub module
mock_module = MagicMock()
mock_client = MagicMock()
# Configure the mock to always return English
mock_client.return_value.post.return_value = {"detected_language": "en"}
mock_module.InferenceClient = mock_client

# Add the mock to sys.modules before the real import happens
sys.modules['huggingface_hub'] = mock_module

# Now we can safely import our module
from src.process_instructions_output import process_output
from src.create_instructions_input import create_input_file

class TestCreateInstructionsInput(unittest.TestCase):
    """Test the create_instructions_input.py script."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.seed_file = os.path.join(self.test_dir, "seed_instructions.txt")
        self.output_file = os.path.join(self.test_dir, "output.jsonl")
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
        
    def test_basic_functionality(self):
        """Test the script produces correct output format and includes seed instructions."""
        # Create sample seed instructions
        seed_instructions = [
            "Format the response as a bullet list.",
            "Include exactly 3 examples in your answer.",
            "Start each paragraph with a different letter of the alphabet."
        ]
        
        with open(self.seed_file, "w") as f:
            f.write("\n".join(seed_instructions))
        
        # Run the script with 2 prompts
        num_prompts = 2
        create_input_file(self.seed_file, self.output_file, num_prompts)
        
        # Verify output file exists
        self.assertTrue(os.path.exists(self.output_file), "Output file wasn't created")
        
        # Check output format and content
        with open(self.output_file, 'r') as f:
            lines = f.readlines()
        
        # Check correct number of prompts were generated
        self.assertEqual(len(lines), num_prompts, 
                         f"Expected {num_prompts} lines, got {len(lines)}")
        
        # Check each prompt
        for line in lines:
            data = json.loads(line)
            
            # Check format has only "prompt" key
            self.assertEqual(list(data.keys()), ["prompt"], 
                            f"Expected only 'prompt' key, got {list(data.keys())}")
            
            # Check all seed instructions are included in the prompt
            for instruction in seed_instructions:
                self.assertIn(instruction, data["prompt"], 
                             f"Instruction '{instruction}' not found in prompt")
    
    def test_num_prompts_parameter(self):
        """Test that the num_prompts parameter creates the correct number of outputs."""
        # Create a minimal seed file
        with open(self.seed_file, "w") as f:
            f.write("Test instruction.")
        
        # Test with different numbers of prompts
        for num_prompts in [1, 3, 5]:
            test_output = os.path.join(self.test_dir, f"output_{num_prompts}.jsonl")
            
            create_input_file(self.seed_file, test_output, num_prompts)
            
            # Verify correct number of prompts
            with open(test_output, 'r') as f:
                lines = f.readlines()
                
            self.assertEqual(len(lines), num_prompts, 
                            f"Expected {num_prompts} prompts, got {len(lines)}")

    def test_empty_seed_file(self):
        """Test handling of empty seed file."""
        # Create empty seed file
        with open(self.seed_file, "w") as f:
            pass  # Empty file
            
        # Run script
        create_input_file(self.seed_file, self.output_file, 1)
        
        # Verify output file exists and contains a prompt despite empty seed
        with open(self.output_file, 'r') as f:
            lines = f.readlines()
            
        self.assertEqual(len(lines), 1, "Expected 1 line for 1 prompt")
        data = json.loads(lines[0])
        self.assertIn("prompt", data)
        # The prompt should still contain the template text even without examples
        self.assertIn("Please provide 50 different instructions", data["prompt"])


class TestProcessInstructionsOutput(unittest.TestCase):
    """Test the process_instructions_output.py functionality."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.seed_file = os.path.join(self.test_dir, "seed.txt")
        self.input_file = os.path.join(self.test_dir, "input.jsonl")
        self.output_file = os.path.join(self.test_dir, "output.txt")
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_numbered_format(self):
        """Test processing of numbered instruction format."""
        # Create seed file
        with open(self.seed_file, "w") as f:
            f.write("Original instruction 1.\n")
            f.write("Original instruction 2.\n")
        
        # Create input file with numbered instructions
        with open(self.input_file, "w") as f:
            f.write(json.dumps({
                "prompt": "Generate instructions",
                "responses": ["1. New instruction one.\n2. New instruction two.\n3. New instruction three."]
            }) + "\n")
        
        # Process the file
        process_output(self.input_file, self.output_file, self.seed_file, language="en")
        
        # Verify output
        with open(self.output_file, 'r') as f:
            lines = f.read().splitlines()
        
        self.assertEqual(len(lines), 5)  # 2 original + 3 new
        self.assertIn("Original instruction 1.", lines)
        self.assertIn("Original instruction 2.", lines)
        self.assertIn("New instruction one.", lines)
        self.assertIn("New instruction two.", lines)
        self.assertIn("New instruction three.", lines)
    
    def test_unnumbered_format(self):
        """Test processing of unnumbered instruction format."""        
        # Create seed file
        with open(self.seed_file, "w") as f:
            f.write("Original seed instruction.\n")
        
        # Create input file with unnumbered instructions
        with open(self.input_file, "w") as f:
            f.write(json.dumps({
                "prompt": "Generate instructions",
                "responses": ["First plain instruction.\nSecond plain instruction.\nThird plain instruction."]
            }) + "\n")
        
        # Process the file
        process_output(self.input_file, self.output_file, self.seed_file, language="en")
        
        # Verify output
        with open(self.output_file, 'r') as f:
            lines = f.read().splitlines()
        
        # Should have 1 original + 3 new instructions
        self.assertEqual(len(lines), 4, f"Expected 4 lines, got {len(lines)}")
        
        # Check for required instructions
        self.assertIn("Original seed instruction.", lines)
        self.assertIn("First plain instruction.", lines)
        self.assertIn("Second plain instruction.", lines)
        self.assertIn("Third plain instruction.", lines)
    
    def test_duplicate_handling(self):
        """Test handling of duplicate instructions."""        
        # Create seed file with one instruction
        with open(self.seed_file, "w") as f:
            f.write("Duplicate instruction.\n")
        
        # Create input with duplicates
        with open(self.input_file, "w") as f:
            f.write(json.dumps({
                "prompt": "Generate instructions",
                "responses": ["Duplicate instruction.\nNew instruction.\nAnother new instruction."]
            }) + "\n")
        
        # Process the file
        process_output(self.input_file, self.output_file, self.seed_file, language="en")
        
        # Verify output - should have no duplicates
        with open(self.output_file, 'r') as f:
            lines = f.read().splitlines()
        
        # Should have 3 unique instructions
        self.assertEqual(len(lines), 3, f"Expected 3 unique lines, got {len(lines)}")
        
        # Count occurrences of each line
        counts = {}
        for line in lines:
            counts[line] = counts.get(line, 0) + 1
        
        # Check each line appears exactly once
        for line, count in counts.items():
            self.assertEqual(count, 1, f"Line '{line}' appears {count} times")
    
if __name__ == "__main__":
    unittest.main()