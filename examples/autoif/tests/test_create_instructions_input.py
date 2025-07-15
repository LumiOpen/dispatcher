import json
import tempfile
import os
import sys
import unittest
from unittest.mock import patch

# Add src directory to path for importing modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# Import the function to test
from src.create_instructions_input import create_input_file, main


class TestCreateInstructionsInput(unittest.TestCase):
    
    def test_create_input_file_success(self):
        """Test successful creation of input file with valid seed file."""
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as seed_file:
            seed_file.write("Start your response with 'Answer:'\nEnd your response with a period\n")
            seed_file_path = seed_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as output_file:
            output_file_path = output_file.name
        
        try:
            # Call the function
            create_input_file(seed_file_path, output_file_path, 50)
            
            # Verify output file was created and has correct content
            self.assertTrue(os.path.exists(output_file_path), "Output file was not created")
            
            with open(output_file_path, 'r') as f:
                data = json.load(f)
            
            self.assertIn('prompt', data, "Prompt key not found in output")
            self.assertIn('50 different instructions', data['prompt'], "Number of instructions not found in prompt")
            self.assertIn("Start your response with 'Answer:'", data['prompt'], "Seed content not found in prompt")
            self.assertIn("End your response with a period", data['prompt'], "Seed content not found in prompt")
            
        finally:
            # Clean up
            os.unlink(seed_file_path)
            os.unlink(output_file_path)
    
    def test_create_input_file_missing_seed_file(self):
        """Test handling of missing seed file."""
        non_existent_file = "non_existent_seed.txt"
        output_file = "test_output.json"
        
        with self.assertRaises(SystemExit) as cm:
            create_input_file(non_existent_file, output_file)
        
        # Check that exit code is 1
        self.assertEqual(cm.exception.code, 1, f"Expected exit code 1, got {cm.exception.code}")
    
    def test_create_input_file_default_num_instructions(self):
        """Test that default number of instructions (100) is used correctly."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as seed_file:
            seed_file.write("Test instruction\n")
            seed_file_path = seed_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as output_file:
            output_file_path = output_file.name
        
        try:
            # Call with default num_instructions
            create_input_file(seed_file_path, output_file_path)
            
            with open(output_file_path, 'r') as f:
                data = json.load(f)
            
            self.assertIn('100 different instructions', data['prompt'], "Default number of instructions not found")
            
        finally:
            os.unlink(seed_file_path)
            os.unlink(output_file_path)
    
    def test_prompt_content_structure(self):
        """Test that the generated prompt has the correct structure and content."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as seed_file:
            seed_file.write("Format as JSON\nUse bullet points\n")
            seed_file_path = seed_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as output_file:
            output_file_path = output_file.name
        
        try:
            create_input_file(seed_file_path, output_file_path, 25)
            
            with open(output_file_path, 'r') as f:
                data = json.load(f)
            
            prompt = data['prompt']
            
            # Check key elements are in the prompt
            self.assertIn("You are an expert for writing instructions", prompt, "Expert instruction not found")
            self.assertIn("Instructions are about the format but not style", prompt, "Format instruction not found")
            self.assertIn("evaluated by a Python function", prompt, "Python function reference not found")
            self.assertIn("Do not generate instructions about writing style", prompt, "Style restriction not found")
            self.assertIn("Format as JSON", prompt, "Seed content 'Format as JSON' not found")
            self.assertIn("Use bullet points", prompt, "Seed content 'Use bullet points' not found")
            self.assertIn("Please generate one instruction per line", prompt, "Line instruction not found")
            
        finally:
            os.unlink(seed_file_path)
            os.unlink(output_file_path)
    
    def test_main_function(self):
        """Test the main function with mocked arguments."""
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as seed_file:
            seed_file.write("Test seed instruction\n")
            seed_file_path = seed_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as output_file:
            output_file_path = output_file.name
        
        # Mock the parsed arguments
        mock_args = type('Args', (), {
            'seed_file': seed_file_path,
            'output_file': output_file_path,
            'num_instructions': 10
        })()
        
        try:
            with patch('argparse.ArgumentParser.parse_args', return_value=mock_args):
                # Capture stdout
                import io
                from contextlib import redirect_stdout
                
                captured_output = io.StringIO()
                with redirect_stdout(captured_output):
                    main()
                
                # Verify file was created
                self.assertTrue(os.path.exists(output_file_path), "Output file was not created by main function")
                
                # Check success message
                output = captured_output.getvalue()
                self.assertIn("Created input file with a prompt to generate 10 instructions", output, "Success message not found")
                
        finally:
            os.unlink(seed_file_path)
            os.unlink(output_file_path)
    
    def test_json_output_format(self):
        """Test that the output is valid JSON format."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as seed_file:
            seed_file.write("Simple instruction\n")
            seed_file_path = seed_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as output_file:
            output_file_path = output_file.name
        
        try:
            create_input_file(seed_file_path, output_file_path, 5)
            
            # Test that the file can be loaded as valid JSON
            with open(output_file_path, 'r') as f:
                data = json.load(f)  # This will raise exception if invalid JSON
            
            # Verify structure
            self.assertIsInstance(data, dict, f"Expected dict, got {type(data)}")
            self.assertEqual(len(data), 1, f"Expected 1 key, got {len(data)}")
            self.assertIn('prompt', data, "Prompt key not found")
            self.assertIsInstance(data['prompt'], str, f"Expected string prompt, got {type(data['prompt'])}")
            
        finally:
            os.unlink(seed_file_path)
            os.unlink(output_file_path)


if __name__ == "__main__":
    unittest.main()