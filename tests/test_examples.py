import unittest
import os
import subprocess
import sys
from pathlib import Path


class TestExamples(unittest.TestCase):
    """Tests that verify all example scripts run without errors."""
    
    def setUp(self):
        # Get the root directory of the project
        self.root_dir = Path(__file__).parent.parent
        self.examples_dir = self.root_dir / "examples"
        
        # Skip the use_cases/cfb_w_lib.py test as it requires external API access
        self.skip_examples = [
            "use_cases/cfb_w_lib.py"
        ]
    
    def test_example_scripts(self):
        """Test that all example scripts run without errors."""
        # Get all Python files in the examples directory
        example_files = []
        for root, _, files in os.walk(self.examples_dir):
            for file in files:
                if file.endswith(".py") and file != "__init__.py":
                    rel_path = os.path.relpath(os.path.join(root, file), self.examples_dir)
                    if rel_path not in self.skip_examples:
                        example_files.append(rel_path)
        
        # Make sure we found some example files
        self.assertGreater(len(example_files), 0, "No example files found")
        
        # Run each example script and check for errors
        for example_file in example_files:
            with self.subTest(example=example_file):
                script_path = os.path.join(self.examples_dir, example_file)
                
                # Run the script with a timeout to prevent hanging
                try:
                    result = subprocess.run(
                        [sys.executable, script_path],
                        capture_output=True,
                        text=True,
                        timeout=10  # 10 second timeout
                    )
                    
                    # Check if the script ran successfully
                    self.assertEqual(
                        result.returncode, 
                        0, 
                        f"Example {example_file} failed with error:\n{result.stderr}"
                    )
                except subprocess.TimeoutExpired:
                    self.fail(f"Example {example_file} timed out after 10 seconds")
    
    def test_individual_examples(self):
        """Test each example script individually with specific assertions."""
        # Test sample_bout.py
        self._test_specific_example("sample_bout.py", expected_output_contains=["Starting ratings:", "After matches"])
        
        # Test prediction.py
        self._test_specific_example("prediction.py", expected_output_contains=["probability of better beating good"])
        
        # Test bout_with_ties.py
        self._test_specific_example("bout_with_ties.py", expected_output_contains=["Starting ratings:", "After matches with ties"])
        
        # Test sample_arena.py
        self._test_specific_example("sample_arena.py", expected_output_contains=["Arena results"])
        
        # Test dwz_arena.py
        self._test_specific_example("dwz_arena.py", expected_output_contains=["Arena results"])
        
        # Test ecf_arena.py
        self._test_specific_example("ecf_arena.py", expected_output_contains=["Arena results"])
        
        # Test glicko_arena.py
        self._test_specific_example("glicko_arena.py", expected_output_contains=["Arena results"])
        
        # Test persist_state_arena.py
        self._test_specific_example("persist_state_arena.py", expected_output_contains=["Arena results"])
        
        # Test bout_with_initialization.py
        self._test_specific_example("bout_with_initialization.py", expected_output_contains=["Starting ratings:", "After matches"])
    
    def _test_specific_example(self, example_file, expected_output_contains):
        """Helper method to test a specific example with expected output."""
        script_path = os.path.join(self.examples_dir, example_file)
        
        # Run the script
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Check if the script ran successfully
            self.assertEqual(
                result.returncode, 
                0, 
                f"Example {example_file} failed with error:\n{result.stderr}"
            )
            
            # Check if the output contains expected strings
            for expected in expected_output_contains:
                self.assertIn(
                    expected, 
                    result.stdout, 
                    f"Example {example_file} output did not contain '{expected}'"
                )
        except subprocess.TimeoutExpired:
            self.fail(f"Example {example_file} timed out after 10 seconds")


if __name__ == '__main__':
    unittest.main() 