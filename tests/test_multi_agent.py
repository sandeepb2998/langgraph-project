"""
Test module for the multi-agent LangGraph system.

This module contains unit tests for the multi-agent workflow components.
"""

import unittest
from unittest.mock import Mock, patch
import os
import tempfile

# Import the functions to test
from src.multi_agent_langgraph import (
    read_document,
    python_executor,
    create_agent,
    build_workflow,
    AgentState,
)


class TestMultiAgentLangGraph(unittest.TestCase):
    """Test cases for the multi-agent LangGraph system."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, "test_instructions.txt")
        
        # Create a test instruction file
        with open(self.test_file, "w") as f:
            f.write("Test instruction content")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_dir)

    def test_read_document_text_file(self):
        """Test reading a text document."""
        result = read_document(self.test_file)
        self.assertEqual(result, "Test instruction content")

    def test_read_document_nonexistent_file(self):
        """Test reading a non-existent file."""
        result = read_document("nonexistent_file.txt")
        self.assertIn("Failed to read document", result)

    @patch('pandas.read_excel')
    def test_read_document_excel_file(self, mock_read_excel):
        """Test reading an Excel file."""
        # Mock pandas DataFrame
        mock_df = Mock()
        mock_df.to_csv.return_value = "Name,DateOfBirth\nJohn,1990-01-01"
        mock_read_excel.return_value = mock_df
        
        excel_file = os.path.join(self.test_dir, "test.xlsx")
        result = read_document(excel_file)
        
        self.assertEqual(result, "Name,DateOfBirth\nJohn,1990-01-01")
        mock_read_excel.assert_called_once_with(excel_file)

    def test_python_executor_simple_code(self):
        """Test executing simple Python code."""
        result = python_executor("print('Hello, World!')")
        self.assertIn("Hello, World!", result)
        self.assertIn("Successfully executed", result)

    def test_python_executor_error_handling(self):
        """Test error handling in Python executor."""
        result = python_executor("print(undefined_variable)")
        self.assertIn("Failed to execute", result)
        self.assertIn("Error", result)

    def test_agent_state_structure(self):
        """Test AgentState structure."""
        state = AgentState(
            messages=[],
            sender="test_agent"
        )
        self.assertEqual(state["sender"], "test_agent")
        self.assertEqual(state["messages"], [])

    @patch('src.multi_agent_langgraph.ChatOpenAI')
    def test_build_workflow(self, mock_chat_openai):
        """Test workflow building."""
        # Mock the ChatOpenAI class
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        
        # Set up environment to use standard OpenAI
        os.environ.pop("AZURE_OPENAI_ENDPOINT", None)
        os.environ.pop("AZURE_OPENAI_DEPLOYMENT_NAME", None)
        
        workflow = build_workflow()
        self.assertIsNotNone(workflow)


if __name__ == "__main__":
    unittest.main() 