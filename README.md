# Multi-Agent LangGraph Project

A sophisticated multi-agent system built with LangGraph that demonstrates how to create intelligent workflows for document processing and code generation.

## Overview

This project implements a multi-agent system with two specialized agents:

1. **InstructionReader Agent** - Reads and interprets instruction documents
2. **CodeWriter Agent** - Generates and executes Python code based on instructions

The system uses LangGraph to orchestrate communication between agents and handle tool execution in a stateful workflow.

## Features

- **Document Processing**: Supports Excel files (.xlsx, .xls) and text files (.txt, .csv, .md, .json)
- **Code Generation**: Automatically generates Python code from natural language instructions
- **Error Handling**: Self-correcting code generation with iterative refinement
- **Multi-Agent Coordination**: Seamless communication between specialized agents
- **Tool Integration**: Built-in tools for document reading and code execution

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sandeepb2998/langgraph-project.git
cd langgraph-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

### API Keys Setup

The system supports both OpenAI and Azure OpenAI endpoints. Configure your API keys as environment variables:

**For OpenAI API:**
```bash
export OPENAI_API_KEY=your-openai-api-key-here
```

**For Azure OpenAI:**
```bash
export AZURE_OPENAI_API_KEY=your-azure-api-key
export AZURE_OPENAI_ENDPOINT=your-azure-endpoint-url
export AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
export AZURE_OPENAI_API_VERSION=2023-06-01-preview
```

## Usage

### Basic Usage

Run the multi-agent workflow with an instruction file:

```bash
python src/multi_agent_langgraph.py instructions.txt
```

### Example Workflow

1. **Create an instruction file** (e.g., `instructions.txt`):
```
The dataset is located at sample_data.xlsx and contains the columns Name and DateOfBirth.
Your task is to create a new column called Age that calculates each person's age in years from their date of birth. Save the result to sample_data_withage.xlsx.
```

2. **Run the workflow**:
```bash
python src/multi_agent_langgraph.py instructions.txt
```

3. **The system will**:
   - Read and interpret the instruction document
   - Generate Python code to process the Excel file
   - Execute the code and handle any errors
   - Iteratively refine the code until successful
   - Save the processed data to a new file

## Project Structure

```
langgraph-project/
├── src/
│   └── multi_agent_langgraph.py    # Main multi-agent system
├── tests/
│   └── test_multi_agent.py         # Unit tests
├── requirements.txt                 # Python dependencies
├── instructions.txt                # Example instruction file
├── sample_data.xlsx                # Sample data file
└── README.md                       # This file
```

## Architecture

### Agent System

- **InstructionReader**: Specialized in document interpretation and instruction parsing
- **CodeWriter**: Focused on code generation, execution, and error correction
- **Tool Node**: Handles tool execution and message routing

### State Management

The system uses a stateful graph with:
- `messages`: Conversation history between agents
- `sender`: Tracks which agent last produced a message

### Workflow Flow

1. User provides instruction document path
2. InstructionReader reads and interprets the document
3. CodeWriter receives instructions and generates code
4. Code is executed via python_executor tool
5. If errors occur, CodeWriter refines the code
6. Process continues until successful completion

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

Or run individual tests:

```bash
python tests/test_multi_agent.py
```

## Dependencies

- **langchain-core**: Core LangChain functionality
- **langchain-openai**: OpenAI integration
- **langchain-experimental**: Experimental features including PythonREPL
- **langgraph**: Multi-agent workflow orchestration
- **pandas**: Data processing
- **openpyxl**: Excel file handling

## Security Notes

- API keys are handled securely via environment variables
- Code execution is sandboxed using LangChain's PythonREPL
- No sensitive data is hardcoded in the source code

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For issues and questions, please open an issue on the GitHub repository. 