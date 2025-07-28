from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

@tool
def read_document(file_path: str) -> str:
    """Read a document and return its contents."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Failed to read document: {e}"

def build_workflow():
    """Build the multi-agent workflow."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    workflow = StateGraph()
    return workflow.compile()
