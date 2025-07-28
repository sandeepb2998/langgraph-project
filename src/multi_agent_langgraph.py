"""
multi_agent_langgraph.py
=======================

This module demonstrates how to construct a simple multi‑agent system using
the `langgraph` library.  The goal of the workflow is to take a
specification document (for example, an Excel sheet describing an ETL
pipeline), extract the relevant instructions, use them to generate a
Python script, execute the script, and iteratively repair it until it
runs without errors.  The design of this system follows the patterns
presented in the LangChain/LangGraph documentation for multi‑agent
workflows【975206377901146†L170-L197】 and self‑correcting code agents【641433007168196†L210-L265】.

Overview
--------

There are three core components in this workflow:

1. **InstructionReader agent** – reads the contents of a document (such as
   an Excel file or plain text) and summarises the instructions into a
   clear prompt for the next agent.  It uses a custom `read_document`
   tool to load the file contents.
2. **CodeWriter agent** – receives the prompt from the InstructionReader,
   writes a Python script that implements the described operations and
   executes it.  It uses a `python_executor` tool, based on
   `langchain_experimental.utilities.PythonREPL`, which returns either
   the standard output of the script or an error message if execution
   fails.  The CodeWriter agent is responsible for inspecting the
   execution results and revising the script when necessary.
3. **Tool node** – executes whichever tool an agent has requested via
   function calling.  This logic is adapted from the generic tool node
   example in the LangGraph multi‑agent tutorial【975206377901146†L223-L249】.

Agents exchange messages through a graph‑level state that stores a list of
`BaseMessage` instances and the name of the last sender.  After each
function call the message is appended to the state, and conditional
edges in the graph determine whether to call a tool, move to another
agent, repeat an agent, or finish the workflow.

To run this example you need to set your OpenAI API key in the
environment variable ``OPENAI_API_KEY`` or configure Azure OpenAI
credentials via ``AZURE_OPENAI_API_KEY``, ``AZURE_OPENAI_ENDPOINT``, and
``AZURE_OPENAI_DEPLOYMENT_NAME``.  The code will automatically detect
these variables and connect to Azure OpenAI if available.  The module
itself does not execute any remote calls when imported, so it is safe
to inspect and adapt.  When ready to test, call the ``run_workflow``
function at the bottom of this file with a path to your instruction
document.

Note
----

This module uses the synchronous LangChain APIs for simplicity.  In
production systems you may wish to use asynchronous equivalents.  Also
be aware that executing arbitrary Python code via the PythonREPL tool
can be unsafe; restrict usage to trusted code or run inside a
sandboxed environment.  The Python REPL tool is wrapped with
exception handling to return any error messages instead of raising
exceptions【975206377901146†L179-L191】.
"""

from __future__ import annotations

import json
import os
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
# Import both the standard OpenAI chat model and the Azure OpenAI chat model.
# AzureChatOpenAI allows you to connect to OpenAI models hosted on the
# Microsoft Azure platform.  The class supports specifying a deployment
# name, endpoint URL, and API version via keyword arguments as described in
# the LangChain documentation【442267202818370†L1216-L1226】【442267202818370†L1232-L1255】.
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_experimental.utilities import PythonREPL
from langgraph.graph import END, StateGraph
# Import the prebuilt ToolNode for executing tools in the workflow.  The
# ToolNode replaces the older ToolExecutor/ToolInvocation classes and is
# available in modern versions of LangGraph.  It accepts a list of tools
# and will automatically handle parsing tool calls from AI messages and
# dispatching to the correct tool.  See the LangGraph documentation for
# examples【281937569266432†L579-L590】.
from langgraph.prebuilt import ToolNode


# ---------------------------------------------------------------------------
# Optional: Set Azure OpenAI credentials here for local development or testing.
# It is best practice to set these as environment variables outside the code.
# Uncomment and fill in your values if you want to set them programmatically.
import os
os.environ["AZURE_OPENAI_API_KEY"] = ""
os.environ["AZURE_OPENAI_ENDPOINT"] = ""
os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "gpt-4.1"
os.environ["AZURE_OPENAI_API_VERSION"] = "2025-01-01-preview"  # Optional
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tool definitions
#
# The InstructionReader agent will use `read_document` to load the contents
# of an instruction document.  The CodeWriter agent uses `python_executor`
# to run generated Python code and capture any errors.

@tool
def read_document(file_path: Annotated[str, "Path to the instruction document"]
                  ) -> str:
    """Read a document from disk and return its contents as a string.

    Supported formats: Excel files (.xls, .xlsx) and plain text files (.txt,
    .csv, .md, .json).  If the file is Excel, it will be converted to a
    comma‑separated string using pandas.  For all other extensions the
    contents are returned as UTF‑8 text.  If the file cannot be read,
    an informative error message is returned instead of raising an
    exception.
    """
    import pandas as pd  # imported here to avoid import when unused
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in {".xlsx", ".xls"}:
            df = pd.read_excel(file_path)
            # Convert to CSV without index for readability
            return df.to_csv(index=False)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        # Return errors as a string so the agent can handle them gracefully
        return f"Failed to read document: {e!r}"


# Create a Python REPL instance
repl = PythonREPL()


@tool
def python_executor(code: Annotated[str, "Python code to execute"]
                   ) -> str:
    """Execute arbitrary Python code and return its output or an error message.

    The tool uses LangChain's PythonREPL to run code safely.  Any
    exceptions are caught and returned as part of the string so that
    the CodeWriter agent can use the error information to revise the
    script【975206377901146†L179-L191】.  The result includes the executed code
    for context.
    """
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"


# Package tools into a list for convenience; each agent will be given only
# the tools it needs.
instruction_tools = [read_document]
execution_tools = [python_executor]


# ---------------------------------------------------------------------------
# State definition
#
# The graph state is a mapping with two fields:
#   * messages: a sequence of messages exchanged between agents and tools
#   * sender: the identifier for the last agent that added a message

class AgentState(TypedDict):
    """State passed between nodes in the graph.

    messages: List of BaseMessage instances representing the conversation.
    sender: Name of the last agent that produced a message.  This is used
        for routing the output of the tool node back to the correct agent
        after a function call【975206377901146†L400-L407】.
    """

    messages: Sequence[BaseMessage]
    sender: str


# ---------------------------------------------------------------------------
# Agent creation helpers


def create_agent(
    llm: ChatOpenAI,
    tools: Sequence,
    system_message: str,
) -> any:
    """Create a function‑calling agent with its own system prompt and tools.

    This helper uses a prompt template similar to the one in the LangGraph
    multi‑agent tutorial【975206377901146†L299-L327】.  It instructs the LLM to
    collaborate with other agents, use the provided tools when necessary,
    and signal completion by prefixing its response with ``FINAL ANSWER``.

    Parameters
    ----------
    llm:
        The underlying chat model (e.g., ChatOpenAI) bound to the user's API
        key.
    tools:
        A sequence of LangChain tools available to the agent.  Each tool
        will be converted to a function for the LLM via
        ``convert_to_openai_function``.
    system_message:
        A role‑specific description used to steer the agent's behaviour.

    Returns
    -------
    A runnable object that can be invoked with ``agent.invoke(state)`` and
    returns either a ``FunctionMessage`` (if the model requested to call a
    tool) or a ``HumanMessage`` representing its output.
    """
    """
    Create a tool‑calling agent with its own system prompt and tools.

    Modern LangGraph workflows prefer to use OpenAI's tool calling via
    ``bind_tools`` rather than the older function‑calling interfaces.
    When a chat model is bound to tools, it will return an ``AIMessage``
    with a ``tool_calls`` attribute whenever it decides to invoke a tool.
    The accompanying ``ToolNode`` can then execute the tool calls.

    Parameters
    ----------
    llm : ChatOpenAI
        The underlying chat model (e.g., ChatOpenAI or AzureChatOpenAI).
    tools : Sequence
        A sequence of LangChain tools available to the agent.  Each tool
        should be decorated with ``@tool`` so that LangChain can infer
        its input schema.
    system_message : str
        A role‑specific description used to steer the agent's behaviour.

    Returns
    -------
    Runnable
        A runnable object that can be invoked with ``agent.invoke(state)``
        and returns a ``BaseMessage`` (typically an ``AIMessage``) which
        may include tool calls.
    """
    # Compose the system prompt.  The general instructions come first and
    # include the list of tool names and the custom system message.  A
    # MessagesPlaceholder is used so the conversation history can be
    # supplied from the graph state at runtime.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a helpful AI assistant collaborating with other agents. "
                    "Use the provided tools to progress towards completing your task. "
                    "If you or any of the other assistants have the final answer or deliverable, "
                    "prefix your response with FINAL ANSWER so the team knows to stop. "
                    "You have access to the following tools: {tool_names}.\n{system_message}"
                ),
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    # Partially fill the template with the system message and the tool names
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([t.name for t in tools]))
    # Bind the tools to the chat model.  This returns a runnable model
    # that produces AIMessage objects with tool calls in their
    # ``tool_calls`` attribute whenever a tool should be invoked【401516166768039†L191-L240】.
    model_with_tools = llm.bind_tools(tools)
    return prompt | model_with_tools


def agent_node(state: AgentState, agent: any, name: str) -> dict:
    """Execute an agent and append its message to the state.

    This helper invokes the agent with the current state.  The agent is
    expected to return a ``BaseMessage`` (typically an ``AIMessage``) that
    may include tool calls in its ``tool_calls`` attribute.  The message
    is appended to the list of messages, and the ``sender`` field is
    updated so that the routing logic knows which agent produced it.
    """
    # Invoke the agent on the current state.  The agent is expected to
    # return a BaseMessage (typically an AIMessage) that may include
    # tool calls in its ``tool_calls`` attribute if the model decides
    # to invoke a tool.  We simply append this message to the state's
    # message list and record which agent produced it.
    message = agent.invoke(state)
    return {
        "messages": state["messages"] + [message],  # Append to history
        "sender": name,
    }


# ---------------------------------------------------------------------------
# Build the workflow
#
# The graph consists of three nodes: InstructionReader, CodeWriter, and
# call_tool.  Conditional edges determine where to route messages based on
# whether a tool is being called or the agent has declared a final answer.


def build_workflow() -> any:
    """Assemble the LangGraph workflow for the multi‑agent coding system.

    This helper initialises the chat model, creates the InstructionReader
    and CodeWriter agents, defines the tool execution node, and stitches
    everything together into a stateful LangGraph.  It first attempts
    to construct an Azure OpenAI client if the appropriate environment
    variables are set; otherwise it falls back to the standard OpenAI
    service.

    **Azure configuration**
    
    To use Azure OpenAI instead of the default OpenAI endpoints you must
    export the following environment variables:

    - ``AZURE_OPENAI_API_KEY`` – your Azure OpenAI API key (the class will
      automatically look for this variable as the default key)
    - ``AZURE_OPENAI_ENDPOINT`` – the base endpoint URL for your Azure
      OpenAI resource, e.g. ``https://my-instance.openai.azure.com``
    - ``AZURE_OPENAI_DEPLOYMENT_NAME`` – the name of the deployment
      configured in the Azure portal (e.g. ``gpt-35-turbo`` or
      ``gpt-4``)
    - ``AZURE_OPENAI_API_VERSION`` – optional; the API version to use
      when calling Azure OpenAI.  If unspecified, a sensible default is
      used (currently ``2023-06-01-preview``).

    With these variables set the function will create an ``AzureChatOpenAI``
    instance bound to your deployment.  If they are missing, the code
    falls back to ``ChatOpenAI`` and expects the standard ``OPENAI_API_KEY``
    to be present【442267202818370†L1216-L1226】.
    """
    # Attempt to configure an Azure OpenAI client if environment variables are
    # provided.  This logic allows the same code to work against both the
    # Azure-hosted and OpenAI-hosted models without changes to the
    # downstream agent logic.
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
    azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-06-01-preview")
    if azure_endpoint and azure_deployment:
        # Instantiate the Azure OpenAI chat model.  AzureChatOpenAI
        # automatically reads the API key from AZURE_OPENAI_API_KEY or
        # OPENAI_API_KEY if set.  For more details see the LangChain
        # documentation on AzureChatOpenAI【442267202818370†L1216-L1226】【442267202818370†L1232-L1255】.
        llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=azure_api_version,
            temperature=0,
        )
    else:
        # Default to the standard OpenAI API using the gpt-4o model.  If
        # ``OPENAI_API_KEY`` is not set, fall back to a dummy key.  This
        # prevents initialisation errors when running unit tests that do
        # not hit the API.  The dummy key will only cause failures at
        # runtime if the model is invoked.
        openai_api_key = os.environ.get("OPENAI_API_KEY", "dummy")
        llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)

    # Create the two agents with their respective tools and system messages.
    instruction_reader = create_agent(
        llm,
        instruction_tools,
        system_message=(
            "You are the InstructionReader. Your task is to understand a document "
            "containing instructions for a data processing pipeline. Use the read_document tool "
            "to load the file specified by the user. Summarise the key operations and steps in "
            "the document, and write a concise prompt that the CodeWriter agent can use to implement "
            "the pipeline in Python. Do not generate code yourself – only describe what needs to be done."
        ),
    )
    code_writer = create_agent(
        llm,
        execution_tools,
        system_message=(
            "You are the CodeWriter. Your job is to write Python code that follows the instructions "
            "provided by the InstructionReader. Produce code that can be executed without errors. "
            "Always wrap your code in a call to the python_executor tool by emitting a function call. "
            "After receiving output from python_executor, inspect whether the execution succeeded. "
            "If the tool response contains an error, revise your code using the error information and "
            "call python_executor again. When the script executes successfully and the task is complete, "
            "prefix your final message with FINAL ANSWER and include the code used to accomplish the task."
        ),
    )

    # Partially bind the agent_node helper to create nodes for each agent
    import functools
    instruction_node = functools.partial(agent_node, agent=instruction_reader, name="InstructionReader")
    code_node = functools.partial(agent_node, agent=code_writer, name="CodeWriter")

    # Create a ToolNode to execute tools.  The ToolNode takes care of
    # parsing tool calls from AI messages and dispatching to the correct
    # tool.  It returns the resulting ToolMessages, which we will
    # append to the conversation state.  See LangGraph documentation for
    # details【401516166768039†L191-L240】.
    tool_node_instance = ToolNode(instruction_tools + execution_tools)

    def call_tool_node(state: AgentState) -> dict:
        """Execute any pending tool calls in the conversation.

        The ToolNode expects a state object with a ``messages`` key whose
        last entry is an AIMessage containing a ``tool_calls`` attribute.
        When invoked, it returns either a dictionary with a ``messages``
        field or a list of ToolMessage objects.  We normalise this to
        return only the new messages, letting LangGraph handle list
        concatenation when merging state updates.
        """
        result = tool_node_instance.invoke(state)
        # Normalise the result to a list of messages
        if isinstance(result, dict):
            tool_messages = result.get("messages", [])
        else:
            tool_messages = result
        return {"messages": state["messages"] + tool_messages}  # Append to history

    # Router for the InstructionReader: determine next step based on last message
    def instruction_router(state: AgentState) -> str:
        """Determine the next step for the InstructionReader.

        If the last message contains tool calls (i.e., the LLM wants to
        invoke a tool), route to the tool execution node.  Otherwise,
        hand off to the CodeWriter unless the message signals completion.
        """
        messages = state["messages"]
        last_message = messages[-1]
        # If the InstructionReader is attempting to call a tool, route to call_tool
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "call_tool"
        # If the InstructionReader somehow produces a final answer, end
        if getattr(last_message, "content", None) and "FINAL ANSWER" in last_message.content:
            return "end"
        # Otherwise, hand off to the CodeWriter
        return "CodeWriter"

    # Router for the CodeWriter: determine next step based on last message
    def code_router(state: AgentState) -> str:
        """Determine the next step for the CodeWriter.

        Route to the tool execution node if the model has requested a tool
        call.  If the agent signals completion, end the workflow.  Otherwise,
        continue iterating on the CodeWriter.
        """
        messages = state["messages"]
        last_message = messages[-1]
        # If the CodeWriter wants to call a tool, go to the tool node
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "call_tool"
        # Stop if the agent has delivered the final answer
        if getattr(last_message, "content", None) and "FINAL ANSWER" in last_message.content:
            return "end"
        # Otherwise, continue iterating on the CodeWriter
        return "CodeWriter"

    # Build the graph
    workflow = StateGraph(AgentState)
    workflow.add_node("InstructionReader", instruction_node)
    workflow.add_node("CodeWriter", code_node)
    workflow.add_node("call_tool", call_tool_node)

    # Define conditional edges for the InstructionReader
    workflow.add_conditional_edges(
        "InstructionReader",
        instruction_router,
        {
            "call_tool": "call_tool",
            "CodeWriter": "CodeWriter",
            "end": END,
        },
    )
    # Define conditional edges for the CodeWriter
    workflow.add_conditional_edges(
        "CodeWriter",
        code_router,
        {
            "call_tool": "call_tool",
            "CodeWriter": "CodeWriter",
            "end": END,
        },
    )
    # Route back from tool to whichever agent invoked it using the sender field
    workflow.add_conditional_edges(
        "call_tool",
        lambda s: s["sender"],
        {
            "InstructionReader": "InstructionReader",
            "CodeWriter": "CodeWriter",
        },
    )
    # Set the starting node
    workflow.set_entry_point("InstructionReader")
    return workflow.compile()


def run_workflow(document_path: str, max_steps: int = 50) -> None:
    workflow = build_workflow()
    initial_state = {
        "messages": [HumanMessage(content=document_path)],
        "sender": "user",
    }
    for step in workflow.stream(initial_state, {"recursion_limit": max_steps}):
        print("Step:", step)
        print("---")
        # Also print any tool execution output or errors
        messages = step.get(list(step.keys())[0], {}).get("messages", [])
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, "content") and last_message.content:
                print("Message content:\n", last_message.content)
            if hasattr(last_message, "name") and last_message.name == "python_executor":
                print("\n[python_executor output]:\n", last_message.content)
            if hasattr(last_message, "content") and "FINAL ANSWER" in last_message.content:
                print("\nFinal Answer:\n", last_message.content)
                break


if __name__ == "__main__":
    # Example usage: specify your own file here.  Ensure that you have
    # exported OPENAI_API_KEY in your environment before running.
    import argparse
    parser = argparse.ArgumentParser(description="Run the multi‑agent ETL pipeline generator")
    parser.add_argument("file", type=str, help="Path to the instruction document")
    parser.add_argument("--max_steps", type=int, default=50, help="Maximum number of iterations")
    args = parser.parse_args()
    run_workflow(args.file, max_steps=args.max_steps) 