"""Tool registry — maps tool names to LangChain BaseTool instances.

Each @tool-decorated function is a BaseTool with .name, .invoke(), and
built-in JSON schema (derived from type hints + docstring).
"""
from langchain_core.tools import BaseTool

from tools.search_tool import search_news
from tools.summarize_tool import summarize_text
from tools.email_tool import send_email
from tools.report_tool import generate_report
from tools.qa_chain import ask_research

_tools: list[BaseTool] = [
    search_news,
    summarize_text,
    send_email,
    generate_report,
    ask_research,
]

_tool_map: dict[str, BaseTool] = {t.name: t for t in _tools}


def execute_tool(name: str, **kwargs) -> object:
    """Execute a registered tool by name."""
    if tool := _tool_map.get(name):
        return tool.invoke(kwargs)
    return {"error": f"Unknown tool: {name}"}


def list_tools() -> list[BaseTool]:
    """Return all LangChain tools for bind_tools()."""
    return _tools
