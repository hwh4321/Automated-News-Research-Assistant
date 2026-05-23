"""ReAct (Reasoning + Acting) agent — LLM thinks, calls tools, observes, repeats."""
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage,
)

from utils.llm import llm
from tools.registry import list_tools
from memory.session import Session, session_store

REACT_SYSTEM = """你是一个新闻调研助手，拥有搜索、总结和生成报告的能力。

处理流程：
1. 分析用户请求，确定需要搜索的内容
2. 调用 search_news 搜索相关新闻
3. 调用 summarize_text 对结果进行总结
4. 调用 generate_report 生成最终研究报告
5. 如果用户要求发送邮件，调用 send_email

始终使用中文回复。每次只调用一个工具，观察结果后再决定下一步。"""

_ROLE_MAP: dict[str, type[BaseMessage]] = {
    "system": SystemMessage, "human": HumanMessage,
    "ai": AIMessage, "tool": ToolMessage,
}


def _serialize(msg: BaseMessage) -> dict:
    role = type(msg).__name__.replace("Message", "").lower()
    result: dict = {"role": role, "content": str(msg.content)}
    if hasattr(msg, "additional_kwargs") and msg.additional_kwargs:
        result["additional_kwargs"] = msg.additional_kwargs
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        result["tool_calls"] = msg.tool_calls
    if hasattr(msg, "name") and msg.name:
        result["name"] = msg.name
    if role == "tool" and hasattr(msg, "tool_call_id"):
        result["tool_call_id"] = msg.tool_call_id
    return result


def _deserialize(raw: dict) -> BaseMessage:
    cls = _ROLE_MAP.get(raw.get("role", ""), AIMessage)
    kwargs: dict = {"content": raw.get("content", "")}
    if "additional_kwargs" in raw:
        kwargs["additional_kwargs"] = raw["additional_kwargs"]
    if "tool_calls" in raw:
        kwargs["tool_calls"] = raw["tool_calls"]
    if "name" in raw:
        kwargs["name"] = raw["name"]
    if raw.get("role") == "tool" and "tool_call_id" in raw:
        kwargs["tool_call_id"] = raw["tool_call_id"]
    return cls(**kwargs)


class ReActAgent:
    """ReAct-pattern agent with session management."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def run(self, instruction: str) -> str:
        """Execute the ReAct loop for a single instruction."""
        messages: list[BaseMessage] = [SystemMessage(content=REACT_SYSTEM)]
        if self.session.messages:
            messages.extend(_deserialize(m) for m in self.session.messages)
        messages.append(HumanMessage(content=instruction))

        result = llm.call_tools(messages, list_tools())

        self.session.messages = [_serialize(m) for m in messages]
        self.session.messages.append({"role": "ai", "content": result})
        session_store.save(self.session)
        return result

    def run_research(self) -> str:
        """Run the full research pipeline for the session's topic."""
        return self.run(
            f"请对以下主题进行新闻调研：{self.session.topic}\n搜索相关新闻 → 总结要点 → 生成研究报告"
        )


react_agent = ReActAgent
