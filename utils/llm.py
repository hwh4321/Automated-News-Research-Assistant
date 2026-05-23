"""LLM client wrapper — ChatOpenAI with tool-calling and ReAct loop."""
import json
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage,
)
from langchain_core.tools import BaseTool

from config import config


class LLMClient:
    """ChatOpenAI wrapper with ReAct-style tool-calling loop."""

    def __init__(self) -> None:
        self._base_llm = ChatOpenAI(
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
            model=config.llm_model,
            extra_body={"thinking": {"type": "disabled"}},
        )

    def ask(self, system: str, user: str, *, temperature: float = 0.3) -> str:
        """Single-turn chat with no tools."""
        llm = self.with_temp(temperature)
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return str(resp.content)

    def call_tools(
        self,
        messages: list[BaseMessage],
        tools: list[BaseTool],
        *,
        max_rounds: int = 5,
    ) -> str:
        """ReAct loop: call tools until the model produces a final text response."""
        llm = self._base_llm.bind_tools(tools)
        tool_map = {t.name: t for t in tools}

        for _ in range(max_rounds):
            resp = llm.invoke(messages)
            if not resp.tool_calls:
                return str(resp.content)

            messages.append(resp)
            for tc in resp.tool_calls:
                args = tc.get("args", {})
                result = tool_map[tc["name"]].invoke(args)
                messages.append(ToolMessage(
                    content=json.dumps(result, ensure_ascii=False),
                    tool_call_id=tc["id"],
                ))

        return "Max tool-calling rounds exceeded."

    def with_temp(self, temperature: float) -> ChatOpenAI:
        return ChatOpenAI(
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
            model=config.llm_model,
            temperature=temperature,
            extra_body={"thinking": {"type": "disabled"}},
        )


llm = LLMClient()
