"""Plan-and-Execute agent: creates a research plan, then executes each step."""
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from utils.llm import llm
from utils import parse_json_response
from tools.search_tool import search_news
from tools.summarize_tool import summarize_text
from tools.report_tool import generate_report
from memory.session import Session, session_store

_plan_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个研究规划专家。分析以下研究主题，生成一个分步执行计划。

返回JSON格式：
{{
    "sub_topics": ["子主题1", "子主题2", ...],
    "steps": ["步骤1: 搜索关键词X", "步骤2: 总结Y", "步骤3: 生成报告"]
}}

每个子主题将独立搜索，确保全面覆盖。"""),
    ("human", "研究主题：{topic}"),
])


class PlanAndExecuteAgent:
    """Plan-and-Execute: plan first, then execute deterministically."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.results: list[dict[str, Any]] = []

    def plan(self) -> dict[str, Any]:
        """Analyze the topic and return a structured execution plan."""
        system, user = _plan_prompt.format_messages(topic=self.session.topic)
        resp = llm.ask(str(system.content), str(user.content))
        try:
            return parse_json_response(resp)
        except Exception:
            return {"sub_topics": [self.session.topic], "steps": ["搜索并总结"]}

    def execute(self) -> str:
        """Execute the plan: search each sub-topic, summarize, then generate report."""
        plan = self.plan()
        sub_topics: list[str] = plan.get("sub_topics", [self.session.topic])

        all_results: list[dict[str, Any]] = []
        summaries: list[str] = []

        for topic in sub_topics:
            if results := search_news.invoke({"query": topic}):
                all_results.extend(results)
                text = "\n\n".join(
                    f"{r['title']}\n{r.get('snippet', '')}" for r in results[:10]
                )
                summaries.append(summarize_text.invoke({"text": text}))

        self.results = all_results
        self.session.search_results = all_results
        session_store.save(self.session)

        report = generate_report.invoke({"topic": self.session.topic, "materials": all_results})
        self.session.report = str(report)
        session_store.save(self.session)
        return str(report)


plan_execute_agent = PlanAndExecuteAgent
