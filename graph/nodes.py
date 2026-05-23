"""Graph nodes — each function is a node in the LangGraph state machine."""
import json
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from graph.state import ResearchState
from tools.search_tool import search_news
from tools.summarize_tool import summarize_text
from tools.report_tool import generate_report
from tools.email_tool import send_email
from utils.llm import llm
from utils import parse_json_response

_planner_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个研究规划专家。根据用户的研究主题，将其拆解为3-5个具体的搜索子任务。
每个子任务应该是一个具体的关键词或问题，用于搜索引擎查询。

返回格式：纯JSON数组，每个元素是一个搜索查询字符串。
示例：["OpenAI最新融资新闻", "AI行业2026年投资趋势", "OpenAI竞争对手动态"]"""),
    ("human", "研究主题：{topic}"),
])


def plan_node(state: ResearchState) -> dict[str, Any]:
    """Plan-and-Execute: break the research topic into search sub-queries."""
    system, user = _planner_prompt.format_messages(topic=state["topic"])
    try:
        plan = parse_json_response(llm.ask(str(system.content), str(user.content)))
    except (json.JSONDecodeError, TypeError):
        plan = [state["topic"]]
    return {"plan": plan, "current_step": "plan", "search_results": []}


def search_node(state: ResearchState) -> dict[str, Any]:
    """Execute searches for all planned sub-queries."""
    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    for query in state.get("plan", [state["topic"]]):
        for item in search_news.invoke({"query": query}):
            if key := item.get("url") or item.get("title"):
                if key not in seen:
                    seen.add(key)
                    results.append(item)

    text = json.dumps(results, ensure_ascii=False)
    from memory.vector_store import vector_store
    try:
        vector_store.add(
            docs=[text],
            metadatas=[{"topic": state["topic"], "session": state["session_id"]}],
            ids=[f"{state['session_id']}-search"],
        )
    except Exception:
        pass

    from memory.relational_store import db
    db.save_query(state["session_id"], json.dumps(state.get("plan", [])), results)
    return {"search_results": results, "current_step": "search"}


def summarize_node(state: ResearchState) -> dict[str, Any]:
    """Summarize the search results."""
    if not (results := state["search_results"]):
        return {"summaries": ["没有找到相关新闻。"]}

    texts = [f"{i+1}. {r['title']}\n{r.get('snippet', '')}" for i, r in enumerate(results[:15])]
    summary = summarize_text.invoke({"text": "\n\n".join(texts)})
    return {"summaries": [str(summary)], "current_step": "summarize"}


def report_node(state: ResearchState) -> dict[str, Any]:
    """Generate the final research report."""
    report = generate_report.invoke({
        "topic": state["topic"],
        "materials": state["search_results"],
    })

    from memory.relational_store import db
    db.save_report(state["session_id"], str(report))
    db.update_session_status(state["session_id"], "completed")

    return {"report": str(report), "current_step": "report", "done": True}


def email_node(state: ResearchState) -> dict[str, Any]:
    """Send the report via email."""
    if not state.get("recipient_email"):
        return {"error": "未指定收件人邮箱"}
    send_email.invoke({
        "to": state["recipient_email"],
        "subject": f"研究报告：{state['topic']}",
        "body": state["report"],
    })
    return {"current_step": "email"}
