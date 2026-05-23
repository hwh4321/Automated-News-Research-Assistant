"""Auto News Research Assistant — CLI entry point.

Usage:
    python main.py run "AI行业最新动态"                    # LangGraph workflow
    python main.py react "OpenAI新产品发布" --email a@b.com  # ReAct agent
    python main.py plan "量子计算突破"                       # Plan-and-Execute agent
    python main.py ask "OpenAI最近有什么新动作"              # RAG Q&A over research history
"""
import argparse
import uuid

from graph.workflow import research_workflow
from agents.react_agent import ReActAgent
from agents.planner import PlanAndExecuteAgent
from memory.session import Session, session_store
from memory.relational_store import db
from memory.vector_store import vector_store
from tools.email_tool import send_email
from tools.qa_chain import ask_research

SEP = "=" * 60


def _make_session(topic: str) -> tuple[str, Session]:
    sid = str(uuid.uuid4())[:8]
    db.create_session(sid, topic)
    session = Session(id=sid, topic=topic)
    session_store.save(session)
    return sid, session


def _print_banner(sid: str, topic: str, mode: str) -> None:
    print(f"\n{SEP}\n  Research Session: {sid}\n  Topic: {topic}\n  Mode: {mode}\n{SEP}\n")


def _print_result(result: str) -> None:
    print(f"\n{SEP}\n{result}\n{SEP}\n")


def cmd_run(topic: str, email: str = "") -> None:
    """Run the LangGraph research workflow."""
    sid, _ = _make_session(topic)
    mode = "LangGraph Workflow (Plan → Search → Summarize → Report"
    mode += " → Email)" if email else ")"

    initial_state = dict(
        topic=topic, session_id=sid, recipient_email=email,
        plan=[], search_results=[], summaries=[], report="",
        current_step="start", error="", done=False,
    )

    _print_banner(sid, topic, mode)
    final = research_workflow.invoke(initial_state)
    _print_result(final.get("report", "No report generated."))
    if error := final.get("error"):
        print(f"[!] Error: {error}")


def cmd_react(topic: str, email: str = "") -> None:
    """Run the ReAct-pattern agent."""
    sid, session = _make_session(topic)
    _print_banner(sid, topic, "ReAct Agent")
    result = ReActAgent(session).run_research()
    _print_result(result)
    if email and result:
        send_email.invoke({"to": email, "subject": f"研究报告：{topic}", "body": result})


def cmd_plan(topic: str, email: str = "") -> None:
    """Run the Plan-and-Execute agent."""
    sid, session = _make_session(topic)
    _print_banner(sid, topic, "Plan-and-Execute Agent")
    report = PlanAndExecuteAgent(session).execute()
    _print_result(report)
    if email and report:
        send_email.invoke({"to": email, "subject": f"研究报告：{topic}", "body": report})


def cmd_ask(question: str) -> None:
    """RAG-style Q&A over research history."""
    print(f"\n{'='*60}\n  Querying research history...\n{'='*60}\n")
    answer = ask_research.invoke({"question": question})
    _print_result(str(answer))


def cmd_stats() -> None:
    """Show memory/store statistics."""
    print(f"\n  Vector store docs: {vector_store.count()}")
    print("  Sessions in DB query available via memory.relational_store.db\n")


def _add_topic_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("topic", help="Research topic")
    parser.add_argument("--email", default="", help="Recipient email address")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto News Research Assistant")
    sub = parser.add_subparsers(dest="command")

    for name, help_text in [("run", "LangGraph workflow"), ("react", "ReAct agent"),
                             ("plan", "Plan-and-Execute agent")]:
        _add_topic_args(sub.add_parser(name, help=help_text))

    ask_p = sub.add_parser("ask", help="RAG Q&A over research history")
    ask_p.add_argument("question", help="Question about past research")

    sub.add_parser("stats", help="Show memory/store statistics")
    args = parser.parse_args()

    match args.command:
        case "run":   cmd_run(args.topic, args.email)
        case "react": cmd_react(args.topic, args.email)
        case "plan":  cmd_plan(args.topic, args.email)
        case "ask":   cmd_ask(args.question)
        case "stats": cmd_stats()
        case _:       parser.print_help()


if __name__ == "__main__":
    main()
