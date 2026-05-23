"""LangGraph workflow builder — compiles the research state graph."""
from typing import Literal

from langgraph.graph import StateGraph, END

from graph.state import ResearchState
from graph.nodes import plan_node, search_node, summarize_node, report_node, email_node


def build_workflow(*, with_email: bool = False) -> StateGraph:
    """Build and compile the research workflow as a LangGraph StateGraph."""
    builder = StateGraph(ResearchState)

    for name, fn in [("plan", plan_node), ("search", search_node),
                      ("summarize", summarize_node), ("report", report_node),
                      ("email", email_node)]:
        builder.add_node(name, fn)

    builder.set_entry_point("plan")
    for src, dst in [("plan", "search"), ("search", "summarize"), ("summarize", "report")]:
        builder.add_edge(src, dst)

    if with_email:
        def _should_send(state: ResearchState) -> Literal["email", END]:
            return "email" if (state.get("recipient_email") and not state.get("error")) else END
        builder.add_conditional_edges("report", _should_send, {"email": "email", END: END})
        builder.add_edge("email", END)
    else:
        builder.add_edge("report", END)

    return builder.compile()


research_workflow = build_workflow(with_email=True)
