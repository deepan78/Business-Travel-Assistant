
from typing import TypedDict, Any

from langgraph.graph import StateGraph, START, END

from rag import answer_policy_question
from tools import estimate_hotel_reimbursement


class TravelAssistantState(TypedDict, total=False):
    action: str
    question: str
    context: str
    sources: list[str]
    documents: list[dict[str, str]]
    tool_inputs: dict[str, Any]
    tool_result: dict[str, Any]
    answer: str


def retrieve_and_generate(state: TravelAssistantState):
    """Retrieve policy documents and generate a grounded answer."""

    result = answer_policy_question(state.get("question", ""))

    context = "\n\n".join(
        document["content"]
        for document in result["documents"]
    )

    return {
        "context": context,
        "sources": result["sources"],
        "documents": result["documents"],
        "answer": result["answer"],
    }


def run_reimbursement_tool(state: TravelAssistantState):
    """Run the hotel reimbursement calculation tool."""

    inputs = state.get("tool_inputs", {})

    result = estimate_hotel_reimbursement.invoke(inputs)

    return {
        "tool_result": result,
        "answer": (
            "Hotel reimbursement estimate calculated successfully."
        ),
    }


def build_graph():
    workflow = StateGraph(TravelAssistantState)

    workflow.add_node("retrieve_and_generate", retrieve_and_generate)
    workflow.add_node("reimbursement_tool", run_reimbursement_tool)

    def choose_workflow(state: TravelAssistantState):
        if state.get("action") == "estimate_reimbursement":
            return "reimbursement_tool"

        return "retrieve_and_generate"

    workflow.add_conditional_edges(
        START,
        choose_workflow,
        {
            "retrieve_and_generate": "retrieve_and_generate",
            "reimbursement_tool": "reimbursement_tool",
        },
    )

    workflow.add_edge("retrieve_and_generate", END)
    workflow.add_edge("reimbursement_tool", END)

    return workflow.compile()


travel_graph = build_graph()


def run_assistant(
    action: str,
    question: str = "",
    tool_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the selected Business Travel Assistant workflow."""

    initial_state: TravelAssistantState = {
        "action": action,
        "question": question,
        "tool_inputs": tool_inputs or {},
    }

    return travel_graph.invoke(initial_state)
