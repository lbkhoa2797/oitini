"""Build and compile the discovery graph."""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from config import GRAPH_DB_PATH
from graph.state import DiscoveryState
from graph.nodes import (
    planner_node, generator_node, simulator_node,
    evidence_node, critic_node, reporter_node, verifier_node,
)


def route_from_critic(state: DiscoveryState) -> str:
    verdict = state["latest_verdict"]
    if verdict["decision"] == "done":
        return "reporter"
    if verdict["decision"] == "abort":
        return "reporter"
    return "generator"


def build_graph(checkpointer=None):
    g = StateGraph(DiscoveryState)
    g.add_node("planner", planner_node)
    g.add_node("generator", generator_node)
    g.add_node("simulator", simulator_node)
    g.add_node("evidence", evidence_node)
    g.add_node("critic", critic_node)
    g.add_node("reporter", reporter_node)
    g.add_node("verifier", verifier_node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "generator")
    g.add_edge("generator", "simulator")
    # Evidence sits between simulation and criticism so the Critic judges a
    # literature target against retrieved chunks rather than from recall.
    g.add_edge("simulator", "evidence")
    g.add_edge("evidence", "critic")
    g.add_conditional_edges(
        "critic", route_from_critic,
        {"generator": "generator", "reporter": "reporter"},
    )
    g.add_edge("reporter", "verifier")
    g.add_edge("verifier", END)

    return g.compile(checkpointer=checkpointer)


def make_default_graph():
    conn = sqlite3.connect(str(GRAPH_DB_PATH), check_same_thread=False)
    return build_graph(checkpointer=SqliteSaver(conn))
