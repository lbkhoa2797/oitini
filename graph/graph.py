"""Build and compile the discovery graph."""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from config import GRAPH_DB_PATH
from graph.state import DiscoveryState
from graph.nodes import (
    planner_node, generator_node, simulator_node,
    critic_node, reporter_node,
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
    g.add_node("critic", critic_node)
    g.add_node("reporter", reporter_node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "generator")
    g.add_edge("generator", "simulator")
    g.add_edge("simulator", "critic")
    g.add_conditional_edges(
        "critic", route_from_critic,
        {"generator": "generator", "reporter": "reporter"},
    )
    g.add_edge("reporter", END)

    return g.compile(checkpointer=checkpointer)


def make_default_graph():
    conn = sqlite3.connect(str(GRAPH_DB_PATH), check_same_thread=False)
    return build_graph(checkpointer=SqliteSaver(conn))
