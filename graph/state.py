"""The State that flows through the LangGraph."""
from typing import TypedDict, Annotated, Optional, Literal
from operator import add


class Candidate(TypedDict):
    """One proposed material plus everything we've learned about it."""
    formula: str
    rationale: str                      # why the Generator proposed it
    structure_source: str               # "materials_project" | "constructed" | etc.
    cif: Optional[str]                  # the CIF text once we have a structure
    dft_status: Literal["pending", "running", "done", "failed", "cached"]
    dft_result: Optional[dict]          # parsed QE output
    critic_notes: Optional[str]         # per-candidate critic feedback


def _merge_candidates(
    existing: list[Candidate], new: list[Candidate]
) -> list[Candidate]:
    """Merge candidates by formula: new entries overwrite existing ones."""
    by_formula = {c["formula"]: c for c in existing}
    for c in new:
        by_formula[c["formula"]] = c
    return list(by_formula.values())


class CriticVerdict(TypedDict):
    decision: Literal["continue", "done", "abort"]
    reason: str
    new_directions: list[str]           # suggestions fed back into the Generator
    n_meeting_targets: int              # how many candidates met ALL targets this pass


class DiscoveryState(TypedDict):
    # Set once at start ----------------------------------------------------
    goal: str
    max_iterations: int

    # Updated by Planner once at start -------------------------------------
    plan: Optional[dict]

    # Updated each loop iteration ------------------------------------------
    iteration: int
    candidates: Annotated[list[Candidate], _merge_candidates]
    evidence: Annotated[list[dict], add]           # retrieved chunks
    critic_history: Annotated[list[CriticVerdict], add]

    # Set by the latest Critic call ---------------------------------------
    latest_verdict: Optional[CriticVerdict]

    # Set by Reporter at the end -------------------------------------------
    final_report: Optional[str]
