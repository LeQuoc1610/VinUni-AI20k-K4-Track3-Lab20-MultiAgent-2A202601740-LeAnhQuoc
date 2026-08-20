"""Unit tests for SupervisorAgent's routing policy.

Replaces the original skeleton guard test (test_agents_todo.py), which only asserted
that SupervisorAgent.run raised StudentTodoError.
"""

from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState


def _state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))


def test_routes_to_researcher_first() -> None:
    state = SupervisorAgent().run(_state())
    assert state.route_history == ["researcher"]
    assert state.iteration == 1


def test_routes_through_full_pipeline_in_order() -> None:
    state = _state()
    state = SupervisorAgent().run(state)
    assert state.route_history[-1] == "researcher"

    state.sources = [SourceDocument(title="t", snippet="s")]
    state.research_notes = "notes"
    state = SupervisorAgent().run(state)
    assert state.route_history[-1] == "analyst"

    state.analysis_notes = "analysis"
    state = SupervisorAgent().run(state)
    assert state.route_history[-1] == "writer"

    state.final_answer = "answer"
    state = SupervisorAgent().run(state)
    assert state.route_history[-1] == "critic"

    state.critic_notes = "looks good"
    state = SupervisorAgent().run(state)
    assert state.route_history[-1] == "done"


def test_stuck_stage_falls_back_to_writer_after_max_attempts() -> None:
    state = _state()
    state.route_history = ["researcher", "researcher"]
    state.iteration = 2
    state = SupervisorAgent().run(state)
    assert state.route_history[-1] == "writer"
    assert state.errors


def test_max_iterations_forces_stop() -> None:
    state = _state()
    state.iteration = 6
    state = SupervisorAgent().run(state)
    assert state.route_history == ["done"]
    assert state.errors
