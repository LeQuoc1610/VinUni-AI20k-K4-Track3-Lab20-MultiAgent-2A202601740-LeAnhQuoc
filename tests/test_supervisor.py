"""Skeleton guard test.

NOTE(student): Test này chỉ xác nhận skeleton còn nguyên TODO. Sau khi bạn implement
SupervisorAgent, test này SẼ FAIL - đó là điều bình thường. Hãy xóa hoặc thay thế nó
bằng unit test thật cho routing policy của bạn.
"""

import pytest

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_is_student_todo() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    with pytest.raises(StudentTodoError):
        SupervisorAgent().run(state)
