from typing import List, Literal

from pydantic import BaseModel, Field

Department = Literal["hr", "it", "finance"]


class DepartmentTask(BaseModel):
    department: Department
    task_message: str = Field(
        ...,
        description=(
            "A complete, standalone message for this department's subagent. It must include "
            "every employee detail (grade, office, joining date, city) that matters, and state "
            "exactly what to find -- the subagent will see nothing else: no chat history, no "
            "other department's findings."
        ),
    )


class Plan(BaseModel):
    departments: List[DepartmentTask] = Field(
        default_factory=list,
        description="Zero to three departments this question needs. Empty if no department's documents are relevant to the question at all.",
    )
    reason: str = Field(..., description="One sentence explaining why these departments (and no others) were chosen.")


class ConflictNote(BaseModel):
    topic: str
    documents: List[str] = Field(..., description="Each entry like 'KSPL-HR-POL-004 sec 4.4'.")
    values: List[str] = Field(..., description="The differing values found, in the same order as documents.")
    applies: str = Field(..., description="Which value applies to the employee, stated plainly.")
    reason: str = Field(..., description="Which conflict-resolution rule was used and why.")


class FinalAnswer(BaseModel):
    answer_markdown: str = Field(..., description="The complete answer to the employee, in markdown, with a citation like [KSPL-FIN-POL-003, section 6.2] after every rule.")
    conflicts: List[ConflictNote] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list, description="Anything no subagent could answer. Suggest a contact (HR Business Partner, hr-help@kestrel.example, or IT service desk) where it helps.")
    departments_consulted: List[Department] = Field(default_factory=list)
