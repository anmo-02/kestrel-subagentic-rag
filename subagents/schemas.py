from typing import List, Literal

from pydantic import BaseModel, Field


class Finding(BaseModel):
    """One rule a subagent found, always traceable to a specific clause."""

    rule: str = Field(..., description="The rule stated in plain, employee-facing words.")
    section: str = Field(..., description="Clause number this rule comes from, e.g. '6.6'.")
    page: int = Field(..., description="Page number the clause appears on.")
    document_id: str
    version: str
    effective_date: str


class SubagentResult(BaseModel):
    """The fixed output format every subagent must return, no exceptions."""

    department: Literal["hr", "it", "finance"]
    covered: Literal["yes", "no"] = Field(
        ..., description="'no' if nothing in this department's document answers any part of the task."
    )
    findings: List[Finding] = Field(default_factory=list)
    gaps: List[str] = Field(
        default_factory=list,
        description="Parts of the task this department's document does not answer. Never leave this implicit.",
    )
