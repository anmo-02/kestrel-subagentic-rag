"""
Stretch goal: skills.

A "skill" here is a Markdown file of formatting/behavioural instructions
that the supervisor loads INTO the synthesis prompt only when it's relevant
-- rather than every synthesis call carrying instructions for every possible
answer shape. This keeps the base synthesis prompt generic and lets a new
skill be added later (e.g. an "exit-checklist" skill) without touching
supervisor/synthesis.py's core instructions.
"""

from __future__ import annotations

import os
import re

_SKILLS_DIR = os.path.join(os.path.dirname(__file__), "..", "skills")

_RELOCATION_PATTERN = re.compile(r"relocat|moving to|transfer(ring)? to|new office", re.IGNORECASE)


def is_relocation_question(question: str) -> bool:
    return bool(_RELOCATION_PATTERN.search(question))


def load_skill(name: str) -> str:
    path = os.path.join(_SKILLS_DIR, name, "SKILL.md")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def skills_for_question(question: str) -> list:
    """Returns the text of every skill that applies to this question."""
    skills = []
    if is_relocation_question(question):
        skills.append(load_skill("relocation-checklist"))
    return skills
