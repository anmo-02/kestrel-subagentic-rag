"""
The 16-question acceptance test set from the assignment, Section 8,
transcribed exactly (profiles, expected departments, and the key points a
correct answer must contain).
"""

from typing import List, Optional

from pydantic import BaseModel

from models import EmployeeProfile


class TestQuestion(BaseModel):
    id: int
    question: str
    profile: EmployeeProfile
    expected_departments: List[str]
    must_contain: List[str]  # key points a correct answer must contain (checked loosely, see eval/run_eval.py)
    expect_conflict: bool = False


ANY_PROFILE = EmployeeProfile(name="Any Employee", grade="G3", office="Pune", joining_date="2024-01-01")

TEST_QUESTIONS: List[TestQuestion] = [
    TestQuestion(
        id=1,
        question="How many earned leaves do I get a year, and how many can I carry forward?",
        profile=ANY_PROFILE,
        expected_departments=["hr"],
        must_contain=["18 days", "1.5", "30", "carry forward"],
    ),
    TestQuestion(
        id=2,
        question="My VPN keeps disconnecting. What should I do?",
        profile=ANY_PROFILE,
        expected_departments=["it"],
        must_contain=["update", "switch", "reset", "P3", "1 business day"],
    ),
    TestQuestion(
        id=3,
        question="What is my daily meal limit on a business trip?",
        profile=EmployeeProfile(name="Test", grade="G4", office="Pune"),
        expected_departments=["finance"],
        must_contain=["1,500", "all cities"],
    ),
    TestQuestion(
        id=4,
        question="How long do I have to submit an expense claim?",
        profile=ANY_PROFILE,
        expected_departments=["finance"],
        must_contain=["30 days", "60 days", "rejected"],
    ),
    TestQuestion(
        id=5,
        question="What is my hotel limit per night in Nashik?",
        profile=EmployeeProfile(name="Test", grade="G2", office="Pune"),
        expected_departments=["finance"],
        must_contain=["non-metro", "75%", "3,000"],
    ),
    TestQuestion(
        id=6,
        question="I lost my laptop at the airport. What now?",
        profile=ANY_PROFILE,
        expected_departments=["it"],
        must_contain=["P1", "24 hours", "police", "wipe", "25%"],
    ),
    TestQuestion(
        id=7,
        question="Can I paste customer data into ChatGPT to summarise it?",
        profile=ANY_PROFILE,
        expected_departments=["it"],
        must_contain=["Confidential", "public", "Kestrel Copilot", "ChatGPT Enterprise"],
    ),
    TestQuestion(
        id=8,
        question="I join next Monday. What documents do I bring, and when do I get my laptop?",
        profile=EmployeeProfile(name="New Joiner", grade="G1", office="Pune"),
        expected_departments=["hr", "it"],
        must_contain=["PAN", "Aadhaar", "day 1", "5 working days", "2 working days"],
    ),
    TestQuestion(
        id=9,
        question="Can I work from Goa for 3 weeks?",
        profile=EmployeeProfile(name="Test", grade="G3", office="Pune"),
        expected_departments=["hr", "it"],
        must_contain=["20 working days", "manager approval", "5 working days", "VPN", "public Wi-Fi"],
    ),
    TestQuestion(
        id=10,
        question="I want to work from Dubai for 2 weeks. Is that allowed?",
        profile=EmployeeProfile(name="Test", grade="G4", office="Pune"),
        expected_departments=["hr", "it"],
        must_contain=["HR Business Partner", "IT Security", "15 working days", "Overseas Access", "10 working days", "sanctioned"],
    ),
    TestQuestion(
        id=11,
        question="I'm resigning. What happens with notice, my laptop and final settlement?",
        profile=EmployeeProfile(name="Test", grade="G4", office="Pune"),
        expected_departments=["hr", "it", "finance"],
        must_contain=["60 days", "HR Head", "30 days", "18:00", "2 working days", "45 days"],
    ),
    TestQuestion(
        id=12,
        question="I'm moving from Pune to the Mumbai office next month. What do I get and what must I do?",
        profile=EmployeeProfile(name="Test", grade="G5", office="Pune", joining_date="2024-06-01"),
        expected_departments=["hr", "it", "finance"],
        must_contain=["HR-12", "30 days", "3 days", "75,000", "60,000", "cabin", "Desk Setup", "10 working days"],
        expect_conflict=True,
    ),
    TestQuestion(
        id=13,
        question="I relocated 8 months ago and now I'm resigning. Do I repay anything?",
        profile=EmployeeProfile(name="Test", grade="G5", office="Mumbai"),
        expected_departments=["finance"],
        must_contain=["4", "25,000", "final settlement"],
    ),
    TestQuestion(
        id=14,
        question="How many nights of temporary accommodation do I get when relocating?",
        profile=ANY_PROFILE,
        expected_departments=["hr", "finance"],
        must_contain=["10 nights", "15 nights", "replaces"],
        expect_conflict=True,
    ),
    TestQuestion(
        id=15,
        question="What is the policy on sabbatical leave?",
        profile=ANY_PROFILE,
        expected_departments=["hr"],
        must_contain=["not covered", "HR Business Partner"],
    ),
    TestQuestion(
        id=16,
        question="What is Kestrel's share price today?",
        profile=ANY_PROFILE,
        expected_departments=[],
        must_contain=["outside", "HR", "IT", "Finance"],
    ),
]


def get_question(qid: int) -> Optional[TestQuestion]:
    return next((q for q in TEST_QUESTIONS if q.id == qid), None)
