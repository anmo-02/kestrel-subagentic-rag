from models import EmployeeProfile
from supervisor.planner import plan_question
from tests.fake_chat_model import FakeChatModel


def test_relocation_question_routes_to_all_three():
    profile = EmployeeProfile(name="Asha", grade="G5", office="Pune", joining_date="2023-01-10")
    plan = plan_question(
        FakeChatModel(),
        "I'm relocating from Pune to Mumbai next month. What happens to my laptop, access card, relocation allowance and leave?",
        profile,
    )
    depts = {t.department for t in plan.departments}
    assert depts == {"hr", "it", "finance"}


def test_vpn_question_routes_to_it_only():
    profile = EmployeeProfile(name="Ravi", grade="G2", office="Bengaluru")
    plan = plan_question(FakeChatModel(), "My VPN keeps disconnecting, what should I do?", profile)
    depts = {t.department for t in plan.departments}
    assert depts == {"it"}


def test_out_of_scope_question_returns_empty_plan():
    profile = EmployeeProfile(name="Meera", grade="G3", office="Pune")
    plan = plan_question(FakeChatModel(), "What is Kestrel's share price today?", profile)
    assert plan.departments == []


def test_profile_is_injected_into_every_task_message():
    profile = EmployeeProfile(name="Dev", grade="G4", office="Mumbai", joining_date="2022-05-01")
    plan = plan_question(FakeChatModel(), "My VPN keeps disconnecting.", profile)
    for task in plan.departments:
        assert "G4" in task.task_message
        assert "Mumbai" in task.task_message


def test_metro_city_note_appended_to_finance_task():
    profile = EmployeeProfile(name="Priya", grade="G2", office="Pune")
    plan = plan_question(FakeChatModel(), "What is my hotel limit per night in Nashik?", profile)
    finance_tasks = [t for t in plan.departments if t.department == "finance"]
    assert finance_tasks, "expected a finance task for a hotel-limit question"
    assert "NON-METRO" in finance_tasks[0].task_message

    profile2 = EmployeeProfile(name="Priya", grade="G2", office="Pune")
    plan2 = plan_question(FakeChatModel(), "What is my hotel limit per night in Mumbai for an expense claim?", profile2)
    finance_tasks2 = [t for t in plan2.departments if t.department == "finance"]
    assert finance_tasks2
    assert "METRO" in finance_tasks2[0].task_message
