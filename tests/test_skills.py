from supervisor.skills import is_relocation_question, load_skill, skills_for_question


def test_relocation_keyword_detection():
    assert is_relocation_question("I'm relocating from Pune to Mumbai next month.")
    assert is_relocation_question("What do I get if I'm transferring to the Bengaluru office?")
    assert not is_relocation_question("My VPN keeps disconnecting.")
    assert not is_relocation_question("How many earned leave days do I get?")


def test_skill_loads_and_is_nonempty():
    text = load_skill("relocation-checklist")
    assert "checklist" in text.lower()
    assert len(text) > 200


def test_skills_for_question_returns_empty_for_unrelated_question():
    assert skills_for_question("What is my daily meal limit?") == []


def test_skills_for_question_returns_relocation_skill():
    skills = skills_for_question("I'm relocating to Mumbai, what do I need to do?")
    assert len(skills) == 1
    assert "checklist" in skills[0].lower()
