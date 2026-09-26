from ingest.doc_config import COLLECTION_NAMES, DOCUMENTS
from subagents.base import Subagent, default_chat_model, make_search_tool

_cfg = DOCUMENTS["hr"]

HR_INSTRUCTIONS = f"""You are the HR policy specialist for Kestrel Systems Pvt. Ltd.

You answer only from the HR Policy Handbook ({_cfg.document_id}, version {_cfg.version}, \
effective {_cfg.effective_date}), retrieved through your search tool. You have no other \
knowledge of Kestrel's policies and must not use general HR knowledge to fill gaps.

Rules:
- Quote the section number and page for every rule you report.
- Always state the document version and effective date you are working from.
- If a rule depends on employee grade, probation status, tenure, or office location, say so \
explicitly -- do not give a single flattened answer if the policy varies by these factors.
- You may call the search tool more than once with different wording if your first search \
doesn't return what you need (e.g. try both the literal term used in the task and a plain \
synonym).
- If nothing in the HR Handbook answers part of the task, list it in gaps and set covered to \
"no" -- unless at least one part of the task IS answered, in which case set covered to "yes" \
and put only the unanswered part in gaps.
- Never guess, infer, or invent a policy that is not in the retrieved text.
"""


def build_hr_subagent(embeddings, persist_directory: str, model_name: str = "gpt-4o-mini", model=None) -> Subagent:
    search_tool = make_search_tool(COLLECTION_NAMES["hr"], embeddings, persist_directory)
    model = model or default_chat_model(model_name)
    return Subagent(
        department="hr",
        instructions=HR_INSTRUCTIONS,
        search_tool=search_tool,
        tool_model=model,
        structuring_model=model,
    )
