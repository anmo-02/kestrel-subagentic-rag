"""
Phase 8: the employee-facing chat UI.

Run with:  streamlit run app/streamlit_app.py
Needs an OPENAI_API_KEY in .env (see .env.example) and a populated
chroma_db/ (run `python -m ingest.build_collections` once first).
"""

from __future__ import annotations

import os
import sys
import time

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from graph.workflow import build_graph
from ingest.build_collections import PERSIST_DIR, build_all_collections
from models import EmployeeProfile

load_dotenv()

st.set_page_config(page_title="Kestrel Policy Assistant", page_icon="🧭", layout="wide")

# Rough, illustrative per-1M-token rates in USD -- OpenAI pricing changes
# over time, so treat these as placeholders and update them from
# https://platform.openai.com/pricing before trusting the cost estimates.
PRICING_PER_1M = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


def estimate_cost(model_name: str, tokens: dict) -> float:
    rates = PRICING_PER_1M.get(model_name, PRICING_PER_1M["gpt-4o-mini"])
    return (
        tokens.get("input_tokens", 0) / 1_000_000 * rates["input"]
        + tokens.get("output_tokens", 0) / 1_000_000 * rates["output"]
    )


@st.cache_resource(show_spinner="Loading policy collections...")
def get_app(disabled_departments: frozenset):
    if not os.path.exists(PERSIST_DIR):
        build_all_collections()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    planning_model = ChatOpenAI(model="gpt-4o", temperature=0)
    synthesis_model = ChatOpenAI(model="gpt-4o", temperature=0)
    return build_graph(
        planning_model=planning_model,
        synthesis_model=synthesis_model,
        embeddings=embeddings,
        persist_directory=PERSIST_DIR,
        subagent_model_name="gpt-4o-mini",
        disabled_departments=set(disabled_departments),
    )


def to_lc_history(messages: list) -> list:
    lc_messages = []
    for m in messages:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        else:
            lc_messages.append(AIMessage(content=m["content"]))
    return lc_messages


# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.header("Your profile")
    name = st.text_input("Name", value=st.session_state.get("name", "Employee"))
    grade = st.selectbox("Grade", ["G1", "G2", "G3", "G4", "G5", "G6", "G7"], index=2)
    office = st.selectbox("Office", ["Pune", "Mumbai", "Bengaluru"])
    joining_date = st.date_input("Joining date", value=None)

    profile = EmployeeProfile(
        name=name or "Employee",
        grade=grade,
        office=office,
        joining_date=str(joining_date) if joining_date else None,
    )

    st.divider()
    with st.expander("Dev tools", expanded=False):
        st.caption("Simulate a department outage to see gap-handling (Phase 6 checkpoint).")
        disabled = st.multiselect("Disable subagent(s)", ["hr", "it", "finance"], default=[])

    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------------------------- main --
st.title("🧭 Kestrel Policy Assistant")
st.caption("Ask about HR, IT, or Finance/travel policy. Answers cite the exact clause.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("trace"):
            trace = msg["trace"]
            if trace.get("conflicts"):
                st.warning(f"⚠️ {len(trace['conflicts'])} conflict(s) between documents were resolved -- see below.")
            with st.expander("Behind this answer", expanded=False):
                st.markdown(f"**Plan:** departments = `{trace['plan_departments']}`  \n**Reason:** {trace['plan_reason']}")
                for dept, run in trace["subagent_runs"].items():
                    st.markdown(f"---\n**{dept.upper()} subagent** &nbsp; "
                                f"({run['elapsed_seconds']:.2f}s, {run['tokens_used']['total_tokens']} tokens"
                                f"{', ERROR: ' + run['error'] if run['error'] else ''})")
                    st.markdown(f"*Task message:* {run['task_message']}")
                    st.markdown(f"*Queries run:* {run['queries']}")
                    result = run["result"]
                    st.markdown(f"*Covered:* {result.covered}")
                    for f in result.findings:
                        st.markdown(f"- {f.rule} `[{f.document_id}, sec {f.section}, p.{f.page}]`")
                    for g in result.gaps:
                        st.markdown(f"- ⚠️ Gap: {g}")
                if trace.get("conflicts"):
                    st.markdown("---\n**Conflicts resolved:**")
                    for c in trace["conflicts"]:
                        st.markdown(f"- **{c.topic}**: {c.documents} -> {c.values}. Applies: *{c.applies}* ({c.reason})")
                st.markdown(
                    f"---\n**Total time:** {trace['total_time']:.2f}s &nbsp; "
                    f"**Estimated cost:** ${trace['estimated_cost']:.5f} "
                    f"*(illustrative -- update PRICING_PER_1M with current rates)*"
                )

question = st.chat_input("Ask a policy question...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    app = get_app(frozenset(disabled))
    chat_history = to_lc_history(st.session_state.messages[:-1])

    with st.chat_message("assistant"):
        with st.spinner("Checking policies..."):
            start = time.time()
            result = app.invoke({"question": question, "profile": profile, "chat_history": chat_history})
            total_time = time.time() - start

        final = result["final_answer"]
        subagent_runs = result.get("subagent_runs", {})
        plan = result.get("plan")

        # Attach the task_message each subagent actually received, for the trace panel.
        task_messages = {t.department: t.task_message for t in (plan.departments if plan else [])}
        for dept, run in subagent_runs.items():
            run["task_message"] = task_messages.get(dept, "(n/a)")

        est_cost = sum(estimate_cost("gpt-4o-mini", run["tokens_used"]) for run in subagent_runs.values())
        est_cost += estimate_cost("gpt-4o", result.get("plan_tokens", {}))
        est_cost += estimate_cost("gpt-4o", result.get("synthesis_tokens", {}))

        st.markdown(final.answer_markdown)
        if final.conflicts:
            st.warning(f"⚠️ {len(final.conflicts)} conflict(s) between documents were resolved -- see below.")

        trace = {
            "plan_departments": [t.department for t in plan.departments] if plan else [],
            "plan_reason": plan.reason if plan else "n/a",
            "subagent_runs": subagent_runs,
            "conflicts": final.conflicts,
            "total_time": total_time,
            "estimated_cost": est_cost,
        }
        with st.expander("Behind this answer", expanded=False):
            st.markdown(f"**Plan:** departments = `{trace['plan_departments']}`  \n**Reason:** {trace['plan_reason']}")
            for dept, run in subagent_runs.items():
                st.markdown(f"---\n**{dept.upper()} subagent** &nbsp; "
                            f"({run['elapsed_seconds']:.2f}s, {run['tokens_used']['total_tokens']} tokens"
                            f"{', ERROR: ' + run['error'] if run['error'] else ''})")
                st.markdown(f"*Task message:* {run['task_message']}")
                st.markdown(f"*Queries run:* {run['queries']}")
                result_obj = run["result"]
                st.markdown(f"*Covered:* {result_obj.covered}")
                for f in result_obj.findings:
                    st.markdown(f"- {f.rule} `[{f.document_id}, sec {f.section}, p.{f.page}]`")
                for g in result_obj.gaps:
                    st.markdown(f"- ⚠️ Gap: {g}")
            if final.conflicts:
                st.markdown("---\n**Conflicts resolved:**")
                for c in final.conflicts:
                    st.markdown(f"- **{c.topic}**: {c.documents} -> {c.values}. Applies: *{c.applies}* ({c.reason})")
            st.markdown(
                f"---\n**Total time:** {total_time:.2f}s &nbsp; "
                f"**Estimated cost:** ${est_cost:.5f} *(illustrative -- update PRICING_PER_1M with current rates)*"
            )

    st.session_state.messages.append({"role": "assistant", "content": final.answer_markdown, "trace": trace})
