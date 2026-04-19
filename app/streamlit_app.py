"""Streamlit UI for the Hybrid Job Recommender.

Tabs:
  1. Job Seeker — select a user or enter skills/resume → top-K jobs with explanations.
  2. Recruiter — pick a job → top candidate users (reverse recommendation via similar_jobs).
  3. Pipeline Inspector — visualize the multi-stage funnel for a user.

Run: `streamlit run app/streamlit_app.py`"""
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st

from api.main import get_state


st.set_page_config(page_title="Job Recommender", layout="wide")


@st.cache_resource
def _load_state():
    return get_state()


state = _load_state()
data = state.data
jobs = data.jobs
users = data.users


def _job_card(row: pd.Series, score: float | None = None, explanation: str = ""):
    with st.container(border=True):
        header = f"**{row['title']}**"
        if score is not None:
            header += f" — score: `{score:.3f}`"
        st.markdown(header)
        st.caption(f"{row.get('category', '')} · {row.get('seniority', '')} · {row.get('location', '')}")
        st.write(f"Skills: `{row.get('skills', '')}`")
        if row.get("salary_min") and row.get("salary_max"):
            st.write(f"Salary: ${int(row['salary_min']):,} – ${int(row['salary_max']):,}")
        if explanation:
            st.info(explanation)


tab1, tab2, tab3, tab4 = st.tabs(["Job Seeker", "Recruiter", "Pipeline Inspector", "Tools"])


with tab1:
    st.header("Find jobs for a user")
    mode = st.radio("Mode", ["Existing user", "New user (resume + skills)"], horizontal=True)
    k = st.slider("Number of recommendations", 3, 20, 10)
    model_choice = st.selectbox(
        "Model", ["multi_stage", "hybrid", "content", "collab", "popularity"])

    if mode == "Existing user":
        uid = st.selectbox("User", users["user_id"].tolist())
        if st.button("Recommend"):
            if model_choice == "multi_stage":
                recs = state.pipeline.recommend(int(uid))
                for r in recs[:k]:
                    row = jobs[jobs["job_id"] == r.job_id].iloc[0]
                    _job_card(row, score=r.score, explanation=r.explanation)
            else:
                if model_choice == "hybrid":
                    out = state.hybrid.recommend(int(uid), k=k)
                elif model_choice == "content":
                    out = state.content.recommend(int(uid), k=k)
                elif model_choice == "collab":
                    out = state.collab.recommend(int(uid), k=k)
                else:
                    out = state.popularity.recommend(k=k)
                for jid, s in out:
                    row = jobs[jobs["job_id"] == jid].iloc[0]
                    _job_card(row, score=s)
    else:
        resume = st.text_area("Resume summary", value="", height=150)
        skills = st.text_input("Skills (comma-separated)", value="")
        if st.button("Recommend for new user"):
            recs = state.hybrid.recommend_for_new_user(resume=resume, skills=skills, k=k)
            for jid, s in recs:
                row = jobs[jobs["job_id"] == jid].iloc[0]
                _job_card(row, score=s)


with tab2:
    st.header("Find candidate jobs similar to a target job (reverse rec)")
    jid = st.selectbox("Job", jobs["job_id"].tolist(),
                       format_func=lambda j: f"{j} — {jobs[jobs['job_id']==j]['title'].iloc[0]}")
    k2 = st.slider("K similar jobs", 3, 20, 10, key="k2")
    if st.button("Find similar"):
        sims = state.content.similar_jobs(int(jid), k=k2)
        for s_jid, s in sims:
            row = jobs[jobs["job_id"] == s_jid].iloc[0]
            _job_card(row, score=s)


with tab3:
    st.header("Multi-stage pipeline funnel inspector")
    uid = st.selectbox("User (inspector)", users["user_id"].tolist(), key="uid_inspect")
    if st.button("Inspect"):
        out = state.pipeline.inspect(int(uid))
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader(f"Retrieval ({len(out['retrieval'])})")
            for jid_, s in out["retrieval"][:20]:
                title = jobs[jobs["job_id"] == jid_]["title"].iloc[0] if jid_ in set(jobs["job_id"]) else "?"
                st.write(f"`{s:.3f}` — {title}")
        with c2:
            st.subheader(f"Ranking ({len(out['ranking'])})")
            for jid_, s in out["ranking"][:20]:
                title = jobs[jobs["job_id"] == jid_]["title"].iloc[0] if jid_ in set(jobs["job_id"]) else "?"
                st.write(f"`{s:.3f}` — {title}")
        with c3:
            st.subheader(f"Final ({len(out['final'])})")
            for jid_, s, exp in out["final"]:
                title = jobs[jobs["job_id"] == jid_]["title"].iloc[0] if jid_ in set(jobs["job_id"]) else "?"
                st.write(f"`{s:.3f}` — {title}")
                if exp:
                    st.caption(exp)


with tab4:
    st.header("Domain tools")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Natural-language query")
        q = st.text_input("Query", value="senior python backend, remote, 180k+")
        if st.button("Parse query"):
            p = state.query_parser.parse(q)
            st.json(p.to_filter_dict())

    with col2:
        st.subheader("Salary predictor")
        sc = st.text_input("Category", value="backend")
        ss = st.text_input("Seniority", value="senior")
        sl = st.text_input("Location", value="Remote")
        sk = st.text_input("Skills", value="python,docker,kubernetes")
        if st.button("Predict salary"):
            if state.salary is None:
                st.warning("Salary model not available (not trained).")
            else:
                out = state.salary.predict(category=sc, seniority=ss, location=sl, skills=sk)
                st.metric("Midpoint", f"${out.midpoint:,.0f}")
                st.caption(f"Range: ${out.salary_min:,.0f} – ${out.salary_max:,.0f}")

    with col3:
        st.subheader("Skill gap")
        us = st.text_input("User skills", value="python,docker")
        ts = st.text_input("Target (job) skills", value="python,kubernetes,terraform")
        if st.button("Analyze gap"):
            from src.features.structured_features import parse_skills
            r = state.gap.analyze(parse_skills(us), parse_skills(ts))
            st.write("Matched:", r.matched)
            st.write("Missing:", r.missing)
            st.write("Adjacent:", r.adjacent)
            st.metric("Match rate", f"{r.match_rate:.0%}")
