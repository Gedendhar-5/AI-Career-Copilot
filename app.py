# app.py — Phase 5 — Full Dashboard + Charts + Polish

import streamlit as st
import numpy as np
import faiss
import sys
import os
import json
from datetime import datetime

st.set_page_config(
    page_title="AI Career Copilot",
    page_icon="🚀",
    layout="wide"
)

# ── Absolute path fix ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ── Package check ─────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    st.error("Run: pip install sentence-transformers")
    st.stop()

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

# ── Module imports (always use module prefix) ─────────────────
# ── Module imports (always use module prefix) ─────────────────
import memory.user_memory    as um
import agents.planner        as planner
import agents.advisor        as advisor
import agents.task_generator as task_gen
import agents.prompt_builder as pb
import agents.llm_engine     as llm
from rag.retriever import Retriever        

# ── Embedding model ───────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

with st.spinner("Loading AI model..."):
    model = load_model()

# ── Session state ─────────────────────────────────────────────
defaults = {
    "documents":    [],
    "index":        None,
    "indexed":      False,
    "current_user": None,
    "page":         "profile",
    "analysis":     None,
    "plan":         None,
    "tasks":        None,
    "ai_tasks":     None,
    "chat_history": [],
    "retriever":    None,      # ← ADD
    "resume_text":  "",        # ← ADD
    "job_text":     ""         # ← ADD
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚀 AI Career Copilot")
    st.markdown("---")

    if st.session_state.current_user:
        user = um.get_user(st.session_state.current_user)
        if user:
            st.success(f"👤 {st.session_state.current_user}")
            st.markdown(f"**Goal:** {user['goal']}")

            # Mini progress bar
            score = user['progress']['skill_score']
            level = min(score // 100, 10)
            st.markdown(f"**Level {level}** — {score} pts")
            st.progress(min(score / 1000, 1.0))
            st.markdown(f"Sessions: `{user['progress']['sessions']}`")

    st.markdown("---")

    # LLM status
    if llm.LLM_AVAILABLE:
        st.success("🤖 Groq AI: Connected")
        st.caption("Llama 3.3 70B | Free | 1000 req/day")
    else:
        st.warning("⚠️ Add GROQ_API_KEY to .env")
        st.caption("Get key: console.groq.com")

    st.markdown("---")

    pages = {
        "👤 Profile":       "profile",
        "📄 Resume Analyzer": "analyzer",
        "💬 AI Career Chat": "chat",
        "🗺️ Career Roadmap": "roadmap",
        "📅 Daily Tasks":   "tasks",
        "📊 Dashboard":     "dashboard",
    }
    for label, key in pages.items():
        if st.button(label, use_container_width=True):
            st.session_state.page = key

    st.markdown("---")
    st.caption("— Full Dashboard--")


# ══════════════════════════════════════════════════════════════
# PAGE 1 — PROFILE
# ══════════════════════════════════════════════════════════════
if st.session_state.page == "profile":
    st.title("👤 Profile Setup")
    st.markdown("---")

    tab1, tab2 = st.tabs(["✨ Create Profile", "📂 Load Profile"])

    with tab1:
        st.markdown("### Create Your Career Profile")
        col1, col2 = st.columns(2)
        with col1:
            username   = st.text_input("Username", placeholder="e.g. john_doe")
            goal       = st.text_input("Career Goal", placeholder="e.g. Become a Data Scientist")
        with col2:
            experience = st.selectbox("Experience Level", ["beginner", "intermediate", "advanced"])
            skills_raw = st.text_input("Current Skills (comma separated)", placeholder="Python, SQL, Excel")

        if st.button("💾 Save Profile", type="primary", use_container_width=True):
            if not username.strip() or not goal.strip():
                st.warning("Username and Goal are required.")
            elif um.get_user(username):
                st.warning("Username exists. Load it from the other tab.")
            else:
                skills = [s.strip() for s in skills_raw.split(",") if s.strip()]
                user   = um.create_user(username, goal, skills, experience)
                st.session_state.current_user = username
                st.success(f"✅ Profile saved to D drive!")
                st.balloons()

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Experience", experience.title())
                col_b.metric("Skills Added", len(skills))
                col_c.metric("Starting Score", "0 pts")

    with tab2:
        st.markdown("### Load Existing Profile")
        existing = um.list_users()
        if not existing:
            st.info("No profiles yet. Create one first.")
        else:
            selected = st.selectbox("Select your profile", existing)
            if st.button("🚀 Load Profile", type="primary", use_container_width=True):
                um.update_progress(selected)
                st.session_state.current_user = selected
                user = um.get_user(selected)
                st.success(f"Welcome back, {selected}!")

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Sessions", user['progress']['sessions'])
                col_b.metric("Skill Score", f"{user['progress']['skill_score']} pts")
                col_c.metric("Tasks Done", len(user['progress']['completed_tasks']))


# ══════════════════════════════════════════════════════════════
# PAGE 2 — RESUME ANALYZER
# ══════════════════════════════════════════════════════════════
elif st.session_state.page == "analyzer":
    st.title("📄 Resume & Job Analyzer")

    if not st.session_state.current_user:
        st.warning("Set up your profile first.")
        st.stop()

    user = um.get_user(st.session_state.current_user)
    st.markdown(f"Analyzing for: **{st.session_state.current_user}** | Goal: *{user['goal']}*")
    st.markdown("---")

    # Initialize retriever once
    if st.session_state.retriever is None:
        st.session_state.retriever = Retriever()
        st.session_state.retriever.load()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📄 Your Resume")
        resume_text = st.text_area(
            "Paste resume", height=220,
            placeholder="Paste your full resume here...",
            value=st.session_state.resume_text
        )
    with col2:
        st.markdown("### 💼 Job Description")
        job_text = st.text_area(
            "Paste job description", height=220,
            placeholder="Paste the target job description here...",
            value=st.session_state.job_text
        )

    if st.button("⚡ Index Documents", type="primary", use_container_width=True):
        if not resume_text.strip() and not job_text.strip():
            st.warning("Paste at least one document.")
        else:
            with st.spinner("Parsing and building structured index..."):
                count = st.session_state.retriever.build_from_texts(
                    resume_text, job_text
                )
                st.session_state.indexed     = True
                st.session_state.resume_text = resume_text
                st.session_state.job_text    = job_text

            st.success(f"✅ Indexed {count} structured chunks!")

            # Auto comparison on index
            if resume_text.strip() and job_text.strip():
                comparison = st.session_state.retriever.compare_resume_to_jd(
                    resume_text, job_text
                )
                st.session_state.analysis = {
                    "matched_skills": comparison["matched_skills"],
                    "missing_skills": comparison["missing_skills"],
                    "match_score":    comparison["match_score"],
                    "verdict":        comparison["verdict"],
                    "advice": (
                        f"You match {comparison['match_score']}% of requirements. "
                        f"Focus on: {', '.join(comparison['missing_skills'][:3])}."
                        if comparison["missing_skills"]
                        else "Excellent match! Prepare for interviews."
                    ),
                    "resources": {}
                }

                st.markdown("---")
                st.markdown("### ⚡ Instant Match Analysis")

                if PLOTLY_OK:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=comparison["match_score"],
                        title={"text": "Resume Match Score"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar":  {"color": "#1D9E75"},
                            "steps": [
                                {"range": [0,  40],  "color": "#FCEBEB"},
                                {"range": [40, 70],  "color": "#FAEEDA"},
                                {"range": [70, 100], "color": "#EAF3DE"}
                            ]
                        }
                    ))
                    fig.update_layout(
                        height=240,
                        margin=dict(t=40, b=0, l=20, r=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Match Score",     f"{comparison['match_score']}%")
                c2.metric("Skills Matched",  len(comparison["matched_skills"]))
                c3.metric("Skills Missing",  len(comparison["missing_skills"]))

                col_a, col_b = st.columns(2)
                with col_a:
                    st.success(
                        f"✅ **You have:** "
                        f"{', '.join(comparison['matched_skills']) or 'None detected'}"
                    )
                with col_b:
                    st.error(
                        f"❌ **Learn these:** "
                        f"{', '.join(comparison['missing_skills']) or 'Strong match!'}"
                    )

    st.markdown("---")
    st.markdown("### 🔍 Ask a Specific Question")

    query = st.text_input(
        "Your question",
        placeholder="What skills am I missing for this job?"
    )

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        source_filter = st.selectbox(
            "Search in",
            ["Both", "Resume only", "Job description only"]
        )
    with col_q2:
        top_k = st.slider("Max results", 1, 6, 3)

    source_map = {
        "Resume only":           "resume",
        "Job description only":  "jd",
        "Both":                  None
    }

    if st.button("🔎 Search & Analyze", use_container_width=True):
        if not query.strip():
            st.warning("Enter a question.")
        elif not st.session_state.indexed:
            st.warning("Index your documents first.")
        else:
            results = st.session_state.retriever.query(
                query_text=query,
                top_k=top_k,
                source_filter=source_map[source_filter]
            )

            st.markdown("### 📊 Structured RAG Results")

            if not results:
                st.warning(
                    "No results above threshold. "
                    "Try a broader question or select 'Both' as source."
                )
            else:
                for r in results:
                    badge = "🟢 Resume" if r["source"] == "resume" else "🔵 Job Description"
                    with st.expander(
                        f"{badge} | {r['label']} | Score: {r['score']}",
                        expanded=True
                    ):
                        st.markdown(
                            f"**Section:** `{r['section'].replace('_',' ').title()}`"
                        )
                        st.write(r["text"])

                rag_chunks = [r["text"] for r in results]

                st.markdown("---")
                st.markdown("### 🤖 Groq AI Career Mentor")
                with st.spinner("Llama 3.3 70B analyzing..."):
                    prompt   = pb.build_career_prompt(
                        user, query, rag_chunks, st.session_state.analysis
                    )
                    response = llm.ask_claude(prompt)
                st.markdown(response)

                score_earned = (
                    len(st.session_state.analysis["matched_skills"]) * 10
                    if st.session_state.analysis else 5
                )
                um.update_progress(
                    st.session_state.current_user,
                    task=f"RAG query: {query}",
                    skill_score=score_earned
                )
                st.info(f"💰 +{score_earned} pts saved!")
# ══════════════════════════════════════════════════════════════
# PAGE 3 — AI CAREER CHAT
# ══════════════════════════════════════════════════════════════
elif st.session_state.page == "chat":
    st.title("💬 AI Career Chat")
    st.markdown("Chat with your AI Career Mentor — powered by Groq (Free)")
    st.markdown("---")

    if not st.session_state.current_user:
        st.warning("Set up your profile first.")
        st.stop()

    user = um.get_user(st.session_state.current_user)

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask your career mentor anything...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        rag_chunks = []
        if st.session_state.indexed and st.session_state.index is not None:
            q_vec = model.encode([user_input], convert_to_numpy=True).astype(np.float32)
            _, idxs = st.session_state.index.search(q_vec, 2)
            for i in idxs[0]:
                if i != -1:
                    rag_chunks.append(st.session_state.documents[i])

        with st.spinner("Groq is thinking..."):
            prompt   = pb.build_career_prompt(
                user, user_input, rag_chunks, st.session_state.analysis
            )
            response = llm.ask_claude(prompt)

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        um.update_progress(
            st.session_state.current_user,
            task=f"Chat: {user_input[:50]}",
            skill_score=5
        )

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ══════════════════════════════════════════════════════════════
# PAGE 4 — CAREER ROADMAP
# ══════════════════════════════════════════════════════════════
elif st.session_state.page == "roadmap":
    st.title("🗺️ Career Roadmap")
    st.markdown("---")

    if not st.session_state.current_user:
        st.warning("Set up your profile first.")
        st.stop()

    user          = um.get_user(st.session_state.current_user)
    existing_plan = planner.get_plan(st.session_state.current_user)

    missing = []
    if st.session_state.analysis:
        missing = st.session_state.analysis["missing_skills"]

    missing_input = st.text_input(
        "Skills to learn (comma separated)",
        value=", ".join(missing) if missing else "",
        placeholder="e.g. docker, kubernetes, aws"
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Structured Roadmap", type="primary", use_container_width=True):
            skills_list = [s.strip() for s in missing_input.split(",") if s.strip()]
            with st.spinner("Planner Agent building roadmap..."):
                plan = planner.generate_plan(
                    username=st.session_state.current_user,
                    goal=user["goal"],
                    experience=user["experience"],
                    missing_skills=skills_list
                )
                st.session_state.plan = plan
            st.success("✅ Roadmap saved to D drive!")

    with col2:
        if st.button("🤖 AI Roadmap (Groq)", use_container_width=True):
            skills_list = [s.strip() for s in missing_input.split(",") if s.strip()]
            with st.spinner("Llama 3.3 70B generating your personalised roadmap..."):
                prompt   = pb.build_roadmap_prompt(user, skills_list)
                response = llm.ask_claude(prompt, max_tokens=1500)
            st.markdown("---")
            st.markdown("### 🤖 Your AI-Generated Roadmap")
            st.markdown(response)

            um.update_progress(
                st.session_state.current_user,
                task="Generated AI career roadmap",
                skill_score=15
            )

    # Display structured plan as timeline
    display_plan = st.session_state.plan or existing_plan
    if display_plan:
        st.markdown("---")
        st.markdown(f"### 🗺️ Roadmap — {display_plan['goal']}")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Duration", display_plan['duration'])
        col_b.metric("Daily Hours", f"{display_plan['hours_per_day']}h")
        col_c.metric("Phases", len(display_plan['phases']))

        st.markdown("---")

        colors = ["#E1F5EE", "#E6F1FB", "#FAEEDA"]
        for phase in display_plan["phases"]:
            with st.expander(
                f"Phase {phase['phase']} — {phase['title']} ({phase['duration']})",
                expanded=True
            ):
                st.markdown(f"**Focus:** {phase['focus']}")
                skills = phase['skills']
                if skills:
                    skill_cols = st.columns(min(len(skills), 4))
                    for i, skill in enumerate(skills):
                        skill_cols[i % 4].markdown(
                            f"<span style='background:#E1F5EE;padding:4px 10px;"
                            f"border-radius:12px;font-size:12px;color:#085041'>"
                            f"{skill}</span>",
                            unsafe_allow_html=True
                        )
                else:
                    st.markdown("General practice and portfolio building")


# ══════════════════════════════════════════════════════════════
# PAGE 5 — DAILY TASKS
# ══════════════════════════════════════════════════════════════
elif st.session_state.page == "tasks":
    st.title("📅 Daily Learning Tasks")
    st.markdown("---")

    if not st.session_state.current_user:
        st.warning("Set up your profile first.")
        st.stop()

    user           = um.get_user(st.session_state.current_user)
    existing_tasks = task_gen.get_tasks(st.session_state.current_user)

    missing = []
    if st.session_state.analysis:
        missing = st.session_state.analysis["missing_skills"]

    missing_input = st.text_input(
        "Skills to focus on (comma separated)",
        value=", ".join(missing[:5]) if missing else "",
        placeholder="e.g. docker, aws, python"
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📋 Generate 7-Day Plan", type="primary", use_container_width=True):
            skills_list = [s.strip() for s in missing_input.split(",") if s.strip()]
            with st.spinner("Task Generator building your plan..."):
                task_data = task_gen.generate_tasks(
                    username=st.session_state.current_user,
                    missing_skills=skills_list,
                    experience=user["experience"]
                )
            # ✅ Phase 4 lesson — handle both dict and list safely
            if isinstance(task_data, list):
                st.session_state.tasks    = task_data
                st.session_state.ai_tasks = None
            else:
                st.session_state.tasks    = task_data.get("tasks", [])
                st.session_state.ai_tasks = task_data.get("ai_tasks", None)

            st.success(f"✅ 7-day plan saved to D drive!")

    with col2:
        if st.button("🤖 AI Task for Today (Groq)", use_container_width=True):
            skills_list = [s.strip() for s in missing_input.split(",") if s.strip()]
            skill_focus = skills_list[0] if skills_list else "Python"
            with st.spinner("Groq generating your personalised task..."):
                prompt   = pb.build_task_prompt(user, skill_focus, 1)
                response = llm.ask_claude(prompt, max_tokens=300)
            st.markdown("### 🤖 Today's AI-Generated Task")
            st.info(response)

    # Display AI tasks if available
    if st.session_state.ai_tasks:
        st.markdown("---")
        st.markdown("### 🤖 AI Task Suggestions")
        st.markdown(st.session_state.ai_tasks)

    # Display 7-day structured plan
    display_tasks = st.session_state.tasks or existing_tasks
    if display_tasks:
        st.markdown("---")

        # Task progress bar
        done_count  = sum(1 for t in display_tasks if t["completed"])
        total_count = len(display_tasks)
        st.markdown(f"### 📅 Your 7-Day Plan — {done_count}/{total_count} completed")
        st.progress(done_count / total_count if total_count > 0 else 0)

        for task in display_tasks:
            status = "✅" if task["completed"] else "⬜"
            diff_color = {
                "Easy":   "#EAF3DE",
                "Medium": "#FAEEDA",
                "Hard":   "#FCEBEB"
            }.get(task["difficulty"], "#F1EFE8")

            c1, c2 = st.columns([5, 1])
            with c1:
                with st.expander(
                    f"{status} **{task['day']}** ({task['date']}) — {task['skill_focus'].title()}",
                    expanded=not task["completed"]
                ):
                    st.markdown(f"**Task:** {task['task']}")

                    col_x, col_y = st.columns(2)
                    col_x.markdown(
                        f"<span style='background:{diff_color};padding:3px 10px;"
                        f"border-radius:10px;font-size:12px'>"
                        f"🎯 {task['difficulty']}</span>",
                        unsafe_allow_html=True
                    )
                    col_y.markdown(f"⏱️ {task['estimated_hours']} hour(s)")

            with c2:
                if not task["completed"]:
                    if st.button("✅ Done", key=f"done_{task['day']}"):
                        task_gen.mark_task_done(
                            st.session_state.current_user, task["day"]
                        )
                        um.update_progress(
                            st.session_state.current_user,
                            task=f"Completed: {task['task']}",
                            skill_score=20
                        )
                        st.rerun()


# ══════════════════════════════════════════════════════════════
# PAGE 6 — FULL DASHBOARD (Phase 5 New)
# ══════════════════════════════════════════════════════════════
elif st.session_state.page == "dashboard":
    st.title("📊 Career Progress Dashboard")
    st.markdown("---")

    if not st.session_state.current_user:
        st.warning("Set up your profile first.")
        st.stop()

    user = um.get_user(st.session_state.current_user)

    # ── Top metric cards ──────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    score    = user['progress']['skill_score']
    sessions = user['progress']['sessions']
    done     = len(user['progress']['completed_tasks'])
    level    = min(score // 100, 10)

    tasks_7day = task_gen.get_tasks(st.session_state.current_user)
    weekly_done = sum(1 for t in tasks_7day if t["completed"]) if tasks_7day else 0

    c1.metric("Level", f"Lv {level}", f"+{score % 100} to next")
    c2.metric("Skill Score", f"{score} pts")
    c3.metric("Sessions", sessions)
    c4.metric("Tasks Done", done)
    c5.metric("Weekly", f"{weekly_done}/7")

    st.markdown("---")

    # ── Row 1 — Skill score over time + Weekly task bar ───────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 📈 Score Over Time")
        completed_tasks = user["progress"]["completed_tasks"]

        if PLOTLY_OK and completed_tasks:
            dates  = [t["completed_at"].split(" ")[0] for t in completed_tasks]
            scores = list(range(10, (len(completed_tasks) + 1) * 10, 10))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, y=scores,
                mode="lines+markers",
                line=dict(color="#1D9E75", width=2),
                marker=dict(size=6, color="#1D9E75"),
                fill="tozeroy",
                fillcolor="rgba(29,158,117,0.1)",
                name="Score"
            ))
            fig.update_layout(
                height=280,
                margin=dict(t=10, b=10, l=10, r=10),
                showlegend=False,
                xaxis_title="Date",
                yaxis_title="Score"
            )
            st.plotly_chart(fig, use_container_width=True)
        elif not PLOTLY_OK:
            st.info("Install plotly for charts: pip install plotly")
        else:
            st.info("Complete tasks to see your progress chart here.")

    with col_right:
        st.markdown("### 📅 Weekly Task Completion")
        if PLOTLY_OK and tasks_7day:
            days   = [t["day"] for t in tasks_7day]
            status = [1 if t["completed"] else 0 for t in tasks_7day]
            colors = ["#1D9E75" if s else "#D3D1C7" for s in status]

            fig = go.Figure(go.Bar(
                x=days, y=[1] * 7,
                marker_color=colors,
                text=["✅" if s else "⬜" for s in status],
                textposition="inside"
            ))
            fig.update_layout(
                height=280,
                margin=dict(t=10, b=10, l=10, r=10),
                showlegend=False,
                yaxis_visible=False,
                xaxis_title="Day"
            )
            st.plotly_chart(fig, use_container_width=True)
        elif not PLOTLY_OK:
            st.info("Install plotly for charts: pip install plotly")
        else:
            st.info("Generate a 7-day task plan to see this chart.")

    st.markdown("---")

    # ── Row 2 — Skill radar + Profile summary ─────────────────
    col_left2, col_right2 = st.columns(2)

    with col_left2:
        st.markdown("### 🎯 Skill Radar")
        if PLOTLY_OK and st.session_state.analysis:
            matched = st.session_state.analysis["matched_skills"]
            missing = st.session_state.analysis["missing_skills"]

            all_skills = matched + missing
            values     = [90] * len(matched) + [20] * len(missing)

            if all_skills:
                fig = go.Figure(go.Scatterpolar(
                    r=values + [values[0]],
                    theta=all_skills + [all_skills[0]],
                    fill="toself",
                    fillcolor="rgba(29,158,117,0.2)",
                    line=dict(color="#1D9E75"),
                    name="Your Skills"
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    height=300,
                    margin=dict(t=10, b=10, l=40, r=40),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run Resume Analyzer first to see your skill radar.")

    with col_right2:
        st.markdown("### 👤 Profile Summary")

        st.markdown(f"**Goal:** {user['goal']}")
        st.markdown(f"**Experience:** {user['experience'].title()}")
        st.markdown(f"**Member since:** {user['created_at']}")
        st.markdown(f"**Last active:** {user['last_active']}")

        if user["current_skills"]:
            st.markdown("**Your Skills:**")
            skill_html = " ".join([
                f"<span style='background:#E1F5EE;color:#085041;padding:3px 10px;"
                f"border-radius:12px;font-size:12px;margin:2px;display:inline-block'>"
                f"{s}</span>"
                for s in user["current_skills"]
            ])
            st.markdown(skill_html, unsafe_allow_html=True)

        if st.session_state.analysis:
            st.markdown(f"**Match Score:** {st.session_state.analysis['match_score']}%")
            st.progress(st.session_state.analysis["match_score"] / 100)

    st.markdown("---")

    # ── Recent Activity ───────────────────────────────────────
    st.markdown("### 📋 Recent Activity")
    completed = user["progress"]["completed_tasks"]

    if not completed:
        st.info("No activity yet. Use Resume Analyzer, Chat, or complete tasks to earn points!")
    else:
        for t in reversed(completed[-8:]):
            col_t, col_s = st.columns([5, 1])
            with col_t:
                st.markdown(f"`{t['completed_at']}` — {t['task']}")
            with col_s:
                st.markdown("🏅 +pts")

    st.markdown("---")

    # ── Download Progress Report ──────────────────────────────
    st.markdown("### 📥 Download Progress Report")

    report = {
        "user":     st.session_state.current_user,
        "goal":     user["goal"],
        "score":    score,
        "sessions": sessions,
        "level":    level,
        "skills":   user["current_skills"],
        "tasks_completed": done,
        "weekly_tasks_done": weekly_done,
        "analysis": st.session_state.analysis,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M")
    }

    st.download_button(
        label="📥 Download My Progress Report (JSON)",
        data=json.dumps(report, indent=2),
        file_name=f"{st.session_state.current_user}_progress_report.json",
        mime="application/json",
        use_container_width=True
    )

st.markdown("---")
st.caption("🚀 AI Career Copilot —| RAG + Agents + Groq AI + Full Dashboard")