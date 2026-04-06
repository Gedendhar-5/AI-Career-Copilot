# agents/prompt_builder.py

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, ".."))

print("prompt_builder.py loaded")


def build_career_prompt(user: dict, query: str, rag_results: list, analysis: dict = None):
    """
    Combines user profile + RAG chunks + skill gap
    into one structured prompt for Gemini.
    """
    rag_context = "\n".join([
        f"- {chunk}" for chunk in rag_results
    ]) if rag_results else "No resume or job description provided yet."

    if analysis:
        matched  = ", ".join(analysis.get("matched_skills", [])) or "None"
        missing  = ", ".join(analysis.get("missing_skills", [])) or "None"
        score    = analysis.get("match_score", 0)
        advice   = analysis.get("advice", "")
        gap_text = f"""
Skill Gap Analysis:
- Match Score   : {score}%
- Matched Skills: {matched}
- Missing Skills: {missing}
- Advisor Note  : {advice}
"""
    else:
        gap_text = "Skill gap analysis not yet run."

    prompt = f"""You are an expert AI Career Mentor helping a user with their career growth.

=== USER PROFILE ===
Name            : {user.get('username', 'User')}
Career Goal     : {user.get('goal', 'Not specified')}
Experience Level: {user.get('experience', 'beginner')}
Current Skills  : {', '.join(user.get('current_skills', [])) or 'Not specified'}
Sessions Done   : {user.get('progress', {}).get('sessions', 0)}
Skill Score     : {user.get('progress', {}).get('skill_score', 0)} pts

=== CONTEXT FROM RESUME AND JOB DESCRIPTION ===
{rag_context}

=== SKILL GAP ===
{gap_text}

=== USER QUESTION ===
{query}

=== YOUR INSTRUCTIONS ===
- Give a warm, personalised, and actionable response
- Mention the user's actual skills and goal by name
- Be specific — avoid generic advice
- Suggest only FREE learning resources when relevant
- Keep response under 250 words
- Use bullet points for clarity
- End with one encouraging sentence

Your response:"""

    return prompt


def build_roadmap_prompt(user: dict, missing_skills: list):
    """Prompt for generating a week-by-week career roadmap."""
    skills_str = ", ".join(missing_skills) if missing_skills else "core programming fundamentals"

    prompt = f"""You are an expert career coach building a personalised learning roadmap.

USER PROFILE:
- Career Goal    : {user.get('goal', 'Software Developer')}
- Experience     : {user.get('experience', 'beginner')}
- Current Skills : {', '.join(user.get('current_skills', [])) or 'None listed'}
- Skills to Learn: {skills_str}

Create a clear 4-week roadmap:
- Each week: 3 to 5 specific daily tasks
- Use only FREE resources (YouTube, official docs, freeCodeCamp, roadmap.sh)
- Be realistic — 1 to 2 hours per day max
- Format exactly as: Week 1, Week 2, Week 3, Week 4
- End with one motivating sentence

Roadmap:"""

    return prompt


def build_task_prompt(user: dict, skill: str, day: int):
    """Prompt for a single day learning task."""
    prompt = f"""You are an AI learning coach generating a daily task.

DETAILS:
- Skill to learn : {skill}
- User level     : {user.get('experience', 'beginner')}
- Day number     : {day}

Generate ONE specific hands-on task:
- Must be completable in 1 to 2 hours
- Must involve building or practicing — not just reading
- Include what to build or practice (specific)
- Include one FREE resource link
- Write 2 to 3 sentences only

Task:"""

    return prompt