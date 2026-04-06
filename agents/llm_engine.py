# agents/llm_engine.py
# Using Groq (Free) — Llama 3.3 70B model
# 1000 requests/day, no credit card, fastest free LLM available

import os
import sys
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, ".."))

load_dotenv()

print("llm_engine.py loaded — Groq backend")

# ── Setup Groq client ─────────────────────────────────────────
try:
    from groq import Groq
    GROQ_KEY = os.getenv("GROQ_API_KEY", "")

    print("GROQ KEY:", GROQ_KEY)

    if GROQ_KEY:
        CLIENT = Groq(api_key=GROQ_KEY)
        LLM_AVAILABLE = True
        print("Groq client ready")
    else:
        CLIENT = None
        LLM_AVAILABLE = False
        print("No GROQ_API_KEY found in .env")

    print("LLM_AVAILABLE:", LLM_AVAILABLE)

except ImportError:
    CLIENT = None
    LLM_AVAILABLE = False
    print("groq package not installed — run: pip install groq")


def ask_claude(prompt: str, max_tokens: int = 1024):
    print("🔥 GROQ FUNCTION CALLED")
    """
    Sends prompt to Groq (Llama 3.3 70B) and returns response.
    Named ask_claude so nothing else in the app needs to change.

    Free tier: 1000 requests/day, 30 requests/minute
    Model    : llama-3.3-70b-versatile
    """
    if not LLM_AVAILABLE or CLIENT is None:
        return _fallback_response(prompt)

    try:
        response = CLIENT.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert AI Career Mentor. Give personalised, actionable career advice. Be specific, warm, and encouraging. Use bullet points for clarity."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content

    except Exception as e:
        error = str(e).lower()
        if "rate_limit" in error or "429" in error:
            return "Rate limit reached (30 req/min). Wait 60 seconds and try again. Daily limit: 1000 requests."
        if "invalid_api_key" in error or "401" in error:
            return "Invalid Groq API key. Check your .env file — key should start with gsk_"
        return f"Groq error: {str(e)}\n\n{_fallback_response(prompt)}"


def _fallback_response(prompt: str):
    """
    Works without any API key.
    App stays usable during testing even without internet.
    """
    prompt_lower = prompt.lower()

    if "roadmap" in prompt_lower:
        return """**Your 4-Week Career Roadmap**

**Week 1 — Foundation**
- Day 1-2: Read official docs for your top missing skill
- Day 3-4: Watch a beginner YouTube tutorial (search: skill + crash course)
- Day 5-7: Complete one beginner hands-on exercise

**Week 2 — Practice**
- Day 1-3: Build a small project using your new skill
- Day 4-7: Solve 5 problems on HackerRank or LeetCode

**Week 3 — Build**
- Build one portfolio project combining old and new skills
- Document everything on GitHub with a clear README

**Week 4 — Apply**
- Update resume with new skills and GitHub project link
- Apply to 3 matching job postings
- Prepare answers for 5 common interview questions

*Add your Groq API key to .env for AI-personalised roadmap.*"""

    if "task" in prompt_lower:
        return """**Today's Learning Task**

Spend 30 minutes reading the official documentation for your target skill,
then 60 minutes building a small working example from scratch.
Push your code to GitHub when finished.

Free resource: https://roadmap.sh

*Add GROQ_API_KEY to .env for personalised AI tasks.*"""

    return """**Career Guidance**

Here are three actions to take this week:

- **Build** one real project using your existing skills
- **Learn** one missing skill from the job description — 1 hour/day is enough  
- **Document** everything on GitHub — your portfolio is your resume

Free resources:
- https://roadmap.sh — structured learning paths
- https://freecodecamp.org — free full courses
- https://learngitbranching.js.org — learn Git interactively

*Add GROQ_API_KEY to .env for personalised AI career mentoring.*"""