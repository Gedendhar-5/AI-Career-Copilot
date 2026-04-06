import os
import json
from datetime import datetime



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLANS_PATH = os.path.join(BASE_DIR, "..", "data", "plans.json")


def _load_plans():
    if not os.path.exists(PLANS_PATH):
        return {}
    try:
        with open(PLANS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_plans(data: dict):
    os.makedirs(os.path.dirname(PLANS_PATH), exist_ok=True)
    with open(PLANS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def generate_plan(username: str, goal: str, experience: str, missing_skills: list):

    if experience == "beginner":
        duration = "6 months"
        hours_per_day = 2
    elif experience == "intermediate":
        duration = "3 months"
        hours_per_day = 3
    else:
        duration = "6 weeks"
        hours_per_day = 4

    skills = missing_skills if missing_skills else ["Core fundamentals", "Project building"]
    total = len(skills)
    chunk = max(1, total // 3)

    plan = {
        "username": username,
        "goal": goal,
        "experience": experience,
        "duration": duration,
        "hours_per_day": hours_per_day,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "phases": [
            {
                "phase": 1,
                "title": "Foundation",
                "duration": "Week 1-3",
                "skills": skills[:chunk],
                "focus": "Learn core concepts"
            },
            {
                "phase": 2,
                "title": "Practice",
                "duration": "Week 4-6",
                "skills": skills[chunk:chunk*2],
                "focus": "Build small projects"
            },
            {
                "phase": 3,
                "title": "Production",
                "duration": "Week 7+",
                "skills": skills[chunk*2:],
                "focus": "Build portfolio"
            }
        ]
    }

    all_plans = _load_plans()
    all_plans[username] = plan
    _save_plans(all_plans)

    return plan


def get_plan(username: str):
    plans = _load_plans()
    return plans.get(username, None)


print("Functions loaded:", dir())