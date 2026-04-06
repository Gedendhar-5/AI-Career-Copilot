# memory/user_memory.py
# Handles all user profile and progress storage using JSON on D drive

import json
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "users.json")

def _load_all():
    """Load entire users file from disk."""
    if not os.path.exists(DATA_PATH):
        return {}

    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}   # prevents crash if file is empty/corrupt


def _save_all(data: dict):
    """Save entire users file to disk."""
    os.makedirs("data", exist_ok=True)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def create_user(username: str, goal: str, current_skills: list, experience: str):
    """
    Create a new user profile and save to D drive.
    Args:
        username: unique name
        goal: career goal e.g. 'Become a Data Scientist'
        current_skills: list of skills e.g. ['Python', 'SQL']
        experience: 'beginner' / 'intermediate' / 'advanced'
    """
    all_users = _load_all()

    all_users[username] = {
        "username": username,
        "goal": goal,
        "current_skills": current_skills,
        "experience": experience,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "last_active": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "progress": {
            "completed_tasks": [],
            "sessions": 0,
            "skill_score": 0
        }
    }

    _save_all(all_users)
    return all_users[username]


def get_user(username: str):
    """Fetch a user profile. Returns None if not found."""
    all_users = _load_all()
    return all_users.get(username, None)


def update_progress(username: str, task: str = None, skill_score: int = 0):
    """
    Update user progress after a session.
    Args:
        username: the user
        task: completed task string (optional)
        skill_score: points to add
    """
    all_users = _load_all()

    if username not in all_users:
        return None

    user = all_users[username]
    user["last_active"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    user["progress"]["sessions"] += 1
    user["progress"]["skill_score"] += skill_score

    if task:
        user["progress"]["completed_tasks"].append({
            "task": task,
            "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M")
        })

    all_users[username] = user
    _save_all(all_users)
    return user


def list_users():
    """Return list of all usernames."""
    return list(_load_all().keys())


def delete_user(username: str):
    """Delete a user profile."""
    all_users = _load_all()
    if username in all_users:
        del all_users[username]
        _save_all(all_users)
        return True
    return False