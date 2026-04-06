# agents/task_generator.py
print("🔥 TASK GENERATOR FILE LOADED")
import os
import json
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASKS_PATH = os.path.join(BASE_DIR, "..", "data", "tasks.json")


def _load_tasks():
    if not os.path.exists(TASKS_PATH):
        return {}
    try:
        with open(TASKS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}


def _save_tasks(data: dict):
    os.makedirs(os.path.dirname(TASKS_PATH), exist_ok=True)
    with open(TASKS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def generate_tasks(username: str, missing_skills: list, experience: str):
    """
    Generate a 7-day daily task plan based on missing skills.

    Args:
        username: user identifier
        missing_skills: list of skills to focus on
        experience: beginner / intermediate / advanced

    Returns:
        list of 7 daily task dicts
    """
    if not missing_skills:
        missing_skills = ["problem solving", "system design"]

    # Task templates per skill
    task_templates = {
        "python":           ["Complete Python basics tutorial", "Write 3 Python functions", "Build a CLI tool in Python"],
        "sql":              ["Practice 10 SQL queries on SQLZoo", "Write JOIN queries", "Design a small database schema"],
        "docker":           ["Install Docker and run hello-world", "Dockerise a Python app", "Learn docker-compose basics"],
        "kubernetes":       ["Read Kubernetes core concepts", "Set up minikube locally", "Deploy a pod on minikube"],
        "aws":              ["Create a free AWS account", "Launch an EC2 instance", "Upload files to S3"],
        "fastapi":          ["Build a FastAPI hello-world app", "Add 3 API endpoints", "Connect FastAPI to SQLite"],
        "machine learning": ["Complete ML crash course Day 1", "Train a linear regression model", "Evaluate model accuracy"],
        "deep learning":    ["Read intro to neural networks", "Build a simple NN with PyTorch", "Train on MNIST dataset"],
        "git":              ["Learn git init, add, commit", "Push a repo to GitHub", "Practice branching and merging"],
        "pandas":           ["Load a CSV with pandas", "Clean and filter data", "Plot data with matplotlib"],
        "react":            ["Build a React hello-world app", "Create 3 React components", "Handle state with useState"],
        "linux":            ["Learn basic Linux commands", "Practice file permissions", "Write a bash script"],
        "tensorflow":       ["Install TensorFlow and run test", "Build a simple model", "Train on sample dataset"],
        "pytorch":          ["Install PyTorch and run test", "Build a neural network", "Train on MNIST"],
    }

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    today = datetime.now()
    task_list = []

    for i in range(7):
        skill = missing_skills[i % len(missing_skills)]
        templates = task_templates.get(
            skill.lower(),
            [f"Study {skill} basics", f"Practice {skill}", f"Build a {skill} mini project"]
        )
        task_desc = templates[i % len(templates)]

        task_list.append({
            "day": days[i],
            "date": (today + timedelta(days=i)).strftime("%Y-%m-%d"),
            "skill_focus": skill,
            "task": task_desc,
            "difficulty": _get_difficulty(experience, i),
            "estimated_hours": 1 if experience == "beginner" else 2,
            "completed": False
        })

    # Save to D drive
    all_tasks = _load_tasks()
    all_tasks[username] = {
        "generated_at": today.strftime("%Y-%m-%d %H:%M"),
        "tasks": task_list
    }
    _save_tasks(all_tasks)

    return task_list


def get_tasks(username: str):
    """Retrieve saved tasks for a user."""
    all_tasks = _load_tasks()
    user_data = all_tasks.get(username, None)
    return user_data["tasks"] if user_data else None


def mark_task_done(username: str, day: str):
    """Mark a specific day's task as completed."""
    all_tasks = _load_tasks()
    if username not in all_tasks:
        return False
    for task in all_tasks[username]["tasks"]:
        if task["day"] == day:
            task["completed"] = True
    _save_tasks(all_tasks)
    return True


def _get_difficulty(experience: str, day_index: int):
    base = {"beginner": 1, "intermediate": 2, "advanced": 3}
    level = base.get(experience, 1) + (day_index // 3)
    labels = {1: "Easy", 2: "Medium", 3: "Hard", 4: "Hard"}
    return labels.get(min(level, 4), "Medium")