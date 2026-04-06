# agents/advisor.py

def analyze_skill_gap(current_skills: list, job_text: str):
    tech_keywords = {
        "python", "sql", "docker", "kubernetes", "aws", "azure", "gcp",
        "fastapi", "flask", "django", "react", "nodejs", "java",
        "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy",
        "git", "linux", "mongodb", "postgresql", "redis", "spark"
    }

    job_words = set(job_text.lower().split())
    user_skill_set = set(s.lower() for s in current_skills)

    required_in_job = tech_keywords & job_words
    matched = required_in_job & user_skill_set
    missing = required_in_job - user_skill_set

    total = len(required_in_job)
    score = round((len(matched) / total * 100) if total > 0 else 0, 1)

    if score >= 80:
        verdict = "Strong match"
        advice = "You are well-qualified. Focus on polishing your resume and interview prep."
    elif score >= 50:
        verdict = "Moderate match"
        advice = f"You match {score}% of requirements. Learn {', '.join(list(missing)[:3])} to strengthen your profile."
    else:
        verdict = "Needs work"
        advice = f"Only {score}% match. Prioritise learning: {', '.join(list(missing)[:5])}."

    resources = _suggest_resources(list(missing)[:5])

    return {
        "matched_skills": sorted(matched),
        "missing_skills": sorted(missing),
        "match_score": score,
        "verdict": verdict,
        "advice": advice,
        "resources": resources
    }


def _suggest_resources(skills: list):
    resource_map = {
        "python":           "https://docs.python.org/3/tutorial",
        "sql":              "https://sqlzoo.net",
        "docker":           "https://docs.docker.com/get-started",
        "kubernetes":       "https://kubernetes.io/docs/tutorials",
        "aws":              "https://aws.amazon.com/free",
        "azure":            "https://learn.microsoft.com/azure",
        "fastapi":          "https://fastapi.tiangolo.com/tutorial",
        "tensorflow":       "https://www.tensorflow.org/tutorials",
        "pytorch":          "https://pytorch.org/tutorials",
        "react":            "https://react.dev/learn",
        "machine learning": "https://www.coursera.org/learn/machine-learning",
        "deep learning":    "https://www.deeplearning.ai/courses",
        "git":              "https://learngitbranching.js.org",
        "linux":            "https://linuxjourney.com",
        "pandas":           "https://pandas.pydata.org/docs/getting_started",
        "numpy":            "https://numpy.org/learn"
    }
    return {
        skill: resource_map.get(
            skill, "https://www.google.com/search?q=learn+" + skill
        )
        for skill in skills
    }