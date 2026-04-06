# rag/parser.py

print("parser.py loaded")

RESUME_SECTIONS = {
    "skills":     ["skill", "technical skill", "technologies", "tools",
                   "programming", "languages", "frameworks", "stack"],
    "experience": ["experience", "work experience", "employment",
                   "professional experience", "career"],
    "education":  ["education", "qualification", "degree", "university",
                   "college", "academic", "certification"],
    "projects":   ["project", "portfolio", "built", "developed", "created"],
    "summary":    ["summary", "objective", "about", "profile", "overview"]
}

JD_SECTIONS = {
    "required_skills":  ["required", "must have", "technical requirement",
                         "skills required", "qualifications", "you need"],
    "responsibilities": ["responsibilities", "duties", "you will",
                         "role", "what you'll do"],
    "nice_to_have":     ["nice to have", "preferred", "bonus", "desirable"],
    "about_role":       ["about", "overview", "description", "we are looking"]
}

TECH_SKILLS = {
    "python", "sql", "java", "javascript", "typescript", "c++", "c#",
    "go", "rust", "docker", "kubernetes", "aws", "azure", "gcp",
    "terraform", "ansible", "fastapi", "flask", "django", "react",
    "nodejs", "vue", "angular", "tensorflow", "pytorch", "scikit-learn",
    "pandas", "numpy", "matplotlib", "git", "linux", "bash", "jenkins",
    "mongodb", "postgresql", "mysql", "redis", "elasticsearch",
    "machine learning", "deep learning", "nlp", "computer vision",
    "rest", "api", "graphql", "microservices", "spark", "kafka",
    "bert", "hugging face", "snowflake", "power bi", "tableau",
    "excel", "jira", "agile", "scrum", "openCV", "github actions"
}


def _detect_section(line: str, section_map: dict) -> str:
    line_lower = line.lower().strip()
    for section, keywords in section_map.items():
        for kw in keywords:
            if kw in line_lower:
                return section
    return "general"


def _parse_text(text: str, section_map: dict, default_sections: dict) -> dict:
    sections = {k: [] for k in default_sections}
    sections["general"] = []
    current_section = "general"
    buffer = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        detected = _detect_section(line, section_map)
        if detected != "general" and len(line) < 80:
            if buffer:
                chunk = " ".join(buffer).strip()
                if chunk:
                    sections[current_section].append(chunk)
                buffer = []
            current_section = detected
        else:
            buffer.append(line)

    if buffer:
        chunk = " ".join(buffer).strip()
        if chunk:
            sections[current_section].append(chunk)

    return sections


def parse_resume(text: str) -> dict:
    return _parse_text(text, RESUME_SECTIONS, {
        "skills": [], "experience": [], "education": [],
        "projects": [], "summary": []
    })


def parse_job_description(text: str) -> dict:
    return _parse_text(text, JD_SECTIONS, {
        "required_skills": [], "responsibilities": [],
        "nice_to_have": [], "about_role": []
    })


def extract_skills_from_text(text: str) -> list:
    text_lower = text.lower()
    return sorted([s for s in TECH_SKILLS if s in text_lower])


def build_structured_chunks(resume_text: str, jd_text: str) -> list:
    chunks = []

    if resume_text.strip():
        resume_sections = parse_resume(resume_text)
        for section, texts in resume_sections.items():
            for text in texts:
                if len(text) > 20:
                    chunks.append({
                        "label":   f"Resume — {section.replace('_', ' ').title()}",
                        "section": section,
                        "text":    text,
                        "source":  "resume"
                    })

    if jd_text.strip():
        jd_sections = parse_job_description(jd_text)
        for section, texts in jd_sections.items():
            for text in texts:
                if len(text) > 20:
                    chunks.append({
                        "label":   f"Job Description — {section.replace('_', ' ').title()}",
                        "section": section,
                        "text":    text,
                        "source":  "jd"
                    })

    # Fallback — if parser found no structure use sentence splitting
    if len(chunks) < 3:
        for source, raw_text in [("resume", resume_text), ("jd", jd_text)]:
            if not raw_text.strip():
                continue
            sentences = [
                s.strip() for s in raw_text.replace("\n", ". ").split(". ")
                if len(s.strip()) > 30
            ]
            for i, sentence in enumerate(sentences):
                chunks.append({
                    "label":   f"{source.title()} — chunk {i+1}",
                    "section": "general",
                    "text":    sentence,
                    "source":  source
                })

    return chunks