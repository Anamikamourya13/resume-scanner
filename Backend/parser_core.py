"""Resume parsing + job-description matching logic (no web code here)."""
import io
import re
from pypdf import PdfReader
from docx import Document

SKILLS = [
    "python", "java", "c++", "c#", "c", "javascript", "typescript", "go", "rust", "kotlin",
    "sql", "mysql", "postgresql", "mongodb", "redis", "dynamodb",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "linux", "git",
    "react", "node.js", "flask", "fastapi", "django", "spring", "html", "css",
    "rest", "microservices", "ci/cd", "agile", "machine learning", "deep learning",
    "nlp", "tensorflow", "pytorch", "pandas", "numpy",
    "data structures", "algorithms", "system design", "oop", "cloud",
]

DEGREE_RE = re.compile(
    r"\b(b\.?\s?tech|b\.?\s?e\b|bachelor|b\.?\s?sc|bca|m\.?\s?tech|master|m\.?\s?sc|mca|ph\.?d)",
    re.I,
)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s-]?)?(?:\d[\s-]?){9,11}\d")
YEARS_RE = re.compile(r"(\d{1,2})\+?\s*(?:years?|yrs?)", re.I)
STOP = set("the a an and or of to in for with on at by is are be as we you your will this that from".split())


def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    if name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                parts.extend(c.text for c in row.cells)
        return "\n".join(parts)
    raise ValueError("Only PDF and DOCX files are supported")


def find_skills(text: str) -> list[str]:
    found = []
    for skill in SKILLS:
        pattern = r"(?<![A-Za-z0-9+#])" + re.escape(skill) + r"(?![A-Za-z0-9+#])"
        if re.search(pattern, text, re.I):
            found.append(skill)
    return found


def guess_name(text: str, filename: str) -> str:
    for line in text.splitlines()[:8]:
        line = line.strip()
        words = line.split()
        if 2 <= len(words) <= 4 and all(re.fullmatch(r"[A-Za-z.'-]+", w) for w in words):
            if not re.search(r"resume|curriculum|vitae|profile|objective", line, re.I):
                return line.title()
    stem = re.sub(r"\.(pdf|docx)$", "", filename, flags=re.I)
    return re.sub(r"[_\-]+", " ", stem).title()


def keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z+#.]{2,}", text.lower()) if w not in STOP}


def parse_resume(filename: str, data: bytes) -> dict:
    text = extract_text(filename, data)
    if not text.strip():
        raise ValueError("No readable text found (scanned PDF?)")
    email = EMAIL_RE.search(text)
    phone = PHONE_RE.search(text)
    years = [int(y) for y in YEARS_RE.findall(text) if int(y) < 40]
    return {
        "file": filename,
        "candidate_name": guess_name(text, filename),
        "email": email.group(0) if email else None,
        "phone": phone.group(0).strip() if phone else None,
        "skills": find_skills(text),
        "has_degree": bool(DEGREE_RE.search(text)),
        "years_experience": max(years) if years else None,
        "_text": text,
    }


def score_candidate(candidate: dict, job_description: str) -> dict:
    required = find_skills(job_description)
    have = set(candidate["skills"])
    matched = [s for s in required if s in have]
    missing = [s for s in required if s not in have]
    skill_pct = len(matched) / len(required) if required else 0

    jd_kw, cv_kw = keywords(job_description), keywords(candidate["_text"])
    kw_pct = len(jd_kw & cv_kw) / len(jd_kw) if jd_kw else 0

    score = round(100 * (0.70 * skill_pct + 0.20 * min(kw_pct * 2, 1) + 0.10 * candidate["has_degree"]))
    result = {k: v for k, v in candidate.items() if k != "_text"}
    result.update(score=score, matched_skills=matched, missing_skills=missing)
    return result