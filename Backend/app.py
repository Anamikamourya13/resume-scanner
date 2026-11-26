"""Flask backend: serves the frontend and exposes /api/analyze."""
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from parser_core import find_skills, parse_resume, score_candidate

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB per request

DEFAULT_JD = """Amazon Software Development Engineer

Responsibilities:
- Collaborate with teams to design and ship features
- Develop scalable cloud-native services on AWS
- Participate in code reviews
- Build resilient and fault-tolerant systems
- Use AI-powered tools to improve productivity
- Write clean and maintainable code
- Follow Agile and CI/CD practices
- Monitor and resolve production issues

Basic Qualifications:
- Experience with Python, Java or C++
- Knowledge of data structures and algorithms
- Bachelor's degree in a related field
- Must be at least 18 years of age
"""


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/api/health")
def health():
    return jsonify(status="ok")


@app.get("/api/default-jd")
def default_jd():
    return jsonify(job_description=DEFAULT_JD)


@app.post("/api/analyze")
def analyze():
    jd = (request.form.get("job_description") or "").strip()
    files = request.files.getlist("resumes")
    if not jd:
        return jsonify(error="Job description is required."), 400
    if not files:
        return jsonify(error="Upload at least one resume (PDF or DOCX)."), 400

    results, errors = [], []
    for f in files:
        try:
            candidate = parse_resume(f.filename, f.read())
            results.append(score_candidate(candidate, jd))
        except Exception as exc:  # one bad file should not stop the rest
            errors.append({"file": f.filename, "error": str(exc)})
    results.sort(key=lambda r: r["score"], reverse=True)
    return jsonify(results=results, errors=errors, required_skills=find_skills(jd))


if __name__ == "__main__":
    # Local development only. In production gunicorn runs the app (see render.yaml).
    import os
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8000)),
            debug=os.environ.get("FLASK_DEBUG") == "1")