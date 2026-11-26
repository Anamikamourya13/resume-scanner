
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field
from pypdf import PdfReader
from docx import Document


# ==========================================
# 1. GROQ API SETUP
# ==========================================

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("GROQ_API_KEY .env file mein nahi mili.")

client = Groq(api_key=my_api_key)
model = "openai/gpt-oss-120b"


# ==========================================
# 2. JOB DESCRIPTION
# ==========================================

job_description = """
Amazon Software Development Engineer (SDE-I)

Responsibilities:
- Collaborate with teams to design, build, and operate products.
- Develop scalable cloud-native solutions and microservices.
- Participate in code reviews and technical documentation.
- Build resilient and fault-tolerant distributed systems.
- Use AI-powered tools to improve productivity.
- Write clean and maintainable code.
- Follow Agile and CI/CD practices.
- Monitor and resolve production issues.

Basic Qualifications:
- Experience with Python, Java, C++, or another programming language.
- Knowledge of data structures and algorithms.
- Bachelor's degree in a relevant STEM field.
- Must be at least 18 years old.

Preferred Qualifications:
- Technical internship or project experience.
- Knowledge of AI tools, AWS, databases, Git and debugging.
- Ability to learn new technologies.
- Understanding of SDLC.
- Strong problem-solving and communication skills.
"""


# ==========================================
# 3. PYDANTIC MODELS
# ==========================================

class JobD(BaseModel):
    role: str
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    minimum_experience: float | None = None
    education_requirements: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)


class MatchResult(BaseModel):
    score: float
    details: dict


class Experience(BaseModel):
    company: str | None = None
    role: str | None = None
    duration: str | None = None
    description: str | None = None
    skills_used: list[str] = Field(default_factory=list)


class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    total_experience_years: float | None = None
    skills: list[str] = Field(default_factory=list)
    experiences: list[Experience] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)


# ==========================================
# 4. EXTRACT JOB DETAILS
# ==========================================

jobd_schema = JobD.model_json_schema()

job_system_prompt = f"""
You are an expert HR assistant.

Extract structured information from the job description.

Return ONLY valid JSON matching this schema:

{json.dumps(jobd_schema, indent=2)}

Rules:
- Extract actual information.
- Do not return the schema itself.
- If experience is not mentioned, return null.
- If information is missing, return an empty list.
- Do not invent information.
"""

job_response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": job_system_prompt},
        {
            "role": "user",
            "content": f"Analyze this job description:\n{job_description}"
        }
    ],
    response_format={"type": "json_object"}
)

job_data = json.loads(
    job_response.choices[0].message.content
)

job = JobD(**job_data)

print("\nJOB DETAILS EXTRACTED")
print("Role:", job.role)
print(
    "Minimum Experience:",
    job.minimum_experience
    if job.minimum_experience is not None
    else "Not specified"
)
print("Required Skills:", job.required_skills)


# ==========================================
# 5. READ PDF AND DOCX FILES
# ==========================================

def read_pdf(file_path: Path) -> str:
    """Extract text from PDF."""

    reader = PdfReader(str(file_path))
    text_parts = []

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text_parts.append(page_text)

    return "\n".join(text_parts)


def read_docx(file_path: Path) -> str:
    """Extract text from DOCX."""

    document = Document(str(file_path))
    text_parts = []

    # Read paragraphs
    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            text_parts.append(paragraph.text)

    # Read tables
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    text_parts.append(cell.text)

    return "\n".join(text_parts)


def read_resume(file_path: Path) -> str:

    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return read_pdf(file_path)

    elif extension == ".docx":
        return read_docx(file_path)

    return ""


# ==========================================
# 6. PARSE RESUME USING GROQ
# ==========================================

resume_schema = Resume.model_json_schema()


def parse_resume(resume_text: str) -> Resume:

    system_prompt = f"""
You are an expert resume parser.

Extract information based on meaning,
not only exact section headings.

Return ONLY valid JSON matching this schema:

{json.dumps(resume_schema, indent=2)}

Rules:
1. Do not invent information.
2. Missing values should be null.
3. Missing lists should be empty.
4. Include internships inside experiences.
5. Extract skills from the entire resume.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"Parse this resume:\n{resume_text}"
            }
        ],
        response_format={"type": "json_object"}
    )

    raw_output = response.choices[0].message.content

    data = json.loads(raw_output)

    return Resume(**data)


# ==========================================
# 7. COMPARE RESUME WITH JOB DESCRIPTION
# ==========================================

def final_score(job: JobD, resume: Resume) -> MatchResult:

    match_schema = MatchResult.model_json_schema()

    prompt = f"""
You are an expert HR recruiter.

Compare the candidate's resume with the job description.

JOB DESCRIPTION:
{job.model_dump_json(indent=2)}

CANDIDATE RESUME:
{resume.model_dump_json(indent=2)}

Return ONLY valid JSON matching this schema:

{json.dumps(match_schema, indent=2)}

The details object must contain:

1. candidate_name
2. matching_skills
3. missing_important_skills
4. experience_requirement_met
5. overall_match_percentage (0 to 100)
6. final_verdict

Evaluate:
- Required skills
- Relevant experience
- Education
- Projects
- Technical knowledge

Base the percentage on resume evidence.
Do not invent qualifications.

IMPORTANT:
Use the same percentage in:
score
details.overall_match_percentage
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={"type": "json_object"}
    )

    data = json.loads(
        response.choices[0].message.content
    )

    details = data.get("details", {})

    detail_score = details.get(
        "overall_match_percentage"
    )

    if detail_score is None:
        raise ValueError(
            "overall_match_percentage missing in AI response"
        )

    score = float(detail_score)

    if not 0 <= score <= 100:
        raise ValueError("Invalid match percentage")

    # Keep one consistent score for ranking
    details["overall_match_percentage"] = score
    data["score"] = score
    data["details"] = details

    return MatchResult(**data)


# ==========================================
# 8. PROCESS ALL RESUMES
# ==========================================

resume_folder = Path("resumes")

all_results = []

if not resume_folder.exists():
    raise FileNotFoundError(
        "resumes folder nahi mila. "
        "Project root mein resumes folder banao."
    )

resume_files = sorted(
    file_path
    for file_path in resume_folder.iterdir()
    if file_path.is_file()
    and file_path.suffix.lower() in [".pdf", ".docx"]
)

if not resume_files:
    raise FileNotFoundError(
        "resumes folder mein PDF ya DOCX file nahi mili."
    )

print("\nTOTAL RESUMES FOUND:", len(resume_files))


for file_path in resume_files:

    print("\n" + "=" * 50)
    print("Processing:", file_path.name)

    try:

        # Read resume
        resume_text = read_resume(file_path)

        # Skip empty resumes
        if not resume_text or not resume_text.strip():
            print("Skipped: No readable text found.")
            continue

        # Parse resume
        parsed_resume = parse_resume(resume_text)

        time.sleep(2)

        # Calculate match score
        result = final_score(job, parsed_resume)

        print(
            "Candidate:",
            parsed_resume.name or file_path.stem
        )

        print("Match Score:", f"{result.score:.1f}%")

        # Save result
        all_results.append({
            "name": parsed_resume.name or file_path.stem,
            "score": result.score,
            "details": result.details
        })

        time.sleep(2)

    except Exception as error:

        print("Could not process:", file_path.name)
        print("Reason:", str(error))

        continue


# ==========================================
# 9. RANK CANDIDATES
# ==========================================

if not all_results:

    print("\nNo resumes processed successfully.")

else:

    # Highest score first
    all_results.sort(
        key=lambda candidate: candidate["score"],
        reverse=True
    )

    # Top 2 candidates
    top_2 = all_results[:2]

    # Lowest 2 candidates
    worst_2 = sorted(
        all_results,
        key=lambda candidate: candidate["score"]
    )[:2]

    print("\n" + "=" * 50)
    print("TOP 2 CANDIDATES")
    print("=" * 50)

    for candidate in top_2:

        print(
            candidate["name"],
            "-",
            f'{candidate["score"]:.1f}%'
        )

        print(
            json.dumps(
                candidate["details"],
                indent=2
            )
        )

    print("\n" + "=" * 50)
    print("LOWEST 2 CANDIDATES")
    print("=" * 50)

    for candidate in worst_2:

        print(
            candidate["name"],
            "-",
            f'{candidate["score"]:.1f}%'
        )

    print("\nResume screening completed.")
