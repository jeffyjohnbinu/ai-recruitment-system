"""
Skill & Role Synonym Tables
-----------------------------
Static lookup tables mapping the many ways a skill or role can be
written in a job posting to one canonical form, so a JD asking for
"JS" and one asking for "Javascript" both normalize to "JavaScript",
and "Software Engineer II" / "SWE" / "SDE" all normalize to
"Software Engineer".

These tables are intentionally simple dict-of-lists structures so they
are easy to extend as new postings surface new phrasing — no code
changes needed, just add an alias to the relevant list.
"""

from __future__ import annotations

from typing import Dict, List

# --------------------------------------------------------------------- #
# Skills: canonical name -> list of aliases / abbreviations / variants.
# Matching is case-insensitive and uses word-boundary regex, so entries
# don't need every capitalization variant.
# --------------------------------------------------------------------- #
SKILL_ALIASES: Dict[str, List[str]] = {
    "JavaScript": ["javascript", "js", "java script", "ecmascript", "es6"],
    "TypeScript": ["typescript", "ts"],
    "Python": ["python", "py"],
    "Java": ["java"],
    "C++": ["c++", "cpp", "c plus plus"],
    "C#": ["c#", "c sharp", "csharp"],
    "Go": ["golang", "go lang", " go "],
    "Ruby": ["ruby"],
    "PHP": ["php"],
    "SQL": ["sql", "structured query language"],
    "NoSQL": ["nosql", "no-sql"],
    "React": ["react", "react.js", "reactjs"],
    "Angular": ["angular", "angular.js", "angularjs"],
    "Vue.js": ["vue", "vue.js", "vuejs"],
    "Node.js": ["node", "node.js", "nodejs"],
    "Django": ["django"],
    "Flask": ["flask"],
    "FastAPI": ["fastapi", "fast api"],
    "Spring Boot": ["spring boot", "spring framework", "springboot"],
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure", "microsoft azure"],
    "GCP": ["gcp", "google cloud", "google cloud platform"],
    "Docker": ["docker", "containerization"],
    "Kubernetes": ["kubernetes", "k8s"],
    "Terraform": ["terraform", "iac", "infrastructure as code"],
    "CI/CD": ["ci/cd", "ci-cd", "continuous integration", "continuous deployment"],
    "Git": ["git", "github", "gitlab", "version control"],
    "REST APIs": ["rest api", "rest apis", "restful", "restful api", "rest"],
    "GraphQL": ["graphql"],
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning", "dl"],
    "Natural Language Processing": ["nlp", "natural language processing"],
    "Data Analysis": ["data analysis", "data analytics"],
    "Data Engineering": ["data engineering", "etl", "etl pipelines"],
    "PyTorch": ["pytorch", "torch"],
    "TensorFlow": ["tensorflow", "tf"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "PostgreSQL": ["postgresql", "postgres"],
    "MySQL": ["mysql"],
    "MongoDB": ["mongodb", "mongo"],
    "Redis": ["redis"],
    "Kafka": ["kafka", "apache kafka"],
    "Spark": ["spark", "apache spark", "pyspark"],
    "Agile/Scrum": ["agile", "scrum", "kanban"],
    "Project Management": ["project management", "pm"],
    "Product Management": ["product management"],
    "UI/UX Design": ["ui/ux", "ui design", "ux design", "user experience design"],
    "Figma": ["figma"],
    "Excel": ["excel", "microsoft excel", "ms excel"],
    "Salesforce": ["salesforce"],
    "Tableau": ["tableau"],
    "Power BI": ["power bi", "powerbi"],
    "Linux": ["linux", "unix"],
    "Bash": ["bash", "shell scripting", "shell script"],
    "Swift": ["swift"],
    "Kotlin": ["kotlin"],
    "Android Development": ["android", "android development"],
    "iOS Development": ["ios", "ios development"],
    "Cybersecurity": ["cybersecurity", "cyber security", "infosec", "information security"],
    "Communication": ["communication skills", "strong communication", "communication"],
    "Leadership": ["leadership", "team leadership", "people management"],
    "Problem Solving": ["problem solving", "problem-solving", "analytical skills"],
}

# Reverse lookup: alias (lowercase) -> canonical name, longest aliases
# first so multi-word aliases match before shorter substrings do.
_SKILL_LOOKUP: Dict[str, str] = {}
for _canonical, _aliases in SKILL_ALIASES.items():
    for _alias in _aliases:
        _SKILL_LOOKUP[_alias.strip().lower()] = _canonical


def skill_lookup() -> Dict[str, str]:
    """Alias (lowercase, trimmed) -> canonical skill name."""
    return _SKILL_LOOKUP


# --------------------------------------------------------------------- #
# Roles: canonical role title -> list of aliases / level variants /
# abbreviations commonly seen in postings.
# --------------------------------------------------------------------- #
ROLE_ALIASES: Dict[str, List[str]] = {
    "Software Engineer": [
        "software engineer",
        "swe",
        "sde",
        "software developer",
        "programmer",
        "software development engineer",
    ],
    "Backend Engineer": [
        "backend engineer",
        "backend developer",
        "back-end engineer",
        "back end developer",
    ],
    "Frontend Engineer": [
        "frontend engineer",
        "frontend developer",
        "front-end engineer",
        "front end developer",
    ],
    "Full Stack Engineer": [
        "full stack engineer",
        "full-stack engineer",
        "fullstack developer",
        "full stack developer",
    ],
    "DevOps Engineer": ["devops engineer", "dev ops engineer", "site reliability engineer", "sre"],
    "Data Scientist": ["data scientist"],
    "Data Engineer": ["data engineer"],
    "Data Analyst": ["data analyst"],
    "Machine Learning Engineer": ["machine learning engineer", "ml engineer", "ai engineer"],
    "Product Manager": ["product manager", "pm", "associate product manager", "apm"],
    "Project Manager": ["project manager", "technical project manager", "tpm"],
    "Engineering Manager": ["engineering manager", "em", "head of engineering"],
    "QA Engineer": ["qa engineer", "quality assurance engineer", "test engineer", "sdet"],
    "UX Designer": ["ux designer", "user experience designer"],
    "UI Designer": ["ui designer", "visual designer"],
    "Product Designer": ["product designer"],
    "Business Analyst": ["business analyst", "ba"],
    "Sales Representative": ["sales representative", "sales rep", "account executive", "ae"],
    "Marketing Manager": ["marketing manager", "digital marketing manager"],
    "HR Manager": ["hr manager", "human resources manager", "people operations manager"],
    "Recruiter": ["recruiter", "talent acquisition specialist", "technical recruiter"],
    "Customer Support Specialist": [
        "customer support specialist",
        "customer success manager",
        "support engineer",
    ],
    "Solutions Architect": ["solutions architect", "cloud architect", "technical architect"],
    "Security Engineer": ["security engineer", "cybersecurity engineer", "infosec engineer"],
    "Mobile Engineer": ["mobile engineer", "mobile developer", "app developer"],
}

_ROLE_LOOKUP: Dict[str, str] = {}
for _canonical, _aliases in ROLE_ALIASES.items():
    for _alias in _aliases:
        _ROLE_LOOKUP[_alias.strip().lower()] = _canonical


def role_lookup() -> Dict[str, str]:
    """Alias (lowercase, trimmed) -> canonical role title."""
    return _ROLE_LOOKUP


# --------------------------------------------------------------------- #
# Seniority / level keywords, checked against the role title line.
# --------------------------------------------------------------------- #
SENIORITY_KEYWORDS: Dict[str, List[str]] = {
    "Intern": ["intern", "internship", "trainee"],
    "Entry-Level": ["entry level", "entry-level", "junior", "jr.", "jr ", "associate", "graduate"],
    "Mid-Level": ["mid level", "mid-level", "ii", "2"],
    "Senior": ["senior", "sr.", "sr ", "iii", "3", "lead"],
    "Staff": ["staff", "principal"],
    "Manager": ["manager", "head of", "director", "vp", "vice president"],
    "Executive": ["chief", "cto", "ceo", "cfo", "coo"],
}
