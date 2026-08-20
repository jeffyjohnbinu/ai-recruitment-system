"""
skill_dictionary.py
--------------------
Master skill dictionary for the Skill Extraction Engine (Day 9).

Structure:
    SKILL_DICTIONARY: canonical_skill -> SkillDefinition
        - category:     "technical" | "business" | "creative"
        - subcategory:  finer grouping (e.g. "language", "cloud", "finance")
        - aliases:      alternate names / abbreviations / common spelling
                         variants that should resolve to the canonical name

    SKILL_STACKS: stack_name -> list[canonical_skill]
        Multi-skill "bundles" (MERN, MEAN, LAMP, ...) that, when mentioned
        as a single token in a resume, expand into their constituent
        skills (tagged with a lower "inferred_from_stack" confidence).

Extending the dictionary:
    Add new entries to SKILL_DICTIONARY / SKILL_STACKS. Canonical names
    are the single source of truth used everywhere downstream (ATS
    matching, JD parsing in Day 6, screening). Keep aliases lowercase;
    matching is case-insensitive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SkillDefinition:
    category: str  # "technical" | "business" | "creative"
    subcategory: str
    aliases: List[str] = field(default_factory=list)


# --------------------------------------------------------------------- #
# Technical skills
# --------------------------------------------------------------------- #
_TECHNICAL: Dict[str, SkillDefinition] = {
    "Python": SkillDefinition("technical", "language", ["python3", "py"]),
    "JavaScript": SkillDefinition(
        "technical", "language", ["js", "java script", "ecmascript", "es6"]
    ),
    "TypeScript": SkillDefinition("technical", "language", ["ts"]),
    "Java": SkillDefinition("technical", "language", []),
    "C++": SkillDefinition("technical", "language", ["cpp", "c plus plus"]),
    "C#": SkillDefinition("technical", "language", ["c sharp", "csharp"]),
    "Go": SkillDefinition("technical", "language", ["golang"]),
    "Rust": SkillDefinition("technical", "language", []),
    "PHP": SkillDefinition("technical", "language", []),
    "Ruby": SkillDefinition("technical", "language", []),
    "SQL": SkillDefinition("technical", "language", ["structured query language"]),
    "R": SkillDefinition("technical", "language", ["r programming", "r language"]),
    "Swift": SkillDefinition("technical", "language", []),
    "Kotlin": SkillDefinition("technical", "language", []),
    "Scala": SkillDefinition("technical", "language", []),
    "React": SkillDefinition("technical", "framework", ["react.js", "reactjs"]),
    "Angular": SkillDefinition("technical", "framework", ["angular.js", "angularjs"]),
    "Vue.js": SkillDefinition("technical", "framework", ["vue", "vuejs"]),
    "Node.js": SkillDefinition("technical", "framework", ["node", "nodejs", "node js"]),
    "Express.js": SkillDefinition("technical", "framework", ["express", "expressjs"]),
    "Django": SkillDefinition("technical", "framework", []),
    "Flask": SkillDefinition("technical", "framework", []),
    "FastAPI": SkillDefinition("technical", "framework", ["fast api"]),
    "Spring Boot": SkillDefinition("technical", "framework", ["spring", "springboot"]),
    ".NET": SkillDefinition("technical", "framework", ["dotnet", "dot net", "asp.net"]),
    "Next.js": SkillDefinition("technical", "framework", ["nextjs", "next js"]),
    "AWS": SkillDefinition("technical", "cloud", ["amazon web services", "amazon aws"]),
    "Azure": SkillDefinition("technical", "cloud", ["microsoft azure"]),
    "GCP": SkillDefinition("technical", "cloud", ["google cloud platform", "google cloud"]),
    "Docker": SkillDefinition("technical", "devops", []),
    "Kubernetes": SkillDefinition("technical", "devops", ["k8s"]),
    "Terraform": SkillDefinition("technical", "devops", []),
    "Jenkins": SkillDefinition("technical", "devops", []),
    "CI/CD": SkillDefinition(
        "technical", "devops", ["ci cd", "continuous integration", "continuous deployment"]
    ),
    "Git": SkillDefinition("technical", "devops", ["github", "gitlab", "version control"]),
    "Linux": SkillDefinition("technical", "devops", ["unix"]),
    "Apache": SkillDefinition("technical", "devops", ["apache http server", "apache server"]),
    "MySQL": SkillDefinition("technical", "database", ["my sql"]),
    "PostgreSQL": SkillDefinition("technical", "database", ["postgres", "postgre sql"]),
    "MongoDB": SkillDefinition("technical", "database", ["mongo", "mongo db"]),
    "Redis": SkillDefinition("technical", "database", []),
    "Elasticsearch": SkillDefinition("technical", "database", ["elastic search"]),
    "Oracle DB": SkillDefinition("technical", "database", ["oracle database", "oracle sql"]),
    "GraphQL": SkillDefinition("technical", "api", ["graph ql"]),
    "REST APIs": SkillDefinition("technical", "api", ["rest api", "restful api", "restful apis"]),
    "Machine Learning": SkillDefinition("technical", "ai_ml", ["ml", "machine-learning"]),
    "Deep Learning": SkillDefinition("technical", "ai_ml", ["dl"]),
    "Natural Language Processing": SkillDefinition(
        "technical", "ai_ml", ["nlp", "natural-language-processing"]
    ),
    "Computer Vision": SkillDefinition("technical", "ai_ml", ["cv"]),
    "TensorFlow": SkillDefinition("technical", "ai_ml", ["tensor flow"]),
    "PyTorch": SkillDefinition("technical", "ai_ml", ["torch"]),
    "scikit-learn": SkillDefinition("technical", "ai_ml", ["sklearn", "scikit learn"]),
    "Pandas": SkillDefinition("technical", "data", []),
    "NumPy": SkillDefinition("technical", "data", []),
    "Apache Spark": SkillDefinition("technical", "data", ["spark", "pyspark"]),
    "Airflow": SkillDefinition("technical", "data", ["apache airflow"]),
    "Kafka": SkillDefinition("technical", "data", ["apache kafka"]),
    "Tableau": SkillDefinition("technical", "data", []),
    "Power BI": SkillDefinition("technical", "data", ["powerbi", "power-bi"]),
    "Excel": SkillDefinition("technical", "data", ["microsoft excel", "ms excel"]),
    "pytest": SkillDefinition("technical", "testing", ["py.test"]),
    "Selenium": SkillDefinition("technical", "testing", []),
    "Jest": SkillDefinition("technical", "testing", []),
    "JUnit": SkillDefinition("technical", "testing", []),
    "Android": SkillDefinition("technical", "mobile", ["android development"]),
    "iOS": SkillDefinition("technical", "mobile", ["ios development"]),
    "Flutter": SkillDefinition("technical", "mobile", []),
    "React Native": SkillDefinition("technical", "mobile", ["react-native"]),
    "HTML": SkillDefinition("technical", "web", ["html5"]),
    "CSS": SkillDefinition("technical", "web", ["css3"]),
    "Tailwind CSS": SkillDefinition("technical", "web", ["tailwind", "tailwindcss"]),
    "Sentence-Transformers": SkillDefinition("technical", "ai_ml", ["sentence transformers"]),
}

# --------------------------------------------------------------------- #
# Business / soft skills
# --------------------------------------------------------------------- #
_BUSINESS: Dict[str, SkillDefinition] = {
    "Project Management": SkillDefinition("business", "management", ["project mgmt", "pm"]),
    "Agile": SkillDefinition("business", "management", ["agile methodology"]),
    "Scrum": SkillDefinition("business", "management", ["scrum master"]),
    "Kanban": SkillDefinition("business", "management", []),
    "Stakeholder Management": SkillDefinition("business", "management", []),
    "Product Management": SkillDefinition("business", "management", ["product mgmt"]),
    "Business Analysis": SkillDefinition("business", "analysis", ["business analyst", "ba"]),
    "Financial Modeling": SkillDefinition("business", "finance", ["financial models"]),
    "Budgeting": SkillDefinition("business", "finance", ["budget management"]),
    "Forecasting": SkillDefinition("business", "finance", ["financial forecasting"]),
    "Sales": SkillDefinition("business", "sales", ["business development", "bd"]),
    "Negotiation": SkillDefinition("business", "sales", []),
    "Marketing Strategy": SkillDefinition(
        "business", "marketing", ["marketing strategy development"]
    ),
    "SEO": SkillDefinition("business", "marketing", ["search engine optimization"]),
    "SEM": SkillDefinition("business", "marketing", ["search engine marketing"]),
    "Content Marketing": SkillDefinition("business", "marketing", []),
    "Public Speaking": SkillDefinition("business", "communication", ["presentation skills"]),
    "Leadership": SkillDefinition("business", "communication", ["team leadership"]),
    "Communication": SkillDefinition(
        "business", "communication", ["verbal communication", "written communication"]
    ),
    "Mentoring": SkillDefinition("business", "communication", ["coaching"]),
    "Client Relations": SkillDefinition(
        "business", "communication", ["client management", "account management"]
    ),
    "Problem Solving": SkillDefinition("business", "cognitive", ["problem-solving"]),
    "Critical Thinking": SkillDefinition("business", "cognitive", []),
    "Data Analysis": SkillDefinition("business", "analysis", ["data analytics"]),
    "Recruitment": SkillDefinition("business", "hr", ["talent acquisition", "hiring"]),
    "Human Resources": SkillDefinition("business", "hr", ["hr", "hr management"]),
}

# --------------------------------------------------------------------- #
# Creative skills
# --------------------------------------------------------------------- #
_CREATIVE: Dict[str, SkillDefinition] = {
    "UI/UX Design": SkillDefinition(
        "creative", "design", ["ui ux design", "ux design", "ui design", "ui/ux"]
    ),
    "Graphic Design": SkillDefinition("creative", "design", []),
    "Figma": SkillDefinition("creative", "tool", []),
    "Adobe Photoshop": SkillDefinition("creative", "tool", ["photoshop"]),
    "Adobe Illustrator": SkillDefinition("creative", "tool", ["illustrator"]),
    "Adobe XD": SkillDefinition("creative", "tool", ["xd"]),
    "Sketch": SkillDefinition("creative", "tool", []),
    "Video Editing": SkillDefinition("creative", "media", ["video production"]),
    "Adobe Premiere Pro": SkillDefinition("creative", "tool", ["premiere pro", "premiere"]),
    "After Effects": SkillDefinition("creative", "tool", ["adobe after effects"]),
    "Copywriting": SkillDefinition("creative", "writing", ["copy writing"]),
    "Content Writing": SkillDefinition("creative", "writing", ["content creation"]),
    "Storyboarding": SkillDefinition("creative", "media", []),
    "Photography": SkillDefinition("creative", "media", []),
    "Illustration": SkillDefinition("creative", "media", []),
    "3D Modeling": SkillDefinition("creative", "media", ["3d design"]),
    "Motion Graphics": SkillDefinition("creative", "media", []),
    "Brand Identity Design": SkillDefinition("creative", "design", ["branding", "brand design"]),
}

SKILL_DICTIONARY: Dict[str, SkillDefinition] = {**_TECHNICAL, **_BUSINESS, **_CREATIVE}

# --------------------------------------------------------------------- #
# Skill stacks: a single mentioned token expands into multiple canonical
# skills. Matched *in addition to* (not instead of) direct mentions.
# --------------------------------------------------------------------- #
SKILL_STACKS: Dict[str, List[str]] = {
    "MERN": ["MongoDB", "Express.js", "React", "Node.js"],
    "MEAN": ["MongoDB", "Express.js", "Angular", "Node.js"],
    "MEVN": ["MongoDB", "Express.js", "Vue.js", "Node.js"],
    "LAMP": ["Linux", "Apache", "MySQL", "PHP"],
    "JAMstack": ["JavaScript", "REST APIs"],
}

_STACK_ALIASES = {
    "mern": "MERN",
    "mern stack": "MERN",
    "mean": "MEAN",
    "mean stack": "MEAN",
    "mevn": "MEVN",
    "mevn stack": "MEVN",
    "lamp": "LAMP",
    "lamp stack": "LAMP",
    "jamstack": "JAMstack",
    "jam stack": "JAMstack",
}


def build_alias_lookup() -> Dict[str, str]:
    """Return a lowercased alias/name -> canonical-skill-name lookup table."""
    lookup: Dict[str, str] = {}
    for canonical, definition in SKILL_DICTIONARY.items():
        lookup[canonical.lower()] = canonical
        for alias in definition.aliases:
            lookup[alias.lower()] = canonical
    return lookup


def build_stack_lookup() -> Dict[str, str]:
    """Return a lowercased alias -> canonical-stack-name lookup table."""
    lookup: Dict[str, str] = {name.lower(): name for name in SKILL_STACKS}
    lookup.update(_STACK_ALIASES)
    return lookup
