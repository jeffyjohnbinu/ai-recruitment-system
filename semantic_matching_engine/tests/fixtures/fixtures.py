"""
Fixture resumes & job descriptions spanning multiple job types (engineering,
data science, product management), used both for unit tests and for the
matching-accuracy validation set (Day 12 deliverable: "Validate across
multiple job types").

Each entry has a `label` indicating the ground-truth match decision a human
recruiter would make, so the accuracy report can be computed against it.
"""

from __future__ import annotations

# ---------------------------------------------------------------- resumes --
RESUME_BACKEND_STRONG = {
    "skills": "Python, FastAPI, PostgreSQL, Docker, Kubernetes, AWS, REST APIs, CI/CD",
    "experience": (
        "Senior backend engineer with 6 years building scalable microservices. "
        "Led migration of a monolith to microservices, cutting deploy time by 40%. "
        "Designed REST APIs consumed by 2M+ daily active users. Mentored junior engineers."
    ),
    "projects": "Built an ETL pipeline processing 10TB/day using Python and Airflow.",
}

RESUME_BACKEND_WEAK = {
    "skills": "Adobe Photoshop, Illustrator, Figma, brand guidelines",
    "experience": (
        "Graphic designer for 4 years creating marketing collateral and " "social media assets."
    ),
    "projects": "Designed a full rebrand for a mid-size retail client.",
}

RESUME_DATA_SCIENTIST = {
    "skills": "Python, scikit-learn, pandas, SQL, AWS SageMaker, statistics",
    "experience": (
        "Data scientist with strong background in machine learning and statistical "
        "modeling for fintech products. Built fraud detection models reducing false "
        "positives by 30%. Deployed models on AWS SageMaker."
    ),
    "projects": "Automated a reporting pipeline saving 10 hours per week.",
}

RESUME_PRODUCT_MANAGER = {
    "skills": "Roadmapping, SQL, Excel, Amplitude, A/B testing, stakeholder management",
    "experience": (
        "Product manager who owned the roadmap for a payments platform, shipping 12 "
        "features across 3 quarters. Partnered closely with engineering and design."
    ),
    "projects": "Ran user research for an onboarding flow, increasing activation rate by 18%.",
}

# ------------------------------------------------------------ job records --
JOB_BACKEND_ENGINEER = {
    "role": "Backend Software Engineer",
    "skills": "Python, FastAPI, PostgreSQL, AWS, Docker, Kubernetes, microservices",
    "experience": (
        "Looking for an engineer who has led backend teams, built and scaled REST "
        "APIs, and worked with containerized microservices in production."
    ),
}

JOB_DATA_SCIENTIST = {
    "role": "Data Scientist, Fraud",
    "skills": "Python, machine learning, scikit-learn, SQL, AWS, statistical modeling",
    "experience": (
        "Seeking a data scientist to build and deploy fraud-detection models and "
        "improve model performance in production."
    ),
}

JOB_PRODUCT_MANAGER = {
    "role": "Senior Product Manager",
    "skills": "roadmapping, SQL, A/B testing, stakeholder management, analytics",
    "experience": (
        "Own the product roadmap for a payments platform, partner with engineering "
        "and design, and run user research to improve activation."
    ),
}

# --------------------------------------------------------- validation set --
# (resume, job, job_type, expected_is_match)
VALIDATION_PAIRS = [
    (RESUME_BACKEND_STRONG, JOB_BACKEND_ENGINEER, "engineering", True),
    (RESUME_BACKEND_WEAK, JOB_BACKEND_ENGINEER, "engineering", False),
    (RESUME_DATA_SCIENTIST, JOB_DATA_SCIENTIST, "data_science", True),
    (RESUME_BACKEND_STRONG, JOB_DATA_SCIENTIST, "data_science", False),
    (RESUME_PRODUCT_MANAGER, JOB_PRODUCT_MANAGER, "product", True),
    (RESUME_DATA_SCIENTIST, JOB_PRODUCT_MANAGER, "product", False),
]
