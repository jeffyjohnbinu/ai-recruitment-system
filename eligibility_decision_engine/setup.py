from setuptools import find_packages, setup

setup(
    name="eligibility_decision_engine",
    version="1.0.0",
    description="Day 21 - Zecpath: candidate eligibility decision engine for AI screening calls",
    packages=find_packages(exclude=["tests", "tests.*", "examples"]),
    python_requires=">=3.9",
    extras_require={"test": ["pytest>=7.0"]},
)
