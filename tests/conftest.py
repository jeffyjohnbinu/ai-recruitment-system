"""Shared pytest fixtures for the test suite."""

import pytest


@pytest.fixture
def sample_resume_text() -> str:
    return (
        "Backend engineer with experience in Python, FastAPI, PostgreSQL, "
        "and AWS. Comfortable writing tests with pytest and deploying with Docker."
    )


@pytest.fixture
def sample_job_description() -> str:
    return (
        "Looking for a Python engineer experienced with FastAPI, PostgreSQL, "
        "and AWS. Pytest and Docker experience preferred."
    )
