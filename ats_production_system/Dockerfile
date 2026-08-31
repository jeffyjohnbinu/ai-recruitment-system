# Day 20 deliverable — Zecpath AI Job Portal
# Containerizes the ATS AI microservice (ats_production_system.api:app).
#
# Build (from the repository root, so sibling Day 5-18 packages are included):
#   docker build -f ats_production_system/Dockerfile -t zecpath-ats-ai:1.0.0 .
#
# Run:
#   docker run -p 8000:8000 zecpath-ats-ai:1.0.0

FROM python:3.11-slim

WORKDIR /app

# System deps for pdfplumber/PyPDF2 font handling and healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install the full repo's consolidated requirements plus this service's own
COPY requirements.txt ./requirements.txt
COPY ats_production_system/requirements.txt ./ats_production_system/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r ats_production_system/requirements.txt

# Copy the whole repo (additive pipeline needs every Day 5-18 package as a sibling import)
COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "ats_production_system.api:app", "--host", "0.0.0.0", "--port", "8000"]
