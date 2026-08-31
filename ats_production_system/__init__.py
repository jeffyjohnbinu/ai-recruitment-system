"""
ATS Production System
Day 20 deliverable — Zecpath AI Job Portal

The real, verified end-to-end orchestrator for the ATS AI microservice:
wires Days 5-18 together with corrected import paths and field mappings
(see component_adapters.py), exposes them over a FastAPI service
(api.py), and a CLI (cli.py) for batch/demo runs.
"""

from .pipeline import ATSPipeline

__all__ = ["ATSPipeline"]
__version__ = "1.0.0"
