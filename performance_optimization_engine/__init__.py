"""
Performance Optimization & Tuning Engine
Day 18 deliverable — Zecpath AI Job Portal

Makes the ATS pipeline production-ready by optimizing extraction speed,
reducing model/embedding response time, improving memory handling,
refining entity detection, and improving noisy-resume handling.

Additive-only: this package never modifies Day 5-17 modules. It wraps
and post-processes their outputs via alias-tolerant loaders, exactly
like every prior day's integration pattern.
"""

from .optimizer import PerformanceOptimizationEngine

__all__ = ["PerformanceOptimizationEngine"]
__version__ = "1.0.0"
