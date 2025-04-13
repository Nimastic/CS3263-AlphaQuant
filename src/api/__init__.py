"""
AlphaQuant API Module.

This module implements the API for the AlphaQuant system, providing
endpoints for data access, model predictions, and portfolio recommendations.
"""

from .main import app
from .routes import router

__all__ = [
    'app',
    'router',
] 