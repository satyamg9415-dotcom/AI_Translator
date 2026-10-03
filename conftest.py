"""
conftest.py — pytest configuration and shared fixtures.
"""
import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests that require model downloads or GPU (deselect with -m 'not slow')"
    )
