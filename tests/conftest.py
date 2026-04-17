"""Shared pytest configuration and fixtures."""

import warnings

import pytest


def pytest_configure(config):
    warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")
    warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
