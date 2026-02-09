"""
Model Proxy package for SWE Agent.

This package provides the ModelProxy server for intercepting SWE-Agent's
OpenAI API calls and forwarding them to VERL for processing.
"""

from .proxy_server import ModelProxy, ModelRequest

__all__ = ["ModelProxy", "ModelRequest"]
