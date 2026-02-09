"""
Compatibility wrapper.

Primary implementation lives in models/qwennew.py.
"""

from models.myqwen import (  # noqa: F401
    QwenConfig,
    QwenModel,
    TraditionalEmbedding,
    SinusoidalPositionalEncoding,
    SinusoidalEmbedding,
    RotaryEmbedding,
)

