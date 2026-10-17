"""
Apourvoir - Canonical Quantization SDK for Mistral AI MoE Models

This package provides:
- Canonical quantization methods (torus-based angular quantization)
- Hardware-aware optimization SDK
- Adaptive batching and memory management
- Expert parallelism for MoE models
- Dynamic quantization based on available resources

Author: Abel Pouillet (Albo-co)
Project: Apourvoir - A pour designer l'angle et la mesure de l'omnipotence
"""

from . import quantization
from . import optimizer

__version__ = "1.0.0"
__all__ = ["quantization", "optimizer"]
