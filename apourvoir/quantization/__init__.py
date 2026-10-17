"""
Quantization Module for Apourvoir

Provides canonical quantization methods for MoE models.
"""

from .canonical import (
    normaliser_vecteur,
    coordonnées_sphériques,
    projeter_sur_tore,
    quantifier_angles,
    déquantifier_angles,
    quantifier_tensor_canonique,
    déquantifier_tensor_canonique,
)

__all__ = [
    "normaliser_vecteur",
    "coordonnées_sphériques",
    "projeter_sur_tore",
    "quantifier_angles",
    "déquantifier_angles",
    "quantifier_tensor_canonique",
    "déquantifier_tensor_canonique",
]
