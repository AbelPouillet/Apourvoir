"""
Quantization Module for Apourvoir

Quantisation canonique (sphérique par triplets) et baselines de référence.
"""

from .canonical import (
    normaliser_vecteur,
    coordonnées_sphériques,
    cartésiennes,
    projeter_sur_tore,
    quantifier_angles,
    déquantifier_angles,
    quantifier_tensor_canonique,
    déquantifier_tensor_canonique,
    taille_octets,
    erreur_relative,
    CanonicalQuantizer,
    ThalesTriangle,
)
from .baselines import BASELINES

__all__ = [
    "normaliser_vecteur",
    "coordonnées_sphériques",
    "cartésiennes",
    "projeter_sur_tore",
    "quantifier_angles",
    "déquantifier_angles",
    "quantifier_tensor_canonique",
    "déquantifier_tensor_canonique",
    "taille_octets",
    "erreur_relative",
    "CanonicalQuantizer",
    "ThalesTriangle",
    "BASELINES",
]
