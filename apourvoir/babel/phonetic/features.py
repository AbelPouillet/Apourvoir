"""
Articulatory features for IPA segments

Chaque segment IPA est décrit par des traits articulatoires standards
(voisement, nasalité, lieu et mode d'articulation pour les consonnes ;
aperture, antériorité et arrondissement pour les voyelles). Ces vecteurs sont
déterministes et interprétables : deux sons proches articulatoirement ont des
vecteurs proches (ex. /p/ et /b/ ne diffèrent que par le voisement).

Disposition des 21 traits :
    0      syllabique (voyelle)
    1      voisé
    2      nasal
    3-11   lieu : bilabial, labiodental, dental, alvéolaire, postalvéolaire,
           palatal, vélaire, uvulaire, glottal
    12-17  mode : occlusive, fricative, affriquée, approximante,
           battue/roulée, latérale
    18     aperture (voyelles : 1 = fermée, 0 = ouverte)
    19     antériorité (voyelles : 0 = antérieure, 1 = postérieure)
    20     arrondie
"""

from typing import Dict, List

import torch

FEATURE_NAMES = [
    "syllabic", "voiced", "nasal",
    "bilabial", "labiodental", "dental", "alveolar", "postalveolar",
    "palatal", "velar", "uvular", "glottal",
    "stop", "fricative", "affricate", "approximant", "trill_tap", "lateral",
    "height", "backness", "rounded",
]
NUM_FEATURES = len(FEATURE_NAMES)
_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def _consonant(places: List[str], manner: List[str], voiced: bool, nasal: bool = False) -> torch.Tensor:
    v = torch.zeros(NUM_FEATURES)
    for p in places:
        v[_IDX[p]] = 1.0
    for m in manner:
        v[_IDX[m]] = 1.0
    v[_IDX["voiced"]] = float(voiced)
    v[_IDX["nasal"]] = float(nasal)
    return v


def _vowel(height: float, backness: float, rounded: bool) -> torch.Tensor:
    v = torch.zeros(NUM_FEATURES)
    v[_IDX["syllabic"]] = 1.0
    v[_IDX["voiced"]] = 1.0
    v[_IDX["height"]] = height
    v[_IDX["backness"]] = backness
    v[_IDX["rounded"]] = float(rounded)
    return v


SEGMENT_FEATURES: Dict[str, torch.Tensor] = {
    # Occlusives
    "p": _consonant(["bilabial"], ["stop"], False),
    "b": _consonant(["bilabial"], ["stop"], True),
    "t": _consonant(["alveolar"], ["stop"], False),
    "d": _consonant(["alveolar"], ["stop"], True),
    "k": _consonant(["velar"], ["stop"], False),
    "ɡ": _consonant(["velar"], ["stop"], True),
    "g": _consonant(["velar"], ["stop"], True),
    "ʔ": _consonant(["glottal"], ["stop"], False),
    # Nasales
    "m": _consonant(["bilabial"], ["stop"], True, nasal=True),
    "n": _consonant(["alveolar"], ["stop"], True, nasal=True),
    "ɲ": _consonant(["palatal"], ["stop"], True, nasal=True),
    "ŋ": _consonant(["velar"], ["stop"], True, nasal=True),
    # Fricatives
    "f": _consonant(["labiodental"], ["fricative"], False),
    "v": _consonant(["labiodental"], ["fricative"], True),
    "β": _consonant(["bilabial"], ["fricative"], True),
    "θ": _consonant(["dental"], ["fricative"], False),
    "ð": _consonant(["dental"], ["fricative"], True),
    "s": _consonant(["alveolar"], ["fricative"], False),
    "z": _consonant(["alveolar"], ["fricative"], True),
    "ʃ": _consonant(["postalveolar"], ["fricative"], False),
    "ʒ": _consonant(["postalveolar"], ["fricative"], True),
    "ç": _consonant(["palatal"], ["fricative"], False),
    "x": _consonant(["velar"], ["fricative"], False),
    "ʁ": _consonant(["uvular"], ["fricative"], True),
    "h": _consonant(["glottal"], ["fricative"], False),
    # Affriquées
    "tʃ": _consonant(["postalveolar"], ["stop", "affricate"], False),
    "dʒ": _consonant(["postalveolar"], ["stop", "affricate"], True),
    "ts": _consonant(["alveolar"], ["stop", "affricate"], False),
    "dz": _consonant(["alveolar"], ["stop", "affricate"], True),
    # Approximantes, battues, roulées, latérales
    "w": _consonant(["bilabial", "velar"], ["approximant"], True),
    "j": _consonant(["palatal"], ["approximant"], True),
    "ɥ": _consonant(["bilabial", "palatal"], ["approximant"], True),
    "ɹ": _consonant(["alveolar"], ["approximant"], True),
    "ʋ": _consonant(["labiodental"], ["approximant"], True),
    "ɾ": _consonant(["alveolar"], ["trill_tap"], True),
    "r": _consonant(["alveolar"], ["trill_tap"], True),
    "l": _consonant(["alveolar"], ["approximant", "lateral"], True),
    "ʎ": _consonant(["palatal"], ["approximant", "lateral"], True),
    # Voyelles (aperture, antériorité, arrondie)
    "i": _vowel(1.0, 0.0, False),
    "y": _vowel(1.0, 0.0, True),
    "ɪ": _vowel(0.83, 0.1, False),
    "ʏ": _vowel(0.83, 0.1, True),
    "ʊ": _vowel(0.83, 0.9, True),
    "u": _vowel(1.0, 1.0, True),
    "e": _vowel(0.67, 0.0, False),
    "ø": _vowel(0.67, 0.0, True),
    "o": _vowel(0.67, 1.0, True),
    "ə": _vowel(0.5, 0.5, False),
    "ɛ": _vowel(0.33, 0.0, False),
    "ɜ": _vowel(0.33, 0.5, False),
    "ʌ": _vowel(0.33, 1.0, False),
    "ɔ": _vowel(0.33, 1.0, True),
    "æ": _vowel(0.17, 0.0, False),
    "ɐ": _vowel(0.17, 0.5, False),
    "a": _vowel(0.0, 0.2, False),
    "ɑ": _vowel(0.0, 1.0, False),
    "ɒ": _vowel(0.0, 1.0, True),
}

# Diacritiques et marques ignorées (accent, longueur, non-syllabicité, abaissement)
_IGNORED = set("/ˈˌː̯̞.")
_NASAL_TILDE = "̃"
_MULTI = ("tʃ", "dʒ", "ts", "dz")


def segment_ipa(ipa: str) -> List[str]:
    """
    Découpe une transcription IPA (ex. "/vwaʁ/", "/tɑ̃/") en segments connus.

    Les voyelles nasales (tilde combinant) sont marquées par le suffixe "~".
    Les symboles inconnus sont ignorés.
    """
    segments: List[str] = []
    i = 0
    while i < len(ipa):
        if ipa[i] in _IGNORED:
            i += 1
            continue
        if ipa[i] == _NASAL_TILDE:
            if segments:
                segments[-1] = segments[-1] + "~"
            i += 1
            continue
        pair = ipa[i:i + 2]
        if pair in _MULTI:
            segments.append(pair)
            i += 2
            continue
        if ipa[i] in SEGMENT_FEATURES:
            segments.append(ipa[i])
        i += 1
    return segments


def segment_features(segment: str) -> torch.Tensor:
    """Vecteur de traits d'un segment (gère le suffixe nasal "~")."""
    nasal = segment.endswith("~")
    base = segment[:-1] if nasal else segment
    v = SEGMENT_FEATURES[base].clone()
    if nasal:
        v[_IDX["nasal"]] = 1.0
    return v
