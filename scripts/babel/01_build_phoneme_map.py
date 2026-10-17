#!/usr/bin/env python3
"""
Script 01: Build Phoneme Map

Generates and validates phoneme mappings for multiple languages.
This script:
- Validates existing phoneme maps
- Generates missing phoneme entries
- Tests phonetic encoding
- Saves updated phoneme maps

Usage:
    python scripts/babel/01_build_phoneme_map.py [--lang fr] [--validate]
"""

import argparse
import json
import logging
import sys
import os

from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from apourvoir.babel.phonetic import PhoneticEncoder, PHONEME_MAP, PHONEME_WEIGHTS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_phoneme_map(lang: str = "fr") -> bool:
    """Validate the phoneme map for a language."""
    logger.info(f"Validating phoneme map for {lang}...")
    
    phoneme_map = PHONEME_MAP.get(lang, {})
    
    if not phoneme_map:
        logger.warning(f"No phoneme map found for {lang}")
        return False
    
    # Check required tokens
    required_tokens = ["pouvoir", "voir", "savoir", "lumiere", "connaissance"]
    missing = []
    
    for token in required_tokens:
        if token not in phoneme_map:
            missing.append(token)
    
    if missing:
        logger.warning(f"Missing tokens in {lang}: {missing}")
        return False
    
    # Check phoneme format
    for token, phonemes in phoneme_map.items():
        if not isinstance(phonemes, list):
            logger.warning(f"Invalid phoneme format for {token}: {phonemes}")
            return False
        
        for phoneme in phonemes:
            if not phoneme.startswith("/") or not phoneme.endswith("/"):
                logger.warning(f"Invalid phoneme format: {phoneme}")
                return False
    
    logger.info(f"Phoneme map for {lang} is valid")
    return True


def generate_missing_phonemes(lang: str = "fr") -> Dict[str, list]:
    """Generate missing phoneme entries."""
    logger.info(f"Generating missing phonemes for {lang}...")
    
    # Get existing map
    existing = PHONEME_MAP.get(lang, {}).copy()
    
    # Common French words for Apourvoir
    new_entries = {
        "pouvoir": ["/pu/", "/vwaR/"],
        "voir": ["/vwaR/"],
        "savoir": ["/sa/", "/vwaR/"],
        "lumiere": ["/ly/", "/mjeR/"],
        "connaissance": ["/ko/", "/nes/", "/s/"],
        "espace": ["/es/", "/pas/"],
        "canonique": ["/ka/", "/no/", "/nik/"],
        "geometrique": ["/Zo/", "/me/", "/trik/"],
        "semantique": ["/se/", "/mAn/", "/tik/"],
        "token": ["/to/", "/kEn/"],
        "mot": ["/mo/"],
        "definition": ["/de/", "/fi/", "/ni/", "/sjO/"],
        "processus": ["/pRo/", "/se/", "/sys/"],
        "etape": ["/e/", "/tap/"],
        "concept": ["/kOn/", "/sEp/"],
        "notion": ["/no/", "/sjO/"],
        "referentiel": ["/Re/", "/fe/", "/REn/", "/sjEl/"],
        "coordonnee": ["/ko/", "/Or/", "/do/", "/ne/"],
        "angle": ["/An/", "/gl/"],
        "vecteur": ["/vEk/", "/tOE/", "/R/"],
        "dimension": ["/di/", "/mAn/", "/sjO/"],
        "compression": ["/kOm/", "/pRe/", "/sjO/"],
        "quantisation": ["/kAn/", "/ti/", "/za/", "/sjO/"],
        "modele": ["/mo/", "/dEl/"],
        "moe": ["/moe/"],
        "mistral": ["/mis/", "/tRal/"],
    }
    
    # Merge with existing
    for token, phonemes in new_entries.items():
        if token not in existing:
            existing[token] = phonemes
            logger.info(f"Added phoneme entry for '{token}': {phonemes}")
    
    return existing


def test_phonetic_encoder(lang: str = "fr") -> bool:
    """Test the phonetic encoder."""
    logger.info(f"Testing phonetic encoder for {lang}...")
    
    try:
        encoder = PhoneticEncoder(lang=lang)
        
        # Test encoding
        test_words = ["pouvoir", "voir", "savoir"]
        for word in test_words:
            embedding = encoder.get_phonetic_embedding(word)
            logger.info(f"Encoded '{word}': shape={embedding.shape}, norm={torch.norm(embedding).item():.4f}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing phonetic encoder: {e}")
        return False


def save_phoneme_map(phoneme_map: Dict[str, list], lang: str, output_dir: str = "results") -> str:
    """Save phoneme map to file."""
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.join(output_dir, f"phoneme_map_{lang}.json")
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(phoneme_map, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved phoneme map to {filename}")
    return filename


def main():
    parser = argparse.ArgumentParser(description="Build and validate phoneme map")
    parser.add_argument("--lang", default="fr", help="Language to process")
    parser.add_argument("--validate", action="store_true", help="Validate existing map")
    parser.add_argument("--generate", action="store_true", help="Generate missing entries")
    parser.add_argument("--test", action="store_true", help="Test phonetic encoder")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Babel Phoneme Map Builder")
    logger.info("=" * 60)
    
    # Validate
    if args.validate:
        valid = validate_phoneme_map(args.lang)
        if not valid:
            logger.warning("Phoneme map validation failed")
    
    # Generate
    if args.generate:
        phoneme_map = generate_missing_phonemes(args.lang)
        save_phoneme_map(phoneme_map, args.lang, args.output_dir)
    
    # Test
    if args.test:
        test_phonetic_encoder(args.lang)
    
    # Default: do all
    if not (args.validate or args.generate or args.test):
        validate_phoneme_map(args.lang)
        phoneme_map = generate_missing_phonemes(args.lang)
        save_phoneme_map(phoneme_map, args.lang, args.output_dir)
        test_phonetic_encoder(args.lang)
    
    logger.info("Phoneme map building complete")


if __name__ == "__main__":
    # Import torch for the encoder
    import torch
    main()
