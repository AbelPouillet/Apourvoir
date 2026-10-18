#!/usr/bin/env python3
"""
Script 05: Validate Babel Space

Validates the Babel space implementation with comprehensive tests.
This script:
- Tests canonical relationships (Thales' theorem)
- Validates layer organization
- Tests multilingual alignment
- Evaluates reconstruction quality
- Generates validation report

Usage:
    python scripts/babel/05_validate_babel.py [--texts pouvoir voir savoir] [--lang fr en]
"""

from typing import List, Dict, Any, Optional, Tuple

import argparse
import torch
import numpy as np
import json
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from apourvoir.babel import BabelSpace, BabelConfig, BabelMetrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_thales_triangles(babel: BabelSpace, triangles: List[Tuple[str, str, str]], lang: str = "fr") -> Dict[str, Any]:
    """Validate Thales' theorem for a set of triangles."""
    logger.info("\n--- Validating Thales' Triangles ---")
    
    results = {
        "triangles": [],
        "valid_count": 0,
        "total_count": len(triangles),
        "validation_rate": 0.0
    }
    
    metrics = BabelMetrics()
    
    for v1, v2, v3 in triangles:
        result = babel.get_relationship(v1, v2, v3, lang)
        
        # Additional validation with metrics
        v1_emb = babel._encode_semantic(v1, lang)
        v2_emb = babel._encode_semantic(v2, lang)
        v3_emb = babel._encode_semantic(v3, lang)
        thales_validation = metrics.validate_thales(v1_emb, v2_emb, v3_emb)
        
        triangle_result = {
            "tokens": [v1, v2, v3],
            "thales_valid": result["thales_valid"],
            "angles_deg": {
                "v1_v2": result["angle_v1v2_deg"],
                "v2_v3": result["angle_v2v3_deg"],
                "v1_v3": result["angle_v1v3_deg"]
            },
            "thales_check": thales_validation
        }
        
        results["triangles"].append(triangle_result)
        
        if result["thales_valid"]:
            results["valid_count"] += 1
        
        logger.info(f"  {v1} - {v2} - {v3}: Thales valid={result['thales_valid']}, angle_v2={result['angle_v2v3_deg']:.2f}")
    
    results["validation_rate"] = results["valid_count"] / results["total_count"] if results["total_count"] > 0 else 0.0
    
    return results


def validate_layer_organization(babel: BabelSpace, texts: List[str], lang: str = "fr") -> Dict[str, Any]:
    """Validate layer organization."""
    logger.info("\n--- Validating Layer Organization ---")
    
    results = {
        "texts": [],
        "layer_distribution": {},
        "expected_layers": {
            "fundamentals": ["pouvoir", "voir", "savoir"],
            "sub_concepts": ["lumiere", "connaissance", "espace"],
            "actions": ["comprendre", "apprendre", "decouvrir"],
            "techniques": ["quantisation", "compression", "projection"]
        }
    }
    
    # Classify all texts
    for text in texts:
        layer_idx, layer_name = babel.get_layer(text, lang)
        
        text_result = {
            "text": text,
            "layer_index": layer_idx,
            "layer_name": layer_name
        }
        results["texts"].append(text_result)
        
        # Update distribution
        if layer_name not in results["layer_distribution"]:
            results["layer_distribution"][layer_name] = 0
        results["layer_distribution"][layer_name] += 1
        
        logger.info(f"  {text}: layer {layer_idx} ({layer_name})")
    
    # Check expected layers
    for layer_name, expected_texts in results["expected_layers"].items():
        actual_texts = [t["text"] for t in results["texts"] if t["layer_name"] == layer_name]
        match_count = len(set(expected_texts) & set(actual_texts))
        results["expected_layers"][layer_name] = {
            "expected": expected_texts,
            "actual": actual_texts,
            "match_count": match_count,
            "match_rate": match_count / len(expected_texts) if expected_texts else 1.0
        }
    
    return results


def validate_multilingual(babel: BabelSpace, multilingual_texts: Dict[str, List[str]]) -> Dict[str, Any]:
    """Validate multilingual alignment."""
    logger.info("\n--- Validating Multilingual Alignment ---")
    
    results = {
        "languages": list(multilingual_texts.keys()),
        "texts": {},
        "alignment": {}
    }
    
    metrics = BabelMetrics()
    
    # Encode all texts
    all_embeddings = {}
    for lang, texts in multilingual_texts.items():
        all_embeddings[lang] = {}
        for text in texts:
            embedding = babel.encode(text, lang)
            all_embeddings[lang][text] = embedding
            results["texts"][f"{lang}:{text}"] = embedding.tolist()
    
    # Check alignment for each concept across languages
    for concept_texts in zip(*[multilingual_texts[lang] for lang in multilingual_texts.keys()]):
        concept = concept_texts[0]  # Use first as concept name
        embeddings = [all_embeddings[lang][text] for lang, text in zip(multilingual_texts.keys(), concept_texts)]
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = metrics.semantic_similarity(embeddings[i], embeddings[j], "cosine")
                similarities.append(sim)
        
        avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
        
        results["alignment"][concept] = {
            "texts": [f"{lang}:{text}" for lang, text in zip(multilingual_texts.keys(), concept_texts)],
            "similarities": similarities,
            "average_similarity": avg_sim
        }
        
        logger.info(f"  {concept}: avg_similarity={avg_sim:.4f}")
    
    return results


def validate_reconstruction(babel: BabelSpace, texts: List[str], lang: str = "fr") -> Dict[str, Any]:
    """Validate reconstruction quality."""
    logger.info("\n--- Validating Reconstruction Quality ---")
    
    results = {
        "texts": [],
        "statistics": {}
    }
    
    metrics = BabelMetrics()
    
    originals = []
    reconstructions = []
    
    for text in texts:
        # Encode
        original = babel.encode(text, lang)
        
        # Quantize and dequantize
        (theta_q, phi_q), canonical = babel.quantize(original)
        reconstructed = babel.dequantize(theta_q, phi_q)
        
        # Calculate errors
        # Two angles only encode the direction of the 3D canonical point: the full
        # embedding cannot be reconstructed from them. The error below therefore
        # measures angular quantization of the canonical 3D point only.
        cosine_error = metrics.reconstruction_error(canonical, reconstructed, "cosine")
        euclidean_error = metrics.reconstruction_error(canonical, reconstructed, "euclidean")
        
        text_result = {
            "text": text,
            "cosine_error": cosine_error,
            "euclidean_error": euclidean_error,
            "original": original.tolist(),
            "reconstructed": reconstructed.tolist()
        }
        results["texts"].append(text_result)
        
        originals.append(canonical)
        reconstructions.append(reconstructed)
        
        logger.info(f"  {text}: cosine_error={cosine_error:.6f}, euclidean_error={euclidean_error:.6f}")
    
    # Calculate batch statistics
    batch_metrics = metrics.batch_metrics(originals, reconstructions)
    results["statistics"] = batch_metrics
    
    return results


def generate_validation_report(
    thales_results: Dict[str, Any],
    layer_results: Dict[str, Any],
    multilingual_results: Dict[str, Any],
    reconstruction_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate comprehensive validation report."""
    report = {
        "thales_validation": thales_results,
        "layer_validation": layer_results,
        "multilingual_validation": multilingual_results,
        "reconstruction_validation": reconstruction_results,
        "summary": {
            "thales_validation_rate": thales_results["validation_rate"],
            "layer_entropy": layer_results.get("layer_entropy", 0.0),
            "avg_multilingual_similarity": np.mean([
                v["average_similarity"] for v in multilingual_results.get("alignment", {}).values()
            ]) if multilingual_results.get("alignment") else 0.0,
            "avg_cosine_error": reconstruction_results["statistics"]["cosine_error"]["mean"]
        }
    }
    
    return report


def save_validation_report(report: Dict[str, Any], output_file: str) -> str:
    """Save validation report to file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved validation report to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Validate Babel space")
    parser.add_argument("--texts", nargs="+", 
                        default=["pouvoir", "voir", "savoir", "lumiere", "connaissance",
                                "espace", "canonique", "geometrique", "semantique", "token"],
                        help="Texts to validate")
    parser.add_argument("--lang", default="fr", help="Primary language")
    parser.add_argument("--multilingual", action="store_true", help="Test multilingual alignment")
    parser.add_argument("--save", action="store_true", help="Save validation report")
    parser.add_argument("--output", default="results/babel_validation.json", help="Output file")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Babel Space Validation")
    logger.info("=" * 60)
    
    # Import torch and numpy
    import torch
    import numpy as np
    
    # Initialize Babel space
    config = BabelConfig(
        phonetic_dim=64,
        syntax_dim=64,
        semantic_dim=3,
        output_dim=128,
        n_bits=8,
        default_lang=args.lang
    )
    babel = BabelSpace(config)
    babel.initialize_references(args.lang)
    
    # Define Thales triangles
    thales_triangles = [
        ("pouvoir", "lumiere", "savoir"),
        ("pou", "voir", "savoir"),
        ("pouvoir", "voir", "lumiere"),
        ("pouvoir", "savoir", "connaissance"),
        ("voir", "lumiere", "connaissance"),
    ]
    
    # Validate Thales
    thales_results = validate_thales_triangles(babel, thales_triangles, args.lang)
    
    # Validate layers
    layer_results = validate_layer_organization(babel, args.texts, args.lang)
    
    # Validate reconstruction
    reconstruction_results = validate_reconstruction(babel, args.texts, args.lang)
    
    # Validate multilingual
    multilingual_results = {"alignment": {}}
    if args.multilingual:
        multilingual_texts = {
            "fr": ["pouvoir", "voir", "savoir", "lumiere"],
            "en": ["power", "see", "know", "light"],
            "es": ["poder", "ver", "saber", "luz"],
        }
        multilingual_results = validate_multilingual(babel, multilingual_texts)
    
    # Generate report
    report = generate_validation_report(
        thales_results, layer_results, multilingual_results, reconstruction_results
    )
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Thales validation rate: {report['summary']['thales_validation_rate']:.2%}")
    logger.info(f"Average cosine error: {report['summary']['avg_cosine_error']:.6f}")
    if args.multilingual:
        logger.info(f"Average multilingual similarity: {report['summary']['avg_multilingual_similarity']:.4f}")
    
    # Save report
    if args.save:
        save_validation_report(report, args.output)
    
    logger.info("Validation complete")


if __name__ == "__main__":
    main()
