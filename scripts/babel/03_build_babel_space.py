#!/usr/bin/env python3
"""
Script 03: Build Babel Space

Constructs and tests the complete Babel space with P+S+M fusion.
This script:
- Initializes the Babel space
- Tests encoding at different dimensions
- Validates canonical relationships
- Tests quantization and dequantization
- Saves results

Usage:
    python scripts/babel/03_build_babel_space.py [--text "pouvoir voir savoir"] [--test-thales]
"""

from typing import List, Dict, Any, Optional, Tuple

import argparse
import torch
import json
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from apourvoir.babel import (
    BabelSpace, BabelConfig, BabelAlgebra, BabelMetrics,
    PhoneticEncoder, SyntaxTreeBuilder, SyntaxEncoder, CanonicalProjector
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_babel_space(texts: List[str], lang: str = "fr") -> BabelSpace:
    """Test the Babel space with a set of texts."""
    logger.info(f"Testing Babel space with {len(texts)} texts...")
    
    # Initialize Babel space
    config = BabelConfig(
        phonetic_dim=64,
        syntax_dim=64,
        semantic_dim=3,
        output_dim=128,
        p_weight=1.0,
        s_weight=1.0,
        m_weight=1.5,
        n_bits=8,
        default_lang=lang
    )
    
    babel = BabelSpace(config)
    
    # Initialize references
    babel.initialize_references(lang)
    
    logger.info(f"Babel space initialized: {config}")
    
    # Test encoding
    for text in texts:
        logger.info(f"\n--- Encoding '{text}' ---")
        
        # Test individual dimensions
        P = babel._encode_phonetic(text, lang)
        S = babel._encode_syntax(text, lang)
        M = babel._encode_semantic(text, lang)
        
        logger.info(f"  P (phonetic): shape={P.shape}, norm={torch.norm(P).item():.4f}")
        logger.info(f"  S (syntax): shape={S.shape}, norm={torch.norm(S).item():.4f}")
        logger.info(f"  M (semantic): shape={M.shape}, norm={torch.norm(M).item():.4f}")
        
        # Test combined encoding
        combined = babel.encode(text, lang)
        logger.info(f"  Combined: shape={combined.shape}, norm={torch.norm(combined).item():.4f}")
        
        # Test quantization
        (theta_q, phi_q), canonical = babel.quantize(combined)
        logger.info(f"  Quantized: theta_q={theta_q}, phi_q={phi_q}")
        
        # Test dequantization
        reconstructed = babel.dequantize(theta_q, phi_q)
        logger.info(f"  Reconstructed: {reconstructed}")
        
        # Test layer
        layer_idx, layer_name = babel.get_layer(text, lang)
        logger.info(f"  Layer: {layer_idx} ({layer_name})")
    
    return babel


def test_thales_relationships(babel: BabelSpace, lang: str = "fr") -> None:
    """Test Thales' theorem relationships."""
    logger.info("\n--- Testing Thales' Theorem Relationships ---")
    
    # Test the canonical triangle: pouvoir, lumiere, savoir
    triangles = [
        ("pouvoir", "lumiere", "savoir"),
        ("pou", "voir", "savoir"),
        ("pouvoir", "voir", "lumiere"),
    ]
    
    for v1, v2, v3 in triangles:
        result = babel.get_relationship(v1, v2, v3, lang)
        logger.info(f"\nTriangle: {v1} - {v2} - {v3}")
        logger.info(f"  Thales valid: {result['thales_valid']}")
        logger.info(f"  Angles: v1-v2={result['angle_v1v2_deg']:.2f}, v2-v3={result['angle_v2v3_deg']:.2f}, v1-v3={result['angle_v1v3_deg']:.2f}")


def test_quantization_roundtrip(babel: BabelSpace, texts: List[str], lang: str = "fr") -> None:
    """Test quantization and dequantization roundtrip."""
    logger.info("\n--- Testing Quantization Roundtrip ---")
    
    metrics = BabelMetrics()
    
    for text in texts:
        # Encode
        original = babel.encode(text, lang)
        
        # Quantize
        (theta_q, phi_q), canonical = babel.quantize(original)
        
        # Dequantize
        reconstructed = babel.dequantize(theta_q, phi_q)
        
        # Calculate error
        error = metrics.reconstruction_error(canonical, reconstructed, "cosine")
        
        logger.info(f"\n'{text}':")
        logger.info(f"  Original: {original}")
        logger.info(f"  Canonical: {canonical}")
        logger.info(f"  Reconstructed: {reconstructed}")
        logger.info(f"  Cosine error: {error:.6f}")


def test_algebra(babel: BabelSpace, lang: str = "fr") -> None:
    """Test algebraic operations."""
    logger.info("\n--- Testing Algebraic Operations ---")
    
    algebra = BabelAlgebra()
    
    # Get embeddings for operations
    texts = ["pouvoir", "voir", "savoir"]
    embeddings = [babel.encode(text, lang) for text in texts]
    
    # Test composition
    composed = algebra.compose(embeddings[0], embeddings[1], "geometric_mean")
    logger.info(f"Composed (pouvoir + voir): {composed}")
    
    # Test inversion
    inverted = algebra.invert(embeddings[0], "antipodal")
    logger.info(f"Inverted (pouvoir): {inverted}")
    
    # Test fusion
    fused = algebra.fuse(embeddings, method="weighted_sum")
    logger.info(f"Fused (all): {fused}")
    
    # Test Thales validation
    is_thales = algebra.validate_thales(embeddings[0], embeddings[1], embeddings[2])
    logger.info(f"Thales valid (pouvoir, voir, savoir): {is_thales}")


def save_babel_results(babel: BabelSpace, texts: List[str], lang: str, output_file: str) -> str:
    """Save Babel space results to file."""
    results = {
        "config": {
            "phonetic_dim": babel.config.phonetic_dim,
            "syntax_dim": babel.config.syntax_dim,
            "semantic_dim": babel.config.semantic_dim,
            "p_weight": babel.config.p_weight,
            "s_weight": babel.config.s_weight,
            "m_weight": babel.config.m_weight,
        },
        "texts": {},
        "statistics": {}
    }
    
    # Encode all texts
    for text in texts:
        combined = babel.encode(text, lang)
        (theta_q, phi_q), canonical = babel.quantize(combined)
        reconstructed = babel.dequantize(theta_q, phi_q)
        layer_idx, layer_name = babel.get_layer(text, lang)
        
        results["texts"][text] = {
            "combined": combined.tolist(),
            "theta_q": theta_q,
            "phi_q": phi_q,
            "canonical": canonical.tolist(),
            "reconstructed": reconstructed.tolist(),
            "layer": {"index": layer_idx, "name": layer_name}
        }
    
    # Calculate statistics
    metrics = BabelMetrics()
    originals = [torch.tensor(results["texts"][t]["combined"]) for t in texts]
    reconstructions = [torch.tensor(results["texts"][t]["reconstructed"]) for t in texts]
    batch_metrics = metrics.batch_metrics(originals, reconstructions)
    results["statistics"] = batch_metrics
    
    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved results to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Build and test Babel space")
    parser.add_argument("--texts", nargs="+", 
                        default=["pouvoir", "voir", "savoir", "lumiere", "connaissance"],
                        help="Texts to process")
    parser.add_argument("--lang", default="fr", help="Language")
    parser.add_argument("--test-thales", action="store_true", help="Test Thales relationships")
    parser.add_argument("--test-quantization", action="store_true", help="Test quantization roundtrip")
    parser.add_argument("--test-algebra", action="store_true", help="Test algebraic operations")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    parser.add_argument("--output", default="results/babel_space.json", help="Output file")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Babel Space Builder")
    logger.info("=" * 60)
    
    # Import torch
    import torch
    
    # Initialize and test Babel space
    babel = test_babel_space(args.texts, args.lang)
    
    # Test Thales relationships
    if args.test_thales:
        test_thales_relationships(babel, args.lang)
    
    # Test quantization
    if args.test_quantization:
        test_quantization_roundtrip(babel, args.texts, args.lang)
    
    # Test algebra
    if args.test_algebra:
        test_algebra(babel, args.lang)
    
    # Save results
    if args.save:
        save_babel_results(babel, args.texts, args.lang, args.output)
    
    logger.info("Babel space building complete")


if __name__ == "__main__":
    main()
