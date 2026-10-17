#!/usr/bin/env python3
"""
Script 04: Quantize Babel Space

Performs quantization of Babel space embeddings with different bit configurations.
This script:
- Tests quantization at different bit depths (4, 8, 12, 16 bits)
- Evaluates reconstruction quality
- Compares compression ratios
- Saves quantized embeddings

Usage:
    python scripts/babel/04_quantize_babel.py [--texts pouvoir voir savoir] [--bits 8 12 16]
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

from apourvoir.babel import BabelSpace, BabelConfig, BabelMetrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def quantize_at_bits(babel: BabelSpace, texts: List[str], n_bits: int, lang: str = "fr") -> Dict[str, Any]:
    """Quantize embeddings at a specific bit depth."""
    logger.info(f"\n--- Quantizing at {n_bits} bits ---")
    
    results = {
        "n_bits": n_bits,
        "texts": {},
        "statistics": {}
    }
    
    metrics = BabelMetrics()
    
    # Temporarily update config
    original_bits = babel.config.n_bits
    babel.config.n_bits = n_bits
    babel.canonical_projector.config.n_bits = n_bits
    
    for text in texts:
        # Encode
        original = babel.encode(text, lang)
        
        # Quantize
        (theta_q, phi_q), canonical = babel.quantize(original)
        
        # Dequantize
        reconstructed = babel.dequantize(theta_q, phi_q)
        
        # Calculate errors
        cosine_error = metrics.reconstruction_error(original, reconstructed, "cosine")
        euclidean_error = metrics.reconstruction_error(original, reconstructed, "euclidean")
        
        results["texts"][text] = {
            "original": original.tolist(),
            "theta_q": theta_q,
            "phi_q": phi_q,
            "canonical": canonical.tolist(),
            "reconstructed": reconstructed.tolist(),
            "cosine_error": cosine_error,
            "euclidean_error": euclidean_error
        }
        
        logger.info(f"  {text}: cosine_error={cosine_error:.6f}, euclidean_error={euclidean_error:.6f}")
    
    # Calculate statistics
    originals = [torch.tensor(results["texts"][t]["original"]) for t in texts]
    reconstructions = [torch.tensor(results["texts"][t]["reconstructed"]) for t in texts]
    batch_metrics = metrics.batch_metrics(originals, reconstructions)
    results["statistics"] = batch_metrics
    
    # Restore original bits
    babel.config.n_bits = original_bits
    babel.canonical_projector.config.n_bits = original_bits
    
    return results


def compare_bit_depths(babel: BabelSpace, texts: List[str], bit_depths: List[int], lang: str = "fr") -> Dict[str, Any]:
    """Compare quantization at different bit depths."""
    logger.info("\n--- Comparing Bit Depths ---")
    
    results = {
        "bit_depths": bit_depths,
        "texts": texts,
        "comparisons": {}
    }
    
    # Quantize at each bit depth
    bit_results = {}
    for n_bits in bit_depths:
        bit_results[n_bits] = quantize_at_bits(babel, texts, n_bits, lang)
    
    # Compare results
    for n_bits in bit_depths:
        results["comparisons"][f"{n_bits}_bits"] = bit_results[n_bits]["statistics"]
    
    # Calculate compression ratios
    original_size = len(texts) * 128  # Assuming 128-dim embeddings at 32 bits
    for n_bits in bit_depths:
        # Quantized size: 2 angles * n_bits per angle
        compressed_size = len(texts) * 2 * n_bits
        ratio = original_size / compressed_size
        results["comparisons"][f"{n_bits}_bits"]["compression_ratio"] = ratio
    
    return results


def test_compression_ratios(texts: List[str], bit_depths: List[int]) -> None:
    """Test theoretical compression ratios."""
    logger.info("\n--- Theoretical Compression Ratios ---")
    
    metrics = BabelMetrics()
    
    for n_bits in bit_depths:
        # Original: 128-dim float32 = 128 * 32 bits
        original_bits = 128 * 32
        
        # Compressed: 2 angles * n_bits
        compressed_bits = 2 * n_bits
        
        ratio = original_bits / compressed_bits
        
        logger.info(f"  {n_bits} bits: ratio={ratio:.2f}x")
        
        # Per element
        original_per_element = 128 * 32
        compressed_per_element = 2 * n_bits
        bits_per_element = compressed_per_element
        
        logger.info(f"    Original: {original_per_element} bits/element")
        logger.info(f"    Compressed: {compressed_per_element} bits/element")
        logger.info(f"    Compression: {ratio:.2f}x")


def save_quantization_results(results: Dict[str, Any], output_file: str) -> str:
    """Save quantization results to file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved quantization results to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Quantize Babel space")
    parser.add_argument("--texts", nargs="+", 
                        default=["pouvoir", "voir", "savoir", "lumiere", "connaissance"],
                        help="Texts to quantize")
    parser.add_argument("--lang", default="fr", help="Language")
    parser.add_argument("--bits", nargs="+", type=int, default=[4, 8, 12, 16],
                        help="Bit depths to test")
    parser.add_argument("--compare", action="store_true", help="Compare bit depths")
    parser.add_argument("--theoretical", action="store_true", help="Show theoretical compression")
    parser.add_argument("--save", action="store_true", help="Save results")
    parser.add_argument("--output", default="results/babel_quantization.json", help="Output file")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Babel Space Quantization")
    logger.info("=" * 60)
    
    # Import torch
    import torch
    
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
    
    # Test theoretical compression
    if args.theoretical:
        test_compression_ratios(args.texts, args.bits)
    
    # Compare bit depths
    if args.compare:
        results = compare_bit_depths(babel, args.texts, args.bits, args.lang)
        
        # Print comparison
        logger.info("\n--- Comparison Results ---")
        for n_bits in args.bits:
            stats = results["comparisons"][f"{n_bits}_bits"]
            logger.info(f"{n_bits} bits:")
            logger.info(f"  Avg cosine error: {stats['cosine_error']['mean']:.6f}")
            logger.info(f"  Compression ratio: {stats.get('compression_ratio', 'N/A')}")
        
        if args.save:
            save_quantization_results(results, args.output)
    else:
        # Just quantize at default bits
        for n_bits in args.bits:
            quantize_at_bits(babel, args.texts, n_bits, args.lang)
    
    logger.info("Quantization complete")


if __name__ == "__main__":
    main()
