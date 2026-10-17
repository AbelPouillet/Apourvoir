#!/usr/bin/env python3
"""
Script 06: Benchmark Babel Space

Comprehensive benchmarking of Babel space performance.
This script:
- Measures encoding/decoding speed
- Evaluates memory usage
- Tests scalability with batch sizes
- Compares with baseline methods
- Generates performance report

Usage:
    python scripts/babel/06_benchmark_babel.py [--texts pouvoir voir savoir] [--batch-sizes 1 10 100]
"""

from typing import List, Dict, Any, Optional, Tuple

import argparse
import torch
import numpy as np
import json
import logging
import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from apourvoir.babel import BabelSpace, BabelConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def benchmark_encoding_speed(babel: BabelSpace, texts: List[str], batch_sizes: List[int], lang: str = "fr") -> Dict[str, Any]:
    """Benchmark encoding speed."""
    logger.info("\n--- Benchmarking Encoding Speed ---")
    
    results = {
        "batch_sizes": [],
        "times": {},
        "speeds": {}
    }
    
    for batch_size in batch_sizes:
        # Prepare batch
        batch = texts * (batch_size // len(texts) + 1)
        batch = batch[:batch_size]
        
        # Warmup
        for _ in range(3):
            _ = babel.encode_batch(batch, lang)
        
        # Measure
        start_time = time.time()
        for _ in range(10):
            embeddings = babel.encode_batch(batch, lang)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        time_per_item = avg_time / batch_size
        items_per_second = batch_size / avg_time
        
        results["batch_sizes"].append(batch_size)
        results["times"][batch_size] = avg_time
        results["speeds"][batch_size] = {
            "time_per_item_ms": time_per_item * 1000,
            "items_per_second": items_per_second
        }
        
        logger.info(f"  Batch size {batch_size}: {avg_time*1000:.2f}ms, {time_per_item*1000:.4f}ms/item, {items_per_second:.2f} items/s")
    
    return results


def benchmark_memory_usage(babel: BabelSpace, texts: List[str], lang: str = "fr") -> Dict[str, Any]:
    """Benchmark memory usage."""
    logger.info("\n--- Benchmarking Memory Usage ---")
    
    try:
        import psutil
        import gc
        
        process = psutil.Process(os.getpid())
        
        results = {
            "initial_memory_mb": process.memory_info().rss / 1024 / 1024,
            "after_encoding_mb": 0,
            "memory_per_item_kb": 0
        }
        
        # Measure initial memory
        gc.collect()
        initial_memory = process.memory_info().rss
        
        # Encode all texts
        embeddings = babel.encode_batch(texts, lang)
        
        # Measure after encoding
        gc.collect()
        after_memory = process.memory_info().rss
        
        memory_used = after_memory - initial_memory
        memory_per_item = memory_used / len(texts)
        
        results["after_encoding_mb"] = after_memory / 1024 / 1024
        results["memory_per_item_kb"] = memory_per_item / 1024
        
        logger.info(f"  Initial memory: {results['initial_memory_mb']:.2f} MB")
        logger.info(f"  After encoding: {results['after_encoding_mb']:.2f} MB")
        logger.info(f"  Memory per item: {results['memory_per_item_kb']:.2f} KB")
        
    except ImportError:
        logger.warning("psutil not available, skipping memory benchmark")
        results = {"error": "psutil not available"}
    
    return results


def benchmark_quantization_speed(babel: BabelSpace, texts: List[str], batch_sizes: List[int], lang: str = "fr") -> Dict[str, Any]:
    """Benchmark quantization speed."""
    logger.info("\n--- Benchmarking Quantization Speed ---")
    
    results = {
        "batch_sizes": [],
        "quantization_times": {},
        "dequantization_times": {}
    }
    
    for batch_size in batch_sizes:
        # Prepare batch
        batch = texts * (batch_size // len(texts) + 1)
        batch = batch[:batch_size]
        
        # Encode batch
        embeddings = babel.encode_batch(batch, lang)
        
        # Warmup
        for _ in range(3):
            for emb in embeddings:
                _ = babel.quantize(emb)
                
        # Measure quantization
        start_time = time.time()
        for _ in range(10):
            for emb in embeddings:
                (theta_q, phi_q), canonical = babel.quantize(emb)
        end_time = time.time()
        
        avg_quant_time = (end_time - start_time) / 10 / batch_size
        
        # Measure dequantization
        start_time = time.time()
        for _ in range(10):
            for emb in embeddings:
                (theta_q, phi_q), _ = babel.quantize(emb)
                _ = babel.dequantize(theta_q, phi_q)
        end_time = time.time()
        
        avg_dequant_time = (end_time - start_time) / 10 / batch_size
        
        results["batch_sizes"].append(batch_size)
        results["quantization_times"][batch_size] = avg_quant_time * 1000  # ms
        results["dequantization_times"][batch_size] = avg_dequant_time * 1000  # ms
        
        logger.info(f"  Batch size {batch_size}: quant={avg_quant_time*1000:.4f}ms, dequant={avg_dequant_time*1000:.4f}ms")
    
    return results


def benchmark_compression_ratio(texts: List[str], bit_depths: List[int]) -> Dict[str, Any]:
    """Benchmark compression ratios."""
    logger.info("\n--- Benchmarking Compression Ratios ---")
    
    results = {
        "texts": len(texts),
        "bit_depths": bit_depths,
        "ratios": {}
    }
    
    # Original size calculation
    # Assuming 128-dim float32 embeddings
    original_bits_per_embedding = 128 * 32
    total_original_bits = original_bits_per_embedding * len(texts)
    
    for n_bits in bit_depths:
        # Quantized size: 2 angles * n_bits per angle per embedding
        compressed_bits_per_embedding = 2 * n_bits
        total_compressed_bits = compressed_bits_per_embedding * len(texts)
        
        ratio = total_original_bits / total_compressed_bits
        
        results["ratios"][n_bits] = {
            "ratio": ratio,
            "original_bits": total_original_bits,
            "compressed_bits": total_compressed_bits,
            "bits_per_embedding": compressed_bits_per_embedding
        }
        
        logger.info(f"  {n_bits} bits: ratio={ratio:.2f}x, {compressed_bits_per_embedding} bits/embedding")
    
    return results


def generate_benchmark_report(
    encoding_results: Dict[str, Any],
    memory_results: Dict[str, Any],
    quantization_results: Dict[str, Any],
    compression_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate comprehensive benchmark report."""
    report = {
        "encoding_benchmark": encoding_results,
        "memory_benchmark": memory_results,
        "quantization_benchmark": quantization_results,
        "compression_benchmark": compression_results,
        "summary": {
            "max_items_per_second": max(
                v["items_per_second"] for v in encoding_results["speeds"].values()
            ) if encoding_results.get("speeds") else 0,
            "min_time_per_item_ms": min(
                v["time_per_item_ms"] for v in encoding_results["speeds"].values()
            ) if encoding_results.get("speeds") else float('inf'),
            "memory_per_item_kb": memory_results.get("memory_per_item_kb", 0),
            "compression_ratios": compression_results["ratios"]
        }
    }
    
    return report


def save_benchmark_report(report: Dict[str, Any], output_file: str) -> str:
    """Save benchmark report to file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved benchmark report to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Benchmark Babel space")
    parser.add_argument("--texts", nargs="+", 
                        default=["pouvoir", "voir", "savoir", "lumiere", "connaissance"],
                        help="Texts to benchmark")
    parser.add_argument("--lang", default="fr", help="Language")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 10, 100],
                        help="Batch sizes to test")
    parser.add_argument("--bits", nargs="+", type=int, default=[4, 8, 12, 16],
                        help="Bit depths to test")
    parser.add_argument("--save", action="store_true", help="Save benchmark report")
    parser.add_argument("--output", default="results/babel_benchmark.json", help="Output file")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Babel Space Benchmark")
    logger.info("=" * 60)
    
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
    
    # Benchmark encoding
    encoding_results = benchmark_encoding_speed(babel, args.texts, args.batch_sizes, args.lang)
    
    # Benchmark memory
    memory_results = benchmark_memory_usage(babel, args.texts, args.lang)
    
    # Benchmark quantization
    quantization_results = benchmark_quantization_speed(babel, args.texts, args.batch_sizes, args.lang)
    
    # Benchmark compression
    compression_results = benchmark_compression_ratio(args.texts, args.bits)
    
    # Generate report
    report = generate_benchmark_report(
        encoding_results, memory_results, quantization_results, compression_results
    )
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Max encoding speed: {report['summary']['max_items_per_second']:.2f} items/s")
    logger.info(f"Min time per item: {report['summary']['min_time_per_item_ms']:.4f} ms")
    if report['summary']['memory_per_item_kb']:
        logger.info(f"Memory per item: {report['summary']['memory_per_item_kb']:.2f} KB")
    logger.info("Compression ratios:")
    for n_bits, data in report['summary']['compression_ratios'].items():
        logger.info(f"  {n_bits} bits: {data['ratio']:.2f}x")
    
    # Save report
    if args.save:
        save_benchmark_report(report, args.output)
    
    logger.info("Benchmark complete")


if __name__ == "__main__":
    main()
