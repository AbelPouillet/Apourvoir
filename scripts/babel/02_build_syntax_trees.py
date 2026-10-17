import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
#!/usr/bin/env python3
"""
Script 02: Build Syntax Trees

Constructs and validates syntax trees for text analysis.
This script:
- Builds syntax trees at different granularities
- Tests tree construction
- Validates tree properties
- Saves example trees

Usage:
    python scripts/babel/02_build_syntax_trees.py [--text "pouvoir voir savoir"] [--granularity word]
"""

import argparse
import json
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from apourvoir.babel.syntax import SyntaxTreeBuilder, SyntaxEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_syntax_tree(text: str, granularity: str = "word", lang: str = "fr") -> SyntaxTreeBuilder:
    """Build a syntax tree for the given text."""
    logger.info(f"Building syntax tree for '{text}' at {granularity} granularity...")
    
    builder = SyntaxTreeBuilder(
        lang=lang,
        granularity=granularity
    )
    
    tree = builder.build_tree(text)
    
    logger.info(f"Tree built: {tree}")
    logger.info(f"  Nodes: {tree.num_nodes}")
    logger.info(f"  Depth: {tree.depth}")
    logger.info(f"  Tokens: {tree.tokens}")
    
    return builder, tree


def test_all_granularities(text: str, lang: str = "fr") -> None:
    """Test tree building at all granularities."""
    granularities = ["char", "morpheme", "word", "phrase"]
    
    for granularity in granularities:
        logger.info(f"\n--- Testing {granularity} granularity ---")
        try:
            builder, tree = build_syntax_tree(text, granularity, lang)
            
            # Print tree structure
            print_tree(tree.root, indent=0)
        except Exception as e:
            logger.error(f"Error with {granularity}: {e}")


def print_tree(node: 'SyntaxNode', indent: int = 0) -> None:
    """Print tree structure."""
    prefix = "  " * indent
    info = f"{node.token}"
    if node.pos:
        info += f"[{node.pos}]"
    if node.is_leaf:
        info += " (leaf)"
    else:
        info += f" ({len(node.children)} children)"
    
    logger.info(f"{prefix}{info}")
    
    for child in node.children:
        print_tree(child, indent + 1)


def test_syntax_encoder(tree: 'SyntaxTree', method: str = "weighted_sum") -> torch.Tensor:
    """Test syntax encoding."""
    logger.info(f"Testing syntax encoder with {method} method...")
    
    encoder = SyntaxEncoder()
    embedding = encoder.encode_tree(tree, method)
    
    logger.info(f"Encoding result: shape={embedding.shape}, norm={torch.norm(embedding).item():.4f}")
    
    return embedding


def compare_granularities(text: str, lang: str = "fr") -> None:
    """Compare encodings at different granularities."""
    logger.info("\n--- Comparing granularities ---")
    
    granularities = ["char", "morpheme", "word", "phrase"]
    embeddings = {}
    
    for granularity in granularities:
        try:
            builder = SyntaxTreeBuilder(lang=lang, granularity=granularity)
            tree = builder.build_tree(text)
            encoder = SyntaxEncoder()
            embedding = encoder.encode_tree(tree)
            embeddings[granularity] = embedding
            logger.info(f"{granularity}: {embedding.shape}")
        except Exception as e:
            logger.error(f"Error with {granularity}: {e}")
    
    # Compare embeddings
    logger.info("\nEmbedding similarities:")
    for g1 in granularities:
        for g2 in granularities:
            if g1 < g2:  # Avoid duplicate comparisons
                e1 = embeddings.get(g1)
                e2 = embeddings.get(g2)
                if e1 is not None and e2 is not None:
                    sim = torch.dot(e1, e2).item()
                    logger.info(f"  {g1} vs {g2}: {sim:.4f}")


def save_tree(tree: 'SyntaxTree', filename: str) -> str:
    """Save tree to file."""
    def node_to_dict(node: 'SyntaxNode') -> dict:
        return {
            "token": node.token,
            "pos": node.pos,
            "depth": node.depth,
            "is_leaf": node.is_leaf,
            "children": [node_to_dict(child) for child in node.children]
        }
    
    tree_dict = {
        "root": node_to_dict(tree.root),
        "num_nodes": tree.num_nodes,
        "depth": tree.depth,
        "tokens": tree.tokens
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(tree_dict, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved tree to {filename}")
    return filename


def main():
    parser = argparse.ArgumentParser(description="Build and test syntax trees")
    parser.add_argument("--text", default="pouvoir voir savoir", help="Text to process")
    parser.add_argument("--lang", default="fr", help="Language")
    parser.add_argument("--granularity", default="word", 
                        choices=["char", "morpheme", "word", "phrase"],
                        help="Granularity level")
    parser.add_argument("--test-all", action="store_true", help="Test all granularities")
    parser.add_argument("--compare", action="store_true", help="Compare granularities")
    parser.add_argument("--save", action="store_true", help="Save tree to file")
    parser.add_argument("--output", default="results/syntax_tree.json", help="Output file")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Babel Syntax Tree Builder")
    logger.info("=" * 60)
    
    # Import torch for encoding
    import torch
    
    # Test single granularity
    if not (args.test_all or args.compare):
        builder, tree = build_syntax_tree(args.text, args.granularity, args.lang)
        
        # Test encoding
        test_syntax_encoder(tree)
        
        # Save if requested
        if args.save:
            save_tree(tree, args.output)
    
    # Test all granularities
    if args.test_all:
        test_all_granularities(args.text, args.lang)
    
    # Compare granularities
    if args.compare:
        compare_granularities(args.text, args.lang)
    
    logger.info("Syntax tree building complete")


if __name__ == "__main__":
    main()
