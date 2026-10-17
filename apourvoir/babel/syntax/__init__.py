"""
Syntax Module for Babel Space

Provides syntactic analysis and encoding for the S (syntax) dimension.

Components:
- SyntaxTreeBuilder: Constructs syntactic trees from text
- SyntaxEncoder: Converts syntax trees to vector embeddings
- SyntaxTree: Data structure for syntax tree nodes
"""

from .tree_builder import SyntaxTreeBuilder, SyntaxTree, SyntaxNode
from .syntax_encoder import SyntaxEncoder

__all__ = [
    "SyntaxTreeBuilder",
    "SyntaxTree",
    "SyntaxNode",
    "SyntaxEncoder",
]
