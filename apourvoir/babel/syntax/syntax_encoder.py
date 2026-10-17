"""
Syntax Encoder for Babel Space

Converts syntax trees to vector embeddings for the S (syntax) dimension.

Encoding strategy:
- Each node gets an embedding based on its properties (token, POS, depth, position)
- Tree structure is encoded via recursive composition
- Final embedding is a weighted sum of all node embeddings

Features:
- Positional encoding for word order
- Depth encoding for hierarchy
- POS tag encoding for grammatical role
- Normalized output vectors
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

from .tree_builder import SyntaxTree, SyntaxNode

logger = logging.getLogger(__name__)


@dataclass
class SyntaxConfig:
    """Configuration for syntax encoding."""
    embedding_dim: int = 64
    use_pos: bool = True
    use_depth: bool = True
    use_position: bool = True
    use_features: bool = True
    max_depth: int = 10
    max_position: int = 100
    
    # Weighting parameters
    pos_weight: float = 1.5
    depth_weight: float = 1.2
    position_weight: float = 1.0
    token_weight: float = 1.0


class SyntaxEncoder:
    """
    Encodes syntax trees into vector embeddings.
    
    The encoder uses:
    - Token embeddings (from a vocabulary)
    - POS tag embeddings
    - Depth embeddings (for hierarchy)
    - Position embeddings (for order)
    
    Example:
        >>> builder = SyntaxTreeBuilder(granularity="word")
        >>> tree = builder.build_tree("pouvoir voir savoir")
        >>> encoder = SyntaxEncoder()
        >>> embedding = encoder.encode_tree(tree)
        >>> print(embedding.shape)  # torch.Size([64])
    """
    
    def __init__(self, config: Optional[SyntaxConfig] = None):
        """
        Initialize the syntax encoder.
        
        Args:
            config: Configuration for encoding
        """
        self.config = config or SyntaxConfig()
        
        # Initialize embeddings
        self.token_embeddings = {}
        self.pos_embeddings = {}
        self.depth_embeddings = None
        self.position_embeddings = None
        
        # Initialize vocabulary
        self._init_vocabulary()
        self._init_pos_embeddings()
        self._init_depth_embeddings()
        self._init_position_embeddings()
        
        logger.info(f"SyntaxEncoder initialized: dim={self.config.embedding_dim}")
    
    def _init_vocabulary(self) -> None:
        """Initialize token vocabulary with random embeddings."""
        # Common tokens for our domain
        domain_tokens = [
            "pouvoir", "voir", "savoir", "lumière", "connaissance", 
            "espace", "canonique", "géométrique", "sémantique", "token",
            "mot", "définition", "processus", "étape", "concept", "notion",
            "referentiel", "coordonnée", "angle", "vecteur", "dimension",
            "compression", "quantisation", "modèle", "moe", "mistral",
        ]
        
        for token in domain_tokens:
            self.token_embeddings[token] = torch.randn(self.config.embedding_dim)
            self.token_embeddings[token] = self.token_embeddings[token] / torch.norm(self.token_embeddings[token])
    
    def _init_pos_embeddings(self) -> None:
        """Initialize POS tag embeddings."""
        pos_tags = ['noun', 'verb', 'adj', 'adv', 'prep', 'conj', 'pron', 'word', 'char', 'morpheme', 'phrase', 'sentence']
        
        for pos in pos_tags:
            self.pos_embeddings[pos] = torch.randn(self.config.embedding_dim)
            self.pos_embeddings[pos] = self.pos_embeddings[pos] / torch.norm(self.pos_embeddings[pos])
    
    def _init_depth_embeddings(self) -> None:
        """Initialize depth embeddings."""
        self.depth_embeddings = torch.zeros(self.config.max_depth, self.config.embedding_dim)
        for d in range(self.config.max_depth):
            # Use sinusoidal encoding for depth
            for i in range(0, self.config.embedding_dim, 2):
                if i < self.config.embedding_dim:
                    self.depth_embeddings[d, i] = np.sin(d / (10000 ** (i / self.config.embedding_dim)))
                if i + 1 < self.config.embedding_dim:
                    self.depth_embeddings[d, i + 1] = np.cos(d / (10000 ** (i / self.config.embedding_dim)))
        
        # Normalize
        for d in range(self.config.max_depth):
            self.depth_embeddings[d] = self.depth_embeddings[d] / torch.norm(self.depth_embeddings[d])
    
    def _init_position_embeddings(self) -> None:
        """Initialize position embeddings."""
        self.position_embeddings = torch.zeros(self.config.max_position, self.config.embedding_dim)
        for p in range(self.config.max_position):
            # Use sinusoidal encoding for position
            for i in range(0, self.config.embedding_dim, 2):
                if i < self.config.embedding_dim:
                    self.position_embeddings[p, i] = np.sin(p / (10000 ** (i / self.config.embedding_dim)))
                if i + 1 < self.config.embedding_dim:
                    self.position_embeddings[p, i + 1] = np.cos(p / (10000 ** (i / self.config.embedding_dim)))
        
        # Normalize
        for p in range(self.config.max_position):
            self.position_embeddings[p] = self.position_embeddings[p] / torch.norm(self.position_embeddings[p])
    
    def _get_token_embedding(self, token: str) -> torch.Tensor:
        """Get embedding for a token."""
        token_lower = token.lower()
        if token_lower in self.token_embeddings:
            return self.token_embeddings[token_lower]
        
        # For unknown tokens, use hash-based embedding
        hash_val = hash(token_lower) % 10000
        unknown_emb = torch.randn(self.config.embedding_dim)
        unknown_emb = unknown_emb / torch.norm(unknown_emb)
        self.token_embeddings[token_lower] = unknown_emb
        return unknown_emb
    
    def _get_pos_embedding(self, pos: Optional[str]) -> torch.Tensor:
        """Get embedding for a POS tag."""
        if pos is None or pos not in self.pos_embeddings:
            return torch.zeros(self.config.embedding_dim)
        return self.pos_embeddings[pos]
    
    def _get_depth_embedding(self, depth: int) -> torch.Tensor:
        """Get embedding for a depth."""
        if depth >= self.config.max_depth:
            depth = self.config.max_depth - 1
        return self.depth_embeddings[depth]
    
    def _get_position_embedding(self, position: int) -> torch.Tensor:
        """Get embedding for a position."""
        if position >= self.config.max_position:
            position = self.config.max_position - 1
        return self.position_embeddings[position]
    
    def encode_node(self, node: SyntaxNode) -> torch.Tensor:
        """
        Encode a single syntax node into a vector.
        
        Args:
            node: The syntax node to encode
        
        Returns:
            torch.Tensor: The node embedding
        """
        # Token embedding
        token_emb = self._get_token_embedding(node.token) * self.config.token_weight
        
        # POS embedding
        pos_emb = self._get_pos_embedding(node.pos) * self.config.pos_weight
        
        # Depth embedding
        depth_emb = self._get_depth_embedding(node.depth) * self.config.depth_weight
        
        # Position embedding (if available in features)
        position = node.features.get("index", 0)
        position_emb = self._get_position_embedding(position) * self.config.position_weight
        
        # Combine all embeddings
        combined = token_emb + pos_emb + depth_emb + position_emb
        
        # Normalize
        if torch.norm(combined) > 0:
            combined = combined / torch.norm(combined)
        
        return combined
    
    def encode_tree(self, tree: SyntaxTree, method: str = "weighted_sum") -> torch.Tensor:
        """
        Encode a complete syntax tree into a vector.
        
        Args:
            tree: The syntax tree to encode
            method: Encoding method ('weighted_sum', 'root', 'average')
        
        Returns:
            torch.Tensor: The tree embedding
        """
        if method == "root":
            # Just encode the root node
            return self.encode_node(tree.root)
        
        elif method == "average":
            # Average of all node embeddings
            embeddings = []
            self._collect_node_embeddings(tree.root, embeddings)
            if not embeddings:
                return torch.zeros(self.config.embedding_dim)
            
            stacked = torch.stack(embeddings)
            avg = torch.mean(stacked, dim=0)
            if torch.norm(avg) > 0:
                avg = avg / torch.norm(avg)
            return avg
        
        else:  # weighted_sum (default)
            # Weighted sum based on depth (deeper nodes have less weight)
            embeddings = []
            weights = []
            self._collect_weighted_embeddings(tree.root, embeddings, weights, weight=1.0)
            
            if not embeddings:
                return torch.zeros(self.config.embedding_dim)
            
            stacked = torch.stack(embeddings)
            weight_tensor = torch.tensor(weights, dtype=torch.float32)
            
            # Weighted sum
            weighted_sum = torch.sum(stacked * weight_tensor.unsqueeze(1), dim=0)
            
            # Normalize
            if torch.norm(weighted_sum) > 0:
                weighted_sum = weighted_sum / torch.norm(weighted_sum)
            
            return weighted_sum
    
    def _collect_node_embeddings(self, node: SyntaxNode, embeddings: List[torch.Tensor]) -> None:
        """Recursively collect all node embeddings."""
        embeddings.append(self.encode_node(node))
        for child in node.children:
            self._collect_node_embeddings(child, embeddings)
    
    def _collect_weighted_embeddings(
        self,
        node: SyntaxNode,
        embeddings: List[torch.Tensor],
        weights: List[float],
        weight: float = 1.0,
        depth_factor: float = 0.8
    ) -> None:
        """
        Recursively collect weighted node embeddings.
        
        Args:
            node: Current node
            embeddings: List to store embeddings
            weights: List to store corresponding weights
            weight: Current weight (attenuated by depth)
            depth_factor: Factor to reduce weight with depth
        """
        embeddings.append(self.encode_node(node))
        weights.append(weight)
        
        # Reduce weight for children
        child_weight = weight * depth_factor
        for child in node.children:
            self._collect_weighted_embeddings(child, embeddings, weights, child_weight, depth_factor)
    
    def encode_batch(self, trees: List[SyntaxTree], method: str = "weighted_sum") -> torch.Tensor:
        """
        Encode a batch of syntax trees.
        
        Args:
            trees: List of syntax trees
            method: Encoding method
        
        Returns:
            torch.Tensor: Batch of embeddings (shape: [num_trees, embedding_dim])
        """
        embeddings = [self.encode_tree(tree, method) for tree in trees]
        return torch.stack(embeddings)
    
    def get_tree_structure_vector(self, tree: SyntaxTree) -> torch.Tensor:
        """
        Get a vector representing the tree structure (without token content).
        
        This captures the shape of the tree independent of the specific tokens.
        
        Args:
            tree: The syntax tree
        
        Returns:
            torch.Tensor: Structure vector
        """
        # Collect structural features
        features = {
            "num_nodes": tree.num_nodes,
            "depth": tree.depth,
            "num_leaves": len(tree.tokens),
            "avg_children": self._get_avg_children(tree.root),
            "max_branching": self._get_max_branching(tree.root),
        }
        
        # Normalize features
        normalized = {}
        for key, value in features.items():
            if key == "num_nodes":
                normalized[key] = value / 100.0
            elif key == "depth":
                normalized[key] = value / self.config.max_depth
            elif key == "num_leaves":
                normalized[key] = value / 100.0
            else:
                normalized[key] = value / 10.0
        
        # Create vector from normalized features
        vector = torch.tensor([
            normalized["num_nodes"],
            normalized["depth"],
            normalized["num_leaves"],
            normalized["avg_children"],
            normalized["max_branching"],
        ])
        
        # Pad or truncate to embedding dimension
        if len(vector) < self.config.embedding_dim:
            padding = torch.zeros(self.config.embedding_dim - len(vector))
            vector = torch.cat([vector, padding])
        else:
            vector = vector[:self.config.embedding_dim]
        
        # Normalize
        if torch.norm(vector) > 0:
            vector = vector / torch.norm(vector)
        
        return vector
    
    def _get_avg_children(self, node: SyntaxNode) -> float:
        """Get average number of children per node."""
        if node.is_leaf:
            return 0.0
        
        total_children = len(node.children)
        total_nodes = 1  # Count this node
        
        for child in node.children:
            total_children += self._get_avg_children(child) * self._count_nodes(child)
            total_nodes += self._count_nodes(child)
        
        return total_children / total_nodes if total_nodes > 0 else 0.0
    
    def _get_max_branching(self, node: SyntaxNode) -> int:
        """Get maximum branching factor in the tree."""
        max_branching = len(node.children)
        for child in node.children:
            child_branching = self._get_max_branching(child)
            max_branching = max(max_branching, child_branching)
        return max_branching
    
    def _count_nodes(self, node: SyntaxNode) -> int:
        """Count nodes in subtree."""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count
