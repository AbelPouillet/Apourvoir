"""
Syntax Tree Builder for Babel Space

Constructs hierarchical syntax trees from text using:
- Tokenization (word, morpheme, character levels)
- Part-of-speech tagging
- Dependency parsing
- Constituency parsing

The tree structure captures:
- Word boundaries and morphology
- Grammatical relationships
- Sentence structure
- Hierarchical composition
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class SyntaxNode:
    """
    A node in the syntax tree.
    
    Attributes:
        token: The text token (word, morpheme, or character)
        pos: Part-of-speech tag (if available)
        lemma: Base form of the token
        depth: Depth in the tree
        parent: Parent node (None for root)
        children: Child nodes
        is_leaf: Whether this is a leaf node
        features: Additional linguistic features
        embedding: Optional pre-computed embedding for this node
    """
    token: str
    pos: Optional[str] = None
    lemma: Optional[str] = None
    depth: int = 0
    parent: Optional['SyntaxNode'] = None
    children: List['SyntaxNode'] = field(default_factory=list)
    is_leaf: bool = True
    features: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        # Update is_leaf based on children
        self.is_leaf = len(self.children) == 0
        
    def add_child(self, child: 'SyntaxNode') -> None:
        """Add a child node."""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)
        self.is_leaf = False
        
    def get_path(self) -> List[str]:
        """Get the path from root to this node."""
        path = []
        node = self
        while node is not None:
            path.insert(0, node.token)
            node = node.parent
        return path
    
    def get_subtree_tokens(self) -> List[str]:
        """Get all tokens in the subtree (pre-order traversal)."""
        tokens = [self.token]
        for child in self.children:
            tokens.extend(child.get_subtree_tokens())
        return tokens
    
    def __repr__(self) -> str:
        return f"SyntaxNode(token='{self.token}', pos={self.pos}, depth={self.depth}, children={len(self.children)})"


@dataclass
class SyntaxTree:
    """
    Complete syntax tree for a sentence or text.
    
    Attributes:
        root: Root node of the tree
        tokens: List of all tokens (leaves)
        num_nodes: Total number of nodes
        depth: Maximum depth of the tree
        metadata: Additional metadata about the tree
    """
    root: SyntaxNode
    tokens: List[str] = field(default_factory=list)
    num_nodes: int = 0
    depth: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Calculate metadata if not provided
        if self.num_nodes == 0:
            self.num_nodes = self._count_nodes(self.root)
        if self.depth == 0:
            self.depth = self._calculate_depth(self.root)
        if not self.tokens:
            self.tokens = self._get_all_tokens(self.root)
    
    def _count_nodes(self, node: SyntaxNode) -> int:
        """Count all nodes in the tree."""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count
    
    def _calculate_depth(self, node: SyntaxNode) -> int:
        """Calculate maximum depth of the tree."""
        if not node.children:
            return node.depth
        return max(self._calculate_depth(child) for child in node.children)
    
    def _get_all_tokens(self, node: SyntaxNode) -> List[str]:
        """Get all leaf tokens in the tree."""
        if node.is_leaf:
            return [node.token]
        tokens = []
        for child in node.children:
            tokens.extend(self._get_all_tokens(child))
        return tokens
    
    def get_node_by_path(self, path: List[str]) -> Optional[SyntaxNode]:
        """Get a node by its path from the root."""
        current = self.root
        for token in path:
            found = None
            for child in current.children:
                if child.token == token:
                    found = child
                    break
            if found is None:
                return None
            current = found
        return current
    
    def __repr__(self) -> str:
        return f"SyntaxTree(root='{self.root.token}', nodes={self.num_nodes}, depth={self.depth}, tokens={len(self.tokens)})"


class SyntaxTreeBuilder:
    """
    Builds syntax trees from text.
    
    Supports multiple levels of granularity:
    - Character level: Individual characters
    - Morpheme level: Word parts (prefixes, suffixes, roots)
    - Word level: Whole words
    - Phrase level: Phrases and clauses
    
    Example:
        >>> builder = SyntaxTreeBuilder(granularity="word")
        >>> tree = builder.build_tree("pouvoir voir savoir")
        >>> print(tree)
    """
    
    # Common morpheme patterns (simplified)
    MORPHEME_PATTERNS = {
        'fr': [
            (r'^(re|de|un|une|le|la|les|des|du|au|aux|en|ne|se|ce|je|tu|il|elle|nous|vous|ils|elles)$', 'prefix'),
            (r'^(pou|sav|vou|com|re|de|trans|in|im|il|ir|a|ab|ad|an|ante|anti|arch|auto)$', 'prefix'),
            (r'(tion|ment|age|eur|able|ible|ance|ence|esse|isme|iste|ite|if|ive|euse|eur|ard)$', 'suffix'),
            (r'(er|re|ir|oir|re|ir)$', 'verb_suffix'),
        ],
        'en': [
            (r'^(un|re|in|im|il|ir|dis|mis|pre|pro|sub|trans|over|under|out|up|down)$', 'prefix'),
            (r'(tion|ment|ness|able|ible|ance|ence|hood|dom|ship|ism|ist|ity|er|or|ly)$', 'suffix'),
            (r'(ing|ed|es|s|en)$', 'verb_suffix'),
        ],
        'es': [
            (r'^(re|des|in|im|sub|trans|pre|pro|a|an|ante|anti|sobre|bajo)$', 'prefix'),
            (r'(cion|miento|dad|tad|ble|able|ible|anza|encia|ismo|ista|idad|ar|er|ir)$', 'suffix'),
        ],
        'de': [
            (r'^(un|ver|zer|er|ur|miss|über|unter|wieder|vor|nach|zu|ab|auf|aus|bei|durch|für|gegen|ohne|um|mit)$', 'prefix'),
            (r'(ung|keit|heit|schaft|lichkeit|bar|lich|isch|end|ig|en|er|eln|ern|in|ung)$', 'suffix'),
        ],
    }
    
    # Part-of-speech tags (simplified)
    POS_TAGS = {
        'fr': {
            'noun': ['pouvoir', 'savoir', 'lumière', 'vision', 'connaissance', 'espace', 'token', 'mot', 'définition'],
            'verb': ['voir', 'pouvoir', 'savoir', 'faire', 'être', 'avoir', 'devoir', 'comprendre', 'désigner'],
            'adj': ['possible', 'canonique', 'géométrique', 'sémantique', 'mathématique', 'concentrique'],
            'adv': ['bien', 'mieux', 'très', 'trop', 'peu', 'beaucoup'],
            'prep': ['de', 'à', 'dans', 'par', 'pour', 'sur', 'en', 'avec', 'sans'],
            'conj': ['et', 'ou', 'mais', 'car', 'donc', 'or', 'ni'],
            'pron': ['je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles', 'on', 'ce', 'ça'],
        },
        'en': {
            'noun': ['power', 'knowledge', 'light', 'vision', 'space', 'token', 'word', 'definition'],
            'verb': ['see', 'can', 'know', 'do', 'be', 'have', 'must', 'understand', 'designate'],
            'adj': ['possible', 'canonical', 'geometric', 'semantic', 'mathematical', 'concentric'],
            'adv': ['well', 'better', 'very', 'too', 'little', 'much'],
            'prep': ['of', 'to', 'in', 'by', 'for', 'on', 'with', 'without'],
            'conj': ['and', 'or', 'but', 'because', 'so', 'yet', 'nor'],
            'pron': ['I', 'you', 'he', 'she', 'we', 'they', 'it', 'this', 'that'],
        },
    }
    
    def __init__(
        self,
        lang: str = "fr",
        granularity: str = "word",
        use_pos: bool = True,
    ):
        """
        Initialize the syntax tree builder.
        
        Args:
            lang: Language for morpheme patterns and POS tagging
            granularity: Level of granularity ('char', 'morpheme', 'word', 'phrase')
            use_pos: Whether to use part-of-speech tagging
        """
        self.lang = lang
        self.granularity = granularity
        self.use_pos = use_pos
        
        # Load patterns
        self.morpheme_patterns = self.MORPHEME_PATTERNS.get(lang, [])
        self.pos_tags = self.POS_TAGS.get(lang, {})
        
        logger.info(f"SyntaxTreeBuilder initialized: lang={lang}, granularity={granularity}")
    
    def build_tree(self, text: str, root_label: str = "ROOT") -> SyntaxTree:
        """
        Build a syntax tree from text.
        
        Args:
            text: Input text to parse
            root_label: Label for the root node
        
        Returns:
            SyntaxTree: The constructed syntax tree
        """
        if self.granularity == "char":
            return self._build_character_tree(text, root_label)
        elif self.granularity == "morpheme":
            return self._build_morpheme_tree(text, root_label)
        elif self.granularity == "word":
            return self._build_word_tree(text, root_label)
        elif self.granularity == "phrase":
            return self._build_phrase_tree(text, root_label)
        else:
            raise ValueError(f"Unknown granularity: {self.granularity}")
    
    def _build_character_tree(self, text: str, root_label: str) -> SyntaxTree:
        """Build tree at character level."""
        root = SyntaxNode(token=root_label)
        
        # Split into words first
        words = text.split()
        for word in words:
            word_node = SyntaxNode(token=word, pos="word")
            root.add_child(word_node)
            
            # Split word into characters
            for char in word:
                char_node = SyntaxNode(
                    token=char,
                    pos="char",
                    lemma=char,
                    features={"type": "character"}
                )
                word_node.add_child(char_node)
        
        return SyntaxTree(root=root)
    
    def _build_morpheme_tree(self, text: str, root_label: str) -> SyntaxTree:
        """Build tree at morpheme level."""
        root = SyntaxNode(token=root_label)
        
        words = text.split()
        for word in words:
            word_node = SyntaxNode(token=word, pos="word")
            root.add_child(word_node)
            
            # Split word into morphemes
            morphemes = self._split_morphemes(word)
            for morpheme in morphemes:
                morpheme_node = SyntaxNode(
                    token=morpheme,
                    pos="morpheme",
                    lemma=morpheme,
                    features={"type": "morpheme", "original": word}
                )
                word_node.add_child(morpheme_node)
        
        return SyntaxTree(root=root)
    
    def _split_morphemes(self, word: str) -> List[str]:
        """Split a word into morphemes using patterns."""
        # For now, use simple prefix/suffix matching
        # In production, use a proper morphological analyzer
        morphemes = []
        remaining = word
        
        # Try to match prefixes
        for pattern, mtype in self.morpheme_patterns:
            match = re.match(pattern, remaining)
            if match:
                prefix = match.group()
                morphemes.append(prefix)
                remaining = remaining[len(prefix):]
                break
        else:
            # No prefix matched, add the whole word
            if remaining:
                morphemes.append(remaining)
                remaining = ""
        
        # Try to match suffixes on remaining
        if remaining:
            for pattern, mtype in self.morpheme_patterns:
                match = re.search(pattern + r'$', remaining)
                if match:
                    suffix = match.group()
                    stem = remaining[:match.start()]
                    if stem:
                        morphemes.append(stem)
                    morphemes.append(suffix)
                    remaining = ""
                    break
            else:
                morphemes.append(remaining)
        
        return morphemes if morphemes else [word]
    
    def _build_word_tree(self, text: str, root_label: str) -> SyntaxTree:
        """Build tree at word level with POS tagging."""
        root = SyntaxNode(token=root_label)
        
        words = text.split()
        for i, word in enumerate(words):
            # Get POS tag
            pos = self._get_pos_tag(word.lower())
            
            word_node = SyntaxNode(
                token=word,
                pos=pos,
                lemma=word.lower(),
                features={"index": i, "type": "word"}
            )
            root.add_child(word_node)
        
        return SyntaxTree(root=root)
    
    def _build_phrase_tree(self, text: str, root_label: str) -> SyntaxTree:
        """Build tree at phrase level."""
        root = SyntaxNode(token=root_label)
        
        # Simple phrase chunking (for now)
        # In production, use a proper parser
        sentences = re.split(r'[.!?]', text)
        
        for sent_idx, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            
            sent_node = SyntaxNode(
                token=f"SENT_{sent_idx}",
                pos="sentence",
                features={"type": "sentence"}
            )
            root.add_child(sent_node)
            
            # Split into phrases (simple comma-based for now)
            phrases = [p.strip() for p in sentence.split(',') if p.strip()]
            
            for phrase_idx, phrase in enumerate(phrases):
                phrase_node = SyntaxNode(
                    token=f"PHRASE_{sent_idx}_{phrase_idx}",
                    pos="phrase",
                    features={"type": "phrase"}
                )
                sent_node.add_child(phrase_node)
                
                # Add words to phrase
                words = phrase.split()
                for word_idx, word in enumerate(words):
                    pos = self._get_pos_tag(word.lower())
                    word_node = SyntaxNode(
                        token=word,
                        pos=pos,
                        lemma=word.lower(),
                        features={"index": word_idx, "type": "word"}
                    )
                    phrase_node.add_child(word_node)
        
        return SyntaxTree(root=root)
    
    def _get_pos_tag(self, word: str) -> Optional[str]:
        """Get part-of-speech tag for a word."""
        if not self.use_pos:
            return None
        
        for pos, words in self.pos_tags.items():
            if word in words:
                return pos
        return None
    
    def build_from_tokens(self, tokens: List[str], root_label: str = "ROOT") -> SyntaxTree:
        """
        Build a simple word-level tree from a list of tokens.
        
        Args:
            tokens: List of tokens
            root_label: Label for the root node
        
        Returns:
            SyntaxTree: The constructed syntax tree
        """
        root = SyntaxNode(token=root_label)
        
        for i, token in enumerate(tokens):
            pos = self._get_pos_tag(token.lower())
            node = SyntaxNode(
                token=token,
                pos=pos,
                lemma=token.lower(),
                features={"index": i, "type": "token"}
            )
            root.add_child(node)
        
        return SyntaxTree(root=root)
