"""
Phonetic Encoder for Babel Space

Converts words to phonetic embeddings by:
1. Transcribing words to IPA phonemes
2. Mapping phonemes to pre-trained embeddings
3. Weighting by articulatory complexity
4. Producing normalized phonetic vectors
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

from .phoneme_map import PHONEME_MAP, PHONEME_WEIGHTS, PHONEME_EMBEDDING_DIM

logger = logging.getLogger(__name__)


class PhoneticEncoder:
    """
    Phonetic encoder for the P (phonetic) dimension of Babel space.
    
    Features:
    - Multi-language support (FR, EN, ES, DE)
    - IPA-based phoneme transcription
    - Articulatory complexity weighting
    - Normalized embeddings
    
    Example:
        >>> encoder = PhoneticEncoder(lang="fr")
        >>> embedding = encoder.get_phonetic_embedding("pouvoir")
        >>> print(embedding.shape)  # torch.Size([64])
    """
    
    def __init__(
        self,
        lang: str = "fr",
        embedding_dim: int = PHONEME_EMBEDDING_DIM,
        use_weights: bool = True,
    ):
        """
        Initialize the phonetic encoder.
        
        Args:
            lang: Language to use ('fr', 'en', 'es', 'de')
            embedding_dim: Dimension of phoneme embeddings
            use_weights: Whether to weight by articulatory complexity
        """
        self.lang = lang
        self.embedding_dim = embedding_dim
        self.use_weights = use_weights
        
        # Load phoneme map
        self.phoneme_map = PHONEME_MAP.get(lang, {})
        self.phoneme_weights = PHONEME_WEIGHTS
        
        # Initialize phoneme embeddings (random for now, replace with pre-trained)
        self.phoneme_embeddings = self._init_phoneme_embeddings()
        
        # Special phoneme for unknown
        self.unknown_phoneme = torch.randn(embedding_dim)
        self.unknown_phoneme = self.unknown_phoneme / torch.norm(self.unknown_phoneme)
        
        logger.info(f"PhoneticEncoder initialized: lang={lang}, dim={embedding_dim}")
    
    def _init_phoneme_embeddings(self) -> Dict[str, torch.Tensor]:
        """Initialize phoneme embeddings (random for now)."""
        embeddings = {}
        
        # Get all unique phonemes from the map
        all_phonemes = set()
        for word_phonemes in self.phoneme_map.values():
            for phoneme in word_phonemes:
                all_phonemes.add(phoneme)
        
        # Add phonemes from weights
        for phoneme in self.phoneme_weights.keys():
            all_phonemes.add(phoneme)
        
        # Initialize embeddings
        for phoneme in all_phonemes:
            embeddings[phoneme] = torch.randn(self.embedding_dim)
            embeddings[phoneme] = embeddings[phoneme] / torch.norm(embeddings[phoneme])
        
        return embeddings
    
    def load_pretrained_embeddings(self, filepath: str) -> None:
        """
        Load pre-trained phoneme embeddings.
        
        Args:
            filepath: Path to the embedding file (e.g., .pt, .npy, .json)
        """
        try:
            if filepath.endswith('.pt'):
                embeddings = torch.load(filepath)
                for phoneme, emb in embeddings.items():
                    if isinstance(emb, torch.Tensor):
                        self.phoneme_embeddings[phoneme] = emb
                    else:
                        self.phoneme_embeddings[phoneme] = torch.tensor(emb, dtype=torch.float32)
            elif filepath.endswith('.npy'):
                data = np.load(filepath, allow_pickle=True)
                if isinstance(data, dict):
                    for phoneme, emb in data.items():
                        self.phoneme_embeddings[phoneme] = torch.tensor(emb, dtype=torch.float32)
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
            
            logger.info(f"Loaded {len(self.phoneme_embeddings)} pre-trained phoneme embeddings")
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
    
    def save_embeddings(self, filepath: str) -> None:
        """Save phoneme embeddings to a file."""
        torch.save(self.phoneme_embeddings, filepath)
        logger.info(f"Saved phoneme embeddings to {filepath}")
    
    def word_to_phonemes(self, word: str) -> List[str]:
        """
        Convert a word to a list of IPA phonemes.
        
        Args:
            word: Word to convert
            
        Returns:
            List of phonemes (e.g., ['/p/', '/u/', '/v/', '/waʁ/'])
        """
        word = word.lower().strip()
        
        # Check if word is in the phoneme map
        if word in self.phoneme_map:
            return self.phoneme_map[word]
        
        # Try to decompose the word
        return self._decompose_word(word)
    
    def _decompose_word(self, word: str) -> List[str]:
        """
        Decompose a word into phonemes using rules.
        
        Args:
            word: Word to decompose
            
        Returns:
            List of phonemes
        """
        phonemes = []
        i = 0
        
        # French-specific rules
        if self.lang == "fr":
            while i < len(word):
                # Check for multi-character phonemes first
                if i + 1 < len(word):
                    pair = word[i:i+2]
                    if pair in ["ch", "gn", "ph", "th", "qu"]:
                        if pair == "ch":
                            phonemes.append("/ʃ/")
                        elif pair == "gn":
                            phonemes.append("/ɲ/")
                        elif pair == "ph":
                            phonemes.append("/f/")
                        elif pair == "th":
                            phonemes.append("/t/")
                        elif pair == "qu":
                            phonemes.append("/k/")
                            i += 1  # Skip next character
                        i += 2
                        continue
                
                # Single character
                char = word[i]
                
                if char in ['a', 'à', 'â', 'ä']:
                    phonemes.append("/a/")
                elif char in ['e', 'é', 'è', 'ê', 'ë']:
                    phonemes.append("/e/")
                elif char in ['i', 'î', 'ï']:
                    phonemes.append("/i/")
                elif char in ['o', 'ô', 'ö']:
                    phonemes.append("/o/")
                elif char in ['u', 'û', 'ü', 'ù']:
                    phonemes.append("/y/")
                elif char == 'p':
                    phonemes.append("/p/")
                elif char == 'b':
                    phonemes.append("/b/")
                elif char == 't':
                    phonemes.append("/t/")
                elif char == 'd':
                    phonemes.append("/d/")
                elif char == 'k':
                    phonemes.append("/k/")
                elif char == 'g':
                    phonemes.append("/ɡ/")
                elif char == 'f':
                    phonemes.append("/f/")
                elif char == 'v':
                    phonemes.append("/v/")
                elif char == 's':
                    phonemes.append("/s/")
                elif char == 'z':
                    phonemes.append("/z/")
                elif char == 'j':
                    phonemes.append("/ʒ/")
                elif char == 'm':
                    phonemes.append("/m/")
                elif char == 'n':
                    phonemes.append("/n/")
                elif char == 'l':
                    phonemes.append("/l/")
                elif char == 'r':
                    phonemes.append("/ʁ/")
                elif char == 'c':
                    # 'c' can be /k/ or /s/ depending on context
                    if i + 1 < len(word) and word[i+1] in ['e', 'é', 'è', 'ê', 'ë', 'i', 'î', 'ï']:
                        phonemes.append("/s/")
                    else:
                        phonemes.append("/k/")
                elif char == 'h':
                    # 'h' is silent in French
                    pass
                else:
                    # Unknown character -> use schwa
                    phonemes.append("/ə/")
                
                i += 1
        
        # English-specific rules
        elif self.lang == "en":
            while i < len(word):
                if i + 1 < len(word):
                    pair = word[i:i+2]
                    if pair == "ch":
                        phonemes.append("/tʃ/")
                        i += 2
                        continue
                    elif pair == "sh":
                        phonemes.append("/ʃ/")
                        i += 2
                        continue
                    elif pair == "th":
                        # 'th' can be /θ/ or /ð/
                        phonemes.append("/θ/")  # Default to voiceless
                        i += 2
                        continue
                    elif pair == "wh":
                        phonemes.append("/w/")
                        i += 2
                        continue
                    elif pair == "ph":
                        phonemes.append("/f/")
                        i += 2
                        continue
                    elif pair == "gh":
                        # 'gh' is often silent
                        i += 2
                        continue
                    elif pair == "kn":
                        phonemes.append("/n/")
                        i += 2
                        continue
                
                char = word[i]
                
                if char == 'a':
                    phonemes.append("/æ/")
                elif char == 'e':
                    phonemes.append("/ɛ/")
                elif char == 'i':
                    phonemes.append("/ɪ/")
                elif char == 'o':
                    phonemes.append("/ɒ/")
                elif char == 'u':
                    phonemes.append("/ʊ/")
                elif char == 'p':
                    phonemes.append("/p/")
                elif char == 'b':
                    phonemes.append("/b/")
                elif char == 't':
                    phonemes.append("/t/")
                elif char == 'd':
                    phonemes.append("/d/")
                elif char == 'k':
                    phonemes.append("/k/")
                elif char == 'g':
                    phonemes.append("/ɡ/")
                elif char == 'f':
                    phonemes.append("/f/")
                elif char == 'v':
                    phonemes.append("/v/")
                elif char == 's':
                    phonemes.append("/s/")
                elif char == 'z':
                    phonemes.append("/z/")
                elif char == 'j':
                    phonemes.append("/dʒ/")
                elif char == 'm':
                    phonemes.append("/m/")
                elif char == 'n':
                    phonemes.append("/n/")
                elif char == 'l':
                    phonemes.append("/l/")
                elif char == 'r':
                    phonemes.append("/ɹ/")
                elif char == 'h':
                    # 'h' at start of word
                    if i == 0:
                        phonemes.append("/h/")
                else:
                    phonemes.append("/ə/")
                
                i += 1
        
        # Default for other languages
        else:
            for char in word:
                if char in ['a', 'e', 'i', 'o', 'u']:
                    phonemes.append(f"/{char}/")
                else:
                    phonemes.append("/ə/")
        
        return phonemes if phonemes else ["/ə/"]
    
    def get_phoneme_weight(self, phoneme: str) -> float:
        """Get the articulatory weight for a phoneme."""
        return self.phoneme_weights.get(phoneme, self.phoneme_weights.get("default", 1.0))
    
    def get_phonetic_embedding(self, word: str) -> torch.Tensor:
        """
        Compute the phonetic embedding for a word.
        
        Args:
            word: Word to encode
            
        Returns:
            Normalized phonetic embedding [embedding_dim]
        """
        phonemes = self.word_to_phonemes(word)
        
        if not phonemes:
            return torch.zeros(self.embedding_dim)
        
        # Get embeddings and weights
        embeddings = []
        weights = []
        
        for phoneme in phonemes:
            if phoneme in self.phoneme_embeddings:
                embeddings.append(self.phoneme_embeddings[phoneme])
                weights.append(self.get_phoneme_weight(phoneme))
            else:
                # Use unknown phoneme embedding
                embeddings.append(self.unknown_phoneme)
                weights.append(0.5)  # Lower weight for unknown
        
        # Stack embeddings
        embeddings_tensor = torch.stack(embeddings, dim=0)  # [n_phonemes, embedding_dim]
        weights_tensor = torch.tensor(weights, dtype=torch.float32)  # [n_phonemes]
        
        # Weighted sum
        weighted_embeddings = embeddings_tensor * weights_tensor.unsqueeze(1)
        P = weighted_embeddings.sum(dim=0) / (weights_tensor.sum() + 1e-10)
        
        # Normalize
        P = P / (torch.norm(P) + 1e-10)
        
        return P
    
    def batch_encode(self, words: List[str]) -> torch.Tensor:
        """
        Encode a batch of words to phonetic embeddings.
        
        Args:
            words: List of words
            
        Returns:
            Phonetic embeddings [batch_size, embedding_dim]
        """
        return torch.stack([self.get_phonetic_embedding(word) for word in words], dim=0)
    
    def get_phonetic_features(self, word: str) -> Dict[str, Any]:
        """
        Extract phonetic features for a word.
        
        Args:
            word: Word to analyze
            
        Returns:
            Dictionary of phonetic features
        """
        phonemes = self.word_to_phonemes(word)
        
        features = {
            "phonemes": phonemes,
            "num_phonemes": len(phonemes),
            "place_of_articulation": [],
            "manner_of_articulation": [],
            "voicing": [],
            "is_vowel": False,
            "num_syllables": 0,
        }
        
        # Count vowels (approximate syllable count)
        vowel_phonemes = {"/a/", "/e/", "/ɛ/", "/ə/", "/i/", "/o/", "/u/", "/y/", "/ø/",
                        "/æ/", "/ɑː/", "/ɒ/", "/ʊ/", "/ʌ/", "/ɜː/", "/ɪ/", "/iː/",
                        "/eɪ/", "/aɪ/", "/ɔɪ/", "/aʊ/", "/eə/", "/ɪə/", "/ɔː/", "/uː/"}
        
        for phoneme in phonemes:
            if phoneme in vowel_phonemes:
                features["is_vowel"] = True
                features["num_syllables"] += 1
        
        return features
    
    def set_phoneme_embedding(self, phoneme: str, embedding: torch.Tensor) -> None:
        """Set the embedding for a specific phoneme."""
        embedding = embedding / torch.norm(embedding)
        self.phoneme_embeddings[phoneme] = embedding
    
    def fine_tune(self, word_embeddings: Dict[str, torch.Tensor]) -> None:
        """
        Fine-tune phoneme embeddings based on word embeddings.
        
        Args:
            word_embeddings: Dictionary of {word: embedding}
        """
        # Group words by their phonemes
        phoneme_to_words = {}
        for word, embedding in word_embeddings.items():
            phonemes = self.word_to_phonemes(word)
            for phoneme in phonemes:
                if phoneme not in phoneme_to_words:
                    phoneme_to_words[phoneme] = []
                phoneme_to_words[phoneme].append((word, embedding))
        
        # Update phoneme embeddings as the mean of their words' embeddings
        for phoneme, word_embs in phoneme_to_words.items():
            if len(word_embs) >= 3:  # Only update if we have enough data
                mean_embedding = torch.stack([emb for _, emb in word_embs], dim=0).mean(dim=0)
                mean_embedding = mean_embedding / torch.norm(mean_embedding)
                self.phoneme_embeddings[phoneme] = mean_embedding
        
        logger.info(f"Fine-tuned {len(phoneme_to_words)} phoneme embeddings")


if __name__ == "__main__":
    # Test the phonetic encoder
    encoder = PhoneticEncoder(lang="fr")
    
    # Test with some words
    test_words = ["pouvoir", "voir", "savoir", "lumière", "comprendre"]
    
    print("Phonetic Encoder Test")
    print("=" * 80)
    
    for word in test_words:
        phonemes = encoder.word_to_phonemes(word)
        embedding = encoder.get_phonetic_embedding(word)
        features = encoder.get_phonetic_features(word)
        
        print(f"Word: {word}")
        print(f"  Phonemes: {phonemes}")
        print(f"  Embedding shape: {embedding.shape}")
        print(f"  Features: {features}")
        print()
