"""
Phonetic Encoder for Babel Space

Converts words to phonetic embeddings by:
1. Transcribing words to IPA phonemes (dictionary, then letter rules)
2. Describing each IPA segment by its articulatory features (deterministic,
   see ``features.py``)
3. Weighting by articulatory complexity
4. Producing normalized phonetic vectors laid out as
   [weighted mean | first phoneme | last phoneme], zero-padded to embedding_dim

No random initialisation is used: the same word always gets the same vector,
and words that sound alike get similar vectors.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

from .phoneme_map import PHONEME_MAP, PHONEME_WEIGHTS, PHONEME_EMBEDDING_DIM
from .features import NUM_FEATURES, segment_ipa, segment_features

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
        if embedding_dim < 3 * NUM_FEATURES:
            raise ValueError(f"embedding_dim must be >= {3 * NUM_FEATURES}, got {embedding_dim}")
        self.lang = lang
        self.embedding_dim = embedding_dim
        self.use_weights = use_weights

        # Load phoneme map
        self.phoneme_map = PHONEME_MAP.get(lang, {})
        self.phoneme_weights = PHONEME_WEIGHTS

        # Phoneme-unit embeddings (articulatory features, size NUM_FEATURES)
        self.phoneme_embeddings = self._init_phoneme_embeddings()

        logger.info(f"PhoneticEncoder initialized: lang={lang}, dim={embedding_dim}")

    def _features_for_unit(self, phoneme: str) -> Optional[torch.Tensor]:
        """Weighted mean of the articulatory features of an IPA unit (e.g. '/vwaʁ/')."""
        segments = segment_ipa(phoneme)
        if not segments:
            return None
        vecs = torch.stack([segment_features(s) for s in segments])
        if self.use_weights:
            w = torch.tensor([self.get_phoneme_weight(f"/{s.rstrip('~')}/") for s in segments])
        else:
            w = torch.ones(len(segments))
        return (vecs * w.unsqueeze(1)).sum(dim=0) / w.sum()

    def _init_phoneme_embeddings(self) -> Dict[str, torch.Tensor]:
        """Compute deterministic feature embeddings for every known phoneme unit."""
        embeddings = {}

        all_phonemes = set(self.phoneme_weights.keys()) - {"default"}
        for word_phonemes in self.phoneme_map.values():
            all_phonemes.update(word_phonemes)

        for phoneme in sorted(all_phonemes):
            vec = self._features_for_unit(phoneme)
            if vec is not None:
                embeddings[phoneme] = vec

        return embeddings

    def _unit_embedding(self, phoneme: str) -> Optional[torch.Tensor]:
        if phoneme not in self.phoneme_embeddings:
            vec = self._features_for_unit(phoneme)
            if vec is None:
                return None
            self.phoneme_embeddings[phoneme] = vec
        return self.phoneme_embeddings[phoneme]
    
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
        phonemes = self.word_to_phonemes(word) if word else []

        # Embeddings of known units (unknown symbols are skipped, not randomised)
        embeddings = []
        weights = []
        for phoneme in phonemes:
            emb = self._unit_embedding(phoneme)
            if emb is not None:
                embeddings.append(emb)
                weights.append(self.get_phoneme_weight(phoneme) if self.use_weights else 1.0)

        if not embeddings:
            return torch.zeros(self.embedding_dim)

        embeddings_tensor = torch.stack(embeddings, dim=0)  # [n_phonemes, NUM_FEATURES]
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        mean = (embeddings_tensor * weights_tensor.unsqueeze(1)).sum(dim=0) / weights_tensor.sum()

        P = torch.zeros(self.embedding_dim)
        P[:NUM_FEATURES] = mean
        P[NUM_FEATURES:2 * NUM_FEATURES] = embeddings_tensor[0]
        P[2 * NUM_FEATURES:3 * NUM_FEATURES] = embeddings_tensor[-1]

        return P / (torch.norm(P) + 1e-10)
    
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
        for word, embedding in word_embeddings.items():
            if embedding.numel() != NUM_FEATURES:
                raise ValueError(f"fine_tune expects {NUM_FEATURES}-dim embeddings, got {embedding.numel()} for '{word}'")

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
