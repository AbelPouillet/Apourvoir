# Méthodologie de Quantisation Vectorielle

## 🎯 Objectifs de la Quantisation

La quantisation dans le cadre de l'alignement vectoriel du savoir/pouvoir permet de :

1. **Réduire la complexité** : Représenter les vecteurs et transformations de manière compacte
2. **Améliorer l'efficacité** : Accélérer les calculs et réduire les besoins en mémoire
3. **Faciliter l'interprétation** : Rendre les structures plus compréhensibles
4. **Assurer la robustesse** : Réduire le bruit et les erreurs numériques
5. **Permettre le stockage** : Faciliter le stockage et la transmission des données

## 📐 Approches de Quantisation

### 1. Quantisation Scalaire

#### a. Quantisation Uniforme

```python
import numpy as np

def uniform_quantize(vector, n_bits=8, range_min=None, range_max=None):
    """
    Quantification uniforme d'un vecteur
    
    Args:
        vector: Vecteur à quantifier
        n_bits: Nombre de bits pour la quantification
        range_min: Valeur minimale (optionnel)
        range_max: Valeur maximale (optionnel)
    
    Returns:
        Vecteur quantifié, échelle de quantification
    """
    if range_min is None:
        range_min = np.min(vector)
    if range_max is None:
        range_max = np.max(vector)
    
    # Calculer l'échelle
    scale = (2 ** n_bits - 1) / (range_max - range_min)
    
    # Quantification
    quantized = np.round((vector - range_min) * scale).astype(np.int16)
    
    return quantized, scale, range_min

def uniform_dequantize(quantized, scale, range_min):
    """Déquantification uniforme"""
    return quantized / scale + range_min
```

#### b. Quantisation Non-Uniforme (Lloyd-Max)

```python
from scipy.cluster.vq import kmeans, kmeans2

def lloyd_max_quantize(vectors, n_levels=256):
    """
    Quantification optimale selon l'algorithme de Lloyd-Max
    
    Args:
        vectors: Matrice de vecteurs à quantifier (chaque ligne est un vecteur)
        n_levels: Nombre de niveaux de quantification
    
    Returns:
        Vecteurs quantifiés, codebook (niveaux de quantification)
    """
    # Apprendre le codebook
    codebook, _ = kmeans2(vectors, n_levels, minit='points')
    
    # Quantification
    quantized = np.zeros_like(vectors)
    for i, vec in enumerate(vectors):
        # Trouver le niveau le plus proche
        distances = np.linalg.norm(codebook - vec, axis=1)
        closest_idx = np.argmin(distances)
        quantized[i] = codebook[closest_idx]
    
    return quantized, codebook
```

### 2. Quantisation Vectorielle (VQ)

#### a. Quantification par K-Means

```python
def vector_quantization(vectors, n_clusters=256):
    """
    Quantification vectorielle utilisant K-Means
    
    Args:
        vectors: Matrice de vecteurs à quantifier
        n_clusters: Nombre de clusters (niveaux de quantification)
    
    Returns:
        Indices des clusters, centroïdes (codebook)
    """
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    kmeans.fit(vectors)
    
    # Retourner les indices et le codebook
    indices = kmeans.labels_
    codebook = kmeans.cluster_centers_
    
    return indices, codebook
```

#### b. Quantification par SOM (Self-Organizing Maps)

```python
from minisom import MiniSom

def som_quantization(vectors, grid_size=(10, 10)):
    """
    Quantification vectorielle utilisant des cartes auto-organisatrices
    
    Args:
        vectors: Matrice de vecteurs à quantifier
        grid_size: Taille de la grille SOM
    
    Returns:
        Indices des neurones gagnants, modèle SOM
    """
    som = MiniSom(grid_size[0], grid_size[1], vectors.shape[1], sigma=0.5, learning_rate=0.5)
    som.train_random(vectors, 1000)
    
    # Quantification
    winner_indices = np.array([som.winner(x) for x in vectors])
    
    return winner_indices, som
```

### 3. Quantisation des Matrices de Pouvoir

#### a. Quantisation par Décomposition

```python
def matrix_quantization_by_svd(matrix, n_bits=8, k=None):
    """
    Quantification d'une matrice par décomposition SVD
    
    Args:
        matrix: Matrice à quantifier
        n_bits: Nombre de bits pour la quantification des valeurs singulières
        k: Nombre de valeurs singulières à conserver (optionnel)
    
    Returns:
        Matrice quantifiée
    """
    # Décomposition SVD
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    
    # Quantification des valeurs singulières
    if k is not None:
        U = U[:, :k]
        S = S[:k]
        Vt = Vt[:k, :]
    
    # Quantifier S
    S_quant, S_scale, S_min = uniform_quantize(S, n_bits)
    S_quant = uniform_dequantize(S_quant, S_scale, S_min)
    
    # Reconstruire la matrice
    quantized_matrix = U @ np.diag(S_quant) @ Vt
    
    return quantized_matrix
```

#### b. Quantisation par Blocs

```python
def block_quantization(matrix, block_size=4, n_bits=8):
    """
    Quantification d'une matrice par blocs
    
    Args:
        matrix: Matrice à quantifier
        block_size: Taille des blocs
        n_bits: Nombre de bits pour la quantification
    
    Returns:
        Matrice quantifiée
    """
    m, n = matrix.shape
    quantized = np.zeros_like(matrix)
    
    for i in range(0, m, block_size):
        for j in range(0, n, block_size):
            block = matrix[i:i+block_size, j:j+block_size]
            
            # Quantifier le bloc
            quant_block, scale, min_val = uniform_quantize(block, n_bits)
            quantized[i:i+block_size, j:j+block_size] = uniform_dequantize(quant_block, scale, min_val)
    
    return quantized
```

#### c. Quantisation par Rang

```python
def rank_quantization(matrix, target_rank=2, n_bits=8):
    """
    Quantification d'une matrice en réduisant son rang
    
    Args:
        matrix: Matrice à quantifier
        target_rank: Rang cible
        n_bits: Nombre de bits pour la quantification
    
    Returns:
        Matrice quantifiée de rang réduit
    """
    # Réduction de rang par SVD
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    
    if target_rank < len(S):
        U = U[:, :target_rank]
        S = S[:target_rank]
        Vt = Vt[:target_rank, :]
    
    # Quantification
    U_quant, U_scale, U_min = uniform_quantize(U, n_bits)
    S_quant, S_scale, S_min = uniform_quantize(S, n_bits)
    Vt_quant, Vt_scale, Vt_min = uniform_quantize(Vt, n_bits)
    
    U_quant = uniform_dequantize(U_quant, U_scale, U_min)
    S_quant = uniform_dequantize(S_quant, S_scale, S_min)
    Vt_quant = uniform_dequantize(Vt_quant, Vt_scale, Vt_min)
    
    # Reconstruire
    quantized_matrix = U_quant @ np.diag(S_quant) @ Vt_quant
    
    return quantized_matrix
```

## 📊 Évaluation de la Quantisation

### 1. Métriques de Qualité

#### a. Erreur de Quantification

```python
def quantization_error(original, quantized):
    """Calculer l'erreur de quantification"""
    return np.linalg.norm(original - quantized)

def relative_quantization_error(original, quantized):
    """Calculer l'erreur relative de quantification"""
    return quantization_error(original, quantized) / np.linalg.norm(original)
```

#### b. Rapport Signal/Bruit (SNR)

```python
def quantization_snr(original, quantized):
    """Calculer le rapport signal/bruit de la quantification"""
    error = quantization_error(original, quantized)
    signal_power = np.linalg.norm(original) ** 2
    noise_power = error ** 2
    
    return 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
```

#### c. Taux de Compression

```python
def compression_ratio(original, quantized, original_bits, quantized_bits):
    """Calculer le taux de compression"""
    original_size = original.size * original_bits
    quantized_size = quantized.size * quantized_bits
    
    return original_size / quantized_size
```

### 2. Courbe Taux-Distorsion

```python
def rate_distortion_curve(vectors, max_bits=16, n_samples=10):
    """
    Calculer la courbe taux-distorsion pour la quantification
    
    Args:
        vectors: Matrice de vecteurs
        max_bits: Nombre maximum de bits à tester
        n_samples: Nombre de points sur la courbe
    
    Returns:
        Liste de taux de compression, liste de distorsions
    """
    bits_range = np.linspace(1, max_bits, n_samples)
    compression_ratios = []
    distortions = []
    
    original_bits = 64  # Supposons des floats 64 bits
    
    for n_bits in bits_range:
        # Quantification
        quantized, _, _ = uniform_quantize(vectors, int(n_bits))
        dequantized = uniform_dequantize(quantized, _, _)
        
        # Calculer la distorsion (erreur moyenne)
        distortion = np.mean([np.linalg.norm(v - d) for v, d in zip(vectors, dequantized)])
        
        # Calculer le taux de compression
        ratio = compression_ratio(vectors, quantized, original_bits, int(n_bits))
        
        compression_ratios.append(ratio)
        distortions.append(distortion)
    
    return compression_ratios, distortions
```

## 🔄 Quantisation Adaptative

### 1. Quantisation Adaptative aux Données

```python
class AdaptiveQuantizer:
    def __init__(self, n_bits=8):
        self.n_bits = n_bits
        self.ranges = {}
    
    def fit(self, vectors):
        """Apprendre les plages de quantification à partir des données"""
        for i in range(vectors.shape[1]):
            self.ranges[i] = {
                'min': np.min(vectors[:, i]),
                'max': np.max(vectors[:, i])
            }
    
    def quantize(self, vector):
        """Quantifier un vecteur en utilisant les plages apprises"""
        quantized = np.zeros_like(vector)
        for i in range(len(vector)):
            range_info = self.ranges.get(i, {'min': np.min(vector), 'max': np.max(vector)})
            quantized[i], _, _ = uniform_quantize(
                np.array([vector[i]]),
                self.n_bits,
                range_info['min'],
                range_info['max']
            )[0]
        return quantized
    
    def dequantize(self, quantized):
        """Déquantifier un vecteur"""
        dequantized = np.zeros_like(quantized)
        for i in range(len(quantized)):
            range_info = self.ranges.get(i, {'min': 0, 'max': 1})
            dequantized[i] = uniform_dequantize(
                np.array([quantized[i]]),
                (2 ** self.n_bits - 1) / (range_info['max'] - range_info['min']),
                range_info['min']
            )[0]
        return dequantized
```

### 2. Quantisation Dynamique

```python
class DynamicQuantizer:
    def __init__(self, initial_bits=8):
        self.bits = initial_bits
        self.error_history = []
        self.target_error = 0.01  # Erreur cible relative
    
    def quantize(self, vector, max_bits=16):
        """Quantifier avec ajustement dynamique du nombre de bits"""
        # Essayer avec le nombre de bits actuel
        quantized, scale, min_val = uniform_quantize(vector, self.bits)
        dequantized = uniform_dequantize(quantized, scale, min_val)
        
        error = relative_quantization_error(vector, dequantized)
        self.error_history.append(error)
        
        # Ajuster le nombre de bits
        if error > self.target_error and self.bits < max_bits:
            self.bits += 1
            # Requantifier avec plus de bits
            return self.quantize(vector, max_bits)
        elif error < self.target_error / 2 and self.bits > 1:
            self.bits -= 1
            # Requantifier avec moins de bits
            return self.quantize(vector, max_bits)
        
        return quantized, scale, min_val
```

## 📦 Applications Spécifiques

### 1. Quantisation pour le Stockage

```python
class StorageQuantizer:
    def __init__(self, max_size_kb):
        self.max_size_kb = max_size_kb
    
    def quantize_for_storage(self, vectors):
        """Quantifier des vecteurs pour respecter une limite de stockage"""
        # Calculer la taille actuelle
        current_size = vectors.size * 8  # 64 bits par float
        target_size = self.max_size_kb * 1024 * 8  # en bits
        
        if current_size <= target_size:
            return vectors, 64  # Pas besoin de quantifier
        
        # Calculer le nombre de bits nécessaire
        ratio = current_size / target_size
        required_bits = max(1, int(64 / ratio))
        
        # Quantification
        quantized, scale, min_val = uniform_quantize(vectors, required_bits)
        
        return quantized, required_bits, scale, min_val
```

### 2. Quantisation pour la Transmission

```python
class TransmissionQuantizer:
    def __init__(self, bandwidth_bps, max_latency_ms):
        self.bandwidth_bps = bandwidth_bps
        self.max_latency_ms = max_latency_ms
    
    def quantize_for_transmission(self, vectors):
        """Quantifier des vecteurs pour respecter des contraintes de bande passante"""
        # Calculer le nombre maximum de bits qui peuvent être transmis
        max_bits = self.bandwidth_bps * (self.max_latency_ms / 1000)
        
        # Calculer le nombre de bits par élément
        total_elements = vectors.size
        bits_per_element = max_bits / total_elements
        
        # Limiter à un nombre raisonnable
        n_bits = min(16, max(1, int(bits_per_element)))
        
        # Quantification
        quantized, scale, min_val = uniform_quantize(vectors, n_bits)
        
        return quantized, n_bits, scale, min_val
```

## 🔧 Outils et Bibliothèques

### 1. Bibliothèques Python

- **NumPy** : Quantification de base
- **SciPy** : Algorithmes avancés de quantification
- **scikit-learn** : Quantification vectorielle (K-Means)
- **MiniSom** : Quantification par SOM
- **PyTorch/TensorFlow** : Quantification pour l'apprentissage profond

### 2. Outils Spécialisés

- **FAISS** (Facebook) : Recherche efficace dans des espaces vectoriels quantifiés
- **Annoy** (Spotify) : Approximate Nearest Neighbors avec quantification
- **ScaNN** (Google) : Recherche vectorielle scalable

### 3. Formats de Stockage

- **Binary formats** : Stockage compact des vecteurs quantifiés
- **HDF5** : Stockage efficace de grandes matrices
- **Parquet** : Format colonne pour les données tabulaires
- **Protocol Buffers** : Sérialisation efficace

## 📝 Bonnes Pratiques

### 1. Choix du Nombre de Bits

- **1-4 bits** : Quantification très agressive, perte de qualité significative
- **5-8 bits** : Bon compromis pour de nombreuses applications
- **9-12 bits** : Qualité proche du float 32 bits
- **13-16 bits** : Qualité proche du float 64 bits

### 2. Gestion des Plages

- **Plages fixes** : Simples mais peuvent ne pas être optimales
- **Plages adaptatives** : Mieux adaptées aux données spécifiques
- **Plages dynamiques** : S'adaptent aux changements dans les données

### 3. Évaluation

- **Toujours évaluer** : Mesurer l'impact de la quantification
- **Comparer plusieurs méthodes** : Tester différentes approches
- **Considérer le contexte** : Choisir la quantification en fonction de l'application
- **Documenter** : Garder une trace des paramètres et des résultats

### 4. Optimisation

- **Quantification par blocs** : Pour de meilleures performances locales
- **Quantification hiérarchique** : Pour capturer des patterns à différentes échelles
- **Quantification adaptative** : Pour s'adapter aux caractéristiques des données
- **Quantification apprise** : Utiliser l'apprentissage automatique pour optimiser

## 📚 Références

- [Lloyd, S. P. (1982). "Least Squares Quantization in PCM"]
- [Gray, R. M. (1984). "Vector Quantization"]
- [Gersho, A., & Gray, R. M. (1992). "Vector Quantization and Signal Compression"]
- [NumPy Documentation - Array manipulation](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [SciPy Documentation - Cluster](https://docs.scipy.org/doc/scipy/reference/cluster.html)
