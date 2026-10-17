# Exemple de Quantisation Vectorielle

## 🎯 Introduction

Cet exemple montre comment utiliser les fonctionnalités de quantisation d'Apourvoir pour compresser et manipuler efficacement les vecteurs de savoir et les matrices de pouvoir.

## 📦 Prérequis

```bash
# Installer les dépendances
pip install numpy scipy matplotlib scikit-learn
```

## 🚀 Exemple de Code

```python
"""
Exemple de quantisation avec Apourvoir.
"""

import numpy as np
import matplotlib.pyplot as plt
from apourvoir.core import SavoirVector, PouvoirMatrix, SavoirSpace
from apourvoir.utils import UniformQuantizer, MatrixQuantizer, VectorQuantizer
from apourvoir.utils import (
    quantization_error,
    relative_quantization_error,
    quantization_snr,
    compression_ratio
)


def main():
    print("=== Exemple de Quantisation avec Apourvoir ===\n")
    
    # 1. Créer un espace de savoir
    print("1. Création de l'espace du savoir...")
    space = SavoirSpace(dimension=3)
    
    # 2. Créer des vecteurs de savoir
    print("2. Création des vecteurs de savoir...")
    n_vectors = 100
    savoir_vectors = []
    for i in range(n_vectors):
        vec = np.random.randn(3) * 10  # Vecteurs aléatoires
        savoir_vectors.append(space.create_vector(vec, label=f"Savoir_{i}"))
    
    # 3. Créer une matrice de pouvoir
    print("3. Création de la matrice de pouvoir...")
    pouvoir_matrix = np.random.randn(3, 3)
    pouvoir = PouvoirMatrix(pouvoir_matrix, label="Pouvoir")
    
    print(f"Matrice de pouvoir originale:\n{pouvoir.matrix}\n")
    
    # ========================================================================
    # EXEMPLE 1 : Quantisation Uniforme
    # ========================================================================
    print("=" * 60)
    print("EXEMPLE 1 : Quantisation Uniforme")
    print("=" * 60)
    
    # Créer un quantificateur uniforme
    uniform_quantizer = UniformQuantizer(n_bits=8)
    
    # Quantifier un vecteur de savoir
    sample_savoir = savoir_vectors[0]
    print(f"\nVecteur de savoir original: {sample_savoir.vector}")
    
    quantized_savoir, scale, min_val = uniform_quantizer.quantize_savoir(sample_savoir)
    print(f"Vecteur quantifié: {quantized_savoir.vector}")
    print(f"Échelle: {scale:.4f}, Valeur minimale: {min_val:.4f}")
    
    # Déquantifier
    dequantized_vector = uniform_quantizer.dequantize(
        quantized_savoir.vector, scale, min_val
    )
    print(f"Vecteur déquantifié: {dequantized_vector}")
    
    # Calculer les métriques d'erreur
    error = quantization_error(sample_savoir.vector, dequantized_vector)
    rel_error = relative_quantization_error(sample_savoir.vector, dequantized_vector)
    snr = quantization_snr(sample_savoir.vector, dequantized_vector)
    
    print(f"\nMétriques d'erreur:")
    print(f"  Erreur absolue: {error:.6f}")
    print(f"  Erreur relative: {rel_error:.6f}")
    print(f"  SNR: {snr:.2f} dB")
    
    # ========================================================================
    # EXEMPLE 2 : Quantisation de Matrice par SVD
    # ========================================================================
    print("\n" + "=" * 60)
    print("EXEMPLE 2 : Quantisation de Matrice par SVD")
    print("=" * 60)
    
    matrix_quantizer = MatrixQuantizer(n_bits=8)
    
    # Quantifier par SVD avec différents rangs
    for target_rank in [3, 2, 1]:
        print(f"\nQuantification avec rang {target_rank}:")
        
        quantized_pouvoir, info = matrix_quantizer.quantize_by_rank(
            pouvoir, target_rank=target_rank
        )
        
        error = quantization_error(pouvoir.matrix, quantized_pouvoir.matrix)
        rel_error = relative_quantization_error(pouvoir.matrix, quantized_pouvoir.matrix)
        
        print(f"  Erreur absolue: {error:.6f}")
        print(f"  Erreur relative: {rel_error:.6f}")
        print(f"  Matrice quantifiée:\n{quantized_pouvoir.matrix}")
    
    # ========================================================================
    # EXEMPLE 3 : Quantisation Vectorielle (K-Means)
    # ========================================================================
    print("\n" + "=" * 60)
    print("EXEMPLE 3 : Quantisation Vectorielle (K-Means)")
    print("=" * 60)
    
    vector_quantizer = VectorQuantizer(n_clusters=32)
    
    # Apprendre le codebook
    print("\nApprentissage du codebook...")
    vector_quantizer.fit(savoir_vectors[:50])  # Utiliser la moitié pour l'apprentissage
    
    # Quantifier les vecteurs restants
    indices, quantized_vectors = vector_quantizer.quantize(savoir_vectors[50:])
    
    print(f"Nombre de clusters: {vector_quantizer.n_clusters}")
    print(f"Indices des premiers 5 vecteurs: {indices[:5]}")
    
    # Calculer l'erreur moyenne
    total_error = 0
    for orig, quant in zip(savoir_vectors[50:], quantized_vectors):
        total_error += quantization_error(orig.vector, quant.vector)
    avg_error = total_error / len(savoir_vectors[50:])
    
    print(f"Erreur moyenne de quantification: {avg_error:.6f}")
    
    # Afficher quelques exemples
    print("\nExemples de quantification:")
    for i in range(3):
        orig = savoir_vectors[50 + i]
        quant = quantized_vectors[i]
        error = quantization_error(orig.vector, quant.vector)
        print(f"  Vecteur {i}: erreur = {error:.6f}")
    
    # ========================================================================
    # EXEMPLE 4 : Analyse des Performances
    # ========================================================================
    print("\n" + "=" * 60)
    print("EXEMPLE 4 : Analyse des Performances")
    print("=" * 60)
    
    # Tester différents nombres de bits
    bit_ranges = [4, 6, 8, 10, 12, 16]
    errors = []
    ratios = []
    
    for n_bits in bit_ranges:
        quantizer = UniformQuantizer(n_bits=n_bits)
        
        # Quantifier tous les vecteurs
        quantized_all = []
        for vec in savoir_vectors:
            q, scale, min_val = quantizer.quantize(vec)
            dq = quantizer.dequantize(q, scale, min_val)
            quantized_all.append(dq)
        
        # Calculer l'erreur moyenne
        total_error = sum(
            quantization_error(s.vector, q) 
            for s, q in zip(savoir_vectors, quantized_all)
        )
        avg_error = total_error / len(savoir_vectors)
        errors.append(avg_error)
        
        # Calculer le taux de compression
        original_size = len(savoir_vectors) * 3 * 64  # 64 bits par float
        quantized_size = len(savoir_vectors) * 3 * n_bits
        ratio = original_size / quantized_size
        ratios.append(ratio)
        
        print(f"{n_bits} bits: erreur moyenne = {avg_error:.6f}, ratio = {ratio:.2f}x")
    
    # Tracer la courbe taux-distorsion
    plt.figure(figsize=(10, 6))
    plt.plot(bit_ranges, errors, 'bo-', label='Erreur moyenne')
    plt.xlabel('Nombre de bits')
    plt.ylabel('Erreur moyenne')
    plt.title('Courbe Taux-Distorsion')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('rate_distortion_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========================================================================
    # EXEMPLE 5 : Quantisation Adaptative
    # ========================================================================
    print("\n" + "=" * 60)
    print("EXEMPLE 5 : Quantisation Adaptative")
    print("=" * 60)
    
    from apourvoir.utils import AdaptiveQuantizer
    
    adaptive_quantizer = AdaptiveQuantizer(n_bits=8)
    
    # Apprendre les plages à partir des données
    print("\nApprentissage des plages adaptatives...")
    adaptive_quantizer.fit([s.vector for s in savoir_vectors[:50]])
    
    # Quantifier avec les plages apprises
    sample_vec = savoir_vectors[0]
    quantized, ranges, _ = adaptive_quantizer.quantize(sample_vec)
    dequantized = adaptive_quantizer.dequantize(quantized, ranges)
    
    print(f"Vecteur original: {sample_vec.vector}")
    print(f"Vecteur quantifié: {quantized}")
    print(f"Vecteur déquantifié: {dequantized}")
    print(f"Erreur: {quantization_error(sample_vec.vector, dequantized):.6f}")
    
    print("\nPlages apprises pour chaque dimension:")
    for i, range_info in adaptive_quantizer.ranges.items():
        print(f"  Dimension {i}: [{range_info['min']:.2f}, {range_info['max']:.2f}]")


if __name__ == "__main__":
    main()
```

## 📊 Résultats Attendus

En exécutant ce code, vous devriez voir :

1. **Sortie console** : 
   - Les résultats de la quantisation uniforme
   - Les résultats de la quantisation par SVD
   - Les résultats de la quantisation vectorielle
   - L'analyse des performances
   - Les résultats de la quantisation adaptative

2. **Visualisation** : Une courbe taux-distorsion montrant le compromis entre le nombre de bits et l'erreur de quantification

## 🔍 Interprétation

### Quantisation Uniforme

- **Erreur absolue** : L'erreur moyenne entre le vecteur original et le vecteur déquantifié
- **Erreur relative** : L'erreur normalisée par la norme du vecteur original
- **SNR (Signal-to-Noise Ratio)** : Plus le SNR est élevé, meilleure est la qualité de la quantification

### Quantisation par SVD

La quantification par SVD permet de :
- Réduire le rang de la matrice (compression)
- Quantifier les valeurs singulières et les vecteurs propres
- Reconstruire une matrice proche de l'originale

### Quantisation Vectorielle (K-Means)

- **Codebook** : Ensemble de vecteurs représentatifs appris à partir des données
- **Indices** : Chaque vecteur est représenté par un indice dans le codebook
- **Erreur** : Distance entre le vecteur original et son représentant dans le codebook

### Analyse des Performances

La courbe taux-distorsion montre :
- Plus le nombre de bits augmente, plus l'erreur diminue
- Le taux de compression est l'inverse du nombre de bits (par rapport à 64 bits)
- Il y a un compromis entre la qualité et le taux de compression

## 🎨 Visualisation

La courbe taux-distorsion permet de visualiser :
- **Axe X** : Nombre de bits utilisés pour la quantification
- **Axe Y** : Erreur moyenne de quantification
- **Tendance** : Diminution exponentielle de l'erreur avec l'augmentation des bits

## 🔧 Personnalisation

### Changer le Nombre de Bits

```python
# Utiliser différents nombres de bits
for n_bits in [4, 8, 12, 16]:
    quantizer = UniformQuantizer(n_bits=n_bits)
    # ...
```

### Changer la Taille du Codebook

```python
# Utiliser différents nombres de clusters
for n_clusters in [16, 32, 64, 128]:
    vector_quantizer = VectorQuantizer(n_clusters=n_clusters)
    # ...
```

### Quantification par Blocs

```python
matrix_quantizer = MatrixQuantizer(n_bits=8)

# Quantifier par blocs de 2x2
quantized_pouvoir = matrix_quantizer.quantize_by_blocks(pouvoir, block_size=2)
```

## 📚 Concepts Clés

### Quantisation

La quantification est le processus de mapping des valeurs continues à un ensemble discret de valeurs. Cela permet de :
- Réduire la taille des données
- Accélérer les calculs
- Faciliter le stockage et la transmission

### Types de Quantisation

1. **Quantisation scalaire** : Chaque valeur est quantifiée indépendamment
2. **Quantisation vectorielle** : Des groupes de valeurs sont quantifiés ensemble
3. **Quantisation adaptative** : Les plages de quantification s'adaptent aux données
4. **Quantification par SVD** : Utilise la décomposition en valeurs singulières pour une compression efficace

### Métriques d'Évaluation

1. **Erreur absolue** : Différence absolue entre l'original et le quantifié
2. **Erreur relative** : Erreur normalisée par l'amplitude du signal
3. **SNR** : Rapport signal/bruit
4. **Taux de compression** : Facteur de réduction de la taille des données

## 🚀 Prochaines Étapes

- [ ] Explorer différentes méthodes de quantification
- [ ] Tester avec des vecteurs de dimension supérieure
- [ ] Comparer les performances des différentes méthodes
- [ ] Intégrer la quantification dans des pipelines d'alignement
- [ ] Explorer la quantification pour des applications spécifiques
