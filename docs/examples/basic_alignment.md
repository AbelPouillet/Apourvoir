# Exemple d'Alignement de Base

## 🎯 Introduction

Cet exemple montre comment utiliser le cadre Apourvoir pour créer un alignement de base entre des vecteurs de savoir et des transformations de pouvoir.

## 📦 Prérequis

```bash
# Installer les dépendances
pip install numpy scipy matplotlib
```

## 🚀 Exemple de Code

```python
"""
Exemple d'alignement de base avec Apourvoir.
"""

import numpy as np
import matplotlib.pyplot as plt
from apourvoir.core import SavoirVector, PouvoirMatrix, SavoirSpace, AlignmentEngine


def main():
    # 1. Créer un espace de savoir en 2D
    print("1. Création de l'espace du savoir...")
    space = SavoirSpace(dimension=2)
    
    # 2. Créer des vecteurs de savoir
    print("2. Création des vecteurs de savoir...")
    savoir1 = space.create_vector([1.0, 0.0], label="Savoir A")
    savoir2 = space.create_vector([0.0, 1.0], label="Savoir B")
    savoir3 = space.create_vector([1.0, 1.0], label="Savoir C")
    
    # 3. Créer des vecteurs cibles (objectifs)
    print("3. Création des vecteurs cibles...")
    target1 = space.create_vector([0.0, 1.0], label="Cible A")
    target2 = space.create_vector([-1.0, 0.0], label="Cible B")
    target3 = space.create_vector([0.0, -1.0], label="Cible C")
    
    # 4. Créer une matrice de pouvoir initiale (identité)
    print("4. Création de la matrice de pouvoir initiale...")
    initial_pouvoir = PouvoirMatrix(np.eye(2), label="Pouvoir Initial")
    
    # 5. Créer le moteur d'alignement
    print("5. Initialisation du moteur d'alignement...")
    engine = AlignmentEngine(space)
    
    # 6. Calculer le score d'alignement initial
    print("6. Calcul du score d'alignement initial...")
    initial_score = engine.alignment_score(
        [savoir1, savoir2, savoir3],
        initial_pouvoir,
        [target1, target2, target3]
    )
    print(f"Score d'alignement initial: {initial_score:.4f}")
    
    # 7. Optimiser l'alignement
    print("7. Optimisation de l'alignement...")
    optimized_pouvoir = engine.optimize_alignment(
        [savoir1, savoir2, savoir3],
        [target1, target2, target3],
        initial_pouvoir=initial_pouvoir,
        max_iter=100
    )
    
    # 8. Calculer le score d'alignement optimisé
    print("8. Calcul du score d'alignement optimisé...")
    optimized_score = engine.alignment_score(
        [savoir1, savoir2, savoir3],
        optimized_pouvoir,
        [target1, target2, target3]
    )
    print(f"Score d'alignement optimisé: {optimized_score:.4f}")
    
    # 9. Visualiser les résultats
    print("9. Visualisation des résultats...")
    visualize_alignment(
        space,
        [savoir1, savoir2, savoir3],
        [target1, target2, target3],
        initial_pouvoir,
        optimized_pouvoir
    )
    
    # 10. Afficher la matrice de pouvoir optimisée
    print("\n10. Matrice de pouvoir optimisée:")
    print(optimized_pouvoir.matrix)
    
    # 11. Calculer les scores de cohérence et de justice
    print("\n11. Scores supplémentaires:")
    coherence = engine.coherence_score([savoir1, savoir2, savoir3], optimized_pouvoir)
    print(f"Score de cohérence: {coherence:.4f}")
    
    justice = engine.justice_score([optimized_pouvoir])
    print(f"Score de justice: {justice:.4f}")


def visualize_alignment(
    space: SavoirSpace,
    savoir_vectors: list,
    target_vectors: list,
    initial_pouvoir: PouvoirMatrix,
    optimized_pouvoir: PouvoirMatrix
):
    """Visualiser l'alignement avant et après optimisation."""
    
    # Extraire les vecteurs
    s_vectors = [s.vector for s in savoir_vectors]
    t_vectors = [t.vector for t in target_vectors]
    
    # Appliquer les transformations
    initial_transformed = [initial_pouvoir.apply(s.vector) for s in savoir_vectors]
    optimized_transformed = [optimized_pouvoir.apply(s.vector) for s in savoir_vectors]
    
    # Créer la figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Graphique 1: Avant optimisation
    ax1.set_title("Avant Optimisation")
    ax1.set_xlim([-2, 2])
    ax1.set_ylim([-2, 2])
    ax1.set_xlabel("Dimension 1")
    ax1.set_ylabel("Dimension 2")
    ax1.grid(True, alpha=0.3)
    
    # Vecteurs de savoir originaux
    for i, (s, label) in enumerate(zip(s_vectors, [s.label for s in savoir_vectors])):
        ax1.arrow(0, 0, s[0], s[1], head_width=0.05, head_length=0.1,
                  fc=f'C{i}', ec=f'C{i}', linestyle='-', alpha=0.7,
                  label=f'Savoir {label}')
    
    # Vecteurs cibles
    for i, (t, label) in enumerate(zip(t_vectors, [t.label for t in target_vectors])):
        ax1.scatter(t[0], t[1], c=f'C{i}', marker='x', s=100, label=f'Cible {label}')
    
    # Vecteurs transformés
    for i, (t, label) in enumerate(zip(initial_transformed, [s.label for s in savoir_vectors])):
        ax1.arrow(0, 0, t[0], t[1], head_width=0.05, head_length=0.1,
                  fc=f'C{i}', ec=f'C{i}', linestyle='--', alpha=0.7,
                  label=f'Transformé {label} (initial)')
    
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Graphique 2: Après optimisation
    ax2.set_title("Après Optimisation")
    ax2.set_xlim([-2, 2])
    ax2.set_ylim([-2, 2])
    ax2.set_xlabel("Dimension 1")
    ax2.set_ylabel("Dimension 2")
    ax2.grid(True, alpha=0.3)
    
    # Vecteurs de savoir originaux
    for i, (s, label) in enumerate(zip(s_vectors, [s.label for s in savoir_vectors])):
        ax2.arrow(0, 0, s[0], s[1], head_width=0.05, head_length=0.1,
                  fc=f'C{i}', ec=f'C{i}', linestyle='-', alpha=0.7,
                  label=f'Savoir {label}')
    
    # Vecteurs cibles
    for i, (t, label) in enumerate(zip(t_vectors, [t.label for t in target_vectors])):
        ax2.scatter(t[0], t[1], c=f'C{i}', marker='x', s=100, label=f'Cible {label}')
    
    # Vecteurs transformés optimisés
    for i, (t, label) in enumerate(zip(optimized_transformed, [s.label for s in savoir_vectors])):
        ax2.arrow(0, 0, t[0], t[1], head_width=0.05, head_length=0.1,
                  fc=f'C{i}', ec=f'C{i}', linestyle='--', alpha=0.7,
                  label=f'Transformé {label} (optimisé)')
    
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('alignment_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
```

## 📊 Résultats Attendus

En exécutant ce code, vous devriez voir :

1. **Sortie console** : Les scores d'alignement avant et après optimisation
2. **Visualisation** : Deux graphiques montrant :
   - À gauche : L'alignement initial (avec la matrice identité)
   - À droite : L'alignement optimisé

## 🔍 Interprétation

- **Score d'alignement** : Plus le score est bas, meilleur est l'alignement
- **Matrice de pouvoir optimisée** : La matrice qui transforme au mieux les vecteurs de savoir vers les cibles
- **Cohérence** : Mesure de la préservation des relations entre les vecteurs après transformation
- **Justice** : Mesure de l'équité dans la distribution du pouvoir

## 🎨 Visualisation

La visualisation montre :
- **Vecteurs de savoir** : Flèches pleines partant de l'origine
- **Vecteurs cibles** : Croix colorées
- **Vecteurs transformés** : Flèches en pointillés

Avant optimisation, les vecteurs transformés (pointillés) sont éloignés des cibles. Après optimisation, ils devraient être beaucoup plus proches.

## 🔧 Personnalisation

### Changer les Vecteurs

Modifiez les vecteurs de savoir et les cibles pour explorer différents scénarios :

```python
# Exemple avec des vecteurs différents
savoir1 = space.create_vector([2.0, 1.0], label="Connaissance Technique")
savoir2 = space.create_vector([1.0, 2.0], label="Connaissance Sociale")
savoir3 = space.create_vector([-1.0, 1.0], label="Connaissance Critique")

target1 = space.create_vector([0.0, 3.0], label="Objectif Technique")
target2 = space.create_vector([3.0, 0.0], label="Objectif Social")
target3 = space.create_vector([0.0, -2.0], label="Objectif Critique")
```

### Ajouter des Contraintes

Vous pouvez ajouter des contraintes à l'optimisation :

```python
# Contrainte : la matrice doit être inversible (déterminant > 0)
constraints = [{'type': 'determinant'}]

optimized_pouvoir = engine.optimize_alignment(
    [savoir1, savoir2, savoir3],
    [target1, target2, target3],
    initial_pouvoir=initial_pouvoir,
    constraints=constraints,
    max_iter=100
)
```

### Changer la Métrique de l'Espace

Vous pouvez définir une métrique personnalisée pour l'espace :

```python
# Créer une métrique personnalisée
custom_metric = np.array([
    [2.0, 0.5],
    [0.5, 2.0]
])

space = SavoirSpace(dimension=2, metric=custom_metric)
```

## 📚 Concepts Clés

### Espace du Savoir

L'espace du savoir est un espace vectoriel où chaque dimension représente un aspect du savoir. Dans cet exemple, nous utilisons un espace en 2D pour simplifier la visualisation.

### Matrice de Pouvoir

La matrice de pouvoir représente une transformation linéaire qui agit sur les vecteurs de savoir. Dans cet exemple, nous optimisons cette matrice pour aligner les vecteurs de savoir avec les cibles.

### Alignement

L'alignement est le processus d'optimisation de la matrice de pouvoir pour minimiser la distance entre les vecteurs de savoir transformés et les vecteurs cibles.

## 🚀 Prochaines Étapes

- [ ] Essayer avec des espaces de dimension supérieure
- [ ] Explorer différentes métriques d'espace
- [ ] Ajouter des contraintes supplémentaires
- [ ] Utiliser des matrices de pouvoir multiples
- [ ] Explorer les méthodes de quantisation
