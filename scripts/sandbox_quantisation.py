#!/usr/bin/env python3
"""
🔷 SANDBOX DE QUANTISATION CANONIQUE POUR MoE
   Implémentation des concepts théoriques du document RECETTE_QUANTISATION_MOE_CANONIQUE.md
   
   Auteur: Vibe Code (pour Mistral AI)
   Date: 2024
   
   Ce script implémente:
   1. La décomposition 3D canonique (x=pou, y=voir, z=réalisation)
   2. La projection sur tore concentrique
   3. La quantisation angulaire
   4. Les relations canoniques (Pouvoir-Lumière-Savoir)
   5. La triangulation de Thalès pour l'inférence sémantique
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import itertools
from typing import Tuple, Dict, List


# ============================================================================
# 📚 CONSTANTES CANONIQUES
# ============================================================================

# Rayons du tore
R_MAJEUR = 2.0  # Rayon majeur (distance au centre)
R_MINEUR = 1.0  # Rayon mineur (épaisseur de la couche)

# Tokens de référence et leurs positions canoniques
TOKENS_CANONIQUES = {
    "pouvoir": {"x": 1.0, "y": 0.0, "z": 0.0, "couche": 1, "description": "But pur, intention"},
    "lumière": {"x": 1.0, "y": 1.0, "z": 1.0, "couche": 1, "description": "Référentiel absolu de perception"},
    "voir": {"x": 0.0, "y": 1.0, "z": 0.0, "couche": 2, "description": "Perception pure"},
    "savoir": {"x": 0.8, "y": 0.9, "z": 0.7, "couche": 2, "description": "Connaissance déduite"},
    "comprendre": {"x": 0.7, "y": 0.8, "z": 0.9, "couche": 3, "description": "Assimilation"},
    "appliquer": {"x": 0.9, "y": 0.6, "z": 0.8, "couche": 3, "description": "Action"},
    "implémenter": {"x": 0.95, "y": 0.5, "z": 0.9, "couche": 4, "description": "Technique spécialisée"},
    "optimiser": {"x": 0.85, "y": 0.7, "z": 0.95, "couche": 4, "description": "Amélioration continue"},
    "volonté": {"x": 0.9, "y": 0.3, "z": 0.5, "couche": 1, "description": "Force intérieure"},
    "capacité": {"x": 0.85, "y": 0.4, "z": 0.6, "couche": 2, "description": "Synonyme de pouvoir"},
    "impuissance": {"x": 0.2, "y": 0.1, "z": 0.1, "couche": 2, "description": "Opposé de pouvoir"},
}

# Couleurs par couche
COULEURS_COUCHES = {
    1: "gold",      # Couche 1: Fondamentaux
    2: "lightblue", # Couche 2: Sous-concepts
    3: "lightgreen",# Couche 3: Actions
    4: "lightcoral",# Couche 4: Techniques
}


# ============================================================================
# 🧮 FONCTIONS MATHÉMATIQUES DE BASE
# ============================================================================

def normaliser_vecteur(v: np.ndarray) -> np.ndarray:
    """Normalise un vecteur 3D."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return np.array([0.0, 0.0, 0.0])
    return v / norm


def distance_euclidienne(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calcule la distance euclidienne entre deux vecteurs."""
    return np.linalg.norm(v1 - v2)


def angle_entre_vecteurs(v1: np.ndarray, v2: np.ndarray, en_degres: bool = True) -> float:
    """Calcule l'angle entre deux vecteurs (en degrés ou radians)."""
    dot = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    cos_theta = dot / (norm_v1 * norm_v2)
    # Clamp pour éviter les erreurs numériques
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    return np.degrees(angle_rad) if en_degres else angle_rad


# ============================================================================
# 🌐 PROJECTION SUR TORE CONCENTRIQUE
# ============================================================================

def coordonnées_sphériques(v: np.ndarray) -> Tuple[float, float, float]:
    """
    Convertit un vecteur cartésien en coordonnées sphériques.
    Retourne (r, theta, phi) où:
    - r: norme du vecteur
    - theta: angle dans le plan x-y (azimut)
    - phi: angle par rapport à z (colatitude)
    """
    x, y, z = v
    r = np.sqrt(x**2 + y**2 + z**2)
    if r == 0:
        return (0.0, 0.0, 0.0)
    theta = np.arctan2(y, x)  # [-π, π]
    phi = np.arccos(z / r)    # [0, π]
    return (r, theta, phi)


def projeter_sur_tore(v: np.ndarray, R: float = R_MAJEUR, r: float = R_MINEUR) -> np.ndarray:
    """
    Projette un vecteur 3D sur un tore concentrique.
    
    Équations paramétriques:
    X = (R + r * cos(phi)) * cos(theta)
    Y = (R + r * cos(phi)) * sin(theta)
    Z = r * sin(phi)
    """
    _, theta, phi = coordonnées_sphériques(v)
    X = (R + r * np.cos(phi)) * np.cos(theta)
    Y = (R + r * np.cos(phi)) * np.sin(theta)
    Z = r * np.sin(phi)
    return np.array([X, Y, Z])


def projeter_sur_tore_avec_couche(v: np.ndarray, couche: int) -> np.ndarray:
    """Projette un vecteur sur un tore avec un rayon majeur dépendant de la couche."""
    R = couche * 1.0  # R = 1.0, 2.0, 3.0, 4.0...
    return projeter_sur_tore(v, R=R, r=R_MINEUR)


# ============================================================================
# 🔢 QUANTISATION ANGULAIRE
# ============================================================================

def quantifier_angles(theta: float, phi: float, n_bits: int = 8) -> Tuple[int, int]:
    """
    Quantifie les angles theta et phi sur n_bits.
    
    - theta ∈ [-π, π] → [0, 2^N-1]
    - phi ∈ [0, π] → [0, 2^N-1]
    """
    # Normaliser theta à [0, 2π]
    theta_norm = theta % (2 * np.pi)
    if theta_norm < 0:
        theta_norm += 2 * np.pi
    
    # Quantification
    theta_quant = int(round(theta_norm / (2 * np.pi) * (2**n_bits))) % (2**n_bits)
    phi_quant = int(round(phi / np.pi * (2**n_bits))) % (2**n_bits)
    
    return (theta_quant, phi_quant)


def déquantifier_angles(theta_q: int, phi_q: int, n_bits: int = 8) -> Tuple[float, float]:
    """Déquantifie les angles."""
    theta = (theta_q / (2**n_bits)) * 2 * np.pi
    phi = (phi_q / (2**n_bits)) * np.pi
    return (theta, phi)


def quantifier_vecteur(v: np.ndarray, n_bits: int = 8) -> Tuple[int, int]:
    """Quantifie un vecteur via ses angles sphériques."""
    _, theta, phi = coordonnées_sphériques(v)
    return quantifier_angles(theta, phi, n_bits)


def déquantifier_vecteur(theta_q: int, phi_q: int, n_bits: int = 8) -> np.ndarray:
    """Reconstruit un vecteur à partir d'angles quantifiés."""
    theta, phi = déquantifier_angles(theta_q, phi_q, n_bits)
    # On suppose une norme de 1 pour simplifier
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return np.array([x, y, z])


# ============================================================================
# 🔺 TRIANGULATION DE THALÈS ET RELATIONS CANONIQUES
# ============================================================================

def angle_sémantique(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, en_degres: bool = True) -> float:
    """
    Calcule l'angle sémantique en v0 entre v1 et v2 (théorème de Thalès en 3D).
    
    Formule: cos(α) = (b² + c² - a²) / (2 * b * c)
    où:
    - a = distance(v1, v2)
    - b = distance(v0, v2)
    - c = distance(v0, v1)
    """
    a = distance_euclidienne(v1, v2)
    b = distance_euclidienne(v0, v2)
    c = distance_euclidienne(v0, v1)
    
    if b == 0 or c == 0:
        return 0.0
    
    cos_alpha = (b**2 + c**2 - a**2) / (2 * b * c)
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    angle_rad = np.arccos(cos_alpha)
    return np.degrees(angle_rad) if en_degres else angle_rad


def relation_canonique(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> Dict:
    """
    Analyse la relation canonique entre trois vecteurs (triangle sémantique).
    
    Retourne:
    - distances: distances entre chaque paire
    - angles: angles aux trois sommets
    - type_relation: classification de la relation
    """
    # Distances
    d01 = distance_euclidienne(v0, v1)
    d02 = distance_euclidienne(v0, v2)
    d12 = distance_euclidienne(v1, v2)
    
    # Angles
    a0 = angle_sémantique(v0, v1, v2)
    a1 = angle_sémantique(v1, v0, v2)
    a2 = angle_sémantique(v2, v0, v1)
    
    # Classification
    if d12 < 0.2:
        type_relation = "SYNONYMIE"
    elif a0 < 15:
        type_relation = "LIEN_FORT"
    elif a0 < 45:
        type_relation = "LIEN_MOYEN"
    elif a0 > 135:
        type_relation = "OPPOSITION"
    else:
        type_relation = "NEUTRE"
    
    return {
        "distances": {"v0-v1": d01, "v0-v2": d02, "v1-v2": d12},
        "angles": {"en_v0": a0, "en_v1": a1, "en_v2": a2},
        "type_relation": type_relation,
        "triangle_valide": (d01 + d02 > d12) and (d01 + d12 > d02) and (d02 + d12 > d01)
    }


# ============================================================================
# 🎯 FONCTIONS DE GRANULARISATION ET COUCHES
# ============================================================================

def obtenir_vecteur_token(token: str) -> np.ndarray:
    """Récupère le vecteur canonique d'un token."""
    if token in TOKENS_CANONIQUES:
        return np.array([
            TOKENS_CANONIQUES[token]["x"],
            TOKENS_CANONIQUES[token]["y"],
            TOKENS_CANONIQUES[token]["z"]
        ])
    else:
        # Générer un vecteur aléatoire pour les tokens inconnus
        return np.random.rand(3)


def décomposer_en_sous_concepts(token: str, n_sous_concepts: int = 3) -> List[Dict]:
    """
    Décompose un token en sous-concepts via triangulation de Thalès.
    
    Chaque sous-concept est positionné sur une couche supérieure.
    """
    v0 = obtenir_vecteur_token(token)
    couche = TOKENS_CANONIQUES.get(token, {}).get("couche", 1)
    nouvelle_couche = couche + 1
    
    sous_concepts = []
    for i in range(n_sous_concepts):
        # Générer un vecteur aléatoire dans le cône autour de v0
        angle = np.random.uniform(0, 30)  # 30° autour de l'axe
        phi = np.random.uniform(0, 30)
        
        # Rotation aléatoire
        theta_random = np.random.uniform(0, 2 * np.pi)
        
        # Nouveau vecteur (perturbation de v0)
        v_new = v0 + np.random.normal(0, 0.1, 3)
        v_new = normaliser_vecteur(v_new)
        
        # Projection sur la nouvelle couche
        v_projeté = projeter_sur_tore_avec_couche(v_new, nouvelle_couche)
        
        sous_concepts.append({
            "token": f"{token}_sous_{i+1}",
            "vecteur": v_new,
            "vecteur_projeté": v_projeté,
            "couche": nouvelle_couche,
            "description": f"Sous-concept {i+1} de {token}"
        })
    
    return sous_concepts


# ============================================================================
# 📊 VISUALISATION 3D
# ============================================================================

def tracer_tore_3d(ax: Axes3D, R: float = R_MAJEUR, r: float = R_MINEUR, couleur: str = "gray", alpha: float = 0.2):
    """Trace un tore 3D."""
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, 2 * np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)
    
    X = (R + r * np.cos(phi)) * np.cos(theta)
    Y = (R + r * np.cos(phi)) * np.sin(theta)
    Z = r * np.sin(phi)
    
    ax.plot_surface(X, Y, Z, color=couleur, alpha=alpha, rstride=5, cstride=5)


def tracer_vecteurs_3d(ax: Axes3D, vecteurs: Dict[str, np.ndarray], couleurs: Dict[int, str] = None):
    """Trace des vecteurs 3D avec leurs labels."""
    couleurs = couleurs or COULEURS_COUCHES
    
    for token, v in vecteurs.items():
        couche = TOKENS_CANONIQUES.get(token, {}).get("couche", 1)
        couleur = couleurs.get(couche, "black")
        
        # Tracer le vecteur
        ax.quiver(0, 0, 0, v[0], v[1], v[2], color=couleur, arrow_length_ratio=0.1)
        
        # Ajouter le label
        ax.text(v[0], v[1], v[2], token, color=couleur, fontsize=9)


def tracer_triangle_sémantique(ax: Axes3D, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, 
                              labels: Tuple[str, str, str] = ("A", "B", "C")):
    """Trace un triangle sémantique avec ses angles."""
    # Tracer les vecteurs
    for v, label in zip([v0, v1, v2], labels):
        ax.quiver(0, 0, 0, v[0], v[1], v[2], color="black", arrow_length_ratio=0.1)
        ax.text(v[0], v[1], v[2], label, fontsize=10)
    
    # Tracer les côtés du triangle
    for v_a, v_b in [(v0, v1), (v0, v2), (v1, v2)]:
        ax.plot([v_a[0], v_b[0]], [v_a[1], v_b[1]], [v_a[2], v_b[2]], color="red", linestyle="--", alpha=0.5)
    
    # Calculer et afficher les angles
    relation = relation_canonique(v0, v1, v2)
    ax.text(0, 0, 0, f"α={relation['angles']['en_v0']:.1f}°", fontsize=10, color="blue")
    ax.text(v1[0], v1[1], v1[2], f"β={relation['angles']['en_v1']:.1f}°", fontsize=10, color="blue")
    ax.text(v2[0], v2[1], v2[2], f"γ={relation['angles']['en_v2']:.1f}°", fontsize=10, color="blue")


def visualiser_espace_canonique():
    """Visualise l'espace canonique avec les tokens de référence."""
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Tracer les tores concentriques
    for couche in range(1, 5):
        tracer_tore_3d(ax, R=couche * 1.0, r=R_MINEUR, couleur=COULEURS_COUCHES[couche], alpha=0.1)
    
    # Tracer les vecteurs canoniques
    vecteurs = {token: obtenir_vecteur_token(token) for token in TOKENS_CANONIQUES}
    tracer_vecteurs_3d(ax, vecteurs)
    
    # Tracer le triangle Pouvoir-Lumière-Savoir
    v_pouvoir = obtenir_vecteur_token("pouvoir")
    v_lumière = obtenir_vecteur_token("lumière")
    v_savoir = obtenir_vecteur_token("savoir")
    tracer_triangle_sémantique(ax, v_pouvoir, v_lumière, v_savoir, 
                              labels=("pouvoir", "lumière", "savoir"))
    
    # Configurer la vue
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])
    ax.set_xlabel("X (Possible)")
    ax.set_ylabel("Y (Perception)")
    ax.set_zlabel("Z (Réalisation)")
    ax.set_title("Espace Canonique 3D avec Tore Concentrique\n(Pouvoir-Lumière-Savoir)", fontsize=14)
    
    plt.tight_layout()
    plt.savefig("/workspace/visualisation_espace_canonique.png", dpi=300)
    plt.show()


def visualiser_quantisation():
    """Visualise l'effet de la quantisation angulaire."""
    # Tokens à quantifier
    tokens = ["pouvoir", "lumière", "savoir", "comprendre"]
    vecteurs_originaux = {token: obtenir_vecteur_token(token) for token in tokens}
    
    # Quantification et déquantification
    n_bits_list = [4, 8, 12]
    
    fig, axes = plt.subplots(1, len(n_bits_list), figsize=(15, 5), subplot_kw={'projection': '3d'})
    
    for i, n_bits in enumerate(n_bits_list):
        ax = axes[i]
        
        # Quantifier et déquantifier
        vecteurs_quantifiés = {}
        for token, v in vecteurs_originaux.items():
            theta_q, phi_q = quantifier_vecteur(v, n_bits)
            v_reconstruit = déquantifier_vecteur(theta_q, phi_q, n_bits)
            vecteurs_quantifiés[token] = v_reconstruit
        
        # Tracer les vecteurs originaux et quantifiés
        for token, v_orig in vecteurs_originaux.items():
            v_quant = vecteurs_quantifiés[token]
            
            # Original
            ax.quiver(0, 0, 0, v_orig[0], v_orig[1], v_orig[2], 
                     color="blue", arrow_length_ratio=0.1, label=f"{token} (orig)" if i == 0 else "")
            ax.text(v_orig[0], v_orig[1], v_orig[2], token, color="blue", fontsize=8)
            
            # Quantifié
            ax.quiver(0, 0, 0, v_quant[0], v_quant[1], v_quant[2], 
                     color="red", arrow_length_ratio=0.1, linestyle="--", 
                     label=f"{token} (quant)" if i == 0 else "")
        
        # Calculer l'erreur moyenne
        erreur_moyenne = 0
        for token in tokens:
            erreur = distance_euclidienne(vecteurs_originaux[token], vecteurs_quantifiés[token])
            erreur_moyenne += erreur
        erreur_moyenne /= len(tokens)
        
        ax.set_title(f"Quantisation ({n_bits} bits)\nErreur moyenne: {erreur_moyenne:.4f}")
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        
        if i == 0:
            ax.legend(fontsize=6)
    
    plt.tight_layout()
    plt.savefig("/workspace/visualisation_quantisation.png", dpi=300)
    plt.show()


# ============================================================================
# 🧪 TESTS ET VALIDATION
# ============================================================================

def tester_relations_canoniques():
    """Teste les relations canoniques entre les tokens de référence."""
    print("=" * 80)
    print("🔍 TEST DES RELATIONS CANONIQUES")
    print("=" * 80)
    
    # Triangle Pouvoir-Lumière-Savoir
    v_pouvoir = obtenir_vecteur_token("pouvoir")
    v_lumière = obtenir_vecteur_token("lumière")
    v_savoir = obtenir_vecteur_token("savoir")
    
    print("\n📐 Triangle Pouvoir-Lumière-Savoir:")
    relation = relation_canonique(v_pouvoir, v_lumière, v_savoir)
    print(f"  Distances:")
    print(f"    Pouvoir-Lumière: {relation['distances']['v0-v1']:.4f}")
    print(f"    Pouvoir-Savoir:  {relation['distances']['v0-v2']:.4f}")
    print(f"    Lumière-Savoir:  {relation['distances']['v1-v2']:.4f}")
    print(f"  Angles:")
    print(f"    En Pouvoir:  {relation['angles']['en_v0']:.2f}°")
    print(f"    En Lumière:  {relation['angles']['en_v1']:.2f}°")
    print(f"    En Savoir:   {relation['angles']['en_v2']:.2f}°")
    print(f"  Type de relation: {relation['type_relation']}")
    
    # Test synonymie: Pouvoir vs Capacité
    print("\n🔗 Test Synonymie (Pouvoir vs Capacité):")
    v_capacité = obtenir_vecteur_token("capacité")
    relation = relation_canonique(v_pouvoir, v_lumière, v_capacité)
    print(f"  Distance Pouvoir-Capacité: {distance_euclidienne(v_pouvoir, v_capacité):.4f}")
    print(f"  Angle en Lumière: {relation['angles']['en_v1']:.2f}°")
    print(f"  Type de relation: {relation['type_relation']}")
    
    # Test opposition: Pouvoir vs Impuissance
    print("\n⚔️ Test Opposition (Pouvoir vs Impuissance):")
    v_impuissance = obtenir_vecteur_token("impuissance")
    relation = relation_canonique(v_pouvoir, v_lumière, v_impuissance)
    print(f"  Distance Pouvoir-Impuissance: {distance_euclidienne(v_pouvoir, v_impuissance):.4f}")
    print(f"  Angle en Lumière: {relation['angles']['en_v1']:.2f}°")
    print(f"  Type de relation: {relation['type_relation']}")
    
    print("\n" + "=" * 80)


def tester_quantisation():
    """Teste la quantisation angulaire."""
    print("=" * 80)
    print("🔢 TEST DE QUANTISATION ANGULAIRE")
    print("=" * 80)
    
    tokens = ["pouvoir", "lumière", "savoir", "comprendre", "appliquer"]
    n_bits_list = [4, 8, 12, 16]
    
    for n_bits in n_bits_list:
        print(f"\n📊 Quantisation avec {n_bits} bits:")
        erreur_totale = 0
        
        for token in tokens:
            v_orig = obtenir_vecteur_token(token)
            theta_q, phi_q = quantifier_vecteur(v_orig, n_bits)
            v_reconstruit = déquantifier_vecteur(theta_q, phi_q, n_bits)
            
            erreur = distance_euclidienne(v_orig, v_reconstruit)
            erreur_totale += erreur
            
            print(f"  {token:12s}: erreur = {erreur:.6f}")
        
        erreur_moyenne = erreur_totale / len(tokens)
        print(f"  → Erreur moyenne: {erreur_moyenne:.6f}")
        print(f"  → Taille par vecteur: {2 * n_bits // 8} octets (vs 24 octets en FP32)")
        print(f"  → Ratio de compression: {24 / (2 * n_bits // 8):.1f}×")
    
    print("\n" + "=" * 80)


def tester_projection_tore():
    """Teste la projection sur tore concentrique."""
    print("=" * 80)
    print("🌐 TEST DE PROJECTION SUR TORE CONCENTRIQUE")
    print("=" * 80)
    
    tokens = ["pouvoir", "lumière", "savoir"]
    
    for token in tokens:
        v_orig = obtenir_vecteur_token(token)
        couche = TOKENS_CANONIQUES[token]["couche"]
        v_projeté = projeter_sur_tore_avec_couche(v_orig, couche)
        
        print(f"\n{token} (couche {couche}):")
        print(f"  Original:     ({v_orig[0]:.4f}, {v_orig[1]:.4f}, {v_orig[2]:.4f})")
        print(f"  Projeté:      ({v_projeté[0]:.4f}, {v_projeté[1]:.4f}, {v_projeté[2]:.4f})")
        print(f"  Distance:     {distance_euclidienne(v_orig, v_projeté):.4f}")
    
    print("\n" + "=" * 80)


def tester_granularisation():
    """Teste la décomposition en sous-concepts."""
    print("=" * 80)
    print("🎯 TEST DE GRANULARISATION (DÉCOMPOSITION EN SOUS-CONCEPTS)")
    print("=" * 80)
    
    token = "pouvoir"
    sous_concepts = décomposer_en_sous_concepts(token, n_sous_concepts=3)
    
    print(f"\nDécomposition de '{token}':")
    for i, sc in enumerate(sous_concepts):
        print(f"\n  Sous-concept {i+1}:")
        print(f"    Token: {sc['token']}")
        print(f"    Couche: {sc['couche']}")
        print(f"    Vecteur: ({sc['vecteur'][0]:.4f}, {sc['vecteur'][1]:.4f}, {sc['vecteur'][2]:.4f})")
        print(f"    Projeté: ({sc['vecteur_projeté'][0]:.4f}, {sc['vecteur_projeté'][1]:.4f}, {sc['vecteur_projeté'][2]:.4f})")
        print(f"    Distance à l'original: {distance_euclidienne(obtenir_vecteur_token(token), sc['vecteur']):.4f}")
    
    print("\n" + "=" * 80)


# ============================================================================
# 🏃 MAIN
# ============================================================================

def main():
    """Exécute tous les tests et visualisations."""
    print("\n" + "=" * 80)
    print("🚀 SANDBOX DE QUANTISATION CANONIQUE POUR MoE")
    print("   Implémentation des concepts théoriques pour Mistral AI")
    print("=" * 80 + "\n")
    
    # Tests
    tester_relations_canoniques()
    tester_quantisation()
    tester_projection_tore()
    tester_granularisation()
    
    # Visualisations
    print("\n📊 Génération des visualisations...")
    visualiser_espace_canonique()
    visualiser_quantisation()
    
    print("\n✅ Tous les tests et visualisations ont été exécutés avec succès!")
    print("   Les images ont été sauvegardées dans /workspace/")


if __name__ == "__main__":
    main()
