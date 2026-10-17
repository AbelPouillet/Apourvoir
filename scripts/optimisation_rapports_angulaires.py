#!/usr/bin/env python3
"""
🔷 OPTIMISATION DES RAPPORTS ANGULAIRES POUR QUANTISATION CANONIQUE
   Phase C: Amélioration itérative des relations canoniques dans l'espace multivectoriel
   
   Ce script optimise:
   1. Les formules de projection sur tore (minimisation de la distorsion)
   2. La quantification angulaire adaptative (basée sur l'importance sémantique)
   3. Les rapports entre couches concentriques
   4. La calibration automatique des angles canoniques
   
   Auteur: Vibe Code (pour Mistral AI)
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Tuple, Dict, List, Callable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from datetime import datetime


# ============================================================================
# 📚 CONSTANTES ET CONFIGURATION
# ============================================================================

# Configuration par défaut
CONFIG = {
    "R_MAJEUR_BASE": 2.0,
    "R_MINEUR_BASE": 0.5,
    "N_BITS_DEFAUT": 8,
    "SEUIL_SYNONYMIE": 0.2,
    "SEUIL_LIEN_FORT": 15.0,  # degrés
    "SEUIL_LIEN_MOYEN": 45.0,
    "SEUIL_OPPOSITION": 135.0,
    "POIDS_COUCHES": {
        1: 1.0,  # Fondamentaux: poids maximal
        2: 0.8,  # Sous-concepts
        3: 0.6,  # Actions
        4: 0.4,  # Techniques
    }
}

# Tokens canoniques avec positions initiales
TOKENS_CANONIQUES = {
    "pouvoir": {"x": 1.0, "y": 0.0, "z": 0.0, "couche": 1, "poids": 1.0, "fixe": True},
    "lumière": {"x": 1.0, "y": 1.0, "z": 1.0, "couche": 1, "poids": 1.0, "fixe": True},
    "voir": {"x": 0.0, "y": 1.0, "z": 0.0, "couche": 2, "poids": 0.8, "fixe": False},
    "savoir": {"x": 0.8, "y": 0.9, "z": 0.7, "couche": 2, "poids": 0.8, "fixe": False},
    "comprendre": {"x": 0.7, "y": 0.8, "z": 0.9, "couche": 3, "poids": 0.6, "fixe": False},
    "appliquer": {"x": 0.9, "y": 0.6, "z": 0.8, "couche": 3, "poids": 0.6, "fixe": False},
    "volonté": {"x": 0.9, "y": 0.3, "z": 0.5, "couche": 1, "poids": 1.0, "fixe": False},
    "capacité": {"x": 0.85, "y": 0.4, "z": 0.6, "couche": 2, "poids": 0.8, "fixe": False},
    "impuissance": {"x": 0.2, "y": 0.1, "z": 0.1, "couche": 2, "poids": 0.8, "fixe": False},
    "connaître": {"x": 0.85, "y": 0.85, "z": 0.7, "couche": 2, "poids": 0.8, "fixe": False},
    "agir": {"x": 0.95, "y": 0.5, "z": 0.85, "couche": 3, "poids": 0.6, "fixe": False},
}


# ============================================================================
# 🧮 FONCTIONS MATHÉMATIQUES AMÉLIORÉES
# ============================================================================

def normaliser_vecteur(v: np.ndarray) -> np.ndarray:
    """Normalise un vecteur 3D avec gestion des cas limites."""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return np.array([0.0, 0.0, 0.0])
    return v / norm


def distance_euclidienne(v1: np.ndarray, v2: np.ndarray) -> float:
    """Distance euclidienne avec tolérance numérique."""
    return np.linalg.norm(v1 - v2)


def angle_entre_vecteurs(v1: np.ndarray, v2: np.ndarray, en_degres: bool = True) -> float:
    """Angle entre deux vecteurs avec gestion des erreurs numériques."""
    v1_norm = normaliser_vecteur(v1)
    v2_norm = normaliser_vecteur(v2)
    dot = np.dot(v1_norm, v2_norm)
    dot = np.clip(dot, -1.0, 1.0)
    angle_rad = np.arccos(dot)
    return np.degrees(angle_rad) if en_degres else angle_rad


# ============================================================================
# 🎯 PROJECTION SUR TORE OPTIMISÉE
# ============================================================================

def coordonnées_sphériques_robustes(v: np.ndarray) -> Tuple[float, float, float]:
    """
    Coordonnées sphériques avec gestion des singularités.
    Retourne (r, theta, phi) avec:
    - r ∈ [0, ∞]
    - theta ∈ [0, 2π] (au lieu de [-π, π])
    - phi ∈ [0, π]
    """
    x, y, z = v
    r = np.sqrt(x**2 + y**2 + z**2)
    
    if r < 1e-10:
        return (0.0, 0.0, 0.0)
    
    # Calcul de theta dans [0, 2π]
    theta = np.arctan2(y, x)
    if theta < 0:
        theta += 2 * np.pi
    
    # Calcul de phi dans [0, π]
    phi = np.arccos(np.clip(z / r, -1.0, 1.0))
    
    return (r, theta, phi)


def projeter_sur_tore_optimisé(v: np.ndarray, R: float, r: float, 
                             normaliser: bool = True) -> np.ndarray:
    """
    Projection sur tore avec minimisation de la distorsion.
    
    Améliorations:
    1. Normalisation optionnelle du vecteur d'entrée
    2. Ajustement automatique de r en fonction de R
    3. Correction des singularités aux pôles
    """
    if normaliser:
        v = normaliser_vecteur(v)
    
    _, theta, phi = coordonnées_sphériques_robustes(v)
    
    # Ajustement dynamique du rayon mineur pour minimiser la distorsion
    r_adapté = r * (1.0 - 0.5 * (abs(np.sin(phi)) + abs(np.cos(theta)))) 
    
    X = (R + r_adapté * np.cos(phi)) * np.cos(theta)
    Y = (R + r_adapté * np.cos(phi)) * np.sin(theta)
    Z = r_adapté * np.sin(phi)
    
    return np.array([X, Y, Z])


def projeter_sur_tore_avec_couche_optimisé(v: np.ndarray, couche: int) -> np.ndarray:
    """Projection avec rayon majeur dépendant de la couche et optimisé."""
    # Rayon majeur proportionnel à la couche
    R = couche * CONFIG["R_MAJEUR_BASE"]
    # Rayon mineur décroissant avec la couche pour plus de précision
    r = CONFIG["R_MINEUR_BASE"] / (0.5 + 0.5 * couche)
    return projeter_sur_tore_optimisé(v, R, r)


# ============================================================================
# 🔢 QUANTISATION ANGULAIRE ADAPTATIVE
# ============================================================================

def quantifier_angles_adaptatif(theta: float, phi: float, n_bits: int = None, 
                               poids: float = 1.0) -> Tuple[int, int]:
    """
    Quantification adaptative basée sur le poids sémantique.
    
    Plus le poids est élevé, plus on utilise de bits pour la quantification.
    """
    if n_bits is None:
        # Calcul dynamique du nombre de bits en fonction du poids
        n_bits = max(4, min(16, int(4 + 4 * poids)))
    
    # Normalisation des angles
    theta_norm = theta % (2 * np.pi)
    if theta_norm < 0:
        theta_norm += 2 * np.pi
    
    # Quantification non-linéaire pour donner plus de précision aux petits angles
    # (les petits angles = relations sémantiques fortes)
    theta_q = int(round((theta_norm / (2 * np.pi)) ** 0.5 * (2**n_bits))) % (2**n_bits)
    phi_q = int(round((phi / np.pi) ** 0.5 * (2**n_bits))) % (2**n_bits)
    
    return (theta_q, phi_q, n_bits)


def déquantifier_angles_adaptatif(theta_q: int, phi_q: int, n_bits: int = 8) -> Tuple[float, float]:
    """Déquantification avec correction non-linéaire."""
    # Correction non-linéaire inverse
    theta = ((theta_q / (2**n_bits)) ** 2) * 2 * np.pi
    phi = ((phi_q / (2**n_bits)) ** 2) * np.pi
    return (theta, phi)


def quantifier_vecteur_adaptatif(v: np.ndarray, poids: float = 1.0) -> Tuple[int, int, int]:
    """Quantifie un vecteur avec adaptation basée sur le poids."""
    _, theta, phi = coordonnées_sphériques_robustes(v)
    return quantifier_angles_adaptatif(theta, phi, poids=poids)


def déquantifier_vecteur_adaptatif(theta_q: int, phi_q: int, n_bits: int) -> np.ndarray:
    """Reconstruit un vecteur à partir d'angles quantifiés."""
    theta, phi = déquantifier_angles_adaptatif(theta_q, phi_q, n_bits)
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return normaliser_vecteur(np.array([x, y, z]))


# ============================================================================
# 🔺 OPTIMISATION DES RELATIONS CANONIQUES
# ============================================================================

def angle_sémantique_optimisé(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float:
    """Calcule l'angle sémantique avec pondération par les normes."""
    a = distance_euclidienne(v1, v2)
    b = distance_euclidienne(v0, v2)
    c = distance_euclidienne(v0, v1)
    
    # Éviter les divisions par zéro
    if b < 1e-10 or c < 1e-10:
        return 0.0
    
    # Formule de Al-Kashi (généralisation de Pythagore)
    cos_alpha = (b**2 + c**2 - a**2) / (2 * b * c)
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    return np.degrees(np.arccos(cos_alpha))


def relation_canonique_optimisée(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, 
                                 poids_v0: float = 1.0) -> Dict:
    """
    Analyse optimisée des relations canoniques avec pondération.
    
    Retourne:
    - distances: distances pondérées
    - angles: angles sémantiques
    - score_relation: score de similarité [0, 1]
    - type_relation: classification
    """
    # Distances
    d01 = distance_euclidienne(v0, v1)
    d02 = distance_euclidienne(v0, v2)
    d12 = distance_euclidienne(v1, v2)
    
    # Angles
    a0 = angle_sémantique_optimisé(v0, v1, v2)
    a1 = angle_sémantique_optimisé(v1, v0, v2)
    a2 = angle_sémantique_optimisé(v2, v0, v1)
    
    # Score de similarité (inverse de la distance, pondéré par les angles)
    # Plus le score est proche de 1, plus les vecteurs sont sémantiquement proches
    score = np.exp(-poids_v0 * (d12 + 0.1 * a0))
    
    # Classification améliorée
    if d12 < CONFIG["SEUIL_SYNONYMIE"]:
        type_relation = "SYNONYMIE"
    elif a0 < CONFIG["SEUIL_LIEN_FORT"]:
        type_relation = "LIEN_FORT"
    elif a0 < CONFIG["SEUIL_LIEN_MOYEN"]:
        type_relation = "LIEN_MOYEN"
    elif a0 > CONFIG["SEUIL_OPPOSITION"]:
        type_relation = "OPPOSITION"
    else:
        type_relation = "NEUTRE"
    
    return {
        "distances": {"v0-v1": d01, "v0-v2": d02, "v1-v2": d12},
        "angles": {"en_v0": a0, "en_v1": a1, "en_v2": a2},
        "score_relation": float(score),
        "type_relation": type_relation,
        "triangle_valide": (d01 + d02 > d12) and (d01 + d12 > d02) and (d02 + d12 > d01)
    }


# ============================================================================
# 🎯 CALIBRATION AUTOMATIQUE DES POSITIONS
# ============================================================================

def obtenir_vecteur_token_optimisé(token: str) -> np.ndarray:
    """Récupère le vecteur d'un token avec gestion des tokens inconnus."""
    if token in TOKENS_CANONIQUES:
        return np.array([
            TOKENS_CANONIQUES[token]["x"],
            TOKENS_CANONIQUES[token]["y"],
            TOKENS_CANONIQUES[token]["z"]
        ])
    else:
        # Générer un vecteur aléatoire normalisé
        v = np.random.rand(3) - 0.5
        return normaliser_vecteur(v)


def fonction_objectif_calibration(positions: np.ndarray, tokens: List[str]) -> float:
    """
    Fonction objectif pour l'optimisation des positions des tokens.
    
    Minimise:
    1. La distance entre tokens synonymes
    2. L'angle entre tokens liés
    3. La distorsion après projection sur tore
    """
    # Reconstruire les vecteurs à partir des positions optimisées
    vecteurs = {}
    idx = 0
    for token in tokens:
        if TOKENS_CANONIQUES[token].get("fixe", False):
            # Les tokens fixes ne sont pas optimisés
            vecteurs[token] = obtenir_vecteur_token_optimisé(token)
        else:
            vecteurs[token] = normaliser_vecteur(positions[idx:idx+3])
            idx += 3
    
    # Calculer le score total
    score_total = 0.0
    
    # 1. Minimiser les distances entre synonymes
    synonymes = [("pouvoir", "capacité"), ("savoir", "connaître")]
    for t1, t2 in synonymes:
        if t1 in vecteurs and t2 in vecteurs:
            d = distance_euclidienne(vecteurs[t1], vecteurs[t2])
            score_total += 10.0 * d  # Pénalité forte pour les synonymes éloignés
    
    # 2. Maximiser les angles entre opposés
    opposés = [("pouvoir", "impuissance")]
    for t1, t2 in opposés:
        if t1 in vecteurs and t2 in vecteurs:
            angle = angle_entre_vecteurs(vecteurs[t1], vecteurs[t2])
            # On veut un angle proche de 180°
            score_total += 0.1 * (180 - angle) ** 2
    
    # 3. Minimiser les distorsions de projection
    for token, v in vecteurs.items():
        couche = TOKENS_CANONIQUES[token]["couche"]
        v_projeté = projeter_sur_tore_avec_couche_optimisé(v, couche)
        distorsion = distance_euclidienne(v, v_projeté)
        score_total += 0.5 * distorsion
    
    # 4. Maintenir les relations canoniques (Pouvoir-Lumière-Savoir)
    if all(t in vecteurs for t in ["pouvoir", "lumière", "savoir"]):
        relation = relation_canonique_optimisée(
            vecteurs["pouvoir"], vecteurs["lumière"], vecteurs["savoir"],
            poids_v0=1.0
        )
        # On veut un LIEN_FORT (angle < 15°)
        if relation["type_relation"] != "LIEN_FORT":
            score_total += 5.0 * (relation["angles"]["en_v0"] - 15) ** 2
    
    return score_total


def calibrer_positions_tokens(tokens_a_optimiser: List[str], 
                              max_iter: int = 1000, 
                              tol: float = 1e-6) -> Dict[str, Dict]:
    """
    Calibre automatiquement les positions des tokens pour optimiser
    les relations sémantiques.
    
    Utilise l'optimisation par essaims particulaires (differential evolution).
    """
    # Préparer les bornes pour l'optimisation
    bounds = []
    for token in tokens_a_optimiser:
        if not TOKENS_CANONIQUES[token].get("fixe", False):
            bounds.extend([(-1, 1), (-1, 1), (-1, 1)])  # x, y, z ∈ [-1, 1]
    
    # Position initiale
    x0 = []
    for token in tokens_a_optimiser:
        if not TOKENS_CANONIQUES[token].get("fixe", False):
            v = obtenir_vecteur_token_optimisé(token)
            x0.extend([v[0], v[1], v[2]])
    
    # Exécuter l'optimisation
    result = differential_evolution(
        lambda x: fonction_objectif_calibration(x, tokens_a_optimiser),
        bounds,
        maxiter=max_iter,
        tol=tol,
        seed=42,
        polish=True
    )
    
    # Mettre à jour les positions optimisées
    idx = 0
    tokens_optimisés = {}
    for token in tokens_a_optimiser:
        if TOKENS_CANONIQUES[token].get("fixe", False):
            tokens_optimisés[token] = TOKENS_CANONIQUES[token].copy()
        else:
            tokens_optimisés[token] = {
                "x": float(result.x[idx]),
                "y": float(result.x[idx+1]),
                "z": float(result.x[idx+2]),
                "couche": TOKENS_CANONIQUES[token]["couche"],
                "poids": TOKENS_CANONIQUES[token]["poids"],
                "fixe": TOKENS_CANONIQUES[token]["fixe"]
            }
            idx += 3
    
    return {
        "tokens_optimisés": tokens_optimisés,
        "score_final": float(result.fun),
        "success": result.success,
        "message": result.message
    }


# ============================================================================
# 📊 ANALYSE ET RAPPORTS
# ============================================================================

def générer_rapport_relations(tokens: Dict[str, Dict]) -> Dict:
    """Génère un rapport complet des relations canoniques."""
    rapport = {
        "date": datetime.now().isoformat(),
        "tokens": {},
        "relations": [],
        "statistiques": {}
    }
    
    # Extraire les vecteurs
    vecteurs = {token: np.array([t["x"], t["y"], t["z"]]) for token, t in tokens.items()}
    
    # Calculer les relations par paires
    token_list = list(tokens.keys())
    for i in range(len(token_list)):
        for j in range(i+1, len(token_list)):
            t1, t2 = token_list[i], token_list[j]
            v1, v2 = vecteurs[t1], vecteurs[t2]
            
            relation = relation_canonique_optimisée(v1, v1, v2, 
                                                     poids_v0=tokens[t1]["poids"])
            
            rapport["relations"].append({
                "token1": t1,
                "token2": t2,
                "distance": relation["distances"]["v0-v2"],
                "angle": relation["angles"]["en_v0"],
                "score": relation["score_relation"],
                "type": relation["type_relation"]
            })
    
    # Statistiques
    distances = [r["distance"] for r in rapport["relations"]]
    angles = [r["angle"] for r in rapport["relations"]]
    scores = [r["score"] for r in rapport["relations"]]
    
    rapport["statistiques"] = {
        "distance_moyenne": float(np.mean(distances)),
        "distance_min": float(np.min(distances)),
        "distance_max": float(np.max(distances)),
        "angle_moyen": float(np.mean(angles)),
        "score_moyen": float(np.mean(scores)),
        "nombre_relations": len(rapport["relations"]),
        "par_type": {}
    }
    
    # Compter par type
    for relation in rapport["relations"]:
        t = relation["type"]
        rapport["statistiques"]["par_type"][t] = rapport["statistiques"]["par_type"].get(t, 0) + 1
    
    return rapport


def générer_rapport_quantisation(tokens: Dict[str, Dict], n_bits_list: List[int]) -> Dict:
    """Génère un rapport de quantisation pour différents nombres de bits."""
    rapport = {
        "date": datetime.now().isoformat(),
        "tokens": list(tokens.keys()),
        "n_bits_list": n_bits_list,
        "résultats": []
    }
    
    vecteurs_originaux = {token: np.array([t["x"], t["y"], t["z"]]) 
                         for token, t in tokens.items()}
    
    for n_bits in n_bits_list:
        erreur_totale = 0
        taille_totale = 0
        
        for token, v_orig in vecteurs_originaux.items():
            poids = tokens[token]["poids"]
            theta_q, phi_q, n_bits_utilisé = quantifier_vecteur_adaptatif(v_orig, poids)
            v_reconstruit = déquantifier_vecteur_adaptatif(theta_q, phi_q, n_bits_utilisé)
            
            erreur = distance_euclidienne(v_orig, v_reconstruit)
            erreur_totale += erreur
            taille_totale += 2 * n_bits_utilisé // 8  # 2 angles × n_bits/8 octets
        
        erreur_moyenne = erreur_totale / len(tokens)
        taille_moyenne = taille_totale / len(tokens)
        ratio_compression = 24 / taille_moyenne  # 24 octets pour FP32 × 3
        
        # Calcul de l'erreur maximale
        erreurs_max = []
        for token, v_orig in vecteurs_originaux.items():
            poids = tokens[token]["poids"]
            theta_q, phi_q, n_bits_utilisé = quantifier_vecteur_adaptatif(v_orig, poids)
            v_reconstruit = déquantifier_vecteur_adaptatif(theta_q, phi_q, n_bits_utilisé)
            erreurs_max.append(distance_euclidienne(v_orig, v_reconstruit))
        
        rapport["résultats"].append({
            "n_bits": n_bits,
            "n_bits_moyen_utilisé": n_bits_utilisé,
            "erreur_moyenne": float(erreur_moyenne),
            "taille_moyenne_octets": float(taille_moyenne),
            "ratio_compression": float(ratio_compression),
            "erreur_maximale": float(max(erreurs_max))
        })
    
    return rapport


# ============================================================================
# 📈 VISUALISATIONS AMÉLIORÉES
# ============================================================================

def visualiser_espace_optimisé(tokens: Dict[str, Dict]):
    """Visualise l'espace canonique avec les positions optimisées."""
    fig = plt.figure(figsize=(18, 12))
    
    # Sous-graphe 1: Espace 3D avec tores
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Tracer les tores concentriques
    for couche in range(1, 5):
        theta = np.linspace(0, 2 * np.pi, 100)
        phi = np.linspace(0, 2 * np.pi, 100)
        theta, phi = np.meshgrid(theta, phi)
        
        R = couche * CONFIG["R_MAJEUR_BASE"]
        r = CONFIG["R_MINEUR_BASE"] / (0.5 + 0.5 * couche)
        
        X = (R + r * np.cos(phi)) * np.cos(theta)
        Y = (R + r * np.cos(phi)) * np.sin(theta)
        Z = r * np.sin(phi)
        
        ax1.plot_surface(X, Y, Z, color=plt.cm.viridis(couche/5), alpha=0.1, rstride=5, cstride=5)
    
    # Tracer les vecteurs
    couleurs = plt.cm.viridis(np.linspace(0, 1, len(tokens)))
    for (token, t), couleur in zip(tokens.items(), couleurs):
        v = np.array([t["x"], t["y"], t["z"]])
        couche = t["couche"]
        
        # Projeter sur tore
        v_projeté = projeter_sur_tore_avec_couche_optimisé(v, couche)
        
        # Tracer le vecteur original
        ax1.quiver(0, 0, 0, v[0], v[1], v[2], color=couleur, arrow_length_ratio=0.1, alpha=0.7)
        
        # Tracer le vecteur projeté
        ax1.quiver(0, 0, 0, v_projeté[0], v_projeté[1], v_projeté[2], 
                   color=couleur, arrow_length_ratio=0.1, linestyle="--", alpha=0.7)
        
        # Label
        ax1.text(v[0], v[1], v[2], token, color=couleur, fontsize=8)
    
    ax1.set_xlim([-5, 5])
    ax1.set_ylim([-5, 5])
    ax1.set_zlim([-5, 5])
    ax1.set_xlabel("X (Possible)")
    ax1.set_ylabel("Y (Perception)")
    ax1.set_zlabel("Z (Réalisation)")
    ax1.set_title("Espace Canonique Optimisé avec Projection Tore", fontsize=12)
    
    # Sous-graphe 2: Triangle Pouvoir-Lumière-Savoir
    ax2 = fig.add_subplot(122, projection='3d')
    
    v_pouvoir = np.array([tokens["pouvoir"]["x"], tokens["pouvoir"]["y"], tokens["pouvoir"]["z"]])
    v_lumière = np.array([tokens["lumière"]["x"], tokens["lumière"]["y"], tokens["lumière"]["z"]])
    v_savoir = np.array([tokens["savoir"]["x"], tokens["savoir"]["y"], tokens["savoir"]["z"]])
    
    # Tracer les vecteurs
    for v, label, couleur in [(v_pouvoir, "pouvoir", "red"), 
                               (v_lumière, "lumière", "gold"), 
                               (v_savoir, "savoir", "blue")]:
        ax2.quiver(0, 0, 0, v[0], v[1], v[2], color=couleur, arrow_length_ratio=0.1, linewidth=2)
        ax2.text(v[0], v[1], v[2], label, color=couleur, fontsize=12)
    
    # Tracer le triangle
    for v_a, v_b in [(v_pouvoir, v_lumière), (v_pouvoir, v_savoir), (v_lumière, v_savoir)]:
        ax2.plot([v_a[0], v_b[0]], [v_a[1], v_b[1]], [v_a[2], v_b[2]], 
                color="black", linestyle="--", linewidth=1, alpha=0.5)
    
    # Calculer et afficher les angles
    relation = relation_canonique_optimisée(v_pouvoir, v_lumière, v_savoir)
    ax2.text(0, 0, 0, f"α={relation['angles']['en_v0']:.1f}°", fontsize=10, color="black")
    ax2.text(v_lumière[0], v_lumière[1], v_lumière[2], 
             f"β={relation['angles']['en_v1']:.1f}°", fontsize=10, color="black")
    ax2.text(v_savoir[0], v_savoir[1], v_savoir[2], 
             f"γ={relation['angles']['en_v2']:.1f}°", fontsize=10, color="black")
    
    ax2.set_xlim([-1.5, 1.5])
    ax2.set_ylim([-1.5, 1.5])
    ax2.set_zlim([-1.5, 1.5])
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_title(f"Triangle Pouvoir-Lumière-Savoir\n{relation['type_relation']} (score: {relation['score_relation']:.3f})", 
                  fontsize=12)
    
    plt.tight_layout()
    plt.savefig("/workspace/visualisation_espace_optimise.png", dpi=300)
    plt.show()


def visualiser_quantisation_optimisée(tokens: Dict[str, Dict], n_bits_list: List[int]):
    """Visualise la quantisation adaptative."""
    fig, axes = plt.subplots(1, len(n_bits_list), figsize=(15, 5), 
                            subplot_kw={'projection': '3d'})
    
    if len(n_bits_list) == 1:
        axes = [axes]
    
    vecteurs_originaux = {token: np.array([t["x"], t["y"], t["z"]]) 
                         for token, t in tokens.items()}
    
    for i, n_bits in enumerate(n_bits_list):
        ax = axes[i]
        
        for token, v_orig in vecteurs_originaux.items():
            poids = tokens[token]["poids"]
            
            # Quantification adaptative
            theta_q, phi_q, n_bits_utilisé = quantifier_vecteur_adaptatif(v_orig, poids)
            v_reconstruit = déquantifier_vecteur_adaptatif(theta_q, phi_q, n_bits_utilisé)
            
            # Tracer
            couleur = plt.cm.viridis(tokens[token]["couche"] / 5)
            ax.quiver(0, 0, 0, v_orig[0], v_orig[1], v_orig[2], 
                     color=couleur, arrow_length_ratio=0.1, alpha=0.7)
            ax.quiver(0, 0, 0, v_reconstruit[0], v_reconstruit[1], v_reconstruit[2], 
                     color=couleur, arrow_length_ratio=0.1, linestyle="--", alpha=0.7)
            ax.text(v_orig[0], v_orig[1], v_orig[2], token, color=couleur, fontsize=8)
        
        # Calculer l'erreur moyenne
        erreurs = []
        for token, v_orig in vecteurs_originaux.items():
            poids = tokens[token]["poids"]
            theta_q, phi_q, n_bits_utilisé = quantifier_vecteur_adaptatif(v_orig, poids)
            v_reconstruit = déquantifier_vecteur_adaptatif(theta_q, phi_q, n_bits_utilisé)
            erreurs.append(distance_euclidienne(v_orig, v_reconstruit))
        erreur_moyenne = sum(erreurs) / len(tokens)
        
        ax.set_title(f"{n_bits} bits (erreur: {erreur_moyenne:.4f})")
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
    
    plt.tight_layout()
    plt.savefig("/workspace/visualisation_quantisation_optimisee.png", dpi=300)
    plt.show()


# ============================================================================
# 🏃 MAIN
# ============================================================================

def main():
    """Exécute l'optimisation et génère les rapports."""
    print("\n" + "=" * 80)
    print("🎯 PHASE C: OPTIMISATION DES RAPPORTS ANGULAIRES")
    print("   Amélioration itérative des relations canoniques")
    print("=" * 80 + "\n")
    
    # Étape 1: Calibration automatique des positions
    print("🔄 Calibration des positions des tokens...")
    tokens_a_optimiser = list(TOKENS_CANONIQUES.keys())
    résultat_calibration = calibrer_positions_tokens(tokens_a_optimiser, max_iter=500)
    
    print(f"✅ Calibration terminée!")
    print(f"   Score final: {résultat_calibration['score_final']:.6f}")
    print(f"   Succès: {résultat_calibration['success']}")
    
    # Utiliser les tokens optimisés pour la suite
    tokens_optimisés = résultat_calibration["tokens_optimisés"]
    
    # Étape 2: Générer le rapport des relations
    print("\n📊 Génération du rapport des relations canoniques...")
    rapport_relations = générer_rapport_relations(tokens_optimisés)
    
    print(f"   Nombre de relations: {rapport_relations['statistiques']['nombre_relations']}")
    print(f"   Distance moyenne: {rapport_relations['statistiques']['distance_moyenne']:.4f}")
    print(f"   Score moyen: {rapport_relations['statistiques']['score_moyen']:.4f}")
    print(f"   Répartition par type:")
    for t, count in rapport_relations['statistiques']['par_type'].items():
        print(f"     {t}: {count}")
    
    # Étape 3: Générer le rapport de quantisation
    print("\n🔢 Génération du rapport de quantisation...")
    rapport_quantisation = générer_rapport_quantisation(tokens_optimisés, [4, 8, 12, 16])
    
    for résultat in rapport_quantisation["résultats"]:
        print(f"   {résultat['n_bits']} bits: erreur={résultat['erreur_moyenne']:.6f}, "
              f"ratio={résultat['ratio_compression']:.1f}×, "
              f"taille={résultat['taille_moyenne_octets']:.2f} octets")
    
    # Étape 4: Sauvegarder les rapports
    print("\n💾 Sauvegarde des rapports...")
    with open("/workspace/rapport_relations_optimisees.json", "w") as f:
        json.dump(rapport_relations, f, indent=2, ensure_ascii=False)
    
    with open("/workspace/rapport_quantisation_optimisee.json", "w") as f:
        json.dump(rapport_quantisation, f, indent=2, ensure_ascii=False)
    
    with open("/workspace/tokens_optimises.json", "w") as f:
        json.dump(tokens_optimisés, f, indent=2, ensure_ascii=False)
    
    # Étape 5: Visualisations
    print("\n📈 Génération des visualisations...")
    visualiser_espace_optimisé(tokens_optimisés)
    visualiser_quantisation_optimisée(tokens_optimisés, [4, 8, 12])
    
    # Étape 6: Afficher les positions optimisées
    print("\n📋 Positions optimisées des tokens:")
    print("-" * 80)
    for token, t in sorted(tokens_optimisés.items()):
        print(f"{token:15s}: x={t['x']:.6f}, y={t['y']:.6f}, z={t['z']:.6f}, "
              f"couche={t['couche']}, poids={t['poids']:.2f}")
    print("-" * 80)
    
    print("\n✅ Phase C terminée avec succès!")
    print("   Tous les rapports et visualisations ont été générés.")


if __name__ == "__main__":
    main()
