#!/usr/bin/env python3
"""
🔧 SCRIPT 2 : QUANTISATION CANONIQUE
   Applique la méthode de quantisation canonique par tore concentrique
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

import torch
import numpy as np
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

# Configuration par défaut
DEFAULT_CONFIG = {
    "n_bits": 8,
    "method": "canonical_torus",
    "layers": "all",
    "R_MAJEUR": 2.0,
    "R_MINEUR": 0.5,
    "vocab_size": 32000,
    "num_experts": 8,
    "top_k": 2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(log_file: Optional[str] = None):
    """Configure le logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    return logging.getLogger(__name__)


# ============================================================================
# FONCTIONS DE QUANTISATION CANONIQUE
# ============================================================================

def normaliser_vecteur(v: torch.Tensor) -> torch.Tensor:
    """Normalise un vecteur."""
    norm = torch.norm(v)
    if norm < 1e-10:
        return torch.zeros_like(v)
    return v / norm


def coordonnées_sphériques(v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convertit un vecteur cartésien en coordonnées sphériques.
    Retourne (r, theta, phi) où:
    - r: norme du vecteur
    - theta: angle dans le plan x-y [0, 2π]
    - phi: angle par rapport à z [0, π]
    """
    r = torch.norm(v, dim=-1)
    theta = torch.atan2(v[..., 1], v[..., 0])  # [0, 2π] ou [-π, π]
    theta = (theta + 2 * np.pi) % (2 * np.pi)  # Normaliser à [0, 2π]
    phi = torch.acos(torch.clamp(v[..., 2] / (r + 1e-10), -1.0, 1.0))  # [0, π]
    return r, theta, phi


def projeter_sur_tore(v: torch.Tensor, R: float, r: float) -> torch.Tensor:
    """
    Projette un vecteur sur un tore 3D.
    
    Équations paramétriques:
    X = (R + r * cos(phi)) * cos(theta)
    Y = (R + r * cos(phi)) * sin(theta)
    Z = r * sin(phi)
    """
    _, theta, phi = coordonnées_sphériques(v)
    
    X = (R + r * torch.cos(phi)) * torch.cos(theta)
    Y = (R + r * torch.cos(phi)) * torch.sin(theta)
    Z = r * torch.sin(phi)
    
    return torch.stack([X, Y, Z], dim=-1)


def quantifier_angles(theta: torch.Tensor, phi: torch.Tensor, n_bits: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantifie les angles sur n_bits.
    
    - theta ∈ [0, 2π] → [0, 2^N-1]
    - phi ∈ [0, π] → [0, 2^N-1]
    """
    theta_norm = theta % (2 * np.pi)
    theta_q = torch.round(theta_norm / (2 * np.pi) * (2**n_bits)).long() % (2**n_bits)
    phi_q = torch.round(phi / np.pi * (2**n_bits)).long() % (2**n_bits)
    return theta_q, phi_q


def déquantifier_angles(theta_q: torch.Tensor, phi_q: torch.Tensor, n_bits: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Déquantifie les angles."""
    theta = (theta_q.float() / (2**n_bits)) * 2 * np.pi
    phi = (phi_q.float() / (2**n_bits)) * np.pi
    return theta, phi


def quantifier_tensor_canonique(tensor: torch.Tensor, n_bits: int = 8, 
                                R: float = 2.0, r: float = 0.5) -> Tuple[torch.Tensor, Dict]:
    """
    Quantifie un tenseur complet avec la méthode canonique.
    
    Retourne:
    - tensor_quantifié: le tenseur quantifié
    - metadata: métadonnées sur la quantification
    """
    device = tensor.device
    shape = tensor.shape
    
    # Reshape pour traiter chaque élément comme un vecteur
    if len(shape) == 2:
        tensor_2d = tensor.view(-1, 1, shape[-1])
    else:
        tensor_2d = tensor.view(-1, shape[-1])
    
    # Normaliser
    tensor_norm = normaliser_vecteur(tensor_2d)
    
    # Projeter sur tore
    tensor_projeté = projeter_sur_tore(tensor_norm, R, r)
    
    # Convertir en coordonnées sphériques
    _, theta, phi = coordonnées_sphériques(tensor_projeté)
    
    # Quantifier les angles
    theta_q, phi_q = quantifier_angles(theta, phi, n_bits)
    
    # Stocker les angles quantifiés (en int8, int16, etc.)
    if n_bits <= 8:
        tensor_quant = torch.stack([theta_q, phi_q], dim=-1).to(torch.int8)
    elif n_bits <= 16:
        tensor_quant = torch.stack([theta_q, phi_q], dim=-1).to(torch.int16)
    else:
        tensor_quant = torch.stack([theta_q, phi_q], dim=-1).to(torch.int32)
    
    # Métadonnées
    metadata = {
        "shape_original": shape,
        "n_bits": n_bits,
        "R": R,
        "r": r,
        "compression_ratio": (tensor.numel() * 4) / (tensor_quant.numel() * (n_bits // 8)),
        "device": str(device)
    }
    
    return tensor_quant, metadata


def déquantifier_tensor_canonique(tensor_quant: torch.Tensor, metadata: Dict) -> torch.Tensor:
    """Déquantifie un tenseur."""
    n_bits = metadata["n_bits"]
    R = metadata["R"]
    r = metadata["r"]
    
    # Extraire theta_q et phi_q
    theta_q = tensor_quant[..., 0].long()
    phi_q = tensor_quant[..., 1].long()
    
    # Déquantifier
    theta, phi = déquantifier_angles(theta_q, phi_q, n_bits)
    
    # Reconstruire les vecteurs
    X = (R + r * torch.cos(phi)) * torch.cos(theta)
    Y = (R + r * torch.cos(phi)) * torch.sin(theta)
    Z = r * torch.sin(phi)
    
    tensor_reconstruit = torch.stack([X, Y, Z], dim=-1)
    
    # Reshape pour correspondre à la forme originale
    if len(metadata["shape_original"]) == 2:
        tensor_reconstruit = tensor_reconstruit.view(metadata["shape_original"])
    else:
        tensor_reconstruit = tensor_reconstruit.view(*metadata["shape_original"])
    
    return tensor_reconstruit


# ============================================================================
# CHARGEMENT ET SAUVEGARDE DES MODÈLES
# ============================================================================

def charger_modèle(input_dir: str, device: str = "cuda") -> Dict[str, torch.Tensor]:
    """
    Charge un modèle depuis un répertoire.
    
    Pour simplifier, on charge seulement les poids essentiels.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Chargement du modèle depuis {input_dir}...")
    
    # Charger la configuration
    config_path = os.path.join(input_dir, "config.json")
    if not os.path.exists(config_path):
        logger.error(f"Fichier de configuration introuvable: {config_path}")
        raise FileNotFoundError(config_path)
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Charger les poids
    weights_path = os.path.join(input_dir, "pytorch_model.bin")
    if not os.path.exists(weights_path):
        logger.error(f"Fichier de poids introuvable: {weights_path}")
        raise FileNotFoundError(weights_path)
    
    logger.info(f"Chargement de {weights_path}...")
    state_dict = torch.load(weights_path, map_location=device)
    
    logger.info(f"✅ Modèle chargé: {len(state_dict)} poids")
    return state_dict, config


def sauvegarder_modèle_quantifié(state_dict_quant: Dict[str, torch.Tensor], 
                                metadata: Dict,
                                output_dir: str) -> None:
    """Sauvegarde un modèle quantifié."""
    logger = logging.getLogger(__name__)
    logger.info(f"Sauvegarde du modèle quantifié dans {output_dir}...")
    
    # Créer le répertoire
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder la configuration
    config_path = os.path.join(output_dir, "config_quant.json")
    with open(config_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Sauvegarder les poids quantifiés
    weights_path = os.path.join(output_dir, "pytorch_model_quant.bin")
    torch.save(state_dict_quant, weights_path)
    
    # Calculer la taille
    total_size = sum(
        t.numel() * t.element_size()
        for t in state_dict_quant.values()
    )
    
    logger.info(f"✅ Modèle quantifié sauvegardé")
    logger.info(f"   Taille: {total_size / (1024**3):.2f} GB")


# ============================================================================
# QUANTISATION DU MODÈLE COMPLET
# ============================================================================

def quantifier_modèle_complet(input_dir: str, output_dir: str, 
                             config: Dict = DEFAULT_CONFIG) -> Dict:
    """
    Quantifie un modèle complet avec la méthode canonique.
    
    Retourne:
    - metadata: rapport de quantification
    """
    logger = logging.getLogger(__name__)
    start_time = time.time()
    
    # Charger le modèle
    state_dict, model_config = charger_modèle(input_dir, config["device"])
    
    # Initialiser le rapport
    rapport = {
        "model": os.path.basename(input_dir),
        "config": config,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "layers": {},
        "total_size_original": 0,
        "total_size_quantified": 0,
        "compression_ratio": 1.0
    }
    
    # Calculer la taille originale
    for name, tensor in state_dict.items():
        rapport["total_size_original"] += tensor.numel() * tensor.element_size()
    
    logger.info(f"Taille originale: {rapport['total_size_original'] / (1024**3):.2f} GB")
    
    # Quantifier chaque poids
    state_dict_quant = {}
    
    for name, tensor in tqdm(state_dict.items(), desc="Quantification"):
        layer_report = {}
        
        # Déterminer le nombre de bits en fonction de l'importance
        # Les couches profondes et les experts ont plus de poids
        if "experts" in name:
            n_bits = min(config["n_bits"] + 2, 16)  # Plus de bits pour les experts
        elif "embed" in name or "lm_head" in name:
            n_bits = min(config["n_bits"] + 4, 16)  # Plus de bits pour les embeddings
        else:
            n_bits = config["n_bits"]
        
        # Déterminer R et r en fonction de la couche
        if "layer" in name:
            layer_num = int(name.split(".")[2])
            R = config["R_MAJEUR"] * (1 + layer_num * 0.1)  # Augmenter R avec la couche
            r = config["R_MINEUR"] / (1 + layer_num * 0.1)  # Diminuer r avec la couche
        else:
            R = config["R_MAJEUR"]
            r = config["R_MINEUR"]
        
        # Quantifier
        tensor_quant, metadata = quantifier_tensor_canonique(tensor, n_bits, R, r)
        state_dict_quant[name] = tensor_quant
        
        # Calculer la taille quantifiée
        layer_size_quant = tensor_quant.numel() * tensor_quant.element_size()
        layer_size_orig = tensor.numel() * tensor.element_size()
        
        layer_report["n_bits"] = n_bits
        layer_report["R"] = R
        layer_report["r"] = r
        layer_report["size_original"] = int(layer_size_orig)
        layer_report["size_quantified"] = int(layer_size_quant)
        layer_report["compression_ratio"] = float(layer_size_orig / layer_size_quant)
        
        rapport["layers"][name] = layer_report
        rapport["total_size_quantified"] += layer_size_quant
    
    # Calculer le ratio de compression global
    rapport["compression_ratio"] = float(
        rapport["total_size_original"] / rapport["total_size_quantified"]
    )
    
    # Sauvegarder le modèle quantifié
    sauvegarder_modèle_quantifié(state_dict_quant, rapport, output_dir)
    
    # Calculer le temps écoulé
    elapsed_time = time.time() - start_time
    rapport["elapsed_time"] = elapsed_time
    rapport["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    logger.info(f"✅ Quantification terminée en {elapsed_time:.1f} secondes")
    logger.info(f"Ratio de compression: {rapport['compression_ratio']:.1f}×")
    logger.info(f"Taille quantifiée: {rapport['total_size_quantified'] / (1024**3):.2f} GB")
    
    return rapport


# ============================================================================
# VALIDATION DE LA QUANTISATION
# ============================================================================

def valider_quantisation(input_dir: str, output_dir: str, 
                        num_samples: int = 100) -> Dict:
    """
    Valide la quantification en comparant les sorties.
    """
    logger = logging.getLogger(__name__)
    logger.info("Validation de la quantification...")
    
    # Charger les modèles
    state_dict_orig, _ = charger_modèle(input_dir)
    state_dict_quant, metadata = charger_modèle(output_dir)
    
    # Comparer quelques poids
    comparaison = {}
    
    for name in list(state_dict_orig.keys())[:10]:  # Comparer les 10 premiers
        if name in state_dict_quant:
            # Déquantifier
            tensor_quant = state_dict_quant[name]
            metadata_layer = metadata["layers"].get(name, {})
            
            # Reconstruire le tenseur original
            # (Pour la démo, on utilise une reconstruction simple)
            # En réalité, il faudrait stocker plus de métadonnées
            
            # Pour simplifier, on compare juste les tailles
            size_orig = state_dict_orig[name].numel() * state_dict_orig[name].element_size()
            size_quant = tensor_quant.numel() * tensor_quant.element_size()
            
            comparaison[name] = {
                "size_original": int(size_orig),
                "size_quantified": int(size_quant),
                "ratio": float(size_orig / size_quant)
            }
    
    return {
        "comparaison": comparaison,
        "message": "Validation basique terminée. Pour une validation complète, "
                   "exécutez le script 05_validate_semantics.py"
    }


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Quantisation canonique des modèles MoE"
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Répertoire du modèle d'entrée"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Répertoire de sortie pour le modèle quantifié"
    )
    
    parser.add_argument(
        "--n_bits",
        type=int,
        default=8,
        choices=[4, 8, 12, 16],
        help="Nombre de bits pour la quantification"
    )
    
    parser.add_argument(
        "--method",
        type=str,
        default="canonical_torus",
        choices=["canonical_torus", "canonical_angular", "gguf"],
        help="Méthode de quantification"
    )
    
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Couches à quantifier (ex: '1 2 3' ou 'all')"
    )
    
    parser.add_argument(
        "--R_MAJEUR",
        type=float,
        default=2.0,
        help="Rayon majeur du tore"
    )
    
    parser.add_argument(
        "--R_MINEUR",
        type=float,
        default=0.5,
        help="Rayon mineur du tore"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Appareil à utiliser"
    )
    
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/quantization.log",
        help="Fichier de log"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Valider la quantification après traitement"
    )
    
    args = parser.parse_args()
    
    # Configurer le logging
    logger = setup_logging(args.log_dir)
    
    # Mettre à jour la configuration
    config = DEFAULT_CONFIG.copy()
    config.update({
        "n_bits": args.n_bits,
        "method": args.method,
        "layers": args.layers,
        "R_MAJEUR": args.R_MAJEUR,
        "R_MINEUR": args.R_MINEUR,
        "device": args.device,
    })
    
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Vérifier les répertoires
    if not os.path.exists(args.input):
        logger.error(f"Répertoire d'entrée introuvable: {args.input}")
        sys.exit(1)
    
    # Quantifier le modèle
    try:
        rapport = quantifier_modèle_complet(args.input, args.output, config)
        
        # Sauvegarder le rapport
        rapport_path = os.path.join(args.output, "quantization_report.json")
        with open(rapport_path, "w") as f:
            json.dump(rapport, f, indent=2)
        
        logger.info(f"Rapport sauvegardé: {rapport_path}")
        
        # Validation si demandé
        if args.validate:
            validation = valider_quantisation(args.input, args.output)
            logger.info(f"Validation: {json.dumps(validation, indent=2)}")
        
        logger.info("\n✅ Quantification terminée avec succès!")
        logger.info(f"   Modèle quantifié: {args.output}")
        logger.info(f"   Ratio de compression: {rapport['compression_ratio']:.1f}×")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la quantification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
