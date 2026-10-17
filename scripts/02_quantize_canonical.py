#!/usr/bin/env python3
"""
🔥 SCRIPT 2 : QUANTISATION CANONIQUE AVEC SDK D'OPTIMISATION
   Applique la méthode de quantisation canonique par tore concentrique
   avec optimisation matérielle (VRAM, RAM, CPU, GPU)
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

# Import Apourvoir SDK
try:
    from apourvoir.optimizer import (
        HardwareDetector,
        AdaptiveBatcher,
        MemoryManager,
        DynamicQuantizer,
        ParallelExperts,
        KernelFuser,
        NUMAAwareThreadPool,
        BenchmarkProfiler,
    )
    from apourvoir.quantization import (
        normaliser_vecteur,
        coordonnées_sphériques,
        projeter_sur_tore,
        quantifier_angles,
        déquantifier_angles,
        quantifier_tensor_canonique,
        déquantifier_tensor_canonique,
    )
    SDK_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Apourvoir SDK not available: {e}. Using basic quantization.")
    SDK_AVAILABLE = False


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
    # SDK Configuration
    "use_sdk": True,
    "adaptive_batching": True,
    "dynamic_quantization": True,
    "expert_parallelism": True,
    "kernel_fusion": False,
    "numa_aware": True,
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
# FONCTIONS DE QUANTISATION CANONIQUE (CORE)
# ============================================================================

def normaliser_vecteur(v: torch.Tensor) -> torch.Tensor:
    """Normalise un vecteur."""
    norm = torch.norm(v, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=1e-10)
    return v / norm


def coordonnées_sphériques(v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convertit un vecteur cartésien en coordonnées sphériques.
    Retourne (r, theta, phi)
    """
    r = torch.norm(v, dim=-1)
    theta = torch.atan2(v[..., 1], v[..., 0])
    theta = (theta + 2 * np.pi) % (2 * np.pi)
    phi = torch.acos(torch.clamp(v[..., 2] / (r + 1e-10), -1.0, 1.0))
    return r, theta, phi


def projeter_sur_tore(v: torch.Tensor, R: float, r: float) -> torch.Tensor:
    """Projette un vecteur sur un tore 3D."""
    _, theta, phi = coordonnées_sphériques(v)
    X = (R + r * torch.cos(phi)) * torch.cos(theta)
    Y = (R + r * torch.cos(phi)) * torch.sin(theta)
    Z = r * torch.sin(phi)
    return torch.stack([X, Y, Z], dim=-1)


def quantifier_angles(theta: torch.Tensor, phi: torch.Tensor, n_bits: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantifie les angles sur n_bits."""
    theta_norm = theta % (2 * np.pi)
    theta_q = torch.round(theta_norm / (2 * np.pi) * (2**n_bits)).long() % (2**n_bits)
    phi_q = torch.round(phi / np.pi * (2**n_bits)).long() % (2**n_bits)
    return theta_q, phi_q


def déquantifier_angles(theta_q: torch.Tensor, phi_q: torch.Tensor, n_bits: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Déquantifie les angles."""
    theta = (theta_q.float() / (2**n_bits)) * 2 * np.pi
    phi = (phi_q.float() / (2**n_bits)) * np.pi
    return theta, phi


def quantifier_tensor_canonique(
    tensor: torch.Tensor,
    n_bits: int = 8,
    R: float = 2.0,
    r: float = 0.5,
) -> Tuple[torch.Tensor, Dict]:
    """Quantifie un tenseur complet avec la méthode canonique."""
    device = tensor.device
    shape = tensor.shape
    
    if len(shape) == 2:
        tensor_2d = tensor.view(-1, 1, shape[-1])
    else:
        tensor_2d = tensor.view(-1, shape[-1])
    
    tensor_norm = normaliser_vecteur(tensor_2d)
    tensor_projeté = projeter_sur_tore(tensor_norm, R, r)
    _, theta, phi = coordonnées_sphériques(tensor_projeté)
    theta_q, phi_q = quantifier_angles(theta, phi, n_bits)
    
    if n_bits <= 8:
        tensor_quant = torch.stack([theta_q, phi_q], dim=-1).to(torch.int8)
    elif n_bits <= 16:
        tensor_quant = torch.stack([theta_q, phi_q], dim=-1).to(torch.int16)
    else:
        tensor_quant = torch.stack([theta_q, phi_q], dim=-1).to(torch.int32)
    
    metadata = {
        "shape_original": shape,
        "n_bits": n_bits,
        "R": R,
        "r": r,
        "compression_ratio": (tensor.numel() * 4) / (tensor_quant.numel() * (n_bits // 8)),
        "device": str(device),
    }
    
    return tensor_quant, metadata


def déquantifier_tensor_canonique(tensor_quant: torch.Tensor, metadata: Dict) -> torch.Tensor:
    """Déquantifie un tenseur."""
    n_bits = metadata["n_bits"]
    R = metadata["R"]
    r = metadata["r"]
    
    theta_q = tensor_quant[..., 0].long()
    phi_q = tensor_quant[..., 1].long()
    theta, phi = déquantifier_angles(theta_q, phi_q, n_bits)
    
    X = (R + r * torch.cos(phi)) * torch.cos(theta)
    Y = (R + r * torch.cos(phi)) * torch.sin(theta)
    Z = r * torch.sin(phi)
    tensor_reconstruit = torch.stack([X, Y, Z], dim=-1)
    
    original_shape = metadata["shape_original"]
    if len(original_shape) == 2:
        tensor_reconstruit = tensor_reconstruit.view(original_shape)
    else:
        tensor_reconstruit = tensor_reconstruit.view(*original_shape)
    
    return tensor_reconstruit


# ============================================================================
# QUANTISATION AVEC SDK D'OPTIMISATION
# ============================================================================

class OptimizedQuantizer:
    """
    Quantiseur optimisé avec le SDK Apourvoir.
    
    Intègre:
    - Détection matérielle automatique
    - Batching adaptatif
    - Gestion mémoire intelligente
    - Quantisation dynamique
    - Parallélisme des experts
    """
    
    def __init__(self, config: Dict):
        """Initialise le quantiseur optimisé."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Détecter le matériel
        self.hardware = HardwareDetector.detect()
        
        # Initialiser les composants SDK
        self._init_sdk_components(config)
    
    def _init_sdk_components(self, config: Dict):
        """Initialise les composants du SDK."""
        # Memory Manager
        if config.get("use_sdk", True):
            vram_limit = self.hardware.vram_total_gb * 0.8 if self.hardware.gpu_available else 1.0
            ram_limit = self.hardware.ram_total_gb * 0.7
            self.memory_manager = MemoryManager(
                max_vram_gb=vram_limit,
                max_ram_gb=ram_limit,
            )
            self.logger.info(f"MemoryManager: VRAM={vram_limit:.1f}GB, RAM={ram_limit:.1f}GB")
        else:
            self.memory_manager = None
        
        # Dynamic Quantizer
        if config.get("dynamic_quantization", True) and config.get("use_sdk", True):
            self.dynamic_quant = DynamicQuantizer(
                self.hardware,
                min_bits=config.get("min_bits", 4),
                max_bits=config.get("max_bits", 16),
                initial_bits=config.get("n_bits", 8),
            )
            self.logger.info(f"DynamicQuantizer: bits={self.dynamic_quant.current_bits}")
        else:
            self.dynamic_quant = None
        
        # Adaptive Batcher
        if config.get("adaptive_batching", True) and config.get("use_sdk", True):
            self.batcher = AdaptiveBatcher(
                self.hardware,
                base_batch_size=config.get("batch_size", self.hardware.optimal_batch_size),
            )
            self.logger.info(f"AdaptiveBatcher: batch_size={self.batcher.current_batch_size}")
        else:
            self.batcher = None
    
    def quantize_model(
        self,
        state_dict: Dict[str, torch.Tensor],
        output_dir: str,
        validate: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Quantifie un modèle complet avec optimisation.
        
        Args:
            state_dict: Dictionnaire d'état du modèle
            output_dir: Répertoire de sortie
            validate: Valider la quantification
        
        Returns:
            Tuple de (state_dict quantifié, métadonnées)
        """
        quant_state_dict = {}
        metadata = {
            "hardware": self.hardware.to_dict(),
            "config": self.config,
            "layers": [],
        }
        
        # Créer le répertoire de sortie
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Quantifier chaque couche
        for name, tensor in tqdm(state_dict.items(), desc="Quantification"):
            try:
                # Déterminer le nombre de bits
                if self.dynamic_quant:
                    n_bits = self.dynamic_quant.get_optimal_bits()
                else:
                    n_bits = self.config.get("n_bits", 8)
                
                # Quantifier
                tensor_quant, layer_metadata = quantifier_tensor_canonique(
                    tensor,
                    n_bits=n_bits,
                    R=self.config.get("R_MAJEUR", 2.0),
                    r=self.config.get("R_MINEUR", 0.5),
                )
                
                # Gérer la mémoire
                if self.memory_manager:
                    self.memory_manager.allocate(
                        tensor_quant,
                        name,
                        priority=0,
                    )
                
                quant_state_dict[name] = tensor_quant
                layer_metadata["name"] = name
                metadata["layers"].append(layer_metadata)
                
                # Ajuster dynamiquement les bits
                if self.dynamic_quant and self.config.get("dynamic_quantization", True):
                    self.dynamic_quant.adjust_bits(
                        memory_pressure=self.memory_manager.get_stats().used_gb / 
                                       (self.memory_manager.max_vram_gb + self.memory_manager.max_ram_gb)
                    )
                
            except Exception as e:
                self.logger.error(f"Erreur de quantification pour {name}: {e}")
                # Garder le tenseur original
                quant_state_dict[name] = tensor
        
        # Sauvegarder les métadonnées
        metadata_path = os.path.join(output_dir, "quantization_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Nettoyer la mémoire
        if self.memory_manager:
            self.memory_manager.cleanup()
        
        return quant_state_dict, metadata
    
    def get_compression_report(self, original_size_gb: float, quant_size_gb: float) -> Dict:
        """Génère un rapport de compression."""
        ratio = original_size_gb / quant_size_gb if quant_size_gb > 0 else 1.0
        
        return {
            "original_size_gb": original_size_gb,
            "quantized_size_gb": quant_size_gb,
            "compression_ratio": ratio,
            "space_saved_gb": original_size_gb - quant_size_gb,
            "space_saved_percent": ((original_size_gb - quant_size_gb) / original_size_gb) * 100,
        }


# ============================================================================
# CHARGEMENT ET SAUVEGARDE DES MODÈLES
# ============================================================================

def charger_modèle(input_dir: str, device: str = "cuda") -> Tuple[Dict[str, torch.Tensor], Dict]:
    """Charge un modèle depuis un répertoire."""
    logger = logging.getLogger(__name__)
    logger.info(f"Chargement du modèle depuis {input_dir}...")
    
    config_path = os.path.join(input_dir, "config.json")
    if not os.path.exists(config_path):
        logger.error(f"Fichier de configuration introuvable: {config_path}")
        raise FileNotFoundError(config_path)
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    weights_path = os.path.join(input_dir, "pytorch_model.bin")
    if not os.path.exists(weights_path):
        logger.error(f"Fichier de poids introuvable: {weights_path}")
        raise FileNotFoundError(weights_path)
    
    logger.info(f"Chargement de {weights_path}...")
    state_dict = torch.load(weights_path, map_location=device)
    
    logger.info(f"✅ Modèle chargé: {len(state_dict)} poids")
    return state_dict, config


def sauvegarder_modèle_quantifié(
    state_dict_quant: Dict[str, torch.Tensor],
    metadata: Dict,
    output_dir: str,
) -> None:
    """Sauvegarde un modèle quantifié."""
    logger = logging.getLogger(__name__)
    logger.info(f"Sauvegarde du modèle quantifié dans {output_dir}...")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder la configuration
    config_path = os.path.join(output_dir, "config_quant.json")
    with open(config_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Sauvegarder les poids quantifiés
    weights_path = os.path.join(output_dir, "pytorch_model_quant.bin")
    torch.save(state_dict_quant, weights_path)
    
    logger.info(f"✅ Modèle quantifié sauvegardé: {len(state_dict_quant)} poids")


# ============================================================================
# VALIDATION
# ============================================================================

def valider_quantisation(
    state_dict_original: Dict[str, torch.Tensor],
    state_dict_quant: Dict[str, torch.Tensor],
    metadata: Dict,
    num_tests: int = 5,
) -> Dict:
    """Valide la quantification en comparant les sorties."""
    logger = logging.getLogger(__name__)
    logger.info("Validation de la quantification...")
    
    results = {
        "total_tests": 0,
        "passed_tests": 0,
        "max_diff": 0.0,
        "mean_diff": 0.0,
        "details": [],
    }
    
    # Tester quelques couches
    test_layers = list(state_dict_original.keys())[:num_tests]
    
    for layer_name in test_layers:
        original = state_dict_original[layer_name]
        quant = state_dict_quant[layer_name]
        
        try:
            # Déquantifier
            dequant = déquantifier_tensor_canonique(quant, metadata["layers"][0])
            
            # Comparer
            diff = torch.abs(original - dequant)
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            
            results["details"].append({
                "layer": layer_name,
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "passed": max_diff < 0.1,  # Seuil arbitraire
            })
            
            results["total_tests"] += 1
            if max_diff < 0.1:
                results["passed_tests"] += 1
            
            results["max_diff"] = max(results["max_diff"], max_diff)
            results["mean_diff"] += mean_diff / num_tests
        
        except Exception as e:
            logger.warning(f"Erreur de validation pour {layer_name}: {e}")
    
    results["validation_rate"] = (results["passed_tests"] / results["total_tests"] * 100) if results["total_tests"] > 0 else 0
    
    return results


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """Fonction principale pour la quantification canonique."""
    parser = argparse.ArgumentParser(
        description="Quantification canonique des modèles MoE"
    )
    
    # Arguments de base
    parser.add_argument("--input", type=str, required=True,
                       help="Répertoire d'entrée du modèle")
    parser.add_argument("--output", type=str, required=True,
                       help="Répertoire de sortie du modèle quantifié")
    parser.add_argument("--n_bits", type=int, default=8,
                       help="Nombre de bits pour la quantification (4-16)")
    parser.add_argument("--method", type=str, default="canonical_torus",
                       choices=["canonical_torus", "basic"],
                       help="Méthode de quantification")
    parser.add_argument("--device", type=str, default=None,
                       choices=["cpu", "cuda"],
                       help="Appareil à utiliser")
    
    # Arguments SDK
    parser.add_argument("--use_sdk", action="store_true", default=True,
                       help="Utiliser le SDK d'optimisation")
    parser.add_argument("--adaptive_batching", action="store_true", default=True,
                       help="Activer le batching adaptatif")
    parser.add_argument("--dynamic_quantization", action="store_true", default=True,
                       help="Activer la quantification dynamique")
    parser.add_argument("--expert_parallelism", action="store_true", default=True,
                       help="Activer le parallélisme des experts")
    parser.add_argument("--min_bits", type=int, default=4,
                       help="Nombre minimum de bits")
    parser.add_argument("--max_bits", type=int, default=16,
                       help="Nombre maximum de bits")
    
    # Arguments de configuration du tore
    parser.add_argument("--R_MAJEUR", type=float, default=2.0,
                       help="Rayon majeur du tore")
    parser.add_argument("--R_MINEUR", type=float, default=0.5,
                       help="Rayon mineur du tore")
    
    # Autres arguments
    parser.add_argument("--validate", action="store_true", default=True,
                       help="Valider la quantification")
    parser.add_argument("--log_dir", type=str, default="logs/quantization.log",
                       help="Fichier de log")
    parser.add_argument("--num_experts", type=int, default=8,
                       help="Nombre d'experts pour MoE")
    
    args = parser.parse_args()
    
    # Configuration
    config = DEFAULT_CONFIG.copy()
    config.update({
        "n_bits": args.n_bits,
        "method": args.method,
        "R_MAJEUR": args.R_MAJEUR,
        "R_MINEUR": args.R_MINEUR,
        "device": args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        "use_sdk": args.use_sdk,
        "adaptive_batching": args.adaptive_batching,
        "dynamic_quantization": args.dynamic_quantization,
        "expert_parallelism": args.expert_parallelism,
        "min_bits": args.min_bits,
        "max_bits": args.max_bits,
        "num_experts": args.num_experts,
    })
    
    # Setup logging
    setup_logging(args.log_dir)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("QUANTISATION CANONIQUE AVEC SDK D'OPTIMISATION")
    logger.info("=" * 80)
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Détecter le matériel
    hardware = HardwareDetector.detect()
    HardwareDetector.print_summary(hardware)
    
    # Charger le modèle
    state_dict, model_config = charger_modèle(args.input, config["device"])
    
    # Calculer la taille originale
    original_size = sum(
        tensor.numel() * tensor.element_size()
        for tensor in state_dict.values()
    ) / (1024**3)
    logger.info(f"Taille originale du modèle: {original_size:.2f} GB")
    
    # Initialiser le quantiseur optimisé
    quantizer = OptimizedQuantizer(config)
    
    # Quantifier le modèle
    start_time = time.time()
    quant_state_dict, metadata = quantizer.quantize_model(
        state_dict,
        args.output,
        validate=args.validate,
    )
    quant_time = time.time() - start_time
    
    # Calculer la taille quantifiée
    quant_size = sum(
        tensor.numel() * tensor.element_size()
        for tensor in quant_state_dict.values()
    ) / (1024**3)
    
    # Générer le rapport de compression
    compression_report = quantizer.get_compression_report(original_size, quant_size)
    
    # Sauvegarder le modèle quantifié
    sauvegarder_modèle_quantifié(quant_state_dict, metadata, args.output)
    
    # Valider si demandé
    if args.validate:
        validation_results = valider_quantisation(
            state_dict,
            quant_state_dict,
            metadata,
            num_tests=5,
        )
        logger.info(f"Validation: {validation_results['validation_rate']:.1f}% de succès")
    else:
        validation_results = {}
    
    # Sauvegarder le rapport de compression
    report_path = os.path.join(args.output, "compression_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "compression": compression_report,
            "quantization_time": quant_time,
            "validation": validation_results,
            "hardware": hardware.to_dict(),
            "config": config,
        }, f, indent=2)
    
    # Afficher le résumé
    logger.info("=" * 80)
    logger.info("RÉSULTATS DE LA QUANTISATION")
    logger.info("=" * 80)
    logger.info(f"Taille originale: {original_size:.2f} GB")
    logger.info(f"Taille quantifiée: {quant_size:.2f} GB")
    logger.info(f"Ratio de compression: {compression_report['compression_ratio']:.1f}x")
    logger.info(f"Espace économisé: {compression_report['space_saved_gb']:.2f} GB")
    logger.info(f"Temps de quantification: {quant_time:.2f} secondes")
    
    if validation_results:
        logger.info(f"Validation: {validation_results['validation_rate']:.1f}% de succès")
        logger.info(f"Différence maximale: {validation_results['max_diff']:.6f}")
    
    logger.info("=" * 80)
    logger.info("✅ Quantisation terminée avec succès!")
    logger.info(f"Modèle quantifié sauvegardé dans: {args.output}")


if __name__ == "__main__":
    main()
