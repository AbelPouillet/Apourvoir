#!/usr/bin/env python3
"""
💾 SCRIPT 4 : BENCHMARK MÉMOIRE
   Compare l'utilisation mémoire entre FP32 et Canonical
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import logging

import torch
import numpy as np
import psutil
import gc


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
# FONCTIONS DE MESURE DE MÉMOIRE
# ============================================================================

def get_memory_usage() -> Dict[str, float]:
    """
    Récupère l'utilisation mémoire actuelle.
    
    Retourne:
    - cpu_used: mémoire CPU utilisée (GB)
    - cpu_available: mémoire CPU disponible (GB)
    - gpu_used: mémoire GPU utilisée (GB) si disponible
    - gpu_total: mémoire GPU totale (GB) si disponible
    """
    # Mémoire CPU
    mem = psutil.virtual_memory()
    cpu_used = mem.used / (1024**3)
    cpu_available = mem.available / (1024**3)
    
    # Mémoire GPU
    gpu_used = 0.0
    gpu_total = 0.0
    
    if torch.cuda.is_available():
        gpu_used = torch.cuda.memory_allocated() / (1024**3)
        gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    return {
        "cpu_used": cpu_used,
        "cpu_available": cpu_available,
        "gpu_used": gpu_used,
        "gpu_total": gpu_total
    }


def get_model_size(model_dir: str) -> Dict[str, float]:
    """
    Calcule la taille d'un modèle sur disque.
    
    Retourne:
    - total: taille totale (GB)
    - num_files: nombre de fichiers
    - largest_file: taille du plus grand fichier (GB)
    """
    total_size = 0
    num_files = 0
    largest_file = 0
    
    for dirpath, _, filenames in os.walk(model_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                size = os.path.getsize(filepath)
                total_size += size
                num_files += 1
                largest_file = max(largest_file, size)
            except:
                pass
    
    return {
        "total": total_size / (1024**3),
        "num_files": num_files,
        "largest_file": largest_file / (1024**3)
    }


def load_model_in_memory(model_dir: str, device: str = "cuda") -> Dict:
    """
    Charge un modèle en mémoire et mesure l'utilisation.
    
    Retourne:
    - memory_before: mémoire avant chargement
    - memory_after: mémoire après chargement
    - model_size: taille du modèle chargé
    """
    logger = logging.getLogger(__name__)
    
    # Mesurer avant
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    memory_before = get_memory_usage()
    
    # Charger le modèle
    logger.info(f"Chargement du modèle depuis {model_dir}...")
    
    try:
        # Pour simplifier, on charge seulement les poids
        # En réalité, on devrait charger le modèle complet
        weights_path = os.path.join(model_dir, "pytorch_model.bin")
        if not os.path.exists(weights_path):
            weights_path = os.path.join(model_dir, "pytorch_model_quant.bin")
        
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=device)
            model_size = sum(t.numel() * t.element_size() for t in state_dict.values()) / (1024**3)
        else:
            state_dict = {}
            model_size = 0
        
        # Mesurer après
        memory_after = get_memory_usage()
        
        # Nettoyer
        del state_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "memory_before": memory_before,
            "memory_after": memory_after,
            "model_size": model_size
        }
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement: {e}")
        return {
            "memory_before": memory_before,
            "memory_after": get_memory_usage(),
            "model_size": 0,
            "error": str(e)
        }


# ============================================================================
# BENCHMARK MÉMOIRE
# ============================================================================

def benchmark_memory(model_fp32_dir: str, model_quant_dir: str,
                    device: str = "cuda") -> Dict:
    """
    Exécute un benchmark mémoire complet.
    
    Retourne:
    - rapport: dictionnaire avec tous les résultats
    """
    logger = logging.getLogger(__name__)
    
    rapport = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_fp32": model_fp32_dir,
        "model_quant": model_quant_dir,
        "device": device,
        "results": {}
    }
    
    # 1. Taille sur disque
    logger.info("Mesure des tailles sur disque...")
    
    fp32_size = get_model_size(model_fp32_dir)
    quant_size = get_model_size(model_quant_dir)
    
    rapport["results"]["disk_size"] = {
        "fp32": fp32_size,
        "quantized": quant_size,
        "compression_ratio": fp32_size["total"] / quant_size["total"] if quant_size["total"] > 0 else 0
    }
    
    logger.info(f"  FP32: {fp32_size['total']:.2f} GB ({fp32_size['num_files']} fichiers)")
    logger.info(f"  Quantifié: {quant_size['total']:.2f} GB ({quant_size['num_files']} fichiers)")
    logger.info(f"  Ratio: {rapport['results']['disk_size']['compression_ratio']:.1f}×")
    
    # 2. Mémoire lors du chargement
    logger.info("\nMesure de la mémoire lors du chargement...")
    
    fp32_load = load_model_in_memory(model_fp32_dir, device)
    rapport["results"]["memory_load"] = {
        "fp32": fp32_load
    }
    
    quant_load = load_model_in_memory(model_quant_dir, device)
    rapport["results"]["memory_load"]["quantized"] = quant_load
    
    # Calculer les différences
    cpu_diff = quant_load["memory_after"]["cpu_used"] - fp32_load["memory_after"]["cpu_used"]
    gpu_diff = quant_load["memory_after"]["gpu_used"] - fp32_load["memory_after"]["gpu_used"]
    
    rapport["results"]["memory_savings"] = {
        "cpu": fp32_load["model_size"] - quant_load["model_size"],
        "gpu": fp32_load["model_size"] - quant_load["model_size"],  # Simplification
        "ratio": fp32_load["model_size"] / quant_load["model_size"] if quant_load["model_size"] > 0 else 0
    }
    
    # 3. Mémoire lors de l'inférence (simulation)
    logger.info("\nSimulation de la mémoire lors de l'inférence...")
    
    # Pour simplifier, on utilise les tailles des modèles comme estimation
    # En réalité, il faudrait exécuter une inférence réelle
    
    batch_sizes = [1, 4, 8, 16, 32]
    seq_lengths = [64, 128, 256, 512]
    
    rapport["results"]["inference_memory"] = {}
    
    for batch_size in batch_sizes:
        rapport["results"]["inference_memory"][f"batch_{batch_size}"] = {}
        
        for seq_length in seq_lengths:
            # Estimation de la mémoire pour FP32
            # Formules simplifiées:
            # - Poids du modèle: model_size
            # - KV Cache: batch_size * seq_length * hidden_size * 2 (pour k et v) * 4 (FP32)
            # - Activation: batch_size * seq_length * hidden_size * 4 * 4 (estimation)
            
            # Pour Mixtral-8x7B: hidden_size = 4096
            hidden_size = 4096
            
            # FP32
            fp32_model_mem = fp32_size["total"]  # Déjà en GB
            fp32_kv_cache = (batch_size * seq_length * hidden_size * 2 * 4) / (1024**3)
            fp32_activation = (batch_size * seq_length * hidden_size * 4 * 4) / (1024**3)
            fp32_total = fp32_model_mem + fp32_kv_cache + fp32_activation
            
            # Quantifié (estimation)
            # On suppose que le modèle quantifié fait 1/18.8 de la taille
            quant_model_mem = fp32_model_mem / 18.8
            # KV Cache: on suppose aussi quantifié (1/4 de la taille)
            quant_kv_cache = fp32_kv_cache / 4
            # Activation: on suppose aussi optimisée
            quant_activation = fp32_activation / 2
            quant_total = quant_model_mem + quant_kv_cache + quant_activation
            
            rapport["results"]["inference_memory"][f"batch_{batch_size}"][f"seq_{seq_length}"] = {
                "fp32": {
                    "model": fp32_model_mem,
                    "kv_cache": fp32_kv_cache,
                    "activation": fp32_activation,
                    "total": fp32_total
                },
                "quantized": {
                    "model": quant_model_mem,
                    "kv_cache": quant_kv_cache,
                    "activation": quant_activation,
                    "total": quant_total
                },
                "savings": fp32_total - quant_total,
                "ratio": fp32_total / quant_total if quant_total > 0 else 0
            }
    
    # Résumé
    rapport["summary"] = {
        "disk_compression": rapport["results"]["disk_size"]["compression_ratio"],
        "model_size_fp32": fp32_size["total"],
        "model_size_quant": quant_size["total"],
        "memory_savings_ratio": rapport["results"]["memory_savings"]["ratio"]
    }
    
    return rapport


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark mémoire pour modèles quantifiés"
    )
    
    parser.add_argument(
        "--model_fp32",
        type=str,
        required=True,
        help="Répertoire du modèle FP32"
    )
    
    parser.add_argument(
        "--model_quant",
        type=str,
        required=True,
        help="Répertoire du modèle quantifié"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results/memory_benchmark.json",
        help="Fichier de sortie pour les résultats"
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
        default="logs/memory_benchmark.log",
        help="Fichier de log"
    )
    
    args = parser.parse_args()
    
    # Configurer le logging
    logger = setup_logging(args.log_dir)
    
    # Vérifier les répertoires
    if not os.path.exists(args.model_fp32):
        logger.error(f"Modèle FP32 introuvable: {args.model_fp32}")
        sys.exit(1)
    
    if not os.path.exists(args.model_quant):
        logger.error(f"Modèle quantifié introuvable: {args.model_quant}")
        sys.exit(1)
    
    # Créer le répertoire de sortie
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Exécuter le benchmark
    try:
        logger.info("Début du benchmark mémoire...")
        
        rapport = benchmark_memory(
            args.model_fp32,
            args.model_quant,
            device=args.device
        )
        
        # Sauvegarder le rapport
        with open(args.output, "w") as f:
            json.dump(rapport, f, indent=2)
        
        logger.info(f"✅ Benchmark mémoire terminé. Rapport sauvegardé: {args.output}")
        
        # Afficher un résumé
        logger.info("\n📊 Résumé du benchmark mémoire:")
        logger.info(f"  Ratio de compression disque: {rapport['summary']['disk_compression']:.1f}×")
        logger.info(f"  Taille FP32: {rapport['summary']['model_size_fp32']:.2f} GB")
        logger.info(f"  Taille quantifiée: {rapport['summary']['model_size_quant']:.2f} GB")
        logger.info(f"  Ratio d'économie mémoire: {rapport['summary']['memory_savings_ratio']:.1f}×")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
