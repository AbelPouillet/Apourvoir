#!/usr/bin/env python3
"""
📥 SCRIPT 1 : TÉLÉCHARGEMENT DES MODÈLES
   Télécharge Mixtral-8x7B ou crée des simulations pour tests légers
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional
import subprocess
import shutil
import numpy as np
import torch


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_CONFIGS = {
    "mistralai/Mixtral-8x7B-v0.1": {
        "name": "mixtral_8x7b",
        "size": "47GB",
        "params": "47B",
        "hf_path": "mistralai/Mixtral-8x7B-v0.1"
    },
    "mistralai/Mistral-7B-v0.1": {
        "name": "mistral_7b",
        "size": "14GB",
        "params": "7B",
        "hf_path": "mistralai/Mistral-7B-v0.1"
    },
}

# Taille des modèles simulés
SIMULATED_SIZES = {
    "tiny": {"layers": 2, "hidden_size": 64, "num_heads": 4, "vocab_size": 1000},
    "small": {"layers": 6, "hidden_size": 128, "num_heads": 8, "vocab_size": 5000},
    "medium": {"layers": 12, "hidden_size": 256, "num_heads": 16, "vocab_size": 10000},
}


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def check_huggingface_cli() -> bool:
    """Vérifie si huggingface-cli est installé."""
    try:
        result = subprocess.run(["huggingface-cli", "--version"], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False


def check_disk_space(path: str, required_gb: int) -> bool:
    """Vérifie l'espace disque disponible."""
    try:
        stat = os.statvfs(path)
        free_gb = stat.f_frsize * stat.f_bavail / (1024**3)
        return free_gb >= required_gb
    except:
        return True  # On suppose qu'il y a assez d'espace


def get_model_size(model_name: str) -> int:
    """Récupère la taille du modèle en GB."""
    if model_name in MODEL_CONFIGS:
        size_str = MODEL_CONFIGS[model_name]["size"]
        return int(size_str.replace("GB", ""))
    return 50  # Défaut


def create_directory(path: str) -> None:
    """Crée un répertoire."""
    Path(path).mkdir(parents=True, exist_ok=True)


def download_from_huggingface(model_path: str, output_dir: str) -> bool:
    """Télécharge un modèle depuis HuggingFace."""
    print(f"📥 Téléchargement de {model_path}...")
    
    try:
        cmd = [
            "huggingface-cli", "download",
            model_path,
            "--local-dir", output_dir,
            "--local-dir-use-symlinks", "False"
        ]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        stderr = process.stderr.read()
        if stderr:
            print(f"⚠️  Avertissements: {stderr}", file=sys.stderr)
        
        if process.returncode == 0:
            print(f"✅ Téléchargement terminé: {output_dir}")
            return True
        else:
            print(f"❌ Échec du téléchargement (code: {process.returncode})")
            return False
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def create_simulated_model(output_dir: str, size: str = "small") -> bool:
    """
    Crée un modèle simulé pour tests légers.
    
    Le modèle simulé a la même structure que Mixtral mais avec des poids aléatoires
    et une taille réduite.
    """
    print(f"🎭 Création d'un modèle simulé (taille: {size})...")
    
    if size not in SIMULATED_SIZES:
        print(f"❌ Taille inconnue: {size}")
        return False
    
    config = SIMULATED_SIZES[size]
    
    # Créer le répertoire
    create_directory(output_dir)
    
    # Sauvegarder la configuration
    model_config = {
        "model_type": "mixtral",
        "simulated": True,
        "original_size": size,
        "num_layers": config["layers"],
        "hidden_size": config["hidden_size"],
        "num_heads": config["num_heads"],
        "vocab_size": config["vocab_size"],
        "num_experts": 8,  # Comme Mixtral-8x7B
        "top_k_experts": 2,
    }
    
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    
    # Créer les poids simulés
    print("  ✓ Génération des poids...")
    
    # Sauvegarder les poids en FP32
    weights_dir = os.path.join(output_dir, "pytorch_model.bin")
    
    # Pour simplifier, on crée un seul fichier avec tous les poids
    # En réalité, HuggingFace utilise plusieurs fichiers, mais on simule
    torch.manual_seed(42)
    
    # Générer des poids aléatoires
    weights = {}
    
    # Couches d'experts
    for expert_idx in range(8):
        prefix = f"model.layers.0.experts.experts.{expert_idx}"
        
        # Matrices de poids
        weights[f"{prefix}.wq.weight"] = torch.randn(config["hidden_size"], config["hidden_size"])
        weights[f"{prefix}.wk.weight"] = torch.randn(config["hidden_size"], config["hidden_size"])
        weights[f"{prefix}.wv.weight"] = torch.randn(config["hidden_size"], config["hidden_size"])
        weights[f"{prefix}.wo.weight"] = torch.randn(config["hidden_size"], config["hidden_size"])
        
        # Bias
        weights[f"{prefix}.wq.bias"] = torch.randn(config["hidden_size"])
        weights[f"{prefix}.wk.bias"] = torch.randn(config["hidden_size"])
        weights[f"{prefix}.wv.bias"] = torch.randn(config["hidden_size"])
        weights[f"{prefix}.wo.bias"] = torch.randn(config["hidden_size"])
    
    # Couche de router
    weights["model.layers.0.experts.gate.weight"] = torch.randn(8, config["hidden_size"])
    
    # Embeddings
    weights["model.embed_tokens.weight"] = torch.randn(config["vocab_size"], config["hidden_size"])
    weights["model.norm.weight"] = torch.randn(config["hidden_size"])
    weights["lm_head.weight"] = torch.randn(config["hidden_size"], config["vocab_size"])
    
    # Sauvegarder
    torch.save(weights, weights_dir)
    
    # Créer un tokenizer simulé
    tokenizer_config = {
        "model_type": "mixtral",
        "vocab_size": config["vocab_size"],
        "tokenizer_class": "SentencePiece"
    }
    
    tokenizer_dir = os.path.join(output_dir, "tokenizer.json")
    with open(tokenizer_dir, "w") as f:
        json.dump(tokenizer_config, f, indent=2)
    
    # Créer un vocabulaire simulé
    vocab = {f"token_{i}": i for i in range(config["vocab_size"])}
    vocab["<unk>"] = config["vocab_size"]
    vocab["<s>"] = config["vocab_size"] + 1
    vocab["</s>"] = config["vocab_size"] + 2
    vocab["<pad>"] = config["vocab_size"] + 3
    
    vocab_path = os.path.join(output_dir, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2)
    
    print(f"✅ Modèle simulé créé: {output_dir}")
    print(f"   Taille: ~{os.path.getsize(weights_dir) / (1024**2):.1f} Mo")
    
    return True


def verify_model(output_dir: str) -> bool:
    """Vérifie qu'un modèle a été correctement téléchargé/créé."""
    required_files = ["config.json"]
    
    for f in required_files:
        if not os.path.exists(os.path.join(output_dir, f)):
            print(f"❌ Fichier manquant: {f}")
            return False
    
    # Vérifier la taille
    total_size = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, f))
    )
    
    print(f"✅ Modèle vérifié: {output_dir}")
    print(f"   Taille totale: {total_size / (1024**3):.2f} GB")
    
    return True


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Téléchargement des modèles pour le lab de benchmark"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="mistralai/Mixtral-8x7B-v0.1",
        choices=list(MODEL_CONFIGS.keys()),
        help="Modèle à télécharger"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="models/mixtral_8x7b_fp32",
        help="Répertoire de sortie"
    )
    
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Créer un modèle simulé au lieu de télécharger"
    )
    
    parser.add_argument(
        "--size",
        type=str,
        default="small",
        choices=["tiny", "small", "medium"],
        help="Taille du modèle simulé"
    )
    
    parser.add_argument(
        "--mirror",
        type=str,
        default=None,
        help="URL du miroir HuggingFace"
    )
    
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Forcer le téléchargement même si le répertoire existe"
    )
    
    args = parser.parse_args()
    
    # Vérifier si le modèle existe déjà
    if os.path.exists(args.output) and not args.force:
        print(f"⚠️  Le répertoire {args.output} existe déjà.")
        print("   Utilisez --force pour forcer le téléchargement.")
        
        if verify_model(args.output):
            print("   Le modèle semble valide.")
            sys.exit(0)
        else:
            print("   Le modèle semble incomplet.")
            sys.exit(1)
    
    # Vérifier l'espace disque
    required_gb = get_model_size(args.model) if not args.simulate else 1
    if not check_disk_space("/workspace", required_gb):
        print(f"❌ Espace disque insuffisant. {required_gb} GB requis.")
        sys.exit(1)
    
    # Vérifier huggingface-cli
    if not args.simulate and not check_huggingface_cli():
        print("❌ huggingface-cli non installé.")
        print("   Installez-le avec: pip install huggingface-hub")
        sys.exit(1)
    
    # Créer le répertoire de sortie
    create_directory(args.output)
    
    # Télécharger ou créer le modèle
    if args.simulate:
        success = create_simulated_model(args.output, args.size)
    else:
        success = download_from_huggingface(args.model, args.output)
    
    if not success:
        print("❌ Échec du téléchargement/création du modèle.")
        sys.exit(1)
    
    # Vérifier le modèle
    if not verify_model(args.output):
        print("❌ Le modèle téléchargé/créé est incomplet.")
        sys.exit(1)
    
    print("\n✅ Succès ! Le modèle est prêt pour la phase suivante.")
    print(f"   Emplacement: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
