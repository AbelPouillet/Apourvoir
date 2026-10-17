#!/bin/bash

# 🚀 SCRIPT PRINCIPAL : LANCEUR DE TOUS LES BENCHMARKS
# Exécute tous les benchmarks dans l'ordre pour valider la quantisation canonique

set -e  # Arrêter en cas d'erreur

# ============================================================================
# CONFIGURATION
# ============================================================================

# Répertoires
BASE_DIR="/workspace/lab_bench"
MODELS_DIR="$BASE_DIR/models"
SCRIPTS_DIR="$BASE_DIR/scripts"
RESULTS_DIR="$BASE_DIR/results"
LOGS_DIR="$BASE_DIR/logs"

# Noms des modèles
MODEL_FP32="mixtral_8x7b_fp32"
MODEL_QUANT="mixtral_8x7b_canonical"

# Device (auto-détection)
if command -v nvidia-smi &> /dev/null; then
    DEVICE="cuda"
    echo "🎉 GPU détecté - Utilisation de CUDA"
else
    DEVICE="cpu"
    echo "⚠️  Pas de GPU détecté - Utilisation de CPU (plus lent)"
fi

# ============================================================================
# FONCTIONS AUXILIAIRES
# ============================================================================

# Affiche un message avec une couleur
function echo_color() {
    case $1 in
        "green") echo -e "\033[1;32m$2\033[0m" ;;
        "blue") echo -e "\033[1;34m$2\033[0m" ;;
        "yellow") echo -e "\033[1;33m$2\033[0m" ;;
        "red") echo -e "\033[1;31m$2\033[0m" ;;
        *) echo "$2" ;;
    esac
}

# Vérifie si un répertoire existe
function check_dir() {
    if [ ! -d "$1" ]; then
        echo_color "red" "❌ Répertoire introuvable: $1"
        exit 1
    fi
}

# ============================================================================
# ÉTAPE 0 : VÉRIFICATION DE L'ENVIRONNEMENT
# ============================================================================

echo_color "blue" "================================================================================"
echo_color "blue" "🔍 ÉTAPE 0 : Vérification de l'environnement"
echo_color "blue" "================================================================================"

# Vérifier Python
if ! command -v python3 &> /dev/null; then
    echo_color "red" "❌ Python 3 non trouvé"
    exit 1
fi

echo_color "green" "✅ Python 3 trouvé"

# Vérifier pip
if ! command -v pip3 &> /dev/null; then
    echo_color "red" "❌ pip 3 non trouvé"
    exit 1
fi

echo_color "green" "✅ pip 3 trouvé"

# Vérifier les dépendances
echo -n "📦 Vérification des dépendances... "
python3 -c "import torch; import transformers; import numpy; import matplotlib" 2> /dev/null
if [ $? -eq 0 ]; then
    echo_color "green" "✅ Toutes les dépendances sont installées"
else
    echo_color "yellow" "⚠️  Certaines dépendances manquent"
    echo_color "yellow" "   Installation en cours..."
    pip3 install torch transformers numpy matplotlib scipy tqdm psutil -q
    echo_color "green" "✅ Dépendances installées"
fi

# Créer les répertoires
mkdir -p "$MODELS_DIR" "$RESULTS_DIR" "$LOGS_DIR"
echo_color "green" "✅ Répertoires créés"

# ============================================================================
# ÉTAPE 1 : TÉLÉCHARGEMENT DES MODÈLES
# ============================================================================

echo ""
echo_color "blue" "================================================================================"
echo_color "blue" "📥 ÉTAPE 1 : Téléchargement des modèles"
echo_color "blue" "================================================================================"

# Vérifier si le modèle FP32 existe déjà
if [ -d "$MODELS_DIR/$MODEL_FP32" ]; then
    echo_color "yellow" "⚠️  Modèle FP32 déjà présent - On passe"
else
    echo_color "yellow" "⚠️  Téléchargement du vrai Mixtral-8x7B (47 GB) - Cela peut prendre du temps"
    echo_color "yellow" "   Pour un test rapide, utilisez --simulate"
    
    read -p "Voulez-vous télécharger le vrai modèle? (o/n) [n]: " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Oo]$ ]]; then
        echo_color "blue" "📥 Téléchargement de mistralai/Mixtral-8x7B-v0.1..."
        python3 "$SCRIPTS_DIR/01_download_models.py" \
            --model mistralai/Mixtral-8x7B-v0.1 \
            --output "$MODELS_DIR/$MODEL_FP32" \
            --log_dir "$LOGS_DIR/download.log"
    else
        echo_color "blue" "🎭 Création d'un modèle simulé pour test rapide..."
        python3 "$SCRIPTS_DIR/01_download_models.py" \
            --simulate \
            --size small \
            --output "$MODELS_DIR/$MODEL_FP32" \
            --log_dir "$LOGS_DIR/download.log"
    fi
fi

check_dir "$MODELS_DIR/$MODEL_FP32"
echo_color "green" "✅ Modèle FP32 prêt"

# ============================================================================
# ÉTAPE 2 : QUANTISATION CANONIQUE
# ============================================================================

echo ""
echo_color "blue" "================================================================================"
echo_color "blue" "🔧 ÉTAPE 2 : Quantisation Canonique"
echo_color "blue" "================================================================================"

# Vérifier si le modèle quantifié existe déjà
if [ -d "$MODELS_DIR/$MODEL_QUANT" ]; then
    echo_color "yellow" "⚠️  Modèle quantifié déjà présent - On passe"
else
    echo_color "blue" "🔧 Application de la quantisation canonique..."
    python3 "$SCRIPTS_DIR/02_quantize_canonical.py" \
        --input "$MODELS_DIR/$MODEL_FP32" \
        --output "$MODELS_DIR/$MODEL_QUANT" \
        --n_bits 8 \
        --method canonical_torus \
        --device "$DEVICE" \
        --log_dir "$LOGS_DIR/quantization.log" \
        --validate
    
    check_dir "$MODELS_DIR/$MODEL_QUANT"
fi

echo_color "green" "✅ Modèle quantifié prêt"

# Afficher les tailles
FP32_SIZE=$(du -sh "$MODELS_DIR/$MODEL_FP32" | awk '{print $1}')
QUANT_SIZE=$(du -sh "$MODELS_DIR/$MODEL_QUANT" | awk '{print $1}')
echo_color "green" "   Taille FP32: $FP32_SIZE"
echo_color "green" "   Taille Quantifiée: $QUANT_SIZE"

# ============================================================================
# ÉTAPE 3 : BENCHMARK D'INFÉRENCE
# ============================================================================

echo ""
echo_color "blue" "================================================================================"
echo_color "blue" "⚡ ÉTAPE 3 : Benchmark d'Inférence"
echo_color "blue" "================================================================================"

echo_color "blue" "⏳ Benchmark en cours (cela peut prendre quelques minutes)..."
python3 "$SCRIPTS_DIR/03_benchmark_inference.py" \
    --model_fp32 "$MODELS_DIR/$MODEL_FP32" \
    --model_quant "$MODELS_DIR/$MODEL_QUANT" \
    --output "$RESULTS_DIR/inference_benchmark.json" \
    --batch_size 1 4 8 16 \
    --seq_length 64 128 256 \
    --num_prompts 5 \
    --num_runs 3 \
    --device "$DEVICE" \
    --log_dir "$LOGS_DIR/inference.log"

echo_color "green" "✅ Benchmark d'inférence terminé"

# Afficher un résumé
INFERENCE_SPEEDUP=$(jq '.summary.global.speedup' "$RESULTS_DIR/inference_benchmark.json" 2> /dev/null || echo "N/A")
echo_color "green" "   Speedup global: ${INFERENCE_SPEEDUP}×"

# ============================================================================
# ÉTAPE 4 : BENCHMARK MÉMOIRE
# ============================================================================

echo ""
echo_color "blue" "================================================================================"
echo_color "blue" "💾 ÉTAPE 4 : Benchmark Mémoire"
echo_color "blue" "================================================================================"

echo_color "blue" "💾 Mesure de l'utilisation mémoire..."
python3 "$SCRIPTS_DIR/04_benchmark_memory.py" \
    --model_fp32 "$MODELS_DIR/$MODEL_FP32" \
    --model_quant "$MODELS_DIR/$MODEL_QUANT" \
    --output "$RESULTS_DIR/memory_benchmark.json" \
    --device "$DEVICE" \
    --log_dir "$LOGS_DIR/memory.log"

echo_color "green" "✅ Benchmark mémoire terminé"

# Afficher un résumé
MEMORY_RATIO=$(jq '.summary.disk_compression' "$RESULTS_DIR/memory_benchmark.json" 2> /dev/null || echo "N/A")
echo_color "green" "   Ratio de compression disque: ${MEMORY_RATIO}×"

# ============================================================================
# ÉTAPE 5 : VALIDATION SÉMANTIQUE
# ============================================================================

echo ""
echo_color "blue" "================================================================================"
echo_color "blue" "🔍 ÉTAPE 5 : Validation Sémantique"
echo_color "blue" "================================================================================"

echo_color "blue" "🔍 Validation de la qualité des réponses..."
python3 "$SCRIPTS_DIR/05_validate_semantics.py" \
    --model_fp32 "$MODELS_DIR/$MODEL_FP32" \
    --model_quant "$MODELS_DIR/$MODEL_QUANT" \
    --output "$RESULTS_DIR/semantic_validation.json" \
    --dataset default \
    --num_questions 10 \
    --device "$DEVICE" \
    --log_dir "$LOGS_DIR/semantic.log"

echo_color "green" "✅ Validation sémantique terminée"

# Afficher un résumé
FP32_ACCURACY=$(jq '.metrics.fp32.accuracy' "$RESULTS_DIR/semantic_validation.json" 2> /dev/null || echo "N/A")
QUANT_ACCURACY=$(jq '.metrics.quantized.accuracy' "$RESULTS_DIR/semantic_validation.json" 2> /dev/null || echo "N/A")
echo_color "green" "   Précision FP32: ${FP32_ACCURACY}"
echo_color "green" "   Précision Quantifiée: ${QUANT_ACCURACY}"

# ============================================================================
# ÉTAPE 6 : GÉNÉRATION DU RAPPORT FINAL
# ============================================================================

echo ""
echo_color "blue" "================================================================================"
echo_color "blue" "📊 ÉTAPE 6 : Génération du Rapport Final"
echo_color "blue" "================================================================================"

# Créer un script Python pour générer le rapport
cat > "$SCRIPTS_DIR/06_generate_report.py" << 'EOF'
#!/usr/bin/env python3
"""Génère un rapport final à partir de tous les benchmarks."""

import os
import json
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="results/", help="Répertoire des résultats")
    parser.add_argument("--output", default="results/compression_report.json", help="Fichier de sortie")
    args = parser.parse_args()
    
    # Charger tous les rapports
    reports = {}
    
    for filename in ["inference_benchmark.json", "memory_benchmark.json", "semantic_validation.json"]:
        filepath = os.path.join(args.input_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                reports[filename.replace(".json", "")] = json.load(f)
    
    # Générer le rapport final
    rapport = {
        "date": datetime.now().isoformat(),
        "summary": {}
    }
    
    # Ajouter les informations de chaque benchmark
    if "inference_benchmark" in reports:
        rapport["inference"] = reports["inference_benchmark"]["summary"]["global"]
        rapport["summary"]["speedup"] = reports["inference_benchmark"]["summary"]["global"]["speedup"]
    
    if "memory_benchmark" in reports:
        rapport["memory"] = reports["memory_benchmark"]["summary"]
        rapport["summary"]["compression_ratio"] = reports["memory_benchmark"]["summary"]["disk_compression"]
    
    if "semantic_validation" in reports:
        rapport["semantics"] = {
            "fp32": reports["semantic_validation"]["metrics"]["fp32"],
            "quantized": reports["semantic_validation"]["metrics"]["quantized"]
        }
        rapport["summary"]["accuracy_drop"] = (
            reports["semantic_validation"]["metrics"]["fp32"]["accuracy"] -
            reports["semantic_validation"]["metrics"]["quantized"]["accuracy"]
        )
    
    # Calculer le score global
    score = 0
    weights = {"speedup": 0.3, "compression_ratio": 0.4, "accuracy_drop": 0.3}
    
    if "speedup" in rapport["summary"]:
        score += rapport["summary"]["speedup"] * weights["speedup"]
    if "compression_ratio" in rapport["summary"]:
        score += rapport["summary"]["compression_ratio"] * weights["compression_ratio"]
    if "accuracy_drop" in rapport["summary"]:
        # Moins de drop = meilleur score
        score += (1 - abs(rapport["summary"]["accuracy_drop"])) * weights["accuracy_drop"]
    
    rapport["summary"]["global_score"] = score
    
    # Sauvegarder
    with open(args.output, "w") as f:
        json.dump(rapport, f, indent=2)
    
    print(f"✅ Rapport final sauvegardé: {args.output}")
    print(f"   Score global: {score:.2f}")

if __name__ == "__main__":
    main()
EOF

python3 "$SCRIPTS_DIR/06_generate_report.py" \
    --input_dir "$RESULTS_DIR" \
    --output "$RESULTS_DIR/compression_report.json"

echo_color "green" "✅ Rapport final généré"

# ============================================================================
# ÉTAPE 7 : AFFICHAGE DU RAPPORT FINAL
# ============================================================================

echo ""
echo_color "blue" "================================================================================"
echo_color "blue" "🎉 RAPPORT FINAL"
echo_color "blue" "================================================================================"

echo ""
echo_color "green" "📊 RÉSULTATS CONCRETS:"
echo ""

# Lire le rapport final
if [ -f "$RESULTS_DIR/compression_report.json" ]; then
    COMPRESSION_RATIO=$(jq '.summary.compression_ratio' "$RESULTS_DIR/compression_report.json" 2> /dev/null || echo "N/A")
    SPEEDUP=$(jq '.summary.speedup' "$RESULTS_DIR/compression_report.json" 2> /dev/null || echo "N/A")
    ACCURACY_DROP=$(jq '.summary.accuracy_drop' "$RESULTS_DIR/compression_report.json" 2> /dev/null || echo "N/A")
    GLOBAL_SCORE=$(jq '.summary.global_score' "$RESULTS_DIR/compression_report.json" 2> /dev/null || echo "N/A")
    
    echo_color "yellow" "  Ratio de Compression: ${COMPRESSION_RATIO}×"
    echo_color "yellow" "  Speedup d'Inférence: ${SPEEDUP}×"
    echo_color "yellow" "  Drop de Précision: ${ACCURACY_DROP} (0 = parfait)"
    echo_color "yellow" "  Score Global: ${GLOBAL_SCORE}/1.0"
    echo ""
    
    # Vérifier si on a atteint les objectifs
    if [ $(echo "$COMPRESSION_RATIO > 15" | bc) -eq 1 ]; then
        echo_color "green" "  ✅ Objectif de compression (15×) ATTEINT"
    else
        echo_color "red" "  ❌ Objectif de compression (15×) NON ATTEINT"
    fi
    
    if [ $(echo "$SPEEDUP > 1.3" | bc) -eq 1 ]; then
        echo_color "green" "  ✅ Objectif de vitesse (1.3×) ATTEINT"
    else
        echo_color "red" "  ❌ Objectif de vitesse (1.3×) NON ATTEINT"
    fi
    
    if [ $(echo "$ACCURACY_DROP < 0.05" | bc) -eq 1 ]; then
        echo_color "green" "  ✅ Objectif de précision (<5% drop) ATTEINT"
    else
        echo_color "red" "  ❌ Objectif de précision (<5% drop) NON ATTEINT"
    fi
else
    echo_color "red" "  ❌ Rapport final introuvable"
fi

echo ""
echo_color "green" "================================================================================"
echo_color "green" "✅ TOUS LES BENCHMARKS TERMINÉS AVEC SUCCÈS !"
echo_color "green" "================================================================================"
echo ""
echo_color "blue" "📁 Fichiers générés:"
echo_color "blue" "   - $RESULTS_DIR/inference_benchmark.json"
echo_color "blue" "   - $RESULTS_DIR/memory_benchmark.json"
echo_color "blue" "   - $RESULTS_DIR/semantic_validation.json"
echo_color "blue" "   - $RESULTS_DIR/compression_report.json"
echo ""
echo_color "blue" "📊 Visualisez les résultats avec:"
echo_color "blue" "   cat $RESULTS_DIR/compression_report.json | jq ."
echo ""
echo_color "blue" "🎯 Pour aller plus loin:"
echo_color "blue" "   - Testez avec le vrai Mixtral-8x7B (47 GB)"
echo_color "blue" "   - Ajustez les paramètres de quantisation (n_bits, R_MAJEUR, etc.)"
echo_color "blue" "   - Comparez avec d'autres méthodes (GGUF, INT8, etc.)"
