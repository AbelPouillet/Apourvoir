#!/bin/bash
# Lance toutes les mesures réelles. Chaque script affiche sa commande et ses résultats [MESURÉ].
# Usage : LLAMA_CPP_DIR=/chemin/vers/llama.cpp ./scripts/run_all_benchmarks.sh [modèle_petit]
set -e
cd "$(dirname "$0")/.."
SMALL_MODEL="${1:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"

python scripts/validate_quantization.py --model "$SMALL_MODEL"
python scripts/compare_baselines.py --model "$SMALL_MODEL" --model-level
python scripts/benchmark_real.py --model mistralai/Mixtral-8x7B-v0.1
python scripts/compare_baselines.py --model mistralai/Mixtral-8x7B-v0.1
