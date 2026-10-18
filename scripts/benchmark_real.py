#!/usr/bin/env python3
"""
Benchmark réel de la quantisation canonique sur les poids d'un modèle HuggingFace.

Par défaut : mistralai/Mixtral-8x7B-v0.1 (93 Go de safetensors BF16, téléchargés
si absents du cache). Les poids sont lus tenseur par tenseur : ni le modèle
complet en RAM ni un GPU ne sont nécessaires (--device cuda est possible).

Chaque chiffre affiché est calculé pendant l'exécution :
- taille d'origine = octets des tenseurs safetensors (BF16 pour Mixtral) ;
- taille FP32 = nombre de paramètres * 4 ;
- taille compressée = octets des codes + échelles réellement produits
  (les tenseurs 1-D non quantifiés comptent à leur taille d'origine) ;
- erreur relative / cosinus = original vs reconstruit, pondérés par numel.

Exemples :
    python scripts/benchmark_real.py
    python scripts/benchmark_real.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --n-bits 4
    python scripts/benchmark_real.py --max-tensors 20          # essai rapide
"""

import argparse
import json
import os
import platform
import shlex
import sys
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))

import torch

from apourvoir.quantization.model_bench import measure_model


def gb(n: int) -> str:
    return f"{n / 1e9:.3f} GB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="mistralai/Mixtral-8x7B-v0.1", help="Dépôt HF ou répertoire local")
    parser.add_argument("--n-bits", type=int, default=8, help="Bits par composante (r, theta, phi)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-tensors", type=int, default=None, help="Limiter le nombre de tenseurs (essai partiel)")
    parser.add_argument("--filter", default=None, help="Ne traiter que les tenseurs dont le nom contient ce texte")
    parser.add_argument("--no-error", action="store_true", help="Ne pas mesurer l'erreur (plus rapide)")
    parser.add_argument("--output", default=None, help="Fichier JSON de résultats (défaut: results/benchmark_<modèle>_canonical<N>.json)")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA indisponible ; utilisez --device cpu")
        return 1

    command = "python " + " ".join(shlex.quote(a) for a in sys.argv)
    method = f"canonical{args.n_bits}"
    print("=" * 78)
    print(f"Commande : {command}")
    print(f"Modèle   : {args.model}")
    print(f"Méthode  : {method} (sphérique par triplets, {args.n_bits} bits/composante)")
    print(f"Machine  : {platform.platform()} | torch {torch.__version__} | device={args.device}")
    print("=" * 78)

    start = time.perf_counter()
    last = [start]

    def progress(i, r):
        now = time.perf_counter()
        if now - last[0] > 30:
            last[0] = now
            print(f"  [{now - start:7.0f}s] {i + 1} tenseurs traités (dernier: {r.name})", flush=True)

    result = measure_model(
        args.model, method, device=args.device, max_tensors=args.max_tensors,
        name_filter=args.filter, measure_error=not args.no_error, progress=progress,
    )
    elapsed = time.perf_counter() - start
    s = result.summary()
    partial = args.max_tensors is not None or args.filter is not None

    print()
    print("RÉSULTATS [MESURÉ]" + (" — PARTIEL (sous-ensemble de tenseurs)" if partial else ""))
    print("-" * 78)
    print(f"Fichiers lus                 : {len(result.files)}")
    print(f"Tenseurs (quantifiés/total)  : {s['num_quantized_tensors']}/{s['num_tensors']}")
    print(f"Paramètres                   : {s['numel']:,}")
    print(f"Taille d'origine (safetensors): {gb(s['original_bytes'])}")
    print(f"Taille équivalente FP32      : {gb(s['fp32_bytes'])}")
    print(f"Taille compressée            : {gb(s['stored_bytes'])}")
    print(f"Bits par paramètre           : {s['bits_per_param']:.4f}")
    print(f"Compression ratio: {s['ratio_vs_fp32']:.2f}x [MESURÉ] (vs FP32)")
    print(f"Compression ratio: {s['ratio_vs_original_dtype']:.2f}x [MESURÉ] (vs dtype d'origine)")
    if s["weighted_rel_error"] is not None:
        print(f"Erreur relative (pondérée)   : {s['weighted_rel_error']:.6f} [MESURÉ]")
        print(f"Erreur relative max          : {s['max_rel_error']:.6f} [MESURÉ]")
        print(f"Cosinus moyen (pondéré)      : {s['weighted_cosine']:.6f} [MESURÉ]")
        print(f"Cosinus min                  : {s['min_cosine']:.6f} [MESURÉ]")
    print(f"Temps quantisation           : {s['quant_seconds']:.1f} s")
    print(f"Temps total (lecture incluse): {elapsed:.1f} s")
    print("-" * 78)

    output = args.output or os.path.join(
        "results", f"benchmark_{args.model.replace('/', '__')}_{method}{'_partial' if partial else ''}.json"
    )
    result.save(output)
    with open(output, "r+", encoding="utf-8") as f:
        data = json.load(f)
        data["command"] = command
        data["elapsed_seconds"] = elapsed
        data["machine"] = {"platform": platform.platform(), "torch": torch.__version__, "device": args.device}
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()
    print(f"Détails par tenseur : {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
