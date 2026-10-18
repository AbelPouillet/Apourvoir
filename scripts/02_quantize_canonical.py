#!/usr/bin/env python3
"""
SCRIPT 2 : QUANTISATION CANONIQUE D'UN MODÈLE

Applique la quantisation canonique (sphérique par triplets, voir
apourvoir/quantization/canonical.py) à tous les poids >= 2-D d'un modèle et
écrit le résultat en safetensors. Les tenseurs 1-D sont copiés tels quels.

Les fonctions de quantisation viennent exclusivement du SDK `apourvoir` :
aucune redéfinition locale (l'ancienne version de ce script masquait l'import
du SDK par des copies locales divergentes).

Sorties dans --output :
- model_canonical.safetensors : codes, échelles et tenseurs non quantifiés
- quantization_metadata.json  : métadonnées par tenseur (pour déquantifier)
- compression_report.json     : tailles et erreurs mesurées

Exemple :
    python scripts/02_quantize_canonical.py --input TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output out/tinyllama_c8
"""

import os
import sys
import argparse
import json
import shlex
import time
from pathlib import Path
from typing import Dict, Tuple

import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from apourvoir.quantization import (
    quantifier_tensor_canonique,
    déquantifier_tensor_canonique,
    erreur_relative,
)
from apourvoir.quantization.model_bench import resolve_safetensors, iter_tensors, should_quantize

SCALES_SUFFIX = ".__canonical_scales"
CODES_SUFFIX = ".__canonical_codes"


def quantize_model(model: str, n_bits: int, validate: bool) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """Quantifie chaque tenseur ; retourne (tenseurs à sauvegarder, rapport)."""
    out: Dict[str, torch.Tensor] = {}
    layers = {}
    original_bytes = stored_bytes = numel = 0
    errors = []

    for name, tensor in iter_tensors(resolve_safetensors(model)):
        numel += tensor.numel()
        original_bytes += tensor.numel() * tensor.element_size()
        if not should_quantize(tensor):
            out[name] = tensor.contiguous()
            stored_bytes += tensor.numel() * tensor.element_size()
            continue

        codes, meta = quantifier_tensor_canonique(tensor, n_bits=n_bits)
        out[name + CODES_SUFFIX] = codes.contiguous()
        out[name + SCALES_SUFFIX] = meta.pop("scales").contiguous()
        stored_bytes += meta["stored_bytes"]
        meta["shape_original"] = list(meta["shape_original"])
        layers[name] = meta

        if validate:
            meta_full = dict(meta, scales=out[name + SCALES_SUFFIX], shape_original=tuple(meta["shape_original"]))
            err = erreur_relative(tensor, déquantifier_tensor_canonique(codes, meta_full))
            meta["rel_error"] = err
            errors.append((err, tensor.numel()))

    report = {
        "numel": numel,
        "original_bytes": original_bytes,
        "fp32_bytes": numel * 4,
        "stored_tensor_bytes": stored_bytes,
        "ratio_vs_fp32": numel * 4 / stored_bytes,
        "ratio_vs_original_dtype": original_bytes / stored_bytes,
        "weighted_rel_error": (sum(e * n for e, n in errors) / sum(n for _, n in errors)) if errors else None,
        "max_rel_error": max((e for e, _ in errors), default=None),
    }
    return out, {"n_bits": n_bits, "layers": layers, "report": report}


def load_quantized(output_dir: str) -> Dict[str, torch.Tensor]:
    """Recharge un modèle écrit par ce script et le déquantifie (float32)."""
    from safetensors.torch import load_file
    tensors = load_file(os.path.join(output_dir, "model_canonical.safetensors"))
    with open(os.path.join(output_dir, "quantization_metadata.json"), encoding="utf-8") as f:
        layers = json.load(f)["layers"]
    state = {}
    for name, tensor in tensors.items():
        if name.endswith(SCALES_SUFFIX):
            continue
        if name.endswith(CODES_SUFFIX):
            base = name[: -len(CODES_SUFFIX)]
            meta = dict(layers[base], scales=tensors[base + SCALES_SUFFIX],
                        shape_original=tuple(layers[base]["shape_original"]))
            state[base] = déquantifier_tensor_canonique(tensor, meta)
        else:
            state[name] = tensor
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantification canonique d'un modèle (safetensors)")
    parser.add_argument("--input", required=True, help="Répertoire local ou dépôt HuggingFace")
    parser.add_argument("--output", required=True, help="Répertoire de sortie")
    parser.add_argument("--n_bits", type=int, default=8, help="Bits par composante (2-16)")
    parser.add_argument("--no-validate", action="store_true", help="Ne pas mesurer l'erreur de reconstruction")
    args = parser.parse_args()

    from safetensors.torch import save_file

    command = "python " + " ".join(shlex.quote(a) for a in sys.argv)
    print(f"Commande : {command}")
    start = time.perf_counter()
    tensors, metadata = quantize_model(args.input, args.n_bits, validate=not args.no_validate)
    elapsed = time.perf_counter() - start

    Path(args.output).mkdir(parents=True, exist_ok=True)
    weights_path = os.path.join(args.output, "model_canonical.safetensors")
    save_file(tensors, weights_path)
    with open(os.path.join(args.output, "quantization_metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"n_bits": metadata["n_bits"], "layers": metadata["layers"]}, f, indent=2)

    report = metadata["report"]
    report["file_bytes"] = os.path.getsize(weights_path)
    report["ratio_file_vs_fp32"] = report["fp32_bytes"] / report["file_bytes"]
    report["quantization_seconds"] = elapsed
    report["command"] = command
    with open(os.path.join(args.output, "compression_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Taille d'origine          : {report['original_bytes'] / 1e9:.3f} GB")
    print(f"Taille FP32 équivalente   : {report['fp32_bytes'] / 1e9:.3f} GB")
    print(f"Fichier safetensors écrit : {report['file_bytes'] / 1e9:.3f} GB")
    print(f"Compression ratio: {report['ratio_file_vs_fp32']:.2f}x [MESURÉ] (fichier vs FP32)")
    if report["weighted_rel_error"] is not None:
        print(f"Erreur relative pondérée  : {report['weighted_rel_error']:.6f} [MESURÉ]")
    print(f"Temps                     : {elapsed:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
