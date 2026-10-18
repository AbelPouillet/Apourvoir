#!/usr/bin/env python3
"""
Validation roundtrip : quantize -> dequantize -> erreur.

1. Tenseurs synthétiques (graine fixe) de dimensions 3, 4096, 14336 et une
   distribution à queue lourde (Student t, 3 ddl), pour 2/4/6/8/16 bits et les
   deux encodages du rayon (lin / sqrt).
2. Comparaison avec la réduction PCA à 3 composantes (option suggérée
   initialement) : erreur de reconstruction d'une projection de rang 3.
3. Optionnel (--model) : premiers tenseurs 2-D d'un vrai modèle safetensors.

Exemple :
    python scripts/validate_quantization.py
    python scripts/validate_quantization.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --max-tensors 10
"""

import argparse
import os
import shlex
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))

import torch

from apourvoir.quantization import quantifier_tensor_canonique, déquantifier_tensor_canonique, erreur_relative
from apourvoir.quantization.baselines import BASELINES


def synthetic():
    gen = torch.Generator().manual_seed(0)
    yield "randn(100, 3)", torch.randn(100, 3, generator=gen)
    yield "randn(100, 4096)", torch.randn(100, 4096, generator=gen)
    yield "randn(100, 14336)", torch.randn(100, 14336, generator=gen)
    torch.manual_seed(0)
    yield "student_t3(100, 4096)", torch.distributions.StudentT(3.0).sample((100, 4096))


def pca_rank3_error(x: torch.Tensor) -> float:
    """Erreur relative de la meilleure reconstruction de rang 3 (SVD tronquée, centrée)."""
    mean = x.mean(dim=0, keepdim=True)
    u, s, v = torch.linalg.svd(x - mean, full_matrices=False)
    k = min(3, s.numel())
    recon = (u[:, :k] * s[:k]) @ v[:k] + mean
    return erreur_relative(x, recon)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-tensors", type=int, default=10)
    args = parser.parse_args()
    print("Commande : python " + " ".join(shlex.quote(a) for a in sys.argv))

    print("\n1. Roundtrip canonique sur tenseurs synthétiques [MESURÉ]")
    print(f"{'tenseur':<24}{'bits':>5}{'r_mode':>7}{'err rel.':>10}{'bits/val':>10}{'vs FP32':>9}")
    for label, x in synthetic():
        for n_bits in (16, 8, 6, 4, 2):
            for r_mode in ("sqrt", "lin"):
                q, m = quantifier_tensor_canonique(x, n_bits=n_bits, r_mode=r_mode)
                err = erreur_relative(x, déquantifier_tensor_canonique(q, m))
                print(f"{label:<24}{n_bits:>5}{r_mode:>7}{err:>10.5f}{m['bits_per_value']:>10.3f}{m['compression_ratio']:>8.2f}x")

    print("\n2. Baselines au même budget de bits [MESURÉ]")
    for label, x in synthetic():
        row = []
        for name, (quant, dequant) in BASELINES.items():
            codes, meta = quant(x)
            row.append(f"{name}={erreur_relative(x, dequant(codes, meta)):.5f}")
        print(f"{label:<24}" + "  ".join(row))

    print("\n3. Réduction PCA à 3 composantes (reconstruction de rang 3) [MESURÉ]")
    for label, x in synthetic():
        print(f"{label:<24}err rel. = {pca_rank3_error(x):.5f}")

    if args.model:
        from apourvoir.quantization.model_bench import resolve_safetensors, iter_tensors, should_quantize
        print(f"\n4. Tenseurs réels : {args.model} [MESURÉ]")
        count = 0
        for name, t in iter_tensors(resolve_safetensors(args.model)):
            if not should_quantize(t):
                continue
            q, m = quantifier_tensor_canonique(t, n_bits=8)
            err = erreur_relative(t, déquantifier_tensor_canonique(q, m))
            pca = pca_rank3_error(t.float()) if t.shape[0] <= 8192 else float("nan")
            print(f"{name:<48}{str(tuple(t.shape)):>16}  canonical8={err:.5f}  pca3={pca:.5f}")
            count += 1
            if count >= args.max_tensors:
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
