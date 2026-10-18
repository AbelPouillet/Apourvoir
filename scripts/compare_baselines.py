#!/usr/bin/env python3
"""
Comparaison honnête de la quantisation canonique avec des baselines.

Toutes les méthodes sont appliquées aux MÊMES poids, sur la MÊME machine.

1. Poids (toutes tailles de modèle, lecture tenseur par tenseur) :
   - canonical8 / canonical4 : méthode canonique (sphérique par triplets)
   - int8_per_row            : INT8 symétrique par ligne (PyTorch)
   - q8_0_ref / q4_0_ref     : réimplémentations PyTorch des formats GGML
   - gguf:Q8_0 / gguf:Q4_K_M : vrais GGUF produits par llama.cpp
     (convert_hf_to_gguf.py -> BF16, puis llama-quantize). L'erreur est mesurée
     en déquantifiant chaque tenseur (gguf-py) et en le comparant au GGUF BF16
     (copie sans perte des poids BF16 d'origine, même disposition).
   - awq : mesuré seulement si le paquet `awq` et CUDA sont disponibles,
     sinon reporté [NON MESURÉ] avec la raison.
   Mesures : octets stockés, ratio vs FP32 et vs dtype d'origine, cosinus et
   erreur relative pondérés par le nombre de paramètres.

2. Modèle (--model-level, seulement si le modèle tient en RAM en FP32) :
   - perplexité sur wikitext-2 (test), N chunks de 512 tokens :
     transformers FP32 (référence) vs poids quantifiés-déquantifiés,
     et llama-perplexity pour les GGUF (référence : GGUF BF16) ;
   - vitesse : llama-bench pour les GGUF ; transformers FP32 pour les méthodes
     PyTorch (aucun noyau bas-bit n'existe : l'inférence se fait sur les poids
     déquantifiés, donc à la vitesse FP32).

Exemples :
    python scripts/compare_baselines.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --model-level \
        --llama-cpp-dir /chemin/vers/llama.cpp
    python scripts/compare_baselines.py --model mistralai/Mixtral-8x7B-v0.1 --llama-cpp-dir /chemin/vers/llama.cpp
"""

import argparse
import importlib.util
import io
import json
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))

import numpy as np
import torch

from apourvoir.quantization.model_bench import (
    get_method, iter_tensors, measure_model, resolve_safetensors, should_quantize,
)

DEFAULT_METHODS = "canonical8,canonical4,int8_per_row,q8_0_ref,q4_0_ref,gguf:Q8_0,gguf:Q4_K_M,awq"
WIKITEXT_URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"


def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: List[str], env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    log("  $ " + " ".join(shlex.quote(str(c)) for c in cmd))
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Échec ({proc.returncode}) : {' '.join(map(str, cmd))}\n{proc.stderr[-3000:]}")
    return proc


def find_exe(name: str) -> Optional[str]:
    return shutil.which(name) or shutil.which(name + ".exe")


# ============================================================================
# GGUF (llama.cpp)
# ============================================================================

class GGUFTools:
    def __init__(self, llama_cpp_dir: Optional[str], work_dir: Path):
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.quantize_exe = find_exe("llama-quantize")
        self.perplexity_exe = find_exe("llama-perplexity")
        self.bench_exe = find_exe("llama-bench")
        self.convert_script = None
        self.env = dict(os.environ, PYTHONIOENCODING="utf-8")
        if llama_cpp_dir:
            script = Path(llama_cpp_dir) / "convert_hf_to_gguf.py"
            if script.exists():
                self.convert_script = script
                gguf_py = Path(llama_cpp_dir) / "gguf-py"
                if gguf_py.exists():
                    self.env["PYTHONPATH"] = str(gguf_py) + os.pathsep + self.env.get("PYTHONPATH", "")
                    sys.path.insert(0, str(gguf_py))

    def unavailable_reason(self) -> Optional[str]:
        if not self.quantize_exe:
            return "llama-quantize introuvable dans le PATH"
        if not self.convert_script:
            return "convert_hf_to_gguf.py introuvable (--llama-cpp-dir)"
        if importlib.util.find_spec("gguf") is None:
            return "paquet python gguf absent"
        return None

    def version(self) -> str:
        exe = find_exe("llama-cli")
        if not exe:
            return "inconnue"
        proc = subprocess.run([exe, "--version"], capture_output=True, text=True, errors="replace")
        m = re.search(r"version: (\S+ \(\w+\))", proc.stdout + proc.stderr)
        return m.group(1) if m else "inconnue"

    def path(self, tag: str, qtype: str) -> Path:
        return self.work_dir / f"{tag}-{qtype}.gguf"

    def convert_bf16(self, model_dir: Path, tag: str) -> Path:
        out = self.path(tag, "BF16")
        if not out.exists():
            run([sys.executable, self.convert_script, model_dir, "--outtype", "bf16", "--outfile", out], env=self.env)
        return out

    def quantize(self, src: Path, tag: str, qtype: str, threads: int) -> Path:
        out = self.path(tag, qtype)
        if not out.exists():
            run([self.quantize_exe, src, out, qtype, str(threads)])
        return out


def _gguf_to_float64(data: np.ndarray, tensor_type) -> np.ndarray:
    """Déquantifie des données GGUF brutes (tenseur ou tranche) en float64 aplati."""
    import gguf
    from gguf.constants import GGMLQuantizationType as T
    if tensor_type in (T.F32, T.F16):
        return np.asarray(data).astype(np.float64).ravel()
    return gguf.quants.dequantize(np.asarray(data), tensor_type).astype(np.float64).ravel()


def compare_gguf(ref_path: Path, quant_path: Path) -> Dict:
    """Compare chaque tenseur d'un GGUF quantifié au GGUF BF16 de référence."""
    import gguf
    ref = {t.name: t for t in gguf.GGUFReader(str(ref_path)).tensors}
    qnt = gguf.GGUFReader(str(quant_path))
    numel_total = 0
    stored = 0
    w_cos = w_err = 0.0
    numel_q = 0
    min_cos, max_err = 1.0, 0.0
    types: Dict[str, int] = {}
    for t in qnt.tensors:
        n = int(t.n_elements)
        numel_total += n
        stored += int(t.n_bytes)
        types[t.tensor_type.name] = types.get(t.tensor_type.name, 0) + 1
        r = ref[t.name]
        if len(t.shape) < 2:
            continue
        if r.tensor_type == t.tensor_type:
            # Tenseur 2-D laissé tel quel par llama-quantize : reconstruction exacte.
            w_cos += n
            numel_q += n
            continue
        # Dé-quantification par tranches (experts MoE : [n_expert, ...])
        sq = dot = rr = qq = 0.0
        q_data, r_data = t.data, r.data
        slices = range(q_data.shape[0]) if q_data.ndim == 3 else [None]
        for i in slices:
            qd = q_data if i is None else q_data[i]
            rd = r_data if i is None else r_data[i]
            a = _gguf_to_float64(rd, r.tensor_type)
            b = _gguf_to_float64(qd, t.tensor_type)
            sq += float(np.dot(a - b, a - b))
            dot += float(np.dot(a, b))
            rr += float(np.dot(a, a))
            qq += float(np.dot(b, b))
        cos = dot / math.sqrt(rr * qq) if rr > 0 and qq > 0 else 1.0
        err = math.sqrt(sq / rr) if rr > 0 else 0.0
        w_cos += cos * n
        w_err += err * n
        numel_q += n
        min_cos, max_err = min(min_cos, cos), max(max_err, err)
    return {
        "numel": numel_total,
        "tensor_data_bytes": stored,
        "file_bytes": quant_path.stat().st_size,
        "weighted_cosine": w_cos / numel_q if numel_q else None,
        "weighted_rel_error": w_err / numel_q if numel_q else None,
        "min_cosine": min_cos,
        "max_rel_error": max_err,
        "tensor_types": types,
    }


# ============================================================================
# NIVEAU MODÈLE : perplexité et vitesse
# ============================================================================

def wikitext_test(cache_dir: Path) -> Path:
    out = cache_dir / "wiki.test.raw"
    if not out.exists():
        log(f"  téléchargement {WIKITEXT_URL}")
        with urllib.request.urlopen(WIKITEXT_URL) as resp:
            archive = zipfile.ZipFile(io.BytesIO(resp.read()))
        name = next(n for n in archive.namelist() if n.endswith("wiki.test.raw"))
        out.write_bytes(archive.read(name))
    return out


def hf_perplexity(model, input_ids: torch.Tensor, ctx: int, chunks: int) -> float:
    """Perplexité sur `chunks` fenêtres disjointes de `ctx` tokens (toutes positions sauf la 1re)."""
    nll, count = 0.0, 0
    with torch.no_grad():
        for c in range(chunks):
            ids = input_ids[:, c * ctx:(c + 1) * ctx]
            if ids.shape[1] < ctx:
                break
            out = model(ids, labels=ids)
            nll += out.loss.item() * (ctx - 1)
            count += ctx - 1
    return math.exp(nll / count)


def hf_generation_speed(model, tokenizer, new_tokens: int = 32) -> float:
    ids = tokenizer("The capital of France is", return_tensors="pt").input_ids
    with torch.no_grad():
        model.generate(ids, max_new_tokens=4, do_sample=False)
        t0 = time.perf_counter()
        out = model.generate(ids, max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False)
        dt = time.perf_counter() - t0
    return (out.shape[1] - ids.shape[1]) / dt


def model_level_hf(model_dir: Path, methods: List[str], text_path: Path, ctx: int, chunks: int) -> Dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    model.eval()
    input_ids = tok(text_path.read_text(encoding="utf-8"), return_tensors="pt").input_ids

    # Poids d'origine (dtype du checkpoint) pour repartir de la même source à chaque méthode
    original = dict(iter_tensors(resolve_safetensors(str(model_dir))))
    params = dict(model.named_parameters())

    results = {"reference": "transformers FP32", "ctx": ctx, "chunks": chunks, "methods": {}}
    t0 = time.perf_counter()
    results["fp32_perplexity"] = hf_perplexity(model, input_ids, ctx, chunks)
    results["fp32_ppl_seconds"] = time.perf_counter() - t0
    results["fp32_tokens_per_second"] = hf_generation_speed(model, tok)
    log(f"  FP32 : ppl={results['fp32_perplexity']:.4f}, génération={results['fp32_tokens_per_second']:.2f} tok/s")

    for method in methods:
        quant, dequant = get_method(method)
        t0 = time.perf_counter()
        with torch.no_grad():
            for name, tensor in original.items():
                if name in params and should_quantize(tensor):
                    codes, meta = quant(tensor)
                    params[name].data.copy_(dequant(codes, meta))
        dq_seconds = time.perf_counter() - t0
        ppl = hf_perplexity(model, input_ids, ctx, chunks)
        results["methods"][method] = {"perplexity": ppl, "quant_dequant_seconds": dq_seconds}
        log(f"  {method} : ppl={ppl:.4f} (quantif.+déquantif. {dq_seconds:.1f}s)")
        with torch.no_grad():
            for name, tensor in original.items():
                if name in params:
                    params[name].data.copy_(tensor.to(torch.float32))
    return results


def llama_perplexity(tools: GGUFTools, gguf_path: Path, text_path: Path, ctx: int, chunks: int, threads: int) -> float:
    proc = run([tools.perplexity_exe, "-m", gguf_path, "-f", text_path, "-c", str(ctx), "--chunks", str(chunks),
                "-t", str(threads)])
    m = re.search(r"Final estimate: PPL = ([\d.]+)", proc.stdout + proc.stderr)
    if not m:
        raise RuntimeError("PPL introuvable dans la sortie de llama-perplexity")
    return float(m.group(1))


def llama_bench(tools: GGUFTools, gguf_path: Path, threads: int) -> Dict:
    """
    Vitesse llama.cpp, sur CPU seul (-ngl 0, comparable à transformers CPU) puis
    avec toutes les couches déchargées sur le GPU si un backend GPU existe.
    Le backend réellement utilisé est enregistré.
    """
    out = {}
    for label, ngl in (("cpu", "0"), ("gpu", "99")):
        proc = run([tools.bench_exe, "-m", gguf_path, "-p", "128", "-n", "32", "-t", str(threads),
                    "-ngl", ngl, "-o", "json"])
        rows = json.loads(proc.stdout)
        entry = {"backends": rows[0].get("backends"), "n_gpu_layers": rows[0].get("n_gpu_layers"),
                 "gpu_info": rows[0].get("gpu_info"), "cpu_info": rows[0].get("cpu_info", "").strip()}
        for r in rows:
            key = "prompt_tokens_per_second" if r.get("n_prompt", 0) > 0 else "gen_tokens_per_second"
            entry[key] = r["avg_ts"]
        out[label] = entry
    return out


# ============================================================================
# MAIN
# ============================================================================

def fmt(v, spec: str) -> str:
    return "—" if v is None else format(v, spec)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="mistralai/Mixtral-8x7B-v0.1")
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--llama-cpp-dir", default=os.environ.get("LLAMA_CPP_DIR"),
                        help="Clone de llama.cpp (convert_hf_to_gguf.py, gguf-py) de la même version que les binaires")
    parser.add_argument("--work-dir", default="gguf_work", help="Répertoire des fichiers GGUF générés")
    parser.add_argument("--model-level", action="store_true", help="Perplexité + vitesse (modèle FP32 en RAM)")
    parser.add_argument("--ppl-chunks", type=int, default=20)
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    command = "python " + " ".join(shlex.quote(a) for a in sys.argv)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    torch_methods = [m for m in methods if not m.startswith("gguf:") and m != "awq"]
    gguf_types = [m.split(":", 1)[1] for m in methods if m.startswith("gguf:")]
    tag = args.model.replace("/", "__")
    files = resolve_safetensors(args.model)
    model_dir = files[0].parent
    work_dir = Path(args.work_dir)
    tools = GGUFTools(args.llama_cpp_dir, work_dir)

    report = {
        "command": command,
        "model": args.model,
        "machine": {"platform": platform.platform(), "torch": torch.__version__, "cpu_threads": args.threads,
                    "cuda": torch.cuda.is_available()},
        "weights": {},
        "not_measured": {},
    }
    log("=" * 78)
    log(f"Commande : {command}")
    log(f"Modèle   : {args.model} ({len(files)} fichiers safetensors)")
    log("=" * 78)

    # ---- 1. Poids : méthodes PyTorch
    for method in torch_methods:
        log(f"\n[poids] {method}")
        t0 = time.perf_counter()
        s = measure_model(args.model, method).summary()
        s["seconds"] = time.perf_counter() - t0
        report["weights"][method] = s
        log(f"  ratio vs FP32 {s['ratio_vs_fp32']:.3f}x | cos {s['weighted_cosine']:.6f} | err {s['weighted_rel_error']:.5f}")

    # ---- 1b. Poids : GGUF llama.cpp
    if gguf_types:
        reason = tools.unavailable_reason()
        if reason:
            for q in gguf_types:
                report["not_measured"][f"gguf:{q}"] = reason
            log(f"\n[gguf] [NON MESURÉ] {reason}")
        else:
            report["llama_cpp_version"] = tools.version()
            log(f"\n[gguf] llama.cpp {report['llama_cpp_version']}")
            ref = tools.convert_bf16(model_dir, tag)
            for q in gguf_types:
                log(f"[poids] gguf:{q}")
                t0 = time.perf_counter()
                path = tools.quantize(ref, tag, q, args.threads)
                s = compare_gguf(ref, path)
                s["fp32_bytes"] = s["numel"] * 4
                s["ratio_vs_fp32"] = s["fp32_bytes"] / s["file_bytes"]
                s["ratio_vs_bf16_gguf"] = ref.stat().st_size / s["file_bytes"]
                s["bits_per_param"] = s["file_bytes"] * 8 / s["numel"]
                s["seconds"] = time.perf_counter() - t0
                s["reproduce"] = [f"python convert_hf_to_gguf.py {model_dir} --outtype bf16 --outfile {ref}",
                                  f"llama-quantize {ref} {path} {q}"]
                report["weights"][f"gguf:{q}"] = s
                log(f"  ratio vs FP32 {s['ratio_vs_fp32']:.3f}x | cos {s['weighted_cosine']:.6f} | err {s['weighted_rel_error']:.5f}")

    # ---- AWQ
    if "awq" in methods:
        if importlib.util.find_spec("awq") is None:
            report["not_measured"]["awq"] = "paquet autoawq non installé (AutoAWQ requiert un GPU CUDA)"
        elif not torch.cuda.is_available():
            report["not_measured"]["awq"] = "CUDA indisponible sur cette machine"
        else:
            report["not_measured"]["awq"] = "intégration AWQ non implémentée dans ce script"
        log(f"\n[awq] [NON MESURÉ] {report['not_measured']['awq']}")

    # ---- 2. Niveau modèle
    if args.model_level:
        text = wikitext_test(work_dir)
        log(f"\n[modèle] perplexité wikitext-2 ({args.ppl_chunks} x {args.ctx} tokens) + vitesse")
        report["model_level"] = {"hf": model_level_hf(model_dir, torch_methods, text, args.ctx, args.ppl_chunks)}
        if gguf_types and not tools.unavailable_reason() and tools.perplexity_exe and tools.bench_exe:
            g = {"reference": "llama.cpp GGUF BF16", "methods": {}}
            for q in ["BF16"] + gguf_types:
                path = tools.path(tag, q)
                ppl = llama_perplexity(tools, path, text, args.ctx, args.ppl_chunks, args.threads)
                speed = llama_bench(tools, path, args.threads)
                g["methods"][q] = {"perplexity": ppl, **speed}
                log(f"  gguf {q}: ppl={ppl:.4f} | {speed}")
            report["model_level"]["gguf"] = g
    else:
        report["not_measured"]["model_level"] = "non demandé (--model-level) ; nécessite le modèle FP32 en RAM"

    # ---- Tableau
    log("\n" + "=" * 78)
    log("TABLEAU COMPARATIF — poids [MESURÉ]")
    log("=" * 78)
    log(f"{'méthode':<16}{'Go stockés':>11}{'bits/param':>11}{'vs FP32':>9}{'cos moyen':>11}{'err rel.':>10}")
    for name, s in report["weights"].items():
        size = s.get("file_bytes", s.get("stored_bytes"))
        log(f"{name:<16}{size / 1e9:>11.3f}{s['bits_per_param']:>11.3f}{s['ratio_vs_fp32']:>8.2f}x"
            f"{fmt(s['weighted_cosine'], '.6f'):>11}{fmt(s['weighted_rel_error'], '.5f'):>10}")
    for name, why in report["not_measured"].items():
        log(f"{name:<16} [NON MESURÉ] {why}")

    output = args.output or os.path.join("results", f"compare_{tag}.json")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log(f"\nRésultats complets : {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
