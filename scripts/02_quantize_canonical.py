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
import gc
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

<<<<<<< HEAD
    report = {
        "numel": numel,
        "original_bytes": original_bytes,
        "fp32_bytes": numel * 4,
        "stored_tensor_bytes": stored_bytes,
        "ratio_vs_fp32": numel * 4 / stored_bytes,
        "ratio_vs_original_dtype": original_bytes / stored_bytes,
        "weighted_rel_error": (sum(e * n for e, n in errors) / sum(n for _, n in errors)) if errors else None,
        "max_rel_error": max((e for e, _ in errors), default=None),
=======
def setup_logging(log_file: Optional[str] = None):
    """Configure le logging."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8") if log_file else logging.NullHandler()
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
>>>>>>> 77c4730ddc74a19d57f9067dba39b2384ca4186c
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
<<<<<<< HEAD
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
=======
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
    
    def _quantize_one_tensor(self, name: str, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Quantifie un seul tenseur et renvoie (tensor_quant, metadata_couche)."""
        n_bits = self.dynamic_quant.get_optimal_bits() if self.dynamic_quant else self.config.get("n_bits", 8)
        tensor_quant, layer_metadata = quantifier_tensor_canonique(
            tensor,
            n_bits=n_bits,
            R=self.config.get("R_MAJEUR", 2.0),
            r=self.config.get("R_MINEUR", 0.5),
        )
        layer_metadata["name"] = name

        if self.memory_manager:
            self.memory_manager.allocate(tensor_quant, name, priority=0)
            if self.dynamic_quant and self.config.get("dynamic_quantization", True):
                stats = self.memory_manager.get_stats()
                self.dynamic_quant.adjust_bits(
                    memory_pressure=stats.used_gb /
                                   (self.memory_manager.max_vram_gb + self.memory_manager.max_ram_gb)
                )

        return tensor_quant, layer_metadata

    def quantize_model_streaming(
        self,
        input_dir: str,
        output_dir: str,
        validate: bool = True,
        num_validate_samples: int = 5,
    ) -> Dict:
        """
        Quantifie un modèle shard par shard, sans jamais charger l'intégralité
        des poids en mémoire. Nécessaire pour les modèles dont la taille
        (ex: Mixtral 8x7B, ~87GB en fp32) dépasse la RAM disponible.
        """
        from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors

        logger = self.logger
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        config_path = os.path.join(input_dir, "config.json")
        if not os.path.exists(config_path):
            logger.error(f"Fichier de configuration introuvable: {config_path}")
            raise FileNotFoundError(config_path)
        with open(config_path, "r") as f:
            model_config = json.load(f)
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(model_config, f, indent=2)

        index_path = os.path.join(input_dir, "model.safetensors.index.json")
        single_path = os.path.join(input_dir, "model.safetensors")
        bin_path = os.path.join(input_dir, "pytorch_model.bin")

        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                shard_files = sorted(set(json.load(f)["weight_map"].values()))
        elif os.path.exists(single_path):
            shard_files = ["model.safetensors"]
        elif os.path.exists(bin_path):
            shard_files = ["pytorch_model.bin"]
        else:
            logger.error(f"Aucun fichier de poids trouvé dans {input_dir}")
            raise FileNotFoundError(index_path)

        device = self.config.get("device", "cpu")
        metadata = {
            "hardware": self.hardware.to_dict(),
            "config": self.config,
            "layers": [],
        }
        output_weight_map: Dict[str, str] = {}
        validation = {"total_tests": 0, "passed_tests": 0, "max_diff": 0.0, "mean_diff": 0.0, "details": []}
        original_size_bytes = 0
        quant_size_bytes = 0
        samples_validated = 0

        for shard_file in tqdm(shard_files, desc="Shards"):
            shard_path = os.path.join(input_dir, shard_file)
            if shard_file.endswith(".bin"):
                shard_state = torch.load(shard_path, map_location=device)
            else:
                shard_state = load_safetensors(shard_path, device=device)

            quant_shard = {}
            for name, tensor in shard_state.items():
                original_size_bytes += tensor.numel() * tensor.element_size()
                try:
                    tensor_quant, layer_metadata = self._quantize_one_tensor(name, tensor)
                    metadata["layers"].append(layer_metadata)
                    quant_shard[name] = tensor_quant.contiguous()
                    quant_size_bytes += tensor_quant.numel() * tensor_quant.element_size()
                except Exception as e:
                    logger.error(f"Erreur de quantification pour {name}: {e}")
                    quant_shard[name] = tensor.contiguous()
                    quant_size_bytes += tensor.numel() * tensor.element_size()
                    continue

                if validate and samples_validated < num_validate_samples:
                    try:
                        dequant = déquantifier_tensor_canonique(tensor_quant, layer_metadata)
                        diff = torch.abs(tensor.float() - dequant.float())
                        max_diff = diff.max().item()
                        mean_diff = diff.mean().item()
                        del dequant, diff
                    except Exception as e:
                        logger.warning(f"Erreur de validation pour {name}: {e}")
                        max_diff = mean_diff = float("inf")
                    validation["details"].append({
                        "layer": name, "max_diff": max_diff, "mean_diff": mean_diff,
                        "passed": max_diff < 0.1,
                    })
                    validation["total_tests"] += 1
                    validation["passed_tests"] += int(max_diff < 0.1)
                    validation["max_diff"] = max(validation["max_diff"], max_diff)
                    validation["mean_diff"] += mean_diff / num_validate_samples
                    samples_validated += 1

            out_shard_name = os.path.splitext(os.path.basename(shard_file))[0] + "_quant.safetensors"
            save_safetensors(quant_shard, os.path.join(output_dir, out_shard_name))
            for name in quant_shard:
                output_weight_map[name] = out_shard_name

            if self.memory_manager:
                self.memory_manager.cleanup()
            del shard_state, quant_shard
            gc.collect()

        with open(os.path.join(output_dir, "model_quant.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": quant_size_bytes}, "weight_map": output_weight_map}, f, indent=2)

        with open(os.path.join(output_dir, "quantization_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        original_size_gb = original_size_bytes / (1024**3)
        quant_size_gb = quant_size_bytes / (1024**3)
        compression_report = self.get_compression_report(original_size_gb, quant_size_gb)
        if validate and validation["total_tests"] > 0:
            validation["validation_rate"] = validation["passed_tests"] / validation["total_tests"] * 100
        else:
            validation["validation_rate"] = 0.0

        report = {"compression": compression_report, "validation": validation if validate else {}}
        with open(os.path.join(output_dir, "compression_report.json"), "w") as f:
            json.dump(report, f, indent=2)

        logger.info(
            f"✅ Quantification terminée: {original_size_gb:.2f}GB -> {quant_size_gb:.2f}GB "
            f"(ratio {compression_report['compression_ratio']:.2f}x)"
        )
        return report

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
    
    # Initialiser le quantiseur optimisé
    quantizer = OptimizedQuantizer(config)

    # Quantifier le modèle en streaming (shard par shard, RAM bornée)
    start_time = time.time()
    report = quantizer.quantize_model_streaming(
        args.input,
        args.output,
        validate=args.validate,
    )
    quant_time = time.time() - start_time

    compression_report = report["compression"]
    validation_results = report["validation"]
    report["quantization_time"] = quant_time
    report["hardware"] = hardware.to_dict()
    report["config"] = config
    with open(os.path.join(args.output, "compression_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Afficher le résumé
    logger.info("=" * 80)
    logger.info("RÉSULTATS DE LA QUANTISATION")
    logger.info("=" * 80)
    logger.info(f"Taille originale: {compression_report['original_size_gb']:.2f} GB")
    logger.info(f"Taille quantifiée: {compression_report['quantized_size_gb']:.2f} GB")
    logger.info(f"Ratio de compression: {compression_report['compression_ratio']:.1f}x")
    logger.info(f"Espace économisé: {compression_report['space_saved_gb']:.2f} GB")
    logger.info(f"Temps de quantification: {quant_time:.2f} secondes")

    if validation_results:
        logger.info(f"Validation: {validation_results['validation_rate']:.1f}% de succès")
        logger.info(f"Différence maximale: {validation_results['max_diff']:.6f}")

    logger.info("=" * 80)
    logger.info("✅ Quantisation terminée avec succès!")
    logger.info(f"Modèle quantifié sauvegardé dans: {args.output}")
>>>>>>> 77c4730ddc74a19d57f9067dba39b2384ca4186c


if __name__ == "__main__":
    sys.exit(main())
