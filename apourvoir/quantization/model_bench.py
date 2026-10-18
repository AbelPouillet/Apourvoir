"""
Mesures de quantisation sur les poids réels d'un modèle

Lit les fichiers ``*.safetensors`` d'un modèle (répertoire local ou dépôt
HuggingFace déjà téléchargé / téléchargeable) **tenseur par tenseur**, ce qui
permet de traiter Mixtral-8x7B (93 Go en BF16) avec quelques Go de RAM et sans
GPU. Pour chaque tenseur :

- octets d'origine (dtype stocké) et équivalent FP32 (numel * 4) ;
- octets réellement stockés après quantisation (codes + échelles) ;
- erreur relative de Frobenius et similarité cosinus original / reconstruit.

Les tenseurs 1-D (normes, biais) et non flottants ne sont pas quantifiés : ils
sont comptés à leur taille d'origine dans le total compressé.
"""

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import torch

from .canonical import quantifier_tensor_canonique, déquantifier_tensor_canonique
from .baselines import BASELINES

QuantFn = Callable[[torch.Tensor], Tuple[object, Dict]]
DequantFn = Callable[[object, Dict], torch.Tensor]


def _canonical(n_bits: int) -> Tuple[QuantFn, DequantFn]:
    return (
        lambda t: quantifier_tensor_canonique(t, n_bits=n_bits),
        lambda q, m: déquantifier_tensor_canonique(q, m),
    )


def get_method(name: str) -> Tuple[QuantFn, DequantFn]:
    """
    Méthodes disponibles : ``canonical<N>`` (ex. canonical8, canonical4),
    ``int8_per_row``, ``q8_0_ref``, ``q4_0_ref``.
    """
    if name.startswith("canonical"):
        return _canonical(int(name[len("canonical"):] or 8))
    if name in BASELINES:
        return BASELINES[name]
    raise ValueError(f"Méthode inconnue: {name}")


def resolve_safetensors(model: str, download: bool = True) -> List[Path]:
    """Fichiers safetensors d'un répertoire local ou d'un dépôt HuggingFace."""
    path = Path(model)
    if path.is_dir():
        files = sorted(path.glob("*.safetensors"))
    else:
        from huggingface_hub import snapshot_download
        local = snapshot_download(
            model,
            allow_patterns=["*.safetensors", "*.json"],
            local_files_only=not download,
        )
        files = sorted(Path(local).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"Aucun fichier .safetensors trouvé pour {model}")
    return files


def iter_tensors(files: List[Path]) -> Iterator[Tuple[str, torch.Tensor]]:
    """Itère sur (nom, tenseur) sans charger tout le modèle en mémoire."""
    from safetensors import safe_open
    for f in files:
        with safe_open(str(f), framework="pt") as handle:
            for name in handle.keys():
                yield name, handle.get_tensor(name)


def should_quantize(tensor: torch.Tensor) -> bool:
    return tensor.is_floating_point() and tensor.dim() >= 2


@dataclass
class TensorResult:
    name: str
    shape: List[int]
    numel: int
    dtype: str
    original_bytes: int
    stored_bytes: int
    quantized: bool
    rel_error: Optional[float] = None
    cosine: Optional[float] = None
    quant_seconds: float = 0.0
    dequant_seconds: float = 0.0


@dataclass
class ModelResult:
    model: str
    method: str
    files: List[str]
    tensors: List[TensorResult] = field(default_factory=list)

    @property
    def numel(self) -> int:
        return sum(t.numel for t in self.tensors)

    @property
    def original_bytes(self) -> int:
        return sum(t.original_bytes for t in self.tensors)

    @property
    def fp32_bytes(self) -> int:
        return self.numel * 4

    @property
    def stored_bytes(self) -> int:
        return sum(t.stored_bytes for t in self.tensors)

    def _weighted(self, attr: str) -> Optional[float]:
        rows = [t for t in self.tensors if getattr(t, attr) is not None]
        total = sum(t.numel for t in rows)
        if not total:
            return None
        return sum(getattr(t, attr) * t.numel for t in rows) / total

    def summary(self) -> Dict:
        quantized = [t for t in self.tensors if t.quantized]
        return {
            "model": self.model,
            "method": self.method,
            "num_tensors": len(self.tensors),
            "num_quantized_tensors": len(quantized),
            "numel": self.numel,
            "original_bytes": self.original_bytes,
            "fp32_bytes": self.fp32_bytes,
            "stored_bytes": self.stored_bytes,
            "ratio_vs_fp32": self.fp32_bytes / self.stored_bytes,
            "ratio_vs_original_dtype": self.original_bytes / self.stored_bytes,
            "bits_per_param": self.stored_bytes * 8 / self.numel,
            "weighted_rel_error": self._weighted("rel_error"),
            "weighted_cosine": self._weighted("cosine"),
            "min_cosine": min((t.cosine for t in quantized), default=None),
            "max_rel_error": max((t.rel_error for t in quantized), default=None),
            "quant_seconds": sum(t.quant_seconds for t in self.tensors),
            "dequant_seconds": sum(t.dequant_seconds for t in self.tensors),
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"summary": self.summary(), "tensors": [asdict(t) for t in self.tensors]}, f, indent=2)


def _stored_bytes(codes, meta: Dict) -> int:
    return int(meta["stored_bytes"])


def measure_tensor(name: str, tensor: torch.Tensor, quant: QuantFn, dequant: DequantFn,
                   device: str = "cpu", measure_error: bool = True) -> TensorResult:
    original_bytes = tensor.numel() * tensor.element_size()
    result = TensorResult(
        name=name, shape=list(tensor.shape), numel=tensor.numel(),
        dtype=str(tensor.dtype).replace("torch.", ""), original_bytes=original_bytes,
        stored_bytes=original_bytes, quantized=False,
    )
    if not should_quantize(tensor):
        return result

    x = tensor.to(device)
    t0 = time.perf_counter()
    codes, meta = quant(x)
    t1 = time.perf_counter()
    result.quantized = True
    result.stored_bytes = _stored_bytes(codes, meta)
    result.quant_seconds = t1 - t0
    if measure_error:
        recon = dequant(codes, meta)
        t2 = time.perf_counter()
        result.dequant_seconds = t2 - t1
        xf = x.double().reshape(-1)
        rf = recon.double().reshape(-1)
        nx = torch.linalg.vector_norm(xf)
        nr = torch.linalg.vector_norm(rf)
        result.rel_error = (torch.linalg.vector_norm(xf - rf) / nx).item() if nx > 0 else 0.0
        result.cosine = (torch.dot(xf, rf) / (nx * nr)).item() if nx > 0 and nr > 0 else 1.0
    return result


def measure_model(model: str, method: str, device: str = "cpu", max_tensors: Optional[int] = None,
                  name_filter: Optional[str] = None, measure_error: bool = True,
                  progress: Optional[Callable[[int, TensorResult], None]] = None) -> ModelResult:
    """Quantifie chaque tenseur du modèle avec ``method`` et agrège les mesures."""
    files = resolve_safetensors(model)
    quant, dequant = get_method(method)
    result = ModelResult(model=model, method=method, files=[f.name for f in files])
    for i, (name, tensor) in enumerate(iter_tensors(files)):
        if max_tensors is not None and len(result.tensors) >= max_tensors:
            break
        if name_filter and name_filter not in name:
            continue
        r = measure_tensor(name, tensor, quant, dequant, device=device, measure_error=measure_error)
        result.tensors.append(r)
        if progress:
            progress(i, r)
    return result


def fake_quantize_state_dict(state_dict: Dict[str, torch.Tensor], method: str) -> Dict[str, torch.Tensor]:
    """
    Remplace chaque poids 2-D par sa version quantifiée puis déquantifiée
    (dans le dtype d'origine), pour mesurer l'effet sur les sorties du modèle.
    """
    quant, dequant = get_method(method)
    out = {}
    for name, tensor in state_dict.items():
        if should_quantize(tensor):
            codes, meta = quant(tensor)
            out[name] = dequant(codes, meta).to(tensor.dtype)
        else:
            out[name] = tensor
    return out
