"""
Baselines de quantisation (référence pure PyTorch)

Implémentations simples et lisibles servant de point de comparaison honnête
à la quantisation canonique, avec la même API (quantize -> codes + métadonnées,
dequantize -> float32) et le même comptage d'octets réellement stockés.

- ``int8_per_row``  : INT8 symétrique, une échelle float32 par ligne (absmax).
- ``q8_0_blocks``   : réimplémentation du format GGML Q8_0 (blocs de 32,
  échelle float16, valeurs int8 -> 8.5 bits/valeur).
- ``q4_0_blocks``   : réimplémentation du format GGML Q4_0 (blocs de 32,
  échelle float16, valeurs 4 bits packées par paires -> 4.5 bits/valeur).

Ce ne sont PAS les binaires llama.cpp : pour les vrais GGUF (Q8_0, Q4_K_M),
voir ``scripts/compare_baselines.py`` qui appelle ``llama-quantize``.
"""

from typing import Any, Callable, Dict, Tuple

import torch

_BLOCK = 32


def _rows(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 0:
        return tensor.reshape(1, 1)
    if tensor.dim() == 1:
        return tensor.reshape(1, -1)
    return tensor.reshape(-1, tensor.shape[-1])


def _bytes(*tensors: torch.Tensor) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def quantize_int8_per_row(tensor: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """INT8 symétrique par ligne : q = round(x / s), s = max|x| / 127."""
    x = _rows(tensor.detach()).float()
    scale = x.abs().amax(dim=1) / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.round(x / scale.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
    codes = {"q": q, "scale": scale}
    meta = {"method": "int8_per_row", "shape_original": tuple(tensor.shape), "stored_bytes": _bytes(q, scale)}
    return codes, meta


def dequantize_int8_per_row(codes: Dict[str, torch.Tensor], meta: Dict[str, Any]) -> torch.Tensor:
    return (codes["q"].float() * codes["scale"].unsqueeze(1)).view(meta["shape_original"])


def _blocks(tensor: torch.Tensor) -> Tuple[torch.Tensor, int]:
    x = _rows(tensor.detach()).float()
    dim = x.shape[1]
    pad = (-dim) % _BLOCK
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    return x.view(x.shape[0], -1, _BLOCK), dim


def quantize_q8_0(tensor: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Format Q8_0 : d = max|x| / 127 (float16) par bloc de 32, q = round(x / d) en int8."""
    b, dim = _blocks(tensor)
    d = (b.abs().amax(dim=2) / 127.0).to(torch.float16)
    d_f = d.float()
    inv = torch.where(d_f > 0, 1.0 / d_f, torch.zeros_like(d_f))
    q = torch.round(b * inv.unsqueeze(2)).clamp(-127, 127).to(torch.int8)
    codes = {"q": q, "d": d}
    meta = {"method": "q8_0", "shape_original": tuple(tensor.shape), "dim": dim, "stored_bytes": _bytes(q, d)}
    return codes, meta


def dequantize_q8_0(codes: Dict[str, torch.Tensor], meta: Dict[str, Any]) -> torch.Tensor:
    x = codes["q"].float() * codes["d"].float().unsqueeze(2)
    x = x.reshape(x.shape[0], -1)[:, : meta["dim"]]
    return x.reshape(meta["shape_original"])


def quantize_q4_0(tensor: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Format Q4_0 (comme ggml) : par bloc de 32, d = max_signé / -8 (float16),
    q = clamp(round(x / d) + 8, 0, 15), deux valeurs 4 bits par octet.
    """
    b, dim = _blocks(tensor)
    idx = b.abs().argmax(dim=2, keepdim=True)
    max_signed = torch.gather(b, 2, idx).squeeze(2)
    d = (max_signed / -8.0).to(torch.float16)
    d_f = d.float()
    inv = torch.where(d_f != 0, 1.0 / d_f, torch.zeros_like(d_f))
    q = torch.clamp(torch.round(b * inv.unsqueeze(2)) + 8, 0, 15).to(torch.uint8)
    packed = q[..., 0::2] | (q[..., 1::2] << 4)
    codes = {"q": packed, "d": d}
    meta = {"method": "q4_0", "shape_original": tuple(tensor.shape), "dim": dim, "stored_bytes": _bytes(packed, d)}
    return codes, meta


def dequantize_q4_0(codes: Dict[str, torch.Tensor], meta: Dict[str, Any]) -> torch.Tensor:
    packed = codes["q"]
    low = (packed & 0x0F).float() - 8
    high = (packed >> 4).float() - 8
    q = torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], _BLOCK)
    x = q * codes["d"].float().unsqueeze(2)
    x = x.reshape(x.shape[0], -1)[:, : meta["dim"]]
    return x.reshape(meta["shape_original"])


BASELINES: Dict[str, Tuple[Callable, Callable]] = {
    "int8_per_row": (quantize_int8_per_row, dequantize_int8_per_row),
    "q8_0_ref": (quantize_q8_0, dequantize_q8_0),
    "q4_0_ref": (quantize_q4_0, dequantize_q4_0),
}
