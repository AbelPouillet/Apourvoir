"""
Inference engine for the launcher's /v1/completions endpoint.

Backend: HuggingFace transformers (causal LM), CPU or CUDA.

Quantization methods:
- "none": weights in the requested dtype.
- "canonical": every 2-D weight goes through the canonical quantization
  (apourvoir.quantization) and is dequantized back. Generation therefore uses
  exactly the canonical-quantized values, but there is no low-bit kernel:
  memory use and speed are those of the dequantized dtype, not of the codes.
  The measured compression and reconstruction error are reported in stats().
"""

import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

SUPPORTED_QUANT_METHODS = ("none", "canonical")


class InsufficientMemoryError(RuntimeError):
    pass


def _weights_bytes_on_disk(model: str) -> Optional[int]:
    """Size of the model's safetensors files (local dir, HF cache, or Hub metadata)."""
    path = Path(model)
    if path.is_dir():
        files = list(path.glob("*.safetensors"))
        return sum(f.stat().st_size for f in files) or None
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(model, files_metadata=True)
        sizes = [s.size or 0 for s in info.siblings if s.rfilename.endswith(".safetensors")]
        return sum(sizes) or None
    except Exception as e:  # offline: the check is skipped, loading will tell
        logger.warning(f"Could not determine model size for {model}: {e}")
        return None


class InferenceEngine:
    def __init__(
        self,
        model: str,
        quant_method: str = "none",
        n_bits: int = 8,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        check_memory: bool = True,
    ):
        if quant_method not in SUPPORTED_QUANT_METHODS:
            raise NotImplementedError(
                f"Quantization method '{quant_method}' is not supported by the inference engine "
                f"(supported: {', '.join(SUPPORTED_QUANT_METHODS)})"
            )
        self.model_name = model
        self.quant_method = quant_method
        self.n_bits = n_bits
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = "float16" if self.device == "cuda" else "float32"
        self.dtype = getattr(torch, dtype)
        self.check_memory = check_memory
        self.model = None
        self.tokenizer = None
        self._lock = threading.Lock()
        self._stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------ load
    def _check_memory(self) -> None:
        disk = _weights_bytes_on_disk(self.model_name)
        if disk is None:
            return
        # Checkpoints are usually BF16/FP16 (2 bytes/param)
        needed = disk / 2 * torch.tensor([], dtype=self.dtype).element_size()
        if self.device == "cuda":
            available = torch.cuda.mem_get_info()[0]
            where = "VRAM"
        else:
            import psutil
            available = psutil.virtual_memory().available
            where = "RAM"
        if needed > available:
            raise InsufficientMemoryError(
                f"{self.model_name} needs ~{needed / 1e9:.1f} GB of {where} in {str(self.dtype)[6:]} "
                f"but only {available / 1e9:.1f} GB are available. Use a smaller --model or --dtype bfloat16."
            )

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.check_memory:
            self._check_memory()
        t0 = time.perf_counter()
        logger.info(f"Loading {self.model_name} ({self.dtype}, {self.device})...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=self.dtype)
        self.model.to(self.device).eval()
        self._stats = {"load_seconds": time.perf_counter() - t0}
        if self.quant_method == "canonical":
            self._apply_canonical()
        logger.info(f"Inference engine ready ({self._stats})")

    def _apply_canonical(self) -> None:
        from apourvoir.quantization import (
            déquantifier_tensor_canonique, erreur_relative, quantifier_tensor_canonique,
        )
        t0 = time.perf_counter()
        numel = stored = 0
        err_weighted = 0.0
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.dim() < 2:
                    stored += param.numel() * param.element_size()
                    numel += param.numel()
                    continue
                codes, meta = quantifier_tensor_canonique(param.data, n_bits=self.n_bits)
                recon = déquantifier_tensor_canonique(codes, meta)
                err_weighted += erreur_relative(param.data, recon) * param.numel()
                param.data.copy_(recon.to(param.dtype))
                stored += meta["stored_bytes"]
                numel += param.numel()
        quantized_numel = sum(p.numel() for p in self.model.parameters() if p.dim() >= 2)
        self._stats.update({
            "canonical_n_bits": self.n_bits,
            "canonical_seconds": time.perf_counter() - t0,
            "canonical_stored_bytes": stored,
            "canonical_ratio_vs_fp32": numel * 4 / stored,
            "canonical_weighted_rel_error": err_weighted / quantized_numel if quantized_numel else 0.0,
            "note": "weights are dequantized for inference: no low-bit kernel, runtime memory/speed are unchanged",
        })

    @property
    def ready(self) -> bool:
        return self.model is not None

    def stats(self) -> Dict[str, Any]:
        return {"model": self.model_name, "quant_method": self.quant_method, "device": self.device,
                "dtype": str(self.dtype).replace("torch.", ""), "ready": self.ready, **self._stats}

    # ------------------------------------------------------------- generate
    def _generate(self, input_ids: torch.Tensor, max_tokens: int, temperature: float, top_p: float,
                  stop: Optional[List[str]], seed: Optional[int]) -> Dict[str, Any]:
        if not self.ready:
            raise RuntimeError("Inference engine not loaded")
        kwargs: Dict[str, Any] = {"max_new_tokens": max_tokens}
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            kwargs.update(do_sample=False)
        input_ids = input_ids.to(self.device)
        with self._lock, torch.no_grad():
            if seed is not None:
                torch.manual_seed(seed)
            t0 = time.perf_counter()
            out = self.model.generate(input_ids, attention_mask=torch.ones_like(input_ids), **kwargs)
            seconds = time.perf_counter() - t0
        new_ids = out[0, input_ids.shape[1]:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        finish_reason = "length" if len(new_ids) >= max_tokens else "stop"
        for s in stop or []:
            idx = text.find(s)
            if idx >= 0:
                text, finish_reason = text[:idx], "stop"
        return {
            "text": text,
            "finish_reason": finish_reason,
            "usage": {"prompt_tokens": int(input_ids.shape[1]), "completion_tokens": int(len(new_ids)),
                      "total_tokens": int(input_ids.shape[1] + len(new_ids))},
            "seconds": seconds,
        }

    def complete(self, prompt: str, max_tokens: int = 16, temperature: float = 0.0, top_p: float = 1.0,
                 stop: Optional[List[str]] = None, seed: Optional[int] = None) -> Dict[str, Any]:
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        return self._generate(ids, max_tokens, temperature, top_p, stop, seed)

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 16, temperature: float = 0.0,
             top_p: float = 1.0, stop: Optional[List[str]] = None, seed: Optional[int] = None) -> Dict[str, Any]:
        if getattr(self.tokenizer, "chat_template", None):
            ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            if not isinstance(ids, torch.Tensor):  # BatchEncoding in recent transformers
                ids = ids["input_ids"]
        else:
            prompt = "".join(f"{m['role']}: {m['content']}\n" for m in messages) + "assistant:"
            ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        return self._generate(ids, max_tokens, temperature, top_p, stop, seed)
