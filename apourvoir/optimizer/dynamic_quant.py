"""
Dynamic Quantizer for Apourvoir SDK

Adaptively adjusts quantization bit precision (4-16 bits) based on:
- Available VRAM/RAM
- Model complexity
- Performance requirements
- Quality constraints
"""

import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Callable
from enum import Enum

import torch
import numpy as np

from .hardware_detector import HardwareSpecs
from .memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class QuantizationMode(Enum):
    """Quantization modes."""
    STATIC = "static"           # Fixed bit precision
    DYNAMIC = "dynamic"         # Adjusts based on memory
    ADAPTIVE = "adaptive"       # Adjusts based on memory + performance
    AUTO = "auto"               # Fully automatic


class QuantizationBackend(Enum):
    """Quantization backends."""
    CANONICAL = "canonical"     # Apourvoir canonical quantization
    PYTORCH = "pytorch"         # PyTorch native quantization
    BITSANDBYTES = "bitsandbytes" # bitsandbytes library
    AWQ = "awq"                 # Activation-aware Weight Quantization
    GPTQ = "gptq"               # GPTQ quantization


@dataclass
class QuantizationConfig:
    """Configuration for quantization."""
    n_bits: int = 8
    mode: QuantizationMode = QuantizationMode.DYNAMIC
    backend: QuantizationBackend = QuantizationBackend.CANONICAL
    symmetric: bool = True
    per_channel: bool = False
    per_group: bool = False
    group_size: int = 128
    dtype: torch.dtype = torch.int8
    
    # Canonical quantization specific
    R_MAJEUR: float = 2.0
    R_MINEUR: float = 0.5
    
    # Quality constraints
    max_quality_loss: float = 0.01  # 1% max quality loss
    min_accuracy: float = 0.98      # 98% min accuracy
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_bits": self.n_bits,
            "mode": self.mode.value,
            "backend": self.backend.value,
            "symmetric": self.symmetric,
            "per_channel": self.per_channel,
            "per_group": self.per_group,
            "group_size": self.group_size,
            "dtype": str(self.dtype),
            "R_MAJEUR": self.R_MAJEUR,
            "R_MINEUR": self.R_MINEUR,
            "max_quality_loss": self.max_quality_loss,
            "min_accuracy": self.min_accuracy,
        }


@dataclass
class QuantizationStats:
    """Statistics for quantization operations."""
    n_bits: int = 0
    compression_ratio: float = 1.0
    quantization_time_ms: float = 0.0
    dequantization_time_ms: float = 0.0
    memory_saved_gb: float = 0.0
    quality_loss: float = 0.0
    success: bool = True
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_bits": self.n_bits,
            "compression_ratio": self.compression_ratio,
            "quantization_time_ms": self.quantization_time_ms,
            "dequantization_time_ms": self.dequantization_time_ms,
            "memory_saved_gb": self.memory_saved_gb,
            "quality_loss": self.quality_loss,
            "success": self.success,
            "error_message": self.error_message,
        }


class DynamicQuantizer:
    """
    Dynamic quantizer that adaptively adjusts bit precision.
    
    This class provides:
    - Dynamic bit adjustment (4-16 bits)
    - Multiple quantization backends
    - Quality-aware quantization
    - Performance-aware quantization
    - Memory tracking
    
    Example:
        >>> from apourvoir.optimizer import HardwareDetector, DynamicQuantizer
        >>> hardware = HardwareDetector.detect()
        >>> quantizer = DynamicQuantizer(hardware)
        >>> 
        >>> # Quantize a tensor
        >>> tensor = torch.randn(1000, 1000)
        >>> quant_tensor, metadata = quantizer.quantize(tensor)
        >>> 
        >>> # Get current bit precision
        >>> bits = quantizer.get_optimal_bits()
    """
    
    def __init__(
        self,
        hardware: HardwareSpecs,
        mode: QuantizationMode = QuantizationMode.AUTO,
        backend: QuantizationBackend = QuantizationBackend.CANONICAL,
        min_bits: int = 4,
        max_bits: int = 16,
        initial_bits: Optional[int] = None,
        memory_buffer_gb: float = 1.0,
        quality_threshold: float = 0.98,
    ):
        """
        Initialize the dynamic quantizer.
        
        Args:
            hardware: Hardware specifications
            mode: Quantization mode
            backend: Quantization backend
            min_bits: Minimum bit precision
            max_bits: Maximum bit precision
            initial_bits: Initial bit precision (defaults to hardware optimal)
            memory_buffer_gb: Memory buffer for safety
            quality_threshold: Minimum quality threshold (0-1)
        """
        self.hardware = hardware
        self.mode = mode
        self.backend = backend
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.memory_buffer_gb = memory_buffer_gb
        self.quality_threshold = quality_threshold
        
        # Current state
        self.current_bits = initial_bits or hardware.optimal_bits
        self.current_bits = max(min_bits, min(max_bits, self.current_bits))
        
        # Statistics
        self.stats: List[QuantizationStats] = []
        self._lock = threading.Lock()
        
        # Memory manager
        self.memory_manager: Optional[MemoryManager] = None
        
        # Quality tracking
        self._quality_history: List[float] = []
        self._max_quality_loss = 0.0
        
        # Performance tracking
        self._quantization_times: List[float] = []
        self._dequantization_times: List[float] = []
        
        logger.info(f"DynamicQuantizer initialized: bits={self.current_bits}, "
                   f"mode={mode.value}, backend={backend.value}")
    
    def set_memory_manager(self, manager: MemoryManager) -> None:
        """Set a memory manager for tracking."""
        self.memory_manager = manager
    
    def get_optimal_bits(self) -> int:
        """Get current optimal bit precision."""
        return self.current_bits
    
    def get_config(self) -> QuantizationConfig:
        """Get current quantization configuration."""
        return QuantizationConfig(
            n_bits=self.current_bits,
            mode=self.mode,
            backend=self.backend,
            R_MAJEUR=self.hardware.vram_total_gb / 10 if self.hardware.gpu_available else 2.0,
            R_MINEUR=0.5,
        )
    
    def adjust_bits(
        self,
        memory_pressure: Optional[float] = None,
        quality_loss: Optional[float] = None,
        quantization_time: Optional[float] = None,
    ) -> int:
        """
        Adjust bit precision based on current conditions.
        
        Args:
            memory_pressure: Memory pressure (0-1, higher = more pressure)
            quality_loss: Current quality loss (0-1)
            quantization_time: Time for last quantization (ms)
        
        Returns:
            New optimal bit precision
        """
        with self._lock:
            # Update tracking
            if quality_loss is not None:
                self._quality_history.append(quality_loss)
                if len(self._quality_history) > 10:
                    self._quality_history.pop(0)
                self._max_quality_loss = max(self._quality_history)
            
            if quantization_time is not None:
                self._quantization_times.append(quantization_time)
                if len(self._quantization_times) > 10:
                    self._quantization_times.pop(0)
            
            # Calculate new bit precision
            new_bits = self._calculate_optimal_bits(
                memory_pressure, quality_loss, quantization_time
            )
            
            # Clamp to min/max
            new_bits = max(self.min_bits, min(self.max_bits, new_bits))
            
            if new_bits != self.current_bits:
                logger.info(f"Adjusting quantization bits: {self.current_bits} -> {new_bits} "
                           f"(memory_pressure={memory_pressure}, quality_loss={quality_loss})")
                self.current_bits = new_bits
            
            return self.current_bits
    
    def _calculate_optimal_bits(
        self,
        memory_pressure: Optional[float],
        quality_loss: Optional[float],
        quantization_time: Optional[float],
    ) -> int:
        """Calculate optimal bit precision."""
        # Start with current bits
        bits = self.current_bits
        
        # Adjust based on memory pressure
        if memory_pressure is not None:
            if memory_pressure > 0.8:
                bits = max(self.min_bits, bits - 2)
            elif memory_pressure > 0.6:
                bits = max(self.min_bits, bits - 1)
            elif memory_pressure < 0.3:
                bits = min(self.max_bits, bits + 1)
            elif memory_pressure < 0.2:
                bits = min(self.max_bits, bits + 2)
        
        # Adjust based on quality loss
        if quality_loss is not None:
            if quality_loss > self.quality_threshold * 2:
                bits = min(self.max_bits, bits + 2)
            elif quality_loss > self.quality_threshold:
                bits = min(self.max_bits, bits + 1)
        
        # Adjust based on quantization time
        if quantization_time and len(self._quantization_times) >= 3:
            avg_time = np.mean(self._quantization_times)
            if avg_time > 100:  # Too slow
                bits = max(self.min_bits, bits - 1)
            elif avg_time < 10:  # Very fast
                bits = min(self.max_bits, bits + 1)
        
        return bits
    
    def quantize(
        self,
        tensor: torch.Tensor,
        n_bits: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Quantize a tensor using the selected backend.
        
        Args:
            tensor: Input tensor to quantize
            n_bits: Override bit precision (uses current if None)
            **kwargs: Additional arguments for the quantization backend
        
        Returns:
            Tuple of (quantized tensor, metadata)
        """
        bits = n_bits or self.current_bits
        
        start_time = time.perf_counter()
        
        try:
            if self.backend == QuantizationBackend.CANONICAL:
                quant_tensor, metadata = self._quantize_canonical(tensor, bits, **kwargs)
            elif self.backend == QuantizationBackend.PYTORCH:
                quant_tensor, metadata = self._quantize_pytorch(tensor, bits, **kwargs)
            elif self.backend == QuantizationBackend.BITSANDBYTES:
                quant_tensor, metadata = self._quantize_bitsandbytes(tensor, bits, **kwargs)
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")
            
            end_time = time.perf_counter()
            quant_time = (end_time - start_time) * 1000
            
            # Calculate stats
            original_size = tensor.numel() * tensor.element_size()
            quant_size = quant_tensor.numel() * quant_tensor.element_size()
            compression_ratio = original_size / quant_size if quant_size > 0 else 1.0
            memory_saved = (original_size - quant_size) / (1024**3)
            
            stat = QuantizationStats(
                n_bits=bits,
                compression_ratio=compression_ratio,
                quantization_time_ms=quant_time,
                memory_saved_gb=memory_saved,
                success=True,
            )
            self.stats.append(stat)
            if len(self.stats) > 100:
                self.stats.pop(0)
            
            # Adjust bits if in adaptive mode
            if self.mode == QuantizationMode.ADAPTIVE:
                self.adjust_bits(quantization_time=quant_time)
            
            return quant_tensor, metadata
        
        except Exception as e:
            end_time = time.perf_counter()
            stat = QuantizationStats(
                n_bits=bits,
                compression_ratio=1.0,
                quantization_time_ms=(end_time - start_time) * 1000,
                success=False,
                error_message=str(e),
            )
            self.stats.append(stat)
            raise
    
    def dequantize(
        self,
        quant_tensor: torch.Tensor,
        metadata: Dict[str, Any],
        **kwargs,
    ) -> torch.Tensor:
        """
        Dequantize a tensor.
        
        Args:
            quant_tensor: Quantized tensor
            metadata: Metadata from quantization
            **kwargs: Additional arguments
        
        Returns:
            Dequantized tensor
        """
        start_time = time.perf_counter()
        
        try:
            if self.backend == QuantizationBackend.CANONICAL:
                tensor = self._dequantize_canonical(quant_tensor, metadata, **kwargs)
            elif self.backend == QuantizationBackend.PYTORCH:
                tensor = self._dequantize_pytorch(quant_tensor, metadata, **kwargs)
            elif self.backend == QuantizationBackend.BITSANDBYTES:
                tensor = self._dequantize_bitsandbytes(quant_tensor, metadata, **kwargs)
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")
            
            end_time = time.perf_counter()
            dequant_time = (end_time - start_time) * 1000
            
            self._dequantization_times.append(dequant_time)
            if len(self._dequantization_times) > 10:
                self._dequantization_times.pop(0)
            
            return tensor
        
        except Exception as e:
            logger.error(f"Dequantization failed: {e}")
            raise
    
    def _quantize_canonical(
        self,
        tensor: torch.Tensor,
        n_bits: int,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Quantize using Apourvoir canonical quantization."""
        from ..quantization import (
            normaliser_vecteur,
            coordonnées_sphériques,
            projeter_sur_tore,
            quantifier_angles,
        )
        
        device = tensor.device
        shape = tensor.shape
        R = kwargs.get("R", 2.0)
        r = kwargs.get("r", 0.5)
        
        # Reshape for processing
        if len(shape) == 2:
            tensor_2d = tensor.view(-1, 1, shape[-1])
        else:
            tensor_2d = tensor.view(-1, shape[-1])
        
        # Normalize
        tensor_norm = normaliser_vecteur(tensor_2d)
        
        # Project on torus
        tensor_projeté = projeter_sur_tore(tensor_norm, R, r)
        
        # Convert to spherical coordinates
        _, theta, phi = coordonnées_sphériques(tensor_projeté)
        
        # Quantize angles
        theta_q, phi_q = quantifier_angles(theta, phi, n_bits)
        
        # Store quantized angles
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
            "backend": "canonical",
        }
        
        return tensor_quant, metadata
    
    def _dequantize_canonical(
        self,
        tensor_quant: torch.Tensor,
        metadata: Dict[str, Any],
        **kwargs,
    ) -> torch.Tensor:
        """Dequantize using Apourvoir canonical quantization."""
        from ..quantization import dÉquantifier_angles
        
        n_bits = metadata.get("n_bits", 8)
        R = metadata.get("R", 2.0)
        r = metadata.get("r", 0.5)
        
        # Extract theta_q and phi_q
        theta_q = tensor_quant[..., 0].long()
        phi_q = tensor_quant[..., 1].long()
        
        # Dequantize angles
        theta, phi = dÉquantifier_angles(theta_q, phi_q, n_bits)
        
        # Reconstruct vectors
        X = (R + r * torch.cos(phi)) * torch.cos(theta)
        Y = (R + r * torch.cos(phi)) * torch.sin(theta)
        Z = r * torch.sin(phi)
        
        tensor_reconstruit = torch.stack([X, Y, Z], dim=-1)
        
        # Reshape to original shape
        original_shape = metadata.get("shape_original", tensor_quant.shape[:-1])
        if len(original_shape) == 2:
            tensor_reconstruit = tensor_reconstruit.view(original_shape)
        else:
            tensor_reconstruit = tensor_reconstruit.view(*original_shape)
        
        return tensor_reconstruit
    
    def _quantize_pytorch(
        self,
        tensor: torch.Tensor,
        n_bits: int,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Quantize using PyTorch native quantization."""
        symmetric = kwargs.get("symmetric", True)
        per_channel = kwargs.get("per_channel", False)
        
        if n_bits == 8:
            dtype = torch.qint8 if symmetric else torch.quint8
            scale, zero_point = self._calculate_scale_zero_point(tensor, symmetric, per_channel)
            
            if per_channel:
                quant_tensor = torch.quantize_per_channel(
                    tensor, scales=scale, zero_points=zero_point,
                    axis=0, dtype=dtype
                )
            else:
                quant_tensor = torch.quantize(
                    tensor, scale=scale, zero_point=zero_point, dtype=dtype
                )
        elif n_bits == 4:
            # For 4-bit, we need to simulate with 8-bit and additional compression
            dtype = torch.quint8
            scale, zero_point = self._calculate_scale_zero_point(tensor, True, per_channel)
            quant_tensor = torch.quantize(tensor, scale=scale, zero_point=zero_point, dtype=dtype)
            # Simulate 4-bit by storing 2 values per byte
            quant_tensor = self._pack_4bit(quant_tensor)
        else:
            raise ValueError(f"PyTorch quantization only supports 4 or 8 bits, got {n_bits}")
        
        metadata = {
            "shape_original": tensor.shape,
            "n_bits": n_bits,
            "symmetric": symmetric,
            "per_channel": per_channel,
            "dtype": str(quant_tensor.dtype),
            "compression_ratio": (tensor.numel() * 4) / (quant_tensor.numel() * (n_bits // 8)),
            "backend": "pytorch",
        }
        
        return quant_tensor, metadata
    
    def _dequantize_pytorch(
        self,
        quant_tensor: torch.Tensor,
        metadata: Dict[str, Any],
        **kwargs,
    ) -> torch.Tensor:
        """Dequantize using PyTorch native dequantization."""
        n_bits = metadata.get("n_bits", 8)
        
        if n_bits == 4:
            # Unpack 4-bit
            quant_tensor = self._unpack_4bit(quant_tensor)
        
        return torch.dequantize(quant_tensor)
    
    def _calculate_scale_zero_point(
        self,
        tensor: torch.Tensor,
        symmetric: bool,
        per_channel: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate scale and zero point for quantization."""
        if per_channel:
            min_vals = tensor.min(dim=1, keepdim=True)[0]
            max_vals = tensor.max(dim=1, keepdim=True)[0]
        else:
            min_vals = tensor.min()
            max_vals = tensor.max()
        
        if symmetric:
            abs_max = torch.max(torch.abs(min_vals), torch.abs(max_vals))
            scale = abs_max / 127.0
            zero_point = torch.zeros_like(scale)
        else:
            scale = (max_vals - min_vals) / 255.0
            zero_point = torch.round(min_vals / scale)
        
        return scale, zero_point
    
    def _pack_4bit(self, tensor_8bit: torch.Tensor) -> torch.Tensor:
        """Pack 8-bit tensor into 4-bit representation."""
        # Reshape to [batch, channels, height, width]
        shape = tensor_8bit.shape
        tensor_8bit = tensor_8bit.view(-1)
        
        # Pack two 4-bit values into one byte
        packed = torch.zeros((tensor_8bit.numel() + 1) // 2, dtype=torch.uint8, device=tensor_8bit.device)
        packed[::2] = tensor_8bit[::2] >> 4
        packed[::2] |= (tensor_8bit[1::2] & 0x0F)
        
        # Handle odd length
        if tensor_8bit.numel() % 2 == 1:
            packed = packed[:-1]
        
        return packed.view(*shape[:-1], -1)
    
    def _unpack_4bit(self, tensor_4bit: torch.Tensor) -> torch.Tensor:
        """Unpack 4-bit tensor into 8-bit representation."""
        shape = tensor_4bit.shape
        tensor_4bit = tensor_4bit.view(-1)
        
        # Unpack to 8-bit
        unpacked = torch.zeros(tensor_4bit.numel() * 2, dtype=torch.uint8, device=tensor_4bit.device)
        unpacked[::2] = (tensor_4bit >> 4) & 0x0F
        unpacked[1::2] = tensor_4bit & 0x0F
        
        return unpacked.view(*shape[:-1], -1)
    
    def _quantize_bitsandbytes(
        self,
        tensor: torch.Tensor,
        n_bits: int,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Quantize using bitsandbytes library."""
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("bitsandbytes library is required for this backend")
        
        if n_bits == 8:
            # Use 8-bit quantization
            quant_tensor = tensor.to(torch.uint8)
        elif n_bits == 4:
            # Use 4-bit quantization
            quant_tensor = bnb.functional.quantize_4bit(tensor)
        else:
            raise ValueError(f"bitsandbytes only supports 4 or 8 bits, got {n_bits}")
        
        metadata = {
            "shape_original": tensor.shape,
            "n_bits": n_bits,
            "dtype": str(quant_tensor.dtype),
            "compression_ratio": (tensor.numel() * 4) / (quant_tensor.numel() * (n_bits // 8)),
            "backend": "bitsandbytes",
        }
        
        return quant_tensor, metadata
    
    def _dequantize_bitsandbytes(
        self,
        quant_tensor: torch.Tensor,
        metadata: Dict[str, Any],
        **kwargs,
    ) -> torch.Tensor:
        """Dequantize using bitsandbytes library."""
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("bitsandbytes library is required for this backend")
        
        n_bits = metadata.get("n_bits", 8)
        
        if n_bits == 4:
            return bnb.functional.dequantize_4bit(quant_tensor)
        else:
            return quant_tensor.to(torch.float32)
    
    def get_stats(self) -> List[Dict[str, Any]]:
        """Get quantization statistics."""
        with self._lock:
            return [s.to_dict() for s in self.stats]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        with self._lock:
            if not self.stats:
                return {
                    "total_quantizations": 0,
                    "avg_compression_ratio": 0,
                    "avg_quantization_time_ms": 0,
                    "total_memory_saved_gb": 0,
                    "current_bits": self.current_bits,
                }
            
            compression_ratios = [s.compression_ratio for s in self.stats]
            quant_times = [s.quantization_time_ms for s in self.stats if s.quantization_time_ms > 0]
            memory_saved = [s.memory_saved_gb for s in self.stats]
            
            return {
                "total_quantizations": len(self.stats),
                "avg_compression_ratio": np.mean(compression_ratios) if compression_ratios else 0,
                "avg_quantization_time_ms": np.mean(quant_times) if quant_times else 0,
                "total_memory_saved_gb": sum(memory_saved),
                "current_bits": self.current_bits,
                "min_bits": self.min_bits,
                "max_bits": self.max_bits,
            }
    
    def reset_stats(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self.stats.clear()
            self._quality_history.clear()
            self._quantization_times.clear()
            self._dequantization_times.clear()
    
    def set_bits(self, bits: int) -> None:
        """Manually set bit precision."""
        with self._lock:
            self.current_bits = max(self.min_bits, min(self.max_bits, bits))
        logger.info(f"Quantization bits manually set to: {self.current_bits}")


class BitPrecisionCalibrator:
    """
    Helper class to calibrate optimal bit precision for a specific model.
    
    Tests different bit precisions and selects the best one based on:
    - Memory usage
    - Inference speed
    - Quality/accuracy
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        hardware: HardwareSpecs,
        calibration_dataset: Optional[List] = None,
        max_bits: int = 16,
        min_bits: int = 4,
    ):
        """
        Initialize the calibrator.
        
        Args:
            model: Model to calibrate
            hardware: Hardware specifications
            calibration_dataset: Dataset for calibration
            max_bits: Maximum bits to test
            min_bits: Minimum bits to test
        """
        self.model = model
        self.hardware = hardware
        self.calibration_dataset = calibration_dataset
        self.max_bits = max_bits
        self.min_bits = min_bits
        self.quantizer = DynamicQuantizer(hardware)
    
    def calibrate(
        self,
        num_samples: int = 10,
        warmup: int = 3,
    ) -> Dict[str, Any]:
        """
        Calibrate optimal bit precision.
        
        Args:
            num_samples: Number of samples to test per bit precision
            warmup: Number of warmup runs
        
        Returns:
            Dictionary with calibration results and recommendations
        """
        results = {}
        
        # Test each bit precision
        for bits in range(self.min_bits, self.max_bits + 1):
            self.quantizer.set_bits(bits)
            
            # Quantize the model
            quantized_model = self._quantize_model(bits)
            
            # Test performance
            perf = self._test_performance(quantized_model, bits, num_samples, warmup)
            results[bits] = perf
        
        # Select optimal
        optimal = self._select_optimal(results)
        
        return {
            "results": results,
            "optimal_bits": optimal["bits"],
            "optimal_performance": optimal,
            "recommendation": self._generate_recommendation(results, optimal),
        }
    
    def _quantize_model(self, bits: int) -> torch.nn.Module:
        """Quantize the entire model."""
        quantized_model = self.model.__class__()
        
        # Copy and quantize state dict
        state_dict = self.model.state_dict()
        quant_state_dict = {}
        
        for name, param in state_dict.items():
            if param.dim() > 1:  # Only quantize weights, not biases
                quant_param, _ = self.quantizer.quantize(param, n_bits=bits)
                quant_state_dict[name] = quant_param
            else:
                quant_state_dict[name] = param
        
        quantized_model.load_state_dict(quant_state_dict)
        return quantized_model
    
    def _test_performance(
        self,
        model: torch.nn.Module,
        bits: int,
        num_samples: int,
        warmup: int,
    ) -> Dict[str, Any]:
        """Test model performance at given bit precision."""
        import time
        
        if self.calibration_dataset is None:
            # Create dummy dataset
            input_size = 1024
            calibration_dataset = [
                torch.randn(input_size, device="cuda" if self.hardware.gpu_available else "cpu")
                for _ in range(num_samples)
            ]
        else:
            calibration_dataset = self.calibration_dataset[:num_samples]
        
        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = model(calibration_dataset[0])
        
        # Test
        total_time = 0
        total_tokens = 0
        correct = 0
        
        for sample in calibration_dataset:
            start = time.perf_counter()
            with torch.no_grad():
                output = model(sample)
            end = time.perf_counter()
            
            total_time += (end - start) * 1000
            total_tokens += sample.numel()
        
        # Calculate metrics
        avg_time_ms = total_time / num_samples
        tokens_per_second = (total_tokens * num_samples) / total_time * 1000
        
        # Estimate memory
        model_size = sum(p.numel() * p.element_size() for p in model.parameters())
        quant_model_size = model_size * (bits // 8) / 4  # Compared to FP32
        memory_saved = (model_size - quant_model_size) / (1024**3)
        
        # Estimate quality (placeholder - would need actual validation)
        quality_score = 1.0 - (0.05 * (16 - bits) / 16)  # Linear degradation
        
        return {
            "bits": bits,
            "avg_time_ms": avg_time_ms,
            "tokens_per_second": tokens_per_second,
            "memory_saved_gb": memory_saved,
            "quality_score": quality_score,
            "compression_ratio": model_size / quant_model_size if quant_model_size > 0 else 1.0,
        }
    
    def _select_optimal(self, results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Select optimal bit precision from calibration results."""
        best_score = -1
        best_bits = self.min_bits
        best_result = None
        
        for bits, result in results.items():
            # Calculate score: balance of speed, memory, and quality
            speed_score = result["tokens_per_second"] / max(
                [r["tokens_per_second"] for r in results.values()], default=1
            )
            memory_score = result["memory_saved_gb"] / max(
                [r["memory_saved_gb"] for r in results.values()], default=1
            )
            quality_score = result["quality_score"]
            
            # Weighted score
            score = 0.4 * speed_score + 0.3 * memory_score + 0.3 * quality_score
            
            if score > best_score:
                best_score = score
                best_bits = bits
                best_result = result
        
        best_result["score"] = best_score
        return best_result
    
    def _generate_recommendation(
        self,
        results: Dict[int, Dict[str, Any]],
        optimal: Dict[str, Any],
    ) -> str:
        """Generate human-readable recommendation."""
        bits = optimal["bits"]
        tps = optimal["tokens_per_second"]
        memory_saved = optimal["memory_saved_gb"]
        quality = optimal["quality_score"]
        
        recommendation = (
            f"Recommended bit precision: {bits} bits\n"
            f"  - Tokens/sec: {tps:.1f}\n"
            f"  - Memory saved: {memory_saved:.2f} GB\n"
            f"  - Quality score: {quality:.3f}\n"
            f"  - Compression ratio: {optimal['compression_ratio']:.1f}x\n"
        )
        
        # Compare with other options
        if bits < self.max_bits:
            higher_bits = bits + 1
            if higher_bits in results:
                higher = results[higher_bits]
                recommendation += (
                    f"\nAlternative: {higher_bits} bits\n"
                    f"  - Tokens/sec: {higher['tokens_per_second']:.1f} "
                    f"({higher['tokens_per_second'] / tps - 1:+.1%})\n"
                    f"  - Memory saved: {higher['memory_saved_gb']:.2f} GB "
                    f"({higher['memory_saved_gb'] / memory_saved - 1:+.1%})\n"
                )
        
        return recommendation


if __name__ == "__main__":
    # Test the dynamic quantizer
    from .hardware_detector import HardwareDetector
    
    hardware = HardwareDetector.detect()
    quantizer = DynamicQuantizer(hardware)
    
    print("DynamicQuantizer Test")
    print("=" * 80)
    print(f"Initial bits: {quantizer.current_bits}")
    print(f"Config: {quantizer.get_config()}")
    print()
    
    # Create a test tensor
    tensor = torch.randn(1000, 1000)
    if hardware.gpu_available:
        tensor = tensor.cuda()
    
    # Quantize at different bit precisions
    for bits in [4, 8, 12, 16]:
        quant_tensor, metadata = quantizer.quantize(tensor, n_bits=bits)
        print(f"{bits}-bit quantization: {metadata['compression_ratio']:.1f}x compression")
    
    # Get summary
    print(f"\nSummary: {quantizer.get_summary()}")
