"""
Apourvoir Optimizer SDK

Hardware-aware optimization components for canonical quantization:
- HardwareDetector: Auto-detect VRAM, RAM, CPU cores, GPU model
- AdaptiveBatcher: Dynamic batch size adjustment based on memory
- MemoryManager: Intelligent VRAM/RAM management with caching
- DynamicQuantizer: Adaptive bit precision (4-16 bits) based on resources
- ParallelExperts: Expert parallelism for MoE models
- KernelFuser: Operation fusion to reduce overhead
- NUMAAwareThreadPool: Thread pool optimized for NUMA architectures
- BenchmarkProfiler: Performance profiling and comparison

Usage:
    from apourvoir.optimizer import (
        HardwareDetector,
        AdaptiveBatcher,
        MemoryManager,
        DynamicQuantizer,
        ParallelExperts,
        KernelFuser,
        NUMAAwareThreadPool,
        BenchmarkProfiler
    )
    
    # Detect hardware and get recommendations
    hardware = HardwareDetector.detect()
    
    # Use adaptive batching
    batcher = AdaptiveBatcher(hardware)
    
    # Manage memory intelligently
    memory_manager = MemoryManager(
        max_vram_gb=hardware.vram_total_gb * 0.8,
        max_ram_gb=hardware.ram_total_gb * 0.7
    )
"""

from .hardware_detector import HardwareDetector, HardwareSpecs
from .adaptive_batching import AdaptiveBatcher, BatchConfig
from .memory_manager import MemoryManager
from .dynamic_quant import DynamicQuantizer
from .expert_parallelism import ParallelExperts
from .kernel_fusion import KernelFuser
from .thread_pool import NUMAAwareThreadPool
from .benchmark_profiler import BenchmarkProfiler, BenchmarkResult

__all__ = [
    "HardwareDetector",
    "HardwareSpecs",
    "AdaptiveBatcher",
    "BatchConfig",
    "MemoryManager",
    "DynamicQuantizer",
    "ParallelExperts",
    "KernelFuser",
    "NUMAAwareThreadPool",
    "BenchmarkProfiler",
    "BenchmarkResult",
]
