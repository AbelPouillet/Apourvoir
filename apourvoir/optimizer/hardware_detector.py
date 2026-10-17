"""
Hardware Detector for Apourvoir SDK

Auto-detects system hardware specifications and provides
optimization recommendations for canonical quantization.
"""

import os
import platform
import multiprocessing
import psutil
import subprocess
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import logging

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HardwareSpecs:
    """Complete hardware specifications and optimization recommendations."""
    
    # === GPU Specifications ===
    gpu_available: bool = False
    gpu_count: int = 0
    gpu_model: str = "N/A"
    gpu_vendor: str = "N/A"
    vram_total_gb: float = 0.0
    vram_used_gb: float = 0.0
    vram_free_gb: float = 0.0
    cuda_version: str = "N/A"
    cuDNN_version: str = "N/A"
    cuda_cores: int = 0
    tensor_cores: bool = False
    tensor_core_version: str = "N/A"
    gpu_memory_bandwidth_gbps: float = 0.0
    gpu_compute_capability: str = "N/A"
    multi_gpu: bool = False
    
    # === CPU Specifications ===
    cpu_model: str = "N/A"
    cpu_cores: int = 0
    cpu_threads: int = 0
    cpu_physical_cores: int = 0
    cpu_architecture: str = "N/A"
    cpu_bits: int = 0
    cpu_frequency_mhz: float = 0.0
    cpu_cache_l1_kb: int = 0
    cpu_cache_l2_kb: int = 0
    cpu_cache_l3_kb: int = 0
    
    # === Memory Specifications ===
    ram_total_gb: float = 0.0
    ram_used_gb: float = 0.0
    ram_free_gb: float = 0.0
    ram_available_gb: float = 0.0
    swap_total_gb: float = 0.0
    swap_used_gb: float = 0.0
    
    # === Storage ===
    disk_total_gb: float = 0.0
    disk_used_gb: float = 0.0
    disk_free_gb: float = 0.0
    disk_type: str = "HDD"
    
    # === NUMA ===
    numa_nodes: int = 1
    numa_aware: bool = False
    
    # === OS ===
    os_name: str = "Unknown"
    os_version: str = "Unknown"
    
    # === Python ===
    python_version: str = "Unknown"
    pytorch_version: str = "Unknown"
    
    # === Recommendations ===
    optimal_batch_size: int = 1
    max_experts_parallel: int = 1
    recommended_dtype: str = "float32"
    use_pinned_memory: bool = False
    use_numa: bool = False
    use_amp: bool = False
    use_tensorrt: bool = False
    use_onnx: bool = False
    max_threads: int = 1
    
    # === Quantization Recommendations ===
    optimal_bits: int = 8
    min_bits: int = 4
    max_bits: int = 16
    adaptive_quantization: bool = True
    
    # === Performance Estimates ===
    estimated_tokens_per_second: float = 0.0
    estimated_latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "gpu": {
                "available": self.gpu_available,
                "count": self.gpu_count,
                "model": self.gpu_model,
                "vendor": self.gpu_vendor,
                "vram_total_gb": self.vram_total_gb,
                "vram_used_gb": self.vram_used_gb,
                "vram_free_gb": self.vram_free_gb,
                "cuda_version": self.cuda_version,
                "cudnn_version": self.cuDNN_version,
                "cuda_cores": self.cuda_cores,
                "tensor_cores": self.tensor_cores,
                "tensor_core_version": self.tensor_core_version,
                "memory_bandwidth_gbps": self.gpu_memory_bandwidth_gbps,
                "compute_capability": self.gpu_compute_capability,
                "multi_gpu": self.multi_gpu,
            },
            "cpu": {
                "model": self.cpu_model,
                "cores": self.cpu_cores,
                "threads": self.cpu_threads,
                "physical_cores": self.cpu_physical_cores,
                "architecture": self.cpu_architecture,
                "bits": self.cpu_bits,
                "frequency_mhz": self.cpu_frequency_mhz,
                "cache_l1_kb": self.cpu_cache_l1_kb,
                "cache_l2_kb": self.cpu_cache_l2_kb,
                "cache_l3_kb": self.cpu_cache_l3_kb,
            },
            "memory": {
                "ram_total_gb": self.ram_total_gb,
                "ram_used_gb": self.ram_used_gb,
                "ram_free_gb": self.ram_free_gb,
                "ram_available_gb": self.ram_available_gb,
                "swap_total_gb": self.swap_total_gb,
                "swap_used_gb": self.swap_used_gb,
            },
            "storage": {
                "total_gb": self.disk_total_gb,
                "used_gb": self.disk_used_gb,
                "free_gb": self.disk_free_gb,
                "type": self.disk_type,
            },
            "numa": {
                "nodes": self.numa_nodes,
                "aware": self.numa_aware,
            },
            "os": {
                "name": self.os_name,
                "version": self.os_version,
            },
            "python": {
                "version": self.python_version,
                "pytorch_version": self.pytorch_version,
            },
            "recommendations": {
                "optimal_batch_size": self.optimal_batch_size,
                "max_experts_parallel": self.max_experts_parallel,
                "recommended_dtype": self.recommended_dtype,
                "use_pinned_memory": self.use_pinned_memory,
                "use_numa": self.use_numa,
                "use_amp": self.use_amp,
                "use_tensorrt": self.use_tensorrt,
                "use_onnx": self.use_onnx,
                "max_threads": self.max_threads,
                "optimal_bits": self.optimal_bits,
                "min_bits": self.min_bits,
                "max_bits": self.max_bits,
                "adaptive_quantization": self.adaptive_quantization,
            },
            "performance_estimates": {
                "tokens_per_second": self.estimated_tokens_per_second,
                "latency_ms": self.estimated_latency_ms,
            }
        }


class HardwareDetector:
    """
    Detects and analyzes system hardware to provide optimization recommendations
    for canonical quantization of MoE models.
    
    This class provides comprehensive hardware detection including:
    - GPU specifications (VRAM, CUDA cores, Tensor cores, etc.)
    - CPU specifications (cores, threads, architecture, cache, etc.)
    - Memory specifications (RAM, swap)
    - Storage specifications
    - NUMA support
    - Optimization recommendations
    
    Example:
        >>> from apourvoir.optimizer import HardwareDetector
        >>> hardware = HardwareDetector.detect()
        >>> print(f"GPU: {hardware.gpu_model}")
        >>> print(f"VRAM: {hardware.vram_total_gb} GB")
        >>> print(f"Optimal batch size: {hardware.optimal_batch_size}")
    """
    
    @staticmethod
    def detect() -> HardwareSpecs:
        """
        Detect all hardware specifications and compute optimization recommendations.
        
        Returns:
            HardwareSpecs: Complete hardware specifications with recommendations.
        """
        logger.info("Detecting hardware specifications...")
        
        specs = HardwareSpecs()
        
        # Detect OS
        specs = HardwareDetector._detect_os(specs)
        
        # Detect Python
        specs = HardwareDetector._detect_python(specs)
        
        # Detect GPU
        specs = HardwareDetector._detect_gpu(specs)
        
        # Detect CPU
        specs = HardwareDetector._detect_cpu(specs)
        
        # Detect Memory
        specs = HardwareDetector._detect_memory(specs)
        
        # Detect Storage
        specs = HardwareDetector._detect_storage(specs)
        
        # Detect NUMA
        specs = HardwareDetector._detect_numa(specs)
        
        # Compute recommendations
        specs = HardwareDetector._compute_recommendations(specs)
        
        logger.info(f"Hardware detection complete: {specs.gpu_model}, "
                   f"VRAM: {specs.vram_total_gb:.1f}GB, "
                   f"CPU: {specs.cpu_model}, "
                   f"RAM: {specs.ram_total_gb:.1f}GB")
        
        return specs
    
    @staticmethod
    def _detect_os(specs: HardwareSpecs) -> HardwareSpecs:
        """Detect OS information."""
        specs.os_name = platform.system()
        specs.os_version = platform.version()
        return specs
    
    @staticmethod
    def _detect_python(specs: HardwareSpecs) -> HardwareSpecs:
        """Detect Python and PyTorch versions."""
        import sys
        specs.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        if torch.__version__:
            specs.pytorch_version = torch.__version__
        
        return specs
    
    @staticmethod
    def _detect_gpu(specs: HardwareSpecs) -> HardwareSpecs:
        """Detect GPU specifications."""
        specs.gpu_available = torch.cuda.is_available()
        
        if not specs.gpu_available:
            return specs
        
        try:
            specs.gpu_count = torch.cuda.device_count()
            specs.multi_gpu = specs.gpu_count > 1
            
            # Get primary GPU info
            for i in range(min(specs.gpu_count, 4)):  # Check first 4 GPUs
                props = torch.cuda.get_device_properties(i)
                
                if i == 0:
                    specs.gpu_model = props.name
                    specs.gpu_vendor = "NVIDIA" if "NVIDIA" in props.name else "AMD" if "AMD" in props.name else "Other"
                    specs.vram_total_gb = props.total_memory / (1024**3)
                    specs.cuda_cores = props.multi_processor_count * 128  # Approximate for NVIDIA
                    specs.gpu_memory_bandwidth_gbps = props.memory_bandwidth / 1e9
                    specs.gpu_compute_capability = f"{props.major}.{props.minor}"
                    
                    # Detect Tensor cores
                    specs.tensor_cores = "Tensor" in props.name or "Ampere" in props.name or "Hopper" in props.name
                    if specs.tensor_cores:
                        if "Hopper" in props.name:
                            specs.tensor_core_version = "4th Gen"
                        elif "Ampere" in props.name:
                            specs.tensor_core_version = "3rd Gen"
                        elif "Turing" in props.name:
                            specs.tensor_core_version = "2nd Gen"
                        elif "Volta" in props.name:
                            specs.tensor_core_version = "1st Gen"
                    
                    # Get current VRAM usage
                    specs.vram_used_gb = torch.cuda.memory_allocated(i) / (1024**3)
                    specs.vram_free_gb = specs.vram_total_gb - specs.vram_used_gb
                
                # Get CUDA and cuDNN versions
                if i == 0:
                    try:
                        specs.cuda_version = torch.version.cuda
                    except:
                        specs.cuda_version = "Unknown"
                    
                    try:
                        specs.cuDNN_version = torch.backends.cudnn.version()
                    except:
                        specs.cuDNN_version = "Unknown"
        
        except Exception as e:
            logger.warning(f"Error detecting GPU: {e}")
        
        return specs
    
    @staticmethod
    def _detect_cpu(specs: HardwareSpecs) -> HardwareSpecs:
        """Detect CPU specifications."""
        try:
            # CPU info
            specs.cpu_model = platform.processor() or "Unknown"
            specs.cpu_cores = multiprocessing.cpu_count()
            specs.cpu_threads = psutil.cpu_count(logical=True)
            specs.cpu_physical_cores = psutil.cpu_count(logical=False)
            
            # CPU architecture
            specs.cpu_architecture = platform.machine()
            specs.cpu_bits = 64 if "64" in specs.cpu_architecture else 32
            
            # CPU frequency
            try:
                specs.cpu_frequency_mhz = psutil.cpu_freq().max
            except:
                specs.cpu_frequency_mhz = 0.0
            
            # CPU cache (approximate)
            if "Intel" in specs.cpu_model:
                specs.cpu_cache_l1_kb = 32 * specs.cpu_physical_cores  # 32KB per core
                specs.cpu_cache_l2_kb = 256 * specs.cpu_physical_cores  # 256KB per core
                specs.cpu_cache_l3_kb = 8192 if specs.cpu_physical_cores >= 4 else 4096  # Shared
            elif "AMD" in specs.cpu_model:
                specs.cpu_cache_l1_kb = 32 * specs.cpu_physical_cores
                specs.cpu_cache_l2_kb = 512 * specs.cpu_physical_cores
                specs.cpu_cache_l3_kb = 16384 if specs.cpu_physical_cores >= 8 else 8192
        
        except Exception as e:
            logger.warning(f"Error detecting CPU: {e}")
        
        return specs
    
    @staticmethod
    def _detect_memory(specs: HardwareSpecs) -> HardwareSpecs:
        """Detect memory specifications."""
        try:
            mem = psutil.virtual_memory()
            specs.ram_total_gb = mem.total / (1024**3)
            specs.ram_used_gb = mem.used / (1024**3)
            specs.ram_free_gb = mem.free / (1024**3)
            specs.ram_available_gb = mem.available / (1024**3)
            
            # Swap
            swap = psutil.swap_memory()
            specs.swap_total_gb = swap.total / (1024**3)
            specs.swap_used_gb = swap.used / (1024**3)
        
        except Exception as e:
            logger.warning(f"Error detecting memory: {e}")
        
        return specs
    
    @staticmethod
    def _detect_storage(specs: HardwareSpecs) -> HardwareSpecs:
        """Detect storage specifications."""
        try:
            disk = psutil.disk_usage('/')
            specs.disk_total_gb = disk.total / (1024**3)
            specs.disk_used_gb = disk.used / (1024**3)
            specs.disk_free_gb = disk.free / (1024**3)
            
            # Try to detect disk type
            try:
                with open('/sys/block/sda/queue/rotational', 'r') as f:
                    rotational = f.read().strip()
                    specs.disk_type = "HDD" if rotational == "1" else "SSD"
            except:
                specs.disk_type = "Unknown"
        
        except Exception as e:
            logger.warning(f"Error detecting storage: {e}")
        
        return specs
    
    @staticmethod
    def _detect_numa(specs: HardwareSpecs) -> HardwareSpecs:
        """Detect NUMA nodes."""
        try:
            # Check for NUMA support
            if hasattr(os, 'sched_getaffinity'):
                # Linux with NUMA
                try:
                    import numactl
                    specs.numa_nodes = numactl.NUMA().num_nodes()
                    specs.numa_aware = specs.numa_nodes > 1
                except ImportError:
                    # Try alternative method
                    try:
                        with open('/proc/cpuinfo', 'r') as f:
                            content = f.read()
                            # Count unique physical IDs
                            physical_ids = set()
                            for line in content.split('\n'):
                                if line.startswith('physical id'):
                                    physical_ids.add(line.split(':')[1].strip())
                            specs.numa_nodes = len(physical_ids)
                            specs.numa_aware = specs.numa_nodes > 1
                    except:
                        specs.numa_nodes = 1
                        specs.numa_aware = False
            else:
                # Non-Linux or no NUMA support
                specs.numa_nodes = 1
                specs.numa_aware = False
        
        except Exception as e:
            logger.warning(f"Error detecting NUMA: {e}")
            specs.numa_nodes = 1
            specs.numa_aware = False
        
        return specs
    
    @staticmethod
    def _compute_recommendations(specs: HardwareSpecs) -> HardwareSpecs:
        """Compute optimization recommendations based on hardware."""
        
        # === GPU Recommendations ===
        if specs.gpu_available:
            # Use mixed precision if Tensor cores available
            specs.use_amp = specs.tensor_cores
            
            # Use TensorRT for NVIDIA GPUs
            specs.use_tensorrt = specs.gpu_vendor == "NVIDIA" and specs.tensor_cores
            
            # Use ONNX Runtime for inference optimization
            specs.use_onnx = True
            
            # Pinned memory for data transfer optimization
            specs.use_pinned_memory = True
            
            # Compute optimal batch size based on VRAM
            # Mixtral-8x7B: ~47GB in FP32, ~2.5GB quantized (18.8x)
            # We use ~70% of VRAM for model + batch
            available_vram = specs.vram_total_gb * 0.7
            
            # Estimate: 0.5GB per token for inference
            specs.optimal_batch_size = max(1, int((available_vram - 2.5) // 0.5))
            
            # Max parallel experts (Mixtral has 8 experts)
            specs.max_experts_parallel = min(8, max(1, int(specs.vram_total_gb // 4)))
            
            # Recommended dtype
            if specs.vram_total_gb >= 24:
                specs.recommended_dtype = "float16"
            elif specs.vram_total_gb >= 16:
                specs.recommended_dtype = "bfloat16"
            else:
                specs.recommended_dtype = "float32"
            
            # Use NUMA if available
            specs.use_numa = specs.numa_aware
        else:
            # CPU-only mode
            specs.use_amp = False
            specs.use_tensorrt = False
            specs.use_onnx = False
            specs.use_pinned_memory = False
            specs.use_numa = specs.numa_aware
            
            # CPU batch size (more conservative)
            available_ram = specs.ram_total_gb * 0.6
            specs.optimal_batch_size = max(1, int((available_ram - 2.5) // 2))
            specs.max_experts_parallel = 1
            specs.recommended_dtype = "float32"
        
        # Max threads
        specs.max_threads = min(
            specs.cpu_threads,
            16 if specs.gpu_available else specs.cpu_threads
        )
        
        # === Quantization Recommendations ===
        if specs.gpu_available:
            if specs.vram_total_gb >= 24:
                specs.optimal_bits = 16
                specs.min_bits = 8
            elif specs.vram_total_gb >= 16:
                specs.optimal_bits = 12
                specs.min_bits = 6
            elif specs.vram_total_gb >= 8:
                specs.optimal_bits = 8
                specs.min_bits = 4
            else:
                specs.optimal_bits = 4
                specs.min_bits = 4
        else:
            # CPU mode - be more aggressive with quantization
            if specs.ram_total_gb >= 32:
                specs.optimal_bits = 12
            elif specs.ram_total_gb >= 16:
                specs.optimal_bits = 8
            else:
                specs.optimal_bits = 4
            specs.min_bits = 4
        
        specs.max_bits = 16
        specs.adaptive_quantization = True
        
        # === Performance Estimates ===
        # Estimate tokens per second based on hardware
        base_tps = 10.0  # Base for CPU
        
        if specs.gpu_available:
            # GPU scaling factor
            gpu_factor = 1.0
            if "A100" in specs.gpu_model or "H100" in specs.gpu_model:
                gpu_factor = 4.0
            elif "V100" in specs.gpu_model:
                gpu_factor = 3.0
            elif "RTX 4090" in specs.gpu_model:
                gpu_factor = 3.5
            elif "RTX 3090" in specs.gpu_model:
                gpu_factor = 2.5
            elif "RTX 2080" in specs.gpu_model:
                gpu_factor = 2.0
            elif "T4" in specs.gpu_model:
                gpu_factor = 1.5
            
            # VRAM scaling
            vram_factor = min(2.0, specs.vram_total_gb / 16)
            
            # Tensor core bonus
            tensor_factor = 1.5 if specs.tensor_cores else 1.0
            
            base_tps = 50.0 * gpu_factor * vram_factor * tensor_factor
        
        # Batch size scaling
        batch_factor = min(2.0, specs.optimal_batch_size / 4)
        
        specs.estimated_tokens_per_second = base_tps * batch_factor
        specs.estimated_latency_ms = 1000 / max(1, specs.estimated_tokens_per_second / specs.optimal_batch_size)
        
        return specs
    
    @staticmethod
    def print_summary(specs: HardwareSpecs) -> None:
        """Print a formatted summary of hardware specifications."""
        print("=" * 80)
        print("APOURVOIR - Hardware Detection Summary")
        print("=" * 80)
        
        print("\n[GPU]")
        print(f"  Available: {specs.gpu_available}")
        if specs.gpu_available:
            print(f"  Model: {specs.gpu_model}")
            print(f"  Count: {specs.gpu_count}")
            print(f"  VRAM: {specs.vram_total_gb:.1f} GB (Used: {specs.vram_used_gb:.1f} GB)")
            print(f"  CUDA Cores: {specs.cuda_cores}")
            print(f"  Tensor Cores: {specs.tensor_cores} ({specs.tensor_core_version})")
            print(f"  Compute Capability: {specs.gpu_compute_capability}")
            print(f"  Memory Bandwidth: {specs.gpu_memory_bandwidth_gbps:.1f} GB/s")
        
        print("\n[CPU]")
        print(f"  Model: {specs.cpu_model}")
        print(f"  Physical Cores: {specs.cpu_physical_cores}")
        print(f"  Logical Cores: {specs.cpu_threads}")
        print(f"  Architecture: {specs.cpu_architecture}")
        print(f"  Frequency: {specs.cpu_frequency_mhz:.1f} MHz")
        print(f"  Cache L1: {specs.cpu_cache_l1_kb} KB")
        print(f"  Cache L2: {specs.cpu_cache_l2_kb} KB")
        print(f"  Cache L3: {specs.cpu_cache_l3_kb} KB")
        
        print("\n[Memory]")
        print(f"  RAM: {specs.ram_total_gb:.1f} GB (Used: {specs.ram_used_gb:.1f} GB)")
        print(f"  Swap: {specs.swap_total_gb:.1f} GB (Used: {specs.swap_used_gb:.1f} GB)")
        
        print("\n[Storage]")
        print(f"  Disk: {specs.disk_total_gb:.1f} GB (Free: {specs.disk_free_gb:.1f} GB)")
        print(f"  Type: {specs.disk_type}")
        
        print("\n[NUMA]")
        print(f"  Nodes: {specs.numa_nodes}")
        print(f"  NUMA-aware: {specs.numa_aware}")
        
        print("\n[Recommendations]")
        print(f"  Optimal Batch Size: {specs.optimal_batch_size}")
        print(f"  Max Parallel Experts: {specs.max_experts_parallel}")
        print(f"  Recommended dtype: {specs.recommended_dtype}")
        print(f"  Use Pinned Memory: {specs.use_pinned_memory}")
        print(f"  Use NUMA: {specs.use_numa}")
        print(f"  Use AMP: {specs.use_amp}")
        print(f"  Use TensorRT: {specs.use_tensorrt}")
        print(f"  Use ONNX: {specs.use_onnx}")
        print(f"  Optimal Bits: {specs.optimal_bits} ({specs.min_bits}-{specs.max_bits})")
        print(f"  Adaptive Quantization: {specs.adaptive_quantization}")
        
        print("\n[Performance Estimates]")
        print(f"  Estimated Tokens/sec: {specs.estimated_tokens_per_second:.1f}")
        print(f"  Estimated Latency: {specs.estimated_latency_ms:.2f} ms")
        
        print("\n" + "=" * 80)


if __name__ == "__main__":
    # Test the hardware detector
    specs = HardwareDetector.detect()
    HardwareDetector.print_summary(specs)
