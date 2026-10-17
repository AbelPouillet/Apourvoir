"""
Launch Configuration for Apourvoir Optimized Launcher

Defines optimization presets for different hardware tiers.

Presets include:
- Batch sizes
- Quantization settings
- Memory management
- Parallelism configuration
- Caching strategies

Example:
    >>> config = LaunchConfig.from_tier(HardwareTier.TIER_3)
    >>> print(config.batch_size)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

from .hardware_profile import HardwareTier

logger = logging.getLogger(__name__)


class OptimizationPreset(Enum):
    """Optimization presets for different use cases."""
    BALANCED = "balanced"  # Balanced between speed and quality
    SPEED = "speed"  # Optimize for maximum speed
    QUALITY = "quality"  # Optimize for maximum quality
    MEMORY = "memory"  # Optimize for minimum memory usage
    LATENCY = "latency"  # Optimize for minimum latency


@dataclass
class QuantizationConfig:
    """Quantization configuration."""
    # Bits per angle for canonical quantization
    n_bits: int = 8
    
    # Quantization method
    method: str = "angular"  # angular, linear, logarithmic
    
    # Adaptive quantization
    adaptive: bool = True
    
    # Minimum and maximum bits
    min_bits: int = 4
    max_bits: int = 16
    
    # Quality threshold for adaptive quantization
    quality_threshold: float = 0.95


@dataclass
class MemoryConfig:
    """Memory management configuration."""
    # Maximum VRAM to use (GB)
    max_vram_gb: Optional[float] = None
    
    # Maximum RAM to use (GB)
    max_ram_gb: Optional[float] = None
    
    # Cache size (GB)
    cache_size_gb: float = 2.0
    
    # Preload models
    preload: bool = True
    
    # Swap to disk if needed
    swap_to_disk: bool = False
    
    # Memory allocation strategy
    strategy: str = "auto"  # auto, conservative, aggressive


@dataclass
class ParallelismConfig:
    """Parallelism configuration."""
    # Number of threads for CPU operations
    cpu_threads: int = 4
    
    # Number of streams for GPU operations
    gpu_streams: int = 2
    
    # Enable multi-GPU
    multi_gpu: bool = False
    
    # Number of experts to load in parallel (for MoE)
    parallel_experts: int = 4
    
    # Tensor parallelism degree
    tensor_parallelism: int = 1
    
    # Pipeline parallelism degree
    pipeline_parallelism: int = 1


@dataclass
class BatchConfig:
    """Batching configuration."""
    # Maximum batch size
    max_batch_size: int = 32
    
    # Optimal batch size
    optimal_batch_size: int = 16
    
    # Minimum batch size
    min_batch_size: int = 1
    
    # Dynamic batching
    dynamic: bool = True
    
    # Batch accumulation
    accumulate: bool = False
    
    # Maximum sequence length
    max_seq_length: int = 4096


@dataclass
class CacheConfig:
    """Caching configuration."""
    # Enable KV cache
    kv_cache: bool = True
    
    # KV cache size (number of tokens)
    kv_cache_size: int = 1024
    
    # Enable attention sink
    attention_sink: bool = True
    
    # Cache compression
    compress_cache: bool = True
    
    # Cache quantization bits
    cache_bits: int = 8


@dataclass
class LaunchConfig:
    """
    Complete launch configuration for Apourvoir.
    
    Contains all optimization settings for a specific hardware tier.
    """
    # Hardware tier
    tier: HardwareTier
    
    # Preset type
    preset: OptimizationPreset = OptimizationPreset.BALANCED
    
    # Quantization settings
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    
    # Memory settings
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    
    # Parallelism settings
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    
    # Batching settings
    batch: BatchConfig = field(default_factory=BatchConfig)
    
    # Cache settings
    cache: CacheConfig = field(default_factory=CacheConfig)
    
    # Additional optimization flags
    flags: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Apply tier-specific defaults."""
        self._apply_tier_defaults()
    
    def _apply_tier_defaults(self) -> None:
        """Apply default settings based on hardware tier."""
        if self.tier == HardwareTier.TIER_1:
            # Entry-level: 4GB VRAM, 8GB RAM, 6 cores
            self.quantization.n_bits = 4
            self.quantization.min_bits = 4
            self.quantization.max_bits = 8
            self.memory.max_vram_gb = 3.5
            self.memory.max_ram_gb = 6.0
            self.memory.cache_size_gb = 0.5
            self.parallelism.cpu_threads = 4
            self.parallelism.gpu_streams = 1
            self.batch.max_batch_size = 4
            self.batch.optimal_batch_size = 2
            self.cache.kv_cache_size = 512
            
        elif self.tier == HardwareTier.TIER_2:
            # Mid-range: 8GB VRAM, 16GB RAM, 8-12 cores
            self.quantization.n_bits = 6
            self.quantization.min_bits = 4
            self.quantization.max_bits = 12
            self.memory.max_vram_gb = 7.0
            self.memory.max_ram_gb = 12.0
            self.memory.cache_size_gb = 1.0
            self.parallelism.cpu_threads = 8
            self.parallelism.gpu_streams = 2
            self.batch.max_batch_size = 8
            self.batch.optimal_batch_size = 4
            self.cache.kv_cache_size = 1024
            
        elif self.tier == HardwareTier.TIER_3:
            # High-end: 12GB VRAM, 32GB RAM, 12-16 cores
            self.quantization.n_bits = 8
            self.quantization.min_bits = 6
            self.quantization.max_bits = 12
            self.memory.max_vram_gb = 10.0
            self.memory.max_ram_gb = 24.0
            self.memory.cache_size_gb = 2.0
            self.parallelism.cpu_threads = 12
            self.parallelism.gpu_streams = 4
            self.batch.max_batch_size = 16
            self.batch.optimal_batch_size = 8
            self.cache.kv_cache_size = 2048
            self.parallelism.parallel_experts = 8
            
        elif self.tier == HardwareTier.TIER_4:
            # Workstation: 16GB VRAM, 48GB RAM, 16-24 cores
            self.quantization.n_bits = 10
            self.quantization.min_bits = 8
            self.quantization.max_bits = 14
            self.memory.max_vram_gb = 14.0
            self.memory.max_ram_gb = 40.0
            self.memory.cache_size_gb = 4.0
            self.parallelism.cpu_threads = 16
            self.parallelism.gpu_streams = 8
            self.batch.max_batch_size = 32
            self.batch.optimal_batch_size = 16
            self.cache.kv_cache_size = 4096
            self.parallelism.parallel_experts = 16
            self.parallelism.tensor_parallelism = 2
            
        elif self.tier == HardwareTier.TIER_5:
            # Machine de guerre: 24GB+ VRAM, 64GB+ RAM, 24+ cores
            self.quantization.n_bits = 12
            self.quantization.min_bits = 8
            self.quantization.max_bits = 16
            self.memory.max_vram_gb = None  # Use all available
            self.memory.max_ram_gb = None  # Use all available
            self.memory.cache_size_gb = 8.0
            self.parallelism.cpu_threads = 24
            self.parallelism.gpu_streams = 16
            self.batch.max_batch_size = 64
            self.batch.optimal_batch_size = 32
            self.cache.kv_cache_size = 8192
            self.parallelism.parallel_experts = 32
            self.parallelism.tensor_parallelism = 4
            self.parallelism.pipeline_parallelism = 2
            self.parallelism.multi_gpu = True
    
    def apply_preset(self, preset: OptimizationPreset) -> None:
        """Apply a specific optimization preset."""
        self.preset = preset
        
        if preset == OptimizationPreset.SPEED:
            # Optimize for speed
            self.quantization.n_bits = min(self.quantization.n_bits, 8)
            self.quantization.adaptive = False
            self.batch.dynamic = False
            self.cache.kv_cache = True
            self.cache.attention_sink = True
            
        elif preset == OptimizationPreset.QUALITY:
            # Optimize for quality
            self.quantization.n_bits = min(self.quantization.max_bits, 16)
            self.quantization.adaptive = True
            self.quantization.quality_threshold = 0.98
            self.cache.kv_cache = True
            self.cache.compress_cache = False
            
        elif preset == OptimizationPreset.MEMORY:
            # Optimize for memory
            self.quantization.n_bits = max(self.quantization.min_bits, 4)
            self.memory.strategy = "conservative"
            self.cache.compress_cache = True
            self.cache.cache_bits = 4
            
        elif preset == OptimizationPreset.LATENCY:
            # Optimize for latency
            self.batch.max_batch_size = 1
            self.batch.optimal_batch_size = 1
            self.batch.dynamic = False
            self.parallelism.cpu_threads = min(self.parallelism.cpu_threads, 4)
    
    @classmethod
    def from_tier(cls, tier: HardwareTier, preset: OptimizationPreset = OptimizationPreset.BALANCED) -> 'LaunchConfig':
        """Create a launch config for a specific tier."""
        config = cls(tier=tier, preset=preset)
        return config
    
    def get_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for the current configuration."""
        recommendations = {
            "tier": self.tier.name,
            "preset": self.preset.value,
            "quantization": {
                "n_bits": self.quantization.n_bits,
                "method": self.quantization.method,
                "adaptive": self.quantization.adaptive,
            },
            "memory": {
                "max_vram_gb": self.memory.max_vram_gb,
                "max_ram_gb": self.memory.max_ram_gb,
                "cache_size_gb": self.memory.cache_size_gb,
            },
            "parallelism": {
                "cpu_threads": self.parallelism.cpu_threads,
                "gpu_streams": self.parallelism.gpu_streams,
                "multi_gpu": self.parallelism.multi_gpu,
            },
            "batch": {
                "max_batch_size": self.batch.max_batch_size,
                "optimal_batch_size": self.batch.optimal_batch_size,
            },
            "cache": {
                "kv_cache_size": self.cache.kv_cache_size,
                "attention_sink": self.cache.attention_sink,
            }
        }
        
        # Add tier-specific recommendations
        if self.tier == HardwareTier.TIER_1:
            recommendations["notes"] = [
                "Use smaller models (Mixtral-8x7B or smaller)",
                "Enable swap to disk if VRAM is insufficient",
                "Limit batch size to 2-4",
                "Use 4-bit quantization for maximum compression"
            ]
        elif self.tier == HardwareTier.TIER_2:
            recommendations["notes"] = [
                "Use Mixtral-8x7B with 6-bit quantization",
                "Batch size up to 8 for good performance",
                "Enable KV cache for better throughput"
            ]
        elif self.tier == HardwareTier.TIER_3:
            recommendations["notes"] = [
                "Use Mixtral-8x22B with 8-bit quantization",
                "Batch size up to 16 for optimal performance",
                "Enable tensor parallelism for MoE models"
            ]
        elif self.tier == HardwareTier.TIER_4:
            recommendations["notes"] = [
                "Use Mixtral-8x22B or larger with 10-bit quantization",
                "Batch size up to 32 for maximum throughput",
                "Enable multi-GPU if available"
            ]
        elif self.tier == HardwareTier.TIER_5:
            recommendations["notes"] = [
                "Use largest models with 12-16 bit quantization",
                "Batch size up to 64 for maximum throughput",
                "Enable all parallelism features"
            ]
        
        return recommendations
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tier": self.tier.name,
            "preset": self.preset.value,
            "quantization": {
                "n_bits": self.quantization.n_bits,
                "method": self.quantization.method,
                "adaptive": self.quantization.adaptive,
                "min_bits": self.quantization.min_bits,
                "max_bits": self.quantization.max_bits,
                "quality_threshold": self.quantization.quality_threshold,
            },
            "memory": {
                "max_vram_gb": self.memory.max_vram_gb,
                "max_ram_gb": self.memory.max_ram_gb,
                "cache_size_gb": self.memory.cache_size_gb,
                "preload": self.memory.preload,
                "swap_to_disk": self.memory.swap_to_disk,
                "strategy": self.memory.strategy,
            },
            "parallelism": {
                "cpu_threads": self.parallelism.cpu_threads,
                "gpu_streams": self.parallelism.gpu_streams,
                "multi_gpu": self.parallelism.multi_gpu,
                "parallel_experts": self.parallelism.parallel_experts,
                "tensor_parallelism": self.parallelism.tensor_parallelism,
                "pipeline_parallelism": self.parallelism.pipeline_parallelism,
            },
            "batch": {
                "max_batch_size": self.batch.max_batch_size,
                "optimal_batch_size": self.batch.optimal_batch_size,
                "min_batch_size": self.batch.min_batch_size,
                "dynamic": self.batch.dynamic,
                "accumulate": self.batch.accumulate,
                "max_seq_length": self.batch.max_seq_length,
            },
            "cache": {
                "kv_cache": self.cache.kv_cache,
                "kv_cache_size": self.cache.kv_cache_size,
                "attention_sink": self.cache.attention_sink,
                "compress_cache": self.cache.compress_cache,
                "cache_bits": self.cache.cache_bits,
            },
            "flags": self.flags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LaunchConfig':
        """Create from dictionary."""
        tier = HardwareTier(data["tier"])
        config = cls(tier=tier)
        
        # Update fields from data
        if "preset" in data:
            config.preset = OptimizationPreset(data["preset"])
        
        if "quantization" in data:
            q = data["quantization"]
            config.quantization.n_bits = q.get("n_bits", config.quantization.n_bits)
            config.quantization.method = q.get("method", config.quantization.method)
            config.quantization.adaptive = q.get("adaptive", config.quantization.adaptive)
            config.quantization.min_bits = q.get("min_bits", config.quantization.min_bits)
            config.quantization.max_bits = q.get("max_bits", config.quantization.max_bits)
            config.quantization.quality_threshold = q.get("quality_threshold", config.quantization.quality_threshold)
        
        if "memory" in data:
            m = data["memory"]
            config.memory.max_vram_gb = m.get("max_vram_gb", config.memory.max_vram_gb)
            config.memory.max_ram_gb = m.get("max_ram_gb", config.memory.max_ram_gb)
            config.memory.cache_size_gb = m.get("cache_size_gb", config.memory.cache_size_gb)
            config.memory.preload = m.get("preload", config.memory.preload)
            config.memory.swap_to_disk = m.get("swap_to_disk", config.memory.swap_to_disk)
            config.memory.strategy = m.get("strategy", config.memory.strategy)
        
        if "parallelism" in data:
            p = data["parallelism"]
            config.parallelism.cpu_threads = p.get("cpu_threads", config.parallelism.cpu_threads)
            config.parallelism.gpu_streams = p.get("gpu_streams", config.parallelism.gpu_streams)
            config.parallelism.multi_gpu = p.get("multi_gpu", config.parallelism.multi_gpu)
            config.parallelism.parallel_experts = p.get("parallel_experts", config.parallelism.parallel_experts)
            config.parallelism.tensor_parallelism = p.get("tensor_parallelism", config.parallelism.tensor_parallelism)
            config.parallelism.pipeline_parallelism = p.get("pipeline_parallelism", config.parallelism.pipeline_parallelism)
        
        if "batch" in data:
            b = data["batch"]
            config.batch.max_batch_size = b.get("max_batch_size", config.batch.max_batch_size)
            config.batch.optimal_batch_size = b.get("optimal_batch_size", config.batch.optimal_batch_size)
            config.batch.min_batch_size = b.get("min_batch_size", config.batch.min_batch_size)
            config.batch.dynamic = b.get("dynamic", config.batch.dynamic)
            config.batch.accumulate = b.get("accumulate", config.batch.accumulate)
            config.batch.max_seq_length = b.get("max_seq_length", config.batch.max_seq_length)
        
        if "cache" in data:
            c = data["cache"]
            config.cache.kv_cache = c.get("kv_cache", config.cache.kv_cache)
            config.cache.kv_cache_size = c.get("kv_cache_size", config.cache.kv_cache_size)
            config.cache.attention_sink = c.get("attention_sink", config.cache.attention_sink)
            config.cache.compress_cache = c.get("compress_cache", config.cache.compress_cache)
            config.cache.cache_bits = c.get("cache_bits", config.cache.cache_bits)
        
        if "flags" in data:
            config.flags = data["flags"]
        
        return config
