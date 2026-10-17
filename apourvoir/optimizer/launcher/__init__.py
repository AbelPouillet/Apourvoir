"""
Apourvoir Optimized Launcher

Provides hardware-aware launching and configuration for optimal performance
across different hardware tiers (4GB VRAM/8GB RAM to 24GB VRAM/64GB RAM/32 cores).

Components:
- HardwareProfile: Detect and profile hardware capabilities
- LaunchConfig: Configuration for different hardware tiers
- OptimizedLauncher: Main launcher with automatic optimization

Hardware Tiers:
- TIER_1: 4GB VRAM, 8GB RAM, 6 cores (Entry-level / Laptop)
- TIER_2: 8GB VRAM, 16GB RAM, 8-12 cores (Mid-range / Gaming PC)
- TIER_3: 12GB VRAM, 32GB RAM, 12-16 cores (High-end / Workstation)
- TIER_4: 16GB VRAM, 48GB RAM, 16-24 cores (Power Workstation)
- TIER_5: 24GB+ VRAM, 64GB+ RAM, 24+ cores (Machine de guerre / Server)

Optimization Presets:
- BALANCED: Balanced between speed and quality
- SPEED: Optimize for maximum speed
- QUALITY: Optimize for maximum quality
- MEMORY: Optimize for minimum memory usage
- LATENCY: Optimize for minimum latency

Example:
    >>> from apourvoir.optimizer.launcher import OptimizedLauncher, LaunchOptions
    >>> launcher = OptimizedLauncher()
    >>> config = launcher.get_optimized_config()
    >>> launcher.launch()
"""

from .hardware_profile import HardwareProfile, HardwareTier
from .launch_config import LaunchConfig, OptimizationPreset
from .optimized_launcher import OptimizedLauncher, LaunchOptions, PerformanceMetrics

__all__ = [
    # Hardware
    "HardwareProfile",
    "HardwareTier",
    
    # Configuration
    "LaunchConfig",
    "OptimizationPreset",
    
    # Launcher
    "OptimizedLauncher",
    "LaunchOptions",
    "PerformanceMetrics",
]

__all__ = [
    # Hardware
    "HardwareProfile",
    "HardwareTier",
    
    # Configuration
    "LaunchConfig",
    "OptimizationPreset",
    
    # Launcher
    "OptimizedLauncher",
]
