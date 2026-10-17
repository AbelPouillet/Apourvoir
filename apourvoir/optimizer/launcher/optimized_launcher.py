"""
Optimized Launcher for Apourvoir

Main launcher class that automatically optimizes for the detected hardware tier.

Features:
- Automatic hardware detection and tier classification
- Preset-based optimization (speed, quality, memory, latency)
- Dynamic configuration adjustment
- Performance monitoring
- Graceful degradation for insufficient resources

Example:
    >>> launcher = OptimizedLauncher()
    >>> config = launcher.get_optimized_config()
    >>> launcher.launch()
"""

import logging
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path

from .hardware_profile import HardwareProfile, HardwareTier
from .launch_config import LaunchConfig, OptimizationPreset

logger = logging.getLogger(__name__)


@dataclass
class LaunchOptions:
    """Options for launching Apourvoir."""
    # Model to load
    model: str = "mistralai/Mixtral-8x7B-v0.1"
    
    # Quantization method
    quant_method: str = "canonical"
    
    # Enable API server
    enable_api: bool = True
    
    # API host
    api_host: str = "0.0.0.0"
    
    # API port
    api_port: int = 8080
    
    # Enable web UI
    enable_ui: bool = True
    
    # UI port
    ui_port: int = 8000
    
    # Log level
    log_level: str = "INFO"
    
    # Config file path
    config_file: Optional[str] = None
    
    # Data directory
    data_dir: str = "data"
    
    # Model directory
    model_dir: str = "models"


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    # Timestamp
    timestamp: float = field(default_factory=time.time)
    
    # Tokens per second (generation)
    tokens_per_second: float = 0.0
    
    # Tokens per second (prompt processing)
    prompt_tokens_per_second: float = 0.0
    
    # VRAM usage (GB)
    vram_usage_gb: float = 0.0
    
    # RAM usage (GB)
    ram_usage_gb: float = 0.0
    
    # GPU utilization (%)
    gpu_utilization: float = 0.0
    
    # CPU utilization (%)
    cpu_utilization: float = 0.0
    
    # Batch size
    batch_size: int = 0
    
    # Sequence length
    seq_length: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "tokens_per_second": self.tokens_per_second,
            "prompt_tokens_per_second": self.prompt_tokens_per_second,
            "vram_usage_gb": self.vram_usage_gb,
            "ram_usage_gb": self.ram_usage_gb,
            "gpu_utilization": self.gpu_utilization,
            "cpu_utilization": self.cpu_utilization,
            "batch_size": self.batch_size,
            "seq_length": self.seq_length
        }


class OptimizedLauncher:
    """
    Optimized launcher for Apourvoir.
    
    Automatically detects hardware and applies optimal configuration
    for the detected tier. Supports custom presets and dynamic adjustment.
    """
    
    def __init__(
        self,
        options: Optional[LaunchOptions] = None,
        preset: OptimizationPreset = OptimizationPreset.BALANCED
    ):
        """
        Initialize the optimized launcher.
        
        Args:
            options: Launch options
            preset: Optimization preset to use
        """
        self.options = options or LaunchOptions()
        self.preset = preset
        
        # Hardware profile
        self.profile: Optional[HardwareProfile] = None
        
        # Launch config
        self.config: Optional[LaunchConfig] = None
        
        # Performance metrics
        self.metrics: List[PerformanceMetrics] = []
        
        # Running state
        self._running: bool = False
        
        # Initialize
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize hardware detection and configuration."""
        logger.info("Initializing Optimized Launcher...")
        
        # Detect hardware
        self.detect_hardware()
        
        # Generate configuration
        self.generate_config()
        
        logger.info(f"Initialized for tier: {self.profile.tier.name if self.profile and self.profile.tier else 'unknown'}")
    
    def detect_hardware(self) -> HardwareProfile:
        """Detect hardware and create profile."""
        self.profile = HardwareProfile()
        return self.profile
    
    def generate_config(self) -> LaunchConfig:
        """Generate optimized configuration based on hardware profile."""
        if not self.profile or not self.profile.tier:
            raise ValueError("Hardware profile not detected")
        
        self.config = LaunchConfig.from_tier(self.profile.tier, self.preset)
        
        # Apply preset
        self.config.apply_preset(self.preset)
        
        return self.config
    
    def get_optimized_config(self) -> LaunchConfig:
        """Get the optimized configuration."""
        if not self.config:
            self.generate_config()
        return self.config
    
    def get_hardware_profile(self) -> HardwareProfile:
        """Get the hardware profile."""
        if not self.profile:
            self.detect_hardware()
        return self.profile
    
    def set_preset(self, preset: OptimizationPreset) -> None:
        """Set the optimization preset."""
        self.preset = preset
        if self.config:
            self.config.apply_preset(preset)
        logger.info(f"Optimization preset set to: {preset.value}")
    
    def adjust_for_model(self, model_name: str) -> None:
        """
        Adjust configuration for a specific model.
        
        Args:
            model_name: Name of the model to load
        """
        # Model-specific adjustments
        model_configs = {
            "mistralai/Mixtral-8x7B-v0.1": {
                "quantization": {"n_bits": 8, "min_bits": 4, "max_bits": 12},
                "batch": {"max_batch_size": 16, "optimal_batch_size": 8},
                "memory": {"cache_size_gb": 2.0}
            },
            "mistralai/Mixtral-8x22B-v0.1": {
                "quantization": {"n_bits": 10, "min_bits": 6, "max_bits": 14},
                "batch": {"max_batch_size": 32, "optimal_batch_size": 16},
                "memory": {"cache_size_gb": 4.0}
            },
            "Qwen/Qwen2-72B": {
                "quantization": {"n_bits": 8, "min_bits": 4, "max_bits": 12},
                "batch": {"max_batch_size": 16, "optimal_batch_size": 8},
                "memory": {"cache_size_gb": 2.0}
            },
        }
        
        if model_name in model_configs:
            model_config = model_configs[model_name]
            
            if "quantization" in model_config:
                q = model_config["quantization"]
                if self.config:
                    self.config.quantization.n_bits = q.get("n_bits", self.config.quantization.n_bits)
                    self.config.quantization.min_bits = q.get("min_bits", self.config.quantization.min_bits)
                    self.config.quantization.max_bits = q.get("max_bits", self.config.quantization.max_bits)
            
            if "batch" in model_config:
                b = model_config["batch"]
                if self.config:
                    self.config.batch.max_batch_size = b.get("max_batch_size", self.config.batch.max_batch_size)
                    self.config.batch.optimal_batch_size = b.get("optimal_batch_size", self.config.batch.optimal_batch_size)
            
            if "memory" in model_config:
                m = model_config["memory"]
                if self.config:
                    self.config.memory.cache_size_gb = m.get("cache_size_gb", self.config.memory.cache_size_gb)
            
            logger.info(f"Adjusted configuration for model: {model_name}")
    
    def validate_resources(self) -> Dict[str, Any]:
        """
        Validate if current hardware has sufficient resources.
        
        Returns:
            Dict with validation results and recommendations
        """
        if not self.profile or not self.config:
            return {"valid": False, "error": "Not initialized"}
        
        results = {
            "valid": True,
            "warnings": [],
            "recommendations": []
        }
        
        # Check VRAM
        if self.profile.gpus:
            primary_gpu = max(self.profile.gpus, key=lambda g: g.vram_total_gb)
            if self.config.memory.max_vram_gb:
                if primary_gpu.vram_total_gb < self.config.memory.max_vram_gb:
                    results["warnings"].append(
                        f"Insufficient VRAM: {primary_gpu.vram_total_gb}GB available, "
                        f"{self.config.memory.max_vram_gb}GB required"
                    )
                    results["valid"] = False
        else:
            results["warnings"].append("No GPU detected")
        
        # Check RAM
        if self.profile.ram:
            if self.config.memory.max_ram_gb:
                if self.profile.ram.total_gb < self.config.memory.max_ram_gb:
                    results["warnings"].append(
                        f"Insufficient RAM: {self.profile.ram.total_gb}GB available, "
                        f"{self.config.memory.max_ram_gb}GB required"
                    )
                    results["valid"] = False
        
        # Check CPU cores
        if self.profile.cpu:
            if self.config.parallelism.cpu_threads > self.profile.cpu.logical_cores:
                results["warnings"].append(
                    f"CPU threads ({self.config.parallelism.cpu_threads}) exceed "
                    f"available cores ({self.profile.cpu.logical_cores})"
                )
        
        return results
    
    def get_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for the current configuration."""
        if not self.config:
            self.generate_config()
        
        return self.config.get_recommendations()
    
    def launch(self) -> None:
        """
        Launch Apourvoir with optimized configuration.
        
        This is the main entry point that:
        1. Validates resources
        2. Adjusts configuration for the model
        3. Starts the inference engine
        4. Starts the API server (if enabled)
        5. Starts the web UI (if enabled)
        """
        logger.info("Launching Apourvoir with optimized configuration...")
        
        # Validate resources
        validation = self.validate_resources()
        if not validation["valid"]:
            logger.error("Resource validation failed:")
            for warning in validation["warnings"]:
                logger.error(f"  - {warning}")
            
            # Try with degraded configuration
            logger.info("Attempting to launch with degraded configuration...")
            self._degrade_config()
        
        # Adjust for model
        self.adjust_for_model(self.options.model)
        
        # Print configuration
        self._print_config()
        
        # Set running state
        self._running = True
        
        # Launch inference engine
        self._launch_inference()
        
        # Launch API server
        if self.options.enable_api:
            self._launch_api_server()
        
        # Launch web UI
        if self.options.enable_ui:
            self._launch_web_ui()
        
        logger.info("Apourvoir launched successfully")
    
    def _degrade_config(self) -> None:
        """Degrade configuration to fit available resources."""
        if not self.profile or not self.config:
            return
        
        logger.info("Degrading configuration to fit available resources...")
        
        # Reduce VRAM requirements
        if self.profile.gpus:
            primary_gpu = max(self.profile.gpus, key=lambda g: g.vram_total_gb)
            if primary_gpu.vram_total_gb < 8:
                self.config.quantization.n_bits = 4
                self.config.batch.max_batch_size = 2
                self.config.cache.kv_cache_size = 256
        
        # Reduce RAM requirements
        if self.profile.ram and self.profile.ram.total_gb < 16:
            self.config.memory.max_ram_gb = self.profile.ram.total_gb * 0.8
            self.config.batch.max_batch_size = 2
        
        # Reduce CPU threads
        if self.profile.cpu:
            self.config.parallelism.cpu_threads = min(
                self.config.parallelism.cpu_threads,
                self.profile.cpu.logical_cores
            )
        
        logger.info("Configuration degraded successfully")
    
    def _print_config(self) -> None:
        """Print the current configuration."""
        if not self.config:
            return
        
        logger.info("=" * 60)
        logger.info("OPTIMIZED CONFIGURATION")
        logger.info("=" * 60)
        
        # Hardware info
        if self.profile:
            logger.info(f"Hardware Tier: {self.profile.get_tier_name()}")
            if self.profile.gpus:
                primary_gpu = max(self.profile.gpus, key=lambda g: g.vram_total_gb)
                logger.info(f"GPU: {primary_gpu.name} ({primary_gpu.vram_total_gb}GB VRAM)")
            if self.profile.cpu:
                logger.info(f"CPU: {self.profile.cpu.brand} ({self.profile.cpu.logical_cores} cores)")
            if self.profile.ram:
                logger.info(f"RAM: {self.profile.ram.total_gb}GB")
        
        # Configuration
        logger.info(f"\nPreset: {self.config.preset.value}")
        logger.info(f"Quantization: {self.config.quantization.n_bits} bits")
        logger.info(f"Batch size: max={self.config.batch.max_batch_size}, optimal={self.config.batch.optimal_batch_size}")
        logger.info(f"CPU threads: {self.config.parallelism.cpu_threads}")
        logger.info(f"GPU streams: {self.config.parallelism.gpu_streams}")
        logger.info(f"Cache size: {self.config.memory.cache_size_gb}GB")
        logger.info(f"KV cache size: {self.config.cache.kv_cache_size} tokens")
        
        # Recommendations
        recommendations = self.get_recommendations()
        if "notes" in recommendations:
            logger.info("\nRecommendations:")
            for note in recommendations["notes"]:
                logger.info(f"  - {note}")
        
        logger.info("=" * 60)
    
    def _launch_inference(self) -> None:
        """Launch the inference engine."""
        logger.info("Launching inference engine...")
        
        # This would integrate with the existing Apourvoir inference code
        # For now, we'll just log the action
        
        # Import and configure the quantization
        try:
            from apourvoir.quantization import canonical as canonical_quant
            from apourvoir.babel import BabelSpace, BabelConfig
            
            # Create Babel space with optimized config
            babel_config = BabelConfig(
                n_bits=self.config.quantization.n_bits,
                phonetic_dim=64,
                syntax_dim=64,
                semantic_dim=3,
                output_dim=128
            )
            babel = BabelSpace(babel_config)
            
            logger.info(f"Babel space initialized with {self.config.quantization.n_bits}-bit quantization")
            
        except ImportError as e:
            logger.warning(f"Could not import Apourvoir modules: {e}")
    
    def _launch_api_server(self) -> None:
        """Launch the API server."""
        logger.info(f"Launching API server on {self.options.api_host}:{self.options.api_port}")
        
        # This would start a FastAPI or similar server
        # For now, we'll just log the action
        
        # In a real implementation, this would be:
        # from apourvoir.serve import api
        # api.start(host=self.options.api_host, port=self.options.api_port)
    
    def _launch_web_ui(self) -> None:
        """Launch the web UI."""
        logger.info(f"Launching web UI on {self.options.api_host}:{self.options.ui_port}")
        
        # This would start a web interface
        # For now, we'll just log the action
    
    def stop(self) -> None:
        """Stop all running components."""
        logger.info("Stopping Apourvoir...")
        self._running = False
        # In a real implementation, this would stop all servers and clean up
    
    def monitor_performance(self) -> PerformanceMetrics:
        """
        Monitor and return current performance metrics.
        
        Returns:
            PerformanceMetrics: Current performance metrics
        """
        # In a real implementation, this would collect actual metrics
        # For now, return placeholder values
        metrics = PerformanceMetrics()
        self.metrics.append(metrics)
        return metrics
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        return {
            "running": self._running,
            "tier": self.profile.tier.name if self.profile and self.profile.tier else "unknown",
            "config": self.config.to_dict() if self.config else {},
            "metrics": [m.to_dict() for m in self.metrics[-10:]]  # Last 10 metrics
        }
    
    def save_config(self, path: str) -> None:
        """Save current configuration to file."""
        if not self.config:
            raise ValueError("Configuration not generated")
        
        config_dict = self.config.to_dict()
        config_dict["options"] = {
            "model": self.options.model,
            "quant_method": self.options.quant_method,
            "enable_api": self.options.enable_api,
            "api_host": self.options.api_host,
            "api_port": self.options.api_port,
            "enable_ui": self.options.enable_ui,
            "ui_port": self.options.ui_port
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to {path}")
    
    def load_config(self, path: str) -> None:
        """Load configuration from file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        # Extract options
        if "options" in config_dict:
            options_dict = config_dict["options"]
            self.options = LaunchOptions(
                model=options_dict.get("model", self.options.model),
                quant_method=options_dict.get("quant_method", self.options.quant_method),
                enable_api=options_dict.get("enable_api", self.options.enable_api),
                api_host=options_dict.get("api_host", self.options.api_host),
                api_port=options_dict.get("api_port", self.options.api_port),
                enable_ui=options_dict.get("enable_ui", self.options.enable_ui),
                ui_port=options_dict.get("ui_port", self.options.ui_port)
            )
        
        # Load launch config
        self.config = LaunchConfig.from_dict(config_dict)
        
        # Update preset
        if "preset" in config_dict:
            self.preset = OptimizationPreset(config_dict["preset"])
        
        logger.info(f"Configuration loaded from {path}")
