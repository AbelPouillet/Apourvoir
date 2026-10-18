#!/usr/bin/env python3
"""
Apourvoir Optimized Launcher

Main launch script for Apourvoir with automatic hardware optimization.

Usage:
    python scripts/launcher/launch.py [--model mistralai/Mixtral-8x7B-v0.1]
                                    [--preset balanced|speed|quality|memory|latency]
                                    [--no-api] [--no-ui]
                                    [--config config.json]
                                    [--log-level DEBUG|INFO|WARNING|ERROR]

Examples:
    # Launch with auto-detection
    python scripts/launcher/launch.py
    
    # Launch with specific model
    python scripts/launcher/launch.py --model mistralai/Mixtral-8x22B-v0.1
    
    # Launch with speed optimization
    python scripts/launcher/launch.py --preset speed
    
    # Launch with custom config
    python scripts/launcher/launch.py --config my_config.json
    
    # Launch API only
    python scripts/launcher/launch.py --no-ui
"""

import argparse
import json
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from apourvoir.optimizer.launcher import (
    OptimizedLauncher,
    LaunchOptions,
    HardwareProfile,
    HardwareTier,
    OptimizationPreset
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('apourvoir_launcher.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Apourvoir Optimized Launcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model options
    parser.add_argument(
        '--model',
        type=str,
        default='mistralai/Mixtral-8x7B-v0.1',
        help='Model to load (default: mistralai/Mixtral-8x7B-v0.1)'
    )
    
    # Quantization method
    parser.add_argument(
        '--quant-method',
        type=str,
        default='canonical',
        choices=['canonical', 'none', 'bitsandbytes', 'gguf', 'awq'],
        help='Quantization method (default: canonical). The inference engine supports canonical and none'
    )
    
    # Inference engine
    parser.add_argument('--no-inference', action='store_true',
                        help='Do not load a model (completions answer 501)')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'],
                        help='Inference device (default: cuda if available, else cpu)')
    parser.add_argument('--dtype', type=str, default=None, choices=['float32', 'float16', 'bfloat16'],
                        help='Weights dtype (default: float16 on cuda, float32 on cpu)')
    parser.add_argument('--n-bits', type=int, default=None,
                        help='Bits for canonical quantization (default: from the hardware tier configuration)')
    
    # Optimization preset
    parser.add_argument(
        '--preset',
        type=str,
        default='balanced',
        choices=['balanced', 'speed', 'quality', 'memory', 'latency'],
        help='Optimization preset (default: balanced)'
    )
    
    # API options
    parser.add_argument(
        '--no-api',
        action='store_true',
        help='Disable API server'
    )
    parser.add_argument(
        '--api-host',
        type=str,
        default='0.0.0.0',
        help='API server host (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--api-port',
        type=int,
        default=8080,
        help='API server port (default: 8080)'
    )
    
    # UI options
    parser.add_argument(
        '--no-ui',
        action='store_true',
        help='Disable web UI'
    )
    parser.add_argument(
        '--ui-port',
        type=int,
        default=8000,
        help='Web UI port (default: 8000)'
    )
    
    # Config file
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Load configuration from file'
    )
    parser.add_argument(
        '--save-config',
        type=str,
        default=None,
        help='Save configuration to file'
    )
    
    # Logging
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Log level (default: INFO)'
    )
    
    # Info mode
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show hardware info and exit'
    )
    
    # Validate
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate configuration and exit'
    )
    
    return parser.parse_args()


def get_preset(preset_name: str) -> OptimizationPreset:
    """Get OptimizationPreset from string."""
    presets = {
        'balanced': OptimizationPreset.BALANCED,
        'speed': OptimizationPreset.SPEED,
        'quality': OptimizationPreset.QUALITY,
        'memory': OptimizationPreset.MEMORY,
        'latency': OptimizationPreset.LATENCY
    }
    return presets.get(preset_name, OptimizationPreset.BALANCED)


def show_hardware_info(profile: HardwareProfile) -> None:
    """Show detailed hardware information."""
    print("\n" + "=" * 60)
    print("HARDWARE INFORMATION")
    print("=" * 60)
    
    # Tier
    print(f"\nTier: {profile.get_tier_name()}")
    
    # GPUs
    print("\nGPUs:")
    if profile.gpus:
        for gpu in profile.gpus:
            print(f"  - {gpu.name}")
            print(f"    VRAM: {gpu.vram_total_gb}GB (free: {gpu.vram_free_gb}GB, used: {gpu.vram_used_gb}GB)")
            print(f"    Vendor: {gpu.vendor}")
            if gpu.compute_capability:
                print(f"    Compute Capability: {gpu.compute_capability}")
    else:
        print("  No GPUs detected")
    
    # CPU
    print("\nCPU:")
    if profile.cpu:
        print(f"  Brand: {profile.cpu.brand}")
        print(f"  Architecture: {profile.cpu.architecture}")
        print(f"  Cores: {profile.cpu.physical_cores} physical, {profile.cpu.logical_cores} logical")
        print(f"  Frequency: {profile.cpu.frequency_mhz} MHz")
        print(f"  Cache: L1={profile.cpu.l1_cache_kb}KB, L2={profile.cpu.l2_cache_kb}KB, L3={profile.cpu.l3_cache_kb}KB")
        print(f"  AVX: {'Yes' if profile.cpu.supports_avx else 'No'}")
        print(f"  AVX2: {'Yes' if profile.cpu.supports_avx2 else 'No'}")
        print(f"  AVX512: {'Yes' if profile.cpu.supports_avx512 else 'No'}")
    else:
        print("  CPU information not available")
    
    # RAM
    print("\nRAM:")
    if profile.ram:
        print(f"  Total: {profile.ram.total_gb}GB")
        print(f"  Available: {profile.ram.available_gb}GB")
        print(f"  Used: {profile.ram.used_gb}GB ({profile.ram.percent_used}%)")
    else:
        print("  RAM information not available")
    
    # Storage
    print("\nStorage:")
    if profile.storages:
        for storage in profile.storages:
            print(f"  {storage.mount_point}: {storage.total_gb}GB total, {storage.free_gb}GB free")
    else:
        print("  Storage information not available")
    
    print("=" * 60 + "\n")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set log level
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    
    logger.info("Apourvoir Optimized Launcher starting...")
    
    # Get preset
    preset = get_preset(args.preset)
    
    # Create launch options
    options = LaunchOptions(
        model=args.model,
        quant_method=args.quant_method,
        enable_api=not args.no_api,
        api_host=args.api_host,
        api_port=args.api_port,
        enable_ui=not args.no_ui,
        ui_port=args.ui_port,
        log_level=args.log_level,
        enable_inference=not args.no_inference,
        device=args.device,
        dtype=args.dtype,
        n_bits=args.n_bits,
    )
    
    # Create launcher
    launcher = OptimizedLauncher(options=options, preset=preset)
    
    # Info mode
    if args.info:
        show_hardware_info(launcher.get_hardware_profile())
        return
    
    # Validate mode
    if args.validate:
        validation = launcher.validate_resources()
        print("\n" + "=" * 60)
        print("CONFIGURATION VALIDATION")
        print("=" * 60)
        
        if validation["valid"]:
            print("\n✓ Configuration is valid")
        else:
            print("\n✗ Configuration has issues:")
            for warning in validation["warnings"]:
                print(f"  - {warning}")
        
        # Show recommendations
        recommendations = launcher.get_recommendations()
        if "notes" in recommendations:
            print("\nRecommendations:")
            for note in recommendations["notes"]:
                print(f"  - {note}")
        
        print("=" * 60 + "\n")
        return
    
    # Load config if specified
    if args.config and os.path.exists(args.config):
        launcher.load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    
    # Save config if specified
    if args.save_config:
        launcher.save_config(args.save_config)
        logger.info(f"Configuration saved to {args.save_config}")
        return
    
    # Launch and keep the servers running until Ctrl+C
    try:
        launcher.launch(block=True)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, stopping...")
    except Exception as e:
        logger.error(f"Error during launch: {e}")
        import traceback
        traceback.print_exc()
        launcher.stop()
        sys.exit(1)
    finally:
        launcher.stop()


if __name__ == "__main__":
    main()
