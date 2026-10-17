"""
Hardware Profile for Apourvoir Optimized Launcher

Detects and profiles hardware capabilities to determine optimal configuration.

Hardware Tiers:
- TIER_1: 4GB VRAM, 8GB RAM, 6 cores (Entry-level)
- TIER_2: 8GB VRAM, 16GB RAM, 8-12 cores (Mid-range)
- TIER_3: 12GB VRAM, 32GB RAM, 12-16 cores (High-end)
- TIER_4: 16GB VRAM, 48GB RAM, 16-24 cores (Workstation)
- TIER_5: 24GB VRAM, 64GB RAM, 24+ cores (Machine de guerre)

Example:
    >>> profile = HardwareProfile()
    >>> tier = profile.detect_tier()
    >>> print(f"Detected tier: {tier.name}")
"""

import logging
import platform
import psutil
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)


class HardwareTier(Enum):
    """Hardware tiers for optimization presets."""
    TIER_1 = "tier_1"  # 4GB VRAM, 8GB RAM, 6 cores
    TIER_2 = "tier_2"  # 8GB VRAM, 16GB RAM, 8-12 cores
    TIER_3 = "tier_3"  # 12GB VRAM, 32GB RAM, 12-16 cores
    TIER_4 = "tier_4"  # 16GB VRAM, 48GB RAM, 16-24 cores
    TIER_5 = "tier_5"  # 24GB VRAM, 64GB RAM, 24+ cores


@dataclass
class GPUInfo:
    """GPU information."""
    id: int
    name: str
    vram_total_gb: float
    vram_free_gb: float
    vram_used_gb: float
    driver: str
    vendor: str
    compute_capability: Optional[float] = None
    
    def __repr__(self) -> str:
        return f"GPU(id={self.id}, name='{self.name}', vram={self.vram_total_gb}GB, free={self.vram_free_gb}GB)"


@dataclass
class CPUInfo:
    """CPU information."""
    physical_cores: int
    logical_cores: int
    frequency_mhz: float
    brand: str
    architecture: str
    l1_cache_kb: int
    l2_cache_kb: int
    l3_cache_kb: int
    supports_avx: bool
    supports_avx2: bool
    supports_avx512: bool
    
    def __repr__(self) -> str:
        return f"CPU(cores={self.logical_cores}, brand='{self.brand}', freq={self.frequency_mhz}MHz)"


@dataclass
class RAMInfo:
    """RAM information."""
    total_gb: float
    available_gb: float
    used_gb: float
    percent_used: float
    speed_mhz: Optional[float] = None
    
    def __repr__(self) -> str:
        return f"RAM(total={self.total_gb}GB, available={self.available_gb}GB, used={self.used_percent}%)"
    
    @property
    def used_percent(self) -> float:
        return self.percent_used


@dataclass
class StorageInfo:
    """Storage information."""
    total_gb: float
    free_gb: float
    used_gb: float
    mount_point: str
    
    def __repr__(self) -> str:
        return f"Storage(mount='{self.mount_point}', total={self.total_gb}GB, free={self.free_gb}GB)"


@dataclass
class HardwareProfile:
    """
    Complete hardware profile for optimization.
    
    Detects:
    - GPU(s): VRAM, vendor, compute capability
    - CPU: Cores, frequency, cache, instruction sets
    - RAM: Total, available, usage
    - Storage: Available space
    """
    gpus: List[GPUInfo] = field(default_factory=list)
    cpu: Optional[CPUInfo] = None
    ram: Optional[RAMInfo] = None
    storages: List[StorageInfo] = field(default_factory=list)
    
    # Detected tier
    tier: Optional[HardwareTier] = None
    
    def __post_init__(self):
        """Auto-detect hardware if not provided."""
        if not self.gpus or not self.cpu or not self.ram:
            self.detect_hardware()
        if not self.tier:
            self.detect_tier()
    
    def detect_hardware(self) -> None:
        """Detect all hardware components."""
        logger.info("Detecting hardware...")
        
        # Detect GPUs
        self.detect_gpus()
        
        # Detect CPU
        self.detect_cpu()
        
        # Detect RAM
        self.detect_ram()
        
        # Detect storage
        self.detect_storage()
        
        logger.info(f"Hardware detection complete: {len(self.gpus)} GPU(s), {self.cpu.logical_cores if self.cpu else 0} CPU cores, {self.ram.total_gb if self.ram else 0}GB RAM")
    
    def detect_gpus(self) -> None:
        """Detect available GPUs."""
        if not GPU_AVAILABLE:
            logger.warning("GPUtil not available, GPU detection skipped")
            return
        
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                gpu_info = GPUInfo(
                    id=i,
                    name=gpu.name,
                    vram_total_gb=round(gpu.memoryTotal / 1024, 2),
                    vram_free_gb=round(gpu.memoryFree / 1024, 2),
                    vram_used_gb=round(gpu.memoryUsed / 1024, 2),
                    driver=gpu.driver,
                    vendor="NVIDIA" if "nvidia" in gpu.driver.lower() or "nv" in gpu.name.lower() else \
                           "AMD" if "amd" in gpu.name.lower() or "radeon" in gpu.name.lower() else \
                           "Intel" if "intel" in gpu.name.lower() else "Unknown"
                )
                
                # Try to extract compute capability for NVIDIA
                if gpu_info.vendor == "NVIDIA":
                    gpu_info.compute_capability = self._extract_compute_capability(gpu.name)
                
                self.gpus.append(gpu_info)
                logger.info(f"  GPU {i}: {gpu_info}")
        except Exception as e:
            logger.error(f"Error detecting GPUs: {e}")
    
    def _extract_compute_capability(self, gpu_name: str) -> Optional[float]:
        """Extract compute capability from GPU name."""
        # Common patterns: RTX 3090 (8.6), RTX 4090 (8.9), etc.
        patterns = {
            r'RTX 4090': 8.9,
            r'RTX 4080': 8.9,
            r'RTX 4070': 8.9,
            r'RTX 4060': 8.9,
            r'RTX 3090': 8.6,
            r'RTX 3080': 8.6,
            r'RTX 3070': 8.6,
            r'RTX 3060': 8.6,
            r'RTX 2080': 7.5,
            r'RTX 2070': 7.5,
            r'RTX 2060': 7.5,
            r'GTX 1080': 6.1,
            r'GTX 1070': 6.1,
            r'GTX 1060': 6.1,
            r'A100': 8.0,
            r'H100': 9.0,
            r'V100': 7.0,
        }
        
        for pattern, capability in patterns.items():
            if re.search(pattern, gpu_name, re.IGNORECASE):
                return capability
        
        return None
    
    def detect_cpu(self) -> None:
        """Detect CPU information."""
        try:
            # Physical cores
            physical_cores = psutil.cpu_count(logical=False)
            logical_cores = psutil.cpu_count(logical=True)
            
            # CPU frequency
            freq = psutil.cpu_freq()
            frequency_mhz = freq.max if freq else 0
            
            # Brand and architecture
            brand = platform.processor() or "Unknown"
            architecture = platform.machine() or "Unknown"
            
            # Cache sizes (approximate)
            l1_cache = 32 * physical_cores  # 32KB per core
            l2_cache = 256 * physical_cores  # 256KB per core
            l3_cache = 8 * 1024 * physical_cores  # 8MB per core (approximate)
            
            # Check instruction sets
            import os
            supports_avx = self._check_cpu_flag("avx")
            supports_avx2 = self._check_cpu_flag("avx2")
            supports_avx512 = self._check_cpu_flag("avx512f")
            
            self.cpu = CPUInfo(
                physical_cores=physical_cores,
                logical_cores=logical_cores,
                frequency_mhz=frequency_mhz,
                brand=brand,
                architecture=architecture,
                l1_cache_kb=l1_cache,
                l2_cache_kb=l2_cache,
                l3_cache_kb=l3_cache,
                supports_avx=supports_avx,
                supports_avx2=supports_avx2,
                supports_avx512=supports_avx512
            )
            
            logger.info(f"  CPU: {self.cpu}")
        except Exception as e:
            logger.error(f"Error detecting CPU: {e}")
    
    def _check_cpu_flag(self, flag: str) -> bool:
        """Check if CPU supports a specific instruction set."""
        try:
            # Try to read /proc/cpuinfo on Linux
            if platform.system() == "Linux":
                with open("/proc/cpuinfo", "r") as f:
                    content = f.read()
                    return flag in content
            elif platform.system() == "Windows":
                # Use os.cpu_count and try to detect via registry or WMI
                import subprocess
                try:
                    result = subprocess.run(
                        ["wmic", "cpu", "get", "name"],
                        capture_output=True, text=True
                    )
                    # For now, assume modern CPUs support AVX2
                    return True
                except:
                    return False
            else:
                return False
        except:
            return False
    
    def detect_ram(self) -> None:
        """Detect RAM information."""
        try:
            ram = psutil.virtual_memory()
            self.ram = RAMInfo(
                total_gb=round(ram.total / 1024 / 1024 / 1024, 2),
                available_gb=round(ram.available / 1024 / 1024 / 1024, 2),
                used_gb=round(ram.used / 1024 / 1024 / 1024, 2),
                percent_used=ram.percent
            )
            logger.info(f"  RAM: {self.ram}")
        except Exception as e:
            logger.error(f"Error detecting RAM: {e}")
    
    def detect_storage(self) -> None:
        """Detect storage information."""
        try:
            partitions = psutil.disk_partitions()
            for partition in partitions:
                if partition.fstype and 'tmpfs' not in partition.fstype.lower():
                    usage = psutil.disk_usage(partition.mountpoint)
                    storage = StorageInfo(
                        total_gb=round(usage.total / 1024 / 1024 / 1024, 2),
                        free_gb=round(usage.free / 1024 / 1024 / 1024, 2),
                        used_gb=round(usage.used / 1024 / 1024 / 1024, 2),
                        mount_point=partition.mountpoint
                    )
                    self.storages.append(storage)
                    logger.info(f"  Storage: {storage}")
        except Exception as e:
            logger.error(f"Error detecting storage: {e}")
    
    def detect_tier(self) -> HardwareTier:
        """
        Detect hardware tier based on capabilities.
        
        Returns:
            HardwareTier: The detected tier
        """
        # Get primary GPU (most VRAM)
        primary_gpu = max(self.gpus, key=lambda g: g.vram_total_gb) if self.gpus else None
        total_vram = primary_gpu.vram_total_gb if primary_gpu else 0
        
        # Get total RAM
        total_ram = self.ram.total_gb if self.ram else 0
        
        # Get CPU cores
        total_cores = self.cpu.logical_cores if self.cpu else 0
        
        # Determine tier
        if total_vram >= 24 and total_ram >= 64 and total_cores >= 24:
            self.tier = HardwareTier.TIER_5
        elif total_vram >= 16 and total_ram >= 48 and total_cores >= 16:
            self.tier = HardwareTier.TIER_4
        elif total_vram >= 12 and total_ram >= 32 and total_cores >= 12:
            self.tier = HardwareTier.TIER_3
        elif total_vram >= 8 and total_ram >= 16 and total_cores >= 8:
            self.tier = HardwareTier.TIER_2
        else:
            self.tier = HardwareTier.TIER_1
        
        logger.info(f"Detected hardware tier: {self.tier.name}")
        return self.tier
    
    def get_tier_name(self) -> str:
        """Get human-readable tier name."""
        tier_names = {
            HardwareTier.TIER_1: "Entry-level (4GB VRAM, 8GB RAM, 6 cores)",
            HardwareTier.TIER_2: "Mid-range (8GB VRAM, 16GB RAM, 8-12 cores)",
            HardwareTier.TIER_3: "High-end (12GB VRAM, 32GB RAM, 12-16 cores)",
            HardwareTier.TIER_4: "Workstation (16GB VRAM, 48GB RAM, 16-24 cores)",
            HardwareTier.TIER_5: "Machine de guerre (24GB+ VRAM, 64GB+ RAM, 24+ cores)",
        }
        return tier_names.get(self.tier, "Unknown")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the hardware profile."""
        return {
            "tier": self.tier.name if self.tier else "unknown",
            "tier_description": self.get_tier_name(),
            "gpus": [
                {
                    "id": gpu.id,
                    "name": gpu.name,
                    "vram_total_gb": gpu.vram_total_gb,
                    "vram_free_gb": gpu.vram_free_gb,
                    "vendor": gpu.vendor
                }
                for gpu in self.gpus
            ],
            "cpu": {
                "physical_cores": self.cpu.physical_cores if self.cpu else 0,
                "logical_cores": self.cpu.logical_cores if self.cpu else 0,
                "frequency_mhz": self.cpu.frequency_mhz if self.cpu else 0,
                "brand": self.cpu.brand if self.cpu else "Unknown",
                "supports_avx": self.cpu.supports_avx if self.cpu else False,
                "supports_avx2": self.cpu.supports_avx2 if self.cpu else False,
                "supports_avx512": self.cpu.supports_avx512 if self.cpu else False,
            },
            "ram": {
                "total_gb": self.ram.total_gb if self.ram else 0,
                "available_gb": self.ram.available_gb if self.ram else 0,
                "used_percent": self.ram.percent_used if self.ram else 0,
            },
            "storages": [
                {
                    "mount_point": s.mount_point,
                    "total_gb": s.total_gb,
                    "free_gb": s.free_gb
                }
                for s in self.storages
            ]
        }
