"""
Memory Manager for Apourvoir SDK

Intelligent memory management for VRAM and RAM with caching,
eviection policies, and automatic cleanup.
"""

import logging
import time
import threading
import weakref
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Set, OrderedDict
from collections import OrderedDict
from enum import Enum

import torch
import numpy as np
import psutil
import gc

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory."""
    CPU = "cpu"
    CUDA = "cuda"
    PINNED = "pinned"


@dataclass
class MemoryBlock:
    """Represents a block of allocated memory."""
    name: str
    tensor: torch.Tensor
    size_bytes: int
    memory_type: MemoryType
    timestamp: float = field(default_factory=time.time)
    priority: int = 0
    locked: bool = False
    
    def __post_init__(self):
        if not isinstance(self.tensor, torch.Tensor):
            raise ValueError("tensor must be a torch.Tensor")
        self.size_bytes = self.tensor.numel() * self.tensor.element_size()
        self.memory_type = MemoryType.CUDA if self.tensor.is_cuda else MemoryType.CPU
    
    def release(self) -> None:
        """Release the memory block."""
        if self.locked:
            logger.warning(f"Cannot release locked memory block: {self.name}")
            return
        
        if self.memory_type == MemoryType.CUDA:
            del self.tensor
            torch.cuda.empty_cache()
        else:
            del self.tensor
        
        gc.collect()


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_allocated_gb: float = 0.0
    used_gb: float = 0.0
    free_gb: float = 0.0
    peak_gb: float = 0.0
    num_blocks: int = 0
    num_locked: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_allocated_gb": self.total_allocated_gb,
            "used_gb": self.used_gb,
            "free_gb": self.free_gb,
            "peak_gb": self.peak_gb,
            "num_blocks": self.num_blocks,
            "num_locked": self.num_locked,
        }


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    SMALLEST = "smallest" # Smallest blocks first
    LARGEST = "largest"   # Largest blocks first
    OLDER = "older"       # Oldest blocks first
    PRIORITY = "priority" # Lowest priority first


class MemoryManager:
    """
    Intelligent memory manager for VRAM and RAM.
    
    Features:
    - Memory tracking for CPU and GPU
    - Caching with configurable eviction policies
    - Automatic cleanup when memory is low
    - Locking mechanism for critical tensors
    - Memory statistics and monitoring
    
    Example:
        >>> from apourvoir.optimizer import MemoryManager
        >>> manager = MemoryManager(max_vram_gb=16, max_ram_gb=32)
        >>> 
        >>> # Allocate memory
        >>> tensor = torch.randn(1000, 1000)
        >>> block = manager.allocate(tensor, "my_tensor")
        >>> 
        >>> # Check memory usage
        >>> stats = manager.get_stats()
        >>> print(f"VRAM used: {stats.vram_used_gb:.1f} GB")
        >>> 
        >>> # Cleanup
        >>> manager.cleanup(threshold=0.8)  # Clean if >80% used
    """
    
    def __init__(
        self,
        max_vram_gb: float = 16.0,
        max_ram_gb: float = 32.0,
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
        cleanup_threshold: float = 0.8,
        min_free_gb: float = 1.0,
        auto_cleanup: bool = True,
        cleanup_interval: float = 5.0,
    ):
        """
        Initialize the memory manager.
        
        Args:
            max_vram_gb: Maximum VRAM to use (GB)
            max_ram_gb: Maximum RAM to use (GB)
            eviction_policy: Policy for evicting cached blocks
            cleanup_threshold: Cleanup when memory usage exceeds this fraction
            min_free_gb: Minimum free memory to maintain (GB)
            auto_cleanup: Whether to automatically cleanup
            cleanup_interval: Interval for auto cleanup (seconds)
        """
        self.max_vram_gb = max_vram_gb
        self.max_ram_gb = max_ram_gb
        self.eviction_policy = eviction_policy
        self.cleanup_threshold = cleanup_threshold
        self.min_free_gb = min_free_gb
        self.auto_cleanup = auto_cleanup
        self.cleanup_interval = cleanup_interval
        
        # Memory tracking
        self._cpu_memory_used = 0.0  # bytes
        self._gpu_memory_used = 0.0  # bytes
        self._peak_cpu_memory = 0.0
        self._peak_gpu_memory = 0.0
        
        # Memory blocks
        self._blocks: Dict[str, MemoryBlock] = {}
        self._access_count: Dict[str, int] = {}
        self._last_access: Dict[str, float] = {}
        
        # Locking
        self._lock = threading.Lock()
        self._cleanup_lock = threading.Lock()
        
        # Auto cleanup thread
        self._cleanup_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Callbacks
        self._oom_callbacks: List[Callable[[], None]] = []
        self._cleanup_callbacks: List[Callable[[float, float], None]] = []
        
        # Start auto cleanup if enabled
        if auto_cleanup:
            self.start_auto_cleanup()
        
        logger.info(f"MemoryManager initialized: VRAM={max_vram_gb}GB, RAM={max_ram_gb}GB, "
                   f"policy={eviction_policy.value}")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop_auto_cleanup()
        self.clear()
    
    def start_auto_cleanup(self) -> None:
        """Start automatic cleanup thread."""
        if self._running:
            return
        
        self._running = True
        self._cleanup_thread = threading.Thread(
            target=self._auto_cleanup_loop,
            daemon=True,
        )
        self._cleanup_thread.start()
        logger.info("Auto cleanup thread started")
    
    def stop_auto_cleanup(self) -> None:
        """Stop automatic cleanup thread."""
        self._running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=1.0)
            self._cleanup_thread = None
        logger.info("Auto cleanup thread stopped")
    
    def _auto_cleanup_loop(self) -> None:
        """Background thread for automatic cleanup."""
        while self._running:
            time.sleep(self.cleanup_interval)
            try:
                self.cleanup()
            except Exception as e:
                logger.error(f"Error in auto cleanup: {e}")
    
    def allocate(
        self,
        tensor: torch.Tensor,
        name: str,
        priority: int = 0,
        locked: bool = False,
    ) -> MemoryBlock:
        """
        Allocate and track a memory block.
        
        Args:
            tensor: Tensor to allocate
            name: Name for the memory block
            priority: Priority (higher = more important)
            locked: Whether to lock the block from eviction
        
        Returns:
            MemoryBlock: Allocated memory block
        
        Raises:
            ValueError: If name already exists
            RuntimeError: If allocation would exceed memory limits
        """
        with self._lock:
            if name in self._blocks:
                raise ValueError(f"Memory block '{name}' already exists")
            
            # Create memory block
            block = MemoryBlock(
                name=name,
                tensor=tensor,
                memory_type=MemoryType.CUDA if tensor.is_cuda else MemoryType.CPU,
                priority=priority,
                locked=locked,
            )
            
            # Track size
            size_gb = block.size_bytes / (1024**3)
            
            if block.memory_type == MemoryType.CUDA:
                # Check VRAM limit
                if self._gpu_memory_used + block.size_bytes > self.max_vram_gb * (1024**3):
                    # Try cleanup first
                    self._cleanup_gpu()
                    if self._gpu_memory_used + block.size_bytes > self.max_vram_gb * (1024**3):
                        raise RuntimeError(
                            f"Allocation of {size_gb:.2f} GB would exceed VRAM limit "
                            f"({self.max_vram_gb} GB). Current usage: {self._gpu_memory_used / (1024**3):.2f} GB"
                        )
                self._gpu_memory_used += block.size_bytes
                self._peak_gpu_memory = max(self._peak_gpu_memory, self._gpu_memory_used)
            else:
                # Check RAM limit
                if self._cpu_memory_used + block.size_bytes > self.max_ram_gb * (1024**3):
                    # Try cleanup first
                    self._cleanup_cpu()
                    if self._cpu_memory_used + block.size_bytes > self.max_ram_gb * (1024**3):
                        raise RuntimeError(
                            f"Allocation of {size_gb:.2f} GB would exceed RAM limit "
                            f"({self.max_ram_gb} GB). Current usage: {self._cpu_memory_used / (1024**3):.2f} GB"
                        )
                self._cpu_memory_used += block.size_bytes
                self._peak_cpu_memory = max(self._peak_cpu_memory, self._cpu_memory_used)
            
            # Track block
            self._blocks[name] = block
            self._access_count[name] = 0
            self._last_access[name] = time.time()
            
            logger.debug(f"Allocated '{name}': {size_gb:.2f} GB ({block.memory_type.value})")
            
            return block
    
    def release(self, name: str) -> bool:
        """
        Release a memory block by name.
        
        Args:
            name: Name of the block to release
        
        Returns:
            bool: True if released, False if not found
        """
        with self._lock:
            if name not in self._blocks:
                logger.warning(f"Memory block '{name}' not found")
                return False
            
            block = self._blocks[name]
            size_gb = block.size_bytes / (1024**3)
            
            if block.locked:
                logger.warning(f"Cannot release locked block '{name}'")
                return False
            
            # Release memory
            block.release()
            
            # Update tracking
            if block.memory_type == MemoryType.CUDA:
                self._gpu_memory_used -= block.size_bytes
            else:
                self._cpu_memory_used -= block.size_bytes
            
            # Remove from tracking
            del self._blocks[name]
            del self._access_count[name]
            del self._last_access[name]
            
            logger.debug(f"Released '{name}': {size_gb:.2f} GB ({block.memory_type.value})")
            
            return True
    
    def get(self, name: str) -> Optional[torch.Tensor]:
        """
        Get a tensor by name and update access statistics.
        
        Args:
            name: Name of the block
        
        Returns:
            Tensor or None if not found
        """
        with self._lock:
            if name not in self._blocks:
                return None
            
            self._access_count[name] += 1
            self._last_access[name] = time.time()
            
            return self._blocks[name].tensor
    
    def lock(self, name: str) -> bool:
        """
        Lock a memory block to prevent eviction.
        
        Args:
            name: Name of the block to lock
        
        Returns:
            bool: True if locked, False if not found
        """
        with self._lock:
            if name not in self._blocks:
                return False
            self._blocks[name].locked = True
            logger.debug(f"Locked '{name}'")
            return True
    
    def unlock(self, name: str) -> bool:
        """
        Unlock a memory block to allow eviction.
        
        Args:
            name: Name of the block to unlock
        
        Returns:
            bool: True if unlocked, False if not found
        """
        with self._lock:
            if name not in self._blocks:
                return False
            self._blocks[name].locked = False
            logger.debug(f"Unlocked '{name}'")
            return True
    
    def cleanup(
        self,
        target_fraction: Optional[float] = None,
        min_free: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Clean up memory by evicting blocks.
        
        Args:
            target_fraction: Target memory usage fraction (0-1)
            min_free: Minimum free memory to maintain (GB)
        
        Returns:
            Dictionary with cleanup statistics
        """
        with self._cleanup_lock:
            # Calculate targets
            if target_fraction is None:
                target_fraction = self.cleanup_threshold
            
            vram_target = self.max_vram_gb * target_fraction
            ram_target = self.max_ram_gb * target_fraction
            
            if min_free is None:
                min_free = self.min_free_gb
            
            # Cleanup GPU
            gpu_cleaned = self._cleanup_gpu(
                target_gb=vram_target,
                min_free_gb=min_free,
            )
            
            # Cleanup CPU
            cpu_cleaned = self._cleanup_cpu(
                target_gb=ram_target,
                min_free_gb=min_free,
            )
            
            # Notify callbacks
            for callback in self._cleanup_callbacks:
                try:
                    callback(gpu_cleaned, cpu_cleaned)
                except Exception as e:
                    logger.error(f"Error in cleanup callback: {e}")
            
            return {
                "gpu_cleaned_gb": gpu_cleaned,
                "cpu_cleaned_gb": cpu_cleaned,
                "total_cleaned_gb": gpu_cleaned + cpu_cleaned,
            }
    
    def _cleanup_gpu(
        self,
        target_gb: Optional[float] = None,
        min_free_gb: Optional[float] = None,
    ) -> float:
        """Clean up GPU memory."""
        if not torch.cuda.is_available():
            return 0.0
        
        with self._lock:
            current_gb = self._gpu_memory_used / (1024**3)
            target = target_gb or (self.max_vram_gb * self.cleanup_threshold)
            min_free = min_free_gb or self.min_free_gb
            
            # Check if cleanup needed
            if current_gb <= target and (self.max_vram_gb - current_gb) >= min_free:
                return 0.0
            
            # Get evictable blocks (not locked, GPU memory)
            evictable = [
                (name, block) for name, block in self._blocks.items()
                if not block.locked and block.memory_type == MemoryType.CUDA
            ]
            
            if not evictable:
                return 0.0
            
            # Sort by eviction policy
            evictable.sort(key=self._get_eviction_key)
            
            # Evict blocks
            cleaned = 0.0
            for name, block in evictable:
                if current_gb - cleaned / (1024**3) <= target:
                    break
                
                if self.release(name):
                    cleaned += block.size_bytes
            
            # Force garbage collection
            torch.cuda.empty_cache()
            gc.collect()
            
            return cleaned / (1024**3)
    
    def _cleanup_cpu(
        self,
        target_gb: Optional[float] = None,
        min_free_gb: Optional[float] = None,
    ) -> float:
        """Clean up CPU memory."""
        with self._lock:
            current_gb = self._cpu_memory_used / (1024**3)
            target = target_gb or (self.max_ram_gb * self.cleanup_threshold)
            min_free = min_free_gb or self.min_free_gb
            
            # Check if cleanup needed
            if current_gb <= target and (self.max_ram_gb - current_gb) >= min_free:
                return 0.0
            
            # Get evictable blocks (not locked, CPU memory)
            evictable = [
                (name, block) for name, block in self._blocks.items()
                if not block.locked and block.memory_type == MemoryType.CPU
            ]
            
            if not evictable:
                return 0.0
            
            # Sort by eviction policy
            evictable.sort(key=self._get_eviction_key)
            
            # Evict blocks
            cleaned = 0.0
            for name, block in evictable:
                if current_gb - cleaned / (1024**3) <= target:
                    break
                
                if self.release(name):
                    cleaned += block.size_bytes
            
            # Force garbage collection
            gc.collect()
            
            return cleaned / (1024**3)
    
    def _get_eviction_key(self, item: tuple) -> tuple:
        """Get sort key for eviction policy."""
        name, block = item
        
        if self.eviction_policy == EvictionPolicy.LRU:
            # Least Recently Used
            return (self._last_access.get(name, 0),)
        elif self.eviction_policy == EvictionPolicy.LFU:
            # Least Frequently Used
            return (self._access_count.get(name, 0),)
        elif self.eviction_policy == EvictionPolicy.SMALLEST:
            # Smallest first
            return (block.size_bytes,)
        elif self.eviction_policy == EvictionPolicy.LARGEST:
            # Largest first
            return (-block.size_bytes,)
        elif self.eviction_policy == EvictionPolicy.OLDER:
            # Oldest first
            return (block.timestamp,)
        elif self.eviction_policy == EvictionPolicy.PRIORITY:
            # Lowest priority first
            return (block.priority,)
        else:
            return (0,)
    
    def clear(self, force: bool = False) -> Dict[str, Any]:
        """
        Clear all memory blocks.
        
        Args:
            force: If True, clear even locked blocks
        
        Returns:
            Dictionary with cleanup statistics
        """
        with self._lock:
            cleaned_gpu = 0.0
            cleaned_cpu = 0.0
            
            for name, block in list(self._blocks.items()):
                if block.locked and not force:
                    continue
                
                if self.release(name):
                    if block.memory_type == MemoryType.CUDA:
                        cleaned_gpu += block.size_bytes
                    else:
                        cleaned_cpu += block.size_bytes
            
            # Force cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return {
                "gpu_cleaned_gb": cleaned_gpu / (1024**3),
                "cpu_cleaned_gb": cleaned_cpu / (1024**3),
                "total_cleaned_gb": (cleaned_gpu + cleaned_cpu) / (1024**3),
            }
    
    def get_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        with self._lock:
            return MemoryStats(
                total_allocated_gb=(self._cpu_memory_used + self._gpu_memory_used) / (1024**3),
                used_gb=(self._cpu_memory_used + self._gpu_memory_used) / (1024**3),
                free_gb=(self.max_vram_gb + self.max_ram_gb) - \
                       ((self._cpu_memory_used + self._gpu_memory_used) / (1024**3)),
                peak_gb=max(self._peak_cpu_memory, self._peak_gpu_memory) / (1024**3),
                num_blocks=len(self._blocks),
                num_locked=sum(1 for b in self._blocks.values() if b.locked),
            )
    
    def get_system_memory(self) -> Dict[str, float]:
        """Get system memory usage (not just managed memory)."""
        vram_used = 0.0
        vram_total = 0.0
        
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / (1024**3)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        mem = psutil.virtual_memory()
        ram_used = mem.used / (1024**3)
        ram_total = mem.total / (1024**3)
        
        return {
            "vram_used_gb": vram_used,
            "vram_total_gb": vram_total,
            "vram_percent": (vram_used / vram_total * 100) if vram_total > 0 else 0,
            "ram_used_gb": ram_used,
            "ram_total_gb": ram_total,
            "ram_percent": (ram_used / ram_total * 100) if ram_total > 0 else 0,
        }
    
    def register_oom_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback for out-of-memory events."""
        self._oom_callbacks.append(callback)
    
    def register_cleanup_callback(self, callback: Callable[[float, float], None]) -> None:
        """Register a callback for cleanup events."""
        self._cleanup_callbacks.append(callback)
    
    def __contains__(self, name: str) -> bool:
        """Check if a block exists."""
        with self._lock:
            return name in self._blocks
    
    def __len__(self) -> int:
        """Get number of tracked blocks."""
        with self._lock:
            return len(self._blocks)
    
    def list_blocks(self) -> List[Dict[str, Any]]:
        """List all memory blocks."""
        with self._lock:
            return [
                {
                    "name": name,
                    "size_gb": block.size_bytes / (1024**3),
                    "type": block.memory_type.value,
                    "priority": block.priority,
                    "locked": block.locked,
                    "access_count": self._access_count.get(name, 0),
                }
                for name, block in self._blocks.items()
            ]


class CachedTensor:
    """
    Wrapper for tensors with automatic caching and eviction.
    
    Example:
        >>> manager = MemoryManager()
        >>> cached = CachedTensor(manager, "my_tensor", shape=(1000, 1000))
        >>> tensor = cached.get()  # Loads or creates tensor
        >>> cached.release()  # Releases tensor
    """
    
    def __init__(
        self,
        manager: MemoryManager,
        name: str,
        shape: tuple,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        priority: int = 0,
        creator: Optional[Callable[[], torch.Tensor]] = None,
    ):
        """
        Initialize cached tensor.
        
        Args:
            manager: MemoryManager instance
            name: Name for the tensor
            shape: Shape of the tensor
            dtype: Data type
            device: Device ("cpu" or "cuda")
            priority: Priority for eviction
            creator: Optional callable to create the tensor
        """
        self.manager = manager
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.priority = priority
        self.creator = creator
        self._tensor: Optional[torch.Tensor] = None
        self._loaded = False
    
    def get(self) -> torch.Tensor:
        """Get the tensor, loading or creating if necessary."""
        if not self._loaded or self._tensor is None:
            self._load()
        
        # Update access in manager
        self.manager.get(self.name)
        
        return self._tensor
    
    def _load(self) -> None:
        """Load or create the tensor."""
        if self.creator:
            self._tensor = self.creator()
        else:
            self._tensor = torch.empty(self.shape, dtype=self.dtype, device=self.device)
        
        # Allocate in manager
        self.manager.allocate(self._tensor, self.name, priority=self.priority)
        self._loaded = True
    
    def release(self) -> bool:
        """Release the tensor."""
        if self._tensor is not None:
            result = self.manager.release(self.name)
            self._tensor = None
            self._loaded = False
            return result
        return False
    
    def __del__(self):
        """Cleanup on deletion."""
        self.release()


if __name__ == "__main__":
    # Test the memory manager
    from .hardware_detector import HardwareDetector
    
    hardware = HardwareDetector.detect()
    
    # Create memory manager
    if hardware.gpu_available:
        manager = MemoryManager(
            max_vram_gb=hardware.vram_total_gb * 0.8,
            max_ram_gb=hardware.ram_total_gb * 0.6,
        )
    else:
        manager = MemoryManager(
            max_vram_gb=1.0,
            max_ram_gb=hardware.ram_total_gb * 0.6,
        )
    
    print("MemoryManager Test")
    print("=" * 80)
    
    # Allocate some tensors
    for i in range(5):
        size = 1000 * (i + 1)
        tensor = torch.randn(size, size, device="cuda" if hardware.gpu_available else "cpu")
        manager.allocate(tensor, f"tensor_{i}", priority=i)
        print(f"Allocated tensor_{i}: {tensor.numel() * tensor.element_size() / (1024**3):.3f} GB")
    
    # Get stats
    stats = manager.get_stats()
    print(f"\nStats: {stats.to_dict()}")
    
    # List blocks
    print("\nBlocks:")
    for block in manager.list_blocks():
        print(f"  {block['name']}: {block['size_gb']:.3f} GB")
    
    # Cleanup
    result = manager.cleanup()
    print(f"\nCleanup: {result}")
    
    # Clear
    result = manager.clear()
    print(f"Clear: {result}")
