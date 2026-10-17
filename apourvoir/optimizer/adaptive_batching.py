"""
Adaptive Batching for Apourvoir SDK

Dynamically adjusts batch sizes based on available memory (VRAM/RAM)
and hardware capabilities to maximize throughput.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Iterator
import threading

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .hardware_detector import HardwareSpecs

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for a batch."""
    batch_size: int = 1
    seq_length: int = 1024
    dtype: torch.dtype = torch.float32
    device: str = "cpu"
    use_pinned_memory: bool = False
    max_memory_gb: float = 8.0
    num_workers: int = 0
    prefetch_factor: int = 2
    persistent_workers: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "batch_size": self.batch_size,
            "seq_length": self.seq_length,
            "dtype": str(self.dtype),
            "device": self.device,
            "use_pinned_memory": self.use_pinned_memory,
            "max_memory_gb": self.max_memory_gb,
            "num_workers": self.num_workers,
            "prefetch_factor": self.prefetch_factor,
            "persistent_workers": self.persistent_workers,
        }


@dataclass
class BatchStats:
    """Statistics for batch processing."""
    batch_size: int = 0
    processing_time_ms: float = 0.0
    tokens_per_second: float = 0.0
    memory_used_gb: float = 0.0
    success: bool = True
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "batch_size": self.batch_size,
            "processing_time_ms": self.processing_time_ms,
            "tokens_per_second": self.tokens_per_second,
            "memory_used_gb": self.memory_used_gb,
            "success": self.success,
            "error_message": self.error_message,
        }


class AdaptiveBatcher:
    """
    Adaptive batching manager that dynamically adjusts batch sizes
    based on available memory and hardware capabilities.
    
    This class provides:
    - Dynamic batch size adjustment
    - Memory-aware batching
    - Automatic configuration based on hardware
    - Statistics tracking
    - Thread-safe operations
    
    Example:
        >>> from apourvoir.optimizer import HardwareDetector, AdaptiveBatcher
        >>> hardware = HardwareDetector.detect()
        >>> batcher = AdaptiveBatcher(hardware)
        >>> 
        >>> # Create a dataloader
        >>> dataloader = batcher.create_dataloader(dataset, batch_size=8)
        >>> 
        >>> # Process batches
        >>> for batch in dataloader:
        ...     process(batch)
        >>> 
        >>> # Adjust based on memory usage
        >>> new_batch_size = batcher.adjust_batch_size(memory_used_gb=10.0)
    """
    
    def __init__(
        self,
        hardware: HardwareSpecs,
        base_batch_size: Optional[int] = None,
        min_batch_size: int = 1,
        max_batch_size: int = 64,
        seq_length: int = 1024,
        memory_buffer_gb: float = 1.0,
    ):
        """
        Initialize the adaptive batcher.
        
        Args:
            hardware: Hardware specifications from HardwareDetector
            base_batch_size: Initial batch size (defaults to hardware optimal)
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
            seq_length: Default sequence length
            memory_buffer_gb: Memory buffer in GB for safety
        """
        self.hardware = hardware
        self.seq_length = seq_length
        self.memory_buffer_gb = memory_buffer_gb
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        
        # Current configuration
        self.current_batch_size = base_batch_size or hardware.optimal_batch_size
        self.current_batch_size = max(min_batch_size, min(max_batch_size, self.current_batch_size))
        
        # Statistics
        self.stats: List[BatchStats] = []
        self._lock = threading.Lock()
        
        # Memory tracking
        self._peak_memory_gb = 0.0
        self._current_memory_gb = 0.0
        
        # Performance tracking
        self._last_processing_times: List[float] = []
        self._max_processing_time = 0.0
        
        logger.info(f"AdaptiveBatcher initialized: batch_size={self.current_batch_size}, "
                   f"min={self.min_batch_size}, max={self.max_batch_size}")
    
    @property
    def config(self) -> BatchConfig:
        """Get current batch configuration."""
        return BatchConfig(
            batch_size=self.current_batch_size,
            seq_length=self.seq_length,
            dtype=torch.float16 if self.hardware.gpu_available else torch.float32,
            device="cuda" if self.hardware.gpu_available else "cpu",
            use_pinned_memory=self.hardware.use_pinned_memory,
            max_memory_gb=self._get_max_memory_gb(),
            num_workers=min(4, self.hardware.cpu_threads),
            prefetch_factor=2,
            persistent_workers=True if self.hardware.cpu_threads >= 4 else False,
        )
    
    def _get_max_memory_gb(self) -> float:
        """Get maximum allowed memory in GB."""
        if self.hardware.gpu_available:
            return (self.hardware.vram_total_gb * 0.7) - self.memory_buffer_gb
        else:
            return (self.hardware.ram_total_gb * 0.6) - self.memory_buffer_gb
    
    def _get_available_memory_gb(self) -> float:
        """Get currently available memory in GB."""
        if self.hardware.gpu_available and torch.cuda.is_available():
            total = self.hardware.vram_total_gb
            used = torch.cuda.memory_allocated() / (1024**3)
            return total - used - self.memory_buffer_gb
        else:
            import psutil
            mem = psutil.virtual_memory()
            return mem.available / (1024**3) - self.memory_buffer_gb
    
    def adjust_batch_size(
        self,
        memory_used_gb: Optional[float] = None,
        processing_time_ms: Optional[float] = None,
        success: bool = True,
        error_message: str = "",
    ) -> int:
        """
        Dynamically adjust batch size based on memory usage and performance.
        
        Args:
            memory_used_gb: Current memory usage in GB
            processing_time_ms: Time to process last batch in ms
            success: Whether the last batch was successful
            error_message: Error message if last batch failed
        
        Returns:
            New optimal batch size
        """
        with self._lock:
            # Record statistics
            if memory_used_gb is not None:
                self._current_memory_gb = memory_used_gb
                self._peak_memory_gb = max(self._peak_memory_gb, memory_used_gb)
            
            if processing_time_ms is not None:
                self._last_processing_times.append(processing_time_ms)
                if len(self._last_processing_times) > 10:
                    self._last_processing_times.pop(0)
                self._max_processing_time = max(self._last_processing_times)
            
            # Record batch stats
            if memory_used_gb is not None or processing_time_ms is not None:
                stat = BatchStats(
                    batch_size=self.current_batch_size,
                    processing_time_ms=processing_time_ms or 0.0,
                    memory_used_gb=memory_used_gb or 0.0,
                    success=success,
                    error_message=error_message,
                )
                self.stats.append(stat)
                if len(self.stats) > 100:
                    self.stats.pop(0)
            
            # Calculate new batch size
            new_batch_size = self._calculate_optimal_batch_size(
                memory_used_gb, processing_time_ms, success
            )
            
            # Clamp to min/max
            new_batch_size = max(self.min_batch_size, min(self.max_batch_size, new_batch_size))
            
            if new_batch_size != self.current_batch_size:
                logger.info(f"Adjusting batch size: {self.current_batch_size} -> {new_batch_size} "
                           f"(memory: {memory_used_gb:.1f}GB, time: {processing_time_ms:.1f}ms)")
                self.current_batch_size = new_batch_size
            
            return self.current_batch_size
    
    def _calculate_optimal_batch_size(
        self,
        memory_used_gb: Optional[float],
        processing_time_ms: Optional[float],
        success: bool,
    ) -> int:
        """Calculate optimal batch size based on current conditions."""
        # If batch failed due to OOM, reduce significantly
        if not success and "OOM" in (error_message or ""):
            return max(self.min_batch_size, self.current_batch_size // 2)
        
        # If batch failed for other reasons, reduce moderately
        if not success:
            return max(self.min_batch_size, self.current_batch_size - 1)
        
        # Get available memory
        available_memory = self._get_available_memory_gb()
        
        # Estimate memory per token (GB)
        # For Mixtral-8x7B: ~0.5GB per token in FP32, ~0.03GB in INT8
        if self.hardware.gpu_available:
            memory_per_token = 0.03  # Conservative estimate for quantized
        else:
            memory_per_token = 0.5  # More conservative for CPU
        
        # Calculate max batch size based on memory
        max_by_memory = max(1, int(available_memory // memory_per_token))
        
        # Calculate based on processing time
        if processing_time_ms and len(self._last_processing_times) >= 3:
            avg_time = np.mean(self._last_processing_times)
            # If processing is fast, we can increase batch size
            if avg_time < 50:  # Less than 50ms per batch
                time_factor = 1.2
            elif avg_time < 100:
                time_factor = 1.1
            elif avg_time > 200:
                time_factor = 0.9
            else:
                time_factor = 1.0
        else:
            time_factor = 1.0
        
        # Combine factors
        optimal = int(self.current_batch_size * time_factor)
        optimal = min(optimal, max_by_memory)
        
        # Smooth changes (don't change too dramatically)
        if abs(optimal - self.current_batch_size) > self.current_batch_size // 2:
            optimal = self.current_batch_size + (
                (optimal - self.current_batch_size) // 2
            )
        
        return optimal
    
    def create_dataloader(
        self,
        dataset: Optional[Dataset] = None,
        tensors: Optional[List[torch.Tensor]] = None,
        batch_size: Optional[int] = None,
        seq_length: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = True,
        **kwargs,
    ) -> DataLoader:
        """
        Create a DataLoader with adaptive batching configuration.
        
        Args:
            dataset: PyTorch Dataset
            tensors: List of tensors to create a TensorDataset
            batch_size: Override batch size (uses current if None)
            seq_length: Override sequence length
            shuffle: Whether to shuffle
            drop_last: Whether to drop last incomplete batch
            **kwargs: Additional DataLoader arguments
        
        Returns:
            Configured DataLoader
        """
        # Use current or specified batch size
        bs = batch_size or self.current_batch_size
        sl = seq_length or self.seq_length
        
        # Create dataset if tensors provided
        if dataset is None and tensors is not None:
            dataset = TensorDataset(*tensors)
        elif dataset is None:
            raise ValueError("Either dataset or tensors must be provided")
        
        # Get configuration
        config = self.config
        if batch_size:
            config = BatchConfig(
                batch_size=bs,
                seq_length=sl or config.seq_length,
                dtype=config.dtype,
                device=config.device,
                use_pinned_memory=config.use_pinned_memory,
                max_memory_gb=config.max_memory_gb,
                num_workers=config.num_workers,
                prefetch_factor=config.prefetch_factor,
                persistent_workers=config.persistent_workers,
            )
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=config.num_workers,
            pin_memory=config.use_pinned_memory,
            prefetch_factor=config.prefetch_factor,
            persistent_workers=config.persistent_workers,
            **kwargs,
        )
        
        logger.debug(f"Created DataLoader: batch_size={bs}, num_workers={config.num_workers}, "
                    f"pin_memory={config.use_pinned_memory}")
        
        return dataloader
    
    def process_batch(
        self,
        model: torch.nn.Module,
        batch: Dict[str, torch.Tensor],
        device: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Process a single batch with memory tracking.
        
        Args:
            model: Model to use for processing
            batch: Input batch
            device: Device to use (defaults to config)
            **kwargs: Additional arguments for model
        
        Returns:
            Dictionary with outputs and statistics
        """
        device = device or self.config.device
        
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # Track memory before
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated() / (1024**3)
        else:
            import psutil
            mem_before = psutil.Process().memory_info().rss / (1024**3)
        
        # Process
        start_time = time.perf_counter()
        try:
            with torch.no_grad():
                outputs = model(**batch, **kwargs)
            success = True
            error_message = ""
        except Exception as e:
            outputs = None
            success = False
            error_message = str(e)
        
        # Track memory after
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_after = torch.cuda.memory_allocated() / (1024**3)
        else:
            import psutil
            mem_after = psutil.Process().memory_info().rss / (1024**3)
        
        end_time = time.perf_counter()
        
        # Calculate stats
        processing_time_ms = (end_time - start_time) * 1000
        memory_used_gb = mem_after - mem_before
        
        # Get batch size
        batch_size = len(next(iter(batch.values()))) if batch else 0
        
        # Calculate tokens per second
        tokens = batch_size * self.seq_length
        tokens_per_second = tokens / (processing_time_ms / 1000) if processing_time_ms > 0 else 0
        
        # Record stats
        self.adjust_batch_size(
            memory_used_gb=memory_used_gb,
            processing_time_ms=processing_time_ms,
            success=success,
            error_message=error_message,
        )
        
        # Return results
        return {
            "outputs": outputs,
            "success": success,
            "error_message": error_message,
            "batch_size": batch_size,
            "processing_time_ms": processing_time_ms,
            "tokens_per_second": tokens_per_second,
            "memory_used_gb": memory_used_gb,
        }
    
    def get_stats(self) -> List[Dict[str, Any]]:
        """Get batch statistics as list of dictionaries."""
        with self._lock:
            return [s.to_dict() for s in self.stats]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        with self._lock:
            if not self.stats:
                return {
                    "total_batches": 0,
                    "avg_batch_size": 0,
                    "avg_processing_time_ms": 0,
                    "avg_tokens_per_second": 0,
                    "avg_memory_used_gb": 0,
                    "success_rate": 0,
                    "current_batch_size": self.current_batch_size,
                }
            
            batch_sizes = [s.batch_size for s in self.stats]
            processing_times = [s.processing_time_ms for s in self.stats if s.processing_time_ms > 0]
            tokens_per_sec = [s.tokens_per_second for s in self.stats if s.tokens_per_second > 0]
            memory_used = [s.memory_used_gb for s in self.stats if s.memory_used_gb > 0]
            successes = [s.success for s in self.stats]
            
            return {
                "total_batches": len(self.stats),
                "avg_batch_size": np.mean(batch_sizes) if batch_sizes else 0,
                "avg_processing_time_ms": np.mean(processing_times) if processing_times else 0,
                "avg_tokens_per_second": np.mean(tokens_per_sec) if tokens_per_sec else 0,
                "avg_memory_used_gb": np.mean(memory_used) if memory_used else 0,
                "peak_memory_gb": self._peak_memory_gb,
                "success_rate": np.mean(successes) * 100 if successes else 0,
                "current_batch_size": self.current_batch_size,
                "min_batch_size": self.min_batch_size,
                "max_batch_size": self.max_batch_size,
            }
    
    def reset_stats(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self.stats.clear()
            self._peak_memory_gb = 0.0
            self._current_memory_gb = 0.0
            self._last_processing_times.clear()
            self._max_processing_time = 0.0
    
    def set_batch_size(self, batch_size: int) -> None:
        """Manually set batch size."""
        with self._lock:
            self.current_batch_size = max(
                self.min_batch_size,
                min(self.max_batch_size, batch_size)
            )
        logger.info(f"Batch size manually set to: {self.current_batch_size}")
    
    def get_optimal_batch_size(self) -> int:
        """Get current optimal batch size."""
        return self.current_batch_size


class BatchSizeFinder:
    """
    Helper class to find optimal batch size through binary search.
    
    Useful for initial calibration of the adaptive batcher.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: Dataset,
        hardware: HardwareSpecs,
        max_batch_size: int = 64,
        target_memory_gb: Optional[float] = None,
        target_latency_ms: Optional[float] = None,
    ):
        """
        Initialize batch size finder.
        
        Args:
            model: Model to test
            dataset: Dataset to use
            hardware: Hardware specifications
            max_batch_size: Maximum batch size to test
            target_memory_gb: Target memory usage in GB
            target_latency_ms: Target latency in ms
        """
        self.model = model
        self.dataset = dataset
        self.hardware = hardware
        self.max_batch_size = max_batch_size
        self.target_memory_gb = target_memory_gb
        self.target_latency_ms = target_latency_ms
        
        self.batcher = AdaptiveBatcher(hardware)
    
    def find_optimal(
        self,
        num_tests: int = 10,
        warmup: int = 3,
    ) -> Dict[str, Any]:
        """
        Find optimal batch size through testing.
        
        Args:
            num_tests: Number of tests to run
            warmup: Number of warmup runs
        
        Returns:
            Dictionary with optimal batch size and metrics
        """
        results = []
        
        # Test different batch sizes
        batch_sizes = list(range(1, self.max_batch_size + 1, max(1, self.max_batch_size // num_tests)))
        if batch_sizes[-1] != self.max_batch_size:
            batch_sizes.append(self.max_batch_size)
        
        for bs in batch_sizes:
            self.batcher.set_batch_size(bs)
            
            # Warmup
            for _ in range(warmup):
                try:
                    dataloader = self.batcher.create_dataloader(
                        self.dataset, batch_size=bs
                    )
                    for batch in dataloader:
                        with torch.no_grad():
                            _ = self.model(**batch)
                        break
                except:
                    break
            
            # Test
            try:
                dataloader = self.batcher.create_dataloader(
                    self.dataset, batch_size=bs
                )
                
                total_time = 0
                total_tokens = 0
                peak_memory = 0
                
                for batch in dataloader:
                    result = self.batcher.process_batch(self.model, batch)
                    total_time += result["processing_time_ms"]
                    total_tokens += result["batch_size"] * self.batcher.seq_length
                    peak_memory = max(peak_memory, result["memory_used_gb"])
                    break  # Just test one batch
                
                avg_tps = total_tokens / (total_time / 1000) if total_time > 0 else 0
                
                results.append({
                    "batch_size": bs,
                    "avg_time_ms": total_time,
                    "tokens_per_second": avg_tps,
                    "memory_used_gb": peak_memory,
                    "success": True,
                })
            except Exception as e:
                results.append({
                    "batch_size": bs,
                    "avg_time_ms": 0,
                    "tokens_per_second": 0,
                    "memory_used_gb": 0,
                    "success": False,
                    "error": str(e),
                })
        
        # Find optimal
        successful = [r for r in results if r["success"]]
        if not successful:
            return {
                "optimal_batch_size": 1,
                "results": results,
                "reason": "No successful tests",
            }
        
        # Filter by constraints
        if self.target_memory_gb:
            successful = [r for r in successful if r["memory_used_gb"] <= self.target_memory_gb]
        
        if self.target_latency_ms:
            successful = [r for r in successful if r["avg_time_ms"] <= self.target_latency_ms]
        
        if not successful:
            successful = [r for r in results if r["success"]]
        
        # Find best by throughput
        best = max(successful, key=lambda x: x["tokens_per_second"])
        
        return {
            "optimal_batch_size": best["batch_size"],
            "optimal_tokens_per_second": best["tokens_per_second"],
            "optimal_memory_gb": best["memory_used_gb"],
            "results": results,
            "reason": "Max throughput",
        }


if __name__ == "__main__":
    # Test the adaptive batcher
    from .hardware_detector import HardwareDetector
    
    hardware = HardwareDetector.detect()
    batcher = AdaptiveBatcher(hardware)
    
    print("AdaptiveBatcher Test")
    print("=" * 80)
    print(f"Initial batch size: {batcher.current_batch_size}")
    print(f"Config: {batcher.config}")
    print()
    
    # Simulate some batches
    for i in range(5):
        # Simulate memory usage
        memory_used = np.random.uniform(5, 15)
        processing_time = np.random.uniform(10, 100)
        batcher.adjust_batch_size(
            memory_used_gb=memory_used,
            processing_time_ms=processing_time,
            success=True,
        )
    
    print(f"Final batch size: {batcher.current_batch_size}")
    print(f"Summary: {batcher.get_summary()}")
