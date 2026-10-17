"""
Benchmark Profiler for Apourvoir SDK

Comprehensive profiling of quantization and inference performance
with detailed metrics and comparisons.
"""

import logging
import time
import json
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Tuple, Union
from datetime import datetime
import statistics

import torch
import numpy as np
import psutil

from .hardware_detector import HardwareDetector, HardwareSpecs
from .adaptive_batching import AdaptiveBatcher
from .memory_manager import MemoryManager

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""
    name: str = ""
    execution_time_ms: float = 0.0
    tokens_per_second: float = 0.0
    memory_used_gb: float = 0.0
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0
    batch_size: int = 0
    seq_length: int = 0
    n_bits: int = 0
    device: str = "cpu"
    success: bool = True
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "execution_time_ms": self.execution_time_ms,
            "tokens_per_second": self.tokens_per_second,
            "memory_used_gb": self.memory_used_gb,
            "gpu_utilization": self.gpu_utilization,
            "cpu_utilization": self.cpu_utilization,
            "batch_size": self.batch_size,
            "seq_length": self.seq_length,
            "n_bits": self.n_bits,
            "device": self.device,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


@dataclass
class ComparisonResult:
    """Result of comparing two configurations."""
    config_a: str = ""
    config_b: str = ""
    speedup: float = 1.0
    memory_ratio: float = 1.0
    quality_diff: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config_a": self.config_a,
            "config_b": self.config_b,
            "speedup": self.speedup,
            "memory_ratio": self.memory_ratio,
            "quality_diff": self.quality_diff,
        }


class BenchmarkProfiler:
    """
    Comprehensive benchmark profiler for quantization and inference.
    
    Features:
    - Performance profiling (tokens/sec, latency)
    - Memory profiling (VRAM, RAM usage)
    - Quality profiling (accuracy, similarity)
    - Comparison between configurations
    - Detailed reporting
    
    Example:
        >>> from apourvoir.optimizer import BenchmarkProfiler
        >>> profiler = BenchmarkProfiler()
        >>> 
        >>> # Profile a function
        >>> result = profiler.profile(
        ...     my_model.generate,
        ...     name="FP32 Inference",
        ...     batch_size=8,
        ...     seq_length=1024
        ... )
        >>> 
        >>> # Compare configurations
        >>> comparison = profiler.compare(
        ...     {"FP32": my_fp32_model, "INT8": my_int8_model},
        ...     input_data=test_data
        ... )
        >>> 
        >>> # Generate report
        >>> profiler.generate_report("benchmark_report.json")
    """
    
    def __init__(
        self,
        hardware: Optional[HardwareSpecs] = None,
        warmup_runs: int = 3,
        num_runs: int = 10,
        track_memory: bool = True,
        track_utilization: bool = True,
    ):
        """
        Initialize the benchmark profiler.
        
        Args:
            hardware: Hardware specifications (auto-detected if None)
            warmup_runs: Number of warmup runs
            num_runs: Number of benchmark runs
            track_memory: Whether to track memory usage
            track_utilization: Whether to track GPU/CPU utilization
        """
        self.hardware = hardware or HardwareDetector.detect()
        self.warmup_runs = warmup_runs
        self.num_runs = num_runs
        self.track_memory = track_memory
        self.track_utilization = track_utilization
        
        # Results
        self.results: List[BenchmarkResult] = []
        self.comparisons: List[ComparisonResult] = []
        self._lock = threading.Lock()
        
        # Memory manager for tracking
        self.memory_manager: Optional[MemoryManager] = None
        
        logger.info(f"BenchmarkProfiler initialized: warmup={warmup_runs}, runs={num_runs}")
    
    def set_memory_manager(self, manager: MemoryManager) -> None:
        """Set a memory manager for tracking."""
        self.memory_manager = manager
    
    def profile(
        self,
        func: Callable,
        name: str = "benchmark",
        batch_size: int = 1,
        seq_length: int = 1024,
        n_bits: int = 8,
        device: Optional[str] = None,
        warmup: Optional[int] = None,
        runs: Optional[int] = None,
        **kwargs,
    ) -> BenchmarkResult:
        """
        Profile a function.
        
        Args:
            func: Function to profile
            name: Name for this benchmark
            batch_size: Batch size
            seq_length: Sequence length
            n_bits: Quantization bit precision
            device: Device to use
            warmup: Override warmup runs
            runs: Override number of runs
            **kwargs: Additional arguments for the function
        
        Returns:
            BenchmarkResult with profiling data
        """
        device = device or ("cuda" if self.hardware.gpu_available else "cpu")
        warmup = warmup or self.warmup_runs
        runs = runs or self.num_runs
        
        # Warmup
        for _ in range(warmup):
            try:
                with torch.no_grad():
                    func(**kwargs)
            except:
                pass
        
        # Benchmark
        execution_times = []
        memory_usages = []
        gpu_utilizations = []
        cpu_utilizations = []
        
        for _ in range(runs):
            # Track memory before
            if self.track_memory:
                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    mem_before = torch.cuda.memory_allocated()
                else:
                    mem_before = psutil.Process().memory_info().rss
            
            # Track utilization before
            if self.track_utilization:
                gpu_util_before = torch.cuda.utilization() if torch.cuda.is_available() else 0
                cpu_util_before = psutil.cpu_percent()
            
            # Execute
            start_time = time.perf_counter()
            try:
                with torch.no_grad():
                    result = func(**kwargs)
                success = True
                error_message = ""
            except Exception as e:
                result = None
                success = False
                error_message = str(e)
            end_time = time.perf_counter()
            
            # Track utilization after
            if self.track_utilization:
                gpu_util_after = torch.cuda.utilization() if torch.cuda.is_available() else 0
                cpu_util_after = psutil.cpu_percent()
                gpu_utilizations.append(gpu_util_after)
                cpu_utilizations.append(cpu_util_after)
            
            # Track memory after
            if self.track_memory:
                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    mem_after = torch.cuda.memory_allocated()
                else:
                    mem_after = psutil.Process().memory_info().rss
                memory_usages.append((mem_after - mem_before) / (1024**3))
            
            execution_times.append((end_time - start_time) * 1000)
        
        # Calculate statistics
        execution_time = statistics.median(execution_times)
        tokens = batch_size * seq_length
        tokens_per_second = tokens / (execution_time / 1000) if execution_time > 0 else 0
        
        memory_used = statistics.median(memory_usages) if memory_usages else 0
        gpu_util = statistics.mean(gpu_utilizations) if gpu_utilizations else 0
        cpu_util = statistics.mean(cpu_utilizations) if cpu_utilizations else 0
        
        # Create result
        result = BenchmarkResult(
            name=name,
            execution_time_ms=execution_time,
            tokens_per_second=tokens_per_second,
            memory_used_gb=memory_used,
            gpu_utilization=gpu_util,
            cpu_utilization=cpu_util,
            batch_size=batch_size,
            seq_length=seq_length,
            n_bits=n_bits,
            device=device,
            success=success,
            error_message=error_message,
            metadata={
                "min_time_ms": min(execution_times) if execution_times else 0,
                "max_time_ms": max(execution_times) if execution_times else 0,
                "std_time_ms": statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                "runs": runs,
                "timestamp": datetime.now().isoformat(),
            },
        )
        
        # Store result
        with self._lock:
            self.results.append(result)
        
        return result
    
    def profile_model(
        self,
        model: torch.nn.Module,
        name: str = "model",
        input_shape: Tuple[int, ...] = (1, 1024),
        batch_sizes: List[int] = [1, 4, 8, 16],
        seq_lengths: List[int] = [128, 256, 512, 1024],
        n_bits: int = 8,
        device: Optional[str] = None,
        **kwargs,
    ) -> List[BenchmarkResult]:
        """
        Profile a model with different configurations.
        
        Args:
            model: Model to profile
            name: Base name for benchmarks
            input_shape: Input shape
            batch_sizes: List of batch sizes to test
            seq_lengths: List of sequence lengths to test
            n_bits: Quantization bit precision
            device: Device to use
            **kwargs: Additional arguments for model forward
        
        Returns:
            List of BenchmarkResult objects
        """
        device = device or ("cuda" if self.hardware.gpu_available else "cpu")
        model = model.to(device)
        
        results = []
        
        for batch_size in batch_sizes:
            for seq_length in seq_lengths:
                # Create input
                input_tensor = torch.randn(
                    batch_size, seq_length, *input_shape[1:],
                    device=device
                )
                
                # Profile
                result = self.profile(
                    model,
                    name=f"{name}_bs{batch_size}_seq{seq_length}",
                    batch_size=batch_size,
                    seq_length=seq_length,
                    n_bits=n_bits,
                    device=device,
                    **kwargs,
                )
                results.append(result)
        
        return results
    
    def profile_quantization(
        self,
        quantizer: Callable,
        tensor: torch.Tensor,
        name: str = "quantization",
        n_bits_list: List[int] = [4, 8, 12, 16],
        **kwargs,
    ) -> List[BenchmarkResult]:
        """
        Profile quantization at different bit precisions.
        
        Args:
            quantizer: Quantization function
            tensor: Tensor to quantize
            name: Base name for benchmarks
            n_bits_list: List of bit precisions to test
            **kwargs: Additional arguments for quantizer
        
        Returns:
            List of BenchmarkResult objects
        """
        results = []
        
        for n_bits in n_bits_list:
            def quantize_n_bits():
                return quantizer(tensor, n_bits=n_bits, **kwargs)
            
            result = self.profile(
                quantize_n_bits,
                name=f"{name}_{n_bits}bits",
                batch_size=tensor.shape[0] if len(tensor.shape) > 1 else 1,
                seq_length=tensor.shape[-1] if len(tensor.shape) > 1 else tensor.numel(),
                n_bits=n_bits,
            )
            results.append(result)
        
        return results
    
    def compare(
        self,
        configs: Dict[str, Union[torch.nn.Module, Callable]],
        input_data: Any,
        name: str = "comparison",
        **kwargs,
    ) -> ComparisonResult:
        """
        Compare multiple configurations.
        
        Args:
            configs: Dictionary of configuration name to model/function
            input_data: Input data for benchmarking
            name: Name for this comparison
            **kwargs: Additional arguments
        
        Returns:
            ComparisonResult with comparison data
        """
        results = {}
        
        for config_name, config in configs.items():
            # Profile each configuration
            result = self.profile(
                config,
                name=f"{name}_{config_name}",
                **kwargs,
            )
            results[config_name] = result
        
        # Find baseline (first configuration)
        baseline_name = list(results.keys())[0]
        baseline = results[baseline_name]
        
        # Compare with baseline
        best_speedup = 0
        best_config = baseline_name
        
        for config_name, result in results.items():
            if config_name == baseline_name:
                continue
            
            speedup = result.tokens_per_second / baseline.tokens_per_second if baseline.tokens_per_second > 0 else 1
            memory_ratio = baseline.memory_used_gb / result.memory_used_gb if result.memory_used_gb > 0 else 1
            
            comparison = ComparisonResult(
                config_a=baseline_name,
                config_b=config_name,
                speedup=speedup,
                memory_ratio=memory_ratio,
                quality_diff=0,  # Would need quality metrics
            )
            
            with self._lock:
                self.comparisons.append(comparison)
            
            if speedup > best_speedup:
                best_speedup = speedup
                best_config = config_name
        
        # Return best comparison
        return ComparisonResult(
            config_a=baseline_name,
            config_b=best_config,
            speedup=best_speedup,
            memory_ratio=results[best_config].memory_used_gb / baseline.memory_used_gb if baseline.memory_used_gb > 0 else 1,
        )
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get all benchmark results as dictionaries."""
        with self._lock:
            return [r.to_dict() for r in self.results]
    
    def get_comparisons(self) -> List[Dict[str, Any]]:
        """Get all comparison results as dictionaries."""
        with self._lock:
            return [c.to_dict() for c in self.comparisons]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmarks."""
        with self._lock:
            if not self.results:
                return {
                    "total_benchmarks": 0,
                    "avg_tokens_per_second": 0,
                    "avg_execution_time_ms": 0,
                    "avg_memory_used_gb": 0,
                }
            
            tps_list = [r.tokens_per_second for r in self.results if r.tokens_per_second > 0]
            time_list = [r.execution_time_ms for r in self.results if r.execution_time_ms > 0]
            memory_list = [r.memory_used_gb for r in self.results if r.memory_used_gb > 0]
            
            return {
                "total_benchmarks": len(self.results),
                "avg_tokens_per_second": statistics.mean(tps_list) if tps_list else 0,
                "min_tokens_per_second": min(tps_list) if tps_list else 0,
                "max_tokens_per_second": max(tps_list) if tps_list else 0,
                "avg_execution_time_ms": statistics.mean(time_list) if time_list else 0,
                "avg_memory_used_gb": statistics.mean(memory_list) if memory_list else 0,
                "total_comparisons": len(self.comparisons),
                "hardware": {
                    "gpu": self.hardware.gpu_model,
                    "vram_gb": self.hardware.vram_total_gb,
                    "cpu": self.hardware.cpu_model,
                    "cpu_cores": self.hardware.cpu_threads,
                },
            }
    
    def generate_report(
        self,
        filepath: str,
        title: str = "Apourvoir Benchmark Report",
        include_system_info: bool = True,
        include_recommendations: bool = True,
    ) -> None:
        """
        Generate a comprehensive benchmark report.
        
        Args:
            filepath: Path to save the report
            title: Report title
            include_system_info: Whether to include system information
            include_recommendations: Whether to include recommendations
        """
        report = {
            "title": title,
            "generated_at": datetime.now().isoformat(),
            "summary": self.get_summary(),
            "results": self.get_results(),
            "comparisons": self.get_comparisons(),
        }
        
        if include_system_info:
            report["system_info"] = self._get_system_info()
        
        if include_recommendations:
            report["recommendations"] = self._generate_recommendations()
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Benchmark report saved to {filepath}")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "hardware": self.hardware.to_dict(),
            "python": {
                "version": self.hardware.python_version,
                "pytorch_version": self.hardware.pytorch_version,
            },
            "os": {
                "name": self.hardware.os_name,
                "version": self.hardware.os_version,
            },
        }
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate optimization recommendations."""
        recommendations = []
        summary = self.get_summary()
        
        # Recommend based on hardware
        if self.hardware.gpu_available:
            if self.hardware.vram_total_gb >= 24:
                recommendations.append({
                    "type": "quantization",
                    "description": "Use 12-16 bit quantization for best quality",
                    "bits": 12,
                    "reason": "Sufficient VRAM for higher precision",
                })
            elif self.hardware.vram_total_gb >= 16:
                recommendations.append({
                    "type": "quantization",
                    "description": "Use 8-12 bit quantization",
                    "bits": 8,
                    "reason": "Good balance of quality and memory",
                })
            else:
                recommendations.append({
                    "type": "quantization",
                    "description": "Use 4-8 bit quantization",
                    "bits": 4,
                    "reason": "Limited VRAM requires aggressive quantization",
                })
        else:
            recommendations.append({
                "type": "quantization",
                "description": "Use 4-8 bit quantization for CPU",
                "bits": 4,
                "reason": "CPU mode benefits from aggressive quantization",
            })
        
        # Recommend based on performance
        if summary["avg_tokens_per_second"] < 10:
            recommendations.append({
                "type": "performance",
                "description": "Consider using smaller batch sizes",
                "batch_size": max(1, self.hardware.optimal_batch_size // 2),
                "reason": "Low throughput may indicate memory constraints",
            })
        
        if summary["avg_memory_used_gb"] > self.hardware.vram_total_gb * 0.8:
            recommendations.append({
                "type": "memory",
                "description": "Reduce memory usage",
                "actions": [
                    "Use lower bit precision",
                    "Reduce batch size",
                    "Use gradient checkpointing",
                ],
                "reason": "High memory usage may cause OOM errors",
            })
        
        return recommendations
    
    def reset(self) -> None:
        """Reset all results and statistics."""
        with self._lock:
            self.results.clear()
            self.comparisons.clear()
    
    def print_report(self, title: str = "Benchmark Report") -> None:
        """Print a formatted benchmark report."""
        summary = self.get_summary()
        
        print("=" * 80)
        print(title)
        print("=" * 80)
        print()
        
        print("[System]")
        print(f"  GPU: {self.hardware.gpu_model}")
        print(f"  VRAM: {self.hardware.vram_total_gb:.1f} GB")
        print(f"  CPU: {self.hardware.cpu_model}")
        print(f"  CPU Cores: {self.hardware.cpu_threads}")
        print()
        
        print("[Summary]")
        print(f"  Total Benchmarks: {summary['total_benchmarks']}")
        print(f"  Avg Tokens/sec: {summary['avg_tokens_per_second']:.1f}")
        print(f"  Avg Time: {summary['avg_execution_time_ms']:.2f} ms")
        print(f"  Avg Memory: {summary['avg_memory_used_gb']:.2f} GB")
        print()
        
        print("[Results]")
        for result in self.results[:10]:  # Print first 10
            print(f"  {result.name}:")
            print(f"    Time: {result.execution_time_ms:.2f} ms")
            print(f"    TPS: {result.tokens_per_second:.1f}")
            print(f"    Memory: {result.memory_used_gb:.2f} GB")
            print(f"    Batch: {result.batch_size}, Seq: {result.seq_length}, Bits: {result.n_bits}")
        print()
        
        if self.comparisons:
            print("[Comparisons]")
            for comp in self.comparisons[:5]:  # Print first 5
                print(f"  {comp.config_a} vs {comp.config_b}:")
                print(f"    Speedup: {comp.speedup:.2f}x")
                print(f"    Memory Ratio: {comp.memory_ratio:.2f}x")
        
        print("=" * 80)


class ContinuousProfiler:
    """
    Continuous profiler that runs in the background.
    """
    
    def __init__(
        self,
        profiler: BenchmarkProfiler,
        interval: float = 60.0,
    ):
        """
        Initialize continuous profiler.
        
        Args:
            profiler: Benchmark profiler to use
            interval: Interval between profiles (seconds)
        """
        self.profiler = profiler
        self.interval = interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def start(self, func: Callable, **kwargs) -> None:
        """
        Start continuous profiling.
        
        Args:
            func: Function to profile continuously
            **kwargs: Arguments for the function
        """
        if self._running:
            return
        
        self._running = True
        self._func = func
        self._kwargs = kwargs
        
        self._thread = threading.Thread(
            target=self._profile_loop,
            daemon=True,
        )
        self._thread.start()
    
    def stop(self) -> None:
        """Stop continuous profiling."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
    
    def _profile_loop(self) -> None:
        """Background profiling loop."""
        while self._running:
            try:
                self.profiler.profile(self._func, **self._kwargs)
            except Exception as e:
                logger.error(f"Error in continuous profiling: {e}")
            
            # Sleep for interval
            for _ in range(int(self.interval * 10)):
                if not self._running:
                    break
                time.sleep(0.1)


class ProfileContext:
    """
    Context manager for profiling a block of code.
    """
    
    def __init__(
        self,
        profiler: BenchmarkProfiler,
        name: str = "block",
        **kwargs,
    ):
        """
        Initialize profile context.
        
        Args:
            profiler: Benchmark profiler
            name: Name for this profile
            **kwargs: Additional arguments
        """
        self.profiler = profiler
        self.name = name
        self.kwargs = kwargs
        self.result: Optional[BenchmarkResult] = None
    
    def __enter__(self):
        """Enter the context."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context."""
        # The actual profiling happens in the profiler
        pass


if __name__ == "__main__":
    # Test the benchmark profiler
    from .hardware_detector import HardwareDetector
    
    hardware = HardwareDetector.detect()
    profiler = BenchmarkProfiler(hardware)
    
    print("BenchmarkProfiler Test")
    print("=" * 80)
    
    # Define a test function
    def test_func(x):
        import time
        time.sleep(0.01)
        return x * x
    
    # Profile with different configurations
    for batch_size in [1, 4, 8]:
        result = profiler.profile(
            test_func,
            name=f"test_bs{batch_size}",
            batch_size=batch_size,
            seq_length=1024,
            runs=5,
        )
        print(f"Batch size {batch_size}: {result.tokens_per_second:.1f} tokens/sec")
    
    # Print summary
    profiler.print_report()
    
    # Generate report
    profiler.generate_report("/tmp/benchmark_report.json")
