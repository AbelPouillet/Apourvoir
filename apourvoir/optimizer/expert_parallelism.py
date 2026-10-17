"""
Expert Parallelism for Apourvoir SDK

Implements parallel execution of experts in Mixture of Experts (MoE) models
to maximize throughput on multi-GPU and multi-core systems.
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import Stream

from .hardware_detector import HardwareSpecs
from .memory_manager import MemoryManager
from .thread_pool import NUMAAwareThreadPool

logger = logging.getLogger(__name__)


class ParallelStrategy(Enum):
    """Parallel execution strategies."""
    THREADS = "threads"         # Multi-threading (shared memory)
    PROCESSES = "processes"     # Multi-processing (separate memory)
    CUDA_STREAMS = "cuda_streams" # CUDA streams (GPU parallelism)
    MULTI_GPU = "multi_gpu"     # Multi-GPU parallelism
    HYBRID = "hybrid"           # Hybrid approach
    AUTO = "auto"               # Automatic selection


@dataclass
class ExpertConfig:
    """Configuration for an expert."""
    name: str
    module: nn.Module
    device: str = "cpu"
    weight: float = 1.0
    capacity: int = 1024
    min_batch_size: int = 1
    max_batch_size: int = 1024


@dataclass
class ParallelConfig:
    """Configuration for parallel execution."""
    strategy: ParallelStrategy = ParallelStrategy.CUDA_STREAMS
    num_workers: int = 4
    num_streams: int = 4
    max_parallel: int = 4
    use_numa: bool = False
    pin_memory: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy": self.strategy.value,
            "num_workers": self.num_workers,
            "num_streams": self.num_streams,
            "max_parallel": self.max_parallel,
            "use_numa": self.use_numa,
            "pin_memory": self.pin_memory,
        }


@dataclass
class ParallelStats:
    """Statistics for parallel execution."""
    num_experts: int = 0
    num_parallel: int = 0
    total_tokens: int = 0
    total_time_ms: float = 0.0
    tokens_per_second: float = 0.0
    speedup: float = 1.0
    efficiency: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_experts": self.num_experts,
            "num_parallel": self.num_parallel,
            "total_tokens": self.total_tokens,
            "total_time_ms": self.total_time_ms,
            "tokens_per_second": self.tokens_per_second,
            "speedup": self.speedup,
            "efficiency": self.efficiency,
        }


class ParallelExperts(nn.Module):
    """
    Parallel execution manager for MoE experts.
    
    This class provides:
    - Parallel execution of experts across multiple devices
    - Load balancing
    - Memory-aware scheduling
    - Multiple parallel strategies (threads, processes, CUDA streams, multi-GPU)
    
    Example:
        >>> from apourvoir.optimizer import HardwareDetector, ParallelExperts
        >>> hardware = HardwareDetector.detect()
        >>> 
        >>> # Create experts
        >>> experts = [nn.Linear(1000, 1000) for _ in range(8)]
        >>> 
        >>> # Wrap with parallel execution
        >>> parallel_experts = ParallelExperts(experts, hardware)
        >>> 
        >>> # Execute in parallel
        >>> inputs = [torch.randn(100, 1000) for _ in range(8)]
        >>> outputs = parallel_experts(inputs)
    """
    
    def __init__(
        self,
        experts: List[nn.Module],
        hardware: HardwareSpecs,
        strategy: ParallelStrategy = None,
        max_parallel: Optional[int] = None,
        memory_manager: Optional[MemoryManager] = None,
    ):
        """
        Initialize parallel experts.
        
        Args:
            experts: List of expert modules
            hardware: Hardware specifications
            strategy: Parallel execution strategy
            max_parallel: Maximum number of parallel experts
            memory_manager: Memory manager for tracking
        """
        super().__init__()
        
        self.experts = nn.ModuleList(experts)
        self.hardware = hardware
        self.memory_manager = memory_manager
        
        # Determine optimal strategy
        if strategy is None:
            self.strategy = self._determine_strategy()
        else:
            self.strategy = strategy
        
        # Configuration
        self.max_parallel = max_parallel or min(
            len(experts),
            hardware.max_experts_parallel,
            hardware.cuda_cores // 100 if hardware.gpu_available else hardware.cpu_threads,
        )
        
        # Setup based on strategy
        self._setup_parallel()
        
        # Statistics
        self.stats: List[ParallelStats] = []
        self._lock = threading.Lock()
        
        logger.info(f"ParallelExperts initialized: {len(experts)} experts, "
                   f"strategy={self.strategy.value}, max_parallel={self.max_parallel}")
    
    def _determine_strategy(self) -> ParallelStrategy:
        """Determine optimal parallel strategy based on hardware."""
        if self.hardware.gpu_available and self.hardware.gpu_count > 1:
            return ParallelStrategy.MULTI_GPU
        elif self.hardware.gpu_available:
            return ParallelStrategy.CUDA_STREAMS
        elif self.hardware.cpu_threads >= 8:
            if self.hardware.numa_aware:
                return ParallelStrategy.PROCESSES
            else:
                return ParallelStrategy.THREADS
        else:
            return ParallelStrategy.THREADS
    
    def _setup_parallel(self) -> None:
        """Setup parallel execution resources."""
        if self.strategy == ParallelStrategy.CUDA_STREAMS:
            self._streams = [Stream() for _ in range(self.max_parallel)]
        elif self.strategy == ParallelStrategy.THREADS:
            self._thread_pool = ThreadPoolExecutor(max_workers=self.max_parallel)
        elif self.strategy == ParallelStrategy.PROCESSES:
            self._process_pool = ProcessPoolExecutor(max_workers=self.max_parallel)
        elif self.strategy == ParallelStrategy.MULTI_GPU:
            self._gpu_devices = list(range(min(self.hardware.gpu_count, self.max_parallel)))
            # Distribute experts across GPUs
            self._expert_devices = {}
            for i, expert in enumerate(self.experts):
                gpu_idx = i % len(self._gpu_devices)
                self._expert_devices[i] = self._gpu_devices[gpu_idx]
                expert.to(f"cuda:{self._gpu_devices[gpu_idx]}")
        elif self.strategy == ParallelStrategy.HYBRID:
            self._streams = [Stream() for _ in range(self.max_parallel // 2)]
            self._thread_pool = ThreadPoolExecutor(max_workers=self.max_parallel // 2)
    
    def forward(
        self,
        inputs: List[torch.Tensor],
        gating_scores: Optional[torch.Tensor] = None,
        top_k: int = 2,
    ) -> List[torch.Tensor]:
        """
        Execute experts in parallel.
        
        Args:
            inputs: List of input tensors for each expert
            gating_scores: Optional gating scores for expert selection
            top_k: Number of top experts to use per token
        
        Returns:
            List of output tensors from each expert
        """
        # Validate inputs
        if len(inputs) != len(self.experts):
            raise ValueError(f"Number of inputs ({len(inputs)}) must match number of experts ({len(self.experts)})")
        
        # Select strategy
        if self.strategy == ParallelStrategy.CUDA_STREAMS:
            return self._forward_cuda_streams(inputs)
        elif self.strategy == ParallelStrategy.THREADS:
            return self._forward_threads(inputs)
        elif self.strategy == ParallelStrategy.PROCESSES:
            return self._forward_processes(inputs)
        elif self.strategy == ParallelStrategy.MULTI_GPU:
            return self._forward_multi_gpu(inputs)
        elif self.strategy == ParallelStrategy.HYBRID:
            return self._forward_hybrid(inputs)
        else:
            # Sequential fallback
            return self._forward_sequential(inputs)
    
    def _forward_sequential(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Sequential execution (fallback)."""
        outputs = []
        for expert, inp in zip(self.experts, inputs):
            out = expert(inp)
            outputs.append(out)
        return outputs
    
    def _forward_cuda_streams(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Parallel execution using CUDA streams."""
        if not torch.cuda.is_available():
            return self._forward_sequential(inputs)
        
        outputs = [None] * len(inputs)
        
        # Create events for synchronization
        events = [torch.cuda.Event(enable_timing=True) for _ in range(len(inputs))]
        
        for i, (expert, inp) in enumerate(zip(self.experts, inputs)):
            stream = self._streams[i % len(self._streams)]
            
            with torch.cuda.stream(stream):
                outputs[i] = expert(inp)
                events[i].record(stream=stream)
        
        # Synchronize all streams
        for event in events:
            event.synchronize()
        
        return outputs
    
    def _forward_threads(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Parallel execution using threads."""
        outputs = [None] * len(inputs)
        futures = []
        
        def process_expert(i, expert, inp):
            try:
                with torch.no_grad():
                    outputs[i] = expert(inp)
            except Exception as e:
                logger.error(f"Error in expert {i}: {e}")
                outputs[i] = torch.zeros_like(inp)
        
        for i, (expert, inp) in enumerate(zip(self.experts, inputs)):
            future = self._thread_pool.submit(process_expert, i, expert, inp)
            futures.append(future)
        
        # Wait for all to complete
        for future in futures:
            future.result()
        
        return outputs
    
    def _forward_processes(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Parallel execution using processes."""
        # Note: This requires experts to be picklable
        # For simplicity, we'll use threads as fallback
        logger.warning("Process-based parallelism not fully implemented, falling back to threads")
        return self._forward_threads(inputs)
    
    def _forward_multi_gpu(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Parallel execution across multiple GPUs."""
        if not self.hardware.gpu_available or self.hardware.gpu_count < 2:
            return self._forward_sequential(inputs)
        
        outputs = [None] * len(inputs)
        
        # Move inputs to appropriate GPUs
        for i, inp in enumerate(inputs):
            device = self._expert_devices.get(i, 0)
            inputs[i] = inp.to(f"cuda:{device}")
        
        # Execute on each GPU
        for i, (expert, inp) in enumerate(zip(self.experts, inputs)):
            device = self._expert_devices.get(i, 0)
            with torch.cuda.device(device):
                outputs[i] = expert(inp)
        
        return outputs
    
    def _forward_hybrid(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Hybrid parallel execution (CUDA streams + threads)."""
        half = len(inputs) // 2
        
        # First half with CUDA streams
        outputs_first = self._forward_cuda_streams(inputs[:half])
        
        # Second half with threads
        outputs_second = self._forward_threads(inputs[half:])
        
        return outputs_first + outputs_second
    
    def select_experts(
        self,
        inputs: torch.Tensor,
        gating_scores: torch.Tensor,
        top_k: int = 2,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        Select top-k experts for each input based on gating scores.
        
        Args:
            inputs: Input tensor of shape (batch_size, ...)
            gating_scores: Gating scores of shape (batch_size, num_experts)
            top_k: Number of experts to select per input
        
        Returns:
            Tuple of (list of inputs for selected experts, list of expert indices)
        """
        # Get top-k experts for each input
        top_k_scores, top_k_indices = torch.topk(gating_scores, k=top_k, dim=1)
        
        # Normalize scores
        top_k_scores = F.softmax(top_k_scores, dim=1)
        
        # Create list of (expert_idx, input, weight) tuples
        expert_inputs = []
        expert_indices = []
        
        for batch_idx in range(inputs.shape[0]):
            for rank in range(top_k):
                expert_idx = top_k_indices[batch_idx, rank].item()
                weight = top_k_scores[batch_idx, rank].item()
                
                expert_inputs.append((expert_idx, inputs[batch_idx], weight))
                expert_indices.append(expert_idx)
        
        # Group by expert
        from collections import defaultdict
        grouped_inputs = defaultdict(list)
        for expert_idx, inp, weight in expert_inputs:
            grouped_inputs[expert_idx].append((inp, weight))
        
        # Prepare inputs for each expert
        final_inputs = []
        for i in range(len(self.experts)):
            if i in grouped_inputs:
                # Stack inputs for this expert
                expert_input_tensors = [item[0] for item in grouped_inputs[i]]
                try:
                    stacked = torch.stack(expert_input_tensors)
                except:
                    # If tensors have different shapes, pad them
                    stacked = self._pad_and_stack(expert_input_tensors)
                final_inputs.append(stacked)
            else:
                # Empty input for unused experts
                final_inputs.append(torch.empty(0, *inputs.shape[1:], 
                                                  device=inputs.device, 
                                                  dtype=inputs.dtype))
        
        return final_inputs, list(grouped_inputs.keys())
    
    def _pad_and_stack(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """Pad tensors to same size and stack."""
        # Find max size
        max_size = max(t.shape[0] for t in tensors)
        
        # Pad each tensor
        padded = []
        for t in tensors:
            if t.shape[0] < max_size:
                pad_size = max_size - t.shape[0]
                padding = (0,) * (t.dim() * 2)
                padding = padding[:-2] + (0, pad_size)
                padded_t = F.pad(t, padding, "constant", 0)
            else:
                padded_t = t
            padded.append(padded_t)
        
        return torch.stack(padded)
    
    def get_load_balancing(self) -> Dict[int, int]:
        """
        Get current load balancing across experts.
        
        Returns:
            Dictionary mapping expert index to number of assigned inputs
        """
        # This would need to track actual usage
        # For now, return equal distribution
        return {i: 0 for i in range(len(self.experts))}
    
    def get_stats(self) -> List[Dict[str, Any]]:
        """Get parallel execution statistics."""
        with self._lock:
            return [s.to_dict() for s in self.stats]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        with self._lock:
            if not self.stats:
                return {
                    "total_executions": 0,
                    "avg_tokens_per_second": 0,
                    "avg_speedup": 1.0,
                    "avg_efficiency": 1.0,
                }
            
            tps_list = [s.tokens_per_second for s in self.stats if s.tokens_per_second > 0]
            speedup_list = [s.speedup for s in self.stats if s.speedup > 0]
            efficiency_list = [s.efficiency for s in self.stats if s.efficiency > 0]
            
            return {
                "total_executions": len(self.stats),
                "avg_tokens_per_second": np.mean(tps_list) if tps_list else 0,
                "avg_speedup": np.mean(speedup_list) if speedup_list else 1.0,
                "avg_efficiency": np.mean(efficiency_list) if efficiency_list else 1.0,
                "num_experts": len(self.experts),
                "max_parallel": self.max_parallel,
                "strategy": self.strategy.value,
            }
    
    def reset_stats(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self.stats.clear()
    
    def get_config(self) -> ParallelConfig:
        """Get current parallel configuration."""
        return ParallelConfig(
            strategy=self.strategy,
            num_workers=self.max_parallel,
            num_streams=self.max_parallel if self.strategy == ParallelStrategy.CUDA_STREAMS else 0,
            max_parallel=self.max_parallel,
            use_numa=self.hardware.use_numa,
            pin_memory=self.hardware.use_pinned_memory,
        )


class MoEWrapper(nn.Module):
    """
    Wrapper for MoE models with parallel expert execution.
    
    This wraps a standard MoE model to use ParallelExperts for
    optimized execution.
    """
    
    def __init__(
        self,
        moe_model: nn.Module,
        hardware: HardwareSpecs,
        strategy: Optional[ParallelStrategy] = None,
        **kwargs,
    ):
        """
        Initialize MoE wrapper.
        
        Args:
            moe_model: Original MoE model
            hardware: Hardware specifications
            strategy: Parallel execution strategy
        """
        super().__init__()
        self.moe_model = moe_model
        self.hardware = hardware
        
        # Extract experts from model
        experts = self._extract_experts(moe_model)
        
        # Wrap with parallel execution
        self.parallel_experts = ParallelExperts(
            experts, hardware, strategy=strategy, **kwargs
        )
        
        # Replace original experts with parallel version
        self._replace_experts(moe_model, self.parallel_experts)
    
    def _extract_experts(self, model: nn.Module) -> List[nn.Module]:
        """Extract expert modules from MoE model."""
        experts = []
        
        # Look for common MoE patterns
        if hasattr(model, 'experts'):
            experts.extend(model.experts)
        
        # Look for expert layers
        for name, module in model.named_modules():
            if 'expert' in name.lower() and isinstance(module, nn.ModuleList):
                experts.extend(module)
        
        return experts if experts else [model]
    
    def _replace_experts(self, model: nn.Module, parallel_experts: ParallelExperts) -> None:
        """Replace experts in model with parallel version."""
        if hasattr(model, 'experts'):
            model.experts = parallel_experts
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass with parallel expert execution."""
        return self.moe_model(*args, **kwargs)


class ExpertRouter:
    """
    Dynamic expert router that selects experts based on input and capacity.
    """
    
    def __init__(
        self,
        num_experts: int,
        capacity_per_expert: int = 1024,
        routing_strategy: str = "gating",
    ):
        """
        Initialize expert router.
        
        Args:
            num_experts: Number of experts
            capacity_per_expert: Maximum capacity per expert
            routing_strategy: Routing strategy ("gating", "random", "round_robin", "load_balanced")
        """
        self.num_experts = num_experts
        self.capacity_per_expert = capacity_per_expert
        self.routing_strategy = routing_strategy
        self._counter = 0
        self._load_counts = [0] * num_experts
    
    def route(
        self,
        inputs: torch.Tensor,
        gating_scores: Optional[torch.Tensor] = None,
    ) -> List[Tuple[int, torch.Tensor]]:
        """
        Route inputs to experts.
        
        Args:
            inputs: Input tensor
            gating_scores: Optional gating scores
        
        Returns:
            List of (expert_idx, input) tuples
        """
        if self.routing_strategy == "gating" and gating_scores is not None:
            return self._route_gating(inputs, gating_scores)
        elif self.routing_strategy == "random":
            return self._route_random(inputs)
        elif self.routing_strategy == "round_robin":
            return self._route_round_robin(inputs)
        elif self.routing_strategy == "load_balanced":
            return self._route_load_balanced(inputs)
        else:
            return self._route_gating(inputs, gating_scores)
    
    def _route_gating(
        self,
        inputs: torch.Tensor,
        gating_scores: torch.Tensor,
    ) -> List[Tuple[int, torch.Tensor]]:
        """Route using gating scores."""
        # Get expert assignments
        expert_indices = torch.argmax(gating_scores, dim=1)
        
        # Group by expert
        routes = []
        for i in range(self.num_experts):
            mask = expert_indices == i
            if mask.any():
                routes.append((i, inputs[mask]))
        
        return routes
    
    def _route_random(self, inputs: torch.Tensor) -> List[Tuple[int, torch.Tensor]]:
        """Route randomly."""
        indices = torch.randint(0, self.num_experts, (inputs.shape[0],))
        
        routes = []
        for i in range(self.num_experts):
            mask = indices == i
            if mask.any():
                routes.append((i, inputs[mask]))
        
        return routes
    
    def _route_round_robin(self, inputs: torch.Tensor) -> List[Tuple[int, torch.Tensor]]:
        """Route using round-robin."""
        routes = []
        batch_size = inputs.shape[0]
        
        for i in range(self.num_experts):
            start = (self._counter % self.num_experts) * (batch_size // self.num_experts)
            end = start + (batch_size // self.num_experts)
            routes.append((i, inputs[start:end]))
            self._counter += 1
        
        return routes
    
    def _route_load_balanced(self, inputs: torch.Tensor) -> List[Tuple[int, torch.Tensor]]:
        """Route using load balancing."""
        # Find least loaded experts
        sorted_experts = sorted(range(self.num_experts), key=lambda x: self._load_counts[x])
        
        routes = []
        batch_size = inputs.shape[0]
        batch_per_expert = batch_size // self.num_experts
        
        for i, expert_idx in enumerate(sorted_experts):
            start = i * batch_per_expert
            end = start + batch_per_expert
            routes.append((expert_idx, inputs[start:end]))
            self._load_counts[expert_idx] += batch_per_expert
        
        return routes


if __name__ == "__main__":
    # Test parallel experts
    from .hardware_detector import HardwareDetector
    
    hardware = HardwareDetector.detect()
    
    # Create dummy experts
    experts = [
        nn.Linear(1000, 1000) for _ in range(8)
    ]
    
    # Wrap with parallel execution
    parallel_experts = ParallelExperts(experts, hardware)
    
    print("ParallelExperts Test")
    print("=" * 80)
    print(f"Strategy: {parallel_experts.strategy.value}")
    print(f"Max parallel: {parallel_experts.max_parallel}")
    print()
    
    # Create test inputs
    inputs = [torch.randn(100, 1000) for _ in range(8)]
    
    # Execute
    outputs = parallel_experts(inputs)
    
    print(f"Executed {len(experts)} experts in parallel")
    for i, out in enumerate(outputs):
        print(f"  Expert {i}: output shape = {out.shape}")
    
    # Get summary
    print(f"\nSummary: {parallel_experts.get_summary()}")
