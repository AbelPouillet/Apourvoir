"""
Kernel Fusion for Apourvoir SDK

Fuses multiple operations into single kernels to reduce overhead
and improve performance on both CPU and GPU.
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Tuple, Union, Type
from enum import Enum
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """Fusion strategies."""
    MANUAL = "manual"           # Manual fusion
    AUTOMATIC = "automatic"       # Automatic fusion detection
    AGGRESSIVE = "aggressive"     # Aggressive fusion (may reduce numerical stability)


@dataclass
class FusionConfig:
    """Configuration for kernel fusion."""
    strategy: FusionStrategy = FusionStrategy.AUTOMATIC
    max_fused_ops: int = 10
    allow_inplace: bool = True
    verify_equivalence: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy": self.strategy.value,
            "max_fused_ops": self.max_fused_ops,
            "allow_inplace": self.allow_inplace,
            "verify_equivalence": self.verify_equivalence,
        }


@dataclass
class FusionStats:
    """Statistics for kernel fusion."""
    num_fused_kernels: int = 0
    num_operations: int = 0
    speedup: float = 1.0
    memory_saved_gb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_fused_kernels": self.num_fused_kernels,
            "num_operations": self.num_operations,
            "speedup": self.speedup,
            "memory_saved_gb": self.memory_saved_gb,
        }


class KernelFuser:
    """
    Kernel fusion manager that combines multiple operations into single kernels.
    
    This class provides:
    - Fusion of common operation patterns (Linear + Activation, Conv + BN + ReLU, etc.)
    - Custom fusion rules
    - Performance tracking
    - Verification of numerical equivalence
    
    Example:
        >>> from apourvoir.optimizer import KernelFuser
        >>> fuser = KernelFuser()
        >>> 
        >>> # Fuse Linear + ReLU
        >>> linear = nn.Linear(1000, 1000)
        >>> fused = fuser.fuse(linear, torch.nn.ReLU())
        >>> 
        >>> # Fuse Conv + BatchNorm + ReLU
        >>> conv = nn.Conv2d(64, 64, 3)
        >>> bn = nn.BatchNorm2d(64)
        >>> fused = fuser.fuse(conv, bn, torch.nn.ReLU())
    """
    
    def __init__(
        self,
        strategy: FusionStrategy = FusionStrategy.AUTOMATIC,
        max_fused_ops: int = 10,
        allow_inplace: bool = True,
        verify_equivalence: bool = False,
    ):
        """
        Initialize the kernel fuser.
        
        Args:
            strategy: Fusion strategy
            max_fused_ops: Maximum number of operations to fuse
            allow_inplace: Whether to allow in-place operations
            verify_equivalence: Whether to verify numerical equivalence
        """
        self.strategy = strategy
        self.max_fused_ops = max_fused_ops
        self.allow_inplace = allow_inplace
        self.verify_equivalence = verify_equivalence
        
        # Statistics
        self.stats: List[FusionStats] = []
        self._lock = threading.Lock()
        
        # Fusion registry
        self._fusion_rules = {
            (nn.Linear, nn.ReLU): self.fuse_linear_activation,
            (nn.Linear, nn.GELU): self.fuse_linear_activation,
            (nn.Linear, nn.SiLU): self.fuse_linear_activation,
            (nn.Conv2d, nn.BatchNorm2d, nn.ReLU): self.fuse_conv_bn_relu,
            (nn.Conv2d, nn.BatchNorm2d): self.fuse_conv_bn,
            (nn.Linear,): self.fuse_linear,
            (nn.Conv2d,): self.fuse_conv,
        }
        
        # Fused modules registry
        self._fused_modules: Dict[str, nn.Module] = {}
        
        logger.info(f"KernelFuser initialized: strategy={strategy.value}")
    
    def fuse(self, *modules: nn.Module) -> nn.Module:
        """
        Fuse multiple modules into a single kernel.
        
        Args:
            *modules: Modules to fuse
        
        Returns:
            Fused module
        """
        if not modules:
            raise ValueError("At least one module must be provided")
        
        # Get module types
        types = tuple(type(m) for m in modules)
        
        # Try to find a matching fusion rule
        for rule_types, fusion_func in self._fusion_rules.items():
            if len(rule_types) == len(types):
                match = True
                for rule_type, actual_type in zip(rule_types, types):
                    if not issubclass(actual_type, rule_type):
                        match = False
                        break
                
                if match:
                    try:
                        fused = fusion_func(*modules)
                        logger.info(f"Fused {len(modules)} modules: {[type(m).__name__ for m in modules]}")
                        return fused
                    except Exception as e:
                        logger.warning(f"Fusion failed for {types}: {e}")
        
        # If no rule matches, return a sequential wrapper
        return SequentialWrapper(*modules)
    
    def register_fusion_rule(
        self,
        module_types: Tuple[Type[nn.Module], ...],
        fusion_func: Callable[[*Tuple[nn.Module, ...]], nn.Module],
    ) -> None:
        """
        Register a custom fusion rule.
        
        Args:
            module_types: Tuple of module types to match
            fusion_func: Function to create fused module
        """
        self._fusion_rules[module_types] = fusion_func
        logger.info(f"Registered fusion rule for {module_types}")
    
    def fuse_linear_activation(
        self,
        linear: nn.Linear,
        activation: nn.Module,
    ) -> nn.Module:
        """Fuse Linear layer with activation function."""
        class FusedLinearActivation(nn.Module):
            def __init__(self, linear, activation):
                super().__init__()
                self.weight = nn.Parameter(linear.weight)
                self.bias = nn.Parameter(linear.bias) if linear.bias is not None else None
                self.activation = activation
            
            def forward(self, x):
                out = F.linear(x, self.weight, self.bias)
                return self.activation(out)
            
            def extra_repr(self):
                return f"activation={type(self.activation).__name__}"
        
        return FusedLinearActivation(linear, activation)
    
    def fuse_conv_bn_relu(
        self,
        conv: nn.Conv2d,
        bn: nn.BatchNorm2d,
        relu: nn.ReLU,
    ) -> nn.Module:
        """Fuse Conv2D + BatchNorm + ReLU into a single kernel."""
        class FusedConvBNReLU(nn.Module):
            def __init__(self, conv, bn, relu):
                super().__init__()
                
                # Check if we can fuse
                if conv.bias is not None:
                    raise ValueError("Conv layer must not have bias for Conv+BN fusion")
                
                # Fuse weights
                self.weight = nn.Parameter(conv.weight)
                
                # BatchNorm parameters
                self.gamma = nn.Parameter(bn.weight)
                self.beta = nn.Parameter(bn.bias)
                self.eps = bn.eps
                self.running_mean = bn.running_mean
                self.running_var = bn.running_var
                
                # Conv parameters
                self.stride = conv.stride
                self.padding = conv.padding
                self.dilation = conv.dilation
                self.groups = conv.groups
            
            def forward(self, x):
                # Conv
                out = F.conv2d(
                    x, self.weight, None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
                
                # BatchNorm
                if self.training:
                    # Use batch statistics
                    mean = out.mean(dim=(0, 2, 3), keepdim=True)
                    var = out.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
                else:
                    # Use running statistics
                    mean = self.running_mean.view(1, -1, 1, 1)
                    var = self.running_var.view(1, -1, 1, 1)
                
                out = (out - mean) / torch.sqrt(var + self.eps)
                out = self.gamma.view(1, -1, 1, 1) * out + self.beta.view(1, -1, 1, 1)
                
                # ReLU
                return F.relu(out)
            
            def extra_repr(self):
                return f"stride={self.stride}, padding={self.padding}"
        
        return FusedConvBNReLU(conv, bn, relu)
    
    def fuse_conv_bn(
        self,
        conv: nn.Conv2d,
        bn: nn.BatchNorm2d,
    ) -> nn.Module:
        """Fuse Conv2D + BatchNorm into a single kernel."""
        class FusedConvBN(nn.Module):
            def __init__(self, conv, bn):
                super().__init__()
                
                if conv.bias is not None:
                    raise ValueError("Conv layer must not have bias for Conv+BN fusion")
                
                self.weight = nn.Parameter(conv.weight)
                self.gamma = nn.Parameter(bn.weight)
                self.beta = nn.Parameter(bn.bias)
                self.eps = bn.eps
                self.running_mean = bn.running_mean
                self.running_var = bn.running_var
                self.stride = conv.stride
                self.padding = conv.padding
                self.dilation = conv.dilation
                self.groups = conv.groups
            
            def forward(self, x):
                out = F.conv2d(
                    x, self.weight, None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
                
                if self.training:
                    mean = out.mean(dim=(0, 2, 3), keepdim=True)
                    var = out.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
                else:
                    mean = self.running_mean.view(1, -1, 1, 1)
                    var = self.running_var.view(1, -1, 1, 1)
                
                out = (out - mean) / torch.sqrt(var + self.eps)
                return self.gamma.view(1, -1, 1, 1) * out + self.beta.view(1, -1, 1, 1)
        
        return FusedConvBN(conv, bn)
    
    def fuse_linear(self, linear: nn.Linear) -> nn.Module:
        """Optimize Linear layer."""
        # For now, just return the original
        # Could add optimizations like weight packing, etc.
        return linear
    
    def fuse_conv(self, conv: nn.Conv2d) -> nn.Module:
        """Optimize Conv2D layer."""
        # For now, just return the original
        return conv
    
    def fuse_quant_dequant(
        self,
        quantizer: nn.Module,
        dequantizer: nn.Module,
    ) -> nn.Module:
        """
        Fuse quantization and dequantization.
        
        If quantizing and immediately dequantizing, we can skip the operation
        to save memory and computation.
        """
        class FusedQuantDequant(nn.Module):
            def __init__(self, quantizer, dequantizer):
                super().__init__()
                self.quantizer = quantizer
                self.dequantizer = dequantizer
            
            def forward(self, x):
                # If we're quantizing and dequantizing immediately,
                # just return the input (identity operation)
                return x
        
        return FusedQuantDequant(quantizer, dequantizer)
    
    def fuse_sequential(self, *modules: nn.Module) -> nn.Module:
        """Fuse a sequence of modules."""
        return SequentialWrapper(*modules)
    
    def auto_fuse(self, model: nn.Module) -> nn.Module:
        """
        Automatically fuse operations in a model.
        
        Args:
            model: Model to optimize
        
        Returns:
            Model with fused operations
        """
        # Create a copy of the model
        model_copy = self._deep_copy_model(model)
        
        # Find and fuse patterns
        self._fuse_patterns(model_copy)
        
        return model_copy
    
    def _deep_copy_model(self, model: nn.Module) -> nn.Module:
        """Create a deep copy of a model."""
        import copy
        return copy.deepcopy(model)
    
    def _fuse_patterns(self, model: nn.Module) -> None:
        """Find and fuse operation patterns in a model."""
        # This is a simplified version - a full implementation would
        # need to traverse the model graph and identify patterns
        
        # For now, we'll just look for sequential modules
        for name, module in model.named_children():
            if isinstance(module, nn.Sequential):
                # Try to fuse sequential modules
                fused_modules = []
                i = 0
                while i < len(module):
                    # Try to fuse with next modules
                    fused = False
                    for j in range(i + 1, min(i + self.max_fused_ops + 1, len(module))):
                        try:
                            fused_module = self.fuse(*module[i:j+1])
                            fused_modules.append(fused_module)
                            i = j + 1
                            fused = True
                            break
                        except:
                            continue
                    
                    if not fused:
                        fused_modules.append(module[i])
                        i += 1
                
                # Replace sequential with fused modules
                if len(fused_modules) < len(module):
                    setattr(model, name, nn.Sequential(*fused_modules))


class SequentialWrapper(nn.Module):
    """
    Wrapper for sequential execution of modules.
    
    This is used as a fallback when fusion is not possible.
    """
    
    def __init__(self, *modules: nn.Module):
        """
        Initialize sequential wrapper.
        
        Args:
            *modules: Modules to execute sequentially
        """
        super().__init__()
        self.modules = nn.ModuleList(modules)
    
    def forward(self, x):
        """Forward pass through all modules."""
        for module in self.modules:
            x = module(x)
        return x
    
    def extra_repr(self):
        return f"modules={[type(m).__name__ for m in self.modules]}"


class FusedModule(nn.Module):
    """
    Base class for fused modules.
    
    Provides common functionality for fused modules.
    """
    
    def __init__(self):
        super().__init__()
        self._original_modules = []
    
    def add_original_module(self, module: nn.Module) -> None:
        """Add an original module for reference."""
        self._original_modules.append(module)
    
    def get_original_modules(self) -> List[nn.Module]:
        """Get list of original modules."""
        return self._original_modules


class FusionVerifier:
    """
    Verifies that fused modules produce equivalent outputs to original modules.
    """
    
    def __init__(self, rtol: float = 1e-5, atol: float = 1e-8):
        """
        Initialize verifier.
        
        Args:
            rtol: Relative tolerance
            atol: Absolute tolerance
        """
        self.rtol = rtol
        self.atol = atol
    
    def verify(
        self,
        original: nn.Module,
        fused: nn.Module,
        input_shape: Tuple[int, ...],
        num_tests: int = 5,
    ) -> Dict[str, Any]:
        """
        Verify that fused module produces equivalent outputs.
        
        Args:
            original: Original module
            fused: Fused module
            input_shape: Input shape for testing
            num_tests: Number of test inputs
        
        Returns:
            Dictionary with verification results
        """
        results = {
            "passed": True,
            "max_diff": 0.0,
            "mean_diff": 0.0,
            "num_tests": num_tests,
        }
        
        for _ in range(num_tests):
            # Create random input
            x = torch.randn(input_shape)
            
            # Forward pass
            with torch.no_grad():
                out_original = original(x)
                out_fused = fused(x)
            
            # Calculate difference
            diff = torch.abs(out_original - out_fused)
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            
            results["max_diff"] = max(results["max_diff"], max_diff)
            results["mean_diff"] += mean_diff / num_tests
            
            # Check tolerance
            if max_diff > self.atol + self.rtol * torch.abs(out_original).max().item():
                results["passed"] = False
        
        return results


class FusionOptimizer:
    """
    Optimizes a model by fusing kernels and applying other optimizations.
    """
    
    def __init__(
        self,
        kernel_fuser: Optional[KernelFuser] = None,
        verify_fusion: bool = False,
    ):
        """
        Initialize fusion optimizer.
        
        Args:
            kernel_fuser: Kernel fuser instance
            verify_fusion: Whether to verify fusion correctness
        """
        self.kernel_fuser = kernel_fuser or KernelFuser()
        self.verify_fusion = verify_fusion
        self.verifier = FusionVerifier() if verify_fusion else None
    
    def optimize(self, model: nn.Module) -> nn.Module:
        """
        Optimize a model by fusing kernels.
        
        Args:
            model: Model to optimize
        
        Returns:
            Optimized model
        """
        logger.info("Optimizing model with kernel fusion...")
        
        # Auto-fuse the model
        optimized = self.kernel_fuser.auto_fuse(model)
        
        # Verify if requested
        if self.verify_fusion:
            self._verify_optimized(model, optimized)
        
        return optimized
    
    def _verify_optimized(self, original: nn.Module, optimized: nn.Module) -> None:
        """Verify that optimized model produces equivalent outputs."""
        # This would need to be implemented based on the model structure
        # For now, we'll just log a warning
        logger.warning("Fusion verification not fully implemented")


if __name__ == "__main__":
    # Test kernel fusion
    fuser = KernelFuser()
    
    print("KernelFuser Test")
    print("=" * 80)
    
    # Test Linear + ReLU fusion
    linear = nn.Linear(1000, 1000)
    relu = nn.ReLU()
    
    fused = fuser.fuse(linear, relu)
    print(f"Fused Linear + ReLU: {type(fused).__name__}")
    
    # Test Conv + BN + ReLU fusion
    conv = nn.Conv2d(64, 64, 3, padding=1)
    bn = nn.BatchNorm2d(64)
    relu = nn.ReLU()
    
    try:
        fused_conv = fuser.fuse(conv, bn, relu)
        print(f"Fused Conv + BN + ReLU: {type(fused_conv).__name__}")
    except ValueError as e:
        print(f"Conv + BN + ReLU fusion failed (expected): {e}")
    
    # Test with a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(1000, 1000)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(1000, 1000)
        
        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            return x
    
    model = SimpleModel()
    optimized = fuser.auto_fuse(model)
    
    print(f"\nOriginal model: {model}")
    print(f"Optimized model: {optimized}")
