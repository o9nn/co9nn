"""
CogTorch Torch7 Module
Core tensor operations and module system inspired by torch7

Provides fundamental tensor operations and the base Module class
that all neural network components inherit from.
"""

import numpy as np
from typing import Optional, Union, Tuple, List, Any


class Tensor:
    """
    Multi-dimensional array container inspired by torch.Tensor
    Wraps numpy arrays with torch-style API
    """
    
    def __init__(self, data: Optional[Union[np.ndarray, list, tuple, float]] = None, 
                 shape: Optional[Tuple[int, ...]] = None):
        """
        Initialize a tensor
        
        Args:
            data: Initial data (numpy array, list, tuple, or scalar)
            shape: Shape of tensor if creating empty tensor
        """
        if data is not None:
            if isinstance(data, np.ndarray):
                self.data = data
            elif isinstance(data, (list, tuple)):
                self.data = np.array(data, dtype=np.float32)
            elif isinstance(data, (int, float)):
                self.data = np.array([data], dtype=np.float32)
            elif isinstance(data, Tensor):
                self.data = data.data.copy()
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
        elif shape is not None:
            self.data = np.zeros(shape, dtype=np.float32)
        else:
            self.data = np.array([], dtype=np.float32)
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get tensor shape"""
        return self.data.shape
    
    @property
    def size(self) -> int:
        """Get total number of elements"""
        return self.data.size
    
    @property
    def ndim(self) -> int:
        """Get number of dimensions"""
        return self.data.ndim
    
    def dim(self) -> int:
        """Alias for ndim"""
        return self.ndim
    
    def zero_(self):
        """Zero out the tensor in-place"""
        self.data.fill(0)
        return self
    
    def fill_(self, value: float):
        """Fill tensor with a scalar value in-place"""
        self.data.fill(value)
        return self
    
    def copy_(self, other: 'Tensor'):
        """Copy data from another tensor in-place"""
        self.data = other.data.copy()
        return self
    
    def clone(self) -> 'Tensor':
        """Create a copy of this tensor"""
        return Tensor(self.data.copy())
    
    def resize_(self, *shape):
        """Resize tensor in-place"""
        self.data = np.resize(self.data, shape)
        return self
    
    def view(self, *shape) -> 'Tensor':
        """Return a new tensor with same data but different shape"""
        return Tensor(self.data.reshape(shape))
    
    def add(self, other: Union['Tensor', float]) -> 'Tensor':
        """Add another tensor or scalar"""
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        else:
            return Tensor(self.data + other)
    
    def add_(self, other: Union['Tensor', float]):
        """Add in-place"""
        if isinstance(other, Tensor):
            self.data += other.data
        else:
            self.data += other
        return self
    
    def mul(self, other: Union['Tensor', float]) -> 'Tensor':
        """Multiply by another tensor or scalar"""
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        else:
            return Tensor(self.data * other)
    
    def mul_(self, other: Union['Tensor', float]):
        """Multiply in-place"""
        if isinstance(other, Tensor):
            self.data *= other.data
        else:
            self.data *= other
        return self
    
    def dot(self, other: 'Tensor') -> 'Tensor':
        """Dot product with another tensor"""
        return Tensor(np.dot(self.data, other.data))
    
    def mm(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication"""
        return Tensor(np.matmul(self.data, other.data))
    
    def norm(self, p: int = 2) -> float:
        """Compute p-norm of tensor"""
        return float(np.linalg.norm(self.data, ord=p))
    
    def mean(self, dim: Optional[int] = None) -> Union['Tensor', float]:
        """Compute mean along dimension"""
        if dim is None:
            return float(np.mean(self.data))
        return Tensor(np.mean(self.data, axis=dim))
    
    def sum(self, dim: Optional[int] = None) -> Union['Tensor', float]:
        """Sum along dimension"""
        if dim is None:
            return float(np.sum(self.data))
        return Tensor(np.sum(self.data, axis=dim))
    
    def max(self, dim: Optional[int] = None) -> Union['Tensor', float]:
        """Maximum value along dimension"""
        if dim is None:
            return float(np.max(self.data))
        return Tensor(np.max(self.data, axis=dim))
    
    def __add__(self, other):
        return self.add(other)
    
    def __mul__(self, other):
        return self.mul(other)
    
    def __repr__(self):
        return f"Tensor({self.data})"
    
    def __str__(self):
        return str(self.data)


class Module:
    """
    Base class for all neural network modules
    Inspired by torch.nn.Module
    """
    
    def __init__(self):
        self.training = True
        self._parameters = {}
        self._modules = {}
        self.output = None
        self.gradInput = None
    
    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass computation
        Must be implemented by subclasses
        """
        raise NotImplementedError("forward() must be implemented by subclass")
    
    def backward(self, input: Tensor, gradOutput: Tensor) -> Tensor:
        """
        Backward pass computation
        Must be implemented by subclasses
        """
        raise NotImplementedError("backward() must be implemented by subclass")
    
    def updateOutput(self, input: Tensor) -> Tensor:
        """Compute and store output (alias for forward)"""
        self.output = self.forward(input)
        return self.output
    
    def updateGradInput(self, input: Tensor, gradOutput: Tensor) -> Tensor:
        """Compute and store gradient with respect to input"""
        self.gradInput = self.backward(input, gradOutput)
        return self.gradInput
    
    def accGradParameters(self, input: Tensor, gradOutput: Tensor, scale: float = 1.0):
        """
        Accumulate gradients for parameters
        Default implementation does nothing (for modules without parameters)
        """
        pass
    
    def zeroGradParameters(self):
        """Zero out parameter gradients"""
        pass
    
    def updateParameters(self, learningRate: float):
        """Update parameters using gradients"""
        pass
    
    def parameters(self) -> List[Tensor]:
        """Return list of parameters"""
        return list(self._parameters.values())
    
    def train(self, mode: bool = True):
        """Set training mode"""
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode"""
        return self.train(False)
    
    def type(self, dtype: str = 'float'):
        """Set tensor type"""
        # Placeholder for type conversion
        return self
    
    def __call__(self, input: Tensor) -> Tensor:
        """Allow module to be called like a function"""
        return self.forward(input)
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"


# Utility functions
def randn(*shape) -> Tensor:
    """Create tensor with random normal distribution"""
    return Tensor(np.random.randn(*shape).astype(np.float32))


def zeros(*shape) -> Tensor:
    """Create tensor filled with zeros"""
    return Tensor(np.zeros(shape, dtype=np.float32))


def ones(*shape) -> Tensor:
    """Create tensor filled with ones"""
    return Tensor(np.ones(shape, dtype=np.float32))


def uniform(min_val: float, max_val: float, *shape) -> Tensor:
    """Create tensor with uniform random distribution"""
    return Tensor(np.random.uniform(min_val, max_val, shape).astype(np.float32))


def eye(n: int) -> Tensor:
    """Create identity matrix"""
    return Tensor(np.eye(n, dtype=np.float32))


def isTensor(obj: Any) -> bool:
    """Check if object is a Tensor"""
    return isinstance(obj, Tensor)


__all__ = [
    'Tensor',
    'Module',
    'randn',
    'zeros',
    'ones',
    'uniform',
    'eye',
    'isTensor'
]
