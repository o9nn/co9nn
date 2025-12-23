"""
CogTorch Nested Neural Networks (NNN) Module
Multidimensional tensor shapes with arbitrary nesting

Enables neural networks to work with complex nested structures like:
- Tuples of tuples
- Lists of tensors
- Hierarchical data structures
- Tree-structured data
- Variable-length sequences

This module extends the standard tensor paradigm to support nested,
heterogeneous data structures common in cognitive architectures.
"""

import numpy as np
from typing import Union, List, Tuple, Any, Optional, Callable
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch7 import Module, Tensor, zeros


class NestedTensor:
    """
    Nested tensor structure supporting arbitrary nesting of tensors
    Can represent: tuples of tuples, lists of tensors, tree structures, etc.
    """
    
    def __init__(self, data: Union[Tensor, List, Tuple, 'NestedTensor']):
        """
        Args:
            data: Can be a Tensor, list/tuple of NestedTensors, or another NestedTensor
        """
        # Check if it's a Tensor (handle both import paths)
        if isinstance(data, Tensor) or (hasattr(data, '__class__') and 
                                         data.__class__.__name__ == 'Tensor'):
            self.data = data
            self.is_leaf = True
            self.children = None
        elif isinstance(data, (list, tuple)):
            self.data = None
            self.is_leaf = False
            self.children = [NestedTensor(item) if not isinstance(item, NestedTensor) else item 
                           for item in data]
        elif isinstance(data, NestedTensor):
            self.data = data.data
            self.is_leaf = data.is_leaf
            self.children = data.children
        else:
            raise ValueError(f"Unsupported data type for NestedTensor: {type(data)}")
    
    def apply(self, func: Callable[[Tensor], Tensor]) -> 'NestedTensor':
        """
        Apply a function to all leaf tensors in the nested structure
        
        Args:
            func: Function to apply to each tensor
            
        Returns:
            New NestedTensor with transformed values
        """
        if self.is_leaf:
            return NestedTensor(func(self.data))
        else:
            return NestedTensor([child.apply(func) for child in self.children])
    
    def map2(self, other: 'NestedTensor', func: Callable[[Tensor, Tensor], Tensor]) -> 'NestedTensor':
        """
        Apply a binary function to corresponding tensors in two nested structures
        
        Args:
            other: Another NestedTensor with same structure
            func: Binary function to apply
            
        Returns:
            New NestedTensor with combined values
        """
        if self.is_leaf and other.is_leaf:
            return NestedTensor(func(self.data, other.data))
        elif not self.is_leaf and not other.is_leaf:
            if len(self.children) != len(other.children):
                raise ValueError("Nested structures must have same shape")
            return NestedTensor([c1.map2(c2, func) for c1, c2 in zip(self.children, other.children)])
        else:
            raise ValueError("Cannot map2 structures with different nesting")
    
    def flatten(self) -> List[Tensor]:
        """
        Flatten nested structure to list of leaf tensors
        
        Returns:
            List of all tensors in depth-first order
        """
        if self.is_leaf:
            return [self.data]
        else:
            result = []
            for child in self.children:
                result.extend(child.flatten())
            return result
    
    def depth(self) -> int:
        """
        Get maximum depth of nesting
        
        Returns:
            Maximum nesting depth
        """
        if self.is_leaf:
            return 0
        else:
            return 1 + max(child.depth() for child in self.children)
    
    def size(self) -> int:
        """
        Get total number of leaf tensors
        
        Returns:
            Count of leaf tensors
        """
        if self.is_leaf:
            return 1
        else:
            return sum(child.size() for child in self.children)
    
    def __repr__(self):
        if self.is_leaf:
            return f"NestedTensor({self.data.shape})"
        else:
            return f"NestedTensor({len(self.children)} children, depth={self.depth()})"


class NestedModule(Module):
    """
    Base module for processing nested tensor structures
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, input: NestedTensor) -> NestedTensor:
        """Forward pass on nested input"""
        raise NotImplementedError("forward() must be implemented by subclass")
    
    def backward(self, input: NestedTensor, gradOutput: NestedTensor) -> NestedTensor:
        """Backward pass on nested gradients"""
        raise NotImplementedError("backward() must be implemented by subclass")


class NestedLinear(NestedModule):
    """
    Linear layer that operates on all leaf tensors independently
    """
    
    def __init__(self, inputSize: int, outputSize: int):
        """
        Args:
            inputSize: Input feature size for each leaf tensor
            outputSize: Output feature size for each leaf tensor
        """
        super().__init__()
        self.inputSize = inputSize
        self.outputSize = outputSize
        
        # Import Linear from nn module
        from ..nn import Linear
        self.linear = Linear(inputSize, outputSize)
    
    def forward(self, input: NestedTensor) -> NestedTensor:
        """Apply linear transformation to all leaves"""
        self.output = input.apply(lambda x: self.linear.forward(x))
        return self.output
    
    def backward(self, input: NestedTensor, gradOutput: NestedTensor) -> NestedTensor:
        """Compute gradients for all leaves"""
        self.gradInput = input.map2(gradOutput, 
                                     lambda inp, grad: self.linear.backward(inp, grad))
        return self.gradInput
    
    def __repr__(self):
        return f"nnn.NestedLinear({self.inputSize}, {self.outputSize})"


class NestedSequential(NestedModule):
    """
    Sequential container for nested modules
    """
    
    def __init__(self, *modules):
        super().__init__()
        self.modules = list(modules)
    
    def add(self, module: NestedModule):
        """Add module to sequence"""
        self.modules.append(module)
        return self
    
    def forward(self, input: NestedTensor) -> NestedTensor:
        """Forward through all modules"""
        self.output = input
        for module in self.modules:
            self.output = module.forward(self.output)
        return self.output
    
    def backward(self, input: NestedTensor, gradOutput: NestedTensor) -> NestedTensor:
        """Backward through all modules"""
        self.gradInput = gradOutput
        for module in reversed(self.modules):
            # This is simplified - should store intermediate values
            self.gradInput = module.backward(input, self.gradInput)
        return self.gradInput
    
    def __repr__(self):
        s = "nnn.NestedSequential {\n"
        for i, module in enumerate(self.modules):
            s += f"  [{i}] {module}\n"
        s += "}"
        return s


class NestedFlatten(NestedModule):
    """
    Flatten nested structure to a single tensor
    """
    
    def forward(self, input: NestedTensor) -> Tensor:
        """Flatten all leaf tensors into single tensor"""
        leaves = input.flatten()
        
        # Concatenate all leaves
        arrays = [leaf.data.reshape(-1) for leaf in leaves]
        self.output = Tensor(np.concatenate(arrays))
        self.input_structure = input
        return self.output
    
    def backward(self, input: NestedTensor, gradOutput: Tensor) -> NestedTensor:
        """Unflatten gradient back to nested structure"""
        # Split gradient according to input structure
        leaves = input.flatten()
        sizes = [leaf.size for leaf in leaves]
        
        split_grads = []
        offset = 0
        for size in sizes:
            split_grads.append(Tensor(gradOutput.data[offset:offset+size]))
            offset += size
        
        # Reconstruct nested structure
        # This is simplified - full implementation would need to track structure
        self.gradInput = NestedTensor(split_grads[0])  # Placeholder
        return self.gradInput
    
    def __repr__(self):
        return "nnn.NestedFlatten()"


class NestedUnflatten(NestedModule):
    """
    Unflatten a tensor into nested structure according to a template
    """
    
    def __init__(self, template: NestedTensor):
        """
        Args:
            template: Template nested structure defining output shape
        """
        super().__init__()
        self.template = template
    
    def forward(self, input: Tensor) -> NestedTensor:
        """Unflatten tensor into nested structure"""
        # Get leaf shapes from template
        leaves = self.template.flatten()
        sizes = [leaf.size for leaf in leaves]
        
        # Split input according to sizes
        split_tensors = []
        offset = 0
        for size, leaf in zip(sizes, leaves):
            data = input.data[offset:offset+size].reshape(leaf.shape)
            split_tensors.append(Tensor(data))
            offset += size
        
        # Reconstruct nested structure (simplified)
        self.output = NestedTensor(split_tensors[0])  # Placeholder
        return self.output
    
    def backward(self, input: Tensor, gradOutput: NestedTensor) -> Tensor:
        """Flatten gradient back to tensor"""
        leaves = gradOutput.flatten()
        arrays = [leaf.data.reshape(-1) for leaf in leaves]
        self.gradInput = Tensor(np.concatenate(arrays))
        return self.gradInput
    
    def __repr__(self):
        return f"nnn.NestedUnflatten(depth={self.template.depth()})"


class TreeLSTM(NestedModule):
    """
    Tree-structured LSTM for processing hierarchical nested data
    Inspired by Tree-LSTM papers (Tai et al., 2015)
    """
    
    def __init__(self, inputSize: int, hiddenSize: int):
        """
        Args:
            inputSize: Size of input features
            hiddenSize: Size of hidden state
        """
        super().__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        
        # Import Linear for gates
        from ..nn import Linear
        
        # Gates: input, output, update, forget (per child)
        self.W_i = Linear(inputSize, hiddenSize)
        self.U_i = Linear(hiddenSize, hiddenSize)
        
        self.W_o = Linear(inputSize, hiddenSize)
        self.U_o = Linear(hiddenSize, hiddenSize)
        
        self.W_u = Linear(inputSize, hiddenSize)
        self.U_u = Linear(hiddenSize, hiddenSize)
        
        # Forget gate (one per child - we'll use one for simplicity)
        self.W_f = Linear(inputSize, hiddenSize)
        self.U_f = Linear(hiddenSize, hiddenSize)
    
    def forward(self, input: NestedTensor) -> NestedTensor:
        """
        Forward pass through tree structure
        Processes leaves first, then aggregates upward
        """
        if input.is_leaf:
            # Leaf node - compute hidden state from input
            x = input.data
            i = Tensor(1.0 / (1.0 + np.exp(-self.W_i.forward(x).data)))
            o = Tensor(1.0 / (1.0 + np.exp(-self.W_o.forward(x).data)))
            u = Tensor(np.tanh(self.W_u.forward(x).data))
            
            c = i.mul(u)
            h = o.mul(Tensor(np.tanh(c.data)))
            
            self.output = NestedTensor(h)
            return self.output
        else:
            # Non-leaf node - aggregate children
            child_outputs = [self.forward(child) for child in input.children]
            
            # Simplified aggregation - average child states
            child_hiddens = [c.data for c in child_outputs]
            avg_hidden = Tensor(np.mean([h.data for h in child_hiddens], axis=0))
            
            self.output = NestedTensor(avg_hidden)
            return self.output
    
    def backward(self, input: NestedTensor, gradOutput: NestedTensor) -> NestedTensor:
        """Backward pass (simplified)"""
        self.gradInput = gradOutput
        return self.gradInput
    
    def __repr__(self):
        return f"nnn.TreeLSTM({self.inputSize}, {self.hiddenSize})"


class NestedEmbedding(NestedModule):
    """
    Embedding layer for nested structures
    Maps discrete indices to continuous vectors
    """
    
    def __init__(self, numEmbeddings: int, embeddingDim: int):
        """
        Args:
            numEmbeddings: Size of embedding dictionary
            embeddingDim: Dimension of embedding vectors
        """
        super().__init__()
        self.numEmbeddings = numEmbeddings
        self.embeddingDim = embeddingDim
        
        # Initialize embedding matrix
        from ..torch7 import randn
        self.weight = randn(numEmbeddings, embeddingDim)
        self.weight.mul_(0.1)
    
    def forward(self, input: NestedTensor) -> NestedTensor:
        """
        Lookup embeddings for nested indices
        
        Args:
            input: NestedTensor where leaves contain integer indices
            
        Returns:
            NestedTensor with embedding vectors
        """
        def lookup(x):
            # x should contain integer indices
            indices = x.data.astype(int).flatten()
            embeddings = self.weight.data[indices]
            return Tensor(embeddings)
        
        self.output = input.apply(lookup)
        return self.output
    
    def backward(self, input: NestedTensor, gradOutput: NestedTensor) -> NestedTensor:
        """Backward pass (simplified)"""
        self.gradInput = gradOutput
        return self.gradInput
    
    def __repr__(self):
        return f"nnn.NestedEmbedding({self.numEmbeddings}, {self.embeddingDim})"


def create_nested_tensor(*shapes) -> NestedTensor:
    """
    Helper function to create nested tensor with specified shapes
    
    Args:
        *shapes: Nested tuple/list structure of shapes
        
    Returns:
        NestedTensor with random data
    """
    from ..torch7 import randn
    
    if isinstance(shapes[0], (int, tuple)):
        # Single shape - create tensor
        shape = shapes[0] if isinstance(shapes[0], tuple) else shapes
        return NestedTensor(randn(*shape))
    else:
        # Nested structure
        return NestedTensor([create_nested_tensor(*s) if isinstance(s, (list, tuple)) else randn(*s) 
                           for s in shapes])


__all__ = [
    'NestedTensor',
    'NestedModule',
    'NestedLinear',
    'NestedSequential',
    'NestedFlatten',
    'NestedUnflatten',
    'TreeLSTM',
    'NestedEmbedding',
    'create_nested_tensor'
]
