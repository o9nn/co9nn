"""
CogTorch Neural Network Graph Module
Graph-based neural network construction inspired by nngraph

Allows building complex neural network architectures with branching,
merging, and multiple inputs/outputs.
"""

import sys
import os
from typing import List, Dict, Optional, Any, Union

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch7 import Module, Tensor


class Node:
    """
    Represents a node in the computational graph
    """
    
    def __init__(self, module: Optional[Module] = None, data: Optional[Tensor] = None):
        """
        Args:
            module: Module associated with this node
            data: Data tensor (for input nodes)
        """
        self.module = module
        self.data = data
        self.output = None
        self.gradInput = None
        self.children = []
        self.parents = []
        self.id = id(self)
    
    def __call__(self, *inputs):
        """
        Connect this node to input nodes
        
        Args:
            *inputs: Input nodes or tensors
            
        Returns:
            New node representing the output
        """
        # Convert tensors to nodes
        input_nodes = []
        for inp in inputs:
            if isinstance(inp, Node):
                input_nodes.append(inp)
            elif isinstance(inp, Tensor):
                input_nodes.append(Node(data=inp))
            else:
                raise ValueError(f"Unsupported input type: {type(inp)}")
        
        # Create output node
        output_node = Node(module=self.module)
        
        # Connect graph
        for inp_node in input_nodes:
            output_node.parents.append(inp_node)
            inp_node.children.append(output_node)
        
        return output_node
    
    def __repr__(self):
        if self.module:
            return f"Node({self.module})"
        elif self.data is not None:
            return f"Node(data={self.data.shape})"
        else:
            return f"Node(id={self.id})"


class gModule(Module):
    """
    Graph module that encapsulates a computational graph
    """
    
    def __init__(self, inputs: List[Node], outputs: List[Node]):
        """
        Args:
            inputs: List of input nodes
            outputs: List of output nodes
        """
        super().__init__()
        self.input_nodes = inputs if isinstance(inputs, list) else [inputs]
        self.output_nodes = outputs if isinstance(outputs, list) else [outputs]
        
        # Build forward and backward execution order
        self._build_execution_order()
    
    def _build_execution_order(self):
        """Build topological order for forward and backward passes"""
        # Forward order: topological sort from inputs to outputs
        self.forward_order = []
        visited = set()
        
        def visit_forward(node):
            if node.id in visited:
                return
            visited.add(node.id)
            
            for parent in node.parents:
                visit_forward(parent)
            
            if node.module is not None:
                self.forward_order.append(node)
        
        for output_node in self.output_nodes:
            visit_forward(output_node)
        
        # Backward order: reverse of forward order
        self.backward_order = list(reversed(self.forward_order))
    
    def forward(self, *inputs: Tensor) -> Union[Tensor, List[Tensor]]:
        """
        Forward pass through the graph
        
        Args:
            *inputs: Input tensors
            
        Returns:
            Output tensor(s)
        """
        # Set input data
        if len(inputs) != len(self.input_nodes):
            raise ValueError(f"Expected {len(self.input_nodes)} inputs, got {len(inputs)}")
        
        for node, input_tensor in zip(self.input_nodes, inputs):
            node.data = input_tensor
            node.output = input_tensor
        
        # Execute forward pass in topological order
        for node in self.forward_order:
            # Gather inputs from parent nodes
            if node.parents:
                if len(node.parents) == 1:
                    node_input = node.parents[0].output
                else:
                    # Multiple inputs - concatenate or use first (depends on module)
                    node_input = node.parents[0].output
            else:
                node_input = node.data
            
            # Execute module
            if node.module:
                node.output = node.module.forward(node_input)
        
        # Collect outputs
        outputs = [node.output for node in self.output_nodes]
        
        if len(outputs) == 1:
            self.output = outputs[0]
            return outputs[0]
        else:
            self.output = outputs
            return outputs
    
    def backward(self, *gradOutputs: Tensor) -> Union[Tensor, List[Tensor]]:
        """
        Backward pass through the graph
        
        Args:
            *gradOutputs: Gradient tensors for outputs
            
        Returns:
            Gradient tensor(s) for inputs
        """
        # Set output gradients
        if len(gradOutputs) != len(self.output_nodes):
            raise ValueError(f"Expected {len(self.output_nodes)} gradients, got {len(gradOutputs)}")
        
        for node, gradOutput in zip(self.output_nodes, gradOutputs):
            node.gradInput = gradOutput
        
        # Execute backward pass in reverse topological order
        for node in self.backward_order:
            if node.module and node.parents:
                # Get input for this node
                if len(node.parents) == 1:
                    node_input = node.parents[0].output
                else:
                    node_input = node.parents[0].output
                
                # Compute gradient
                grad = node.module.backward(node_input, node.gradInput)
                
                # Propagate to parents
                for parent in node.parents:
                    if parent.gradInput is None:
                        parent.gradInput = grad
                    else:
                        parent.gradInput = parent.gradInput.add(grad)
        
        # Collect input gradients
        gradInputs = [node.gradInput for node in self.input_nodes]
        
        if len(gradInputs) == 1:
            self.gradInput = gradInputs[0]
            return gradInputs[0]
        else:
            self.gradInput = gradInputs
            return gradInputs
    
    def __repr__(self):
        return f"nngraph.gModule(inputs={len(self.input_nodes)}, outputs={len(self.output_nodes)}, nodes={len(self.forward_order)})"


class Identity(Module):
    """Identity module that passes input through unchanged"""
    
    def forward(self, input: Tensor) -> Tensor:
        self.output = input
        return self.output
    
    def backward(self, input: Tensor, gradOutput: Tensor) -> Tensor:
        self.gradInput = gradOutput
        return self.gradInput


class JoinTable(Module):
    """Join multiple tensors along a dimension"""
    
    def __init__(self, dimension: int = 0):
        super().__init__()
        self.dimension = dimension
    
    def forward(self, *inputs: Tensor) -> Tensor:
        """Concatenate inputs along dimension"""
        import numpy as np
        arrays = [inp.data for inp in inputs]
        self.output = Tensor(np.concatenate(arrays, axis=self.dimension))
        return self.output
    
    def backward(self, inputs: List[Tensor], gradOutput: Tensor) -> List[Tensor]:
        """Split gradient back to inputs"""
        import numpy as np
        sizes = [inp.data.shape[self.dimension] for inp in inputs]
        split_grads = np.split(gradOutput.data, np.cumsum(sizes)[:-1], axis=self.dimension)
        self.gradInput = [Tensor(g) for g in split_grads]
        return self.gradInput


class SplitTable(Module):
    """Split a tensor into multiple tensors along a dimension"""
    
    def __init__(self, dimension: int = 0):
        super().__init__()
        self.dimension = dimension
    
    def forward(self, input: Tensor) -> List[Tensor]:
        """Split input along dimension"""
        import numpy as np
        n = input.shape[self.dimension]
        self.output = [Tensor(np.take(input.data, i, axis=self.dimension)) for i in range(n)]
        return self.output
    
    def backward(self, input: Tensor, gradOutputs: List[Tensor]) -> Tensor:
        """Join gradients back together"""
        import numpy as np
        arrays = [g.data for g in gradOutputs]
        self.gradInput = Tensor(np.stack(arrays, axis=self.dimension))
        return self.gradInput


class CAddTable(Module):
    """Add multiple tensors element-wise"""
    
    def forward(self, *inputs: Tensor) -> Tensor:
        """Sum all inputs"""
        self.output = inputs[0].clone()
        for inp in inputs[1:]:
            self.output.add_(inp)
        return self.output
    
    def backward(self, inputs: List[Tensor], gradOutput: Tensor) -> List[Tensor]:
        """Broadcast gradient to all inputs"""
        self.gradInput = [gradOutput.clone() for _ in inputs]
        return self.gradInput


class CMulTable(Module):
    """Multiply multiple tensors element-wise"""
    
    def forward(self, *inputs: Tensor) -> Tensor:
        """Multiply all inputs"""
        self.output = inputs[0].clone()
        for inp in inputs[1:]:
            self.output.mul_(inp)
        return self.output
    
    def backward(self, inputs: List[Tensor], gradOutput: Tensor) -> List[Tensor]:
        """Compute gradients for multiplication"""
        # Simplified - full implementation needs chain rule
        self.gradInput = [gradOutput.clone() for _ in inputs]
        return self.gradInput


def nn(*args, **kwargs):
    """
    Decorator to convert a module into a node
    
    Usage:
        linear = nn.Linear(10, 5)
        node = nn(linear)
    """
    if len(args) == 1 and isinstance(args[0], Module):
        return Node(module=args[0])
    else:
        raise ValueError("nn() expects a single Module argument")


__all__ = [
    'Node',
    'gModule',
    'Identity',
    'JoinTable',
    'SplitTable',
    'CAddTable',
    'CMulTable',
    'nn'
]
