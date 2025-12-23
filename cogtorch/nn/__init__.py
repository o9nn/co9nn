"""
CogTorch Neural Network Module
Neural network building blocks inspired by torch.nn

Provides layers, activations, and criterions for building neural networks.
"""

import numpy as np
from typing import Optional, List
import sys
import os

# Add parent directory to path for torch7 import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch7 import Module, Tensor, randn, zeros


class Linear(Module):
    """
    Linear (fully connected) layer
    Applies linear transformation: y = Wx + b
    """
    
    def __init__(self, inputSize: int, outputSize: int, bias: bool = True):
        """
        Args:
            inputSize: Size of input features
            outputSize: Size of output features
            bias: Whether to include bias term
        """
        super().__init__()
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.bias = bias
        
        # Initialize weights
        self.weight = randn(outputSize, inputSize)
        self.weight.mul_(np.sqrt(2.0 / inputSize))  # He initialization
        
        if bias:
            self.biasParam = zeros(outputSize)
        else:
            self.biasParam = None
        
        # Gradients
        self.gradWeight = zeros(outputSize, inputSize)
        if bias:
            self.gradBias = zeros(outputSize)
        else:
            self.gradBias = None
    
    def forward(self, input: Tensor) -> Tensor:
        """Forward pass"""
        # Handle both 1D and 2D inputs
        input_is_1d = (input.ndim == 1)
        
        if input_is_1d:
            # For 1D input, do dot product with each row of weight
            output_data = np.dot(self.weight.data, input.data)
            self.output = Tensor(output_data)
        else:
            # For 2D input, do matrix multiplication
            self.output = input.mm(self.weight.view(self.inputSize, self.outputSize))
        
        if self.bias:
            # Add bias
            self.output = self.output.add(self.biasParam)
        return self.output
    
    def backward(self, input: Tensor, gradOutput: Tensor) -> Tensor:
        """Backward pass"""
        # Gradient w.r.t. input
        self.gradInput = gradOutput.mm(self.weight)
        return self.gradInput
    
    def accGradParameters(self, input: Tensor, gradOutput: Tensor, scale: float = 1.0):
        """Accumulate parameter gradients"""
        # Gradient w.r.t. weight
        grad_w = Tensor(np.outer(gradOutput.data.ravel(), input.data.ravel()))
        self.gradWeight.add_(grad_w.mul(scale).view(self.outputSize, self.inputSize))
        
        # Gradient w.r.t. bias
        if self.bias:
            self.gradBias.add_(gradOutput.mul(scale))
    
    def updateParameters(self, learningRate: float):
        """Update parameters using accumulated gradients"""
        self.weight.add_(self.gradWeight.mul(-learningRate))
        if self.bias:
            self.biasParam.add_(self.gradBias.mul(-learningRate))
    
    def zeroGradParameters(self):
        """Zero out gradients"""
        self.gradWeight.zero_()
        if self.bias:
            self.gradBias.zero_()
    
    def __repr__(self):
        return f"nn.Linear({self.inputSize}, {self.outputSize})"


class Tanh(Module):
    """Hyperbolic tangent activation function"""
    
    def forward(self, input: Tensor) -> Tensor:
        self.output = Tensor(np.tanh(input.data))
        return self.output
    
    def backward(self, input: Tensor, gradOutput: Tensor) -> Tensor:
        # Derivative: 1 - tanh^2(x)
        self.gradInput = gradOutput.mul(Tensor(1.0 - np.tanh(input.data) ** 2))
        return self.gradInput


class ReLU(Module):
    """Rectified Linear Unit activation"""
    
    def forward(self, input: Tensor) -> Tensor:
        self.output = Tensor(np.maximum(0, input.data))
        return self.output
    
    def backward(self, input: Tensor, gradOutput: Tensor) -> Tensor:
        # Derivative: 1 if x > 0, else 0
        self.gradInput = gradOutput.mul(Tensor((input.data > 0).astype(np.float32)))
        return self.gradInput


class Sigmoid(Module):
    """Sigmoid activation function"""
    
    def forward(self, input: Tensor) -> Tensor:
        self.output = Tensor(1.0 / (1.0 + np.exp(-input.data)))
        return self.output
    
    def backward(self, input: Tensor, gradOutput: Tensor) -> Tensor:
        # Derivative: sigmoid(x) * (1 - sigmoid(x))
        sig = 1.0 / (1.0 + np.exp(-input.data))
        self.gradInput = gradOutput.mul(Tensor(sig * (1.0 - sig)))
        return self.gradInput


class Sequential(Module):
    """
    Sequential container for modules
    Applies modules in order
    """
    
    def __init__(self, *modules):
        super().__init__()
        self.modules = list(modules)
    
    def add(self, module: Module):
        """Add a module to the sequence"""
        self.modules.append(module)
        return self
    
    def forward(self, input: Tensor) -> Tensor:
        """Forward pass through all modules"""
        self.output = input
        for module in self.modules:
            self.output = module.forward(self.output)
        return self.output
    
    def backward(self, input: Tensor, gradOutput: Tensor) -> Tensor:
        """Backward pass through all modules in reverse"""
        # Store intermediate inputs for each module
        intermediates = [input]
        current = input
        for module in self.modules:
            current = module.forward(current)
            intermediates.append(current)
        
        # Backpropagate
        self.gradInput = gradOutput
        for i in range(len(self.modules) - 1, -1, -1):
            module = self.modules[i]
            module_input = intermediates[i]
            self.gradInput = module.backward(module_input, self.gradInput)
        
        return self.gradInput
    
    def updateParameters(self, learningRate: float):
        """Update parameters of all modules"""
        for module in self.modules:
            module.updateParameters(learningRate)
    
    def zeroGradParameters(self):
        """Zero gradients of all modules"""
        for module in self.modules:
            module.zeroGradParameters()
    
    def __repr__(self):
        s = "nn.Sequential {\n"
        for i, module in enumerate(self.modules):
            s += f"  [{i}] {module}\n"
        s += "}"
        return s


class MSECriterion(Module):
    """
    Mean Squared Error loss criterion
    """
    
    def forward(self, input: Tensor, target: Tensor) -> float:
        """Compute MSE loss"""
        diff = input.data - target.data
        self.output = float(np.mean(diff ** 2))
        return self.output
    
    def backward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute gradient of MSE"""
        n = input.size
        self.gradInput = Tensor(2.0 * (input.data - target.data) / n)
        return self.gradInput


class CrossEntropyCriterion(Module):
    """
    Cross Entropy loss for classification
    """
    
    def forward(self, input: Tensor, target: Tensor) -> float:
        """Compute cross entropy loss"""
        # Apply softmax
        exp_input = np.exp(input.data - np.max(input.data))
        self.probs = exp_input / np.sum(exp_input)
        
        # Cross entropy
        self.output = -float(np.sum(target.data * np.log(self.probs + 1e-8)))
        return self.output
    
    def backward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute gradient"""
        self.gradInput = Tensor(self.probs - target.data)
        return self.gradInput


class Dropout(Module):
    """
    Dropout layer for regularization
    """
    
    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Dropout probability
        """
        super().__init__()
        self.p = p
        self.mask = None
    
    def forward(self, input: Tensor) -> Tensor:
        """Forward pass with dropout"""
        if self.training:
            # Create dropout mask
            self.mask = Tensor(np.random.binomial(1, 1 - self.p, input.shape))
            self.output = input.mul(self.mask).mul(1.0 / (1.0 - self.p))
        else:
            self.output = input.clone()
        return self.output
    
    def backward(self, input: Tensor, gradOutput: Tensor) -> Tensor:
        """Backward pass"""
        if self.training:
            self.gradInput = gradOutput.mul(self.mask).mul(1.0 / (1.0 - self.p))
        else:
            self.gradInput = gradOutput.clone()
        return self.gradInput
    
    def __repr__(self):
        return f"nn.Dropout({self.p})"


class BatchNormalization(Module):
    """
    Batch Normalization layer
    """
    
    def __init__(self, nFeatures: int, eps: float = 1e-5, momentum: float = 0.1):
        """
        Args:
            nFeatures: Number of features
            eps: Small constant for numerical stability
            momentum: Momentum for running statistics
        """
        super().__init__()
        self.nFeatures = nFeatures
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = ones(nFeatures)  # Scale
        self.beta = zeros(nFeatures)   # Shift
        
        # Running statistics
        self.running_mean = zeros(nFeatures)
        self.running_var = ones(nFeatures)
        
        # Gradients
        self.gradGamma = zeros(nFeatures)
        self.gradBeta = zeros(nFeatures)
    
    def forward(self, input: Tensor) -> Tensor:
        """Forward pass"""
        if self.training:
            # Compute batch statistics
            self.batch_mean = input.mean(dim=0)
            self.batch_var = Tensor(np.var(input.data, axis=0))
            
            # Update running statistics
            self.running_mean = self.running_mean.mul(1 - self.momentum).add(
                self.batch_mean.mul(self.momentum)
            )
            self.running_var = self.running_var.mul(1 - self.momentum).add(
                self.batch_var.mul(self.momentum)
            )
            
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var
        
        # Normalize
        std = Tensor(np.sqrt(var.data + self.eps))
        self.normalized = Tensor((input.data - mean.data) / std.data)
        
        # Scale and shift
        self.output = self.normalized.mul(self.gamma).add(self.beta)
        return self.output
    
    def backward(self, input: Tensor, gradOutput: Tensor) -> Tensor:
        """Backward pass"""
        # This is simplified - full BatchNorm backward is more complex
        self.gradInput = gradOutput.mul(self.gamma)
        return self.gradInput
    
    def __repr__(self):
        return f"nn.BatchNormalization({self.nFeatures})"


__all__ = [
    'Linear',
    'Tanh',
    'ReLU', 
    'Sigmoid',
    'Sequential',
    'MSECriterion',
    'CrossEntropyCriterion',
    'Dropout',
    'BatchNormalization'
]
