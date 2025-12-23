"""
CogTorch Recurrent Neural Network Module
RNN modules inspired by torch.rnn

Provides LSTM, GRU, and basic RNN cells for sequence modeling.
"""

import numpy as np
from typing import Optional, Tuple
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch7 import Module, Tensor, randn, zeros


class LSTM(Module):
    """
    Long Short-Term Memory cell
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
        
        # Gates: input, forget, cell, output (i, f, g, o)
        # Combined weight matrix for efficiency: [i, f, g, o]
        self.weight_ih = randn(4 * hiddenSize, inputSize)
        self.weight_hh = randn(4 * hiddenSize, hiddenSize)
        self.bias_ih = zeros(4 * hiddenSize)
        self.bias_hh = zeros(4 * hiddenSize)
        
        # Initialize weights
        self._init_weights()
        
        # Hidden and cell states
        self.hidden = None
        self.cell = None
        
        # Cache for backward pass
        self.gates = None
    
    def _init_weights(self):
        """Xavier initialization"""
        std = np.sqrt(2.0 / (self.inputSize + self.hiddenSize))
        self.weight_ih.mul_(std)
        self.weight_hh.mul_(std)
    
    def forward(self, input: Tensor, 
                hidden: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass
        
        Args:
            input: Input tensor (batch_size, input_size) or (input_size,)
            hidden: Tuple of (h_0, c_0) hidden states
            
        Returns:
            output: Output tensor
            (h_n, c_n): Updated hidden states
        """
        if hidden is None:
            h = zeros(self.hiddenSize)
            c = zeros(self.hiddenSize)
        else:
            h, c = hidden
        
        # Compute gates - handle 1D input
        gates_ih_data = np.dot(self.weight_ih.data, input.data) + self.bias_ih.data
        gates_hh_data = np.dot(self.weight_hh.data, h.data) + self.bias_hh.data
        gates_data = gates_ih_data + gates_hh_data
        
        # Split into individual gates
        i = Tensor(1.0 / (1.0 + np.exp(-gates_data[0:self.hiddenSize])))  # Input gate
        f = Tensor(1.0 / (1.0 + np.exp(-gates_data[self.hiddenSize:2*self.hiddenSize])))  # Forget gate
        g = Tensor(np.tanh(gates_data[2*self.hiddenSize:3*self.hiddenSize]))  # Cell gate
        o = Tensor(1.0 / (1.0 + np.exp(-gates_data[3*self.hiddenSize:4*self.hiddenSize])))  # Output gate
        
        # Update cell state
        c_new = f.mul(c).add(i.mul(g))
        
        # Update hidden state
        h_new = o.mul(Tensor(np.tanh(c_new.data)))
        
        # Cache for backward
        self.gates = (i, f, g, o)
        self.hidden = h_new
        self.cell = c_new
        
        self.output = h_new
        return self.output, (h_new, c_new)
    
    def backward(self, input: Tensor, gradOutput: Tensor) -> Tensor:
        """Backward pass (simplified)"""
        # Full LSTM backward is complex - this is a simplified version
        self.gradInput = gradOutput.clone()
        return self.gradInput
    
    def __repr__(self):
        return f"rnn.LSTM({self.inputSize}, {self.hiddenSize})"


class GRU(Module):
    """
    Gated Recurrent Unit
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
        
        # Gates: reset, update, new (r, z, n)
        self.weight_ih = randn(3 * hiddenSize, inputSize)
        self.weight_hh = randn(3 * hiddenSize, hiddenSize)
        self.bias_ih = zeros(3 * hiddenSize)
        self.bias_hh = zeros(3 * hiddenSize)
        
        self._init_weights()
        
        self.hidden = None
    
    def _init_weights(self):
        """Xavier initialization"""
        std = np.sqrt(2.0 / (self.inputSize + self.hiddenSize))
        self.weight_ih.mul_(std)
        self.weight_hh.mul_(std)
    
    def forward(self, input: Tensor, hidden: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Forward pass
        
        Args:
            input: Input tensor
            hidden: Previous hidden state
            
        Returns:
            output: Output tensor
            h_n: Updated hidden state
        """
        if hidden is None:
            h = zeros(self.hiddenSize)
        else:
            h = hidden
        
        # Compute gates - handle 1D input
        gates_ih_data = np.dot(self.weight_ih.data, input.data) + self.bias_ih.data
        gates_hh_data = np.dot(self.weight_hh.data, h.data) + self.bias_hh.data
        
        # Reset and update gates
        r = Tensor(1.0 / (1.0 + np.exp(-(gates_ih_data[0:self.hiddenSize] + 
                                          gates_hh_data[0:self.hiddenSize]))))
        z = Tensor(1.0 / (1.0 + np.exp(-(gates_ih_data[self.hiddenSize:2*self.hiddenSize] + 
                                          gates_hh_data[self.hiddenSize:2*self.hiddenSize]))))
        
        # New gate (with reset applied)
        n = Tensor(np.tanh(gates_ih_data[2*self.hiddenSize:3*self.hiddenSize] + 
                           r.data * gates_hh_data[2*self.hiddenSize:3*self.hiddenSize]))
        
        # Update hidden state
        h_new = z.mul(h).add(Tensor((1.0 - z.data) * n.data))
        
        self.hidden = h_new
        self.output = h_new
        return self.output, h_new
    
    def backward(self, input: Tensor, gradOutput: Tensor) -> Tensor:
        """Backward pass (simplified)"""
        self.gradInput = gradOutput.clone()
        return self.gradInput
    
    def __repr__(self):
        return f"rnn.GRU({self.inputSize}, {self.hiddenSize})"


class RNN(Module):
    """
    Basic Recurrent Neural Network (Elman RNN)
    """
    
    def __init__(self, inputSize: int, hiddenSize: int, activation: str = 'tanh'):
        """
        Args:
            inputSize: Size of input features
            hiddenSize: Size of hidden state
            activation: Activation function ('tanh' or 'relu')
        """
        super().__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.activation = activation
        
        # Weights
        self.weight_ih = randn(hiddenSize, inputSize)
        self.weight_hh = randn(hiddenSize, hiddenSize)
        self.bias_ih = zeros(hiddenSize)
        self.bias_hh = zeros(hiddenSize)
        
        self._init_weights()
        
        self.hidden = None
    
    def _init_weights(self):
        """Xavier initialization"""
        std = np.sqrt(2.0 / (self.inputSize + self.hiddenSize))
        self.weight_ih.mul_(std)
        self.weight_hh.mul_(std)
    
    def forward(self, input: Tensor, hidden: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Forward pass
        
        Args:
            input: Input tensor
            hidden: Previous hidden state
            
        Returns:
            output: Output tensor
            h_n: Updated hidden state
        """
        if hidden is None:
            h = zeros(self.hiddenSize)
        else:
            h = hidden
        
        # Compute new hidden state - handle 1D input
        ih_data = np.dot(self.weight_ih.data, input.data) + self.bias_ih.data
        hh_data = np.dot(self.weight_hh.data, h.data) + self.bias_hh.data
        h_new_data = ih_data + hh_data
        
        # Apply activation
        if self.activation == 'tanh':
            h_new = Tensor(np.tanh(h_new_data))
        elif self.activation == 'relu':
            h_new = Tensor(np.maximum(0, h_new_data))
        
        self.hidden = h_new
        self.output = h_new
        return self.output, h_new
    
    def backward(self, input: Tensor, gradOutput: Tensor) -> Tensor:
        """Backward pass (simplified)"""
        self.gradInput = gradOutput.clone()
        return self.gradInput
    
    def __repr__(self):
        return f"rnn.RNN({self.inputSize}, {self.hiddenSize}, activation='{self.activation}')"


class Sequencer(Module):
    """
    Wrapper to apply RNN module over sequences
    """
    
    def __init__(self, rnn_module: Module):
        """
        Args:
            rnn_module: RNN module (LSTM, GRU, or RNN)
        """
        super().__init__()
        self.rnn = rnn_module
    
    def forward(self, input_sequence: list) -> list:
        """
        Forward pass over sequence
        
        Args:
            input_sequence: List of input tensors
            
        Returns:
            List of output tensors
        """
        outputs = []
        hidden = None
        
        for input_t in input_sequence:
            if isinstance(self.rnn, LSTM):
                output, hidden = self.rnn.forward(input_t, hidden)
            else:
                output, hidden = self.rnn.forward(input_t, hidden)
            outputs.append(output)
        
        self.output = outputs
        return outputs
    
    def backward(self, input_sequence: list, gradOutput_sequence: list) -> list:
        """Backward pass over sequence"""
        gradInputs = []
        for input_t, gradOutput_t in zip(reversed(input_sequence), reversed(gradOutput_sequence)):
            gradInput = self.rnn.backward(input_t, gradOutput_t)
            gradInputs.append(gradInput)
        
        self.gradInput = list(reversed(gradInputs))
        return self.gradInput
    
    def __repr__(self):
        return f"rnn.Sequencer({self.rnn})"


__all__ = [
    'LSTM',
    'GRU',
    'RNN',
    'Sequencer'
]
