"""
CogTorch: Fundamental Neural Embeddings for Cognitive Architecture
Neural network library inspired by Torch7 ecosystem for the OCC (OpenCog Collection)

CogTorch provides a Python implementation of core Torch concepts optimized for
cognitive architectures and AGI research. It integrates:

- torch7: Core tensor operations and module system
- nn: Neural network building blocks (Linear, activations, containers)
- rnn: Recurrent neural networks (LSTM, GRU, RNN)
- nngraph: Graph-based neural network construction
- sys: System utilities (timing, memory tracking)
- nnn: Nested Neural Networks for multidimensional tensor shapes

The nnn module is the key innovation, enabling neural networks to operate on
complex nested structures (tuples of tuples, hierarchical data, trees) that
are common in symbolic AI and knowledge graphs.

Example Usage:
    
    # Basic neural network
    from cogtorch import nn, torch7
    
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    input = torch7.randn(784)
    output = model.forward(input)
    
    # Nested neural network
    from cogtorch import nnn
    
    nested_input = nnn.NestedTensor([
        torch7.randn(10),
        torch7.randn(5),
        nnn.NestedTensor([torch7.randn(3), torch7.randn(7)])
    ])
    
    nested_model = nnn.NestedLinear(10, 5)
    nested_output = nested_model.forward(nested_input)
    
    # Recurrent network
    from cogtorch import rnn
    
    lstm = rnn.LSTM(inputSize=10, hiddenSize=20)
    output, (h, c) = lstm.forward(input, hidden=None)

Integration with OpenCog:
    
    CogTorch is designed to integrate with OpenCog's AtomSpace for neural-symbolic
    processing. Nested tensors can represent hypergraph structures, and neural
    operations can be performed directly on knowledge graphs.
    
    See examples in the gnn/ module for AtomSpace-GNN integration.

References:
    - Torch7: http://torch.ch/
    - torch/nn: https://github.com/torch/nn
    - torch/rnn: https://github.com/torch/rnn
    - torch/nngraph: https://github.com/torch/nngraph
    - Tree-LSTM: https://arxiv.org/abs/1503.00075
"""

__version__ = "0.1.0"
__author__ = "OpenCog Collection Contributors"
__license__ = "AGPL-3.0"

# Import core modules
from . import torch7
from . import nn
from . import rnn
from . import nngraph
from . import sys
from . import nnn

# Import key classes and functions for convenience
from .torch7 import Tensor, Module, randn, zeros, ones, uniform, eye, isTensor
from .nn import (
    Linear, Tanh, ReLU, Sigmoid, Sequential,
    MSECriterion, CrossEntropyCriterion,
    Dropout, BatchNormalization
)
from .rnn import LSTM, GRU, RNN, Sequencer
from .nngraph import Node, gModule, Identity, JoinTable, SplitTable, CAddTable, CMulTable
from .sys import Clock, tic, toc, Benchmark, get_system_info
from .nnn import (
    NestedTensor, NestedModule, NestedLinear, NestedSequential,
    NestedFlatten, NestedUnflatten, TreeLSTM, NestedEmbedding,
    create_nested_tensor
)

__all__ = [
    # Modules
    'torch7',
    'nn',
    'rnn',
    'nngraph',
    'sys',
    'nnn',
    
    # Core classes
    'Tensor',
    'Module',
    
    # Tensor operations
    'randn',
    'zeros',
    'ones',
    'uniform',
    'eye',
    'isTensor',
    
    # Neural network layers
    'Linear',
    'Tanh',
    'ReLU',
    'Sigmoid',
    'Sequential',
    
    # Loss functions
    'MSECriterion',
    'CrossEntropyCriterion',
    
    # Regularization
    'Dropout',
    'BatchNormalization',
    
    # Recurrent networks
    'LSTM',
    'GRU',
    'RNN',
    'Sequencer',
    
    # Graph networks
    'Node',
    'gModule',
    'Identity',
    'JoinTable',
    'SplitTable',
    'CAddTable',
    'CMulTable',
    
    # System utilities
    'Clock',
    'tic',
    'toc',
    'Benchmark',
    'get_system_info',
    
    # Nested neural networks
    'NestedTensor',
    'NestedModule',
    'NestedLinear',
    'NestedSequential',
    'NestedFlatten',
    'NestedUnflatten',
    'TreeLSTM',
    'NestedEmbedding',
    'create_nested_tensor',
]


def info():
    """Print CogTorch information"""
    print(f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║              CogTorch Neural Embeddings                    ║
    ║          Fundamental Neural Network Library                ║
    ║         for Cognitive Architecture Research                ║
    ╚═══════════════════════════════════════════════════════════╝
    
    Version: {__version__}
    
    Modules:
      • torch7     - Core tensor operations and module system
      • nn         - Neural network layers and activations
      • rnn        - Recurrent neural networks (LSTM, GRU)
      • nngraph    - Graph-based network construction
      • sys        - System utilities and benchmarking
      • nnn        - Nested Neural Networks (NEW!)
    
    Key Features:
      ✓ Torch7-style tensor API backed by NumPy
      ✓ Full neural network module system
      ✓ LSTM, GRU, and basic RNN implementations
      ✓ Graph-based network construction
      ✓ Nested tensor structures for hierarchical data
      ✓ Tree-LSTM for processing tree-structured data
      ✓ Integration-ready for OpenCog AtomSpace
    
    Quick Start:
      >>> from cogtorch import nn, torch7
      >>> model = nn.Sequential(
      ...     nn.Linear(10, 5),
      ...     nn.ReLU()
      ... )
      >>> x = torch7.randn(10)
      >>> y = model(x)
    
    Nested Tensor Example:
      >>> from cogtorch import nnn, torch7
      >>> nested = nnn.NestedTensor([
      ...     torch7.randn(5),
      ...     torch7.randn(3)
      ... ])
      >>> print(f"Depth: {{nested.depth()}}, Size: {{nested.size()}}")
    
    For more information, see: /home/runner/work/co9nn/co9nn/cogtorch/README.md
    """)


def test_basic():
    """Run basic tests to verify installation"""
    print("Running CogTorch basic tests...\n")
    
    # Test 1: Tensor operations
    print("Test 1: Tensor operations")
    x = randn(5)
    y = randn(5)
    z = x.add(y)
    print(f"  ✓ Tensor addition: {x.shape} + {y.shape} = {z.shape}")
    
    # Test 2: Neural network
    print("\nTest 2: Neural network")
    model = Sequential(
        Linear(5, 3),
        ReLU()
    )
    output = model(x)
    print(f"  ✓ Sequential model: input {x.shape} -> output {output.shape}")
    
    # Test 3: LSTM
    print("\nTest 3: LSTM")
    lstm = LSTM(inputSize=5, hiddenSize=10)
    output, (h, c) = lstm.forward(x)
    print(f"  ✓ LSTM forward: hidden size {h.shape}")
    
    # Test 4: Nested tensor
    print("\nTest 4: Nested tensor")
    nested = NestedTensor([randn(3), randn(2)])
    print(f"  ✓ Nested tensor: depth={nested.depth()}, size={nested.size()}")
    
    # Test 5: System utilities
    print("\nTest 5: System utilities")
    clock = Clock()
    clock.tic()
    import time
    time.sleep(0.01)
    elapsed = clock.toc()
    print(f"  ✓ Clock timing: {elapsed:.6f}s")
    
    print("\n✅ All basic tests passed!")


# Show info on import (can be disabled by setting environment variable)
import os
if os.environ.get('COGTORCH_QUIET') != '1':
    print(f"CogTorch v{__version__} loaded. Type 'cogtorch.info()' for details.")
