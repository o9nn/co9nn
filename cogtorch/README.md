# CogTorch: Fundamental Neural Embeddings for Cognitive Architecture

**CogTorch** provides fundamental neural network building blocks inspired by the [Torch7](https://github.com/torch/torch7) ecosystem, optimized for cognitive architecture research and AGI development within the OpenCog Collection (OCC).

## Overview

CogTorch implements Python versions of core Torch concepts with a focus on:

1. **Cognitive Architecture Integration**: Designed to work seamlessly with OpenCog's AtomSpace and hypergraph structures
2. **Nested Tensor Structures**: Unique `nnn` module for handling hierarchical and tree-structured data
3. **Minimal Dependencies**: Built on NumPy for broad compatibility
4. **Research-Friendly**: Clear, readable implementations suitable for experimentation

## Modules

### 📦 torch7 - Core Tensor Operations

Implements the fundamental `Tensor` and `Module` classes:

```python
from cogtorch import torch7

# Create tensors
x = torch7.randn(5, 10)  # Random tensor
y = torch7.zeros(5, 10)  # Zero tensor
z = torch7.ones(3, 3)    # Ones tensor

# Tensor operations
result = x.add(y).mul(2.0)
norm = x.norm()
mean = x.mean()

# Module base class for neural networks
class MyModule(torch7.Module):
    def forward(self, input):
        return input.mul(2.0)
```

**Key Features:**
- Tensor operations: `add`, `mul`, `dot`, `mm` (matrix multiply), `norm`, `mean`, `sum`
- In-place operations: `add_`, `mul_`, `zero_`, `fill_`
- Reshaping: `view`, `resize_`, `clone`
- NumPy backend for performance

### 🧠 nn - Neural Network Layers

Standard neural network building blocks:

```python
from cogtorch import nn, torch7

# Build a feedforward network
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 128),
    nn.Tanh(),
    nn.Linear(128, 10)
)

# Forward pass
input = torch7.randn(784)
output = model(input)

# Loss functions
criterion = nn.MSECriterion()
loss = criterion.forward(output, target)
```

**Available Layers:**
- `Linear` - Fully connected layer
- `Sequential` - Container for sequential modules
- `Dropout` - Regularization
- `BatchNormalization` - Batch normalization

**Activations:**
- `ReLU` - Rectified Linear Unit
- `Tanh` - Hyperbolic tangent
- `Sigmoid` - Sigmoid activation

**Criterions (Loss Functions):**
- `MSECriterion` - Mean Squared Error
- `CrossEntropyCriterion` - Cross Entropy

### 🔄 rnn - Recurrent Neural Networks

LSTM, GRU, and basic RNN implementations:

```python
from cogtorch import rnn, torch7

# LSTM cell
lstm = rnn.LSTM(inputSize=128, hiddenSize=256)
input = torch7.randn(128)
output, (h_new, c_new) = lstm.forward(input, hidden=None)

# GRU cell
gru = rnn.GRU(inputSize=128, hiddenSize=256)
output, h_new = gru.forward(input, hidden=None)

# Basic RNN
basic_rnn = rnn.RNN(inputSize=128, hiddenSize=256, activation='tanh')
output, h_new = basic_rnn.forward(input)

# Process sequences
sequencer = rnn.Sequencer(lstm)
sequence = [torch7.randn(128) for _ in range(10)]
outputs = sequencer.forward(sequence)
```

**Features:**
- Full LSTM implementation with input/forget/output gates
- GRU with reset and update gates
- Basic Elman RNN with tanh/relu activation
- Sequencer wrapper for batch sequence processing

### 🕸️ nngraph - Neural Network Graphs

Build complex architectures with branching and merging:

```python
from cogtorch import nngraph, nn, torch7

# Create modules as nodes
input_node = nngraph.Node()
linear1 = nngraph.Node(nn.Linear(10, 20))
linear2 = nngraph.Node(nn.Linear(10, 20))
add = nngraph.Node(nngraph.CAddTable())
output = nngraph.Node(nn.Tanh())

# Build graph
branch1 = linear1(input_node)
branch2 = linear2(input_node)
merged = add(branch1, branch2)
result = output(merged)

# Create executable graph module
graph_model = nngraph.gModule([input_node], [result])

# Execute
input = torch7.randn(10)
output = graph_model.forward(input)
```

**Components:**
- `Node` - Computational graph node
- `gModule` - Executable graph module
- `JoinTable` - Concatenate tensors
- `SplitTable` - Split tensors
- `CAddTable` - Element-wise addition
- `CMulTable` - Element-wise multiplication

### ⏱️ sys - System Utilities

Timing, benchmarking, and system information:

```python
from cogtorch import sys

# Timing
clock = sys.Clock()
clock.tic()
# ... do work ...
elapsed = clock.toc()
print(f"Elapsed: {elapsed:.6f}s")

# Benchmarking
benchmark = sys.Benchmark("MyBenchmark")
for _ in range(100):
    with benchmark:
        # ... code to benchmark ...
        pass
print(benchmark.report())

# System info
info = sys.get_system_info()
print(f"OS: {info['os']}, CPU Count: {info['cpu_count']}")

# Memory tracking
sys.track_memory("model_weights", 1024 * 1024)  # 1 MB
mem_info = sys.get_memory_info()
```

**Features:**
- High-resolution timing with `Clock`
- Benchmarking with statistics
- Memory tracking
- System information

### 🌳 nnn - Nested Neural Networks ⭐

**The key innovation:** Process nested, hierarchical tensor structures common in symbolic AI and knowledge graphs.

```python
from cogtorch import nnn, torch7

# Create nested tensor structures
nested = nnn.NestedTensor([
    torch7.randn(5),           # First branch
    torch7.randn(3),           # Second branch
    nnn.NestedTensor([         # Nested sub-branch
        torch7.randn(2),
        torch7.randn(4)
    ])
])

print(f"Depth: {nested.depth()}")  # 2
print(f"Size: {nested.size()}")    # 4 leaf tensors

# Apply function to all leaves
scaled = nested.apply(lambda x: x.mul(2.0))

# Neural networks on nested structures
model = nnn.NestedLinear(inputSize=10, outputSize=5)
output = model.forward(nested)

# Tree-LSTM for hierarchical data
tree_lstm = nnn.TreeLSTM(inputSize=128, hiddenSize=256)
tree_output = tree_lstm.forward(nested)

# Flatten and unflatten
flattener = nnn.NestedFlatten()
flat_tensor = flattener.forward(nested)

template = nnn.create_nested_tensor((5,), (3,), (2,))
unflattener = nnn.NestedUnflatten(template)
reconstructed = unflattener.forward(flat_tensor)
```

**Use Cases:**
- **Tree-structured data**: Parse trees, expression trees, program ASTs
- **Hypergraph processing**: OpenCog AtomSpace structures
- **Hierarchical knowledge**: Nested concepts, taxonomies
- **Variable-length sequences**: Ragged arrays, jagged tensors
- **Recursive structures**: Lisp-like S-expressions, JSON-like objects

**Key Classes:**
- `NestedTensor` - Nested tensor container
- `NestedLinear` - Linear layer for nested structures
- `NestedSequential` - Sequential container
- `TreeLSTM` - LSTM for tree-structured data
- `NestedEmbedding` - Embeddings for nested structures
- `NestedFlatten` / `NestedUnflatten` - Structure conversion

## Integration with OpenCog

CogTorch is designed to integrate with OpenCog's AtomSpace:

```python
from cogtorch import nnn, torch7
from opencog.atomspace import AtomSpace, types

# Represent AtomSpace atoms as nested tensors
atomspace = AtomSpace()

# Create hierarchical structure
concept_a = atomspace.add_node(types.ConceptNode, "A")
concept_b = atomspace.add_node(types.ConceptNode, "B")
inheritance = atomspace.add_link(types.InheritanceLink, [concept_a, concept_b])

# Convert to nested tensor (simplified)
def atom_to_nested_tensor(atom):
    # Each atom becomes a tensor based on its features
    features = torch7.randn(128)  # Embedding
    
    if atom.is_node():
        return nnn.NestedTensor(features)
    else:
        # Link contains child atoms
        children = [atom_to_nested_tensor(child) for child in atom.out]
        return nnn.NestedTensor(children)

nested_atom = atom_to_nested_tensor(inheritance)

# Process with TreeLSTM
tree_lstm = nnn.TreeLSTM(inputSize=128, hiddenSize=256)
result = tree_lstm.forward(nested_atom)
```

See the `gnn/` module for full AtomSpace-GNN integration examples.

## Installation

CogTorch is part of the OpenCog Collection. To use:

```bash
# From OCC root directory
cd /path/to/co9nn

# Import in Python
import sys
sys.path.insert(0, '/path/to/co9nn')
import cogtorch

# Or add to PYTHONPATH
export PYTHONPATH=/path/to/co9nn:$PYTHONPATH
python -c "import cogtorch; cogtorch.info()"
```

**Dependencies:**
- Python 3.7+
- NumPy

**Optional:**
- OpenCog AtomSpace (for integration)

## Examples

### Example 1: Simple Feedforward Network

```python
from cogtorch import nn, torch7

# Create model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Training loop (simplified)
for epoch in range(10):
    # Forward pass
    input = torch7.randn(784)
    target = torch7.zeros(10)
    target.data[5] = 1.0  # One-hot encoding
    
    output = model(input)
    
    # Compute loss
    criterion = nn.MSECriterion()
    loss = criterion.forward(output, target)
    
    # Backward pass
    grad = criterion.backward(output, target)
    model.backward(input, grad)
    
    # Update parameters
    model.updateParameters(learningRate=0.01)
    
    print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

### Example 2: LSTM Sequence Modeling

```python
from cogtorch import rnn, torch7

# Create LSTM
lstm = rnn.LSTM(inputSize=50, hiddenSize=100)

# Process sequence
hidden = None
sequence = [torch7.randn(50) for _ in range(20)]

for t, input_t in enumerate(sequence):
    output, hidden = lstm.forward(input_t, hidden)
    print(f"Step {t}: output shape {output.shape}")
```

### Example 3: Tree-Structured Processing

```python
from cogtorch import nnn, torch7

# Create tree structure
#       root
#      /    \
#    child1  child2
#           /      \
#        leaf1    leaf2

leaf1 = torch7.randn(10)
leaf2 = torch7.randn(10)
child2 = nnn.NestedTensor([leaf1, leaf2])
child1 = torch7.randn(10)
root = nnn.NestedTensor([child1, child2])

# Process with TreeLSTM
tree_lstm = nnn.TreeLSTM(inputSize=10, hiddenSize=20)
root_representation = tree_lstm.forward(root)

print(f"Tree depth: {root.depth()}")
print(f"Leaf count: {root.size()}")
print(f"Root representation: {root_representation.data.shape}")
```

### Example 4: Graph Neural Network

```python
from cogtorch import nngraph, nn, torch7

# Build graph with skip connections
input1 = nngraph.Node()
input2 = nngraph.Node()

# Parallel processing
branch1 = nngraph.Node(nn.Linear(10, 20))(input1)
branch2 = nngraph.Node(nn.Linear(15, 20))(input2)

# Merge branches
merged = nngraph.Node(nngraph.CAddTable())(branch1, branch2)

# Final layer
output = nngraph.Node(nn.Tanh())(merged)

# Create graph module
model = nngraph.gModule([input1, input2], [output])

# Execute
x1 = torch7.randn(10)
x2 = torch7.randn(15)
y = model.forward(x1, x2)
```

## Benchmarking

```python
from cogtorch import sys, nn, torch7

# Benchmark forward pass
model = nn.Sequential(
    nn.Linear(1000, 500),
    nn.ReLU(),
    nn.Linear(500, 100)
)

benchmark = sys.Benchmark("Forward Pass")
for _ in range(100):
    with benchmark:
        input = torch7.randn(1000)
        output = model(input)

print(benchmark.report())
```

## Testing

Run basic tests:

```python
import cogtorch
cogtorch.test_basic()
```

## Architecture Comparison

| Feature | PyTorch | Torch7 (Lua) | CogTorch |
|---------|---------|--------------|----------|
| Language | Python | Lua | Python |
| Backend | C++/CUDA | C/CUDA | NumPy |
| Autograd | Yes | No (manual) | No (manual) |
| Nested Tensors | Limited | No | **Yes (nnn)** |
| AtomSpace Integration | No | No | **Yes** |
| Dynamic Graphs | Yes | Yes | Yes (nngraph) |
| Production Ready | Yes | Legacy | Research |

## Contributing

CogTorch is part of the OpenCog Collection. Contributions welcome!

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## References

1. **Torch7**: http://torch.ch/
2. **torch/nn**: https://github.com/torch/nn
3. **torch/rnn**: https://github.com/torch/rnn  
4. **torch/nngraph**: https://github.com/torch/nngraph
5. **torch/sys**: https://github.com/torch/sys
6. **Tree-LSTM**: Tai et al., "Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks", ACL 2015
7. **OpenCog AtomSpace**: https://wiki.opencog.org/w/AtomSpace
8. **Graph Neural Networks**: Battaglia et al., "Relational inductive biases, deep learning, and graph networks", 2018

## License

AGPL-3.0 - See [LICENSE](../LICENSE)

## Authors

OpenCog Collection Contributors

---

**CogTorch v0.1.0** - Fundamental Neural Embeddings for Cognitive Architecture
