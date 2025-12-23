"""
CogTorch Nested Neural Networks Example
Demonstrates the unique nnn module for hierarchical data structures
"""

import sys
sys.path.insert(0, '/home/runner/work/co9nn/co9nn')

import cogtorch
from cogtorch import torch7, nnn

print("=" * 60)
print("CogTorch Nested Neural Networks (nnn) Demo")
print("=" * 60)

# Example 1: Simple nested structure
print("\n1. Simple Nested Structure")
print("-" * 40)
nested = nnn.NestedTensor([
    torch7.randn(5),
    torch7.randn(3),
    torch7.randn(7)
])
print(f"Created nested tensor:")
print(f"  - Depth: {nested.depth()}")
print(f"  - Number of leaves: {nested.size()}")
print(f"  - Leaf shapes: {[leaf.shape for leaf in nested.flatten()]}")

# Example 2: Deeply nested structure (tree-like)
print("\n2. Tree-like Nested Structure")
print("-" * 40)
tree = nnn.NestedTensor([
    torch7.randn(10),  # Root-level leaf
    nnn.NestedTensor([  # First subtree
        torch7.randn(5),
        torch7.randn(3)
    ]),
    nnn.NestedTensor([  # Second subtree
        torch7.randn(4),
        nnn.NestedTensor([  # Nested subtree
            torch7.randn(2),
            torch7.randn(6)
        ])
    ])
])
print(f"Created tree structure:")
print(f"  - Depth: {tree.depth()}")
print(f"  - Number of leaves: {tree.size()}")

# Example 3: Apply function to all leaves
print("\n3. Apply Function to All Leaves")
print("-" * 40)
# Scale all tensors by 2
scaled = nested.apply(lambda x: x.mul(2.0))
print("Applied scaling function (x * 2) to all leaves")
print(f"  - Original first leaf mean: {nested.flatten()[0].mean():.3f}")
print(f"  - Scaled first leaf mean: {scaled.flatten()[0].mean():.3f}")

# Example 4: Neural network on nested structure
print("\n4. Neural Network Processing")
print("-" * 40)
# Create nested input with uniform size
uniform_nested = nnn.NestedTensor([
    torch7.randn(10),
    torch7.randn(10),
    torch7.randn(10)
])

# Process with NestedLinear
nested_linear = nnn.NestedLinear(inputSize=10, outputSize=5)
output = nested_linear.forward(uniform_nested)
print(f"NestedLinear(10 -> 5):")
print(f"  - Input leaves: {uniform_nested.size()}")
print(f"  - Output leaves: {output.size()}")
print(f"  - Output leaf shapes: {[leaf.shape for leaf in output.flatten()]}")

# Example 5: Flatten and reconstruct
print("\n5. Flatten and Reconstruct")
print("-" * 40)
flattener = nnn.NestedFlatten()
flat = flattener.forward(output)
print(f"Flattened nested structure:")
print(f"  - Original structure: {output.size()} leaves")
print(f"  - Flattened shape: {flat.shape}")

# Example 6: Tree-LSTM for hierarchical processing
print("\n6. Tree-LSTM Processing")
print("-" * 40)
tree_lstm = nnn.TreeLSTM(inputSize=10, hiddenSize=20)
tree_input = nnn.NestedTensor([
    torch7.randn(10),
    nnn.NestedTensor([
        torch7.randn(10),
        torch7.randn(10)
    ])
])
tree_output = tree_lstm.forward(tree_input)
print(f"Tree-LSTM processed hierarchical structure:")
print(f"  - Input depth: {tree_input.depth()}")
print(f"  - Output: {tree_output}")

print("\n" + "=" * 60)
print("✅ Nested Neural Networks demo completed!")
print("=" * 60)
print("\nUse cases:")
print("  • Parse trees in NLP")
print("  • Hypergraph structures (OpenCog AtomSpace)")
print("  • Hierarchical knowledge representations")
print("  • Variable-length sequences")
print("  • Recursive data structures")
