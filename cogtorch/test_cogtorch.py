"""
CogTorch Integration Test Suite
Tests all modules: torch7, nn, rnn, nngraph, sys, nnn
"""

import sys
import os

# Add cogtorch to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cogtorch
from cogtorch import torch7, nn, rnn, nngraph, sys as cogsys, nnn


def test_torch7():
    """Test torch7 module"""
    print("\n=== Testing torch7 ===")
    
    # Test tensor creation
    x = torch7.randn(5, 10)
    assert x.shape == (5, 10), "Tensor shape mismatch"
    print(f"✓ Created tensor with shape {x.shape}")
    
    # Test operations
    y = torch7.zeros(5, 10)
    z = x.add(y)
    assert z.shape == x.shape, "Addition shape mismatch"
    print(f"✓ Tensor addition works")
    
    # Test matrix multiplication
    a = torch7.randn(3, 5)
    b = torch7.randn(5, 4)
    c = a.mm(b)
    assert c.shape == (3, 4), "Matrix multiplication shape mismatch"
    print(f"✓ Matrix multiplication: {a.shape} x {b.shape} = {c.shape}")
    
    print("✅ torch7 tests passed")


def test_nn():
    """Test nn module"""
    print("\n=== Testing nn ===")
    
    # Test Linear layer
    linear = nn.Linear(10, 5)
    x = torch7.randn(10)
    y = linear(x)
    assert y.shape == (5,), f"Linear output shape mismatch: {y.shape}"
    print(f"✓ Linear layer: {x.shape} -> {y.shape}")
    
    # Test activations
    relu = nn.ReLU()
    tanh = nn.Tanh()
    y_relu = relu(x)
    y_tanh = tanh(x)
    print(f"✓ ReLU and Tanh activations work")
    
    # Test Sequential
    model = nn.Sequential(
        nn.Linear(10, 8),
        nn.ReLU(),
        nn.Linear(8, 5)
    )
    output = model(x)
    assert output.shape == (5,), "Sequential output shape mismatch"
    print(f"✓ Sequential model: {x.shape} -> {output.shape}")
    
    # Test loss
    target = torch7.randn(5)
    criterion = nn.MSECriterion()
    loss = criterion.forward(output, target)
    print(f"✓ MSE loss computed: {loss:.4f}")
    
    print("✅ nn tests passed")


def test_rnn():
    """Test rnn module"""
    print("\n=== Testing rnn ===")
    
    # Test LSTM
    lstm = rnn.LSTM(inputSize=10, hiddenSize=20)
    x = torch7.randn(10)
    output, (h, c) = lstm.forward(x)
    assert output.shape == (20,), f"LSTM output shape mismatch: {output.shape}"
    assert h.shape == (20,), f"LSTM hidden shape mismatch: {h.shape}"
    assert c.shape == (20,), f"LSTM cell shape mismatch: {c.shape}"
    print(f"✓ LSTM: input {x.shape} -> output {output.shape}, hidden {h.shape}")
    
    # Test GRU
    gru = rnn.GRU(inputSize=10, hiddenSize=15)
    output, h = gru.forward(x)
    assert output.shape == (15,), f"GRU output shape mismatch"
    print(f"✓ GRU: input {x.shape} -> output {output.shape}")
    
    # Test basic RNN
    basic_rnn = rnn.RNN(inputSize=10, hiddenSize=12)
    output, h = basic_rnn.forward(x)
    assert output.shape == (12,), f"RNN output shape mismatch"
    print(f"✓ Basic RNN: input {x.shape} -> output {output.shape}")
    
    # Test Sequencer
    sequencer = rnn.Sequencer(lstm)
    sequence = [torch7.randn(10) for _ in range(5)]
    outputs = sequencer.forward(sequence)
    assert len(outputs) == 5, "Sequencer output length mismatch"
    print(f"✓ Sequencer processed {len(sequence)} timesteps")
    
    print("✅ rnn tests passed")


def test_nngraph():
    """Test nngraph module"""
    print("\n=== Testing nngraph ===")
    
    # Test basic node
    linear = nn.Linear(10, 5)
    node = nngraph.Node(linear)
    print(f"✓ Created node: {node}")
    
    # Test graph operations
    add_table = nngraph.CAddTable()
    x = torch7.randn(5)
    y = torch7.randn(5)
    z = add_table.forward(x, y)
    print(f"✓ CAddTable: {x.shape} + {y.shape} = {z.shape}")
    
    print("✅ nngraph tests passed")


def test_sys():
    """Test sys module"""
    print("\n=== Testing sys ===")
    
    # Test Clock
    clock = cogsys.Clock()
    clock.tic()
    import time
    time.sleep(0.01)
    elapsed = clock.toc()
    assert elapsed >= 0.01, "Clock timing too short"
    print(f"✓ Clock timing: {elapsed:.6f}s")
    
    # Test Benchmark
    benchmark = cogsys.Benchmark("Test")
    for _ in range(3):
        with benchmark:
            time.sleep(0.001)
    avg = benchmark.get_average()
    print(f"✓ Benchmark: {benchmark.get_count()} runs, avg={avg:.6f}s")
    
    # Test system info
    info = cogsys.get_system_info()
    assert 'os' in info, "Missing OS info"
    print(f"✓ System info: OS={info['os']}, CPUs={info['cpu_count']}")
    
    print("✅ sys tests passed")


def test_nnn():
    """Test nnn module"""
    print("\n=== Testing nnn (Nested Neural Networks) ===")
    
    # Test NestedTensor creation
    nested = nnn.NestedTensor([
        torch7.randn(5),
        torch7.randn(3)
    ])
    assert nested.size() == 2, "Nested tensor size mismatch"
    assert nested.depth() == 1, "Nested tensor depth mismatch (should be 1 for list of tensors)"
    print(f"✓ NestedTensor: depth={nested.depth()}, size={nested.size()}")
    
    # Test nested with deeper nesting
    deep_nested = nnn.NestedTensor([
        torch7.randn(5),
        nnn.NestedTensor([
            torch7.randn(3),
            torch7.randn(2)
        ])
    ])
    assert deep_nested.size() == 3, "Deep nested size mismatch"
    assert deep_nested.depth() == 2, "Deep nested depth mismatch (1 + max(0, 1) = 2)"
    print(f"✓ Deep NestedTensor: depth={deep_nested.depth()}, size={deep_nested.size()}")
    
    # Test apply
    scaled = nested.apply(lambda x: x.mul(2.0))
    print(f"✓ Applied function to all leaves")
    
    # Test flatten
    flat = nested.flatten()
    assert len(flat) == 2, "Flatten output length mismatch"
    print(f"✓ Flattened to {len(flat)} tensors")
    
    # Test NestedLinear
    nested_model = nnn.NestedLinear(inputSize=5, outputSize=3)
    # Create nested input with consistent size
    nested_input = nnn.NestedTensor([
        torch7.randn(5),
        torch7.randn(5)
    ])
    output = nested_model.forward(nested_input)
    print(f"✓ NestedLinear processed nested structure")
    
    # Test TreeLSTM
    tree_lstm = nnn.TreeLSTM(inputSize=10, hiddenSize=20)
    tree_input = nnn.NestedTensor(torch7.randn(10))
    tree_output = tree_lstm.forward(tree_input)
    print(f"✓ TreeLSTM processed tree structure")
    
    print("✅ nnn tests passed")


def test_integration():
    """Test integration of multiple modules"""
    print("\n=== Testing Integration ===")
    
    # Build a model using multiple modules
    # Sequential model with RNN on nested data
    
    # 1. Create nested input
    nested_input = nnn.NestedTensor([
        torch7.randn(10),
        torch7.randn(10),
        torch7.randn(10)
    ])
    print(f"✓ Created nested input with {nested_input.size()} leaves")
    
    # 2. Process with NestedLinear
    nested_linear = nnn.NestedLinear(inputSize=10, outputSize=8)
    nested_features = nested_linear.forward(nested_input)
    print(f"✓ Processed with NestedLinear")
    
    # 3. Flatten for sequence processing
    flattener = nnn.NestedFlatten()
    flat_features = flattener.forward(nested_features)
    print(f"✓ Flattened to tensor of shape {flat_features.shape}")
    
    # 4. Build standard neural network
    model = nn.Sequential(
        nn.Linear(flat_features.size, 64),
        nn.ReLU(),
        nn.Linear(64, 32)
    )
    final_output = model.forward(flat_features)
    print(f"✓ Final output shape: {final_output.shape}")
    
    # 5. Benchmark the pipeline
    benchmark = cogsys.Benchmark("Full Pipeline")
    for _ in range(10):
        with benchmark:
            nested_input = nnn.NestedTensor([torch7.randn(10) for _ in range(3)])
            nested_features = nested_linear.forward(nested_input)
            flat_features = flattener.forward(nested_features)
            final_output = model.forward(flat_features)
    
    print(f"✓ Pipeline benchmark: {benchmark.get_average():.6f}s average")
    
    print("✅ Integration tests passed")


def main():
    """Run all tests"""
    print("=" * 60)
    print("CogTorch Test Suite")
    print("=" * 60)
    
    try:
        test_torch7()
        test_nn()
        test_rnn()
        test_nngraph()
        test_sys()
        test_nnn()
        test_integration()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
