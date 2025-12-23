# Inference-Driven Pattern Discovery

**Status:** ✅ Implemented (Phase 1)  
**Autogenesis Rank:** #6 (MEDIUM Impact, MEDIUM Complexity)  
**Emergent Capability:** `semantic_pattern_mining`

---

## Overview

Inference-Driven Pattern Discovery integrates PLN (Probabilistic Logic Networks) with the pattern miner to enable semantic-guided pattern mining. This allows the system to discover patterns that satisfy logical constraints, validate patterns through reasoning, and iteratively refine patterns based on validation feedback.

## Architecture

### Components

1. **Semantic Constraints via PLN**
   - Define logical requirements for patterns
   - Multiple constraint types (structural, logical, temporal, causal, statistical)
   - Importance weighting for constraints
   - Confidence thresholds for validation

2. **Constraint-Guided Pattern Mining**
   - Mine patterns from AtomSpace
   - Filter patterns based on support and confidence
   - Focus search on constraint-satisfying patterns
   - Efficient pattern discovery with semantic guidance

3. **Pattern Validation via Reasoning**
   - Use PLN inference to validate patterns
   - Check each pattern against all constraints
   - Calculate weighted validation scores
   - Determine pattern quality levels

4. **Iterative Pattern Refinement**
   - Refine patterns that fail constraints
   - Adjust pattern structure based on failed constraints
   - Re-mine with updated parameters
   - Converge to high-quality patterns

## Constraint Types

The system supports five types of semantic constraints:

- **STRUCTURAL**: Graph structure requirements (e.g., "must have 3+ nodes")
- **LOGICAL**: Logical relationship constraints (e.g., "must be transitive")
- **TEMPORAL**: Time-based constraints (e.g., "must have temporal ordering")
- **CAUSAL**: Causality constraints (e.g., "A must cause B")
- **STATISTICAL**: Statistical properties (e.g., "support must exceed 0.3")

## Pattern Quality Levels

Discovered patterns are classified into quality levels:

- **EXCELLENT** (4): Validation score ≥ 0.9, satisfies ≥ 90% constraints
- **GOOD** (3): Validation score ≥ 0.75, satisfies ≥ 75% constraints
- **ACCEPTABLE** (2): Validation score ≥ 0.6, satisfies ≥ 60% constraints
- **POOR** (1): Validation score ≥ 0.4
- **INVALID** (0): Validation score < 0.4

## Key Features

### 1. Semantic Constraint Definition

Define constraints that patterns must satisfy:

```python
from synergy.bridges.inference_pattern_discovery import (
    SemanticConstraint,
    ConstraintType
)

constraint = SemanticConstraint(
    constraint_id="transitive_pattern",
    constraint_type=ConstraintType.LOGICAL,
    description="Pattern must be transitive",
    logical_formula="(Transitive ?pattern)",
    confidence_threshold=0.75,
    importance=1.5
)
```

### 2. Mining Task Creation

Create tasks with multiple constraints:

```python
task = bridge.create_mining_task(
    task_id="discover_transitive_patterns",
    description="Find transitive patterns with high support",
    constraint_ids=["transitive_pattern", "high_support"],
    target_atoms=["concept_A", "concept_B", "concept_C"],
    min_support=0.2,
    min_confidence=0.7,
    max_patterns=50
)
```

### 3. Pattern Discovery and Validation

Discover patterns and validate them:

```python
# Execute complete task
results = bridge.execute_task("discover_transitive_patterns")

print(f"Discovered: {results['discovered_count']} patterns")
print(f"Validated: {results['validated_count']} patterns")
print(f"Excellent: {results['excellent_count']} patterns")

# Access validated patterns
for pattern in results['patterns']:
    print(f"Pattern: {pattern.pattern_id}")
    print(f"Quality: {pattern.quality.name}")
    print(f"Validation Score: {pattern.validation_score}")
    print(f"Satisfies: {len(pattern.satisfies_constraints)} constraints")
```

### 4. Iterative Refinement

The system automatically refines patterns that don't meet quality thresholds:

```python
# Refinement happens automatically during validation
bridge.refinement_engine.max_iterations = 5  # Configure max iterations

validated = bridge.validate_and_refine(patterns, constraints)

# Check refinement history
history = bridge.refinement_engine.refinement_history
print(f"Total refinements: {len(history)}")
```

## Usage

### Basic Usage

```python
from synergy.bridges.inference_pattern_discovery import (
    create_inference_pattern_bridge,
    SemanticConstraint,
    ConstraintType
)

# Create bridge
bridge = create_inference_pattern_bridge(
    min_support=0.15,
    min_confidence=0.6,
    refinement_iterations=3
)

# Define constraints
constraints = [
    SemanticConstraint(
        constraint_id="struct_constraint",
        constraint_type=ConstraintType.STRUCTURAL,
        description="Must have 3+ nodes",
        logical_formula="(NumberOf ?pattern nodes) >= 3",
        confidence_threshold=0.7
    ),
    SemanticConstraint(
        constraint_id="logic_constraint",
        constraint_type=ConstraintType.LOGICAL,
        description="Must be transitive",
        logical_formula="(Transitive ?pattern)",
        confidence_threshold=0.75
    )
]

# Add constraints to bridge
for constraint in constraints:
    bridge.add_constraint(constraint)

# Create mining task
task = bridge.create_mining_task(
    task_id="my_task",
    description="Discover transitive patterns",
    constraint_ids=["struct_constraint", "logic_constraint"],
    target_atoms=["concept_A", "concept_B", "concept_C"]
)

# Execute task
results = bridge.execute_task("my_task")

# Display results
print(f"Found {results['validated_count']} valid patterns")
for pattern in results['patterns'][:5]:
    print(f"- {pattern.pattern_id}: {pattern.quality.name}")
```

### Advanced Usage

```python
# Manual pattern discovery and validation
patterns = bridge.discover_patterns(task)
validated = bridge.validate_and_refine(patterns, task.constraints)

# Access constraint validator directly
validator = bridge.validator
satisfies, score = validator.validate_pattern(pattern, constraint)

# Access refinement engine
refiner = bridge.refinement_engine
refined_pattern = refiner.refine_pattern(pattern, results, constraints)

# Get detailed statistics
stats = bridge.get_statistics()
print(f"Total tasks: {stats['total_tasks']}")
print(f"Total patterns discovered: {stats['total_patterns_discovered']}")
print(f"Total refinements: {stats['total_refinements']}")
```

## Implementation Details

### File Structure

```
synergy/bridges/
└── inference_pattern_discovery.py    # Main implementation

tests/synergy/
└── test_inference_pattern_discovery.py    # Test suite (14 tests)

docs/
└── inference-driven-pattern-discovery.md    # This documentation
```

### Classes

- `SemanticConstraint`: Constraint definition with PLN formula
- `DiscoveredPattern`: Pattern discovered by miner
- `ConstraintValidator`: Validates patterns against constraints using PLN
- `PatternRefinementEngine`: Iteratively refines patterns
- `InferencePatternDiscoveryBridge`: Main coordinator

### Data Structures

- `MiningTask`: Task with constraints and parameters
- `ConstraintType`: Enum for constraint types
- `PatternQuality`: Enum for quality levels

## Testing

Run the test suite:

```bash
python3 tests/synergy/test_inference_pattern_discovery.py
```

Run the demonstration:

```bash
python3 synergy/bridges/inference_pattern_discovery.py
```

All 14 tests should pass:
- SemanticConstraints: 2 tests
- ConstraintValidator: 3 tests
- PatternRefinement: 2 tests
- InferencePatternBridge: 6 tests
- Integration: 1 test

## Performance Characteristics

### Time Complexity
- Pattern mining: O(N × P) where N = atoms, P = patterns
- Constraint validation: O(P × C) where P = patterns, C = constraints
- Pattern refinement: O(I × P × C) where I = iterations
- Overall: O(N × P + I × P × C)

### Space Complexity
- Discovered patterns: O(P)
- Validation cache: O(P × C)
- Refinement history: O(R) where R = refinements

## Configuration Parameters

### Bridge Parameters
- **min_support**: Minimum support threshold (default: 0.1)
- **min_confidence**: Minimum confidence threshold (default: 0.5)
- **refinement_iterations**: Max refinement iterations (default: 3)

### Constraint Parameters
- **confidence_threshold**: Minimum validation confidence (default: 0.7)
- **importance**: Constraint weight in scoring (default: 1.0)

### Task Parameters
- **min_support**: Task-specific support threshold
- **min_confidence**: Task-specific confidence threshold
- **max_patterns**: Maximum patterns to discover (default: 100)

## Integration with Existing Systems

### AtomSpace Integration

The bridge works with AtomSpace atoms and patterns:

```python
# Extract patterns from AtomSpace
from opencog.atomspace import AtomSpace

atomspace = AtomSpace()
# ... populate AtomSpace ...

# Create task targeting specific atoms
target_atoms = [str(atom.name) for atom in atomspace.get_atoms_by_type(types.ConceptNode)]
task = bridge.create_mining_task(
    task_id="atomspace_task",
    description="Mine patterns from AtomSpace",
    constraint_ids=constraint_ids,
    target_atoms=target_atoms
)
```

### PLN Integration

In full implementation, PLN validates patterns:

```python
# Real PLN validation (future implementation)
def validate_with_pln(pattern, constraint):
    """
    1. Convert pattern to PLN formula
    2. Convert constraint to PLN goal
    3. Run PLN inference
    4. Return confidence from inference result
    """
    pln_formula = pattern_to_pln(pattern)
    pln_goal = constraint_to_pln(constraint)
    
    result = pln_infer(pln_formula, pln_goal)
    return result.confidence
```

### Pattern Miner Integration

In full implementation, the actual miner is invoked:

```python
# Real miner integration (future implementation)
def mine_patterns_real(task):
    """
    1. Configure miner with task parameters
    2. Run miner on target atoms
    3. Filter by support and confidence
    4. Return discovered patterns
    """
    from opencog.miner import MinePattern
    
    miner = MinePattern(
        min_support=task.min_support,
        min_confidence=task.min_confidence
    )
    
    patterns = miner.mine(task.target_atoms)
    return patterns
```

## Future Enhancements

Based on autogenesis roadmap:

1. **Real PLN Integration**
   - Connect to live PLN inference engine
   - Use actual logical validation
   - Leverage PLN reasoning capabilities

2. **Real Pattern Miner Integration**
   - Connect to OpenCog pattern miner
   - Use actual frequent pattern mining
   - Leverage miner optimizations

3. **Advanced Constraint Types**
   - Semantic similarity constraints
   - Contextual constraints
   - Dynamic constraint generation

4. **Performance Optimization**
   - GPU acceleration for validation
   - Parallel pattern mining
   - Incremental refinement
   - Caching improvements

5. **Meta-Learning**
   - Learn optimal constraint weights
   - Adapt refinement strategies
   - Predict pattern quality

## Related Features

This implementation complements:

- **Cross-Modal Cognitive Fusion** (Rank #1)
- **Attention-Guided Evolutionary Learning** (Rank #2)
- **Distributed Cognitive Shard Network** (Rank #5)

## References

- Autogenesis Roadmap: `autogenesis_roadmap.json`
- Autogenesis Report: `autogenesis_report.md`
- PLN Documentation: [OpenCog Wiki - PLN](https://wiki.opencog.org/w/PLN)
- Pattern Miner: [OpenCog Wiki - Pattern Miner](https://wiki.opencog.org/w/Pattern_Miner)

## Contributing

To contribute to inference-driven pattern discovery:

1. Ensure all existing tests pass
2. Add tests for new functionality
3. Update documentation
4. Follow existing code style
5. Measure and report quality improvements

---

**Implementation Date:** December 23, 2025  
**Authors:** OpenCog Collection Contributors  
**License:** AGPL-3.0
