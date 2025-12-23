"""
Inference-Driven Pattern Discovery Bridge

This module implements the integration between PLN (Probabilistic Logic Networks)
and the pattern miner, using logical inference to guide pattern mining with
semantic constraints.

Emergent Capability: semantic_pattern_mining
- PLN defines semantic constraints for patterns
- Miner finds patterns satisfying those constraints
- Discovered patterns are validated via reasoning
- Iterative refinement improves pattern quality

Architecture:
1. Define semantic pattern constraints in PLN
2. Integrate miner with inference engine
3. Add pattern validation via reasoning
4. Enable iterative refinement of patterns

Based on: Autogenesis Roadmap Feature #6
Implementation Status: Phase 1
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


class ConstraintType(Enum):
    """Types of semantic constraints for patterns"""
    STRUCTURAL = "structural"      # Graph structure constraints
    LOGICAL = "logical"           # Logical relationship constraints
    TEMPORAL = "temporal"         # Time-based constraints
    CAUSAL = "causal"            # Causality constraints
    STATISTICAL = "statistical"   # Statistical properties


class PatternQuality(Enum):
    """Quality levels for discovered patterns"""
    EXCELLENT = 4
    GOOD = 3
    ACCEPTABLE = 2
    POOR = 1
    INVALID = 0


@dataclass
class SemanticConstraint:
    """Represents a semantic constraint for pattern mining"""
    constraint_id: str
    constraint_type: ConstraintType
    description: str
    logical_formula: str  # PLN formula defining the constraint
    confidence_threshold: float = 0.7
    importance: float = 1.0
    
    def __hash__(self):
        return hash(self.constraint_id)


@dataclass
class DiscoveredPattern:
    """Represents a pattern discovered by the miner"""
    pattern_id: str
    structure: str  # Graph structure or formula
    support: float  # Frequency in data
    confidence: float  # Confidence score
    atoms: List[str] = field(default_factory=list)
    satisfies_constraints: List[str] = field(default_factory=list)
    validation_score: float = 0.0
    quality: PatternQuality = PatternQuality.ACCEPTABLE
    discovered_at: float = field(default_factory=time.time)
    
    def __hash__(self):
        return hash(self.pattern_id)


@dataclass
class MiningTask:
    """Represents a pattern mining task with semantic constraints"""
    task_id: str
    description: str
    constraints: List[SemanticConstraint]
    target_atoms: List[str]
    min_support: float = 0.1
    min_confidence: float = 0.5
    max_patterns: int = 100
    status: str = "pending"  # pending, running, completed, failed


class ConstraintValidator:
    """Validates patterns against semantic constraints using PLN"""
    
    def __init__(self):
        self.validation_cache: Dict[Tuple[str, str], float] = {}
        logger.info("ConstraintValidator initialized")
    
    def validate_pattern(
        self,
        pattern: DiscoveredPattern,
        constraint: SemanticConstraint
    ) -> Tuple[bool, float]:
        """
        Validate a pattern against a semantic constraint
        
        Args:
            pattern: Pattern to validate
            constraint: Constraint to check
            
        Returns:
            Tuple of (satisfies, confidence_score)
        """
        cache_key = (pattern.pattern_id, constraint.constraint_id)
        if cache_key in self.validation_cache:
            score = self.validation_cache[cache_key]
            return (score >= constraint.confidence_threshold, score)
        
        # Simulate PLN inference for validation
        # In real implementation, this would invoke PLN
        score = self._simulate_pln_validation(pattern, constraint)
        
        self.validation_cache[cache_key] = score
        satisfies = score >= constraint.confidence_threshold
        
        logger.debug(
            f"Pattern {pattern.pattern_id} vs constraint {constraint.constraint_id}: "
            f"{'PASS' if satisfies else 'FAIL'} (score: {score:.3f})"
        )
        
        return (satisfies, score)
    
    def _simulate_pln_validation(
        self,
        pattern: DiscoveredPattern,
        constraint: SemanticConstraint
    ) -> float:
        """
        Simulate PLN inference for pattern validation
        
        In real implementation, this would:
        1. Convert pattern to PLN formula
        2. Check if pattern satisfies constraint formula
        3. Return confidence from inference
        """
        # Base score from pattern's own confidence
        base_score = pattern.confidence * 0.6
        
        # Bonus for constraint type compatibility
        type_bonus = 0.0
        if constraint.constraint_type == ConstraintType.STRUCTURAL:
            type_bonus = 0.2
        elif constraint.constraint_type == ConstraintType.LOGICAL:
            type_bonus = 0.15
        
        # Bonus for support (frequent patterns more likely valid)
        support_bonus = pattern.support * 0.15
        
        # Add some variance
        noise = np.random.normal(0, 0.05)
        
        score = min(base_score + type_bonus + support_bonus + noise, 1.0)
        return max(score, 0.0)
    
    def validate_all_constraints(
        self,
        pattern: DiscoveredPattern,
        constraints: List[SemanticConstraint]
    ) -> Dict[str, Any]:
        """
        Validate pattern against all constraints
        
        Returns:
            Dict with validation results and overall score
        """
        results = {
            'pattern_id': pattern.pattern_id,
            'constraint_results': [],
            'satisfied_constraints': [],
            'failed_constraints': [],
            'overall_score': 0.0
        }
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for constraint in constraints:
            satisfies, score = self.validate_pattern(pattern, constraint)
            
            result = {
                'constraint_id': constraint.constraint_id,
                'satisfies': satisfies,
                'score': score,
                'importance': constraint.importance
            }
            results['constraint_results'].append(result)
            
            if satisfies:
                results['satisfied_constraints'].append(constraint.constraint_id)
                pattern.satisfies_constraints.append(constraint.constraint_id)
            else:
                results['failed_constraints'].append(constraint.constraint_id)
            
            # Weight by importance
            total_weighted_score += score * constraint.importance
            total_weight += constraint.importance
        
        results['overall_score'] = (
            total_weighted_score / total_weight if total_weight > 0 else 0.0
        )
        pattern.validation_score = results['overall_score']
        
        # Determine quality level
        pattern.quality = self._determine_quality(pattern, results)
        
        return results


    def _determine_quality(
        self,
        pattern: DiscoveredPattern,
        validation_results: Dict[str, Any]
    ) -> PatternQuality:
        """Determine overall quality of pattern"""
        score = validation_results['overall_score']
        satisfied_count = len(validation_results['satisfied_constraints'])
        total_count = len(validation_results['constraint_results'])
        
        satisfaction_ratio = (
            satisfied_count / total_count if total_count > 0 else 0.0
        )
        
        if score >= 0.9 and satisfaction_ratio >= 0.9:
            return PatternQuality.EXCELLENT
        elif score >= 0.75 and satisfaction_ratio >= 0.75:
            return PatternQuality.GOOD
        elif score >= 0.6 and satisfaction_ratio >= 0.6:
            return PatternQuality.ACCEPTABLE
        elif score >= 0.4:
            return PatternQuality.POOR
        else:
            return PatternQuality.INVALID


class PatternRefinementEngine:
    """Iteratively refines patterns based on validation feedback"""
    
    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.refinement_history: List[Dict[str, Any]] = []
        logger.info(f"PatternRefinementEngine initialized (max iterations: {max_iterations})")
    
    def refine_pattern(
        self,
        pattern: DiscoveredPattern,
        validation_results: Dict[str, Any],
        constraints: List[SemanticConstraint]
    ) -> Optional[DiscoveredPattern]:
        """
        Refine a pattern based on validation feedback
        
        Args:
            pattern: Pattern to refine
            validation_results: Results from validation
            constraints: Constraints that failed
            
        Returns:
            Refined pattern or None if cannot be refined
        """
        failed_constraints = [
            c for c in constraints
            if c.constraint_id in validation_results['failed_constraints']
        ]
        
        if not failed_constraints:
            return pattern  # Already satisfies all constraints
        
        logger.info(
            f"Refining pattern {pattern.pattern_id} "
            f"({len(failed_constraints)} failed constraints)"
        )
        
        # Create refined pattern
        refined_pattern = DiscoveredPattern(
            pattern_id=f"{pattern.pattern_id}_refined",
            structure=self._adjust_structure(pattern, failed_constraints),
            support=pattern.support * 0.95,  # Slightly lower support
            confidence=pattern.confidence,
            atoms=pattern.atoms.copy(),
            discovered_at=time.time()
        )
        
        self.refinement_history.append({
            'original_pattern': pattern.pattern_id,
            'refined_pattern': refined_pattern.pattern_id,
            'failed_constraints': [c.constraint_id for c in failed_constraints],
            'timestamp': time.time()
        })
        
        return refined_pattern
    
    def _adjust_structure(
        self,
        pattern: DiscoveredPattern,
        failed_constraints: List[SemanticConstraint]
    ) -> str:
        """
        Adjust pattern structure to better satisfy constraints
        
        In real implementation, this would use PLN to:
        1. Analyze why constraints failed
        2. Modify pattern structure accordingly
        3. Re-mine with adjusted parameters
        """
        # Simulate structural adjustment
        adjustments = []
        for constraint in failed_constraints:
            if constraint.constraint_type == ConstraintType.STRUCTURAL:
                adjustments.append("add_node")
            elif constraint.constraint_type == ConstraintType.LOGICAL:
                adjustments.append("add_relation")
            elif constraint.constraint_type == ConstraintType.TEMPORAL:
                adjustments.append("add_temporal_order")
        
        adjusted = f"{pattern.structure} + [{', '.join(adjustments)}]"
        return adjusted


class InferencePatternDiscoveryBridge:
    """
    Main bridge integrating PLN inference with pattern mining
    
    This bridge coordinates:
    1. Semantic constraint definition via PLN
    2. Constraint-guided pattern mining
    3. Pattern validation via reasoning
    4. Iterative pattern refinement
    """
    
    def __init__(
        self,
        min_support: float = 0.1,
        min_confidence: float = 0.5,
        refinement_iterations: int = 3
    ):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.refinement_iterations = refinement_iterations
        
        # Components
        self.validator = ConstraintValidator()
        self.refinement_engine = PatternRefinementEngine(refinement_iterations)
        
        # State
        self.mining_tasks: Dict[str, MiningTask] = {}
        self.discovered_patterns: Dict[str, DiscoveredPattern] = {}
        self.constraints: Dict[str, SemanticConstraint] = {}
        
        # Statistics
        self.stats = {
            'total_tasks': 0,
            'total_patterns_discovered': 0,
            'total_patterns_validated': 0,
            'total_refinements': 0,
            'excellent_patterns': 0,
            'good_patterns': 0
        }
        
        logger.info(
            f"InferencePatternDiscoveryBridge initialized "
            f"(support: {min_support}, confidence: {min_confidence})"
        )
    
    def add_constraint(self, constraint: SemanticConstraint) -> None:
        """Add a semantic constraint for pattern mining"""
        self.constraints[constraint.constraint_id] = constraint
        logger.info(f"Added constraint: {constraint.constraint_id} ({constraint.constraint_type.value})")
    
    def create_mining_task(
        self,
        task_id: str,
        description: str,
        constraint_ids: List[str],
        target_atoms: List[str],
        **kwargs
    ) -> MiningTask:
        """
        Create a new pattern mining task with semantic constraints
        
        Args:
            task_id: Unique task identifier
            description: Task description
            constraint_ids: IDs of constraints to apply
            target_atoms: Atoms to mine patterns from
            **kwargs: Additional task parameters
            
        Returns:
            Created MiningTask
        """
        # Get constraint objects
        constraints = [
            self.constraints[cid] for cid in constraint_ids
            if cid in self.constraints
        ]
        
        task = MiningTask(
            task_id=task_id,
            description=description,
            constraints=constraints,
            target_atoms=target_atoms,
            min_support=kwargs.get('min_support', self.min_support),
            min_confidence=kwargs.get('min_confidence', self.min_confidence),
            max_patterns=kwargs.get('max_patterns', 100)
        )
        
        self.mining_tasks[task_id] = task
        self.stats['total_tasks'] += 1
        
        logger.info(
            f"Created mining task '{task_id}' with {len(constraints)} constraints"
        )
        
        return task
    
    def discover_patterns(self, task: MiningTask) -> List[DiscoveredPattern]:
        """
        Discover patterns for a mining task
        
        In real implementation, this would invoke the actual miner.
        For now, we simulate pattern discovery.
        
        Args:
            task: Mining task
            
        Returns:
            List of discovered patterns
        """
        task.status = "running"
        logger.info(f"Discovering patterns for task: {task.task_id}")
        
        # Simulate pattern mining
        patterns = self._simulate_pattern_mining(task)
        
        task.status = "completed"
        self.stats['total_patterns_discovered'] += len(patterns)
        
        logger.info(f"Discovered {len(patterns)} patterns for task {task.task_id}")
        
        return patterns
    
    def _simulate_pattern_mining(self, task: MiningTask) -> List[DiscoveredPattern]:
        """
        Simulate pattern mining
        
        In real implementation, this would call the actual pattern miner.
        """
        num_patterns = min(np.random.randint(5, 15), task.max_patterns)
        patterns = []
        
        for i in range(num_patterns):
            pattern = DiscoveredPattern(
                pattern_id=f"{task.task_id}_pattern_{i}",
                structure=f"Pattern_{i}(X, Y, Z)",
                support=np.random.uniform(task.min_support, 0.8),
                confidence=np.random.uniform(task.min_confidence, 0.95),
                atoms=task.target_atoms[:3] if task.target_atoms else []
            )
            patterns.append(pattern)
            self.discovered_patterns[pattern.pattern_id] = pattern
        
        return patterns
    
    def validate_and_refine(
        self,
        patterns: List[DiscoveredPattern],
        constraints: List[SemanticConstraint]
    ) -> List[DiscoveredPattern]:
        """
        Validate patterns and iteratively refine them
        
        Args:
            patterns: Patterns to validate
            constraints: Constraints to validate against
            
        Returns:
            List of validated/refined patterns
        """
        logger.info(f"Validating {len(patterns)} patterns against {len(constraints)} constraints")
        
        validated_patterns = []
        
        for pattern in patterns:
            # Initial validation
            results = self.validator.validate_all_constraints(pattern, constraints)
            self.stats['total_patterns_validated'] += 1
            
            # Track quality
            if pattern.quality == PatternQuality.EXCELLENT:
                self.stats['excellent_patterns'] += 1
            elif pattern.quality == PatternQuality.GOOD:
                self.stats['good_patterns'] += 1
            
            # Refine if needed
            if pattern.quality in [PatternQuality.POOR, PatternQuality.ACCEPTABLE]:
                for iteration in range(self.refinement_iterations):
                    refined = self.refinement_engine.refine_pattern(
                        pattern, results, constraints
                    )
                    
                    if refined and refined.pattern_id != pattern.pattern_id:
                        self.stats['total_refinements'] += 1
                        # Re-validate refined pattern
                        results = self.validator.validate_all_constraints(
                            refined, constraints
                        )
                        
                        if refined.quality.value >= PatternQuality.GOOD.value:
                            pattern = refined
                            break
            
            if pattern.quality != PatternQuality.INVALID:
                validated_patterns.append(pattern)
        
        logger.info(
            f"Validation complete: {len(validated_patterns)}/{len(patterns)} patterns valid"
        )
        
        return validated_patterns
    
    def execute_task(self, task_id: str) -> Dict[str, Any]:
        """
        Execute a complete mining task: discover, validate, refine
        
        Args:
            task_id: ID of task to execute
            
        Returns:
            Results dictionary with discovered patterns and statistics
        """
        if task_id not in self.mining_tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.mining_tasks[task_id]
        start_time = time.time()
        
        logger.info(f"Executing task: {task_id}")
        
        # Step 1: Discover patterns
        discovered = self.discover_patterns(task)
        
        # Step 2: Validate and refine
        validated = self.validate_and_refine(discovered, task.constraints)
        
        # Sort by quality and validation score
        validated.sort(
            key=lambda p: (p.quality.value, p.validation_score),
            reverse=True
        )
        
        execution_time = time.time() - start_time
        
        results = {
            'task_id': task_id,
            'discovered_count': len(discovered),
            'validated_count': len(validated),
            'excellent_count': sum(1 for p in validated if p.quality == PatternQuality.EXCELLENT),
            'good_count': sum(1 for p in validated if p.quality == PatternQuality.GOOD),
            'patterns': validated,
            'execution_time': execution_time
        }
        
        logger.info(
            f"Task {task_id} complete: "
            f"{results['validated_count']}/{results['discovered_count']} patterns valid "
            f"({results['excellent_count']} excellent, {results['good_count']} good) "
            f"in {execution_time:.2f}s"
        )
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get bridge statistics"""
        return {
            **self.stats,
            'active_constraints': len(self.constraints),
            'active_tasks': len(self.mining_tasks),
            'total_discovered_patterns': len(self.discovered_patterns),
            'refinement_history': len(self.refinement_engine.refinement_history)
        }


def create_inference_pattern_bridge(
    min_support: float = 0.1,
    min_confidence: float = 0.5,
    refinement_iterations: int = 3
) -> InferencePatternDiscoveryBridge:
    """
    Factory function to create an inference-driven pattern discovery bridge
    
    Args:
        min_support: Minimum support threshold for patterns
        min_confidence: Minimum confidence threshold for patterns
        refinement_iterations: Maximum iterations for pattern refinement
        
    Returns:
        Configured InferencePatternDiscoveryBridge instance
    """
    return InferencePatternDiscoveryBridge(
        min_support=min_support,
        min_confidence=min_confidence,
        refinement_iterations=refinement_iterations
    )


if __name__ == "__main__":
    # Demo of inference-driven pattern discovery
    print("=" * 70)
    print("Inference-Driven Pattern Discovery Demo")
    print("=" * 70)
    
    logging.basicConfig(level=logging.INFO)
    
    # Create bridge
    bridge = create_inference_pattern_bridge(
        min_support=0.15,
        min_confidence=0.6,
        refinement_iterations=2
    )
    
    # Define semantic constraints
    constraints = [
        SemanticConstraint(
            constraint_id="struct_1",
            constraint_type=ConstraintType.STRUCTURAL,
            description="Pattern must have 3+ nodes",
            logical_formula="(NumberOf ?pattern nodes) >= 3",
            confidence_threshold=0.7,
            importance=1.0
        ),
        SemanticConstraint(
            constraint_id="logic_1",
            constraint_type=ConstraintType.LOGICAL,
            description="Pattern must be transitive",
            logical_formula="(Transitive ?pattern)",
            confidence_threshold=0.75,
            importance=1.5
        ),
        SemanticConstraint(
            constraint_id="stat_1",
            constraint_type=ConstraintType.STATISTICAL,
            description="Pattern must have high support",
            logical_formula="(Support ?pattern) > 0.3",
            confidence_threshold=0.6,
            importance=0.8
        )
    ]
    
    for constraint in constraints:
        bridge.add_constraint(constraint)
    
    # Create mining task
    task = bridge.create_mining_task(
        task_id="demo_task",
        description="Discover transitive patterns with high support",
        constraint_ids=["struct_1", "logic_1", "stat_1"],
        target_atoms=["concept_A", "concept_B", "concept_C", "concept_D"],
        max_patterns=10
    )
    
    # Execute task
    results = bridge.execute_task("demo_task")
    
    # Display results
    print(f"\n{'='*70}")
    print("Results:")
    print(f"{'='*70}")
    print(f"Discovered: {results['discovered_count']} patterns")
    print(f"Validated: {results['validated_count']} patterns")
    print(f"  - Excellent: {results['excellent_count']}")
    print(f"  - Good: {results['good_count']}")
    print(f"Execution time: {results['execution_time']:.2f}s")
    
    print(f"\n{'='*70}")
    print("Top Patterns:")
    print(f"{'='*70}")
    for i, pattern in enumerate(results['patterns'][:5], 1):
        print(f"\n{i}. {pattern.pattern_id}")
        print(f"   Quality: {pattern.quality.name}")
        print(f"   Validation Score: {pattern.validation_score:.3f}")
        print(f"   Support: {pattern.support:.3f}, Confidence: {pattern.confidence:.3f}")
        print(f"   Satisfies: {len(pattern.satisfies_constraints)}/{len(constraints)} constraints")
    
    # Statistics
    stats = bridge.get_statistics()
    print(f"\n{'='*70}")
    print("Bridge Statistics:")
    print(f"{'='*70}")
    for key, value in stats.items():
        print(f"{key}: {value}")
