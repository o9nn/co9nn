#!/usr/bin/env python3
"""
Tests for Inference-Driven Pattern Discovery Bridge

Tests the integration between PLN reasoning and pattern mining
with semantic constraints, validation, and iterative refinement.
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from synergy.bridges.inference_pattern_discovery import (
    InferencePatternDiscoveryBridge,
    create_inference_pattern_bridge,
    SemanticConstraint,
    DiscoveredPattern,
    ConstraintType,
    PatternQuality,
    ConstraintValidator,
    PatternRefinementEngine,
    MiningTask
)


class TestSemanticConstraints(unittest.TestCase):
    """Test semantic constraint functionality"""
    
    def test_create_constraint(self):
        """Test creating semantic constraints"""
        constraint = SemanticConstraint(
            constraint_id="test_1",
            constraint_type=ConstraintType.STRUCTURAL,
            description="Test constraint",
            logical_formula="(Test ?x)",
            confidence_threshold=0.7
        )
        
        self.assertEqual(constraint.constraint_id, "test_1")
        self.assertEqual(constraint.constraint_type, ConstraintType.STRUCTURAL)
        self.assertEqual(constraint.confidence_threshold, 0.7)
    
    def test_constraint_types(self):
        """Test different constraint types"""
        types = [
            ConstraintType.STRUCTURAL,
            ConstraintType.LOGICAL,
            ConstraintType.TEMPORAL,
            ConstraintType.CAUSAL,
            ConstraintType.STATISTICAL
        ]
        
        for ctype in types:
            constraint = SemanticConstraint(
                constraint_id=f"test_{ctype.value}",
                constraint_type=ctype,
                description=f"Test {ctype.value}",
                logical_formula=f"({ctype.value} ?x)"
            )
            self.assertEqual(constraint.constraint_type, ctype)


class TestConstraintValidator(unittest.TestCase):
    """Test constraint validation functionality"""
    
    def setUp(self):
        self.validator = ConstraintValidator()
    
    def test_validate_pattern(self):
        """Test validating a pattern against a constraint"""
        pattern = DiscoveredPattern(
            pattern_id="test_pattern",
            structure="TestPattern(X, Y)",
            support=0.5,
            confidence=0.8
        )
        
        constraint = SemanticConstraint(
            constraint_id="test_constraint",
            constraint_type=ConstraintType.LOGICAL,
            description="Test",
            logical_formula="(Test ?x)",
            confidence_threshold=0.6
        )
        
        satisfies, score = self.validator.validate_pattern(pattern, constraint)
        
        self.assertIsInstance(satisfies, bool)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_validate_all_constraints(self):
        """Test validating against multiple constraints"""
        pattern = DiscoveredPattern(
            pattern_id="multi_test",
            structure="TestPattern(X, Y, Z)",
            support=0.6,
            confidence=0.85
        )
        
        constraints = [
            SemanticConstraint(
                constraint_id=f"constraint_{i}",
                constraint_type=ConstraintType.STRUCTURAL,
                description=f"Constraint {i}",
                logical_formula=f"(Test_{i} ?x)",
                confidence_threshold=0.5
            )
            for i in range(3)
        ]
        
        results = self.validator.validate_all_constraints(pattern, constraints)
        
        self.assertIn('pattern_id', results)
        self.assertIn('overall_score', results)
        self.assertIn('satisfied_constraints', results)
        self.assertEqual(len(results['constraint_results']), 3)
    
    def test_quality_determination(self):
        """Test pattern quality determination"""
        high_quality_pattern = DiscoveredPattern(
            pattern_id="high_quality",
            structure="HighQuality(X)",
            support=0.9,
            confidence=0.95
        )
        
        constraint = SemanticConstraint(
            constraint_id="easy_constraint",
            constraint_type=ConstraintType.STRUCTURAL,
            description="Easy constraint",
            logical_formula="(Easy ?x)",
            confidence_threshold=0.5
        )
        
        results = self.validator.validate_all_constraints(
            high_quality_pattern,
            [constraint]
        )
        
        # High quality patterns should get good scores
        self.assertGreaterEqual(results['overall_score'], 0.5)


class TestPatternRefinement(unittest.TestCase):
    """Test pattern refinement functionality"""
    
    def setUp(self):
        self.engine = PatternRefinementEngine(max_iterations=3)
    
    def test_refine_pattern(self):
        """Test pattern refinement"""
        pattern = DiscoveredPattern(
            pattern_id="needs_refinement",
            structure="SimplePattern(X)",
            support=0.4,
            confidence=0.6
        )
        
        constraint = SemanticConstraint(
            constraint_id="complex_constraint",
            constraint_type=ConstraintType.LOGICAL,
            description="Complex constraint",
            logical_formula="(Complex ?x)",
            confidence_threshold=0.8
        )
        
        validation_results = {
            'pattern_id': pattern.pattern_id,
            'failed_constraints': ['complex_constraint'],
            'overall_score': 0.5
        }
        
        refined = self.engine.refine_pattern(
            pattern,
            validation_results,
            [constraint]
        )
        
        self.assertIsNotNone(refined)
        self.assertNotEqual(refined.pattern_id, pattern.pattern_id)
        self.assertIn("refined", refined.pattern_id)
    
    def test_refinement_history(self):
        """Test that refinement history is tracked"""
        initial_history_len = len(self.engine.refinement_history)
        
        pattern = DiscoveredPattern(
            pattern_id="history_test",
            structure="Test(X)",
            support=0.5,
            confidence=0.7
        )
        
        constraint = SemanticConstraint(
            constraint_id="test_constraint",
            constraint_type=ConstraintType.STRUCTURAL,
            description="Test",
            logical_formula="(Test ?x)"
        )
        
        validation_results = {
            'pattern_id': pattern.pattern_id,
            'failed_constraints': ['test_constraint'],
            'overall_score': 0.4
        }
        
        self.engine.refine_pattern(pattern, validation_results, [constraint])
        
        self.assertEqual(
            len(self.engine.refinement_history),
            initial_history_len + 1
        )


class TestInferencePatternBridge(unittest.TestCase):
    """Test the main inference pattern discovery bridge"""
    
    def setUp(self):
        self.bridge = create_inference_pattern_bridge(
            min_support=0.1,
            min_confidence=0.5,
            refinement_iterations=2
        )
    
    def test_add_constraint(self):
        """Test adding semantic constraints"""
        constraint = SemanticConstraint(
            constraint_id="bridge_test_1",
            constraint_type=ConstraintType.LOGICAL,
            description="Test constraint",
            logical_formula="(Test ?x)"
        )
        
        initial_count = len(self.bridge.constraints)
        self.bridge.add_constraint(constraint)
        
        self.assertEqual(len(self.bridge.constraints), initial_count + 1)
        self.assertIn("bridge_test_1", self.bridge.constraints)
    
    def test_create_mining_task(self):
        """Test creating mining tasks"""
        # Add constraints first
        constraints = [
            SemanticConstraint(
                constraint_id=f"task_constraint_{i}",
                constraint_type=ConstraintType.STRUCTURAL,
                description=f"Constraint {i}",
                logical_formula=f"(C{i} ?x)"
            )
            for i in range(2)
        ]
        
        for c in constraints:
            self.bridge.add_constraint(c)
        
        task = self.bridge.create_mining_task(
            task_id="test_task",
            description="Test mining task",
            constraint_ids=["task_constraint_0", "task_constraint_1"],
            target_atoms=["atom1", "atom2", "atom3"]
        )
        
        self.assertEqual(task.task_id, "test_task")
        self.assertEqual(len(task.constraints), 2)
        self.assertEqual(len(task.target_atoms), 3)
        self.assertEqual(task.status, "pending")
    
    def test_discover_patterns(self):
        """Test pattern discovery"""
        task = MiningTask(
            task_id="discovery_test",
            description="Test discovery",
            constraints=[],
            target_atoms=["a", "b", "c"],
            min_support=0.1,
            min_confidence=0.5
        )
        
        patterns = self.bridge.discover_patterns(task)
        
        self.assertGreater(len(patterns), 0)
        self.assertEqual(task.status, "completed")
        
        for pattern in patterns:
            self.assertIsInstance(pattern, DiscoveredPattern)
            self.assertGreaterEqual(pattern.support, task.min_support)
            self.assertGreaterEqual(pattern.confidence, task.min_confidence)
    
    def test_validate_and_refine(self):
        """Test pattern validation and refinement"""
        patterns = [
            DiscoveredPattern(
                pattern_id=f"validate_test_{i}",
                structure=f"Pattern_{i}(X, Y)",
                support=0.3 + i * 0.1,
                confidence=0.6 + i * 0.1
            )
            for i in range(3)
        ]
        
        constraints = [
            SemanticConstraint(
                constraint_id="validate_constraint",
                constraint_type=ConstraintType.LOGICAL,
                description="Validation constraint",
                logical_formula="(Valid ?x)",
                confidence_threshold=0.6
            )
        ]
        
        validated = self.bridge.validate_and_refine(patterns, constraints)
        
        self.assertGreater(len(validated), 0)
        
        for pattern in validated:
            self.assertNotEqual(pattern.quality, PatternQuality.INVALID)
            self.assertGreater(pattern.validation_score, 0.0)
    
    def test_execute_task_complete(self):
        """Test complete task execution"""
        # Setup constraints
        constraints = [
            SemanticConstraint(
                constraint_id="exec_constraint_1",
                constraint_type=ConstraintType.STRUCTURAL,
                description="Structure constraint",
                logical_formula="(Struct ?x)",
                confidence_threshold=0.6
            ),
            SemanticConstraint(
                constraint_id="exec_constraint_2",
                constraint_type=ConstraintType.LOGICAL,
                description="Logic constraint",
                logical_formula="(Logic ?x)",
                confidence_threshold=0.7
            )
        ]
        
        for c in constraints:
            self.bridge.add_constraint(c)
        
        # Create task
        task = self.bridge.create_mining_task(
            task_id="complete_exec_test",
            description="Complete execution test",
            constraint_ids=["exec_constraint_1", "exec_constraint_2"],
            target_atoms=["atom_x", "atom_y", "atom_z"],
            max_patterns=8
        )
        
        # Execute
        results = self.bridge.execute_task("complete_exec_test")
        
        # Verify results structure
        self.assertIn('task_id', results)
        self.assertIn('discovered_count', results)
        self.assertIn('validated_count', results)
        self.assertIn('excellent_count', results)
        self.assertIn('good_count', results)
        self.assertIn('patterns', results)
        self.assertIn('execution_time', results)
        
        # Verify counts
        self.assertGreater(results['discovered_count'], 0)
        self.assertLessEqual(results['validated_count'], results['discovered_count'])
        self.assertGreater(results['execution_time'], 0)
        
        # Verify patterns are sorted by quality
        patterns = results['patterns']
        if len(patterns) > 1:
            for i in range(len(patterns) - 1):
                self.assertGreaterEqual(
                    patterns[i].quality.value,
                    patterns[i+1].quality.value
                )
    
    def test_statistics_tracking(self):
        """Test that statistics are tracked correctly"""
        initial_stats = self.bridge.get_statistics()
        
        # Add constraint
        constraint = SemanticConstraint(
            constraint_id="stats_constraint",
            constraint_type=ConstraintType.STRUCTURAL,
            description="Stats test",
            logical_formula="(Stats ?x)"
        )
        self.bridge.add_constraint(constraint)
        
        # Create and execute task
        task = self.bridge.create_mining_task(
            task_id="stats_task",
            description="Stats test task",
            constraint_ids=["stats_constraint"],
            target_atoms=["a", "b"]
        )
        
        self.bridge.execute_task("stats_task")
        
        # Check updated statistics
        final_stats = self.bridge.get_statistics()
        
        self.assertGreater(
            final_stats['total_tasks'],
            initial_stats['total_tasks']
        )
        self.assertGreater(
            final_stats['total_patterns_discovered'],
            initial_stats['total_patterns_discovered']
        )


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows"""
    
    def test_full_workflow(self):
        """Test complete inference-driven pattern discovery workflow"""
        # Create bridge
        bridge = create_inference_pattern_bridge(
            min_support=0.15,
            min_confidence=0.6,
            refinement_iterations=2
        )
        
        # Define multiple constraints
        constraints = [
            SemanticConstraint(
                constraint_id="workflow_struct",
                constraint_type=ConstraintType.STRUCTURAL,
                description="Structural requirement",
                logical_formula="(Structure ?x)",
                confidence_threshold=0.65,
                importance=1.0
            ),
            SemanticConstraint(
                constraint_id="workflow_logic",
                constraint_type=ConstraintType.LOGICAL,
                description="Logical requirement",
                logical_formula="(Logic ?x)",
                confidence_threshold=0.70,
                importance=1.5
            ),
            SemanticConstraint(
                constraint_id="workflow_stat",
                constraint_type=ConstraintType.STATISTICAL,
                description="Statistical requirement",
                logical_formula="(Stats ?x)",
                confidence_threshold=0.60,
                importance=0.8
            )
        ]
        
        for c in constraints:
            bridge.add_constraint(c)
        
        # Create mining task
        task = bridge.create_mining_task(
            task_id="full_workflow",
            description="Complete workflow test",
            constraint_ids=[c.constraint_id for c in constraints],
            target_atoms=["concept_A", "concept_B", "concept_C", "concept_D"],
            max_patterns=12
        )
        
        # Execute task
        results = bridge.execute_task("full_workflow")
        
        # Verify complete workflow
        self.assertGreater(results['discovered_count'], 0)
        self.assertGreater(len(results['patterns']), 0)
        
        # Check that patterns are validated
        for pattern in results['patterns']:
            self.assertGreater(pattern.validation_score, 0.0)
            self.assertGreater(len(pattern.satisfies_constraints), 0)
        
        # Check statistics
        stats = bridge.get_statistics()
        self.assertEqual(stats['total_tasks'], 1)
        self.assertGreater(stats['total_patterns_discovered'], 0)
        self.assertGreater(stats['total_patterns_validated'], 0)


def run_tests():
    """Run all tests"""
    print("="*70)
    print("Testing Inference-Driven Pattern Discovery Bridge")
    print("="*70)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSemanticConstraints))
    suite.addTests(loader.loadTestsFromTestCase(TestConstraintValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestPatternRefinement))
    suite.addTests(loader.loadTestsFromTestCase(TestInferencePatternBridge))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary:")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
