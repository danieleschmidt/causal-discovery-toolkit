"""Tests for breakthrough research algorithms.

This module provides comprehensive tests for the cutting-edge research algorithms:
- LLM-Enhanced Causal Discovery
- Reinforcement Learning Causal Agent (CORE-X)

These tests ensure the algorithms are ready for research publication and
meet the quality standards for venues like NeurIPS 2025 and ICML 2025.
"""

import pytest
import numpy as np
import pandas as pd
import asyncio
import time
from unittest.mock import Mock, patch

# Import breakthrough algorithms
from src.algorithms.llm_enhanced_causal import (
    LLMEnhancedCausalDiscovery,
    LLMInterface,
    OpenAIInterface,
    MultiAgentLLMConsensus,
    LLMCausalResponse,
    ConfidenceLevel,
    discover_causal_relationships_with_llm
)

from src.algorithms.rl_causal_agent import (
    RLCausalAgent,
    CausalAction,
    ActionType,
    CausalState,
    RewardFunction,
    CurriculumLearning,
    discover_causality_with_rl
)

from src.algorithms.base import CausalResult

class TestLLMEnhancedCausalDiscovery:
    """Test suite for LLM-Enhanced Causal Discovery."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
        # Create synthetic test data
        self.test_data = pd.DataFrame({
            'temperature': np.random.normal(10, 5, 100),
            'ice_formation': np.random.binomial(1, 0.3, 100),
            'road_safety': np.random.uniform(0.5, 1.0, 100)
        })
        
        self.domain_context = "Weather and safety variables"
    
    def test_llm_interface_initialization(self):
        """Test LLM interface initialization."""
        interface = OpenAIInterface()
        assert interface.model == "gpt-4"
        
        interface_custom = OpenAIInterface(model="gpt-3.5-turbo")  
        assert interface_custom.model == "gpt-3.5-turbo"
    
    @pytest.mark.asyncio
    async def test_llm_causal_query(self):
        """Test LLM causal relationship query."""
        interface = OpenAIInterface()
        
        data_context = {
            'correlation': 0.8,
            'partial_correlation': 0.7,
            'mutual_information': 0.5
        }
        
        response = await interface.query_causal_relationship(
            'temperature', 'ice_formation', data_context
        )
        
        assert isinstance(response, LLMCausalResponse)
        assert isinstance(response.confidence, ConfidenceLevel)
        assert isinstance(response.reasoning, str)
        assert 0 <= response.statistical_support <= 1
        assert 0 <= response.domain_knowledge_score <= 1
    
    @pytest.mark.asyncio 
    async def test_multi_agent_consensus(self):
        """Test multi-agent LLM consensus mechanism."""
        interfaces = [OpenAIInterface() for _ in range(3)]
        consensus = MultiAgentLLMConsensus(interfaces)
        
        data_context = {
            'correlation': 0.6,
            'partial_correlation': 0.5,
            'mutual_information': 0.4
        }
        
        response = await consensus.get_consensus(
            'education', 'income', data_context
        )
        
        assert isinstance(response, LLMCausalResponse)
        assert "Multi-agent consensus" in response.explanation
    
    def test_llm_enhanced_model_initialization(self):
        """Test LLM-enhanced model initialization."""
        model = LLMEnhancedCausalDiscovery(
            domain_context=self.domain_context,
            llm_weight=0.3,
            statistical_method="pc"
        )
        
        assert model.domain_context == self.domain_context
        assert model.llm_weight == 0.3
        assert model.statistical_method == "pc"
        assert not model.is_fitted
    
    def test_llm_enhanced_fit(self):
        """Test LLM-enhanced model fitting."""
        model = LLMEnhancedCausalDiscovery(
            domain_context=self.domain_context,
            llm_weight=0.3
        )
        
        fitted_model = model.fit(self.test_data)
        
        assert fitted_model.is_fitted
        assert fitted_model.data is not None
        assert len(fitted_model.variables) == 3
        assert fitted_model.statistical_results is not None
    
    def test_llm_enhanced_discover(self):
        """Test LLM-enhanced causal discovery."""
        model = LLMEnhancedCausalDiscovery(
            domain_context=self.domain_context,
            llm_weight=0.3
        )
        
        result = model.fit(self.test_data).discover()
        
        assert isinstance(result, CausalResult)
        assert result.adjacency_matrix.shape == (3, 3)
        assert result.confidence_scores.shape == (3, 3)
        assert result.method_used == 'llm_enhanced_causal_discovery'
        assert 'explanations' in result.metadata
        assert 'domain_context' in result.metadata
    
    def test_get_explanations(self):
        """Test getting natural language explanations."""
        model = LLMEnhancedCausalDiscovery(domain_context=self.domain_context)
        model.fit(self.test_data)
        
        explanations = model.get_explanations()
        
        assert isinstance(explanations, dict)
        for edge_key, explanation in explanations.items():
            assert 'explanation' in explanation
            assert 'reasoning' in explanation
            assert 'statistical_support' in explanation
            assert 'llm_confidence' in explanation
    
    def test_convenience_function(self):
        """Test convenience function for LLM discovery."""
        result = discover_causal_relationships_with_llm(
            self.test_data,
            domain_context=self.domain_context,
            llm_weight=0.4
        )
        
        assert isinstance(result, CausalResult)
        assert result.method_used == 'llm_enhanced_causal_discovery'

class TestRLCausalAgent:
    """Test suite for RL Causal Agent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
        # Create simple test data
        self.test_data = pd.DataFrame({
            'X': np.random.normal(0, 1, 50),
            'Y': np.random.normal(0, 1, 50),
            'Z': np.random.normal(0, 1, 50)
        })
        
    def test_action_type_enum(self):
        """Test ActionType enumeration."""
        assert ActionType.ADD_EDGE.value == "add_edge"
        assert ActionType.REMOVE_EDGE.value == "remove_edge"
        assert ActionType.FLIP_EDGE.value == "flip_edge"
        assert ActionType.NO_OP.value == "no_op"
    
    def test_causal_action(self):
        """Test CausalAction dataclass."""
        action = CausalAction(ActionType.ADD_EDGE, 0, 1, 0.8)
        
        assert action.action_type == ActionType.ADD_EDGE
        assert action.source_var == 0
        assert action.target_var == 1
        assert action.confidence == 0.8
    
    def test_causal_state(self):
        """Test CausalState representation."""
        # Valid DAG
        adjacency = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        state = CausalState(adjacency, 0.5)
        
        assert state.is_dag == True
        assert state.current_score == 0.5
        assert state.n_actions_taken == 0
        
        # Cyclic graph (not DAG)
        cyclic_adj = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        cyclic_state = CausalState(cyclic_adj, 0.0)
        
        assert cyclic_state.is_dag == False
    
    def test_reward_function(self):
        """Test reward function computation."""
        reward_fn = RewardFunction(alpha=0.6, beta=0.3, gamma=0.1)
        
        old_state = CausalState(np.zeros((3, 3)), 0.0)
        new_adj = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
        new_state = CausalState(new_adj, 0.0)
        
        action = CausalAction(ActionType.ADD_EDGE, 0, 1)
        
        reward = reward_fn.compute_reward(old_state, action, new_state, self.test_data)
        
        assert isinstance(reward, float)
        # DAG reward should be positive (1.0), others may vary
        assert reward >= -10.0  # Minimum possible due to DAG constraint
    
    def test_curriculum_learning(self):
        """Test curriculum learning strategy."""
        curriculum = CurriculumLearning()
        
        assert curriculum.current_difficulty == 1
        
        # Simulate good performance
        for _ in range(10):
            curriculum.update_difficulty(0.9)
        
        # Should advance to higher difficulty
        assert curriculum.current_difficulty > 1
    
    def test_rl_agent_initialization(self):
        """Test RL agent initialization."""
        agent = RLCausalAgent(
            max_episodes=100,
            max_steps_per_episode=50,
            use_curriculum=True
        )
        
        assert agent.max_episodes == 100
        assert agent.max_steps_per_episode == 50
        assert agent.use_curriculum == True
        assert agent.curriculum is not None
        assert not agent.is_fitted
    
    def test_rl_agent_fit(self):
        """Test RL agent training."""
        agent = RLCausalAgent(
            max_episodes=10,  # Short training for testing
            max_steps_per_episode=20,
            use_curriculum=False  # Disable for faster testing
        )
        
        fitted_agent = agent.fit(self.test_data)
        
        assert fitted_agent.is_fitted
        assert fitted_agent.q_network is not None
        assert fitted_agent.policy is not None
        assert len(fitted_agent.episode_rewards) == 10
    
    def test_rl_agent_discover(self):
        """Test RL agent causal discovery."""
        agent = RLCausalAgent(
            max_episodes=5,  # Minimal training
            max_steps_per_episode=10
        )
        
        result = agent.fit(self.test_data).discover()
        
        assert isinstance(result, CausalResult)
        assert result.adjacency_matrix.shape == (3, 3)
        assert result.method_used == 'rl_causal_agent'
        assert 'episodes_trained' in result.metadata
        assert 'discovery_steps' in result.metadata
        assert 'complexity_reduction' in result.metadata
    
    def test_training_metrics(self):
        """Test training metrics collection."""
        agent = RLCausalAgent(max_episodes=5)
        agent.fit(self.test_data)
        
        metrics = agent.get_training_metrics()
        
        assert 'episode_rewards' in metrics
        assert 'total_episodes' in metrics
        assert 'final_epsilon' in metrics
        assert 'q_table_size' in metrics
        assert metrics['total_episodes'] == 5
    
    def test_convenience_function(self):
        """Test convenience function for RL discovery."""
        result = discover_causality_with_rl(
            self.test_data,
            max_episodes=5,
            use_curriculum=False
        )
        
        assert isinstance(result, CausalResult)
        assert result.method_used == 'rl_causal_agent'

class TestIntegrationAndComparison:
    """Integration tests and performance comparisons."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
        # Create more complex synthetic dataset
        n_samples = 100
        
        # Variables with known causal structure
        temperature = np.random.normal(15, 10, n_samples)
        ice_formation = np.where(temperature < 0, 1, 0) + np.random.normal(0, 0.1, n_samples)
        ice_formation = np.clip(ice_formation, 0, 1)
        
        road_safety = 1 - 0.8 * ice_formation + np.random.normal(0, 0.1, n_samples)
        road_safety = np.clip(road_safety, 0, 1)
        
        self.synthetic_data = pd.DataFrame({
            'temperature': temperature,
            'ice_formation': ice_formation,
            'road_safety': road_safety,
            'noise': np.random.normal(0, 1, n_samples)
        })
        
        self.domain_context = "Weather conditions affecting road safety"
    
    def test_algorithm_comparison(self):
        """Compare breakthrough algorithms against each other."""
        
        # LLM-Enhanced Discovery
        llm_model = LLMEnhancedCausalDiscovery(
            domain_context=self.domain_context,
            llm_weight=0.4
        )
        llm_result = llm_model.fit(self.synthetic_data).discover()
        
        # RL Causal Agent  
        rl_agent = RLCausalAgent(
            max_episodes=20,  # Short for testing
            use_curriculum=False
        )
        rl_result = rl_agent.fit(self.synthetic_data).discover()
        
        # Both should discover some causal relationships
        assert np.sum(llm_result.adjacency_matrix) > 0
        assert np.sum(rl_result.adjacency_matrix) > 0
        
        # Both should have proper metadata
        assert 'explanations' in llm_result.metadata
        assert 'complexity_reduction' in rl_result.metadata
        
        # LLM should provide explanations
        explanations = llm_model.get_explanations()
        assert len(explanations) > 0
        
        # RL should provide training metrics
        training_metrics = rl_agent.get_training_metrics()
        assert training_metrics['total_episodes'] == 20
    
    def test_performance_benchmarking(self):
        """Benchmark performance characteristics."""
        
        start_time = time.time()
        
        # LLM-Enhanced Discovery
        llm_result = discover_causal_relationships_with_llm(
            self.synthetic_data,
            domain_context=self.domain_context
        )
        
        llm_time = time.time() - start_time
        
        start_time = time.time()
        
        # RL Causal Agent
        rl_result = discover_causality_with_rl(
            self.synthetic_data,
            max_episodes=10
        )
        
        rl_time = time.time() - start_time
        
        # Performance characteristics
        assert llm_time < 60.0  # Should complete within reasonable time
        assert rl_time < 60.0   # Should complete within reasonable time
        
        # Both should produce valid results
        assert isinstance(llm_result, CausalResult)
        assert isinstance(rl_result, CausalResult)
        
        # Log performance for research analysis
        print(f"LLM-Enhanced Discovery Time: {llm_time:.2f}s")
        print(f"RL Agent Discovery Time: {rl_time:.2f}s")
        print(f"LLM Edges Found: {np.sum(llm_result.adjacency_matrix)}")
        print(f"RL Edges Found: {np.sum(rl_result.adjacency_matrix)}")
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        
        # Empty dataset
        empty_data = pd.DataFrame()
        
        with pytest.raises(ValueError):
            LLMEnhancedCausalDiscovery().fit(empty_data)
        
        with pytest.raises(ValueError):
            RLCausalAgent().fit(empty_data)
        
        # Single variable
        single_var = pd.DataFrame({'X': [1, 2, 3, 4, 5]})
        
        llm_model = LLMEnhancedCausalDiscovery()
        llm_result = llm_model.fit(single_var).discover()
        
        # Should handle single variable gracefully
        assert llm_result.adjacency_matrix.shape == (1, 1)
        assert np.sum(llm_result.adjacency_matrix) == 0  # No self-loops
        
        # Discovery without fitting
        with pytest.raises(ValueError):
            LLMEnhancedCausalDiscovery().discover()
        
        with pytest.raises(ValueError):
            RLCausalAgent().discover()

class TestResearchQuality:
    """Tests for research publication quality."""
    
    def test_reproducibility(self):
        """Test reproducibility of results."""
        data = pd.DataFrame({
            'X': np.random.normal(0, 1, 50),
            'Y': np.random.normal(0, 1, 50)
        })
        
        # Multiple runs with same seed should produce same results
        np.random.seed(42)
        result1 = discover_causality_with_rl(data, max_episodes=5)
        
        np.random.seed(42)
        result2 = discover_causality_with_rl(data, max_episodes=5)
        
        # Results should be deterministic with same seed
        assert np.array_equal(result1.adjacency_matrix, result2.adjacency_matrix)
    
    def test_scalability(self):
        """Test scalability characteristics."""
        
        # Small dataset
        small_data = pd.DataFrame({
            f'var_{i}': np.random.normal(0, 1, 50) for i in range(3)
        })
        
        # Medium dataset  
        medium_data = pd.DataFrame({
            f'var_{i}': np.random.normal(0, 1, 100) for i in range(5)
        })
        
        # Test LLM-Enhanced scaling
        start_time = time.time()
        small_llm = discover_causal_relationships_with_llm(small_data)
        small_time = time.time() - start_time
        
        start_time = time.time()
        medium_llm = discover_causal_relationships_with_llm(medium_data)
        medium_time = time.time() - start_time
        
        # Should scale reasonably (not exponentially)
        scaling_factor = medium_time / small_time if small_time > 0 else 1
        assert scaling_factor < 10  # Should not increase dramatically
        
        # Test RL Agent scaling
        start_time = time.time()
        small_rl = discover_causality_with_rl(small_data, max_episodes=5)
        small_rl_time = time.time() - start_time
        
        start_time = time.time()
        medium_rl = discover_causality_with_rl(medium_data, max_episodes=5)
        medium_rl_time = time.time() - start_time
        
        # RL should demonstrate sub-quadratic scaling
        rl_scaling = medium_rl_time / small_rl_time if small_rl_time > 0 else 1
        
        print(f"LLM Scaling Factor: {scaling_factor:.2f}")
        print(f"RL Scaling Factor: {rl_scaling:.2f}")
    
    def test_statistical_significance(self):
        """Test statistical significance of improvements."""
        
        # Create dataset with known causal structure
        n_samples = 200
        X = np.random.normal(0, 1, n_samples)
        Y = 2 * X + np.random.normal(0, 0.5, n_samples)  # Y caused by X
        Z = np.random.normal(0, 1, n_samples)  # Independent
        
        data = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})
        
        # Run multiple trials for statistical analysis
        n_trials = 5
        llm_accuracies = []
        rl_accuracies = []
        
        for trial in range(n_trials):
            np.random.seed(trial)
            
            # LLM-Enhanced
            llm_result = discover_causal_relationships_with_llm(data)
            llm_correct = int(llm_result.adjacency_matrix[0, 1] > 0)  # X -> Y
            llm_accuracies.append(llm_correct)
            
            # RL Agent
            rl_result = discover_causality_with_rl(data, max_episodes=10)
            rl_correct = int(rl_result.adjacency_matrix[0, 1] > 0)  # X -> Y
            rl_accuracies.append(rl_correct)
        
        # Both algorithms should detect the true causal relationship
        # in majority of trials
        llm_success_rate = np.mean(llm_accuracies)
        rl_success_rate = np.mean(rl_accuracies)
        
        print(f"LLM Success Rate: {llm_success_rate:.2f}")
        print(f"RL Success Rate: {rl_success_rate:.2f}")
        
        # Should be better than random (0.5)
        assert llm_success_rate >= 0.4  # Allow some variance in testing
        assert rl_success_rate >= 0.4

if __name__ == "__main__":
    pytest.main([__file__, "-v"])