"""
Causal Memory Bank for Autonomous Agents
========================================

Advanced memory system for causal agents with episodic memory, causal indexing,
and experience replay with causal reasoning.

Novel Contributions:
- Causal episodic memory with graph indexing
- Experience replay prioritized by causal importance
- Memory consolidation with causal pattern extraction
- Cross-temporal causal reasoning
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import heapq
import json
import pickle
import time
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
import networkx as nx

try:
    from .causal_reasoning_agent import CausalObservation, CausalAction, CausalResult
except ImportError:
    from causal_reasoning_agent import CausalObservation, CausalAction, CausalResult


@dataclass
class CausalEpisode:
    """Represents a causal episode in memory."""
    episode_id: str
    observations: List[CausalObservation]
    actions: List[CausalAction] 
    rewards: List[float]
    causal_structures: List[np.ndarray]
    reasoning_traces: List[List[str]]
    
    # Episode metadata
    start_time: float
    end_time: float
    total_reward: float
    episode_length: int
    
    # Causal analysis
    causal_importance: float = 0.0
    learned_relationships: Dict[str, Any] = field(default_factory=dict)
    surprise_events: List[int] = field(default_factory=list)  # Timesteps with high surprise
    
    # Indexing
    causal_signature: str = ""  # Hash of causal patterns
    variable_signatures: Dict[str, str] = field(default_factory=dict)


@dataclass 
class CausalMemoryConfig:
    """Configuration for causal memory system."""
    # Memory capacity
    max_episodes: int = 1000
    max_experiences_per_episode: int = 200
    max_total_experiences: int = 10000
    
    # Prioritization
    importance_decay: float = 0.95
    surprise_threshold: float = 0.3
    causal_importance_weight: float = 0.4
    recency_weight: float = 0.3
    surprise_weight: float = 0.3
    
    # Consolidation
    consolidation_frequency: int = 100  # Episodes between consolidation
    min_pattern_support: int = 3  # Minimum occurrences for pattern
    pattern_significance_threshold: float = 0.05
    
    # Retrieval
    retrieval_k: int = 10  # Number of episodes to retrieve
    similarity_threshold: float = 0.5
    enable_causal_reasoning: bool = True


class CausalPatternExtractor:
    """Extracts causal patterns from episodic memory."""
    
    def __init__(self, config: CausalMemoryConfig):
        self.config = config
        self.discovered_patterns = {}
        
    def extract_patterns(self, episodes: List[CausalEpisode]) -> Dict[str, Any]:
        """Extract recurring causal patterns from episodes."""
        patterns = {
            "intervention_patterns": self._extract_intervention_patterns(episodes),
            "causal_chains": self._extract_causal_chains(episodes), 
            "context_patterns": self._extract_context_patterns(episodes),
            "outcome_patterns": self._extract_outcome_patterns(episodes)
        }
        
        # Update discovered patterns
        self.discovered_patterns.update(patterns)
        
        return patterns
    
    def _extract_intervention_patterns(self, episodes: List[CausalEpisode]) -> Dict[str, Any]:
        """Extract patterns of successful interventions."""
        intervention_outcomes = defaultdict(list)
        
        for episode in episodes:
            for i, action in enumerate(episode.actions):
                if (action.intervention_type == "do" and 
                    i < len(episode.rewards) and
                    len(action.target_variables) > 0):
                    
                    # Create intervention signature
                    signature = f"{action.target_variables[0]}_{action.intervention_type}"
                    reward = episode.rewards[i]
                    
                    intervention_outcomes[signature].append({
                        "reward": reward,
                        "magnitude": action.intervention_values[0] if action.intervention_values.size > 0 else 0.0,
                        "confidence": action.confidence,
                        "episode_id": episode.episode_id
                    })
        
        # Analyze patterns
        patterns = {}
        for signature, outcomes in intervention_outcomes.items():
            if len(outcomes) >= self.config.min_pattern_support:
                rewards = [o["reward"] for o in outcomes]
                magnitudes = [o["magnitude"] for o in outcomes]
                
                patterns[signature] = {
                    "count": len(outcomes),
                    "mean_reward": np.mean(rewards),
                    "reward_std": np.std(rewards),
                    "optimal_magnitude": magnitudes[np.argmax(rewards)],
                    "success_rate": sum(1 for r in rewards if r > 0) / len(rewards)
                }
        
        return patterns
    
    def _extract_causal_chains(self, episodes: List[CausalEpisode]) -> Dict[str, Any]:
        """Extract common causal chains A -> B -> C."""
        chains = defaultdict(int)
        
        for episode in episodes:
            for structure in episode.causal_structures:
                # Find paths of length 2 and 3
                for length in [2, 3]:
                    paths = self._find_causal_paths(structure, length)
                    for path in paths:
                        chain_signature = " -> ".join([f"var_{i}" for i in path])
                        chains[chain_signature] += 1
        
        # Filter by minimum support
        significant_chains = {
            chain: count for chain, count in chains.items()
            if count >= self.config.min_pattern_support
        }
        
        return significant_chains
    
    def _find_causal_paths(self, structure: np.ndarray, length: int) -> List[List[int]]:
        """Find causal paths of specified length."""
        paths = []
        n_vars = structure.shape[0]
        
        def dfs(current_path, current_var):
            if len(current_path) == length:
                paths.append(current_path.copy())
                return
            
            # Find variables that current_var affects
            for next_var in range(n_vars):
                if (structure[current_var, next_var] > 0.3 and 
                    next_var not in current_path):
                    current_path.append(next_var)
                    dfs(current_path, next_var)
                    current_path.pop()
        
        # Start DFS from each variable
        for start_var in range(n_vars):
            dfs([start_var], start_var)
        
        return paths
    
    def _extract_context_patterns(self, episodes: List[CausalEpisode]) -> Dict[str, Any]:
        """Extract patterns related to context and conditions."""
        context_patterns = defaultdict(list)
        
        for episode in episodes:
            for obs in episode.observations:
                if obs.context_data:
                    # Group episodes by context features
                    for key, value in obs.context_data.items():
                        context_patterns[key].append({
                            "value": value,
                            "total_reward": episode.total_reward,
                            "episode_length": episode.episode_length
                        })
        
        # Analyze context impact
        analyzed_patterns = {}
        for context_key, data in context_patterns.items():
            if len(data) >= self.config.min_pattern_support:
                rewards = [d["total_reward"] for d in data]
                analyzed_patterns[context_key] = {
                    "count": len(data),
                    "mean_reward": np.mean(rewards),
                    "reward_correlation": np.corrcoef([d["value"] for d in data], rewards)[0, 1] if len(data) > 1 else 0.0
                }
        
        return analyzed_patterns
    
    def _extract_outcome_patterns(self, episodes: List[CausalEpisode]) -> Dict[str, Any]:
        """Extract patterns in outcomes and consequences."""
        outcome_patterns = {}
        
        # Analyze reward patterns
        rewards = [ep.total_reward for ep in episodes]
        lengths = [ep.episode_length for ep in episodes]
        
        outcome_patterns["reward_statistics"] = {
            "mean": np.mean(rewards),
            "std": np.std(rewards),
            "median": np.median(rewards),
            "success_rate": sum(1 for r in rewards if r > 0) / len(rewards)
        }
        
        outcome_patterns["length_statistics"] = {
            "mean": np.mean(lengths),
            "std": np.std(lengths),
            "median": np.median(lengths)
        }
        
        # Correlation between length and reward
        if len(rewards) > 1:
            outcome_patterns["length_reward_correlation"] = np.corrcoef(lengths, rewards)[0, 1]
        
        return outcome_patterns


class CausalIndexer:
    """Indexes episodes by causal signatures for fast retrieval."""
    
    def __init__(self, config: CausalMemoryConfig):
        self.config = config
        self.causal_index = defaultdict(list)  # signature -> episode_ids
        self.variable_index = defaultdict(list)  # variable -> episode_ids
        self.temporal_index = defaultdict(list)  # time_bucket -> episode_ids
        
    def index_episode(self, episode: CausalEpisode):
        """Add episode to indices."""
        # Causal signature indexing
        episode.causal_signature = self._compute_causal_signature(episode)
        self.causal_index[episode.causal_signature].append(episode.episode_id)
        
        # Variable signature indexing
        for var_name in self._extract_variables(episode):
            var_signature = self._compute_variable_signature(episode, var_name)
            episode.variable_signatures[var_name] = var_signature
            self.variable_index[var_signature].append(episode.episode_id)
        
        # Temporal indexing
        time_bucket = int(episode.start_time // 3600)  # Hour buckets
        self.temporal_index[time_bucket].append(episode.episode_id)
    
    def _compute_causal_signature(self, episode: CausalEpisode) -> str:
        """Compute signature representing causal patterns in episode."""
        if not episode.causal_structures:
            return "empty"
        
        # Average causal structure
        avg_structure = np.mean(episode.causal_structures, axis=0)
        
        # Create signature from structure characteristics
        n_edges = np.sum(avg_structure > 0.3)
        density = np.mean(avg_structure)
        max_strength = np.max(avg_structure)
        
        # Intervention signature
        intervention_types = [a.intervention_type for a in episode.actions]
        intervention_signature = "_".join(sorted(set(intervention_types)))
        
        signature = f"edges_{n_edges}_density_{density:.2f}_max_{max_strength:.2f}_int_{intervention_signature}"
        return signature
    
    def _extract_variables(self, episode: CausalEpisode) -> List[str]:
        """Extract variable names involved in episode."""
        variables = set()
        for action in episode.actions:
            variables.update(action.target_variables)
        
        # Add variables from observations if available
        for obs in episode.observations:
            if obs.metadata and "variable_names" in obs.metadata:
                variables.update(obs.metadata["variable_names"])
        
        return list(variables)
    
    def _compute_variable_signature(self, episode: CausalEpisode, var_name: str) -> str:
        """Compute signature for specific variable across episode."""
        # Count interventions on this variable
        intervention_count = sum(1 for action in episode.actions 
                               if var_name in action.target_variables)
        
        # Compute average intervention magnitude
        magnitudes = []
        for action in episode.actions:
            if var_name in action.target_variables and action.intervention_values.size > 0:
                var_idx = action.target_variables.index(var_name)
                if var_idx < len(action.intervention_values):
                    magnitudes.append(abs(action.intervention_values[var_idx]))
        
        avg_magnitude = np.mean(magnitudes) if magnitudes else 0.0
        
        signature = f"{var_name}_int_{intervention_count}_mag_{avg_magnitude:.2f}"
        return signature
    
    def find_similar_episodes(self, query_signature: str, k: int = 10) -> List[str]:
        """Find episodes with similar causal signatures."""
        # Exact match first
        if query_signature in self.causal_index:
            return self.causal_index[query_signature][:k]
        
        # Fuzzy matching based on signature components
        best_matches = []
        for signature, episode_ids in self.causal_index.items():
            similarity = self._compute_signature_similarity(query_signature, signature)
            if similarity > self.config.similarity_threshold:
                for episode_id in episode_ids:
                    heapq.heappush(best_matches, (-similarity, episode_id))
        
        # Return top k matches
        return [episode_id for _, episode_id in heapq.nsmallest(k, best_matches)][:k]
    
    def _compute_signature_similarity(self, sig1: str, sig2: str) -> float:
        """Compute similarity between causal signatures."""
        # Simple token-based similarity
        tokens1 = set(sig1.split("_"))
        tokens2 = set(sig2.split("_"))
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0


class ExperienceReplay:
    """Prioritized experience replay for causal learning."""
    
    def __init__(self, config: CausalMemoryConfig):
        self.config = config
        self.replay_buffer = []  # (priority, episode_id)
        
    def add_episode(self, episode: CausalEpisode):
        """Add episode to replay buffer with computed priority."""
        priority = self._compute_priority(episode)
        heapq.heappush(self.replay_buffer, (-priority, episode.episode_id))
        
        # Maintain buffer size
        if len(self.replay_buffer) > self.config.max_episodes:
            heapq.heappop(self.replay_buffer)
    
    def _compute_priority(self, episode: CausalEpisode) -> float:
        """Compute replay priority for episode."""
        # Combine multiple factors
        causal_importance = episode.causal_importance
        recency = self._compute_recency_score(episode)
        surprise = self._compute_surprise_score(episode)
        
        priority = (self.config.causal_importance_weight * causal_importance +
                   self.config.recency_weight * recency +
                   self.config.surprise_weight * surprise)
        
        return priority
    
    def _compute_recency_score(self, episode: CausalEpisode) -> float:
        """Compute recency score (higher for more recent episodes)."""
        current_time = time.time()
        time_diff = current_time - episode.end_time
        
        # Exponential decay
        return np.exp(-time_diff / (24 * 3600))  # 24 hour half-life
    
    def _compute_surprise_score(self, episode: CausalEpisode) -> float:
        """Compute surprise score based on unexpected events."""
        if not episode.surprise_events:
            return 0.0
        
        # Higher score for more surprise events
        return min(1.0, len(episode.surprise_events) / episode.episode_length)
    
    def sample_episodes(self, k: int) -> List[str]:
        """Sample k episodes for replay."""
        if len(self.replay_buffer) < k:
            return [episode_id for _, episode_id in self.replay_buffer]
        
        # Sample top k episodes
        return [episode_id for _, episode_id in heapq.nsmallest(k, self.replay_buffer)]


class CausalMemoryBank:
    """Advanced memory system for causal agents."""
    
    def __init__(self, config: CausalMemoryConfig):
        self.config = config
        self.episodes = {}  # episode_id -> CausalEpisode
        self.current_episode = None
        self.episode_counter = 0
        
        # Components
        self.pattern_extractor = CausalPatternExtractor(config)
        self.indexer = CausalIndexer(config)
        self.replay_system = ExperienceReplay(config)
        
        # Consolidated knowledge
        self.consolidated_patterns = {}
        self.causal_knowledge_graph = nx.DiGraph()
        
        # Statistics
        self.consolidation_count = 0
        self.total_experiences = 0
        
        # Initialize logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def start_episode(self) -> str:
        """Start a new episode."""
        episode_id = f"episode_{self.episode_counter}_{int(time.time())}"
        
        self.current_episode = CausalEpisode(
            episode_id=episode_id,
            observations=[],
            actions=[],
            rewards=[],
            causal_structures=[],
            reasoning_traces=[],
            start_time=time.time(),
            end_time=0.0,
            total_reward=0.0,
            episode_length=0
        )
        
        self.episode_counter += 1
        self.logger.debug(f"Started episode: {episode_id}")
        
        return episode_id
    
    def add_experience(self, observation: CausalObservation, action: CausalAction,
                      reward: float, causal_structure: np.ndarray,
                      reasoning_trace: List[str]):
        """Add experience to current episode."""
        if self.current_episode is None:
            self.start_episode()
        
        # Add to current episode
        self.current_episode.observations.append(observation)
        self.current_episode.actions.append(action)
        self.current_episode.rewards.append(reward)
        self.current_episode.causal_structures.append(causal_structure)
        self.current_episode.reasoning_traces.append(reasoning_trace)
        
        # Update episode metadata
        self.current_episode.total_reward += reward
        self.current_episode.episode_length += 1
        
        # Check for surprise events
        surprise_level = self._compute_surprise_level(observation, action, reward)
        if surprise_level > self.config.surprise_threshold:
            self.current_episode.surprise_events.append(self.current_episode.episode_length - 1)
        
        self.total_experiences += 1
        
        # Auto-end episode if it gets too long
        if self.current_episode.episode_length >= self.config.max_experiences_per_episode:
            self.end_episode()
    
    def _compute_surprise_level(self, observation: CausalObservation,
                               action: CausalAction, reward: float) -> float:
        """Compute surprise level for experience."""
        # Simple heuristic: extreme rewards are surprising
        if not hasattr(self, '_recent_rewards'):
            self._recent_rewards = []
        
        self._recent_rewards.append(reward)
        if len(self._recent_rewards) > 20:
            self._recent_rewards.pop(0)
        
        if len(self._recent_rewards) < 3:
            return 0.0
        
        mean_reward = np.mean(self._recent_rewards[:-1])
        std_reward = np.std(self._recent_rewards[:-1]) + 1e-6
        
        # Z-score based surprise
        z_score = abs(reward - mean_reward) / std_reward
        return min(1.0, z_score / 3.0)  # Normalize to [0, 1]
    
    def end_episode(self):
        """End current episode and store in memory."""
        if self.current_episode is None:
            return
        
        # Finalize episode
        self.current_episode.end_time = time.time()
        
        # Compute causal importance
        self.current_episode.causal_importance = self._compute_causal_importance(self.current_episode)
        
        # Store episode
        self.episodes[self.current_episode.episode_id] = self.current_episode
        
        # Index episode
        self.indexer.index_episode(self.current_episode)
        
        # Add to replay buffer
        self.replay_system.add_episode(self.current_episode)
        
        # Memory management
        self._manage_memory()
        
        # Periodic consolidation
        if len(self.episodes) % self.config.consolidation_frequency == 0:
            self.consolidate_memory()
        
        self.logger.info(f"Ended episode: {self.current_episode.episode_id} "
                        f"(length: {self.current_episode.episode_length}, "
                        f"reward: {self.current_episode.total_reward:.3f})")
        
        self.current_episode = None
    
    def _compute_causal_importance(self, episode: CausalEpisode) -> float:
        """Compute causal importance score for episode."""
        importance_factors = []
        
        # Factor 1: Number of causal interventions
        intervention_count = sum(1 for action in episode.actions 
                               if action.intervention_type == "do")
        intervention_score = min(1.0, intervention_count / 10.0)
        importance_factors.append(intervention_score)
        
        # Factor 2: Strength of causal relationships discovered
        if episode.causal_structures:
            avg_structure = np.mean(episode.causal_structures, axis=0)
            structure_strength = np.mean(avg_structure)
            importance_factors.append(structure_strength)
        
        # Factor 3: Surprise events
        surprise_score = len(episode.surprise_events) / max(1, episode.episode_length)
        importance_factors.append(surprise_score)
        
        # Factor 4: Reward magnitude
        reward_score = min(1.0, abs(episode.total_reward) / 10.0)
        importance_factors.append(reward_score)
        
        return np.mean(importance_factors)
    
    def _manage_memory(self):
        """Manage memory capacity by removing old episodes."""
        if len(self.episodes) > self.config.max_episodes:
            # Remove least important episodes
            episode_importance = [(ep.causal_importance, ep_id) 
                                for ep_id, ep in self.episodes.items()]
            episode_importance.sort()
            
            # Remove bottom 10%
            num_to_remove = len(self.episodes) // 10
            for _, ep_id in episode_importance[:num_to_remove]:
                del self.episodes[ep_id]
                
            self.logger.info(f"Removed {num_to_remove} episodes to manage memory")
    
    def consolidate_memory(self):
        """Consolidate memory by extracting patterns and updating knowledge."""
        self.logger.info("Starting memory consolidation...")
        
        # Extract patterns from recent episodes
        recent_episodes = list(self.episodes.values())[-self.config.consolidation_frequency:]
        new_patterns = self.pattern_extractor.extract_patterns(recent_episodes)
        
        # Update consolidated patterns
        self.consolidated_patterns.update(new_patterns)
        
        # Update causal knowledge graph
        self._update_knowledge_graph(new_patterns)
        
        self.consolidation_count += 1
        self.logger.info(f"Memory consolidation #{self.consolidation_count} completed")
    
    def _update_knowledge_graph(self, patterns: Dict[str, Any]):
        """Update causal knowledge graph with new patterns."""
        # Add causal chains to graph
        if "causal_chains" in patterns:
            for chain, count in patterns["causal_chains"].items():
                variables = chain.split(" -> ")
                for i in range(len(variables) - 1):
                    cause = variables[i]
                    effect = variables[i + 1]
                    
                    if self.causal_knowledge_graph.has_edge(cause, effect):
                        # Update edge weight
                        current_weight = self.causal_knowledge_graph[cause][effect]["weight"]
                        self.causal_knowledge_graph[cause][effect]["weight"] = current_weight + count
                    else:
                        # Add new edge
                        self.causal_knowledge_graph.add_edge(cause, effect, weight=count)
    
    def retrieve_similar_experiences(self, query_observation: CausalObservation,
                                   query_action: CausalAction,
                                   k: int = None) -> List[CausalEpisode]:
        """Retrieve similar experiences from memory."""
        if k is None:
            k = self.config.retrieval_k
        
        # Create query signature
        query_episode = CausalEpisode(
            episode_id="query",
            observations=[query_observation],
            actions=[query_action],
            rewards=[0.0],
            causal_structures=[np.eye(5)],  # Dummy structure
            reasoning_traces=[[]],
            start_time=time.time(),
            end_time=time.time(),
            total_reward=0.0,
            episode_length=1
        )
        
        query_signature = self.indexer._compute_causal_signature(query_episode)
        
        # Find similar episodes
        similar_episode_ids = self.indexer.find_similar_episodes(query_signature, k)
        
        # Return episode objects
        similar_episodes = [self.episodes[ep_id] for ep_id in similar_episode_ids 
                          if ep_id in self.episodes]
        
        return similar_episodes
    
    def get_causal_knowledge(self, cause: str, effect: str) -> Dict[str, Any]:
        """Get consolidated causal knowledge about relationship."""
        knowledge = {
            "relationship_exists": False,
            "strength": 0.0,
            "confidence": 0.0,
            "evidence_count": 0,
            "patterns": []
        }
        
        # Check knowledge graph
        if self.causal_knowledge_graph.has_edge(cause, effect):
            edge_data = self.causal_knowledge_graph[cause][effect]
            knowledge["relationship_exists"] = True
            knowledge["strength"] = min(1.0, edge_data["weight"] / 10.0)
            knowledge["evidence_count"] = edge_data["weight"]
        
        # Check consolidated patterns
        for pattern_type, patterns in self.consolidated_patterns.items():
            if isinstance(patterns, dict):
                for pattern_name, pattern_data in patterns.items():
                    if cause in pattern_name and effect in pattern_name:
                        knowledge["patterns"].append({
                            "type": pattern_type,
                            "pattern": pattern_name,
                            "data": pattern_data
                        })
        
        # Compute overall confidence
        if knowledge["evidence_count"] > 0:
            knowledge["confidence"] = min(1.0, knowledge["evidence_count"] / 5.0)
        
        return knowledge
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        return {
            "total_episodes": len(self.episodes),
            "total_experiences": self.total_experiences,
            "consolidation_count": self.consolidation_count,
            "knowledge_graph_nodes": self.causal_knowledge_graph.number_of_nodes(),
            "knowledge_graph_edges": self.causal_knowledge_graph.number_of_edges(),
            "consolidated_patterns": {
                pattern_type: len(patterns) if isinstance(patterns, dict) else 0
                for pattern_type, patterns in self.consolidated_patterns.items()
            },
            "current_episode_length": self.current_episode.episode_length if self.current_episode else 0,
            "replay_buffer_size": len(self.replay_system.replay_buffer)
        }
    
    def save_memory(self, filepath: str):
        """Save memory bank to file."""
        memory_data = {
            "config": self.config.__dict__,
            "episodes": {ep_id: self._serialize_episode(ep) for ep_id, ep in self.episodes.items()},
            "consolidated_patterns": self.consolidated_patterns,
            "knowledge_graph": dict(self.causal_knowledge_graph.edges(data=True)),
            "statistics": self.get_memory_statistics()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(memory_data, f)
        
        self.logger.info(f"Memory bank saved to {filepath}")
    
    def _serialize_episode(self, episode: CausalEpisode) -> Dict[str, Any]:
        """Serialize episode to dictionary."""
        return {
            "episode_id": episode.episode_id,
            "start_time": episode.start_time,
            "end_time": episode.end_time,
            "total_reward": episode.total_reward,
            "episode_length": episode.episode_length,
            "causal_importance": episode.causal_importance,
            "causal_signature": episode.causal_signature,
            # Note: Observations and actions would need custom serialization for full save
            "summary": {
                "n_observations": len(episode.observations),
                "n_actions": len(episode.actions),
                "n_surprises": len(episode.surprise_events)
            }
        }
    
    def load_memory(self, filepath: str):
        """Load memory bank from file."""
        with open(filepath, 'rb') as f:
            memory_data = pickle.load(f)
        
        self.consolidated_patterns = memory_data.get("consolidated_patterns", {})
        
        # Rebuild knowledge graph
        self.causal_knowledge_graph = nx.DiGraph()
        for (cause, effect), edge_data in memory_data.get("knowledge_graph", {}).items():
            self.causal_knowledge_graph.add_edge(cause, effect, **edge_data)
        
        self.logger.info(f"Memory bank loaded from {filepath}")


def demo_causal_memory_bank():
    """Demonstrate causal memory bank capabilities."""
    print("🧠 Causal Memory Bank Demo")
    print("=" * 50)
    
    # Configuration
    config = CausalMemoryConfig(
        max_episodes=100,
        consolidation_frequency=10,
        enable_causal_reasoning=True
    )
    
    # Initialize memory bank
    memory_bank = CausalMemoryBank(config)
    
    # Simulate multiple episodes
    for episode_num in range(15):
        episode_id = memory_bank.start_episode()
        print(f"\n📖 Episode {episode_num + 1}: {episode_id}")
        
        # Simulate episode with 5-10 experiences
        episode_length = np.random.randint(5, 11)
        
        for step in range(episode_length):
            # Create mock observation
            observation = CausalObservation(
                state_data=np.random.randn(5),
                timestamp=time.time() + step,
                metadata={"step": step, "variable_names": [f"var_{i}" for i in range(5)]}
            )
            
            # Create mock action
            if np.random.random() < 0.4:  # 40% intervention rate
                action = CausalAction(
                    target_variables=[f"var_{np.random.randint(0, 3)}"],
                    intervention_values=np.array([np.random.randn() * 0.5]),
                    intervention_type="do",
                    confidence=np.random.uniform(0.6, 0.9),
                    reasoning_chain=[f"Step {step} intervention"]
                )
            else:
                action = CausalAction(
                    target_variables=[],
                    intervention_values=np.array([]),
                    intervention_type="observe",
                    confidence=1.0,
                    reasoning_chain=[f"Step {step} observation"]
                )
            
            # Mock reward and causal structure
            reward = np.random.randn() * 2.0
            causal_structure = np.random.random((5, 5)) * 0.3
            np.fill_diagonal(causal_structure, 0)
            
            # Add experience
            memory_bank.add_experience(
                observation=observation,
                action=action,
                reward=reward,
                causal_structure=causal_structure,
                reasoning_trace=[f"Reasoning for step {step}"]
            )
        
        # End episode
        memory_bank.end_episode()
        
        # Show progress
        if (episode_num + 1) % 5 == 0:
            stats = memory_bank.get_memory_statistics()
            print(f"  📊 Memory stats: {stats['total_episodes']} episodes, "
                  f"{stats['total_experiences']} experiences, "
                  f"{stats['knowledge_graph_edges']} causal relationships")
    
    # Test retrieval
    print(f"\n🔍 Testing memory retrieval...")
    
    # Create query
    query_obs = CausalObservation(
        state_data=np.array([1.0, 0.5, -0.2, 0.3, -0.1]),
        timestamp=time.time(),
        metadata={"variable_names": [f"var_{i}" for i in range(5)]}
    )
    
    query_action = CausalAction(
        target_variables=["var_1"],
        intervention_values=np.array([0.8]),
        intervention_type="do",
        confidence=0.9,
        reasoning_chain=["Query intervention"]
    )
    
    similar_episodes = memory_bank.retrieve_similar_experiences(query_obs, query_action, k=3)
    print(f"Found {len(similar_episodes)} similar episodes")
    
    for i, episode in enumerate(similar_episodes):
        print(f"  {i+1}. {episode.episode_id}: reward={episode.total_reward:.3f}, "
              f"length={episode.episode_length}, importance={episode.causal_importance:.3f}")
    
    # Test causal knowledge
    print(f"\n🔗 Testing causal knowledge...")
    
    knowledge = memory_bank.get_causal_knowledge("var_0", "var_2")
    print(f"Causal relationship var_0 -> var_2:")
    print(f"  Exists: {knowledge['relationship_exists']}")
    print(f"  Strength: {knowledge['strength']:.3f}")
    print(f"  Confidence: {knowledge['confidence']:.3f}")
    print(f"  Evidence count: {knowledge['evidence_count']}")
    
    # Final statistics
    final_stats = memory_bank.get_memory_statistics()
    print(f"\n📈 Final Memory Statistics:")
    print(f"  Total episodes: {final_stats['total_episodes']}")
    print(f"  Total experiences: {final_stats['total_experiences']}")
    print(f"  Consolidations: {final_stats['consolidation_count']}")
    print(f"  Knowledge graph: {final_stats['knowledge_graph_nodes']} nodes, {final_stats['knowledge_graph_edges']} edges")
    print(f"  Patterns discovered: {sum(final_stats['consolidated_patterns'].values())}")
    
    return memory_bank


if __name__ == "__main__":
    memory_bank = demo_causal_memory_bank()
    print(f"\n✅ Causal memory bank demo completed!")