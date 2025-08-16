"""Neuromorphic causal discovery algorithms inspired by brain dynamics."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import scipy.signal as signal
from .base import CausalDiscoveryModel, CausalResult


@dataclass
class SynapticConnection:
    """Represents a synaptic connection between neurons."""
    pre_neuron: int
    post_neuron: int
    weight: float
    plasticity: float
    delay: int
    spike_history: List[float]


@dataclass
class NeuronState:
    """State of a leaky integrate-and-fire neuron."""
    membrane_potential: float
    threshold: float
    refractory_period: int
    tau_membrane: float
    tau_synaptic: float
    spike_times: List[float]


class SpikingNeuralCausal(CausalDiscoveryModel):
    """Causal discovery using spiking neural network dynamics."""
    
    def __init__(self,
                 n_neurons: Optional[int] = None,
                 membrane_time_constant: float = 20.0,
                 synaptic_time_constant: float = 5.0,
                 threshold_voltage: float = -55.0,
                 reset_voltage: float = -70.0,
                 refractory_period: int = 2,
                 plasticity_rate: float = 0.01,
                 stdp_window: float = 20.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.n_neurons = n_neurons
        self.membrane_time_constant = membrane_time_constant
        self.synaptic_time_constant = synaptic_time_constant
        self.threshold_voltage = threshold_voltage
        self.reset_voltage = reset_voltage
        self.refractory_period = refractory_period
        self.plasticity_rate = plasticity_rate
        self.stdp_window = stdp_window
        self.neurons = []
        self.synapses = {}
        
    def _initialize_neurons(self, n_variables: int) -> List[NeuronState]:
        """Initialize leaky integrate-and-fire neurons."""
        neurons = []
        
        for i in range(n_variables):
            neuron = NeuronState(
                membrane_potential=self.reset_voltage,
                threshold=self.threshold_voltage,
                refractory_period=0,
                tau_membrane=self.membrane_time_constant,
                tau_synaptic=self.synaptic_time_constant,
                spike_times=[]
            )
            neurons.append(neuron)
        
        return neurons
    
    def _initialize_synapses(self, n_variables: int) -> Dict[Tuple[int, int], SynapticConnection]:
        """Initialize synaptic connections between neurons."""
        synapses = {}
        
        for i in range(n_variables):
            for j in range(n_variables):
                if i != j:
                    synapse = SynapticConnection(
                        pre_neuron=i,
                        post_neuron=j,
                        weight=np.random.normal(0, 0.1),
                        plasticity=1.0,
                        delay=np.random.randint(1, 5),
                        spike_history=[]
                    )
                    synapses[(i, j)] = synapse
        
        return synapses
    
    def _convert_data_to_spikes(self, data: pd.DataFrame) -> Dict[int, List[float]]:
        """Convert continuous data to spike trains using Poisson encoding."""
        spike_trains = {}
        dt = 1.0  # Time step in milliseconds
        
        for i, column in enumerate(data.columns):
            # Normalize data to [0, 1]
            normalized_data = (data[column] - data[column].min()) / (data[column].max() - data[column].min())
            
            # Convert to firing rates (Hz)
            max_rate = 100.0  # Maximum firing rate
            firing_rates = normalized_data * max_rate
            
            # Generate Poisson spike train
            spike_times = []
            for t, rate in enumerate(firing_rates):
                # Probability of spike in time step dt
                spike_prob = rate * dt / 1000.0  # Convert Hz to probability per ms
                
                if np.random.random() < spike_prob:
                    spike_times.append(t * dt)
            
            spike_trains[i] = spike_times
        
        return spike_trains
    
    def _update_membrane_potential(self, neuron: NeuronState, synaptic_input: float, dt: float) -> bool:
        """Update neuron membrane potential and check for spike."""
        if neuron.refractory_period > 0:
            neuron.refractory_period -= 1
            return False
        
        # Leaky integrate dynamics
        leak_current = -(neuron.membrane_potential - self.reset_voltage) / neuron.tau_membrane
        
        # Update membrane potential
        dv_dt = leak_current + synaptic_input
        neuron.membrane_potential += dv_dt * dt
        
        # Check for spike
        if neuron.membrane_potential >= neuron.threshold:
            neuron.membrane_potential = self.reset_voltage
            neuron.refractory_period = self.refractory_period
            return True
        
        return False
    
    def _calculate_synaptic_current(self, synapse: SynapticConnection, pre_spike_times: List[float], 
                                  current_time: float) -> float:
        """Calculate synaptic current from presynaptic spikes."""
        current = 0.0
        
        for spike_time in pre_spike_times:
            # Check if spike is within delay window
            time_diff = current_time - spike_time - synapse.delay
            
            if 0 <= time_diff <= 5 * synapse.tau_synaptic:
                # Exponential decay synaptic current
                current += synapse.weight * np.exp(-time_diff / synapse.tau_synaptic)
        
        return current
    
    def _apply_stdp(self, synapse: SynapticConnection, pre_spike_time: float, post_spike_time: float):
        """Apply spike-timing-dependent plasticity (STDP)."""
        time_diff = post_spike_time - pre_spike_time
        
        if abs(time_diff) <= self.stdp_window:
            if time_diff > 0:  # Post before pre - LTP
                weight_change = self.plasticity_rate * np.exp(-time_diff / 10.0)
            else:  # Pre before post - LTD
                weight_change = -self.plasticity_rate * np.exp(time_diff / 10.0)
            
            synapse.weight += weight_change
            synapse.weight = np.clip(synapse.weight, -1.0, 1.0)  # Bound synaptic weights
    
    def _simulate_network(self, spike_trains: Dict[int, List[float]], simulation_time: float) -> Dict[int, List[float]]:
        """Simulate the spiking neural network."""
        dt = 0.1  # Time step in milliseconds
        n_steps = int(simulation_time / dt)
        
        output_spike_trains = {i: [] for i in range(len(self.neurons))}
        
        for step in range(n_steps):
            current_time = step * dt
            
            # Calculate synaptic inputs for each neuron
            synaptic_inputs = [0.0] * len(self.neurons)
            
            for (pre_idx, post_idx), synapse in self.synapses.items():
                if pre_idx in spike_trains:
                    synaptic_current = self._calculate_synaptic_current(
                        synapse, spike_trains[pre_idx], current_time
                    )
                    synaptic_inputs[post_idx] += synaptic_current
            
            # Update each neuron
            for i, neuron in enumerate(self.neurons):
                spiked = self._update_membrane_potential(neuron, synaptic_inputs[i], dt)
                
                if spiked:
                    output_spike_trains[i].append(current_time)
                    neuron.spike_times.append(current_time)
                    
                    # Apply STDP to all synapses targeting this neuron
                    for (pre_idx, post_idx), synapse in self.synapses.items():
                        if post_idx == i:
                            # Find recent presynaptic spikes
                            for pre_spike_time in spike_trains.get(pre_idx, []):
                                if abs(current_time - pre_spike_time) <= self.stdp_window:
                                    self._apply_stdp(synapse, pre_spike_time, current_time)
        
        return output_spike_trains
    
    def _analyze_causal_connectivity(self) -> Tuple[np.ndarray, np.ndarray]:
        """Analyze synaptic weights to determine causal connectivity."""
        n_vars = len(self.neurons)
        adjacency_matrix = np.zeros((n_vars, n_vars))
        confidence_scores = np.zeros((n_vars, n_vars))
        
        for (pre_idx, post_idx), synapse in self.synapses.items():
            # Significant positive weights indicate causal connections
            if abs(synapse.weight) > 0.1:  # Threshold for significant connection
                adjacency_matrix[pre_idx, post_idx] = 1 if synapse.weight > 0 else 0
            
            # Confidence based on absolute weight strength
            confidence_scores[pre_idx, post_idx] = abs(synapse.weight)
        
        return adjacency_matrix, confidence_scores
    
    def fit(self, data: pd.DataFrame) -> 'SpikingNeuralCausal':
        """Fit the spiking neural network causal model."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        
        if data.empty:
            raise ValueError("Data cannot be empty")
        
        n_variables = len(data.columns)
        self.variable_names = list(data.columns)
        
        # Initialize network components
        self.neurons = self._initialize_neurons(n_variables)
        self.synapses = self._initialize_synapses(n_variables)
        
        self._fitted_data = data
        self.is_fitted = True
        return self
    
    def predict(self, data: pd.DataFrame) -> CausalResult:
        """Predict causal relationships using spiking neural dynamics."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Convert data to spike trains
        spike_trains = self._convert_data_to_spikes(data)
        
        # Simulate network for learning
        simulation_time = len(data) * 10.0  # 10ms per data point
        output_spikes = self._simulate_network(spike_trains, simulation_time)
        
        # Analyze learned connectivity
        adjacency_matrix, confidence_scores = self._analyze_causal_connectivity()
        
        # Calculate network statistics
        avg_firing_rate = np.mean([len(spikes) / simulation_time * 1000 for spikes in output_spikes.values()])
        avg_synaptic_weight = np.mean([abs(syn.weight) for syn in self.synapses.values()])
        
        metadata = {
            'simulation_time_ms': simulation_time,
            'average_firing_rate_hz': avg_firing_rate,
            'average_synaptic_weight': avg_synaptic_weight,
            'n_neurons': len(self.neurons),
            'n_synapses': len(self.synapses),
            'plasticity_applied': True,
            'stdp_window_ms': self.stdp_window,
            'variable_names': self.variable_names
        }
        
        return CausalResult(
            adjacency_matrix=adjacency_matrix,
            confidence_scores=confidence_scores,
            method_used="SpikingNeuralCausal",
            metadata=metadata
        )
    
    def discover(self, data: Optional[pd.DataFrame] = None) -> CausalResult:
        """Discover causal relationships."""
        if data is None:
            if not hasattr(self, '_fitted_data'):
                raise ValueError("No data provided and model has no fitted data")
            data = self._fitted_data
        
        return self.predict(data)


class ReservoirComputingCausal(CausalDiscoveryModel):
    """Causal discovery using reservoir computing principles."""
    
    def __init__(self,
                 reservoir_size: int = 100,
                 spectral_radius: float = 0.9,
                 input_scaling: float = 0.1,
                 leak_rate: float = 0.1,
                 ridge_regression_param: float = 1e-6,
                 **kwargs):
        super().__init__(**kwargs)
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leak_rate = leak_rate
        self.ridge_regression_param = ridge_regression_param
        self.reservoir_weights = None
        self.input_weights = None
        self.output_weights = None
        
    def _initialize_reservoir(self, n_inputs: int):
        """Initialize the reservoir with random weights."""
        # Random reservoir weights
        self.reservoir_weights = np.random.randn(self.reservoir_size, self.reservoir_size)
        
        # Scale to desired spectral radius
        eigenvals = np.linalg.eigvals(self.reservoir_weights)
        current_spectral_radius = np.max(np.abs(eigenvals))
        self.reservoir_weights *= self.spectral_radius / current_spectral_radius
        
        # Random input weights
        self.input_weights = np.random.randn(self.reservoir_size, n_inputs) * self.input_scaling
    
    def _run_reservoir(self, inputs: np.ndarray) -> np.ndarray:
        """Run the reservoir with given inputs."""
        n_timesteps = inputs.shape[0]
        reservoir_states = np.zeros((n_timesteps, self.reservoir_size))
        state = np.zeros(self.reservoir_size)
        
        for t in range(n_timesteps):
            # Update reservoir state
            input_activation = np.dot(self.input_weights, inputs[t])
            reservoir_activation = np.dot(self.reservoir_weights, state)
            
            # Leaky integration with tanh nonlinearity
            new_state = np.tanh(input_activation + reservoir_activation)
            state = (1 - self.leak_rate) * state + self.leak_rate * new_state
            
            reservoir_states[t] = state
        
        return reservoir_states
    
    def _train_readout(self, reservoir_states: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Train linear readout using ridge regression."""
        # Add bias term
        extended_states = np.column_stack([reservoir_states, np.ones(reservoir_states.shape[0])])
        
        # Ridge regression
        regularization = self.ridge_regression_param * np.eye(extended_states.shape[1])
        
        try:
            self.output_weights = np.linalg.solve(
                extended_states.T @ extended_states + regularization,
                extended_states.T @ targets
            )
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse if matrix is singular
            self.output_weights = np.linalg.pinv(extended_states.T @ extended_states + regularization) @ extended_states.T @ targets
        
        return extended_states @ self.output_weights
    
    def _analyze_reservoir_connectivity(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Analyze causal connectivity through reservoir dynamics."""
        n_variables = len(data.columns)
        adjacency_matrix = np.zeros((n_variables, n_variables))
        confidence_scores = np.zeros((n_variables, n_variables))
        
        # For each variable, assess its causal influence on others
        for target_var in range(n_variables):
            # Create input where only one variable changes at a time
            for source_var in range(n_variables):
                if source_var != target_var:
                    # Create perturbation in source variable
                    perturbed_data = data.values.copy()
                    perturbation = np.random.normal(0, 0.1, len(data))
                    perturbed_data[:, source_var] += perturbation
                    
                    # Run original and perturbed data through reservoir
                    original_states = self._run_reservoir(data.values)
                    perturbed_states = self._run_reservoir(perturbed_data)
                    
                    # Measure difference in reservoir response
                    state_difference = np.mean(np.abs(perturbed_states - original_states), axis=0)
                    causal_strength = np.mean(state_difference)
                    
                    # Threshold for determining causal connection
                    if causal_strength > np.std(state_difference) * 2:
                        adjacency_matrix[source_var, target_var] = 1
                    
                    confidence_scores[source_var, target_var] = causal_strength
        
        return adjacency_matrix, confidence_scores
    
    def fit(self, data: pd.DataFrame) -> 'ReservoirComputingCausal':
        """Fit the reservoir computing causal model."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        
        if data.empty:
            raise ValueError("Data cannot be empty")
        
        n_variables = len(data.columns)
        self.variable_names = list(data.columns)
        
        # Initialize reservoir
        self._initialize_reservoir(n_variables)
        
        self._fitted_data = data
        self.is_fitted = True
        return self
    
    def predict(self, data: pd.DataFrame) -> CausalResult:
        """Predict causal relationships using reservoir computing."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Analyze causal connectivity through reservoir dynamics
        adjacency_matrix, confidence_scores = self._analyze_reservoir_connectivity(data)
        
        # Calculate reservoir statistics
        reservoir_states = self._run_reservoir(data.values)
        reservoir_activity = np.mean(np.abs(reservoir_states))
        state_diversity = np.mean(np.std(reservoir_states, axis=0))
        
        metadata = {
            'reservoir_size': self.reservoir_size,
            'spectral_radius': self.spectral_radius,
            'leak_rate': self.leak_rate,
            'reservoir_activity': reservoir_activity,
            'state_diversity': state_diversity,
            'input_scaling': self.input_scaling,
            'variable_names': self.variable_names
        }
        
        return CausalResult(
            adjacency_matrix=adjacency_matrix,
            confidence_scores=confidence_scores,
            method_used="ReservoirComputingCausal",
            metadata=metadata
        )
    
    def discover(self, data: Optional[pd.DataFrame] = None) -> CausalResult:
        """Discover causal relationships."""
        if data is None:
            if not hasattr(self, '_fitted_data'):
                raise ValueError("No data provided and model has no fitted data")
            data = self._fitted_data
        
        return self.predict(data)