"""Specialized data processing utilities for bioneuro-olfactory research."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from scipy import signal, stats
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

try:
    from .validation import DataValidator
    from .security import DataSecurityValidator
    from .monitoring import monitor_performance
except ImportError:
    # For direct execution
    from validation import DataValidator
    from security import DataSecurityValidator
    from monitoring import monitor_performance

logger = logging.getLogger(__name__)


@dataclass
class OlfactoryDataProcessingConfig:
    """Configuration for olfactory data processing."""
    sampling_rate_hz: float = 1000.0
    filter_low_cutoff: float = 1.0
    filter_high_cutoff: float = 100.0
    spike_detection_threshold: float = 3.0
    artifact_removal: bool = True
    normalization_method: str = "z_score"  # "z_score", "min_max", "robust"
    temporal_window_ms: int = 100
    baseline_correction: bool = True


class BioneuroDataProcessor:
    """Specialized data processor for bioneuro-olfactory research."""
    
    def __init__(self, config: Optional[OlfactoryDataProcessingConfig] = None):
        """
        Initialize the bioneuro data processor.
        
        Args:
            config: Processing configuration, uses defaults if None
        """
        self.config = config or OlfactoryDataProcessingConfig()
        self.validator = DataValidator(strict=False)
        self.security = DataSecurityValidator()
        self.scalers = {}
        
        logger.info(f"Initialized BioneuroDataProcessor with sampling_rate={self.config.sampling_rate_hz}Hz")
    
    @monitor_performance()
    def process_olfactory_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw olfactory neural signals.
        
        Args:
            data: Raw signal data with columns for different sensors/neurons
            
        Returns:
            Processed DataFrame with cleaned and normalized signals
        """
        # Validate input
        validation_result = self.validator.validate_input_data(data)
        if not validation_result.is_valid:
            if validation_result.errors:
                raise ValueError(f"Invalid input data: {validation_result.errors}")
            else:
                # Only warnings, log them but continue
                if validation_result.warnings:
                    logger.warning(f"Data validation warnings: {validation_result.warnings}")
        security_result = self.security.validate_data_security(data)
        if not security_result.is_secure:
            logger.warning(f"Security issues detected: {security_result.issues}")
        
        processed_data = data.copy()
        
        # 1. Artifact removal
        if self.config.artifact_removal:
            processed_data = self._remove_artifacts(processed_data)
        
        # 2. Filtering
        processed_data = self._apply_bandpass_filter(processed_data)
        
        # 3. Spike detection and processing
        processed_data = self._detect_and_process_spikes(processed_data)
        
        # 4. Baseline correction
        if self.config.baseline_correction:
            processed_data = self._apply_baseline_correction(processed_data)
        
        # 5. Normalization
        processed_data = self._normalize_signals(processed_data)
        
        # 6. Feature extraction
        processed_data = self._extract_temporal_features(processed_data)
        
        logger.info(f"Processed olfactory signals: {len(data)} -> {len(processed_data)} samples")
        return processed_data
    
    @monitor_performance()
    def process_neural_activity(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process neural activity data (firing rates, LFP, etc.).
        
        Args:
            data: Neural activity data
            
        Returns:
            Processed neural activity DataFrame
        """
        validation_result = self.validator.validate_input_data(data)
        if not validation_result.is_valid:
            if validation_result.errors:
                raise ValueError(f"Invalid input data: {validation_result.errors}")
            else:
                # Only warnings, log them but continue
                if validation_result.warnings:
                    logger.warning(f"Data validation warnings: {validation_result.warnings}")
        processed_data = data.copy()
        
        # Neural-specific processing
        neural_cols = [col for col in data.columns if any(term in col.lower() 
                      for term in ['neural', 'firing', 'spike', 'lfp', 'potential'])]
        
        for col in neural_cols:
            if pd.api.types.is_numeric_dtype(processed_data[col]):
                # Firing rate smoothing
                processed_data[col] = self._smooth_firing_rates(processed_data[col].values)
                
                # Burst detection
                burst_features = self._detect_burst_patterns(processed_data[col].values)
                processed_data[f"{col}_burst_rate"] = burst_features['burst_rate']
                processed_data[f"{col}_interburst_interval"] = burst_features['interburst_interval']
        
        return processed_data
    
    @monitor_performance()
    def process_odor_stimuli(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process odor stimulus data (concentrations, timing, etc.).
        
        Args:
            data: Odor stimulus data
            
        Returns:
            Processed odor stimulus DataFrame
        """
        processed_data = data.copy()
        
        # Odor-specific processing
        odor_cols = [col for col in data.columns if any(term in col.lower() 
                    for term in ['odor', 'concentration', 'stimulus', 'odorant'])]
        
        for col in odor_cols:
            if pd.api.types.is_numeric_dtype(processed_data[col]):
                # Log transform for concentration data
                if 'concentration' in col.lower():
                    processed_data[f"{col}_log"] = np.log1p(np.maximum(processed_data[col], 0))
                
                # Stimulus onset/offset detection
                stimulus_events = self._detect_stimulus_events(processed_data[col].values)
                processed_data[f"{col}_onset"] = stimulus_events['onset']
                processed_data[f"{col}_offset"] = stimulus_events['offset']
        
        return processed_data
    
    def _remove_artifacts(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove common artifacts from neural data."""
        cleaned_data = data.copy()
        
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                values = cleaned_data[col].values
                
                # Remove extreme outliers (beyond 5 standard deviations)
                z_scores = np.abs(stats.zscore(values, nan_policy='omit'))
                outlier_mask = z_scores < 5
                
                # Interpolate outliers
                if not np.all(outlier_mask):
                    cleaned_values = values.copy()
                    cleaned_values[~outlier_mask] = np.nan
                    cleaned_data[col] = pd.Series(cleaned_values).interpolate()
                
                # Remove line noise (50/60 Hz and harmonics)
                cleaned_data[col] = self._remove_line_noise(cleaned_data[col].values)
        
        return cleaned_data
    
    def _apply_bandpass_filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply bandpass filter to remove unwanted frequencies."""
        filtered_data = data.copy()
        
        # Design Butterworth bandpass filter
        nyquist = self.config.sampling_rate_hz / 2
        low = self.config.filter_low_cutoff / nyquist
        high = self.config.filter_high_cutoff / nyquist
        
        if high >= 1.0:
            high = 0.99  # Avoid filter instability
        
        try:
            b, a = signal.butter(4, [low, high], btype='band')
            
            for col in data.columns:
                if pd.api.types.is_numeric_dtype(data[col]):
                    values = filtered_data[col].values
                    if len(values) > 10:  # Minimum length for filtering
                        filtered_values = signal.filtfilt(b, a, values)
                        filtered_data[col] = filtered_values
        except Exception as e:
            logger.warning(f"Bandpass filtering failed: {str(e)}")
        
        return filtered_data
    
    def _detect_and_process_spikes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect and process neural spikes."""
        spike_data = data.copy()
        
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]) and 'neural' in col.lower():
                values = data[col].values
                
                # Spike detection using threshold crossing
                threshold = self.config.spike_detection_threshold * np.std(values)
                spike_times = self._find_spike_times(values, threshold)
                
                # Add spike-related features
                spike_data[f"{col}_spike_rate"] = self._compute_spike_rate(spike_times, len(values))
                spike_data[f"{col}_spike_amplitude"] = self._compute_spike_amplitudes(values, spike_times)
        
        return spike_data
    
    def _apply_baseline_correction(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply baseline correction to remove drift."""
        corrected_data = data.copy()
        
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                values = data[col].values
                
                # Remove linear trend
                detrended = signal.detrend(values, type='linear')
                
                # Subtract moving baseline
                baseline_window = int(self.config.sampling_rate_hz * 0.1)  # 100ms window
                if len(values) > baseline_window:
                    baseline = pd.Series(detrended).rolling(
                        window=baseline_window, center=True, min_periods=1
                    ).median()
                    corrected_data[col] = detrended - baseline
                else:
                    corrected_data[col] = detrended
        
        return corrected_data
    
    def _normalize_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize signals using specified method."""
        normalized_data = data.copy()
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.config.normalization_method == "z_score":
                scaler = StandardScaler()
            elif self.config.normalization_method == "min_max":
                scaler = MinMaxScaler()
            elif self.config.normalization_method == "robust":
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
            else:
                continue
            
            values = data[col].values.reshape(-1, 1)
            normalized_values = scaler.fit_transform(values).flatten()
            normalized_data[col] = normalized_values
            
            # Store scaler for inverse transform if needed
            self.scalers[col] = scaler
        
        return normalized_data
    
    def _extract_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from signals."""
        feature_data = data.copy()
        
        window_samples = int(self.config.temporal_window_ms * self.config.sampling_rate_hz / 1000)
        
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                values = data[col].values
                
                if len(values) > window_samples:
                    # Windowed statistics
                    windowed_mean = pd.Series(values).rolling(window=window_samples).mean()
                    windowed_std = pd.Series(values).rolling(window=window_samples).std()
                    
                    feature_data[f"{col}_windowed_mean"] = windowed_mean
                    feature_data[f"{col}_windowed_std"] = windowed_std
                    
                    # Spectral features
                    if len(values) > 2 * window_samples:
                        spectral_features = self._compute_spectral_features(values, window_samples)
                        feature_data[f"{col}_dominant_freq"] = spectral_features['dominant_frequency']
                        feature_data[f"{col}_spectral_power"] = spectral_features['total_power']
        
        return feature_data
    
    def _smooth_firing_rates(self, firing_rates: np.ndarray) -> np.ndarray:
        """Smooth neural firing rate data."""
        if len(firing_rates) < 5:
            return firing_rates
        
        # Gaussian smoothing
        sigma = self.config.sampling_rate_hz * 0.01  # 10ms smoothing
        smoothed = gaussian_filter1d(firing_rates, sigma=sigma)
        return smoothed
    
    def _detect_burst_patterns(self, neural_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Detect burst patterns in neural activity."""
        if len(neural_data) < 10:
            return {
                'burst_rate': np.zeros_like(neural_data),
                'interburst_interval': np.zeros_like(neural_data)
            }
        
        # Simple burst detection based on activity level
        threshold = np.mean(neural_data) + 2 * np.std(neural_data)
        burst_mask = neural_data > threshold
        
        # Compute burst rate (bursts per second)
        burst_rate = pd.Series(burst_mask.astype(float)).rolling(
            window=int(self.config.sampling_rate_hz)
        ).sum() * self.config.sampling_rate_hz / len(neural_data)
        
        # Compute interburst intervals
        burst_times = np.where(burst_mask)[0]
        interburst_intervals = np.zeros_like(neural_data)
        
        if len(burst_times) > 1:
            intervals = np.diff(burst_times) / self.config.sampling_rate_hz
            for i, burst_time in enumerate(burst_times[1:]):
                interburst_intervals[burst_time] = intervals[i]
        
        return {
            'burst_rate': burst_rate.fillna(0).values,
            'interburst_interval': interburst_intervals
        }
    
    def _detect_stimulus_events(self, stimulus_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Detect stimulus onset and offset events."""
        # Simple threshold-based event detection
        threshold = np.mean(stimulus_data) + np.std(stimulus_data)
        
        # Find onset and offset
        above_threshold = stimulus_data > threshold
        onset_mask = np.diff(above_threshold.astype(int), prepend=0) == 1
        offset_mask = np.diff(above_threshold.astype(int), prepend=0) == -1
        
        return {
            'onset': onset_mask.astype(float),
            'offset': offset_mask.astype(float)
        }
    
    def _remove_line_noise(self, signal_data: np.ndarray) -> np.ndarray:
        """Remove 50/60 Hz line noise using notch filter."""
        if len(signal_data) < 10:
            return signal_data
        
        try:
            # Remove 50 Hz and 60 Hz noise
            for freq in [50, 60]:
                nyquist = self.config.sampling_rate_hz / 2
                if freq < nyquist:
                    Q = 30  # Quality factor
                    b, a = signal.iirnotch(freq, Q, self.config.sampling_rate_hz)
                    signal_data = signal.filtfilt(b, a, signal_data)
        except Exception as e:
            logger.warning(f"Line noise removal failed: {str(e)}")
        
        return signal_data
    
    def _find_spike_times(self, signal_data: np.ndarray, threshold: float) -> np.ndarray:
        """Find spike times using threshold crossing."""
        # Find peaks above threshold
        peaks, _ = signal.find_peaks(signal_data, height=threshold, distance=5)
        return peaks
    
    def _compute_spike_rate(self, spike_times: np.ndarray, signal_length: int) -> np.ndarray:
        """Compute instantaneous spike rate."""
        spike_rate = np.zeros(signal_length)
        
        if len(spike_times) > 0:
            # Convert to spike rate (spikes per second)
            window_size = int(self.config.sampling_rate_hz * 0.05)  # 50ms window
            
            for spike_time in spike_times:
                start_idx = max(0, spike_time - window_size // 2)
                end_idx = min(signal_length, spike_time + window_size // 2)
                spike_rate[start_idx:end_idx] += 1
            
            spike_rate = spike_rate / (window_size / self.config.sampling_rate_hz)
        
        return spike_rate
    
    def _compute_spike_amplitudes(self, signal_data: np.ndarray, spike_times: np.ndarray) -> np.ndarray:
        """Compute spike amplitudes."""
        amplitude_signal = np.zeros_like(signal_data)
        
        for spike_time in spike_times:
            if spike_time < len(signal_data):
                amplitude_signal[spike_time] = signal_data[spike_time]
        
        return amplitude_signal
    
    def _compute_spectral_features(self, signal_data: np.ndarray, window_size: int) -> Dict[str, np.ndarray]:
        """Compute spectral features using windowed FFT."""
        try:
            # Compute power spectral density
            freqs, psd = signal.welch(
                signal_data, 
                fs=self.config.sampling_rate_hz,
                nperseg=min(window_size, len(signal_data) // 4)
            )
            
            # Find dominant frequency
            dominant_freq_idx = np.argmax(psd)
            dominant_frequency = freqs[dominant_freq_idx]
            
            # Total spectral power
            total_power = np.sum(psd)
            
            return {
                'dominant_frequency': np.full_like(signal_data, dominant_frequency),
                'total_power': np.full_like(signal_data, total_power)
            }
        except Exception as e:
            logger.warning(f"Spectral feature computation failed: {str(e)}")
            return {
                'dominant_frequency': np.zeros_like(signal_data),
                'total_power': np.zeros_like(signal_data)
            }


class OlfactoryFeatureExtractor:
    """Specialized feature extractor for olfactory neural data."""
    
    def __init__(self, sampling_rate: float = 1000.0):
        """
        Initialize feature extractor.
        
        Args:
            sampling_rate: Sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        logger.info(f"Initialized OlfactoryFeatureExtractor with fs={sampling_rate}Hz")
    
    @monitor_performance()
    def extract_olfactory_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract comprehensive olfactory-specific features.
        
        Args:
            data: Processed olfactory data
            
        Returns:
            DataFrame with extracted features
        """
        features = data.copy()
        
        # 1. Receptor response features
        features = self._extract_receptor_features(features)
        
        # 2. Neural encoding features
        features = self._extract_neural_encoding_features(features)
        
        # 3. Temporal dynamics features
        features = self._extract_temporal_dynamics(features)
        
        # 4. Cross-modal integration features
        features = self._extract_cross_modal_features(features)
        
        logger.info(f"Extracted features: {data.shape[1]} -> {features.shape[1]} columns")
        return features
    
    def _extract_receptor_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract receptor-specific features."""
        receptor_cols = [col for col in data.columns if 'receptor' in col.lower()]
        
        for col in receptor_cols:
            if pd.api.types.is_numeric_dtype(data[col]):
                # Response magnitude
                data[f"{col}_magnitude"] = np.abs(data[col])
                
                # Response latency (time to peak)
                if len(data[col]) > 1:
                    peak_idx = np.argmax(np.abs(data[col]))
                    data[f"{col}_latency"] = peak_idx / self.sampling_rate
                
                # Adaptation rate
                if len(data[col]) > 10:
                    adaptation_rate = self._compute_adaptation_rate(data[col].values)
                    data[f"{col}_adaptation_rate"] = adaptation_rate
        
        return data
    
    def _extract_neural_encoding_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract neural encoding features."""
        neural_cols = [col for col in data.columns if 'neural' in col.lower()]
        
        for col in neural_cols:
            if pd.api.types.is_numeric_dtype(data[col]):
                # Information content (approximate entropy)
                if len(data[col]) > 10:
                    entropy = self._compute_approximate_entropy(data[col].values)
                    data[f"{col}_entropy"] = entropy
                
                # Synchronization index
                sync_index = self._compute_synchronization_index(data[col].values)
                data[f"{col}_synchronization"] = sync_index
        
        return data
    
    def _extract_temporal_dynamics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal dynamics features."""
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                # Rate of change
                if len(data[col]) > 1:
                    rate_of_change = np.gradient(data[col])
                    data[f"{col}_rate_of_change"] = rate_of_change
                
                # Periodicity
                if len(data[col]) > 20:
                    periodicity = self._detect_periodicity(data[col].values)
                    data[f"{col}_periodicity"] = periodicity
        
        return data
    
    def _extract_cross_modal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract cross-modal integration features."""
        receptor_cols = [col for col in data.columns if 'receptor' in col.lower()]
        neural_cols = [col for col in data.columns if 'neural' in col.lower()]
        
        # Cross-correlation features
        for r_col in receptor_cols:
            for n_col in neural_cols:
                if (pd.api.types.is_numeric_dtype(data[r_col]) and 
                    pd.api.types.is_numeric_dtype(data[n_col])):
                    
                    cross_corr = np.corrcoef(data[r_col], data[n_col])[0, 1]
                    if not np.isnan(cross_corr):
                        data[f"{r_col}_{n_col}_cross_correlation"] = cross_corr
        
        return data
    
    def _compute_adaptation_rate(self, signal: np.ndarray) -> np.ndarray:
        """Compute receptor adaptation rate."""
        if len(signal) < 5:
            return np.zeros_like(signal)
        
        # Fit exponential decay to signal
        try:
            t = np.arange(len(signal))
            # Simple approximation: rate of decay in moving window
            window_size = min(10, len(signal) // 2)
            adaptation_rate = []
            
            for i in range(len(signal)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(signal), i + window_size // 2)
                window_signal = signal[start_idx:end_idx]
                
                if len(window_signal) > 2:
                    # Linear fit in log space (exponential decay)
                    try:
                        pos_signal = np.maximum(window_signal, 1e-10)
                        slope = np.polyfit(range(len(pos_signal)), np.log(pos_signal), 1)[0]
                        adaptation_rate.append(-slope)  # Negative slope = decay
                    except:
                        adaptation_rate.append(0.0)
                else:
                    adaptation_rate.append(0.0)
            
            return np.array(adaptation_rate)
        except:
            return np.zeros_like(signal)
    
    def _compute_approximate_entropy(self, signal: np.ndarray) -> np.ndarray:
        """Compute approximate entropy as measure of regularity."""
        if len(signal) < 5:
            return np.zeros_like(signal)
        
        # Simplified approximate entropy calculation
        # Using local variance as proxy for entropy
        window_size = min(5, len(signal) // 2)
        entropy_signal = []
        
        for i in range(len(signal)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(signal), i + window_size // 2)
            window_signal = signal[start_idx:end_idx]
            
            # Use variance as entropy proxy
            entropy_val = np.var(window_signal) if len(window_signal) > 1 else 0
            entropy_signal.append(entropy_val)
        
        return np.array(entropy_signal)
    
    def _compute_synchronization_index(self, signal: np.ndarray) -> np.ndarray:
        """Compute neural synchronization index."""
        if len(signal) < 3:
            return np.zeros_like(signal)
        
        # Phase synchronization approximation
        # Using Hilbert transform for phase extraction
        try:
            analytic_signal = signal + 1j * np.imag(signal)  # Simplified
            phase = np.angle(analytic_signal)
            
            # Compute phase locking value
            window_size = min(10, len(signal) // 3)
            sync_index = []
            
            for i in range(len(signal)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(signal), i + window_size // 2)
                window_phase = phase[start_idx:end_idx]
                
                # Circular variance as synchronization measure
                if len(window_phase) > 1:
                    mean_phase = np.angle(np.mean(np.exp(1j * window_phase)))
                    sync_val = 1 - np.var(np.cos(window_phase - mean_phase))
                    sync_index.append(max(0, sync_val))
                else:
                    sync_index.append(0.0)
            
            return np.array(sync_index)
        except:
            return np.zeros_like(signal)
    
    def _detect_periodicity(self, signal: np.ndarray) -> np.ndarray:
        """Detect periodic patterns in signal."""
        if len(signal) < 10:
            return np.zeros_like(signal)
        
        try:
            # Autocorrelation-based periodicity detection
            autocorr = np.correlate(signal, signal, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Find peaks in autocorrelation (indicating periodicity)
            if len(autocorr) > 5:
                peaks, _ = signal.find_peaks(autocorr[1:], height=0.1 * np.max(autocorr))
                
                # Dominant period
                if len(peaks) > 0:
                    dominant_period = peaks[0] + 1  # +1 for offset
                    periodicity_strength = autocorr[dominant_period] / autocorr[0]
                else:
                    periodicity_strength = 0
            else:
                periodicity_strength = 0
            
            return np.full_like(signal, periodicity_strength)
        except:
            return np.zeros_like(signal)