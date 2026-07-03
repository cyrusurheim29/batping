"""
Iterative Local Tangent Estimation for Pulse Optimization

This module implements a gradient-based optimization method that experimentally
estimates the Jacobian of the piezo-acoustic system and iteratively refines
the input waveform to match a target output pulse.

Based on the "Iterative Local Tangent Estimation" proposal.
"""

import numpy as np
import pyaudio
import time
from datetime import datetime
from scipy import signal
from dataclasses import dataclass
from typing import Optional, Tuple, List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from agilent_33220a import Agilent33220A, generate_gaussian_pulse, generate_shaped_pulse


@dataclass
class OptimizationResult:
    """Results from tangent optimization."""
    final_coeffs: np.ndarray
    final_error: float
    error_history: List[float]
    coeffs_history: List[np.ndarray]
    iterations: int
    converged: bool


class TangentOptimizer:
    """
    Iterative Local Tangent Estimation optimizer.

    Uses Fourier-parameterized waveforms and experimentally estimates
    the Jacobian to find optimal input waveforms.
    """

    def __init__(
        self,
        n_points: int = 4000,
        carrier_freq: float = 40000,
        sample_rate_awg: float = 1e6,
        sample_rate_mic: int = 384000,
        drive_amplitude: float = 5.0,
        n_fourier_coeffs: int = 8,
        epsilon: float = 0.05,
        lambda_reg: float = 0.01,
        alpha: float = 0.5,
        max_iterations: int = 20,
        tolerance: float = 1e-4,
        settling_time: float = 0.1,
        record_duration: float = 0.05,
    ):
        """
        Initialize the tangent optimizer.

        Args:
            n_points: Number of waveform samples
            carrier_freq: Carrier frequency (Hz)
            sample_rate_awg: AWG sample rate (Sa/s)
            sample_rate_mic: Microphone sample rate (Hz)
            drive_amplitude: AWG output amplitude (Vpp)
            n_fourier_coeffs: Number of Fourier coefficients to optimize
            epsilon: Perturbation size for Jacobian estimation
            lambda_reg: Regularization parameter for least-squares
            alpha: Step size for waveform updates
            max_iterations: Maximum optimization iterations
            tolerance: Convergence tolerance on error norm
            settling_time: Time to wait after AWG output before recording (s)
            record_duration: Recording duration (s)
        """
        self.n_points = n_points
        self.carrier_freq = carrier_freq
        self.sample_rate_awg = sample_rate_awg
        self.sample_rate_mic = sample_rate_mic
        self.drive_amplitude = drive_amplitude
        self.n_fourier_coeffs = n_fourier_coeffs
        self.epsilon = epsilon
        self.lambda_reg = lambda_reg
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.settling_time = settling_time
        self.record_duration = record_duration

        # Hardware handles
        self.awg: Optional[Agilent33220A] = None
        self.pa: Optional[pyaudio.PyAudio] = None
        self.mic_index: Optional[int] = None

        # Derived parameters
        self.waveform_duration_s = n_points / sample_rate_awg
        self.rep_freq = 1.0 / self.waveform_duration_s

        # Signal processing parameters
        self.bandpass_low = 38000
        self.bandpass_high = 42000

        # Target pulse parameters (will be set by generate_target)
        self.target_envelope: Optional[np.ndarray] = None
        self.target_time_ms: Optional[np.ndarray] = None

    def connect_hardware(self):
        """Connect to AWG and find microphone."""
        print("Connecting to Agilent 33220A...")
        self.awg = Agilent33220A()
        self.awg.connect()

        print("Initializing audio...")
        self.pa = pyaudio.PyAudio()

        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            if "UltraMic" in info['name'] and info['maxInputChannels'] > 0:
                self.mic_index = i
                break

        if self.mic_index is None:
            raise RuntimeError("UltraMic not found")

        print(f"Found UltraMic at index {self.mic_index}")

    def disconnect_hardware(self):
        """Disconnect from hardware."""
        if self.awg:
            self.awg.set_output(False)
            self.awg.disconnect()
            self.awg = None
        if self.pa:
            self.pa.terminate()
            self.pa = None

    def record_audio(self) -> np.ndarray:
        """Record audio from microphone."""
        n_samples = int(self.record_duration * self.sample_rate_mic)
        chunk_size = 4096

        stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate_mic,
            input=True,
            input_device_index=self.mic_index,
            frames_per_buffer=chunk_size
        )

        frames = []
        remaining = n_samples

        while remaining > 0:
            chunk = min(chunk_size, remaining)
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)
            remaining -= chunk

        stream.stop_stream()
        stream.close()

        audio = np.frombuffer(b''.join(frames), dtype=np.int16)
        audio = audio.astype(np.float64) / 32768.0

        return audio

    def extract_envelope(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract the envelope of the 40kHz signal.

        Args:
            audio: Raw audio samples

        Returns:
            Envelope of the bandpass-filtered signal
        """
        # Bandpass filter
        nyq = self.sample_rate_mic / 2
        low = self.bandpass_low / nyq
        high = min(self.bandpass_high / nyq, 0.99)
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, audio)

        # Extract envelope via Hilbert transform
        analytic = signal.hilbert(filtered)
        envelope = np.abs(analytic)

        return envelope

    def generate_target_pulse(
        self,
        arrival_time_ms: float,
        sigma_cycles: float = 2.0,
        amplitude: float = 1.0
    ) -> np.ndarray:
        """
        Generate target output pulse (narrow Gaussian at arrival time).

        Args:
            arrival_time_ms: Expected pulse arrival time (ms)
            sigma_cycles: Width of target pulse in carrier cycles
            amplitude: Target amplitude (will be scaled to match measurements)

        Returns:
            Target envelope array
        """
        n_samples = int(self.record_duration * self.sample_rate_mic)
        t_ms = np.arange(n_samples) / self.sample_rate_mic * 1000

        # Gaussian envelope centered at arrival time
        sigma_ms = sigma_cycles / self.carrier_freq * 1000
        target = amplitude * np.exp(-0.5 * ((t_ms - arrival_time_ms) / sigma_ms) ** 2)

        self.target_envelope = target
        self.target_time_ms = t_ms

        return target

    def generate_target_shape(
        self,
        window_samples: int,
        sigma_cycles: float = 2.0,
        amplitude: float = 1.0
    ) -> np.ndarray:
        """
        Generate target pulse SHAPE (centered at window midpoint for alignment).

        This is used for shape-matching rather than fixed-time matching.

        Args:
            window_samples: Number of samples in the window
            sigma_cycles: Width of target pulse in carrier cycles
            amplitude: Target amplitude

        Returns:
            Target shape array (centered)
        """
        t = np.arange(window_samples) / self.sample_rate_mic
        t_center = t[window_samples // 2]
        sigma_s = sigma_cycles / self.carrier_freq

        target = amplitude * np.exp(-0.5 * ((t - t_center) / sigma_s) ** 2)

        return target

    def extract_pulse_window(
        self,
        envelope: np.ndarray,
        window_ms: float = 3.0
    ) -> Tuple[np.ndarray, int]:
        """
        Extract a window around the pulse peak from envelope.

        Args:
            envelope: Full envelope array
            window_ms: Window size in ms (centered on peak)

        Returns:
            Tuple of (windowed envelope, peak index in original)
        """
        peak_idx = np.argmax(envelope)
        window_samples = int(window_ms * self.sample_rate_mic / 1000)
        half_window = window_samples // 2

        # Handle edge cases
        start_idx = max(0, peak_idx - half_window)
        end_idx = min(len(envelope), peak_idx + half_window)

        # Pad if necessary to maintain window size
        window = envelope[start_idx:end_idx]
        if len(window) < window_samples:
            pad_left = max(0, half_window - peak_idx)
            pad_right = max(0, (peak_idx + half_window) - len(envelope))
            window = np.pad(window, (pad_left, pad_right), mode='constant')

        return window[:window_samples], peak_idx

    def generate_waveform(self, coeffs: np.ndarray) -> np.ndarray:
        """
        Generate waveform from Fourier coefficients.

        Args:
            coeffs: Fourier coefficients [a0, a1, b1, a2, b2, ...]

        Returns:
            Normalized waveform array
        """
        return generate_shaped_pulse(
            self.n_points,
            self.carrier_freq,
            self.sample_rate_awg,
            fourier_coeffs=coeffs
        )

    def measure_output(self, coeffs: np.ndarray) -> np.ndarray:
        """
        Send waveform and measure acoustic output envelope.

        Args:
            coeffs: Fourier coefficients

        Returns:
            Measured envelope
        """
        waveform = self.generate_waveform(coeffs)

        self.awg.output_arbitrary(waveform, freq_hz=self.rep_freq, vpp=self.drive_amplitude)
        time.sleep(self.settling_time)

        audio = self.record_audio()
        self.awg.set_output(False)

        envelope = self.extract_envelope(audio)

        return envelope

    def measure_aligned_window(
        self,
        coeffs: np.ndarray,
        window_ms: float = 3.0
    ) -> Tuple[np.ndarray, float]:
        """
        Send waveform and return envelope window aligned to peak.

        This provides timing-invariant measurements for shape optimization.

        Args:
            coeffs: Fourier coefficients
            window_ms: Window size in ms

        Returns:
            Tuple of (windowed envelope centered on peak, peak amplitude)
        """
        envelope = self.measure_output(coeffs)
        window, _ = self.extract_pulse_window(envelope, window_ms)
        peak_amp = np.max(window)

        return window, peak_amp

    def estimate_jacobian(
        self,
        coeffs: np.ndarray,
        y_current: np.ndarray
    ) -> np.ndarray:
        """
        Estimate Jacobian by perturbing each coefficient.

        Args:
            coeffs: Current Fourier coefficients
            y_current: Current output envelope

        Returns:
            Jacobian matrix K (M x N) where M = len(y), N = len(coeffs)
        """
        n_params = len(coeffs)
        n_output = len(y_current)

        K = np.zeros((n_output, n_params))

        for j in range(n_params):
            # Create perturbation vector
            coeffs_perturbed = coeffs.copy()
            coeffs_perturbed[j] += self.epsilon

            # Measure perturbed output
            y_perturbed = self.measure_output(coeffs_perturbed)

            # Finite difference estimate of column j
            K[:, j] = (y_perturbed - y_current) / self.epsilon

            print(f"    Jacobian column {j+1}/{n_params}: ||dy/d{j}|| = {np.linalg.norm(K[:, j]):.4f}")

        return K

    def compute_correction(
        self,
        K: np.ndarray,
        error: np.ndarray
    ) -> np.ndarray:
        """
        Compute optimal correction via regularized least-squares.

        Solves: (K^T K + λI) Δf = -K^T error

        Args:
            K: Jacobian matrix
            error: Current error vector (y - y_target)

        Returns:
            Correction vector Δf
        """
        n_params = K.shape[1]

        # Form normal equations
        KtK = K.T @ K
        Kt_error = K.T @ error

        # Add regularization
        A = KtK + self.lambda_reg * np.eye(n_params)
        b = -Kt_error

        # Solve
        delta_f = np.linalg.solve(A, b)

        return delta_f

    def optimize(
        self,
        initial_coeffs: np.ndarray,
        arrival_time_ms: float,
        target_sigma_cycles: float = 2.0,
        verbose: bool = True
    ) -> OptimizationResult:
        """
        Run the iterative local tangent optimization (fixed-time version).

        Args:
            initial_coeffs: Starting Fourier coefficients
            arrival_time_ms: Expected pulse arrival time (ms)
            target_sigma_cycles: Width of target pulse in cycles
            verbose: Print progress information

        Returns:
            OptimizationResult with final coefficients and history
        """
        if self.awg is None:
            raise RuntimeError("Hardware not connected. Call connect_hardware() first.")

        # Generate target pulse
        if verbose:
            print(f"\nGenerating target pulse at {arrival_time_ms:.2f} ms, σ = {target_sigma_cycles} cycles")

        # First, measure baseline to get amplitude scale
        y_baseline = self.measure_output(initial_coeffs)
        baseline_amp = np.max(y_baseline)

        # Scale target to match baseline amplitude
        self.generate_target_pulse(arrival_time_ms, target_sigma_cycles, amplitude=baseline_amp)

        if verbose:
            print(f"Target amplitude scaled to {baseline_amp:.4f}")

        # Initialize
        coeffs = initial_coeffs.copy()
        error_history = []
        coeffs_history = [coeffs.copy()]

        if verbose:
            print(f"\n{'='*60}")
            print("Starting Iterative Local Tangent Optimization")
            print(f"{'='*60}")
            print(f"Parameters: ε={self.epsilon}, λ={self.lambda_reg}, α={self.alpha}")
            print(f"Max iterations: {self.max_iterations}, tolerance: {self.tolerance}")

        converged = False

        for n in range(self.max_iterations):
            if verbose:
                print(f"\n--- Iteration {n+1} ---")

            # 1. Measure current output
            y_current = self.measure_output(coeffs)

            # 2. Compute error
            error = y_current - self.target_envelope
            error_norm = np.linalg.norm(error) / len(error)  # Normalized MSE
            error_history.append(error_norm)

            if verbose:
                print(f"  Error norm: {error_norm:.6f}")

            # 3. Check convergence
            if error_norm < self.tolerance:
                if verbose:
                    print(f"  Converged! Error below tolerance.")
                converged = True
                break

            # 4. Estimate Jacobian
            if verbose:
                print(f"  Estimating Jacobian...")
            K = self.estimate_jacobian(coeffs, y_current)

            # 5. Compute correction
            delta_f = self.compute_correction(K, error)

            if verbose:
                print(f"  Correction norm: {np.linalg.norm(delta_f):.4f}")

            # 6. Update coefficients
            coeffs = coeffs + self.alpha * delta_f
            coeffs_history.append(coeffs.copy())

            if verbose:
                coeffs_str = ', '.join([f'{c:.3f}' for c in coeffs])
                print(f"  New coeffs: [{coeffs_str}]")

        if verbose:
            print(f"\n{'='*60}")
            print("Optimization Complete")
            print(f"{'='*60}")
            print(f"Iterations: {len(error_history)}")
            print(f"Final error: {error_history[-1]:.6f}")
            print(f"Converged: {converged}")

        return OptimizationResult(
            final_coeffs=coeffs,
            final_error=error_history[-1],
            error_history=error_history,
            coeffs_history=coeffs_history,
            iterations=len(error_history),
            converged=converged
        )

    def optimize_shape(
        self,
        initial_coeffs: np.ndarray,
        target_sigma_cycles: float = 2.0,
        window_ms: float = 3.0,
        verbose: bool = True
    ) -> OptimizationResult:
        """
        Run shape-aligned optimization (timing-invariant).

        Instead of matching a pulse at a fixed time, this aligns each measurement
        to its peak and compares SHAPES. This avoids timing jitter issues.

        Args:
            initial_coeffs: Starting Fourier coefficients
            target_sigma_cycles: Width of target pulse in cycles
            window_ms: Window size for shape comparison (ms)
            verbose: Print progress information

        Returns:
            OptimizationResult with final coefficients and history
        """
        if self.awg is None:
            raise RuntimeError("Hardware not connected. Call connect_hardware() first.")

        window_samples = int(window_ms * self.sample_rate_mic / 1000)

        # First, measure baseline to get amplitude scale
        if verbose:
            print("Measuring baseline for amplitude scaling...")

        y_window_baseline, baseline_amp = self.measure_aligned_window(initial_coeffs, window_ms)

        # Generate target shape (centered Gaussian)
        target_shape = self.generate_target_shape(window_samples, target_sigma_cycles, amplitude=baseline_amp)

        if verbose:
            print(f"Target shape: σ = {target_sigma_cycles} cycles, amplitude = {baseline_amp:.4f}")
            print(f"Window: {window_ms} ms ({window_samples} samples)")

        # Store for plotting
        self.target_envelope = target_shape
        self.target_time_ms = np.arange(window_samples) / self.sample_rate_mic * 1000

        # Initialize
        coeffs = initial_coeffs.copy()
        error_history = []
        coeffs_history = [coeffs.copy()]

        if verbose:
            print(f"\n{'='*60}")
            print("Starting Shape-Aligned Tangent Optimization")
            print(f"{'='*60}")
            print(f"Parameters: ε={self.epsilon}, λ={self.lambda_reg}, α={self.alpha}")
            print(f"Max iterations: {self.max_iterations}, tolerance: {self.tolerance}")

        converged = False

        for n in range(self.max_iterations):
            if verbose:
                print(f"\n--- Iteration {n+1} ---")

            # 1. Measure current output (aligned window)
            y_window, peak_amp = self.measure_aligned_window(coeffs, window_ms)

            # 2. Compute shape error (normalize both to peak=1 for shape comparison)
            y_normalized = y_window / (peak_amp + 1e-10)
            target_normalized = target_shape / (baseline_amp + 1e-10)
            error = y_normalized - target_normalized

            error_norm = np.linalg.norm(error) / len(error)
            error_history.append(error_norm)

            if verbose:
                print(f"  Peak amplitude: {peak_amp:.4f}")
                print(f"  Shape error norm: {error_norm:.6f}")

            # 3. Check convergence
            if error_norm < self.tolerance:
                if verbose:
                    print(f"  Converged! Error below tolerance.")
                converged = True
                break

            # 4. Estimate Jacobian (using aligned windows)
            if verbose:
                print(f"  Estimating Jacobian...")
            K = self._estimate_jacobian_aligned(coeffs, y_window, window_ms)

            # 5. Compute correction (on normalized error)
            delta_f = self.compute_correction(K, error)

            if verbose:
                print(f"  Correction norm: {np.linalg.norm(delta_f):.4f}")

            # 6. Update coefficients
            coeffs = coeffs + self.alpha * delta_f
            coeffs_history.append(coeffs.copy())

            if verbose:
                coeffs_str = ', '.join([f'{c:.3f}' for c in coeffs])
                print(f"  New coeffs: [{coeffs_str}]")

        if verbose:
            print(f"\n{'='*60}")
            print("Optimization Complete")
            print(f"{'='*60}")
            print(f"Iterations: {len(error_history)}")
            print(f"Final error: {error_history[-1]:.6f}")
            print(f"Converged: {converged}")

        return OptimizationResult(
            final_coeffs=coeffs,
            final_error=error_history[-1],
            error_history=error_history,
            coeffs_history=coeffs_history,
            iterations=len(error_history),
            converged=converged
        )

    def _estimate_jacobian_aligned(
        self,
        coeffs: np.ndarray,
        y_current: np.ndarray,
        window_ms: float
    ) -> np.ndarray:
        """
        Estimate Jacobian using aligned windows for shape comparison.

        Args:
            coeffs: Current Fourier coefficients
            y_current: Current output window (aligned)
            window_ms: Window size in ms

        Returns:
            Jacobian matrix K
        """
        n_params = len(coeffs)
        n_output = len(y_current)

        # Get baseline peak for normalization
        baseline_peak = np.max(y_current) + 1e-10
        y_normalized = y_current / baseline_peak

        K = np.zeros((n_output, n_params))

        for j in range(n_params):
            # Create perturbation
            coeffs_perturbed = coeffs.copy()
            coeffs_perturbed[j] += self.epsilon

            # Measure perturbed output (aligned)
            y_perturbed, peak_perturbed = self.measure_aligned_window(coeffs_perturbed, window_ms)

            # Normalize for shape comparison
            y_perturbed_normalized = y_perturbed / (peak_perturbed + 1e-10)

            # Finite difference
            K[:, j] = (y_perturbed_normalized - y_normalized) / self.epsilon

            if (j + 1) % 2 == 0 or j == n_params - 1:
                print(f"    Jacobian columns {j+1}/{n_params} done")

        return K

    def plot_results(
        self,
        result: OptimizationResult,
        initial_coeffs: np.ndarray,
        save_path: str = "tangent_optimization_results.png"
    ):
        """
        Generate visualization of optimization results.

        Args:
            result: OptimizationResult from optimize()
            initial_coeffs: Initial coefficients for comparison
            save_path: Path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Error convergence
        axes[0, 0].semilogy(range(1, len(result.error_history) + 1), result.error_history, 'b.-')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Error Norm (log scale)')
        axes[0, 0].set_title('Convergence')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Coefficient evolution
        coeffs_array = np.array(result.coeffs_history)
        for i in range(coeffs_array.shape[1]):
            axes[0, 1].plot(range(len(result.coeffs_history)), coeffs_array[:, i],
                          label=f'c{i}' if i == 0 else f'a{(i+1)//2}' if i % 2 == 1 else f'b{i//2}')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Coefficient Value')
        axes[0, 1].set_title('Coefficient Evolution')
        axes[0, 1].legend(fontsize=8, ncol=2)
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Initial vs Final waveform (input)
        waveform_initial = self.generate_waveform(initial_coeffs)
        waveform_final = self.generate_waveform(result.final_coeffs)
        t_input = np.arange(self.n_points) / self.sample_rate_awg * 1000

        axes[1, 0].plot(t_input, waveform_initial, 'b-', alpha=0.5, lw=0.5, label='Initial')
        axes[1, 0].plot(t_input, waveform_final, 'r-', alpha=0.5, lw=0.5, label='Final')
        axes[1, 0].set_xlabel('Time (ms)')
        axes[1, 0].set_ylabel('Amplitude')
        axes[1, 0].set_title('Input Waveforms')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Output comparison with target
        y_initial = self.measure_output(initial_coeffs)
        y_final = self.measure_output(result.final_coeffs)

        axes[1, 1].plot(self.target_time_ms, y_initial, 'b-', alpha=0.7, label='Initial output')
        axes[1, 1].plot(self.target_time_ms, y_final, 'r-', alpha=0.7, label='Final output')
        axes[1, 1].plot(self.target_time_ms, self.target_envelope, 'g--', lw=2, label='Target')
        axes[1, 1].set_xlabel('Time (ms)')
        axes[1, 1].set_ylabel('Envelope')
        axes[1, 1].set_title('Output Envelopes vs Target')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Add session info
        session_time = datetime.now().strftime('%Y-%m-%d %H:%M')
        fig.suptitle(f'Tangent Optimization Results — {session_time}', fontsize=12)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"Saved: {save_path}")


    def detect_arrival_time(self, coeffs: np.ndarray, verbose: bool = True) -> float:
        """
        Auto-detect pulse arrival time from a measurement.

        Uses a longer recording to capture multiple pulses, finds the strongest
        peak that is above a noise threshold.

        Args:
            coeffs: Fourier coefficients for test waveform
            verbose: Print debug info

        Returns:
            Detected arrival time in ms
        """
        if verbose:
            print("  Auto-detecting arrival time...")

        # Use longer recording for detection
        old_duration = self.record_duration
        self.record_duration = 0.02  # 20ms to capture multiple pulses

        envelope = self.measure_output(coeffs)

        self.record_duration = old_duration

        # Find noise floor (use first 0.3ms which should be before any signal)
        noise_samples = int(0.3 * self.sample_rate_mic / 1000)
        noise_floor = np.std(envelope[:noise_samples]) if noise_samples > 10 else 0.001

        # Find peaks above 5x noise floor
        threshold = max(5 * noise_floor, 0.01 * np.max(envelope))

        # Find the maximum peak
        peak_idx = np.argmax(envelope)
        peak_val = envelope[peak_idx]
        peak_time_ms = peak_idx / self.sample_rate_mic * 1000

        if verbose:
            print(f"    Noise floor: {noise_floor:.6f}")
            print(f"    Peak value: {peak_val:.4f} at {peak_time_ms:.2f} ms")

        # Verify it's a real signal (peak should be well above noise)
        if peak_val < threshold:
            if verbose:
                print(f"    Warning: Peak below threshold, using default 2.0ms")
            return 2.0

        return peak_time_ms


def main():
    """Run tangent optimization experiment."""
    print("=" * 60)
    print("BATPING Tangent Optimization (Shape-Aligned)")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create optimizer with tuned parameters
    optimizer = TangentOptimizer(
        n_points=4000,
        carrier_freq=40000,
        sample_rate_awg=1e6,
        n_fourier_coeffs=8,
        epsilon=0.1,        # Perturbation size
        lambda_reg=0.1,     # Regularization
        alpha=0.3,          # Step size
        max_iterations=15,
        tolerance=1e-5,
        settling_time=0.15,
        record_duration=0.02,
    )

    # Initial coefficients (Gaussian-like envelope)
    initial_coeffs = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    try:
        optimizer.connect_hardware()

        # Run SHAPE-ALIGNED optimization (timing-invariant)
        result = optimizer.optimize_shape(
            initial_coeffs=initial_coeffs,
            target_sigma_cycles=2.0,
            window_ms=3.0,
            verbose=True
        )

        # Save results
        np.save('tangent_optimized_coeffs.npy', result.final_coeffs)
        print(f"\nSaved coefficients to tangent_optimized_coeffs.npy")

        # Generate plots
        optimizer.plot_results(result, initial_coeffs, 'tangent_optimization_results.png')

    finally:
        optimizer.disconnect_hardware()

    print("\nDone.")


if __name__ == "__main__":
    main()
