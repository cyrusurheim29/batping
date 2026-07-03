"""
Full Waveform Iterative Local Tangent Estimation

Implements the optimization method from the proposal exactly:
- f ∈ ℝ^N is the raw waveform (N=4000 samples)
- y_target is adaptive: centered at argmax(y^(n)) each iteration
- Jacobian estimated by perturbing each sample f_j

This is computationally expensive (~7 min per iteration) but faithful to the method.
"""

import numpy as np
import pyaudio
import time
import os
from datetime import datetime
from scipy import signal
from dataclasses import dataclass
from typing import Optional, Tuple, List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from agilent_33220a import Agilent33220A, generate_gaussian_pulse


@dataclass
class IterationState:
    """State that can be saved/restored between iterations."""
    iteration: int
    f_current: np.ndarray
    error_history: List[float]
    f_history: List[np.ndarray]
    f_best: Optional[np.ndarray] = None
    best_error: float = float('inf')


class FullWaveformOptimizer:
    """
    Full waveform tangent optimization.

    Perturbs each of the N input samples to estimate the Jacobian,
    then computes the optimal correction via regularized least-squares.
    """

    def __init__(
        self,
        n_points: int = 4000,
        carrier_freq: float = 40000,
        sample_rate_awg: float = 1e6,
        sample_rate_mic: int = 384000,
        drive_amplitude: float = 5.0,
        epsilon: float = 0.05,
        lambda_reg: float = 0.01,
        alpha: float = 0.5,
        max_iterations: int = 20,
        tolerance: float = 1e-6,
        settling_time: float = 0.1,
        record_duration: float = 0.015,
        target_sigma_cycles: float = 1.5,
    ):
        self.n_points = n_points
        self.carrier_freq = carrier_freq
        self.sample_rate_awg = sample_rate_awg
        self.sample_rate_mic = sample_rate_mic
        self.drive_amplitude = drive_amplitude
        self.epsilon = epsilon
        self.lambda_reg = lambda_reg
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.settling_time = settling_time
        self.record_duration = record_duration
        self.target_sigma_cycles = target_sigma_cycles

        # Hardware
        self.awg: Optional[Agilent33220A] = None
        self.pa: Optional[pyaudio.PyAudio] = None
        self.mic_index: Optional[int] = None

        # Derived
        self.waveform_duration_s = n_points / sample_rate_awg
        self.rep_freq = 1.0 / self.waveform_duration_s
        self.n_output = int(record_duration * sample_rate_mic)

        # Bandpass - precompute filter coefficients once
        self.bandpass_low = 38000
        self.bandpass_high = 42000
        nyq = sample_rate_mic / 2
        low = self.bandpass_low / nyq
        high = min(self.bandpass_high / nyq, 0.99)
        self.filter_b, self.filter_a = signal.butter(4, [low, high], btype='band')

        # Downsample output for faster matrix operations
        self.output_downsample = 10  # Keep every 10th sample
        self.n_output_downsampled = self.n_output // self.output_downsample

        # Subsample input perturbations (key speedup!)
        # Instead of perturbing all 4000 samples, perturb every Kth sample
        self.input_subsample = 10  # Perturb every 10th input sample
        self.n_perturb = n_points // self.input_subsample  # 400 perturbations

    def connect_hardware(self):
        """Connect to AWG and microphone."""
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
        return audio.astype(np.float64) / 32768.0

    def extract_envelope(self, audio: np.ndarray, downsample: bool = True) -> np.ndarray:
        """Extract envelope of 40kHz signal via bandpass + Hilbert.

        Uses pre-computed filter coefficients for speed.
        Optionally downsamples for faster matrix operations.
        """
        filtered = signal.filtfilt(self.filter_b, self.filter_a, audio)
        analytic = signal.hilbert(filtered)
        envelope = np.abs(analytic)

        if downsample:
            return envelope[::self.output_downsample]
        return envelope

    def measure_output(self, f: np.ndarray) -> np.ndarray:
        """
        Send waveform f and measure acoustic output.

        Args:
            f: Input waveform (N samples, will be normalized to [-1,1])

        Returns:
            Output envelope (M samples)
        """
        # Normalize waveform to [-1, 1] for AWG
        f_normalized = f.copy()
        max_abs = np.max(np.abs(f_normalized))
        if max_abs > 1.0:
            f_normalized = f_normalized / max_abs

        self.awg.output_arbitrary(f_normalized, freq_hz=self.rep_freq, vpp=self.drive_amplitude)
        time.sleep(self.settling_time)

        audio = self.record_audio()
        self.awg.set_output(False)

        envelope = self.extract_envelope(audio)
        return envelope

    def generate_adaptive_target(self, y: np.ndarray) -> np.ndarray:
        """
        Generate y_target centered at the peak of y.

        This is the key insight: y_target adapts to meet y where it is,
        so we're always comparing shapes at the correct alignment.

        Args:
            y: Current measured output (possibly downsampled)

        Returns:
            y_target: Narrow Gaussian centered at argmax(y), matched amplitude
        """
        peak_idx = np.argmax(y)
        peak_amp = y[peak_idx]

        # Effective sample rate (accounting for downsampling)
        effective_rate = self.sample_rate_mic / self.output_downsample

        # Time array
        t = np.arange(len(y)) / effective_rate
        t_peak = peak_idx / effective_rate

        # Narrow Gaussian centered at peak
        sigma_s = self.target_sigma_cycles / self.carrier_freq
        y_target = peak_amp * np.exp(-0.5 * ((t - t_peak) / sigma_s) ** 2)

        return y_target

    def estimate_jacobian(
        self,
        f: np.ndarray,
        y_current: np.ndarray,
        progress_callback=None
    ) -> np.ndarray:
        """
        Estimate Jacobian by perturbing subsampled input points.

        Instead of perturbing all N samples, we perturb every Kth sample
        and interpolate the Jacobian for intermediate points.

        Args:
            f: Current input waveform (N samples)
            y_current: Current output (M samples, downsampled)
            progress_callback: Optional callback

        Returns:
            K: Jacobian matrix (M x N), with interpolated columns
        """
        N = len(f)
        M = len(y_current)

        # Indices we'll actually perturb
        perturb_indices = np.arange(0, N, self.input_subsample)
        n_perturb = len(perturb_indices)

        # Subsampled Jacobian
        K_sub = np.zeros((M, n_perturb))

        start_time = time.time()
        print(f"    Subsampled Jacobian: {M} x {n_perturb} = {M*n_perturb:,} elements")
        print(f"    Perturbing every {self.input_subsample}th sample ({n_perturb} measurements)")

        for i, j in enumerate(perturb_indices):
            # Perturb sample j
            f_perturbed = f.copy()
            f_perturbed[j] += self.epsilon

            # Measure
            y_perturbed = self.measure_output(f_perturbed)

            # Finite difference
            K_sub[:, i] = (y_perturbed - y_current) / self.epsilon

            # Progress every 20 perturbations
            if (i + 1) % 20 == 0 or i == n_perturb - 1:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (n_perturb - i - 1) / rate if rate > 0 else 0
                print(f"    Jacobian: {i+1}/{n_perturb} ({100*(i+1)/n_perturb:.1f}%) - {rate:.1f} meas/s - ETA: {eta/60:.1f} min", flush=True)

        # Interpolate to full Jacobian
        print(f"    Interpolating to full {M}x{N} Jacobian...", flush=True)
        K = np.zeros((M, N))
        for i in range(n_perturb):
            j_start = perturb_indices[i]
            j_end = perturb_indices[i + 1] if i < n_perturb - 1 else N

            # Linear interpolation between sampled columns
            if i < n_perturb - 1:
                for j in range(j_start, j_end):
                    alpha = (j - j_start) / (j_end - j_start)
                    K[:, j] = (1 - alpha) * K_sub[:, i] + alpha * K_sub[:, i + 1]
            else:
                # Last segment: extrapolate or just use last value
                for j in range(j_start, j_end):
                    K[:, j] = K_sub[:, i]

        return K

    def compute_correction(self, K: np.ndarray, error: np.ndarray) -> np.ndarray:
        """
        Compute optimal correction via regularized least-squares.

        Solves: (K^T K + λI) Δf = -K^T e

        Args:
            K: Jacobian (M x N)
            error: y - y_target (M,)

        Returns:
            Δf: Correction vector (N,)
        """
        N = K.shape[1]

        print(f"    Computing K^T @ K ({N}x{N})...", flush=True)
        t0 = time.time()
        KtK = K.T @ K
        print(f"    K^T @ K done in {time.time()-t0:.1f}s", flush=True)

        print(f"    Computing K^T @ error...", flush=True)
        Kt_e = K.T @ error

        print(f"    Solving {N}x{N} linear system...", flush=True)
        t0 = time.time()
        A = KtK + self.lambda_reg * np.eye(N)
        b = -Kt_e
        delta_f = np.linalg.solve(A, b)
        print(f"    Solve done in {time.time()-t0:.1f}s", flush=True)

        return delta_f

    def save_state(self, state: IterationState, path: str = "optimizer_state.npz"):
        """Save current state for resumability."""
        save_dict = {
            'iteration': state.iteration,
            'f_current': state.f_current,
            'error_history': np.array(state.error_history),
        }
        if state.f_best is not None:
            save_dict['f_best'] = state.f_best
            save_dict['best_error'] = state.best_error
        np.savez(path, **save_dict)
        print(f"  State saved to {path}")

    def load_state(self, path: str = "optimizer_state.npz") -> Optional[IterationState]:
        """Load state if exists."""
        if not os.path.exists(path):
            return None
        data = np.load(path)
        f_best = data['f_best'] if 'f_best' in data.files else None
        best_error = float(data['best_error']) if 'best_error' in data.files else float('inf')
        return IterationState(
            iteration=int(data['iteration']),
            f_current=data['f_current'],
            error_history=list(data['error_history']),
            f_history=[],
            f_best=f_best,
            best_error=best_error,
        )

    def optimize(
        self,
        f_initial: np.ndarray,
        resume: bool = True,
        verbose: bool = True
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Run full waveform optimization.

        Args:
            f_initial: Initial input waveform (N samples)
            resume: Try to resume from saved state
            verbose: Print progress

        Returns:
            Tuple of (optimized waveform, error history)
        """
        if self.awg is None:
            raise RuntimeError("Hardware not connected")

        # Try to resume
        state = None
        if resume:
            state = self.load_state()
            if state is not None:
                print(f"Resuming from iteration {state.iteration + 1}")

        if state is None:
            state = IterationState(
                iteration=0,
                f_current=f_initial.copy(),
                error_history=[],
                f_history=[f_initial.copy()],
                f_best=None,
                best_error=float('inf'),
            )

        f = state.f_current.copy()
        error_history = state.error_history.copy()
        start_iter = state.iteration
        f_best = state.f_best.copy() if state.f_best is not None else None
        best_error = state.best_error

        if verbose:
            print(f"\n{'='*60}")
            print("Full Waveform Tangent Optimization")
            print(f"{'='*60}")
            print(f"Input samples: {self.n_points}")
            print(f"Output samples: {self.n_output}")
            print(f"Parameters: ε={self.epsilon}, λ={self.lambda_reg}, α={self.alpha}")
            print(f"Target σ: {self.target_sigma_cycles} cycles")
            print(f"Input subsample: every {self.input_subsample}th sample")
            print(f"Measurements per iteration: {self.n_perturb + 1}")
            print(f"Estimated time per iteration: ~{(self.n_perturb + 1) * 1.0 / 60:.1f} min")

        for n in range(start_iter, self.max_iterations):
            iter_start = time.time()

            if verbose:
                print(f"\n{'='*60}")
                print(f"ITERATION {n + 1}/{self.max_iterations}")
                print(f"{'='*60}")

            # 1. Measure current output
            if verbose:
                print("  Measuring current output...")
            y = self.measure_output(f)

            # 2. Generate adaptive target (centered at peak of y)
            y_target = self.generate_adaptive_target(y)

            # 3. Compute error
            error = y - y_target
            error_norm = np.sqrt(np.mean(error**2))  # RMSE
            error_history.append(error_norm)

            # Track best waveform (before update, using f that produced this error)
            if error_norm < best_error:
                best_error = error_norm
                f_best = f.copy()
                if verbose:
                    print(f"  ** New best! **")

            if verbose:
                peak_idx = np.argmax(y)
                print(f"  Peak at sample {peak_idx} ({peak_idx/self.sample_rate_mic*1000:.2f} ms)")
                print(f"  Peak amplitude: {y[peak_idx]:.4f}")
                print(f"  Error (RMSE): {error_norm:.6f}")

            # 4. Check convergence
            if error_norm < self.tolerance:
                if verbose:
                    print(f"  CONVERGED! Error below tolerance.")
                break

            # 5. Estimate Jacobian (the expensive part)
            if verbose:
                print(f"  Estimating Jacobian ({self.n_points} perturbations)...")
            K = self.estimate_jacobian(f, y)

            # 6. Compute correction
            if verbose:
                print("  Computing correction...")
            delta_f = self.compute_correction(K, error)

            if verbose:
                print(f"  Correction norm: {np.linalg.norm(delta_f):.4f}")
                print(f"  Correction max: {np.max(np.abs(delta_f)):.4f}")

            # 7. Update waveform
            f = f + self.alpha * delta_f

            # 8. Enforce constraints: clip to [-1, 1]
            f = np.clip(f, -1.0, 1.0)

            # Save state with best waveform
            state = IterationState(
                iteration=n + 1,
                f_current=f.copy(),
                error_history=error_history.copy(),
                f_history=[],
                f_best=f_best,
                best_error=best_error,
            )
            self.save_state(state)

            iter_time = time.time() - iter_start
            if verbose:
                print(f"  Iteration time: {iter_time/60:.1f} min")

        # Save best waveform separately
        if f_best is not None:
            np.save('best_waveform.npy', f_best)
            if verbose:
                print(f"\nBest waveform saved to best_waveform.npy (error={best_error:.4f})")

        return f, error_history, f_best


def create_initial_waveform(n_points: int, carrier_freq: float, sample_rate: float) -> np.ndarray:
    """Create initial Gaussian pulse waveform."""
    return generate_gaussian_pulse(n_points, carrier_freq, sample_rate, sigma_cycles=5)


def plot_results(
    f_initial: np.ndarray,
    f_final: np.ndarray,
    error_history: List[float],
    sample_rate_awg: float,
    save_path: str = "full_waveform_results.png"
):
    """Generate results plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Error convergence
    axes[0, 0].semilogy(range(1, len(error_history) + 1), error_history, 'b.-', markersize=8)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Error (RMSE)')
    axes[0, 0].set_title('Convergence')
    axes[0, 0].grid(True, alpha=0.3)

    # Initial waveform
    t_ms = np.arange(len(f_initial)) / sample_rate_awg * 1000
    axes[0, 1].plot(t_ms, f_initial, 'b-', lw=0.5)
    axes[0, 1].set_xlabel('Time (ms)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_title('Initial Waveform')
    axes[0, 1].grid(True, alpha=0.3)

    # Final waveform
    axes[1, 0].plot(t_ms, f_final, 'r-', lw=0.5)
    axes[1, 0].set_xlabel('Time (ms)')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].set_title('Optimized Waveform')
    axes[1, 0].grid(True, alpha=0.3)

    # Difference
    diff = f_final - f_initial
    axes[1, 1].plot(t_ms, diff, 'g-', lw=0.5)
    axes[1, 1].set_xlabel('Time (ms)')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].set_title(f'Correction (max={np.max(np.abs(diff)):.3f})')
    axes[1, 1].grid(True, alpha=0.3)

    session_time = datetime.now().strftime('%Y-%m-%d %H:%M')
    fig.suptitle(f'Full Waveform Optimization — {session_time}', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    """Run full waveform optimization."""
    print("=" * 60)
    print("BATPING Full Waveform Tangent Optimization")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nWARNING: This will take ~7 minutes per iteration!")
    print("State is saved after each iteration for resumability.\n")

    optimizer = FullWaveformOptimizer(
        n_points=4000,
        carrier_freq=40000,
        sample_rate_awg=1e6,
        epsilon=0.05,
        lambda_reg=0.01,
        alpha=0.5,
        max_iterations=10,
        tolerance=1e-6,
        settling_time=0.15,
        record_duration=0.025,  # 25ms to capture multiple pulses
        target_sigma_cycles=1.5,
    )

    # Initial waveform: Gaussian pulse
    f_initial = create_initial_waveform(
        optimizer.n_points,
        optimizer.carrier_freq,
        optimizer.sample_rate_awg
    )

    try:
        optimizer.connect_hardware()

        f_optimized, error_history = optimizer.optimize(
            f_initial=f_initial,
            resume=True,
            verbose=True
        )

        # Save final waveform
        np.save('optimized_waveform_full.npy', f_optimized)
        print(f"\nSaved optimized waveform to optimized_waveform_full.npy")

        # Plot results
        plot_results(
            f_initial, f_optimized, error_history,
            optimizer.sample_rate_awg,
            'full_waveform_results.png'
        )

    finally:
        optimizer.disconnect_hardware()

    print("\nOptimization complete.")


if __name__ == "__main__":
    main()
