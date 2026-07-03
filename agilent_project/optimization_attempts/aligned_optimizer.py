"""
Aligned Waveform Optimizer

Fixes two critical issues from full_waveform_optimizer.py:
1. Timing jitter: Aligns all measurements to peak before Jacobian computation
2. Duty cycle: Constrains optimization to a pulse window (forces silence outside)

The key insight: we want to shape the pulse, not when it arrives. By extracting
a fixed-size window around each measurement's peak, we compare shapes rather
than absolute timing - making the Jacobian estimation robust to jitter.

Window structure:
    [silence] [active region: main pulse + compensation tail] [silence]

The active region is intentionally large enough to allow the optimizer to
discover inverse-filter-like compensation tails that cancel piezo ringdown.
The forced silence regions ensure discrete pulses for TOF measurement.

Example with 4000 samples @ 1 MSa/s (4ms period):
    - Silence: 0-500 (0.5ms) - pre-pulse quiet
    - Active: 500-3200 (2.7ms) - main pulse + room for ~1ms ringdown compensation
    - Silence: 3200-4000 (0.8ms) - post-pulse quiet for arrival detection
"""

import numpy as np
import pyaudio
import time
import os
from datetime import datetime
from scipy import signal
from scipy.signal import windows
from dataclasses import dataclass
from typing import Optional, Tuple, List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from agilent_33220a import Agilent33220A, generate_gaussian_pulse


@dataclass
class IterationState:
    """State for resumability."""
    iteration: int
    f_current: np.ndarray
    error_history: List[float]
    f_best: Optional[np.ndarray] = None
    best_error: float = float('inf')


class AlignedWaveformOptimizer:
    """
    Waveform optimizer with peak alignment and duty cycle constraints.

    Key differences from full_waveform_optimizer:
    1. Extracts fixed-size window around peak for all comparisons
    2. Only optimizes samples within pulse window; rest stays zero
    3. Smooth tapering at window edges to avoid discontinuities
    """

    def __init__(
        self,
        n_points: int = 4000,
        carrier_freq: float = 40000,
        sample_rate_awg: float = 1e6,
        sample_rate_mic: int = 384000,
        drive_amplitude: float = 5.0,
        # Window parameters - designed for ringdown compensation
        # Structure: [pre-silence][active: pulse + tail][post-silence]
        pre_silence_samples: int = 500,   # 0.5ms forced silence before pulse
        post_silence_samples: int = 800,  # 0.8ms forced silence after (for TOF)
        taper_samples: int = 50,          # Smooth edges to avoid spectral splatter
        # Optimization parameters
        epsilon: float = 0.05,
        lambda_reg: float = 0.01,
        alpha: float = 0.3,
        max_iterations: int = 20,
        tolerance: float = 1e-6,
        # Measurement parameters
        settling_time: float = 0.15,
        record_duration: float = 0.025,
        # Target parameters
        target_sigma_cycles: float = 2.0,
        # Subsampling for speed
        input_subsample: int = 5,   # Perturb every Nth sample in window
        output_downsample: int = 10,
        # Alignment window
        align_window_ms: float = 2.0,  # Window around peak for comparison
    ):
        self.n_points = n_points
        self.carrier_freq = carrier_freq
        self.sample_rate_awg = sample_rate_awg
        self.sample_rate_mic = sample_rate_mic
        self.drive_amplitude = drive_amplitude

        # Window structure: [pre-silence][active][post-silence]
        self.pre_silence_samples = pre_silence_samples
        self.post_silence_samples = post_silence_samples
        self.taper_samples = taper_samples

        # Derive active region (everything not silence)
        self.pulse_window_start = pre_silence_samples
        self.pulse_window_end = n_points - post_silence_samples
        self.pulse_window_size = self.pulse_window_end - self.pulse_window_start

        # Sanity check
        if self.pulse_window_size < 100:
            raise ValueError(f"Active region too small: {self.pulse_window_size} samples")

        # Calculate duty cycle for logging
        self.duty_cycle = self.pulse_window_size / n_points

        # Build taper window
        self.taper_window = self._build_taper_window()

        # Optimization
        self.epsilon = epsilon
        self.lambda_reg = lambda_reg
        self.alpha = alpha
        self.alpha_initial = alpha  # Store for adaptive scheduling
        self.alpha_min = 0.05       # Minimum step size
        self.alpha_decay = 0.7      # Decay factor when overshooting
        self.alpha_grow = 1.1       # Growth factor when improving
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Measurement
        self.settling_time = settling_time
        self.record_duration = record_duration
        self.target_sigma_cycles = target_sigma_cycles

        # Subsampling
        self.input_subsample = input_subsample
        self.output_downsample = output_downsample

        # Alignment
        self.align_window_samples = int(align_window_ms * sample_rate_mic / 1000)

        # Hardware
        self.awg: Optional[Agilent33220A] = None
        self.pa: Optional[pyaudio.PyAudio] = None
        self.mic_index: Optional[int] = None

        # Derived
        self.waveform_duration_s = n_points / sample_rate_awg
        self.rep_freq = 1.0 / self.waveform_duration_s

        # Precompute filter
        self.bandpass_low = 38000
        self.bandpass_high = 42000
        nyq = sample_rate_mic / 2
        low = self.bandpass_low / nyq
        high = min(self.bandpass_high / nyq, 0.99)
        self.filter_b, self.filter_a = signal.butter(4, [low, high], btype='band')

        # Number of perturbations (only within pulse window, subsampled)
        self.n_perturb = self.pulse_window_size // input_subsample

        # Output window size (downsampled)
        self.n_output = self.align_window_samples // output_downsample

    def _build_taper_window(self) -> np.ndarray:
        """Build smooth taper window for pulse edges."""
        window = np.ones(self.pulse_window_size)

        # Tukey-style taper at edges
        taper_in = np.linspace(0, 1, self.taper_samples)
        taper_in = 0.5 * (1 - np.cos(np.pi * taper_in))  # Hann rise

        taper_out = np.linspace(1, 0, self.taper_samples)
        taper_out = 0.5 * (1 + np.cos(np.pi * (1 - taper_out)))  # Hann fall

        window[:self.taper_samples] = taper_in
        window[-self.taper_samples:] = taper_out

        return window

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

    def extract_envelope(self, audio: np.ndarray) -> np.ndarray:
        """Extract envelope of 40kHz signal."""
        filtered = signal.filtfilt(self.filter_b, self.filter_a, audio)
        analytic = signal.hilbert(filtered)
        return np.abs(analytic)

    def extract_aligned_window(self, envelope: np.ndarray) -> Tuple[np.ndarray, int, float]:
        """
        Extract fixed-size window centered on peak.

        Returns:
            (window, peak_idx, peak_amplitude)
        """
        peak_idx = np.argmax(envelope)
        peak_amp = envelope[peak_idx]

        half_window = self.align_window_samples // 2

        # Handle edge cases
        start = max(0, peak_idx - half_window)
        end = min(len(envelope), peak_idx + half_window)

        window = envelope[start:end]

        # Pad if necessary
        if len(window) < self.align_window_samples:
            pad_left = max(0, half_window - peak_idx)
            pad_right = self.align_window_samples - len(window) - pad_left
            window = np.pad(window, (pad_left, max(0, pad_right)), mode='edge')

        # Downsample
        window = window[::self.output_downsample]

        return window[:self.n_output], peak_idx, peak_amp

    def build_waveform(self, pulse_samples: np.ndarray) -> np.ndarray:
        """
        Build full waveform from pulse samples.

        Args:
            pulse_samples: Samples for the active region (pulse_window_size,)

        Returns:
            Full waveform (n_points,) with zeros outside pulse window
        """
        waveform = np.zeros(self.n_points)

        # Apply taper and insert
        tapered = pulse_samples * self.taper_window
        waveform[self.pulse_window_start:self.pulse_window_end] = tapered

        return waveform

    def measure_aligned(self, pulse_samples: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Send waveform and return aligned output window.

        Args:
            pulse_samples: Active region samples

        Returns:
            (aligned window, peak amplitude)
        """
        waveform = self.build_waveform(pulse_samples)

        # Normalize
        max_abs = np.max(np.abs(waveform))
        if max_abs > 1.0:
            waveform = waveform / max_abs

        self.awg.output_arbitrary(waveform, freq_hz=self.rep_freq, vpp=self.drive_amplitude)
        time.sleep(self.settling_time)

        audio = self.record_audio()
        self.awg.set_output(False)

        envelope = self.extract_envelope(audio)
        window, _, peak_amp = self.extract_aligned_window(envelope)

        return window, peak_amp

    def generate_target(self, peak_amplitude: float) -> np.ndarray:
        """
        Generate target pulse shape (narrow Gaussian).

        The target is centered in the alignment window.
        """
        t = np.arange(self.n_output)
        t_center = self.n_output // 2

        # Sigma in downsampled output samples
        effective_rate = self.sample_rate_mic / self.output_downsample
        sigma_samples = self.target_sigma_cycles / self.carrier_freq * effective_rate

        target = peak_amplitude * np.exp(-0.5 * ((t - t_center) / sigma_samples) ** 2)

        return target

    def estimate_jacobian(
        self,
        pulse_samples: np.ndarray,
        y_current: np.ndarray,
        peak_amp: float
    ) -> np.ndarray:
        """
        Estimate Jacobian using aligned measurements.

        Only perturbs samples within the pulse window (subsampled).
        All measurements are aligned to peak before computing differences.
        """
        N = len(pulse_samples)  # pulse_window_size
        M = len(y_current)      # n_output (downsampled aligned window)

        # Indices to perturb
        perturb_indices = np.arange(0, N, self.input_subsample)
        n_perturb = len(perturb_indices)

        K_sub = np.zeros((M, n_perturb))

        start_time = time.time()
        print(f"    Jacobian: {M} x {n_perturb} ({M * n_perturb:,} elements)")

        for i, j in enumerate(perturb_indices):
            # Perturb
            pulse_perturbed = pulse_samples.copy()
            pulse_perturbed[j] += self.epsilon

            # Measure (aligned)
            y_perturbed, _ = self.measure_aligned(pulse_perturbed)

            # Finite difference
            K_sub[:, i] = (y_perturbed - y_current) / self.epsilon

            if (i + 1) % 20 == 0 or i == n_perturb - 1:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (n_perturb - i - 1) / rate if rate > 0 else 0
                print(f"    {i+1}/{n_perturb} ({100*(i+1)/n_perturb:.0f}%) - {rate:.1f}/s - ETA {eta/60:.1f}m")

        # Interpolate to full resolution
        K = np.zeros((M, N))
        for i in range(n_perturb):
            j_start = perturb_indices[i]
            j_end = perturb_indices[i + 1] if i < n_perturb - 1 else N

            if i < n_perturb - 1:
                for j in range(j_start, j_end):
                    alpha = (j - j_start) / (j_end - j_start)
                    K[:, j] = (1 - alpha) * K_sub[:, i] + alpha * K_sub[:, i + 1]
            else:
                K[:, j_start:j_end] = K_sub[:, i:i+1]

        return K

    def compute_correction(self, K: np.ndarray, error: np.ndarray) -> np.ndarray:
        """Regularized least-squares correction."""
        N = K.shape[1]

        KtK = K.T @ K
        Kt_e = K.T @ error

        A = KtK + self.lambda_reg * np.eye(N)
        b = -Kt_e

        delta_f = np.linalg.solve(A, b)

        return delta_f

    def create_initial_pulse(self) -> np.ndarray:
        """Create initial Gaussian pulse samples for the active window."""
        t = np.arange(self.pulse_window_size) / self.sample_rate_awg
        t_center = t[self.pulse_window_size // 2]
        sigma_t = 5 / self.carrier_freq  # 5 cycles

        envelope = np.exp(-0.5 * ((t - t_center) / sigma_t) ** 2)
        carrier = np.sin(2 * np.pi * self.carrier_freq * t)

        return envelope * carrier

    def save_state(self, state: IterationState, path: str = "aligned_optimizer_state.npz"):
        """Save state for resumability."""
        np.savez(
            path,
            iteration=state.iteration,
            f_current=state.f_current,
            error_history=np.array(state.error_history),
            f_best=state.f_best if state.f_best is not None else np.array([]),
            best_error=state.best_error,
        )
        print(f"  Saved state to {path}")

    def load_state(self, path: str = "aligned_optimizer_state.npz") -> Optional[IterationState]:
        """Load state if exists."""
        if not os.path.exists(path):
            return None

        data = np.load(path)
        f_best = data['f_best'] if data['f_best'].size > 0 else None

        return IterationState(
            iteration=int(data['iteration']),
            f_current=data['f_current'],
            error_history=list(data['error_history']),
            f_best=f_best,
            best_error=float(data['best_error']),
        )

    def optimize(
        self,
        pulse_initial: Optional[np.ndarray] = None,
        resume: bool = True,
        verbose: bool = True
    ) -> Tuple[np.ndarray, List[float], np.ndarray]:
        """
        Run aligned waveform optimization.

        Returns:
            (final_pulse, error_history, best_pulse)
        """
        if self.awg is None:
            raise RuntimeError("Hardware not connected")

        # Initialize or resume
        state = self.load_state() if resume else None

        if state is not None:
            print(f"Resuming from iteration {state.iteration + 1}")
            pulse = state.f_current.copy()
            error_history = state.error_history.copy()
            start_iter = state.iteration
            f_best = state.f_best.copy() if state.f_best is not None else None
            best_error = state.best_error
        else:
            pulse = pulse_initial if pulse_initial is not None else self.create_initial_pulse()
            error_history = []
            start_iter = 0
            f_best = None
            best_error = float('inf')

        # Adaptive step size state
        current_alpha = self.alpha
        prev_error = None

        if verbose:
            pre_ms = self.pre_silence_samples / self.sample_rate_awg * 1000
            post_ms = self.post_silence_samples / self.sample_rate_awg * 1000
            active_ms = self.pulse_window_size / self.sample_rate_awg * 1000
            print(f"\n{'='*60}")
            print("Aligned Waveform Optimization")
            print(f"{'='*60}")
            print(f"Window structure:")
            print(f"  Pre-silence:  {self.pre_silence_samples:4d} samples ({pre_ms:.1f}ms)")
            print(f"  Active:       {self.pulse_window_size:4d} samples ({active_ms:.1f}ms) [for pulse + ringdown compensation]")
            print(f"  Post-silence: {self.post_silence_samples:4d} samples ({post_ms:.1f}ms) [for TOF detection]")
            print(f"  Duty cycle:   {self.duty_cycle*100:.1f}%")
            print(f"Parameters: ε={self.epsilon}, λ={self.lambda_reg}, α={self.alpha}")
            print(f"Input subsample: {self.input_subsample} → {self.n_perturb} perturbations")
            print(f"Target σ: {self.target_sigma_cycles} cycles")
            est_time = (self.n_perturb + 1) * (self.settling_time + 0.1)
            print(f"Est. time/iteration: ~{est_time/60:.1f} min")

        for n in range(start_iter, self.max_iterations):
            iter_start = time.time()

            if verbose:
                print(f"\n{'='*60}")
                print(f"ITERATION {n + 1}/{self.max_iterations}")
                print(f"{'='*60}")

            # Measure current (aligned)
            y_current, peak_amp = self.measure_aligned(pulse)

            # Generate adaptive target (matches current peak amplitude)
            y_target = self.generate_target(peak_amp)

            # Error
            error = y_current - y_target
            error_norm = np.sqrt(np.mean(error**2))
            error_history.append(error_norm)

            if verbose:
                print(f"  Peak amplitude: {peak_amp:.4f}")
                print(f"  Error (RMSE): {error_norm:.6f}")

            # Track best
            if error_norm < best_error:
                best_error = error_norm
                f_best = pulse.copy()
                if verbose:
                    print(f"  ** New best! **")

            # Convergence check
            if error_norm < self.tolerance:
                if verbose:
                    print("  CONVERGED!")
                break

            # Estimate Jacobian
            if verbose:
                print(f"  Estimating Jacobian...")
            K = self.estimate_jacobian(pulse, y_current, peak_amp)

            # Compute correction
            delta_f = self.compute_correction(K, error)

            if verbose:
                print(f"  Correction norm: {np.linalg.norm(delta_f):.4f}")
                print(f"  Correction max: {np.max(np.abs(delta_f)):.4f}")

            # Adaptive step size
            if prev_error is not None:
                if error_norm > prev_error * 1.05:  # Error increased by >5%
                    current_alpha = max(self.alpha_min, current_alpha * self.alpha_decay)
                    if verbose:
                        print(f"  α reduced to {current_alpha:.3f} (error increased)")
                elif error_norm < prev_error * 0.98:  # Good improvement
                    current_alpha = min(self.alpha_initial, current_alpha * self.alpha_grow)
                    if verbose:
                        print(f"  α increased to {current_alpha:.3f} (good progress)")

            prev_error = error_norm

            # Update with adaptive step size
            pulse = pulse + current_alpha * delta_f

            # Clip to prevent runaway
            pulse = np.clip(pulse, -2.0, 2.0)

            # Save state
            state = IterationState(
                iteration=n + 1,
                f_current=pulse.copy(),
                error_history=error_history.copy(),
                f_best=f_best,
                best_error=best_error,
            )
            self.save_state(state)

            iter_time = time.time() - iter_start
            if verbose:
                print(f"  Iteration time: {iter_time/60:.1f} min")

        return pulse, error_history, f_best


def plot_results(
    optimizer: AlignedWaveformOptimizer,
    pulse_initial: np.ndarray,
    pulse_final: np.ndarray,
    error_history: List[float],
    save_path: str = "aligned_optimization_results.png"
):
    """Generate results plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Error convergence
    axes[0, 0].semilogy(range(1, len(error_history) + 1), error_history, 'b.-', markersize=8)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Error (RMSE)')
    axes[0, 0].set_title('Convergence')
    axes[0, 0].grid(True, alpha=0.3)

    # Initial waveform (full)
    waveform_initial = optimizer.build_waveform(pulse_initial)
    waveform_final = optimizer.build_waveform(pulse_final)
    t_ms = np.arange(optimizer.n_points) / optimizer.sample_rate_awg * 1000

    axes[0, 1].plot(t_ms, waveform_initial, 'b-', lw=0.5, alpha=0.7, label='Initial')
    axes[0, 1].plot(t_ms, waveform_final, 'r-', lw=0.5, alpha=0.7, label='Final')
    axes[0, 1].axvspan(
        optimizer.pulse_window_start / optimizer.sample_rate_awg * 1000,
        optimizer.pulse_window_end / optimizer.sample_rate_awg * 1000,
        alpha=0.1, color='green', label='Active window'
    )
    axes[0, 1].set_xlabel('Time (ms)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_title('Input Waveforms')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Pulse region comparison
    t_pulse = np.arange(optimizer.pulse_window_size) / optimizer.sample_rate_awg * 1000
    axes[1, 0].plot(t_pulse, pulse_initial, 'b-', lw=0.5, alpha=0.7, label='Initial')
    axes[1, 0].plot(t_pulse, pulse_final, 'r-', lw=0.5, alpha=0.7, label='Final')
    axes[1, 0].set_xlabel('Time within window (ms)')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].set_title('Pulse Region Detail')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Correction
    diff = pulse_final - pulse_initial
    axes[1, 1].plot(t_pulse, diff, 'g-', lw=0.5)
    axes[1, 1].set_xlabel('Time within window (ms)')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].set_title(f'Correction (max={np.max(np.abs(diff)):.3f})')
    axes[1, 1].grid(True, alpha=0.3)

    session_time = datetime.now().strftime('%Y-%m-%d %H:%M')
    fig.suptitle(f'Aligned Waveform Optimization — {session_time}', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    """Run aligned optimization."""
    print("=" * 60)
    print("BATPING Aligned Waveform Optimization")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Window structure for 4ms waveform (4000 samples @ 1 MSa/s):
    #   - Pre-silence: 0.5ms (500 samples) - quiet before pulse
    #   - Active: 2.7ms (2700 samples) - main pulse + room for 1ms+ compensation tail
    #   - Post-silence: 0.8ms (800 samples) - quiet for TOF arrival detection
    #
    # The active region is 67.5% duty cycle, but this is the MAXIMUM the optimizer
    # can use. In practice it will find a compact solution if that's optimal.
    # The key constraint is the forced silence, not the active region size.

    optimizer = AlignedWaveformOptimizer(
        n_points=4000,
        carrier_freq=40000,
        sample_rate_awg=1e6,
        drive_amplitude=5.0,
        # Window structure
        pre_silence_samples=500,   # 0.5ms before pulse
        post_silence_samples=800,  # 0.8ms after (ensures discrete arrival)
        taper_samples=50,
        # Optimization
        epsilon=0.05,
        lambda_reg=0.01,
        alpha=0.3,
        max_iterations=15,
        tolerance=1e-6,
        # Measurement
        settling_time=0.15,
        record_duration=0.025,
        target_sigma_cycles=2.0,
        # Subsampling - more aggressive since active region is larger
        input_subsample=10,  # ~270 perturbations
        output_downsample=10,
        align_window_ms=2.0,
    )

    pulse_initial = optimizer.create_initial_pulse()

    try:
        optimizer.connect_hardware()

        pulse_final, error_history, pulse_best = optimizer.optimize(
            pulse_initial=pulse_initial,
            resume=True,
            verbose=True,
        )

        # Save results
        np.save('aligned_pulse_final.npy', pulse_final)
        np.save('aligned_pulse_best.npy', pulse_best if pulse_best is not None else pulse_final)
        print(f"\nSaved: aligned_pulse_final.npy, aligned_pulse_best.npy")

        # Plot
        plot_results(
            optimizer,
            pulse_initial,
            pulse_best if pulse_best is not None else pulse_final,
            error_history,
            'aligned_optimization_results.png',
        )

    finally:
        optimizer.disconnect_hardware()

    print("\nOptimization complete.")


if __name__ == "__main__":
    main()
