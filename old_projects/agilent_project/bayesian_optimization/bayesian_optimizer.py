"""
Bayesian Optimization for Piezo Pulse Shaping

Uses Gaussian Process surrogate to optimize FWHM directly,
avoiding the pitfalls of the Jacobian-based L2 approach.

Key differences from aligned_optimizer.py:
1. Low-dimensional parameterization (~10 params) instead of all samples
2. Direct FWHM objective (what we actually care about)
3. GP learns global model - each measurement informs everywhere
4. Expected Improvement acquisition balances exploration/exploitation
"""

import numpy as np
import pyaudio
import time
from datetime import datetime
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real
from skopt.plots import plot_convergence, plot_objective
from typing import Optional, List, Tuple
from dataclasses import dataclass

from agilent_33220a import Agilent33220A


@dataclass
class OptimizationResult:
    """Results from Bayesian optimization."""
    best_params: np.ndarray
    best_fwhm: float
    best_amplitude: float
    best_waveform: np.ndarray
    all_fwhms: List[float]
    all_amplitudes: List[float]
    all_params: List[np.ndarray]
    baseline_fwhm: float
    baseline_amplitude: float


class BayesianPulseOptimizer:
    """
    Bayesian optimization for piezo pulse shaping.

    Parameterizes the waveform with ~10 physically meaningful parameters
    and uses GP-based optimization to minimize FWHM directly.
    """

    def __init__(
        self,
        n_points: int = 4000,
        carrier_freq: float = 40000,
        sample_rate_awg: float = 1e6,
        sample_rate_mic: int = 384000,
        drive_amplitude: float = 5.0,
        settling_time: float = 0.2,
        record_duration: float = 0.03,
        # Waveform structure
        pre_silence_samples: int = 500,
        post_silence_samples: int = 800,
        # Amplitude constraint
        min_amplitude_fraction: float = 0.7,
        amplitude_penalty_weight: float = 1000.0,
    ):
        self.n_points = n_points
        self.carrier_freq = carrier_freq
        self.sample_rate_awg = sample_rate_awg
        self.sample_rate_mic = sample_rate_mic
        self.drive_amplitude = drive_amplitude
        self.settling_time = settling_time
        self.record_duration = record_duration

        self.pre_silence = pre_silence_samples
        self.post_silence = post_silence_samples
        self.active_start = pre_silence_samples
        self.active_end = n_points - post_silence_samples
        self.active_size = self.active_end - self.active_start

        self.min_amp_frac = min_amplitude_fraction
        self.amp_penalty = amplitude_penalty_weight

        # Hardware
        self.awg: Optional[Agilent33220A] = None
        self.pa: Optional[pyaudio.PyAudio] = None
        self.mic_index: Optional[int] = None

        # Derived
        self.rep_freq = 1.0 / (n_points / sample_rate_awg)

        # Bandpass filter for envelope extraction
        nyq = sample_rate_mic / 2
        low = 38000 / nyq
        high = min(42000 / nyq, 0.99)
        self.filter_b, self.filter_a = signal.butter(4, [low, high], btype='band')

        # Tracking
        self.baseline_amplitude: Optional[float] = None
        self.eval_count = 0
        self.all_fwhms: List[float] = []
        self.all_amplitudes: List[float] = []
        self.all_params: List[np.ndarray] = []
        self.best_fwhm = float('inf')
        self.best_params: Optional[np.ndarray] = None
        self.best_waveform: Optional[np.ndarray] = None
        self.best_amplitude: Optional[float] = None

    def connect(self):
        """Connect to hardware."""
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

    def disconnect(self):
        """Disconnect from hardware."""
        if self.awg:
            self.awg.set_output(False)
            self.awg.disconnect()
        if self.pa:
            self.pa.terminate()

    def record(self) -> np.ndarray:
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
        """Extract envelope via bandpass + Hilbert."""
        filtered = signal.filtfilt(self.filter_b, self.filter_a, audio)
        analytic = signal.hilbert(filtered)
        return np.abs(analytic)

    def measure_fwhm(self, envelope: np.ndarray) -> Tuple[float, float]:
        """
        Measure FWHM and peak amplitude from envelope.

        Returns:
            (fwhm_us, peak_amplitude)
        """
        peak_idx = np.argmax(envelope)
        peak_amp = envelope[peak_idx]
        half_max = peak_amp / 2

        # Find left crossing
        left_idx = peak_idx
        for i in range(peak_idx, -1, -1):
            if envelope[i] < half_max:
                # Linear interpolation for sub-sample precision
                if i < peak_idx - 1:
                    frac = (half_max - envelope[i]) / (envelope[i+1] - envelope[i])
                    left_idx = i + frac
                else:
                    left_idx = i
                break

        # Find right crossing
        right_idx = peak_idx
        for i in range(peak_idx, len(envelope)):
            if envelope[i] < half_max:
                if i > peak_idx + 1:
                    frac = (half_max - envelope[i-1]) / (envelope[i] - envelope[i-1])
                    right_idx = i - 1 + frac
                else:
                    right_idx = i
                break

        fwhm_samples = right_idx - left_idx
        fwhm_us = fwhm_samples / self.sample_rate_mic * 1e6

        return fwhm_us, peak_amp

    def params_to_waveform(self, params: np.ndarray) -> np.ndarray:
        """
        Convert parameter vector to full waveform.

        Parameters (10 total):
            [0] sigma: Gaussian envelope width in cycles (1-15)
            [1] phase: Carrier phase in radians (0-2π)
            [2-4] pre_comp[3]: Pre-pulse compensation amplitudes (-0.5 to 0.5)
            [5-9] post_comp[5]: Post-pulse compensation amplitudes (-0.5 to 0.5)
        """
        sigma_cycles = params[0]
        phase = params[1]
        pre_comp = params[2:5]
        post_comp = params[5:10]

        # Time arrays
        t_awg = np.arange(self.n_points) / self.sample_rate_awg
        t_center = t_awg[self.n_points // 2]

        # Main Gaussian pulse
        sigma_t = sigma_cycles / self.carrier_freq
        envelope = np.exp(-0.5 * ((t_awg - t_center) / sigma_t) ** 2)
        carrier = np.sin(2 * np.pi * self.carrier_freq * t_awg + phase)
        main_pulse = envelope * carrier

        # Pre-pulse compensation: 3 half-cycle bursts before main pulse
        pre_pulse = np.zeros(self.n_points)
        burst_duration = 0.5 / self.carrier_freq  # Half cycle
        burst_samples = int(burst_duration * self.sample_rate_awg)

        for i, amp in enumerate(pre_comp):
            # Place bursts before the main pulse center
            start = self.n_points // 2 - int((3-i) * 2 * burst_samples) - burst_samples
            if start >= 0 and start + burst_samples < self.n_points:
                t_burst = np.arange(burst_samples) / self.sample_rate_awg
                pre_pulse[start:start+burst_samples] = amp * np.sin(2 * np.pi * self.carrier_freq * t_burst)

        # Post-pulse compensation: 5 decaying bursts after main pulse
        post_pulse = np.zeros(self.n_points)
        for i, amp in enumerate(post_comp):
            start = self.n_points // 2 + int((i+1) * 2 * burst_samples)
            if start >= 0 and start + burst_samples < self.n_points:
                t_burst = np.arange(burst_samples) / self.sample_rate_awg
                # Apply decay envelope to post-compensation
                decay = np.exp(-i * 0.3)
                post_pulse[start:start+burst_samples] = amp * decay * np.sin(2 * np.pi * self.carrier_freq * t_burst)

        # Combine
        waveform = main_pulse + pre_pulse + post_pulse

        # Apply window: zero outside active region
        full_waveform = np.zeros(self.n_points)
        full_waveform[self.active_start:self.active_end] = waveform[self.active_start:self.active_end]

        # Taper edges
        taper_len = 50
        taper_in = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, taper_len)))
        taper_out = 0.5 * (1 + np.cos(np.pi * np.linspace(0, 1, taper_len)))
        full_waveform[self.active_start:self.active_start+taper_len] *= taper_in
        full_waveform[self.active_end-taper_len:self.active_end] *= taper_out

        # Normalize to [-1, 1]
        max_abs = np.max(np.abs(full_waveform))
        if max_abs > 0:
            full_waveform = full_waveform / max_abs

        return full_waveform

    def measure_waveform(self, waveform: np.ndarray) -> Tuple[float, float]:
        """
        Send waveform and measure FWHM + amplitude.

        Returns:
            (fwhm_us, amplitude)
        """
        self.awg.output_arbitrary(waveform, freq_hz=self.rep_freq, vpp=self.drive_amplitude)
        time.sleep(self.settling_time)

        audio = self.record()
        envelope = self.extract_envelope(audio)
        fwhm_us, amplitude = self.measure_fwhm(envelope)

        return fwhm_us, amplitude

    def objective(self, params: List[float]) -> float:
        """
        Objective function for optimization.

        Minimizes FWHM with penalty for low amplitude.
        """
        params = np.array(params)
        self.eval_count += 1

        # Build waveform
        waveform = self.params_to_waveform(params)

        # Measure
        fwhm_us, amplitude = self.measure_waveform(waveform)

        # Amplitude penalty
        if self.baseline_amplitude is not None:
            amp_threshold = self.min_amp_frac * self.baseline_amplitude
            if amplitude < amp_threshold:
                penalty = self.amp_penalty * (amp_threshold - amplitude) ** 2
            else:
                penalty = 0.0
        else:
            penalty = 0.0

        cost = fwhm_us + penalty

        # Track
        self.all_fwhms.append(fwhm_us)
        self.all_amplitudes.append(amplitude)
        self.all_params.append(params.copy())

        # Update best
        if fwhm_us < self.best_fwhm and (self.baseline_amplitude is None or
            amplitude >= self.min_amp_frac * self.baseline_amplitude):
            self.best_fwhm = fwhm_us
            self.best_params = params.copy()
            self.best_waveform = waveform.copy()
            self.best_amplitude = amplitude
            print(f"  [{self.eval_count}] NEW BEST: FWHM={fwhm_us:.1f}μs, amp={amplitude:.4f}")
        else:
            print(f"  [{self.eval_count}] FWHM={fwhm_us:.1f}μs, amp={amplitude:.4f}, cost={cost:.1f}")

        return cost

    def measure_baseline(self, sigma_cycles: float = 5.0) -> Tuple[float, float]:
        """Measure baseline with standard Gaussian pulse."""
        print("\nMeasuring baseline...")

        # Standard Gaussian parameters
        baseline_params = np.array([
            sigma_cycles,  # sigma
            0.0,           # phase
            0.0, 0.0, 0.0, # pre_comp
            0.0, 0.0, 0.0, 0.0, 0.0,  # post_comp
        ])

        waveform = self.params_to_waveform(baseline_params)

        # Average multiple measurements
        fwhms = []
        amps = []
        for i in range(3):
            fwhm, amp = self.measure_waveform(waveform)
            fwhms.append(fwhm)
            amps.append(amp)
            print(f"  Baseline {i+1}/3: FWHM={fwhm:.1f}μs, amp={amp:.4f}")

        baseline_fwhm = np.mean(fwhms)
        baseline_amp = np.mean(amps)

        print(f"Baseline: FWHM={baseline_fwhm:.1f}μs, amplitude={baseline_amp:.4f}")

        self.baseline_amplitude = baseline_amp
        return baseline_fwhm, baseline_amp

    def optimize(
        self,
        n_calls: int = 100,
        n_initial_points: int = 20,
        sigma_range: Tuple[float, float] = (1.0, 15.0),
    ) -> OptimizationResult:
        """
        Run Bayesian optimization.

        Args:
            n_calls: Total number of objective evaluations
            n_initial_points: Random samples before GP kicks in
            sigma_range: Bounds for envelope width parameter
        """
        print("\n" + "=" * 60)
        print("Bayesian Optimization for Pulse Shaping")
        print("=" * 60)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Evaluations: {n_initial_points} initial + {n_calls - n_initial_points} GP-guided")
        print(f"Parameters: 10 (sigma, phase, 3 pre-comp, 5 post-comp)")

        # Reset tracking
        self.eval_count = 0
        self.all_fwhms = []
        self.all_amplitudes = []
        self.all_params = []
        self.best_fwhm = float('inf')
        self.best_params = None
        self.best_waveform = None
        self.best_amplitude = None

        # Measure baseline
        baseline_fwhm, baseline_amp = self.measure_baseline()

        # Define parameter space
        space = [
            Real(sigma_range[0], sigma_range[1], name='sigma'),
            Real(0.0, 2*np.pi, name='phase'),
            Real(-0.5, 0.5, name='pre_comp_0'),
            Real(-0.5, 0.5, name='pre_comp_1'),
            Real(-0.5, 0.5, name='pre_comp_2'),
            Real(-0.5, 0.5, name='post_comp_0'),
            Real(-0.5, 0.5, name='post_comp_1'),
            Real(-0.5, 0.5, name='post_comp_2'),
            Real(-0.5, 0.5, name='post_comp_3'),
            Real(-0.5, 0.5, name='post_comp_4'),
        ]

        print(f"\nStarting optimization...")
        print("-" * 60)

        t0 = time.time()

        result = gp_minimize(
            self.objective,
            space,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            acq_func='EI',
            noise=1e-10,
            random_state=42,
            verbose=False,
        )

        elapsed = time.time() - t0

        print("-" * 60)
        print(f"\nOptimization complete in {elapsed/60:.1f} minutes")
        print(f"Best FWHM: {self.best_fwhm:.1f}μs (baseline: {baseline_fwhm:.1f}μs)")
        print(f"Improvement: {(1 - self.best_fwhm/baseline_fwhm)*100:.1f}%")
        print(f"Best amplitude: {self.best_amplitude:.4f} (baseline: {baseline_amp:.4f})")

        # Turn off output
        self.awg.set_output(False)

        return OptimizationResult(
            best_params=self.best_params,
            best_fwhm=self.best_fwhm,
            best_amplitude=self.best_amplitude,
            best_waveform=self.best_waveform,
            all_fwhms=self.all_fwhms,
            all_amplitudes=self.all_amplitudes,
            all_params=self.all_params,
            baseline_fwhm=baseline_fwhm,
            baseline_amplitude=baseline_amp,
        )


def plot_results(result: OptimizationResult, save_path: str = 'bayesian_optimization_results.png'):
    """Plot optimization results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Convergence
    evals = np.arange(1, len(result.all_fwhms) + 1)
    best_so_far = np.minimum.accumulate(result.all_fwhms)

    axes[0, 0].plot(evals, result.all_fwhms, 'b.', alpha=0.5, label='Measured')
    axes[0, 0].plot(evals, best_so_far, 'r-', lw=2, label='Best so far')
    axes[0, 0].axhline(result.baseline_fwhm, color='gray', ls='--', label=f'Baseline ({result.baseline_fwhm:.0f}μs)')
    axes[0, 0].set_xlabel('Evaluation')
    axes[0, 0].set_ylabel('FWHM (μs)')
    axes[0, 0].set_title('Convergence')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Amplitude tracking
    axes[0, 1].plot(evals, result.all_amplitudes, 'g.-', alpha=0.7)
    axes[0, 1].axhline(result.baseline_amplitude, color='gray', ls='--', label='Baseline')
    axes[0, 1].axhline(0.7 * result.baseline_amplitude, color='red', ls=':', label='70% threshold')
    axes[0, 1].set_xlabel('Evaluation')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_title('Amplitude Tracking')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Best waveform
    t_ms = np.arange(len(result.best_waveform)) / 1e6 * 1000
    axes[1, 0].plot(t_ms, result.best_waveform, 'b-', lw=0.5)
    axes[1, 0].set_xlabel('Time (ms)')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].set_title('Best Waveform')
    axes[1, 0].grid(True, alpha=0.3)

    # Summary
    summary = f"""
    Bayesian Optimization Results
    =============================

    Baseline:
      FWHM: {result.baseline_fwhm:.1f} μs
      Amplitude: {result.baseline_amplitude:.4f}

    Best Found:
      FWHM: {result.best_fwhm:.1f} μs
      Amplitude: {result.best_amplitude:.4f}

    Improvement:
      FWHM: {(1 - result.best_fwhm/result.baseline_fwhm)*100:+.1f}%
      Amplitude: {(result.best_amplitude/result.baseline_amplitude - 1)*100:+.1f}%

    Best Parameters:
      σ = {result.best_params[0]:.2f} cycles
      φ = {result.best_params[1]:.2f} rad
      pre_comp = [{result.best_params[2]:.3f}, {result.best_params[3]:.3f}, {result.best_params[4]:.3f}]
      post_comp = [{result.best_params[5]:.3f}, {result.best_params[6]:.3f}, {result.best_params[7]:.3f}, {result.best_params[8]:.3f}, {result.best_params[9]:.3f}]

    Total evaluations: {len(result.all_fwhms)}
    """

    axes[1, 1].text(0.05, 0.95, summary, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("=" * 60)
    print("BATPING Bayesian Pulse Optimization")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    optimizer = BayesianPulseOptimizer(
        settling_time=0.2,
        record_duration=0.03,
        min_amplitude_fraction=0.7,
        amplitude_penalty_weight=1000.0,
    )

    try:
        optimizer.connect()

        result = optimizer.optimize(
            n_calls=100,
            n_initial_points=20,
            sigma_range=(1.0, 15.0),
        )

        # Save results
        plot_results(result)

        np.savez(
            'bayesian_optimizer_state.npz',
            best_params=result.best_params,
            best_fwhm=result.best_fwhm,
            best_amplitude=result.best_amplitude,
            best_waveform=result.best_waveform,
            all_fwhms=result.all_fwhms,
            all_amplitudes=result.all_amplitudes,
            baseline_fwhm=result.baseline_fwhm,
            baseline_amplitude=result.baseline_amplitude,
        )
        print("Saved: bayesian_optimizer_state.npz")

    finally:
        optimizer.disconnect()


if __name__ == "__main__":
    main()
