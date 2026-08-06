"""
Pulse Optimizer for BATPING (Agilent 33220A version)

Optimizes arbitrary waveforms sent to piezoelectric transducer array to achieve:
- Maximum received signal amplitude
- Minimum pulse width (for time-of-flight precision)

Hardware:
- Agilent 33220A Function Generator (arbitrary waveform)
- Dodotronic Ultramic 384K EVO (ultrasonic microphone)
- 40 kHz piezoelectric transducer array
"""

import numpy as np
import pyaudio
import time
from typing import Tuple, Optional
from dataclasses import dataclass
from scipy import signal
from scipy.optimize import minimize, differential_evolution
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from agilent_33220a import Agilent33220A, generate_gaussian_pulse, generate_shaped_pulse, generate_compact_pulse


@dataclass
class PulseMetrics:
    """Metrics extracted from received pulse."""
    peak_amplitude: float
    pulse_width_samples: int
    pulse_width_us: float
    energy: float
    peak_to_rms: float
    snr_db: float
    arrival_time_us: float
    rise_time_us: float = 0.0  # 10-90% rise time


def calculate_duty_cycle(waveform: np.ndarray, threshold_fraction: float = 0.1) -> float:
    """
    Calculate duty cycle of a waveform.

    Args:
        waveform: Input waveform array
        threshold_fraction: Fraction of max amplitude to consider "active"

    Returns:
        Duty cycle as fraction (0.0 to 1.0)
    """
    max_amp = np.max(np.abs(waveform))
    if max_amp == 0:
        return 0.0
    threshold = threshold_fraction * max_amp
    active_samples = np.sum(np.abs(waveform) > threshold)
    return active_samples / len(waveform)


@dataclass
class TimeGate:
    """Time gate for isolating acoustic signal from noise."""
    start_ms: float
    duration_ms: float

    @classmethod
    def from_distance(cls, distance_m: float, margin_ms: float = 0.3,
                      speed_of_sound: float = 343.0,
                      min_start_ms: float = 0.2) -> 'TimeGate':
        """Create time gate based on acoustic travel distance."""
        travel_time_ms = (distance_m / speed_of_sound) * 1000
        start = max(min_start_ms, travel_time_ms - margin_ms)
        pulse_duration_ms = 0.5
        duration = margin_ms + pulse_duration_ms + margin_ms
        return cls(start_ms=start, duration_ms=duration)

    @classmethod
    def for_direct_path(cls, separation_m: float, margin_ms: float = 0.5,
                        speed_of_sound: float = 343.0,
                        system_latency_ms: float = 2.0) -> 'TimeGate':
        """Create time gate for direct transducer-to-mic measurement."""
        acoustic_travel_ms = (separation_m / speed_of_sound) * 1000
        expected_arrival_ms = acoustic_travel_ms + system_latency_ms
        start = max(0.5, expected_arrival_ms - margin_ms - 0.5)
        pulse_duration_ms = 1.0
        duration = 1.0 + pulse_duration_ms + 1.0

        print(f"  Acoustic travel time: {acoustic_travel_ms:.2f} ms")
        print(f"  System latency estimate: {system_latency_ms:.1f} ms")
        print(f"  Expected signal arrival: {expected_arrival_ms:.2f} ms")
        print(f"  Gate window: {start:.2f} - {start + duration:.2f} ms")

        return cls(start_ms=start, duration_ms=duration)

    @classmethod
    def auto_detect(cls, audio: np.ndarray, sample_rate: int,
                    threshold_factor: float = 5.0,
                    margin_ms: float = 0.5,
                    min_start_ms: float = 0.5,
                    pulse_window_ms: float = 5.0) -> 'TimeGate':
        """Auto-detect signal arrival and create time gate."""
        envelope = np.abs(signal.hilbert(audio))

        window_samples = int(0.05 * sample_rate / 1000)
        if window_samples > 1:
            kernel = np.ones(window_samples) / window_samples
            envelope_smooth = np.convolve(envelope, kernel, mode='same')
        else:
            envelope_smooth = envelope

        noise_start = int(0.1 * sample_rate / 1000)
        noise_end = int(0.5 * sample_rate / 1000)
        if noise_end > noise_start:
            noise_rms = np.sqrt(np.mean(envelope_smooth[noise_start:noise_end]**2))
        else:
            noise_rms = np.sqrt(np.mean(envelope_smooth[:1000]**2))

        threshold = noise_rms * threshold_factor
        above_threshold = envelope_smooth > threshold

        search_start = int(min_start_ms * sample_rate / 1000)

        if np.any(above_threshold[search_start:]):
            first_above = search_start + np.argmax(above_threshold[search_start:])
            first_above_ms = first_above / sample_rate * 1000

            start_ms = max(min_start_ms, first_above_ms - margin_ms)
            duration_ms = pulse_window_ms

            peak_amplitude = np.max(envelope_smooth)
            print(f"  Noise RMS: {noise_rms:.4f}, Threshold: {threshold:.4f}")
            print(f"  Peak amplitude: {peak_amplitude:.4f}")
            print(f"  First detection at: {first_above_ms:.2f} ms")
            print(f"  Gate window: {start_ms:.2f} - {start_ms + duration_ms:.2f} ms")

            return cls(start_ms=start_ms, duration_ms=duration_ms)
        else:
            print(f"  Warning: No signal detected above threshold ({threshold:.4f})")
            print(f"  Peak in recording: {np.max(envelope_smooth):.4f}")
            print("  Using wide gate as fallback")
            return cls(start_ms=1.0, duration_ms=15.0)

    def to_samples(self, sample_rate: int) -> Tuple[int, int]:
        """Convert gate to sample indices."""
        start_sample = int(self.start_ms * sample_rate / 1000)
        end_sample = int((self.start_ms + self.duration_ms) * sample_rate / 1000)
        return start_sample, end_sample


class UltramicRecorder:
    """Interface to Dodotronic Ultramic 384K EVO."""

    def __init__(self, device_index: Optional[int] = None, sample_rate: int = 384000):
        self.sample_rate = sample_rate
        self.pa = pyaudio.PyAudio()

        if device_index is None:
            device_index = self._find_ultramic()

        self.device_index = device_index

        info = self.pa.get_device_info_by_index(device_index)
        if "UltraMic" not in info['name']:
            print(f"Warning: Device {device_index} is '{info['name']}', expected UltraMic")
        else:
            print(f"Found UltraMic at index {device_index}")

    def _find_ultramic(self) -> int:
        """Auto-detect Ultramic device index."""
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            if "UltraMic" in info['name'] and info['maxInputChannels'] > 0:
                return i
        raise RuntimeError("UltraMic not found")

    def record(self, duration_s: float, pre_delay_s: float = 0.0) -> np.ndarray:
        """Record audio from Ultramic."""
        if pre_delay_s > 0:
            time.sleep(pre_delay_s)

        n_samples = int(duration_s * self.sample_rate)
        chunk_size = 4096

        stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=chunk_size
        )

        frames = []
        samples_remaining = n_samples

        while samples_remaining > 0:
            chunk = min(chunk_size, samples_remaining)
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)
            samples_remaining -= chunk

        stream.stop_stream()
        stream.close()

        audio = np.frombuffer(b''.join(frames), dtype=np.int16)
        audio = audio.astype(np.float64) / 32768.0

        return audio

    def close(self):
        """Clean up PyAudio."""
        self.pa.terminate()


def analyze_pulse(audio: np.ndarray, sample_rate: int = 384000,
                  time_gate: Optional[TimeGate] = None,
                  bandpass: Optional[Tuple[float, float]] = None) -> PulseMetrics:
    """Analyze recorded audio to extract pulse metrics."""
    if time_gate is not None:
        gate_start, gate_end = time_gate.to_samples(sample_rate)
        gate_start = max(0, gate_start)
        gate_end = min(len(audio), gate_end)
        gated_audio = audio[gate_start:gate_end]
        gate_offset = gate_start
    else:
        gated_audio = audio
        gate_offset = 0

    if bandpass is not None:
        nyq = sample_rate / 2
        low = bandpass[0] / nyq
        high = min(bandpass[1] / nyq, 0.99)
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, gated_audio)
    else:
        filtered = gated_audio

    analytic = signal.hilbert(filtered)
    envelope = np.abs(analytic)

    peak_idx_local = np.argmax(envelope)
    peak_amplitude = envelope[peak_idx_local]

    peak_idx_absolute = peak_idx_local + gate_offset
    arrival_time_us = peak_idx_absolute / sample_rate * 1e6

    half_max = peak_amplitude / 2
    above_half = envelope > half_max

    crossings = np.diff(above_half.astype(int))
    rising = np.where(crossings == 1)[0]
    falling = np.where(crossings == -1)[0]

    if len(rising) > 0 and len(falling) > 0:
        start_idx = rising[rising < peak_idx_local]
        end_idx = falling[falling > peak_idx_local]

        if len(start_idx) > 0 and len(end_idx) > 0:
            pulse_start = start_idx[-1]
            pulse_end = end_idx[0]
            pulse_width_samples = pulse_end - pulse_start
        else:
            pulse_width_samples = len(gated_audio)
    else:
        pulse_width_samples = len(gated_audio)

    pulse_width_us = pulse_width_samples / sample_rate * 1e6

    energy = np.sum(envelope**2)

    rms = np.sqrt(np.mean(filtered**2))
    peak_to_rms = peak_amplitude / rms if rms > 0 else 0

    noise_start = int(0.1 * sample_rate / 1000)
    noise_end = int(0.3 * sample_rate / 1000)
    noise_start = max(0, noise_start)
    noise_end = min(len(audio), noise_end)

    if noise_end > noise_start:
        noise_region = audio[noise_start:noise_end]
        noise_rms = np.sqrt(np.mean(noise_region**2))
    else:
        noise_rms = 1e-10

    snr_db = 20 * np.log10(rms / noise_rms) if noise_rms > 0 else 0

    # Calculate 10-90% rise time
    rise_time_us = 0.0
    threshold_10 = 0.1 * peak_amplitude
    threshold_90 = 0.9 * peak_amplitude

    # Search for first crossing of 10% and 90% thresholds before the peak
    pre_peak = envelope[:peak_idx_local]
    if len(pre_peak) > 0:
        above_10 = pre_peak > threshold_10
        above_90 = pre_peak > threshold_90

        if np.any(above_10) and np.any(above_90):
            # First sample above 10%
            first_10_idx = np.argmax(above_10)
            # First sample above 90%
            first_90_idx = np.argmax(above_90)

            if first_90_idx > first_10_idx:
                rise_time_samples = first_90_idx - first_10_idx
                rise_time_us = rise_time_samples / sample_rate * 1e6

    return PulseMetrics(
        peak_amplitude=peak_amplitude,
        pulse_width_samples=pulse_width_samples,
        pulse_width_us=pulse_width_us,
        energy=energy,
        peak_to_rms=peak_to_rms,
        snr_db=snr_db,
        arrival_time_us=arrival_time_us,
        rise_time_us=rise_time_us
    )


class PulseOptimizer:
    """Optimize waveform shape for best transducer response."""

    def __init__(self, awg: Agilent33220A, recorder: UltramicRecorder,
                 n_waveform_points: int = 4000,
                 carrier_freq_hz: float = 40000,
                 sample_rate_hz: float = 1e6,
                 amplitude_vpp: float = 2.0,
                 record_duration_s: float = 0.05,
                 time_gate: Optional[TimeGate] = None,
                 acoustic_distance_m: Optional[float] = None):
        """
        Initialize optimizer.

        Args:
            awg: Connected Agilent33220A instance
            recorder: UltramicRecorder instance
            n_waveform_points: Points in arbitrary waveform (max 16384 for 33220A)
            carrier_freq_hz: Carrier frequency (transducer resonance)
            sample_rate_hz: Effective sample rate (determines rep rate)
            amplitude_vpp: Output amplitude (Vpp)
            record_duration_s: Recording duration per trial
            time_gate: TimeGate for signal isolation
            acoustic_distance_m: Acoustic path length (for auto time gate)
        """
        self.awg = awg
        self.recorder = recorder
        self.n_points = n_waveform_points
        self.carrier_freq = carrier_freq_hz
        self.sample_rate = sample_rate_hz
        self.amplitude = amplitude_vpp
        self.record_duration = record_duration_s

        if time_gate is not None:
            self.time_gate = time_gate
        elif acoustic_distance_m is not None:
            self.time_gate = TimeGate.from_distance(acoustic_distance_m)
            print(f"Auto time gate: {self.time_gate.start_ms:.1f}ms to "
                  f"{self.time_gate.start_ms + self.time_gate.duration_ms:.1f}ms")
        else:
            self.time_gate = TimeGate(start_ms=2.0, duration_ms=4.0)
            print("Warning: No time gate specified, using default (2-6ms)")

        self.history = []

    def waveform_from_params(self, params: np.ndarray) -> np.ndarray:
        """Generate waveform from Fourier coefficient parameters."""
        return generate_shaped_pulse(
            n_points=self.n_points,
            center_freq_hz=self.carrier_freq,
            sample_rate_hz=self.sample_rate,
            fourier_coeffs=params
        )

    def evaluate_waveform(self, waveform: np.ndarray,
                          settling_time_s: float = 0.02) -> PulseMetrics:
        """Upload waveform, output, record, and analyze."""
        # Repetition frequency: one waveform period = n_points / sample_rate
        waveform_duration_s = self.n_points / self.sample_rate
        rep_freq_hz = 1.0 / waveform_duration_s

        self.awg.output_arbitrary(waveform, freq_hz=rep_freq_hz, vpp=self.amplitude)
        time.sleep(settling_time_s)

        audio = self.recorder.record(self.record_duration)

        self.awg.set_output(False)

        metrics = analyze_pulse(
            audio,
            sample_rate=self.recorder.sample_rate,
            time_gate=self.time_gate
        )

        return metrics

    def objective_function(self, params: np.ndarray,
                           amplitude_weight: float = 1.0,
                           rise_time_weight: float = 0.5,
                           duty_cycle_weight: float = 10.0,
                           max_duty_cycle: float = 0.25) -> float:
        """
        Objective function: maximize amplitude, minimize rise time, penalize high duty cycle.

        Args:
            params: Fourier coefficients for waveform envelope
            amplitude_weight: Weight for peak amplitude term (default 1.0)
            rise_time_weight: Weight for rise time penalty (default 0.5)
            duty_cycle_weight: Weight for duty cycle penalty (default 10.0)
            max_duty_cycle: Maximum allowed duty cycle (default 0.25 = 25%)

        Returns:
            Negative of score (for minimization)
        """
        waveform = self.waveform_from_params(params)

        # Check duty cycle of INPUT waveform BEFORE evaluation
        duty_cycle = calculate_duty_cycle(waveform)
        duty_cycle_violation = max(duty_cycle - max_duty_cycle, 0)
        duty_penalty = duty_cycle_violation ** 2

        # Evaluate acoustic output
        metrics = self.evaluate_waveform(waveform)

        self.history.append({
            'params': params.copy(),
            'metrics': metrics,
            'duty_cycle': duty_cycle
        })

        # Score: maximize amplitude, minimize rise time, penalize high duty cycle
        score = (amplitude_weight * metrics.peak_amplitude
                 - rise_time_weight * (metrics.rise_time_us / 100)
                 - duty_cycle_weight * duty_penalty)

        print(f"  Eval {len(self.history)}: amp={metrics.peak_amplitude:.4f}, "
              f"rise={metrics.rise_time_us:.1f}us, duty={duty_cycle:.1%}, score={score:.4f}")

        return -score

    def optimize(self, n_params: int = 10, method: str = "nelder-mead",
                 max_iter: int = 50, amplitude_weight: float = 1.0,
                 rise_time_weight: float = 0.5,
                 duty_cycle_weight: float = 10.0,
                 max_duty_cycle: float = 0.25,
                 initial_params: Optional[np.ndarray] = None) -> dict:
        """
        Run optimization with reformulated objective function.

        Args:
            n_params: Number of Fourier coefficients
            method: Optimization method ('nelder-mead' or 'differential_evolution')
            max_iter: Maximum iterations
            amplitude_weight: Weight for amplitude term (default 1.0)
            rise_time_weight: Weight for rise time penalty (default 0.5)
            duty_cycle_weight: Weight for duty cycle penalty (default 10.0)
            max_duty_cycle: Maximum allowed duty cycle (default 0.25)
            initial_params: Optional initial parameter vector

        Returns:
            Dict with optimal_params, best_metrics, history, etc.
        """
        self.history = []

        # Initial parameters: start with a centered pulse envelope
        # Using a0 + a1*cos(2*pi*t/T) with a0=a1 gives envelope peaking at center
        # that goes to zero at edges: e(t) = a0*(1 - cos(2*pi*t/T))
        # But this still has ~67% duty cycle due to how 10% threshold works.
        # The duty cycle penalty will guide the optimizer to find better solutions.
        if initial_params is not None:
            x0 = initial_params
        else:
            x0 = np.zeros(n_params)
            x0[0] = 0.5   # DC component
            if n_params > 1:
                x0[1] = -0.5  # First cosine (creates centered pulse)

        print(f"Starting optimization with {n_params} Fourier coefficients")
        print(f"Method: {method}, max_iter: {max_iter}")
        print(f"Weights: amplitude={amplitude_weight}, rise_time={rise_time_weight}, "
              f"duty_cycle={duty_cycle_weight}")
        print(f"Max duty cycle: {max_duty_cycle:.0%}")
        print()

        obj_func = lambda p: self.objective_function(
            p, amplitude_weight, rise_time_weight, duty_cycle_weight, max_duty_cycle
        )

        if method == "differential_evolution":
            bounds = [(-2, 2)] * n_params
            result = differential_evolution(obj_func, bounds, maxiter=max_iter,
                                           seed=42, disp=True, polish=False)
            optimal_params = result.x
        else:
            result = minimize(obj_func, x0, method=method,
                            options={'maxiter': max_iter, 'disp': True})
            optimal_params = result.x

        # Find best result (highest score, accounting for duty cycle constraint)
        valid_results = [
            (i, h) for i, h in enumerate(self.history)
            if h.get('duty_cycle', 0) <= max_duty_cycle + 0.05  # Allow small tolerance
        ]

        if valid_results:
            # Among valid results, find best amplitude
            best_idx = max(valid_results, key=lambda x: x[1]['metrics'].peak_amplitude)[0]
        else:
            # If no valid results, take best amplitude anyway (shouldn't happen)
            best_idx = np.argmax([h['metrics'].peak_amplitude for h in self.history])
            print("Warning: No results satisfied duty cycle constraint!")

        return {
            'optimal_params': optimal_params,
            'best_metrics': self.history[best_idx]['metrics'],
            'best_params': self.history[best_idx]['params'],
            'best_duty_cycle': self.history[best_idx].get('duty_cycle', None),
            'history': self.history,
            'n_evaluations': len(self.history)
        }


def run_baseline_test(awg: Agilent33220A, recorder: UltramicRecorder,
                      time_gate: Optional[TimeGate] = None,
                      n_points: int = 4000,
                      sample_rate: float = 1e6,
                      amplitude_vpp: float = 2.0) -> dict:
    """Run baseline tests with standard pulse shapes."""
    results = {}

    pulse_configs = [
        ("gaussian_narrow", 2.0),
        ("gaussian_medium", 5.0),
        ("gaussian_wide", 10.0),
        ("continuous_sine", None),
    ]

    for name, sigma_cycles in pulse_configs:
        print(f"\nTesting {name}...")

        if sigma_cycles is not None:
            waveform = generate_gaussian_pulse(
                n_points=n_points,
                center_freq_hz=40000,
                sample_rate_hz=sample_rate,
                sigma_cycles=sigma_cycles
            )
            waveform_duration_s = n_points / sample_rate
            rep_freq_hz = 1.0 / waveform_duration_s
            awg.output_arbitrary(waveform, freq_hz=rep_freq_hz, vpp=amplitude_vpp)
        else:
            # Continuous sine wave
            awg.output_sine(40000, amplitude_vpp)

        time.sleep(0.02)
        audio = recorder.record(0.05)
        awg.set_output(False)

        metrics = analyze_pulse(audio, recorder.sample_rate, time_gate=time_gate)
        results[name] = metrics

        print(f"  Amplitude: {metrics.peak_amplitude:.4f}")
        print(f"  Width: {metrics.pulse_width_us:.1f} us")
        print(f"  Arrival: {metrics.arrival_time_us:.1f} us")
        print(f"  SNR: {metrics.snr_db:.1f} dB")

    return results


def plot_recording_with_gate(audio: np.ndarray, sample_rate: int,
                             time_gate: TimeGate, save_path: str):
    """Plot recording with time gate overlay."""
    t_ms = np.arange(len(audio)) / sample_rate * 1000

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    axes[0].plot(t_ms, audio, 'b-', linewidth=0.5)
    axes[0].axvspan(time_gate.start_ms, time_gate.start_ms + time_gate.duration_ms,
                    alpha=0.3, color='green', label='Time Gate')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Full Recording with Time Gate')
    axes[0].legend()
    axes[0].grid(True)

    margin = 2.0
    zoom_start = max(0, time_gate.start_ms - margin)
    zoom_end = time_gate.start_ms + time_gate.duration_ms + margin

    mask = (t_ms >= zoom_start) & (t_ms <= zoom_end)
    if np.any(mask):
        axes[1].plot(t_ms[mask], audio[mask], 'b-', linewidth=0.5)
        axes[1].axvspan(time_gate.start_ms, time_gate.start_ms + time_gate.duration_ms,
                        alpha=0.3, color='green', label='Time Gate')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Zoomed to Gate Region')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved plot to {save_path}")


def plot_optimization_history(history: list, save_path: str):
    """Plot optimization progress."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    iterations = range(1, len(history) + 1)
    amplitudes = [h['metrics'].peak_amplitude for h in history]
    widths = [h['metrics'].pulse_width_us for h in history]
    snrs = [h['metrics'].snr_db for h in history]

    axes[0, 0].plot(iterations, amplitudes, 'b.-')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Peak Amplitude')
    axes[0, 0].set_title('Amplitude vs Iteration')
    axes[0, 0].grid(True)

    axes[0, 1].plot(iterations, widths, 'r.-')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Pulse Width (µs)')
    axes[0, 1].set_title('Pulse Width vs Iteration')
    axes[0, 1].grid(True)

    axes[1, 0].plot(iterations, snrs, 'g.-')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('SNR (dB)')
    axes[1, 0].set_title('SNR vs Iteration')
    axes[1, 0].grid(True)

    sc = axes[1, 1].scatter(widths, amplitudes, c=iterations, cmap='viridis', alpha=0.7)
    axes[1, 1].set_xlabel('Pulse Width (µs)')
    axes[1, 1].set_ylabel('Peak Amplitude')
    axes[1, 1].set_title('Amplitude vs Width Trade-off')
    plt.colorbar(sc, ax=axes[1, 1], label='Iteration')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='BATPING Pulse Optimizer (Agilent 33220A)')
    parser.add_argument('--distance', type=float, default=0.25,
                        help='Acoustic path distance in meters')
    parser.add_argument('--amplitude', type=float, default=2.0,
                        help='Output amplitude in Vpp')
    parser.add_argument('--baseline-only', action='store_true',
                        help='Run only baseline tests')
    parser.add_argument('--optimize', action='store_true',
                        help='Run full optimization')
    parser.add_argument('--max-iter', type=int, default=30,
                        help='Maximum optimization iterations')
    parser.add_argument('--n-params', type=int, default=8,
                        help='Number of Fourier coefficients')
    parser.add_argument('--plot-recording', action='store_true',
                        help='Save recording plot')

    args = parser.parse_args()

    print("BATPING Pulse Optimizer (Agilent 33220A)")
    print("=" * 50)

    # Connect hardware
    print("\nConnecting to Agilent 33220A...")
    awg = Agilent33220A()
    awg.connect()

    print("\nInitializing Ultramic recorder...")
    recorder = UltramicRecorder(sample_rate=384000)

    # Calibration: auto-detect signal arrival
    print("\n" + "=" * 50)
    print("CALIBRATION: Auto-detecting signal arrival time")
    print("=" * 50)

    # Send test pulse
    pulse = generate_gaussian_pulse(4000, 40000, 1e6, sigma_cycles=5)
    awg.output_arbitrary(pulse, freq_hz=250, vpp=args.amplitude)
    time.sleep(0.02)
    calibration_audio = recorder.record(0.05)
    awg.set_output(False)

    time_gate = TimeGate.auto_detect(calibration_audio, recorder.sample_rate)

    if args.plot_recording:
        plot_recording_with_gate(calibration_audio, recorder.sample_rate,
                                time_gate, "calibration_recording.png")

    # Baseline tests
    print("\n" + "=" * 50)
    print("BASELINE TESTS")
    print("=" * 50)
    baseline_results = run_baseline_test(awg, recorder, time_gate=time_gate,
                                         amplitude_vpp=args.amplitude)

    print("\n" + "=" * 50)
    print("BASELINE SUMMARY")
    print("=" * 50)
    for ptype, metrics in baseline_results.items():
        print(f"{ptype:20s}: amp={metrics.peak_amplitude:.4f}, "
              f"width={metrics.pulse_width_us:.1f}us, snr={metrics.snr_db:.1f}dB")

    # Optimization
    if args.optimize:
        print("\n" + "=" * 50)
        print("OPTIMIZATION")
        print("=" * 50)

        optimizer = PulseOptimizer(
            awg=awg,
            recorder=recorder,
            n_waveform_points=4000,
            carrier_freq_hz=40000,
            sample_rate_hz=1e6,
            amplitude_vpp=args.amplitude,
            time_gate=time_gate
        )

        results = optimizer.optimize(
            n_params=args.n_params,
            method="nelder-mead",
            max_iter=args.max_iter
        )

        print("\n" + "=" * 50)
        print("OPTIMIZATION RESULTS")
        print("=" * 50)
        best = results['best_metrics']
        print(f"Best amplitude: {best.peak_amplitude:.4f}")
        print(f"Best width: {best.pulse_width_us:.1f} us")
        print(f"Best SNR: {best.snr_db:.1f} dB")
        print(f"Total evaluations: {results['n_evaluations']}")

        # Save results
        np.save("optimized_params.npy", results['best_params'])
        plot_optimization_history(results['history'], "optimization_history.png")

    # Cleanup
    awg.set_output(False)
    awg.disconnect()
    recorder.close()
    print("\nDone!")
