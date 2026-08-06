"""
Pulse Optimizer for BATPING

Optimizes arbitrary waveforms sent to piezoelectric transducer array to achieve:
- Maximum received signal amplitude
- Minimum pulse width (for time-of-flight precision)

Hardware:
- Teledyne LeCroy WaveStation 2052 (arbitrary waveform generator)
- Dodotronic Ultramic 384K EVO (ultrasonic microphone)
- 40 kHz piezoelectric transducer array

Optimization approach:
1. Parameterize the waveform (e.g., Fourier coefficients, time-domain samples)
2. Send waveform to transducer via WaveStation
3. Record response with Ultramic
4. Extract metrics (peak amplitude, pulse width at half-max)
5. Update parameters to improve objective function
6. Repeat until convergence
"""

import numpy as np
import pyaudio
import time
from typing import Tuple, Optional, Callable
from dataclasses import dataclass
from scipy import signal
from scipy.optimize import minimize, differential_evolution
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

from wavestation import WaveStation, generate_gaussian_pulse, generate_chirp


@dataclass
class PulseMetrics:
    """Metrics extracted from received pulse."""
    peak_amplitude: float      # Maximum absolute amplitude
    pulse_width_samples: int   # Width at half maximum (samples)
    pulse_width_us: float      # Width at half maximum (microseconds)
    energy: float              # Total signal energy
    peak_to_rms: float         # Peak to RMS ratio (indicates pulse sharpness)
    snr_db: float              # Signal to noise ratio estimate
    arrival_time_us: float     # Time of peak arrival (microseconds from recording start)


@dataclass
class TimeGate:
    """
    Time gate configuration for filtering by arrival time.

    The gate isolates the acoustic signal from:
    - EM coupling (arrives at t≈0, speed of light)
    - Motion sensor interference (continuous 33 kHz)
    - Other transient noise outside the expected arrival window
    """
    start_ms: float      # Gate opens at this time (ms from recording start)
    duration_ms: float   # Gate stays open for this duration (ms)

    @classmethod
    def from_distance(cls, distance_m: float, margin_ms: float = 0.3,
                      speed_of_sound: float = 343.0,
                      min_start_ms: float = 0.2) -> 'TimeGate':
        """
        Create time gate based on expected acoustic travel distance.

        Args:
            distance_m: Acoustic path length in meters (one-way for direct,
                       round-trip for echo)
            margin_ms: Extra time margin on each side of expected arrival
            speed_of_sound: Speed of sound in m/s (343 at 20°C sea level)
            min_start_ms: Minimum gate start time to skip initial EM transient

        Returns:
            TimeGate configured for the expected arrival
        """
        travel_time_ms = (distance_m / speed_of_sound) * 1000

        # Gate starts slightly before expected arrival, but after EM spike
        start = max(min_start_ms, travel_time_ms - margin_ms)

        # Gate duration: margin before + pulse time + margin after
        # For 40 kHz, one cycle = 25 µs, typical pulse might be 10-20 cycles = 0.25-0.5 ms
        pulse_duration_ms = 0.5
        duration = margin_ms + pulse_duration_ms + margin_ms

        return cls(start_ms=start, duration_ms=duration)

    @classmethod
    def for_direct_path(cls, separation_m: float, margin_ms: float = 0.2,
                        speed_of_sound: float = 343.0,
                        system_latency_ms: float = 2.5) -> 'TimeGate':
        """
        Create time gate for direct transducer-to-mic measurement (no reflection).

        Accounts for system latency (recording buffer + output startup).

        Args:
            separation_m: Distance between transducer and microphone
            margin_ms: Time margin (tighter than echo mode)
            speed_of_sound: Speed of sound in m/s
            system_latency_ms: Latency from pyaudio buffering and WaveStation startup

        Returns:
            TimeGate configured for direct path
        """
        acoustic_travel_ms = (separation_m / speed_of_sound) * 1000

        # Total expected arrival = acoustic travel + system latency
        expected_arrival_ms = acoustic_travel_ms + system_latency_ms

        # Gate opens before expected arrival
        start = max(0.5, expected_arrival_ms - margin_ms - 0.5)

        # Duration covers margin + pulse + margin + some extra for variability
        pulse_duration_ms = 1.0
        duration = 1.0 + pulse_duration_ms + 1.0  # Wider gate to be safe

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
        """
        Auto-detect FIRST signal arrival and create appropriate time gate.

        Finds the first significant signal (not reverberations) and creates
        a gate around just that initial pulse.

        Args:
            audio: Recorded audio samples
            sample_rate: Sample rate in Hz
            threshold_factor: Signal must exceed noise_rms * threshold_factor
            margin_ms: Margin before first detection
            min_start_ms: Minimum gate start time (to skip any EM transient)
            pulse_window_ms: Expected duration of direct pulse (excludes reverb)

        Returns:
            TimeGate positioned around first signal arrival
        """
        # Compute envelope using Hilbert transform
        envelope = np.abs(signal.hilbert(audio))

        # Smooth envelope slightly to reduce noise spikes
        window_samples = int(0.05 * sample_rate / 1000)  # 50 µs window
        if window_samples > 1:
            kernel = np.ones(window_samples) / window_samples
            envelope_smooth = np.convolve(envelope, kernel, mode='same')
        else:
            envelope_smooth = envelope

        # Estimate noise from samples between 0.1 and 0.5 ms (after EM, before acoustic)
        noise_start = int(0.1 * sample_rate / 1000)
        noise_end = int(0.5 * sample_rate / 1000)
        if noise_end > noise_start:
            noise_rms = np.sqrt(np.mean(envelope_smooth[noise_start:noise_end]**2))
        else:
            noise_rms = np.sqrt(np.mean(envelope_smooth[:1000]**2))

        # Find where signal first exceeds threshold
        threshold = noise_rms * threshold_factor
        above_threshold = envelope_smooth > threshold

        # Start searching after min_start_ms
        search_start = int(min_start_ms * sample_rate / 1000)

        if np.any(above_threshold[search_start:]):
            # First sample above threshold (after search_start)
            first_above = search_start + np.argmax(above_threshold[search_start:])
            first_above_ms = first_above / sample_rate * 1000

            # Gate starts slightly before first detection
            start_ms = max(min_start_ms, first_above_ms - margin_ms)

            # Gate duration covers just the direct pulse, not reverberations
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
        """
        Initialize Ultramic recorder.

        Args:
            device_index: PyAudio device index for Ultramic. If None, auto-detect.
            sample_rate: Sample rate in Hz (max 384000 for Ultramic)
        """
        self.sample_rate = sample_rate
        self.pa = pyaudio.PyAudio()

        # Auto-detect Ultramic if no index provided
        if device_index is None:
            device_index = self._find_ultramic()

        self.device_index = device_index

        # Verify device exists
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
        raise RuntimeError("UltraMic not found. Available input devices: " +
                          ", ".join(self.pa.get_device_info_by_index(i)['name']
                                   for i in range(self.pa.get_device_count())
                                   if self.pa.get_device_info_by_index(i)['maxInputChannels'] > 0))

    def record(self, duration_s: float, pre_delay_s: float = 0.0) -> np.ndarray:
        """
        Record audio from Ultramic.

        Args:
            duration_s: Recording duration in seconds
            pre_delay_s: Delay before starting recording (for synchronization)

        Returns:
            Audio samples as numpy array (float64, normalized)
        """
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

        # Convert to numpy array
        audio = np.frombuffer(b''.join(frames), dtype=np.int16)
        audio = audio.astype(np.float64) / 32768.0  # Normalize to [-1, 1]

        return audio

    def close(self):
        """Clean up PyAudio."""
        self.pa.terminate()


def analyze_pulse(audio: np.ndarray, sample_rate: int = 384000,
                  time_gate: Optional[TimeGate] = None,
                  bandpass: Optional[Tuple[float, float]] = None,
                  noise_gate: Optional[Tuple[float, float]] = None) -> PulseMetrics:
    """
    Analyze recorded audio to extract pulse metrics using time gating.

    Time gating is the primary filtering method - it isolates the acoustic
    signal based on expected arrival time, rejecting:
    - EM coupling spike at t≈0
    - Continuous interference (motion sensors at 33 kHz)
    - Other noise outside the expected arrival window

    Args:
        audio: Recorded audio samples
        sample_rate: Sample rate in Hz
        time_gate: TimeGate specifying when to look for the signal.
                   If None, analyzes the entire recording (not recommended).
        bandpass: Optional (low_hz, high_hz) bandpass filter. Only use if
                  needed for additional noise rejection. Time gating alone
                  is usually sufficient.
        noise_gate: Optional (start_ms, end_ms) window for noise estimation.
                    Defaults to first 0.3ms (before acoustic arrival but
                    after initial EM transient settles).

    Returns:
        PulseMetrics with extracted values
    """
    # Apply time gate to isolate signal region
    if time_gate is not None:
        gate_start, gate_end = time_gate.to_samples(sample_rate)
        gate_start = max(0, gate_start)
        gate_end = min(len(audio), gate_end)
        gated_audio = audio[gate_start:gate_end]
        gate_offset = gate_start  # For converting back to absolute time
    else:
        gated_audio = audio
        gate_offset = 0

    # Optional bandpass filter (only if specified)
    if bandpass is not None:
        nyq = sample_rate / 2
        low = bandpass[0] / nyq
        high = min(bandpass[1] / nyq, 0.99)
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, gated_audio)
    else:
        filtered = gated_audio

    # Compute envelope using Hilbert transform
    analytic = signal.hilbert(filtered)
    envelope = np.abs(analytic)

    # Find peak within gated region
    peak_idx_local = np.argmax(envelope)
    peak_amplitude = envelope[peak_idx_local]

    # Absolute peak index (in original recording)
    peak_idx_absolute = peak_idx_local + gate_offset
    arrival_time_us = peak_idx_absolute / sample_rate * 1e6

    # Compute pulse width at half maximum
    half_max = peak_amplitude / 2
    above_half = envelope > half_max

    # Find first and last crossing of half-max
    crossings = np.diff(above_half.astype(int))
    rising = np.where(crossings == 1)[0]
    falling = np.where(crossings == -1)[0]

    if len(rising) > 0 and len(falling) > 0:
        # Find the pulse containing the peak
        start_idx = rising[rising < peak_idx_local]
        end_idx = falling[falling > peak_idx_local]

        if len(start_idx) > 0 and len(end_idx) > 0:
            pulse_start = start_idx[-1]
            pulse_end = end_idx[0]
            pulse_width_samples = pulse_end - pulse_start
        else:
            pulse_width_samples = len(gated_audio)  # Fallback
    else:
        pulse_width_samples = len(gated_audio)  # Fallback

    pulse_width_us = pulse_width_samples / sample_rate * 1e6

    # Energy within gated region
    energy = np.sum(envelope**2)

    # Peak to RMS ratio (within gated region)
    rms = np.sqrt(np.mean(filtered**2))
    peak_to_rms = peak_amplitude / rms if rms > 0 else 0

    # SNR estimate using noise from a separate time region
    # Default: use samples from 0.1-0.3ms (after EM spike settles, before acoustic arrival)
    if noise_gate is not None:
        noise_start = int(noise_gate[0] * sample_rate / 1000)
        noise_end = int(noise_gate[1] * sample_rate / 1000)
    else:
        noise_start = int(0.1 * sample_rate / 1000)  # 0.1 ms
        noise_end = int(0.3 * sample_rate / 1000)    # 0.3 ms

    noise_start = max(0, noise_start)
    noise_end = min(len(audio), noise_end)

    if noise_end > noise_start:
        noise_region = audio[noise_start:noise_end]
        if bandpass is not None:
            noise_region = signal.filtfilt(b, a, noise_region)
        noise_rms = np.sqrt(np.mean(noise_region**2))
    else:
        noise_rms = 1e-10  # Fallback

    signal_rms = rms
    snr_db = 20 * np.log10(signal_rms / noise_rms) if noise_rms > 0 else 0

    return PulseMetrics(
        peak_amplitude=peak_amplitude,
        pulse_width_samples=pulse_width_samples,
        pulse_width_us=pulse_width_us,
        energy=energy,
        peak_to_rms=peak_to_rms,
        snr_db=snr_db,
        arrival_time_us=arrival_time_us
    )


class PulseOptimizer:
    """
    Optimize waveform shape for best transducer response.

    The waveform is parameterized and optimization adjusts these parameters
    to maximize amplitude and minimize pulse width.
    """

    def __init__(self, wavestation: WaveStation, recorder: UltramicRecorder,
                 n_waveform_points: int = 1024,
                 carrier_freq_hz: float = 40000,
                 sample_rate_hz: float = 1e6,
                 amplitude_vpp: float = 1.0,
                 record_duration_s: float = 0.1,
                 time_gate: Optional[TimeGate] = None,
                 acoustic_distance_m: Optional[float] = None,
                 bandpass_hz: Optional[Tuple[float, float]] = None):
        """
        Initialize optimizer.

        Args:
            wavestation: Connected WaveStation instance
            recorder: UltramicRecorder instance
            n_waveform_points: Points in arbitrary waveform
            carrier_freq_hz: Carrier frequency (transducer resonance)
            sample_rate_hz: WaveStation sample rate for playback
            amplitude_vpp: Output amplitude (Vpp)
            record_duration_s: Recording duration per trial
            time_gate: TimeGate for signal isolation. If None and acoustic_distance_m
                       is provided, will be computed automatically.
            acoustic_distance_m: Total acoustic path length (round-trip for echo).
                                Used to auto-compute time_gate if not provided.
            bandpass_hz: Optional bandpass filter (usually not needed with time gating)
        """
        self.ws = wavestation
        self.recorder = recorder
        self.n_points = n_waveform_points
        self.carrier_freq = carrier_freq_hz
        self.sample_rate = sample_rate_hz
        self.amplitude = amplitude_vpp
        self.record_duration = record_duration_s
        self.bandpass = bandpass_hz

        # Set up time gate
        if time_gate is not None:
            self.time_gate = time_gate
        elif acoustic_distance_m is not None:
            self.time_gate = TimeGate.from_distance(acoustic_distance_m)
            print(f"Auto time gate: {self.time_gate.start_ms:.1f}ms to "
                  f"{self.time_gate.start_ms + self.time_gate.duration_ms:.1f}ms "
                  f"(for {acoustic_distance_m}m path)")
        else:
            # Default: assume ~1m path
            self.time_gate = TimeGate(start_ms=2.0, duration_ms=4.0)
            print("Warning: No time gate specified, using default (2-6ms)")

        # Optimization history
        self.history = []

    def waveform_from_params(self, params: np.ndarray,
                             param_type: str = "fourier") -> np.ndarray:
        """
        Generate waveform from parameter vector.

        Args:
            params: Parameter vector
            param_type: "fourier" - Fourier coefficients for envelope
                       "envelope" - Direct envelope samples (interpolated)
                       "predistort" - Pre-distortion filter coefficients

        Returns:
            Waveform samples
        """
        t = np.linspace(0, 1, self.n_points)

        if param_type == "fourier":
            # params = [a0, a1, a2, ..., b1, b2, ...]  (DC + cosine + sine coefficients)
            n_harmonics = len(params) // 2
            envelope = params[0] * np.ones_like(t)  # DC component

            for k in range(1, n_harmonics):
                envelope += params[k] * np.cos(2 * np.pi * k * t)
                envelope += params[n_harmonics + k - 1] * np.sin(2 * np.pi * k * t)

            # Ensure non-negative envelope
            envelope = np.maximum(envelope, 0)

            # Normalize envelope
            if np.max(envelope) > 0:
                envelope = envelope / np.max(envelope)

        elif param_type == "envelope":
            # params are sparse envelope samples, interpolate to full length
            envelope_points = np.concatenate([[0], params, [0]])  # Zero at edges
            x_sparse = np.linspace(0, 1, len(envelope_points))
            envelope = np.interp(t, x_sparse, envelope_points)
            envelope = np.maximum(envelope, 0)

        elif param_type == "predistort":
            # Start with Gaussian envelope, apply pre-distortion filter
            base_envelope = np.exp(-((t - 0.5)**2) / (2 * 0.1**2))
            # params are FIR filter coefficients
            envelope = signal.lfilter(params, [1.0], base_envelope)
            envelope = np.maximum(envelope, 0)
            if np.max(envelope) > 0:
                envelope = envelope / np.max(envelope)

        else:
            raise ValueError(f"Unknown param_type: {param_type}")

        # Apply carrier
        n_carrier_cycles = self.carrier_freq * self.n_points / self.sample_rate
        carrier = np.sin(2 * np.pi * n_carrier_cycles * t)
        waveform = envelope * carrier

        return waveform

    def evaluate_waveform(self, waveform: np.ndarray,
                          settling_time_s: float = 0.05) -> PulseMetrics:
        """
        Upload waveform, trigger output, record response, and analyze.

        Args:
            waveform: Waveform samples to test
            settling_time_s: Time to wait after enabling output

        Returns:
            PulseMetrics from recorded response
        """
        # Upload and configure
        self.ws.upload_arbitrary_waveform(waveform, name="opt_test")
        self.ws.set_arbitrary_waveform(
            channel=1,
            name="opt_test",
            sample_rate_hz=self.sample_rate,
            amp_vpp=self.amplitude
        )

        # Enable output
        self.ws.set_output(channel=1, enable=True)
        time.sleep(settling_time_s)

        # Record response
        audio = self.recorder.record(self.record_duration)

        # Disable output
        self.ws.set_output(channel=1, enable=False)

        # Analyze with time gating
        metrics = analyze_pulse(
            audio,
            sample_rate=self.recorder.sample_rate,
            time_gate=self.time_gate,
            bandpass=self.bandpass
        )

        return metrics

    def objective_function(self, params: np.ndarray,
                           param_type: str = "fourier",
                           amplitude_weight: float = 1.0,
                           width_weight: float = 0.5) -> float:
        """
        Objective function for optimization.

        We want to MAXIMIZE amplitude and MINIMIZE width.
        Since optimizers typically minimize, we return negative score.

        Args:
            params: Parameter vector
            param_type: Parameterization type
            amplitude_weight: Weight for amplitude term
            width_weight: Weight for width term (negative contribution)

        Returns:
            Negative score (lower is better for minimization)
        """
        waveform = self.waveform_from_params(params, param_type)
        metrics = self.evaluate_waveform(waveform)

        # Store in history
        self.history.append({
            'params': params.copy(),
            'metrics': metrics
        })

        # Compute score: maximize amplitude, minimize width
        # Normalize width to reasonable scale (target ~50us would be ~19 samples at 384kHz)
        # Score = amplitude - width_weight * (width_us / 100)
        score = amplitude_weight * metrics.peak_amplitude - width_weight * (metrics.pulse_width_us / 100)

        print(f"  Eval {len(self.history)}: amp={metrics.peak_amplitude:.4f}, "
              f"width={metrics.pulse_width_us:.1f}us, score={score:.4f}")

        return -score  # Negative because we minimize

    def optimize(self, n_params: int = 10, param_type: str = "fourier",
                 method: str = "nelder-mead", max_iter: int = 50,
                 amplitude_weight: float = 1.0, width_weight: float = 0.5) -> dict:
        """
        Run optimization.

        Args:
            n_params: Number of parameters
            param_type: Parameterization type
            method: Optimization method ("nelder-mead", "powell", "differential_evolution")
            max_iter: Maximum iterations
            amplitude_weight: Weight for amplitude term
            width_weight: Weight for width term

        Returns:
            Dict with optimal parameters, metrics, and history
        """
        self.history = []

        # Initial parameters (start with Gaussian-like envelope)
        if param_type == "fourier":
            # Start with DC and gentle rolloff
            x0 = np.zeros(n_params)
            x0[0] = 1.0  # DC
            x0[1] = 0.5  # First harmonic
        elif param_type == "envelope":
            # Start with Gaussian shape
            t = np.linspace(0, 1, n_params)
            x0 = np.exp(-((t - 0.5)**2) / (2 * 0.2**2))
        else:
            x0 = np.ones(n_params) / n_params

        print(f"Starting optimization with {n_params} parameters ({param_type})")
        print(f"Method: {method}, max_iter: {max_iter}")
        print(f"Weights: amplitude={amplitude_weight}, width={width_weight}")
        print()

        obj_func = lambda p: self.objective_function(p, param_type, amplitude_weight, width_weight)

        if method == "differential_evolution":
            bounds = [(-2, 2)] * n_params
            result = differential_evolution(obj_func, bounds, maxiter=max_iter,
                                           seed=42, disp=True, polish=False)
            optimal_params = result.x
        else:
            result = minimize(obj_func, x0, method=method,
                            options={'maxiter': max_iter, 'disp': True})
            optimal_params = result.x

        # Get best result from history
        best_idx = np.argmin([h['metrics'].peak_amplitude * -1 +
                             width_weight * h['metrics'].pulse_width_us / 100
                             for h in self.history])

        return {
            'optimal_params': optimal_params,
            'best_metrics': self.history[best_idx]['metrics'],
            'best_params': self.history[best_idx]['params'],
            'history': self.history,
            'n_evaluations': len(self.history)
        }


def run_baseline_test(ws: WaveStation, recorder: UltramicRecorder,
                      pulse_types: list = None,
                      time_gate: Optional[TimeGate] = None,
                      acoustic_distance_m: Optional[float] = None) -> dict:
    """
    Run baseline tests with standard pulse shapes.

    Args:
        ws: Connected WaveStation
        recorder: UltramicRecorder instance
        pulse_types: List of pulse types to test
        time_gate: TimeGate for signal isolation
        acoustic_distance_m: Acoustic path length (used if time_gate not provided)

    Returns:
        Dict mapping pulse type to metrics
    """
    if pulse_types is None:
        pulse_types = ["gaussian_narrow", "gaussian_wide", "rectangular", "chirp"]

    # Set up time gate
    if time_gate is None and acoustic_distance_m is not None:
        time_gate = TimeGate.from_distance(acoustic_distance_m)
        print(f"Time gate: {time_gate.start_ms:.1f}ms to "
              f"{time_gate.start_ms + time_gate.duration_ms:.1f}ms")

    results = {}
    n_points = 1024
    carrier_cycles = 40  # ~40 kHz at 1 MSa/s playback

    for ptype in pulse_types:
        print(f"\nTesting {ptype}...")

        if ptype == "gaussian_narrow":
            waveform = generate_gaussian_pulse(n_points, sigma_frac=0.05,
                                              carrier_freq_cycles=carrier_cycles)
        elif ptype == "gaussian_wide":
            waveform = generate_gaussian_pulse(n_points, sigma_frac=0.15,
                                              carrier_freq_cycles=carrier_cycles)
        elif ptype == "rectangular":
            t = np.linspace(0, 1, n_points)
            envelope = np.where((t > 0.3) & (t < 0.7), 1.0, 0.0)
            carrier = np.sin(2 * np.pi * carrier_cycles * t)
            waveform = envelope * carrier
        elif ptype == "chirp":
            waveform = generate_chirp(n_points, carrier_cycles * 0.8,
                                     carrier_cycles * 1.2, envelope="gaussian")
        else:
            continue

        # Upload and test
        ws.upload_arbitrary_waveform(waveform, name="baseline_test")
        ws.set_arbitrary_waveform(channel=1, name="baseline_test",
                                 sample_rate_hz=1e6, amp_vpp=1.0)
        ws.set_output(channel=1, enable=True)
        time.sleep(0.05)

        audio = recorder.record(0.1)
        ws.set_output(channel=1, enable=False)

        metrics = analyze_pulse(audio, recorder.sample_rate, time_gate=time_gate)
        results[ptype] = metrics

        print(f"  Amplitude: {metrics.peak_amplitude:.4f}")
        print(f"  Width: {metrics.pulse_width_us:.1f} us")
        print(f"  Arrival: {metrics.arrival_time_us:.1f} us")
        print(f"  SNR: {metrics.snr_db:.1f} dB")

    return results


def plot_optimization_history(history: list, save_path: Optional[str] = None):
    """Plot optimization progress."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    iterations = range(1, len(history) + 1)
    amplitudes = [h['metrics'].peak_amplitude for h in history]
    widths = [h['metrics'].pulse_width_us for h in history]
    snrs = [h['metrics'].snr_db for h in history]
    energies = [h['metrics'].energy for h in history]

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

    # Scatter plot of amplitude vs width
    axes[1, 1].scatter(widths, amplitudes, c=iterations, cmap='viridis', alpha=0.7)
    axes[1, 1].set_xlabel('Pulse Width (µs)')
    axes[1, 1].set_ylabel('Peak Amplitude')
    axes[1, 1].set_title('Amplitude vs Width Trade-off')
    plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1], label='Iteration')
    axes[1, 1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        default_path = "optimization_history.png"
        plt.savefig(default_path, dpi=150)
        print(f"Saved plot to {default_path}")

    plt.close()


def plot_recording_with_gate(audio: np.ndarray, sample_rate: int,
                             time_gate: TimeGate, save_path: Optional[str] = None):
    """
    Plot full recording showing time gate region.

    Useful for verifying the gate is positioned correctly.
    """
    t_ms = np.arange(len(audio)) / sample_rate * 1000

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    # Full recording
    axes[0].plot(t_ms, audio, 'b-', linewidth=0.5)
    axes[0].axvspan(time_gate.start_ms, time_gate.start_ms + time_gate.duration_ms,
                    alpha=0.3, color='green', label='Time Gate')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Full Recording with Time Gate')
    axes[0].legend()
    axes[0].grid(True)

    # Zoomed to gate region (with some margin)
    margin = 2.0  # ms
    zoom_start = max(0, time_gate.start_ms - margin)
    zoom_end = time_gate.start_ms + time_gate.duration_ms + margin

    mask = (t_ms >= zoom_start) & (t_ms <= zoom_end)
    axes[1].plot(t_ms[mask], audio[mask], 'b-', linewidth=0.5)
    axes[1].axvspan(time_gate.start_ms, time_gate.start_ms + time_gate.duration_ms,
                    alpha=0.3, color='green', label='Time Gate')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Zoomed to Gate Region')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        # Default save location
        default_path = "recording_with_gate.png"
        plt.savefig(default_path, dpi=150)
        print(f"Saved plot to {default_path}")

    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='BATPING Pulse Optimizer')
    parser.add_argument('--distance', type=float, default=0.235,
                        help='Acoustic path distance in meters')
    parser.add_argument('--direct', action='store_true', default=True,
                        help='Direct path mode (transducer to mic, no reflection)')
    parser.add_argument('--echo', action='store_true',
                        help='Echo/reflection mode (round-trip path)')
    parser.add_argument('--gate-start', type=float, default=None,
                        help='Manual time gate start (ms)')
    parser.add_argument('--gate-duration', type=float, default=None,
                        help='Manual time gate duration (ms)')
    parser.add_argument('--baseline-only', action='store_true',
                        help='Run only baseline tests, no optimization')
    parser.add_argument('--plot-recording', action='store_true',
                        help='Plot a test recording showing time gate')

    args = parser.parse_args()

    print("BATPING Pulse Optimizer")
    print("=" * 50)

    # Set up time gate
    if args.gate_start is not None and args.gate_duration is not None:
        time_gate = TimeGate(start_ms=args.gate_start, duration_ms=args.gate_duration)
        print(f"\nManual time gate: {time_gate.start_ms:.2f}ms to "
              f"{time_gate.start_ms + time_gate.duration_ms:.2f}ms")
    elif args.echo:
        time_gate = TimeGate.from_distance(args.distance)
        print(f"\nEcho mode time gate for {args.distance}m round-trip:")
        print(f"  Gate: {time_gate.start_ms:.2f}ms to "
              f"{time_gate.start_ms + time_gate.duration_ms:.2f}ms")
    else:
        print(f"\nDirect path mode, separation: {args.distance}m")
        time_gate = TimeGate.for_direct_path(args.distance)

    # Connect to hardware
    print("\nConnecting to WaveStation...")
    ws = WaveStation()
    ws.connect()

    print("\nInitializing Ultramic recorder...")
    recorder = UltramicRecorder(sample_rate=384000)  # Auto-detect device

    # Calibration: send a test pulse and auto-detect actual arrival time
    print("\n" + "=" * 50)
    print("CALIBRATION: Auto-detecting signal arrival time")
    print("=" * 50)

    # Send a test pulse with wide recording window
    pulse = generate_gaussian_pulse(1024, sigma_frac=0.1, carrier_freq_cycles=40)
    ws.upload_arbitrary_waveform(pulse, name="calibration")
    ws.set_arbitrary_waveform(channel=1, name="calibration", sample_rate_hz=1e6, amp_vpp=1.0)
    ws.set_output(channel=1, enable=True)
    time.sleep(0.05)
    calibration_audio = recorder.record(0.05)  # 50ms recording
    ws.set_output(channel=1, enable=False)

    # Auto-detect where the signal actually is
    calibrated_gate = TimeGate.auto_detect(calibration_audio, recorder.sample_rate)
    time_gate = calibrated_gate  # Use auto-detected gate

    # Plot the recording with the calibrated gate
    if args.plot_recording:
        print("\nSaving calibration plot...")
        plot_recording_with_gate(calibration_audio, recorder.sample_rate, time_gate)

    # Run baseline tests
    print("\n" + "=" * 50)
    print("BASELINE TESTS")
    print("=" * 50)
    baseline_results = run_baseline_test(ws, recorder, time_gate=time_gate)

    # Summary
    print("\n" + "=" * 50)
    print("BASELINE SUMMARY")
    print("=" * 50)
    for ptype, metrics in baseline_results.items():
        print(f"{ptype:20s}: amp={metrics.peak_amplitude:.4f}, "
              f"width={metrics.pulse_width_us:.1f}us, "
              f"arrival={metrics.arrival_time_us:.0f}us, "
              f"snr={metrics.snr_db:.1f}dB")

    # Clean up
    ws.disconnect()
    recorder.close()
    print("\nDone!")
