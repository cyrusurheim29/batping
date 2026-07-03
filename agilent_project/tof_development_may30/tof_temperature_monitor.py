"""
TOF Temperature Monitor for BATPING

Measures time-of-flight of ultrasonic pulses to infer temperature changes
via the speed of sound relationship: c(T) = 331.3 + 0.6*T m/s

Uses matched filtering (cross-correlation) for precise TOF estimation
with sub-sample interpolation.
"""

import numpy as np
import pyaudio
import time
from datetime import datetime
from scipy import signal
from scipy.fft import fft, ifft
from typing import Optional, Tuple, List
from dataclasses import dataclass
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from agilent_33220a import Agilent33220A, generate_gaussian_pulse


@dataclass
class TOFMeasurement:
    """Single TOF measurement result."""
    tof_s: float
    peak_amplitude: float
    correlation_snr: float
    timestamp: float


@dataclass
class RepeatabilityResult:
    """Results from repeatability test."""
    tof_mean_us: float
    tof_std_us: float
    temp_noise_c: float
    n_measurements: int
    all_tofs_us: np.ndarray
    pass_threshold: bool


def parabolic_interpolation(y: np.ndarray, peak_idx: int) -> float:
    """
    Sub-sample peak refinement using parabolic interpolation.

    Fits parabola to peak and two neighbors, returns refined peak location.

    Args:
        y: Signal array
        peak_idx: Index of discrete peak

    Returns:
        Refined peak location (float index)
    """
    if peak_idx <= 0 or peak_idx >= len(y) - 1:
        return float(peak_idx)

    y0, y1, y2 = y[peak_idx - 1], y[peak_idx], y[peak_idx + 1]

    # Parabola vertex: x = -b/(2a) where y = ax^2 + bx + c
    # With points at x=-1, 0, +1: a = (y0 - 2*y1 + y2)/2, b = (y2 - y0)/2
    denom = 2 * (2 * y1 - y0 - y2)
    if abs(denom) < 1e-10:
        return float(peak_idx)

    offset = (y0 - y2) / denom
    return peak_idx + offset


def xcorr_peak(template: np.ndarray, signal_data: np.ndarray) -> Tuple[float, float, float]:
    """
    Standard cross-correlation with parabolic interpolation.

    Args:
        template: Template signal (from averaging)
        signal_data: New measurement to correlate

    Returns:
        (peak_lag, peak_value, snr)
    """
    # Normalize
    template_norm = template - np.mean(template)
    signal_norm = signal_data - np.mean(signal_data)

    # Cross-correlation via FFT
    n = len(template_norm) + len(signal_norm) - 1
    n_fft = 2 ** int(np.ceil(np.log2(n)))

    T = fft(template_norm, n_fft)
    S = fft(signal_norm, n_fft)
    R = ifft(np.conj(T) * S).real

    # Find peak
    peak_idx = np.argmax(R)
    peak_val = R[peak_idx]

    # Sub-sample refinement
    refined_idx = parabolic_interpolation(R, peak_idx)

    # SNR: peak / RMS of correlation outside peak region
    mask = np.ones(len(R), dtype=bool)
    exclude_width = max(100, len(R) // 20)
    start = max(0, peak_idx - exclude_width)
    end = min(len(R), peak_idx + exclude_width)
    mask[start:end] = False

    if np.sum(mask) > 10:
        noise_rms = np.sqrt(np.mean(R[mask]**2))
        snr = peak_val / noise_rms if noise_rms > 0 else 0
    else:
        snr = 0

    return refined_idx, peak_val, snr


def gcc_phat(template: np.ndarray, signal_data: np.ndarray,
             fs: int, freq_range: Optional[Tuple[float, float]] = None) -> Tuple[float, float, float]:
    """
    Generalized Cross-Correlation with Phase Transform (GCC-PHAT).

    Whitens the spectrum for sharper correlation peak.

    Args:
        template: Template signal
        signal_data: New measurement
        fs: Sample rate
        freq_range: Optional (low, high) Hz to restrict PHAT weighting

    Returns:
        (peak_lag, peak_value, snr)
    """
    n = len(template) + len(signal_data) - 1
    n_fft = 2 ** int(np.ceil(np.log2(n)))

    T = fft(template, n_fft)
    S = fft(signal_data, n_fft)

    # Cross-spectrum
    G = np.conj(T) * S

    # PHAT weighting (normalize by magnitude)
    mag = np.abs(G)

    if freq_range is not None:
        # Only apply PHAT in specified frequency range
        freqs = np.fft.fftfreq(n_fft, 1/fs)
        mask = (np.abs(freqs) >= freq_range[0]) & (np.abs(freqs) <= freq_range[1])

        G_phat = np.zeros_like(G)
        G_phat[mask] = G[mask] / (mag[mask] + 1e-10)
        # Keep original outside range (or zero)
        G_phat[~mask] = 0
    else:
        G_phat = G / (mag + 1e-10)

    R = ifft(G_phat).real

    # Find peak
    peak_idx = np.argmax(R)
    peak_val = R[peak_idx]

    # Sub-sample refinement
    refined_idx = parabolic_interpolation(R, peak_idx)

    # SNR estimate
    mask = np.ones(len(R), dtype=bool)
    exclude_width = max(100, len(R) // 20)
    start = max(0, peak_idx - exclude_width)
    end = min(len(R), peak_idx + exclude_width)
    mask[start:end] = False

    if np.sum(mask) > 10:
        noise_rms = np.sqrt(np.mean(R[mask]**2))
        snr = peak_val / noise_rms if noise_rms > 0 else 0
    else:
        snr = 0

    return refined_idx, peak_val, snr


class TOFTemperatureMonitor:
    """
    Time-of-flight based temperature monitor using matched filtering.

    Uses cross-correlation between a template pulse and new measurements
    to precisely estimate TOF changes, which map to temperature changes.
    """

    def __init__(
        self,
        path_length_m: float = 2.812,
        reference_temp_c: float = 20.0,
        sample_rate_mic: int = 384000,
        sample_rate_awg: float = 1e6,
        n_waveform_points: int = 4000,
        carrier_freq: float = 40000,
        drive_amplitude: float = 10.0,
        rep_freq: float = 50.0,
        settling_time: float = 0.3,
        record_duration: float = 0.025,
        time_gate_ms: Optional[Tuple[float, float]] = None,
    ):
        """
        Initialize TOF monitor.

        Args:
            path_length_m: Round-trip acoustic path length in meters
            reference_temp_c: Reference temperature for calculations
            sample_rate_mic: Microphone sample rate (Hz)
            sample_rate_awg: AWG sample rate (Sa/s)
            n_waveform_points: Points in AWG waveform
            carrier_freq: Ultrasonic carrier frequency (Hz)
            drive_amplitude: AWG output amplitude (Vpp)
            rep_freq: Pulse repetition frequency (Hz)
            settling_time: Wait time after AWG output before recording (s)
            record_duration: Recording duration (s)
            time_gate_ms: Optional (start, end) time gate in ms to isolate reflection
        """
        self.path_length = path_length_m
        self.T_ref = reference_temp_c
        self.sample_rate_mic = sample_rate_mic
        self.sample_rate_awg = sample_rate_awg
        self.n_waveform_points = n_waveform_points
        self.carrier_freq = carrier_freq
        self.drive_amplitude = drive_amplitude
        self.rep_freq = rep_freq
        self.settling_time = settling_time
        self.record_duration = record_duration

        # Speed of sound at reference temp
        self.c_ref = 331.3 + 0.6 * reference_temp_c

        # Expected TOF
        self.expected_tof_s = path_length_m / self.c_ref

        # Template storage
        self.template: Optional[np.ndarray] = None
        self.template_raw: Optional[np.ndarray] = None
        self.template_tof_s: Optional[float] = None
        self.input_waveform: Optional[np.ndarray] = None

        # Hardware
        self.awg: Optional[Agilent33220A] = None
        self.pa: Optional[pyaudio.PyAudio] = None
        self.mic_index: Optional[int] = None

        # Time gate for isolating reflection from direct signal
        if time_gate_ms is not None:
            self.time_gate_ms = time_gate_ms
        else:
            # Default: gate around expected TOF with margin
            expected_tof_ms = path_length_m / (331.3 + 0.6 * reference_temp_c) * 1000
            self.time_gate_ms = (expected_tof_ms - 2.0, expected_tof_ms + 4.0)

        # Bandpass filter for envelope extraction
        nyq = sample_rate_mic / 2
        low = 38000 / nyq
        high = min(42000 / nyq, 0.99)
        self.filter_b, self.filter_a = signal.butter(4, [low, high], btype='band')

        print(f"TOF Temperature Monitor initialized:")
        print(f"  Path length: {path_length_m:.3f} m")
        print(f"  Expected TOF: {self.expected_tof_s*1000:.2f} ms")
        print(f"  Time gate: {self.time_gate_ms[0]:.1f} - {self.time_gate_ms[1]:.1f} ms")
        print(f"  Reference temp: {reference_temp_c}°C")
        print(f"  Rep rate: {rep_freq} Hz ({1000/rep_freq:.1f} ms period)")

    def connect(self):
        """Connect to hardware."""
        print("\nConnecting to Agilent 33220A...")
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
        """Extract envelope via bandpass + Hilbert transform."""
        filtered = signal.filtfilt(self.filter_b, self.filter_a, audio)
        analytic = signal.hilbert(filtered)
        return np.abs(analytic)

    def set_input_waveform(self, waveform: np.ndarray):
        """Set the input waveform to use."""
        self.input_waveform = waveform
        print(f"Input waveform set: {len(waveform)} points")

    def load_optimized_waveform(self, path: str = 'bayesian_optimizer_state.npz'):
        """Load the Bayesian-optimized waveform."""
        data = np.load(path)
        self.input_waveform = data['best_waveform']
        print(f"Loaded optimized waveform from {path}")
        print(f"  Best FWHM was: {data['best_fwhm']:.1f} μs")

    def create_gaussian_waveform(self, sigma_cycles: float = 5.0):
        """Create a Gaussian pulse waveform."""
        self.input_waveform = generate_gaussian_pulse(
            n_points=self.n_waveform_points,
            center_freq_hz=self.carrier_freq,
            sample_rate_hz=self.sample_rate_awg,
            sigma_cycles=sigma_cycles
        )
        print(f"Created Gaussian waveform with σ={sigma_cycles} cycles")

    def measure_once(self) -> np.ndarray:
        """Send pulse and record response."""
        if self.input_waveform is None:
            raise RuntimeError("No input waveform set")

        self.awg.output_arbitrary(
            self.input_waveform,
            freq_hz=self.rep_freq,
            vpp=self.drive_amplitude
        )
        time.sleep(self.settling_time)

        audio = self.record()
        self.awg.set_output(False)

        return audio

    def apply_time_gate(self, data: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Apply time gate to isolate reflection from direct signal.

        Returns:
            (gated_data, gate_start_idx) - gated data and the starting index
        """
        gate_start_idx = int(self.time_gate_ms[0] / 1000 * self.sample_rate_mic)
        gate_end_idx = int(self.time_gate_ms[1] / 1000 * self.sample_rate_mic)
        gate_start_idx = max(0, gate_start_idx)
        gate_end_idx = min(len(data), gate_end_idx)
        return data[gate_start_idx:gate_end_idx], gate_start_idx

    def create_template(self, n_avg: int = 10, use_envelope: bool = True) -> np.ndarray:
        """
        Create template by averaging multiple measurements.

        Args:
            n_avg: Number of measurements to average
            use_envelope: If True, average envelopes; if False, average raw signals

        Returns:
            Template array (time-gated to reflection region)
        """
        print(f"\nCreating template from {n_avg} measurements...")
        print(f"  Time gate: {self.time_gate_ms[0]:.1f} - {self.time_gate_ms[1]:.1f} ms")

        if self.input_waveform is None:
            print("  No waveform set, using Gaussian σ=5")
            self.create_gaussian_waveform(5.0)

        measurements = []

        for i in range(n_avg):
            audio = self.measure_once()
            if use_envelope:
                processed = self.extract_envelope(audio)
            else:
                # Bandpass filter only
                processed = signal.filtfilt(self.filter_b, self.filter_a, audio)

            # Apply time gate
            gated, gate_start = self.apply_time_gate(processed)
            measurements.append(gated)

            peak_in_gate = np.max(gated)
            print(f"  {i+1}/{n_avg}: peak in gate={peak_in_gate:.4f}")

        # Average
        self.template_raw = measurements
        self.template = np.mean(measurements, axis=0)
        self.gate_start_idx = gate_start

        # Estimate template TOF from peak location (relative to gate start)
        local_peak_idx = np.argmax(self.template)
        global_peak_idx = local_peak_idx + self.gate_start_idx
        self.template_tof_s = global_peak_idx / self.sample_rate_mic

        print(f"Template created:")
        print(f"  Peak at {self.template_tof_s*1000:.3f} ms")
        print(f"  Expected TOF: {self.expected_tof_s*1000:.2f} ms")
        print(f"  Peak amplitude: {np.max(self.template):.4f}")

        return self.template

    def measure_tof(self, method: str = 'xcorr') -> TOFMeasurement:
        """
        Measure TOF using cross-correlation with template.

        Args:
            method: 'xcorr' for standard, 'gcc-phat' for phase transform

        Returns:
            TOFMeasurement with TOF and diagnostics
        """
        if self.template is None:
            raise RuntimeError("No template created. Call create_template() first.")

        # Record and apply time gate
        audio = self.measure_once()
        envelope = self.extract_envelope(audio)
        measured, _ = self.apply_time_gate(envelope)

        # Cross-correlate (both template and measured are gated)
        if method == 'xcorr':
            lag, peak_val, snr = xcorr_peak(self.template, measured)
        elif method == 'gcc-phat':
            lag, peak_val, snr = gcc_phat(
                self.template, measured,
                self.sample_rate_mic,
                freq_range=(38000, 42000)
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # The lag tells us how much the measured signal is shifted relative to template
        # Positive lag = measured arrives later, negative = earlier
        # Template TOF is self.template_tof_s
        # Measured TOF = template_tof_s + lag_adjustment

        # For xcorr: peak at index N means measured is shifted by (N - len(template) + 1) samples
        # But simpler: just find peak in measured directly for absolute TOF
        local_peak_idx = np.argmax(measured)
        global_peak_idx = local_peak_idx + self.gate_start_idx
        measured_tof_s = global_peak_idx / self.sample_rate_mic

        return TOFMeasurement(
            tof_s=measured_tof_s,
            peak_amplitude=np.max(measured),
            correlation_snr=snr,
            timestamp=time.time()
        )

    def tof_to_delta_temp(self, tof_s: float) -> float:
        """
        Convert TOF to temperature change from reference.

        Using: c = d/τ and c(T) = 331.3 + 0.6*T

        ΔT = (c_measured - c_ref) / 0.6
        """
        c_measured = self.path_length / tof_s
        delta_T = (c_measured - self.c_ref) / 0.6
        return delta_T

    def run_repeatability_test(
        self,
        n_measurements: int = 100,
        method: str = 'xcorr',
        interval_s: float = 0.5
    ) -> RepeatabilityResult:
        """
        Run repeatability test to characterize noise floor.

        Args:
            n_measurements: Number of measurements
            method: Correlation method
            interval_s: Time between measurements

        Returns:
            RepeatabilityResult with statistics
        """
        print(f"\n{'='*60}")
        print(f"Repeatability Test: {n_measurements} measurements")
        print(f"Method: {method}")
        print(f"{'='*60}")

        if self.template is None:
            print("Creating template first...")
            self.create_template(n_avg=10)

        tofs = []

        for i in range(n_measurements):
            result = self.measure_tof(method=method)
            tofs.append(result.tof_s * 1e6)  # Convert to μs

            if (i + 1) % 10 == 0:
                current_std = np.std(tofs)
                print(f"  {i+1}/{n_measurements}: TOF={result.tof_s*1e6:.2f}μs, "
                      f"σ={current_std:.3f}μs so far")

            if interval_s > 0:
                time.sleep(interval_s)

        tofs = np.array(tofs)
        tof_mean = np.mean(tofs)
        tof_std = np.std(tofs)

        # Convert to temperature noise
        # σ_T = σ_τ * c²/(0.6*d)
        temp_noise = tof_std * 1e-6 * (self.c_ref**2) / (0.6 * self.path_length)

        # Check against 0.1°C threshold
        threshold_tof_us = 0.1 * 0.6 * self.path_length / (self.c_ref**2) * 1e6
        passed = tof_std < threshold_tof_us

        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"TOF mean: {tof_mean:.2f} μs ({tof_mean/1000:.3f} ms)")
        print(f"TOF std:  {tof_std:.3f} μs")
        print(f"Temperature noise: {temp_noise:.4f} °C")
        print(f"Required for 0.1°C: σ_τ < {threshold_tof_us:.3f} μs")
        print(f"PASS: {'YES' if passed else 'NO'}")

        return RepeatabilityResult(
            tof_mean_us=tof_mean,
            tof_std_us=tof_std,
            temp_noise_c=temp_noise,
            n_measurements=n_measurements,
            all_tofs_us=tofs,
            pass_threshold=passed
        )


def plot_results(result: RepeatabilityResult, template: np.ndarray,
                 sample_rate: int, save_path: str = 'tof_repeatability.png'):
    """Plot repeatability test results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # TOF time series
    axes[0, 0].plot(result.all_tofs_us, 'b.-', markersize=2, linewidth=0.5)
    axes[0, 0].axhline(result.tof_mean_us, color='r', linestyle='--',
                       label=f'Mean: {result.tof_mean_us:.2f} μs')
    axes[0, 0].axhline(result.tof_mean_us + result.tof_std_us, color='orange',
                       linestyle=':', label=f'±1σ: {result.tof_std_us:.3f} μs')
    axes[0, 0].axhline(result.tof_mean_us - result.tof_std_us, color='orange', linestyle=':')
    axes[0, 0].set_xlabel('Measurement #')
    axes[0, 0].set_ylabel('TOF (μs)')
    axes[0, 0].set_title('TOF Time Series')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Histogram
    axes[0, 1].hist(result.all_tofs_us, bins=30, density=True, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(result.tof_mean_us, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('TOF (μs)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title(f'TOF Distribution (σ = {result.tof_std_us:.3f} μs)')
    axes[0, 1].grid(True, alpha=0.3)

    # Template waveform
    t_ms = np.arange(len(template)) / sample_rate * 1000
    axes[1, 0].plot(t_ms, template, 'b-', linewidth=0.5)
    peak_idx = np.argmax(template)
    peak_t = peak_idx / sample_rate * 1000
    axes[1, 0].axvline(peak_t, color='r', linestyle='--', alpha=0.7,
                       label=f'Peak: {peak_t:.2f} ms')
    axes[1, 0].set_xlabel('Time (ms)')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].set_title('Template Waveform')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Summary text
    summary = f"""
    TOF Repeatability Test Results
    {'='*40}

    Measurements: {result.n_measurements}

    TOF Statistics:
      Mean: {result.tof_mean_us:.2f} μs ({result.tof_mean_us/1000:.3f} ms)
      Std:  {result.tof_std_us:.3f} μs

    Temperature Equivalent:
      Noise: {result.temp_noise_c:.4f} °C RMS

    0.1°C Resolution: {'PASS' if result.pass_threshold else 'FAIL'}

    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """

    axes[1, 1].text(0.05, 0.95, summary, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nSaved: {save_path}")


def main():
    """Run TOF characterization."""
    print("="*60)
    print("BATPING TOF Temperature Monitor")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Configuration for reflector setup: 1m one-way, 2m round-trip
    # Expected TOF: 2m / 343 m/s ≈ 5.83 ms
    monitor = TOFTemperatureMonitor(
        path_length_m=2.0,        # Round-trip: 2 × 1.0 m
        reference_temp_c=20.0,
        rep_freq=50.0,            # 20 ms period (> 5.83 ms TOF)
        settling_time=0.2,
        record_duration=0.015,    # 15 ms recording
        time_gate_ms=(4.0, 8.0),  # Gate around expected 5.83 ms TOF
    )

    try:
        monitor.connect()

        # Start with baseline Gaussian
        print("\n" + "="*60)
        print("TEST 1: Gaussian σ=5 waveform")
        print("="*60)

        monitor.create_gaussian_waveform(sigma_cycles=5.0)
        monitor.create_template(n_avg=10)

        # Quick repeatability test
        result = monitor.run_repeatability_test(
            n_measurements=50,
            method='xcorr',
            interval_s=0.3
        )

        # Plot
        plot_results(
            result,
            monitor.template,
            monitor.sample_rate_mic,
            'tof_repeatability_gaussian.png'
        )

        # Save data
        np.savez(
            'tof_repeatability_data.npz',
            tofs_us=result.all_tofs_us,
            tof_mean_us=result.tof_mean_us,
            tof_std_us=result.tof_std_us,
            temp_noise_c=result.temp_noise_c,
            template=monitor.template,
            path_length_m=monitor.path_length,
        )
        print("Saved: tof_repeatability_data.npz")

    finally:
        monitor.disconnect()

    print("\nDone!")


if __name__ == "__main__":
    main()
