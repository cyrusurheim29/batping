"""
Compare optimized waveform to baseline.

Measures actual FWHM and amplitude for both waveforms with
proper settling time and multiple measurements for statistics.
"""

import numpy as np
import pyaudio
import time
from datetime import datetime
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from agilent_33220a import Agilent33220A


class WaveformComparator:
    """Compare two waveforms with proper measurement methodology."""

    def __init__(
        self,
        n_points: int = 4000,
        carrier_freq: float = 40000,
        sample_rate_awg: float = 1e6,
        sample_rate_mic: int = 384000,
        drive_amplitude: float = 5.0,
        settling_time: float = 0.3,  # Longer settling for accurate comparison
        record_duration: float = 0.05,  # Longer recording
        n_measurements: int = 5,  # Multiple measurements for statistics
    ):
        self.n_points = n_points
        self.carrier_freq = carrier_freq
        self.sample_rate_awg = sample_rate_awg
        self.sample_rate_mic = sample_rate_mic
        self.drive_amplitude = drive_amplitude
        self.settling_time = settling_time
        self.record_duration = record_duration
        self.n_measurements = n_measurements

        self.rep_freq = 1.0 / (n_points / sample_rate_awg)

        # Hardware
        self.awg = None
        self.pa = None
        self.mic_index = None

        # Filter
        nyq = sample_rate_mic / 2
        low = 38000 / nyq
        high = min(42000 / nyq, 0.99)
        self.filter_b, self.filter_a = signal.butter(4, [low, high], btype='band')

    def connect(self):
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
        if self.awg:
            self.awg.set_output(False)
            self.awg.disconnect()
        if self.pa:
            self.pa.terminate()

    def record(self) -> np.ndarray:
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
        filtered = signal.filtfilt(self.filter_b, self.filter_a, audio)
        analytic = signal.hilbert(filtered)
        return np.abs(analytic)

    def measure_fwhm(self, envelope: np.ndarray) -> tuple:
        """Measure FWHM and peak amplitude."""
        peak_idx = np.argmax(envelope)
        peak_amp = envelope[peak_idx]
        half_max = peak_amp / 2

        # Find left crossing
        left_idx = peak_idx
        for i in range(peak_idx, -1, -1):
            if envelope[i] < half_max:
                left_idx = i
                break

        # Find right crossing
        right_idx = peak_idx
        for i in range(peak_idx, len(envelope)):
            if envelope[i] < half_max:
                right_idx = i
                break

        fwhm_samples = right_idx - left_idx
        fwhm_us = fwhm_samples / self.sample_rate_mic * 1e6

        return peak_amp, fwhm_us, peak_idx

    def measure_waveform(self, waveform: np.ndarray, name: str) -> dict:
        """Take multiple measurements and return statistics."""
        # Normalize
        max_abs = np.max(np.abs(waveform))
        if max_abs > 1.0:
            waveform = waveform / max_abs

        print(f"\nMeasuring {name}...")
        self.awg.output_arbitrary(waveform, freq_hz=self.rep_freq, vpp=self.drive_amplitude)
        time.sleep(self.settling_time)

        amplitudes = []
        fwhms = []
        envelopes = []

        for i in range(self.n_measurements):
            audio = self.record()
            envelope = self.extract_envelope(audio)
            peak_amp, fwhm_us, _ = self.measure_fwhm(envelope)

            amplitudes.append(peak_amp)
            fwhms.append(fwhm_us)
            envelopes.append(envelope)

            print(f"  {i+1}/{self.n_measurements}: amp={peak_amp:.4f}, FWHM={fwhm_us:.1f}μs")
            time.sleep(0.1)

        self.awg.set_output(False)

        return {
            'name': name,
            'amplitude_mean': np.mean(amplitudes),
            'amplitude_std': np.std(amplitudes),
            'fwhm_mean': np.mean(fwhms),
            'fwhm_std': np.std(fwhms),
            'envelopes': envelopes,
            'waveform': waveform,
        }


def create_gaussian_baseline(n_points, carrier_freq, sample_rate, sigma_cycles=5):
    """Create Gaussian pulse baseline."""
    t = np.arange(n_points) / sample_rate
    t_center = t[n_points // 2]
    sigma_t = sigma_cycles / carrier_freq

    envelope = np.exp(-0.5 * ((t - t_center) / sigma_t) ** 2)
    carrier = np.sin(2 * np.pi * carrier_freq * t)

    return envelope * carrier


def load_optimized_waveform(state_path='aligned_optimizer_state.npz'):
    """Load optimized pulse and embed in full waveform."""
    data = np.load(state_path)
    pulse_best = data['f_best']

    # Parameters from optimizer
    n_points = 4000
    pre_silence = 500
    post_silence = 800
    taper_samples = 50

    # Build taper
    pulse_size = len(pulse_best)
    taper = np.ones(pulse_size)
    taper_in = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, taper_samples)))
    taper_out = 0.5 * (1 + np.cos(np.pi * np.linspace(0, 1, taper_samples)))
    taper[:taper_samples] = taper_in
    taper[-taper_samples:] = taper_out

    # Build full waveform
    waveform = np.zeros(n_points)
    waveform[pre_silence:pre_silence + pulse_size] = pulse_best * taper

    return waveform


def plot_comparison(baseline_result, optimized_result, save_path='waveform_comparison.png'):
    """Generate comparison plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    t_input = np.arange(len(baseline_result['waveform'])) / 1e6 * 1000  # ms

    # Input waveforms
    axes[0, 0].plot(t_input, baseline_result['waveform'], 'b-', lw=0.5, alpha=0.7, label='Gaussian baseline')
    axes[0, 0].plot(t_input, optimized_result['waveform'], 'r-', lw=0.5, alpha=0.7, label='Optimized')
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title('Input Waveforms')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Output envelopes (overlay all measurements)
    t_output = np.arange(len(baseline_result['envelopes'][0])) / 384000 * 1000  # ms

    for env in baseline_result['envelopes']:
        axes[0, 1].plot(t_output, env, 'b-', alpha=0.3, lw=0.5)
    for env in optimized_result['envelopes']:
        axes[0, 1].plot(t_output, env, 'r-', alpha=0.3, lw=0.5)

    # Plot mean
    baseline_mean = np.mean(baseline_result['envelopes'], axis=0)
    optimized_mean = np.mean(optimized_result['envelopes'], axis=0)
    axes[0, 1].plot(t_output, baseline_mean, 'b-', lw=2, label=f"Baseline (FWHM={baseline_result['fwhm_mean']:.0f}μs)")
    axes[0, 1].plot(t_output, optimized_mean, 'r-', lw=2, label=f"Optimized (FWHM={optimized_result['fwhm_mean']:.0f}μs)")
    axes[0, 1].set_xlabel('Time (ms)')
    axes[0, 1].set_ylabel('Envelope')
    axes[0, 1].set_title('Acoustic Output Envelopes')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Bar chart comparison
    metrics = ['Amplitude', 'FWHM (μs)']
    baseline_vals = [baseline_result['amplitude_mean'], baseline_result['fwhm_mean']]
    baseline_errs = [baseline_result['amplitude_std'], baseline_result['fwhm_std']]
    optimized_vals = [optimized_result['amplitude_mean'], optimized_result['fwhm_mean']]
    optimized_errs = [optimized_result['amplitude_std'], optimized_result['fwhm_std']]

    x = np.arange(len(metrics))
    width = 0.35

    axes[1, 0].bar(x - width/2, baseline_vals, width, yerr=baseline_errs, label='Baseline', color='blue', alpha=0.7)
    axes[1, 0].bar(x + width/2, optimized_vals, width, yerr=optimized_errs, label='Optimized', color='red', alpha=0.7)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(metrics)
    axes[1, 0].set_title('Metrics Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Summary text
    amp_change = (optimized_result['amplitude_mean'] / baseline_result['amplitude_mean'] - 1) * 100
    fwhm_change = (optimized_result['fwhm_mean'] / baseline_result['fwhm_mean'] - 1) * 100

    summary = f"""
    Comparison Summary
    ==================

    Baseline (Gaussian σ=5 cycles):
      Amplitude: {baseline_result['amplitude_mean']:.4f} ± {baseline_result['amplitude_std']:.4f}
      FWHM: {baseline_result['fwhm_mean']:.1f} ± {baseline_result['fwhm_std']:.1f} μs

    Optimized (iter 5 best):
      Amplitude: {optimized_result['amplitude_mean']:.4f} ± {optimized_result['amplitude_std']:.4f}
      FWHM: {optimized_result['fwhm_mean']:.1f} ± {optimized_result['fwhm_std']:.1f} μs

    Changes:
      Amplitude: {amp_change:+.1f}%
      FWHM: {fwhm_change:+.1f}%
    """

    axes[1, 1].text(0.1, 0.5, summary, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='center', fontfamily='monospace')
    axes[1, 1].axis('off')

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    fig.suptitle(f'Waveform Comparison — {timestamp}', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nSaved: {save_path}")

    return amp_change, fwhm_change


def main():
    print("=" * 60)
    print("Waveform Comparison: Baseline vs Optimized")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    comparator = WaveformComparator(
        settling_time=0.3,
        record_duration=0.05,
        n_measurements=5,
    )

    # Create waveforms
    baseline_waveform = create_gaussian_baseline(
        n_points=4000,
        carrier_freq=40000,
        sample_rate=1e6,
        sigma_cycles=5,
    )

    optimized_waveform = load_optimized_waveform('aligned_optimizer_state.npz')

    try:
        comparator.connect()

        # Measure both - baseline first, then optimized, then baseline again
        # to check for drift
        print("\n--- First baseline measurement ---")
        baseline1 = comparator.measure_waveform(baseline_waveform, "Baseline (1)")

        print("\n--- Optimized measurement ---")
        optimized = comparator.measure_waveform(optimized_waveform, "Optimized")

        print("\n--- Second baseline measurement (drift check) ---")
        baseline2 = comparator.measure_waveform(baseline_waveform, "Baseline (2)")

        # Average the two baseline measurements
        baseline_result = {
            'name': 'Baseline',
            'amplitude_mean': (baseline1['amplitude_mean'] + baseline2['amplitude_mean']) / 2,
            'amplitude_std': np.sqrt(baseline1['amplitude_std']**2 + baseline2['amplitude_std']**2) / 2,
            'fwhm_mean': (baseline1['fwhm_mean'] + baseline2['fwhm_mean']) / 2,
            'fwhm_std': np.sqrt(baseline1['fwhm_std']**2 + baseline2['fwhm_std']**2) / 2,
            'envelopes': baseline1['envelopes'] + baseline2['envelopes'],
            'waveform': baseline_waveform,
        }

        # Check for drift
        drift = abs(baseline1['amplitude_mean'] - baseline2['amplitude_mean']) / baseline1['amplitude_mean'] * 100
        print(f"\nDrift check: {drift:.1f}% amplitude change between baseline measurements")
        if drift > 10:
            print("WARNING: Significant drift detected!")

        # Plot comparison
        amp_change, fwhm_change = plot_comparison(baseline_result, optimized, 'waveform_comparison.png')

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Amplitude change: {amp_change:+.1f}%")
        print(f"FWHM change: {fwhm_change:+.1f}%")

        if fwhm_change < 0:
            print(f"\n*** SUCCESS: Pulse narrowed by {-fwhm_change:.1f}% ***")
        else:
            print(f"\n*** Pulse widened by {fwhm_change:.1f}% ***")

    finally:
        comparator.disconnect()


if __name__ == "__main__":
    main()
