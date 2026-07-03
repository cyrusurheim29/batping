#!/usr/bin/env python3
"""
Bayesian Optimization for Piezo Pulse Shaping - SDG1062X Version

Optimizes AUTOCORRELATION FWHM of reflection pulses using SDG1062X
waveform generator via LAN connection.

Why autocorrelation instead of cross-correlation:
- Avoids all alignment issues (peak is always at lag=0)
- Directly measures intrinsic pulse sharpness/bandwidth
- Simpler interpretation: narrower autocorr = sharper pulse = better TOF precision

Key differences from Agilent version:
1. SDG1062X via LAN socket (not USB VISA)
2. 6400 points @ 400 kSa/s for 62.5 Hz rep rate (not 4000 @ 1 MSa/s)
3. Objective: autocorrelation FWHM of reflection pulses
"""

import numpy as np
import pyaudio
import socket
import time
from datetime import datetime
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real
from typing import Optional, List, Tuple
from dataclasses import dataclass


# === Configuration ===
SDG_IP = "10.11.13.230"
SDG_PORT = 5025

# Waveform parameters
N_POINTS = 6400
CARRIER_FREQ = 40000
SAMPLE_RATE_AWG = 400000  # 400 kSa/s for 62.5 Hz @ 6400 points
SAMPLE_RATE_MIC = 384000
REP_RATE = 62.5  # Hz

# Recording
RECORD_DURATION = 0.5  # seconds (~31 pulses)
SETTLING_TIME = 0.3  # seconds after waveform upload

# Amplitude constraint
MIN_AMPLITUDE_FRACTION = 0.7
AMPLITUDE_PENALTY_WEIGHT = 1000.0


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


class SDG1062X_LAN:
    """SDG1062X control via LAN socket."""

    def __init__(self, ip=SDG_IP, port=SDG_PORT, timeout=10.0):
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self.sock = None

    def connect(self, retries=3):
        for attempt in range(retries):
            try:
                if self.sock:
                    self.sock.close()
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(self.timeout)
                self.sock.connect((self.ip, self.port))
                print(f"Connected to SDG @ {self.ip}:{self.port}")
                return
            except socket.timeout:
                print(f"Connection attempt {attempt+1}/{retries} timed out, retrying...")
                time.sleep(1)
        raise ConnectionError(f"Failed to connect to SDG after {retries} attempts")

    def disconnect(self):
        if self.sock:
            self.sock.close()
            self.sock = None

    def write(self, cmd):
        self.sock.sendall((cmd + '\n').encode('utf-8'))

    def write_raw(self, data):
        self.sock.sendall(data)

    def read(self, bufsize=4096):
        return self.sock.recv(bufsize).decode('utf-8').strip()

    def query(self, cmd, retries=2):
        for attempt in range(retries):
            try:
                self.write(cmd)
                time.sleep(0.15)
                return self.read()
            except socket.timeout:
                if attempt < retries - 1:
                    print(f"Query timeout, reconnecting...")
                    self.connect(retries=1)
                else:
                    raise

    def upload_waveform(self, name, wave_data, freq_hz=REP_RATE, ampl_vpp=2.0, channel=1):
        """Upload waveform with little-endian 16-bit encoding."""
        wave_int16 = np.clip(wave_data, -1.0, 1.0)
        wave_int16 = (wave_int16 * 32767).astype(np.int16)
        bin_data = wave_int16.astype('<i2').tobytes()

        cmd_prefix = f"C{channel}:WVDT WVNM,{name},FREQ,{freq_hz},AMPL,{ampl_vpp},OFST,0,PHASE,0,WAVEDATA,"
        full_cmd = cmd_prefix.encode('utf-8') + bin_data + b'\n'

        self.write_raw(full_cmd)
        time.sleep(0.5)

        self.write(f"C{channel}:ARWV NAME,{name}")
        time.sleep(0.2)

    def set_arb_mode(self, channel=1):
        self.write(f"C{channel}:BSWV WVTP,ARB")
        time.sleep(0.2)

    def output_on(self, channel=1):
        self.write(f"C{channel}:OUTP ON")

    def output_off(self, channel=1):
        self.write(f"C{channel}:OUTP OFF")


class BayesianXCorrOptimizer:
    """
    Bayesian optimization for piezo pulse shaping targeting xcorr FWHM.
    """

    def __init__(self):
        self.n_points = N_POINTS
        self.carrier_freq = CARRIER_FREQ
        self.sample_rate_awg = SAMPLE_RATE_AWG
        self.sample_rate_mic = SAMPLE_RATE_MIC
        self.rep_rate = REP_RATE
        self.record_duration = RECORD_DURATION
        self.settling_time = SETTLING_TIME

        # Hardware
        self.sdg: Optional[SDG1062X_LAN] = None
        self.pa: Optional[pyaudio.PyAudio] = None
        self.mic_index: Optional[int] = None

        # Bandpass filter for 40kHz
        nyq = self.sample_rate_mic / 2
        self.filter_b, self.filter_a = signal.butter(4, [35000/nyq, 45000/nyq], btype='band')

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
        print("Connecting to SDG1062X...")
        self.sdg = SDG1062X_LAN()
        self.sdg.connect()

        idn = self.sdg.query("*IDN?")
        print(f"Device: {idn}")

        self.sdg.set_arb_mode(channel=1)

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
        if self.sdg:
            self.sdg.output_off(channel=1)
            self.sdg.disconnect()
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

    def find_reflection_peaks(self, envelope: np.ndarray) -> np.ndarray:
        """Find strong reflection peaks (ignore weak incident peaks)."""
        threshold = 0.15 * envelope.max()
        period_samples = int(self.sample_rate_mic / self.rep_rate)

        peaks, _ = signal.find_peaks(envelope, height=threshold, distance=period_samples//2)

        # Filter to keep only strong peaks (reflections)
        peak_amps = envelope[peaks]
        strong_threshold = 0.4 * envelope.max()
        strong_peaks = peaks[peak_amps > strong_threshold]

        return strong_peaks

    def measure_autocorr_fwhm(self, envelope: np.ndarray, peaks: np.ndarray) -> Tuple[float, float]:
        """
        Compute autocorrelation FWHM of each reflection pulse.

        Autocorrelation avoids all alignment issues - peak is always at lag=0.
        FWHM measures intrinsic pulse sharpness/bandwidth.

        Returns:
            (autocorr_fwhm_us, mean_amplitude)
        """
        if len(peaks) < 3:
            return 1000.0, 0.0  # Penalty for too few peaks

        # Extract pulse windows around each peak
        window_half = int(0.001 * self.sample_rate_mic)  # 1ms half-window

        autocorr_fwhms = []
        amplitudes = []

        for pk in peaks:
            start = max(0, pk - window_half)
            end = min(len(envelope), pk + window_half)
            if end - start != 2 * window_half:
                continue

            pulse = envelope[start:end]
            amplitudes.append(envelope[pk])

            # Autocorrelation - correlate pulse with itself
            autocorr = np.correlate(pulse, pulse, mode='full')
            autocorr = autocorr / np.max(autocorr)  # Normalize so peak = 1.0

            # Peak is always at center (lag=0)
            peak_idx = len(autocorr) // 2
            half_max = 0.5

            # Find left crossing
            left_idx = peak_idx
            for j in range(peak_idx, 0, -1):
                if autocorr[j] < half_max:
                    frac = (half_max - autocorr[j]) / (autocorr[j+1] - autocorr[j] + 1e-10)
                    left_idx = j + frac
                    break

            # Find right crossing
            right_idx = peak_idx
            for j in range(peak_idx, len(autocorr)-1):
                if autocorr[j] < half_max:
                    frac = (half_max - autocorr[j-1]) / (autocorr[j] - autocorr[j-1] + 1e-10)
                    right_idx = j - 1 + frac
                    break

            fwhm_samples = right_idx - left_idx
            fwhm_us = fwhm_samples / self.sample_rate_mic * 1e6

            if 10 < fwhm_us < 1000:  # Sanity check
                autocorr_fwhms.append(fwhm_us)

        if len(autocorr_fwhms) < 3:
            return 1000.0, 0.0

        mean_fwhm = np.mean(autocorr_fwhms)
        mean_amplitude = np.mean(amplitudes)

        return mean_fwhm, mean_amplitude

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

        # Time array for SDG (6400 points @ 400 kSa/s)
        t_awg = np.arange(self.n_points) / self.sample_rate_awg
        t_center = t_awg[self.n_points // 2]

        # Main Gaussian pulse
        sigma_t = sigma_cycles / self.carrier_freq
        envelope = np.exp(-0.5 * ((t_awg - t_center) / sigma_t) ** 2)
        carrier = np.sin(2 * np.pi * self.carrier_freq * t_awg + phase)
        main_pulse = envelope * carrier

        # Pre-pulse compensation
        pre_pulse = np.zeros(self.n_points)
        burst_duration = 0.5 / self.carrier_freq
        burst_samples = int(burst_duration * self.sample_rate_awg)

        for i, amp in enumerate(pre_comp):
            start = self.n_points // 2 - int((3-i) * 2 * burst_samples) - burst_samples
            if 0 <= start < self.n_points - burst_samples:
                t_burst = np.arange(burst_samples) / self.sample_rate_awg
                pre_pulse[start:start+burst_samples] = amp * np.sin(2 * np.pi * self.carrier_freq * t_burst)

        # Post-pulse compensation
        post_pulse = np.zeros(self.n_points)
        for i, amp in enumerate(post_comp):
            start = self.n_points // 2 + int((i+1) * 2 * burst_samples)
            if 0 <= start < self.n_points - burst_samples:
                t_burst = np.arange(burst_samples) / self.sample_rate_awg
                decay = np.exp(-i * 0.3)
                post_pulse[start:start+burst_samples] = amp * decay * np.sin(2 * np.pi * self.carrier_freq * t_burst)

        # Combine
        waveform = main_pulse + pre_pulse + post_pulse

        # Apply window
        active_start = int(0.1 * self.n_points)
        active_end = int(0.9 * self.n_points)
        full_waveform = np.zeros(self.n_points)
        full_waveform[active_start:active_end] = waveform[active_start:active_end]

        # Taper edges
        taper_len = 50
        taper_in = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, taper_len)))
        taper_out = 0.5 * (1 + np.cos(np.pi * np.linspace(0, 1, taper_len)))
        full_waveform[active_start:active_start+taper_len] *= taper_in
        full_waveform[active_end-taper_len:active_end] *= taper_out

        # Normalize
        max_abs = np.max(np.abs(full_waveform))
        if max_abs > 0:
            full_waveform = full_waveform / max_abs

        return full_waveform

    def measure_waveform(self, waveform: np.ndarray) -> Tuple[float, float]:
        """Send waveform and measure xcorr FWHM + amplitude."""
        self.sdg.upload_waveform("opt", waveform, freq_hz=self.rep_rate, ampl_vpp=2.0)
        self.sdg.output_on(channel=1)
        time.sleep(self.settling_time)

        audio = self.record()
        envelope = self.extract_envelope(audio)
        peaks = self.find_reflection_peaks(envelope)
        fwhm_us, amplitude = self.measure_autocorr_fwhm(envelope, peaks)

        return fwhm_us, amplitude

    def objective(self, params: List[float]) -> float:
        """Objective function for optimization."""
        params = np.array(params)
        self.eval_count += 1

        waveform = self.params_to_waveform(params)
        fwhm_us, amplitude = self.measure_waveform(waveform)

        # Amplitude penalty
        if self.baseline_amplitude is not None:
            amp_threshold = MIN_AMPLITUDE_FRACTION * self.baseline_amplitude
            if amplitude < amp_threshold:
                penalty = AMPLITUDE_PENALTY_WEIGHT * (amp_threshold - amplitude) ** 2
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
            amplitude >= MIN_AMPLITUDE_FRACTION * self.baseline_amplitude):
            self.best_fwhm = fwhm_us
            self.best_params = params.copy()
            self.best_waveform = waveform.copy()
            self.best_amplitude = amplitude
            print(f"  [{self.eval_count}] NEW BEST: autocorr_FWHM={fwhm_us:.1f}μs, amp={amplitude:.4f}")
        else:
            print(f"  [{self.eval_count}] autocorr_FWHM={fwhm_us:.1f}μs, amp={amplitude:.4f}, cost={cost:.1f}")

        return cost

    def measure_baseline(self, sigma_cycles: float = 5.0) -> Tuple[float, float]:
        """Measure baseline with standard Gaussian pulse."""
        print("\nMeasuring baseline...")

        baseline_params = np.array([
            sigma_cycles, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
        ])

        waveform = self.params_to_waveform(baseline_params)

        fwhms = []
        amps = []
        for i in range(3):
            fwhm, amp = self.measure_waveform(waveform)
            fwhms.append(fwhm)
            amps.append(amp)
            print(f"  Baseline {i+1}/3: autocorr_FWHM={fwhm:.1f}μs, amp={amp:.4f}")

        baseline_fwhm = np.mean(fwhms)
        baseline_amp = np.mean(amps)

        print(f"Baseline: autocorr_FWHM={baseline_fwhm:.1f}μs, amplitude={baseline_amp:.4f}")

        self.baseline_amplitude = baseline_amp
        return baseline_fwhm, baseline_amp

    def optimize(self, n_calls: int = 50, n_initial_points: int = 15) -> OptimizationResult:
        """Run Bayesian optimization."""
        print("\n" + "=" * 60)
        print("Bayesian Optimization - Autocorr FWHM (SDG1062X)")
        print("=" * 60)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Evaluations: {n_initial_points} initial + {n_calls - n_initial_points} GP-guided")

        # Reset
        self.eval_count = 0
        self.all_fwhms = []
        self.all_amplitudes = []
        self.all_params = []
        self.best_fwhm = float('inf')
        self.best_params = None
        self.best_waveform = None
        self.best_amplitude = None

        baseline_fwhm, baseline_amp = self.measure_baseline()

        # Parameter space
        space = [
            Real(1.0, 15.0, name='sigma'),
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

        # Checkpoint callback for periodic saving
        def checkpoint_callback(res):
            iteration = len(res.x_iters)
            if iteration % 50 == 0:
                checkpoint_path = f'bayesian_xcorr_checkpoint_{iteration}.npz'
                np.savez(
                    checkpoint_path,
                    best_fwhm=self.best_fwhm,
                    best_params=self.best_params,
                    best_waveform=self.best_waveform,
                    all_fwhms=self.all_fwhms,
                    iteration=iteration,
                )
                print(f"  >>> Checkpoint saved: {checkpoint_path}")

        result = gp_minimize(
            self.objective,
            space,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            acq_func='EI',
            noise=1e-10,
            random_state=42,
            verbose=False,
            callback=[checkpoint_callback],
        )

        elapsed = time.time() - t0

        print("-" * 60)
        print(f"\nOptimization complete in {elapsed/60:.1f} minutes")
        print(f"Best autocorr_FWHM: {self.best_fwhm:.1f}μs (baseline: {baseline_fwhm:.1f}μs)")
        print(f"Improvement: {(1 - self.best_fwhm/baseline_fwhm)*100:.1f}%")

        self.sdg.output_off(channel=1)

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


def plot_results(result: OptimizationResult, save_path: str):
    """Plot optimization results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    evals = np.arange(1, len(result.all_fwhms) + 1)
    best_so_far = np.minimum.accumulate(result.all_fwhms)

    axes[0, 0].plot(evals, result.all_fwhms, 'b.', alpha=0.5, label='Measured')
    axes[0, 0].plot(evals, best_so_far, 'r-', lw=2, label='Best so far')
    axes[0, 0].axhline(result.baseline_fwhm, color='gray', ls='--', label=f'Baseline ({result.baseline_fwhm:.0f}μs)')
    axes[0, 0].set_xlabel('Evaluation')
    axes[0, 0].set_ylabel('Autocorr FWHM (μs)')
    axes[0, 0].set_title('Convergence')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(evals, result.all_amplitudes, 'g.-', alpha=0.7)
    axes[0, 1].axhline(result.baseline_amplitude, color='gray', ls='--', label='Baseline')
    axes[0, 1].axhline(0.7 * result.baseline_amplitude, color='red', ls=':', label='70% threshold')
    axes[0, 1].set_xlabel('Evaluation')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_title('Amplitude Tracking')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    t_ms = np.arange(len(result.best_waveform)) / SAMPLE_RATE_AWG * 1000
    axes[1, 0].plot(t_ms, result.best_waveform, 'b-', lw=0.5)
    axes[1, 0].set_xlabel('Time (ms)')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].set_title('Best Waveform')
    axes[1, 0].grid(True, alpha=0.3)

    summary = f"""
    Autocorr FWHM Optimization Results (SDG1062X)
    =============================================

    Baseline:
      Autocorr FWHM: {result.baseline_fwhm:.1f} μs
      Amplitude: {result.baseline_amplitude:.4f}

    Best Found:
      Autocorr FWHM: {result.best_fwhm:.1f} μs
      Amplitude: {result.best_amplitude:.4f}

    Improvement: {(1 - result.best_fwhm/result.baseline_fwhm)*100:+.1f}%

    Best Parameters:
      σ = {result.best_params[0]:.2f} cycles
      φ = {result.best_params[1]:.2f} rad
      pre = [{result.best_params[2]:.2f}, {result.best_params[3]:.2f}, {result.best_params[4]:.2f}]
      post = [{result.best_params[5]:.2f}, {result.best_params[6]:.2f}, {result.best_params[7]:.2f}, {result.best_params[8]:.2f}, {result.best_params[9]:.2f}]

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
    print("BATPING Bayesian Autocorr Optimization (SDG1062X)")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    optimizer = BayesianXCorrOptimizer()

    try:
        optimizer.connect()

        result = optimizer.optimize(n_calls=1000, n_initial_points=100)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_results(result, f'bayesian_xcorr_results_v3_{timestamp}.png')

        np.savez(
            f'bayesian_xcorr_state_v3_{timestamp}.npz',
            best_params=result.best_params,
            best_fwhm=result.best_fwhm,
            best_amplitude=result.best_amplitude,
            best_waveform=result.best_waveform,
            all_fwhms=result.all_fwhms,
            all_amplitudes=result.all_amplitudes,
            baseline_fwhm=result.baseline_fwhm,
            baseline_amplitude=result.baseline_amplitude,
        )
        print(f"Saved: bayesian_xcorr_state_v3_{timestamp}.npz")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        optimizer.disconnect()


if __name__ == "__main__":
    main()
