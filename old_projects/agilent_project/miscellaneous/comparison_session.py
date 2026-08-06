#!/usr/bin/env python3
"""
BATPING Comparison Recording Session

Records both unoptimized and optimized waveforms in the same session
for fair comparison, with proper annotations and time windowing.
"""

import numpy as np
import pyaudio
import time
from datetime import datetime
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from agilent_33220a import Agilent33220A, generate_gaussian_pulse, generate_shaped_pulse


def record_audio(pa, device_index, duration_s, sample_rate=384000):
    """Record audio from Ultramic."""
    n_samples = int(duration_s * sample_rate)
    chunk_size = 4096

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        input_device_index=device_index,
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


def find_ultramic(pa):
    """Find Ultramic device index."""
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if "UltraMic" in info['name'] and info['maxInputChannels'] > 0:
            return i
    raise RuntimeError("UltraMic not found")


def analyze_pulse_detailed(audio, sample_rate=384000, bandpass=(38000, 42000)):
    """
    Analyze pulse with separate attack and ringdown metrics.

    Returns dict with:
    - peak_amplitude: Maximum envelope value
    - peak_time_ms: Time of peak
    - fwhm_total_us: Full FWHM including ringdown
    - attack_width_us: Width of initial peak only (to first local minimum after peak)
    - rise_time_10_90_us: Time from 10% to 90% of peak
    """
    # Bandpass filter
    nyq = sample_rate / 2
    low = bandpass[0] / nyq
    high = min(bandpass[1] / nyq, 0.99)
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, audio)

    # Envelope via Hilbert transform
    analytic = signal.hilbert(filtered)
    envelope = np.abs(analytic)

    # Smooth envelope slightly for stable measurements
    smooth_window = int(0.05 * sample_rate / 1000)  # 0.05ms window
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        envelope_smooth = np.convolve(envelope, kernel, mode='same')
    else:
        envelope_smooth = envelope

    # Find peak
    peak_idx = np.argmax(envelope_smooth)
    peak_amplitude = envelope_smooth[peak_idx]
    peak_time_ms = peak_idx / sample_rate * 1000

    # --- Total FWHM (traditional) ---
    half_max = peak_amplitude / 2
    above_half = envelope_smooth > half_max
    crossings = np.diff(above_half.astype(int))
    rising = np.where(crossings == 1)[0]
    falling = np.where(crossings == -1)[0]

    fwhm_total_us = None
    fwhm_start_idx = None
    fwhm_end_idx = None

    if len(rising) > 0 and len(falling) > 0:
        start_candidates = rising[rising < peak_idx]
        end_candidates = falling[falling > peak_idx]

        if len(start_candidates) > 0 and len(end_candidates) > 0:
            fwhm_start_idx = start_candidates[-1]
            fwhm_end_idx = end_candidates[0]
            fwhm_total_us = (fwhm_end_idx - fwhm_start_idx) / sample_rate * 1e6

    # --- Attack width (to first local minimum after peak) ---
    # Find local minima after peak
    search_end = min(peak_idx + int(2 * sample_rate / 1000), len(envelope_smooth) - 1)  # Search up to 2ms after peak

    attack_width_us = None
    attack_end_idx = None

    if search_end > peak_idx + 10:
        # Look for where envelope drops below some fraction of peak, then rises again (local min)
        post_peak = envelope_smooth[peak_idx:search_end]

        # Find local minima in post-peak region
        local_min_indices = []
        for i in range(1, len(post_peak) - 1):
            if post_peak[i] < post_peak[i-1] and post_peak[i] < post_peak[i+1]:
                local_min_indices.append(i)

        if len(local_min_indices) > 0:
            # First local minimum after peak
            attack_end_idx = peak_idx + local_min_indices[0]

            # Find where signal first rose to half-max before peak
            if fwhm_start_idx is not None:
                attack_width_us = (attack_end_idx - fwhm_start_idx) / sample_rate * 1e6

    # --- 10-90% rise time ---
    rise_time_us = None

    threshold_10 = 0.1 * peak_amplitude
    threshold_90 = 0.9 * peak_amplitude

    # Search before peak for crossings
    pre_peak = envelope_smooth[:peak_idx]
    above_10 = pre_peak > threshold_10
    above_90 = pre_peak > threshold_90

    if np.any(above_10) and np.any(above_90):
        # First time we cross 10%
        first_10 = np.argmax(above_10)
        # First time we cross 90%
        first_90 = np.argmax(above_90)

        if first_90 > first_10:
            rise_time_us = (first_90 - first_10) / sample_rate * 1e6

    return {
        'peak_amplitude': peak_amplitude,
        'peak_time_ms': peak_time_ms,
        'peak_idx': peak_idx,
        'fwhm_total_us': fwhm_total_us,
        'fwhm_start_idx': fwhm_start_idx,
        'fwhm_end_idx': fwhm_end_idx,
        'attack_width_us': attack_width_us,
        'attack_end_idx': attack_end_idx,
        'rise_time_10_90_us': rise_time_us,
        'envelope': envelope_smooth,
        'filtered': filtered
    }


def plot_comparison(input_waveform, audio, metrics, title_prefix, waveform_type,
                    session_info, save_path, sample_rate_input=1e6, sample_rate_audio=384000):
    """
    Create comparison plot with input waveform and acoustic response.
    Shows full extent of signal including ringdown.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # --- Top panel: Input waveform ---
    t_input_us = np.arange(len(input_waveform)) / sample_rate_input * 1e6
    axes[0].plot(t_input_us, input_waveform, 'b-', linewidth=0.5)
    axes[0].set_xlabel('Time (μs)')
    axes[0].set_ylabel('Amplitude (normalized)')
    axes[0].set_title(f'{title_prefix}: Electrical Input Waveform ({waveform_type})')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, t_input_us[-1]])

    # --- Bottom panel: Acoustic response with full ringdown ---
    t_audio_ms = np.arange(len(audio)) / sample_rate_audio * 1000

    # Determine plot window: center on peak, extend to show full ringdown
    peak_time = metrics['peak_time_ms']

    # Show 2ms before peak and 8ms after (to capture ringdown)
    window_start_ms = max(0, peak_time - 2)
    window_end_ms = min(t_audio_ms[-1], peak_time + 8)

    mask = (t_audio_ms >= window_start_ms) & (t_audio_ms <= window_end_ms)

    # Plot filtered signal
    axes[1].plot(t_audio_ms[mask], metrics['filtered'][mask], 'b-', alpha=0.5,
                 linewidth=0.5, label='40kHz signal')

    # Plot envelope
    axes[1].plot(t_audio_ms[mask], metrics['envelope'][mask], 'r-',
                 linewidth=1.5, label='Envelope')

    # Mark peak
    axes[1].axvline(peak_time, color='orange', linestyle='--', alpha=0.7, label=f'Peak')

    # Mark half-max level
    half_max = metrics['peak_amplitude'] / 2
    axes[1].axhline(half_max, color='green', linestyle=':', alpha=0.7, label='Half-max')

    # Mark FWHM region if available
    if metrics['fwhm_start_idx'] is not None and metrics['fwhm_end_idx'] is not None:
        fwhm_start_ms = metrics['fwhm_start_idx'] / sample_rate_audio * 1000
        fwhm_end_ms = metrics['fwhm_end_idx'] / sample_rate_audio * 1000
        axes[1].axvspan(fwhm_start_ms, fwhm_end_ms, alpha=0.15, color='green', label='FWHM region')

    # Mark attack end if available
    if metrics['attack_end_idx'] is not None:
        attack_end_ms = metrics['attack_end_idx'] / sample_rate_audio * 1000
        axes[1].axvline(attack_end_ms, color='purple', linestyle='--', alpha=0.7, label='Attack end')

    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Amplitude')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Build metrics string
    metrics_str = f"Peak: {metrics['peak_amplitude']:.4f}"
    if metrics['fwhm_total_us'] is not None:
        metrics_str += f", FWHM(total): {metrics['fwhm_total_us']:.0f}μs"
    if metrics['attack_width_us'] is not None:
        metrics_str += f", Attack: {metrics['attack_width_us']:.0f}μs"
    if metrics['rise_time_10_90_us'] is not None:
        metrics_str += f", Rise(10-90%): {metrics['rise_time_10_90_us']:.0f}μs"

    axes[1].set_title(f'{title_prefix}: Acoustic Response — {metrics_str}')

    # Add session info annotation
    fig.text(0.02, 0.02, session_info, fontsize=8, family='monospace',
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("=" * 60)
    print("BATPING Comparison Recording Session")
    print("=" * 60)

    # Session metadata
    session_time = datetime.now()
    session_id = session_time.strftime("%Y%m%d_%H%M%S")

    # Parameters
    n_points = 4000
    carrier_freq = 40000
    sample_rate_awg = 1e6
    drive_amplitude = 5.0  # Vpp
    record_duration = 0.1  # seconds
    settling_time = 0.05  # seconds after AWG output before recording

    print(f"\nSession ID: {session_id}")
    print(f"Time: {session_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nParameters:")
    print(f"  Waveform points: {n_points}")
    print(f"  Carrier frequency: {carrier_freq} Hz")
    print(f"  AWG sample rate: {sample_rate_awg/1e6} MSa/s")
    print(f"  Drive amplitude: {drive_amplitude} Vpp")
    print(f"  Record duration: {record_duration*1000} ms")

    # Generate waveforms
    print("\nGenerating waveforms...")

    # Unoptimized: Gaussian envelope
    unopt_waveform = generate_gaussian_pulse(n_points, carrier_freq, sample_rate_awg, sigma_cycles=5)

    # Optimized: Load Run 2 parameters
    opt_params = np.load('optimized_params_run2.npy')
    opt_waveform = generate_shaped_pulse(n_points, carrier_freq, sample_rate_awg, fourier_coeffs=opt_params)

    print(f"  Unoptimized: Gaussian, σ=5 cycles")
    print(f"  Optimized: Fourier coeffs = [{', '.join(f'{c:.3f}' for c in opt_params[:4])}...]")

    # Calculate rep rate
    waveform_duration_s = n_points / sample_rate_awg
    rep_freq_hz = 1.0 / waveform_duration_s
    print(f"  Waveform duration: {waveform_duration_s*1000:.1f} ms")
    print(f"  Rep rate: {rep_freq_hz:.1f} Hz")

    # Connect hardware
    print("\nConnecting to Agilent 33220A...")
    awg = Agilent33220A()
    awg.connect()

    print("Initializing Ultramic...")
    pa = pyaudio.PyAudio()
    mic_index = find_ultramic(pa)
    print(f"  Found Ultramic at index {mic_index}")

    sample_rate_mic = 384000

    # Session info string for plot annotations
    session_info = (f"Session: {session_id} | Location: 2nd floor office | "
                    f"Separation: ~22cm | Drive: {drive_amplitude}Vpp | "
                    f"Mic: Ultramic 384K @ {sample_rate_mic/1000:.0f}kHz")

    results = {}

    # --- Record unoptimized ---
    print("\n" + "=" * 60)
    print("Recording UNOPTIMIZED waveform...")
    print("=" * 60)

    awg.output_arbitrary(unopt_waveform, freq_hz=rep_freq_hz, vpp=drive_amplitude)
    print(f"  AWG output enabled, settling for {settling_time*1000:.0f}ms...")
    time.sleep(settling_time)

    print(f"  Recording {record_duration*1000:.0f}ms...")
    audio_unopt = record_audio(pa, mic_index, record_duration, sample_rate_mic)
    awg.set_output(False)

    print("  Analyzing...")
    metrics_unopt = analyze_pulse_detailed(audio_unopt, sample_rate_mic)

    print(f"  Peak amplitude: {metrics_unopt['peak_amplitude']:.4f}")
    print(f"  Peak time: {metrics_unopt['peak_time_ms']:.2f} ms")
    if metrics_unopt['fwhm_total_us']:
        print(f"  FWHM (total): {metrics_unopt['fwhm_total_us']:.0f} μs")
    if metrics_unopt['attack_width_us']:
        print(f"  Attack width: {metrics_unopt['attack_width_us']:.0f} μs")
    if metrics_unopt['rise_time_10_90_us']:
        print(f"  Rise time (10-90%): {metrics_unopt['rise_time_10_90_us']:.0f} μs")

    results['unoptimized'] = {
        'waveform': unopt_waveform,
        'audio': audio_unopt,
        'metrics': metrics_unopt
    }

    # Brief pause between recordings
    print("\n  Pausing 1s before next recording...")
    time.sleep(1.0)

    # --- Record optimized ---
    print("\n" + "=" * 60)
    print("Recording OPTIMIZED waveform...")
    print("=" * 60)

    awg.output_arbitrary(opt_waveform, freq_hz=rep_freq_hz, vpp=drive_amplitude)
    print(f"  AWG output enabled, settling for {settling_time*1000:.0f}ms...")
    time.sleep(settling_time)

    print(f"  Recording {record_duration*1000:.0f}ms...")
    audio_opt = record_audio(pa, mic_index, record_duration, sample_rate_mic)
    awg.set_output(False)

    print("  Analyzing...")
    metrics_opt = analyze_pulse_detailed(audio_opt, sample_rate_mic)

    print(f"  Peak amplitude: {metrics_opt['peak_amplitude']:.4f}")
    print(f"  Peak time: {metrics_opt['peak_time_ms']:.2f} ms")
    if metrics_opt['fwhm_total_us']:
        print(f"  FWHM (total): {metrics_opt['fwhm_total_us']:.0f} μs")
    if metrics_opt['attack_width_us']:
        print(f"  Attack width: {metrics_opt['attack_width_us']:.0f} μs")
    if metrics_opt['rise_time_10_90_us']:
        print(f"  Rise time (10-90%): {metrics_opt['rise_time_10_90_us']:.0f} μs")

    results['optimized'] = {
        'waveform': opt_waveform,
        'audio': audio_opt,
        'metrics': metrics_opt
    }

    # --- Generate plots ---
    print("\n" + "=" * 60)
    print("Generating comparison plots...")
    print("=" * 60)

    plot_comparison(
        unopt_waveform, audio_unopt, metrics_unopt,
        "UNOPTIMIZED", "Gaussian envelope, σ=5 cycles",
        session_info,
        f"comparison_unoptimized_{session_id}.png",
        sample_rate_awg, sample_rate_mic
    )

    plot_comparison(
        opt_waveform, audio_opt, metrics_opt,
        "OPTIMIZED", "Fourier envelope (Run 2 params)",
        session_info,
        f"comparison_optimized_{session_id}.png",
        sample_rate_awg, sample_rate_mic
    )

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)
    print(f"\nSession ID: {session_id}")
    print(f"\n{'Metric':<25} {'Unoptimized':>15} {'Optimized':>15} {'Ratio':>10}")
    print("-" * 65)

    amp_unopt = metrics_unopt['peak_amplitude']
    amp_opt = metrics_opt['peak_amplitude']
    print(f"{'Peak Amplitude':<25} {amp_unopt:>15.4f} {amp_opt:>15.4f} {amp_opt/amp_unopt:>10.2f}x")

    if metrics_unopt['fwhm_total_us'] and metrics_opt['fwhm_total_us']:
        fwhm_unopt = metrics_unopt['fwhm_total_us']
        fwhm_opt = metrics_opt['fwhm_total_us']
        print(f"{'FWHM Total (μs)':<25} {fwhm_unopt:>15.0f} {fwhm_opt:>15.0f} {fwhm_opt/fwhm_unopt:>10.2f}x")

    if metrics_unopt['attack_width_us'] and metrics_opt['attack_width_us']:
        att_unopt = metrics_unopt['attack_width_us']
        att_opt = metrics_opt['attack_width_us']
        print(f"{'Attack Width (μs)':<25} {att_unopt:>15.0f} {att_opt:>15.0f} {att_opt/att_unopt:>10.2f}x")

    if metrics_unopt['rise_time_10_90_us'] and metrics_opt['rise_time_10_90_us']:
        rise_unopt = metrics_unopt['rise_time_10_90_us']
        rise_opt = metrics_opt['rise_time_10_90_us']
        print(f"{'Rise Time 10-90% (μs)':<25} {rise_unopt:>15.0f} {rise_opt:>15.0f} {rise_opt/rise_unopt:>10.2f}x")

    # Save raw data
    np.savez(f'comparison_data_{session_id}.npz',
             audio_unopt=audio_unopt,
             audio_opt=audio_opt,
             waveform_unopt=unopt_waveform,
             waveform_opt=opt_waveform,
             opt_params=opt_params,
             session_info=session_info)
    print(f"\nRaw data saved to: comparison_data_{session_id}.npz")

    # Cleanup
    awg.disconnect()
    pa.terminate()
    print("\nSession complete.")


if __name__ == "__main__":
    main()
