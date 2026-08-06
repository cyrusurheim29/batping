#!/usr/bin/env python3
"""
Batping Signal Analysis Script
Analyzes ultrasonic recordings from the Dodotronic Ultramic 384K

Usage:
    python analyze_signal.py <audio_file.aiff>
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import sys
import os

# For reading AIFF files
import aifc


def read_aiff(filename):
    """Read an AIFF audio file and return sample rate and data."""
    with aifc.open(filename, 'r') as f:
        n_channels = f.getnchannels()
        sample_width = f.getsampwidth()
        sample_rate = f.getframerate()
        n_frames = f.getnframes()

        # Read raw audio data
        raw_data = f.readframes(n_frames)

        # Convert to numpy array based on sample width
        if sample_width == 2:
            data = np.frombuffer(raw_data, dtype='>i2')  # Big-endian 16-bit
        elif sample_width == 3:
            # 24-bit audio - need to handle specially
            data = np.zeros(n_frames * n_channels, dtype=np.int32)
            for i in range(n_frames * n_channels):
                # 24-bit big-endian
                sample = (raw_data[i*3] << 16) | (raw_data[i*3+1] << 8) | raw_data[i*3+2]
                # Sign extend
                if sample & 0x800000:
                    sample -= 0x1000000
                data[i] = sample
        elif sample_width == 4:
            data = np.frombuffer(raw_data, dtype='>i4')  # Big-endian 32-bit
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        # If stereo, just take first channel
        if n_channels == 2:
            data = data[::2]

        # Normalize to float between -1 and 1
        max_val = 2 ** (sample_width * 8 - 1)
        data = data.astype(np.float64) / max_val

        return sample_rate, data


def compute_envelope(data, sample_rate, window_ms=10):
    """Compute signal envelope using RMS in sliding windows."""
    window_samples = int(window_ms * sample_rate / 1000)

    # Compute RMS in sliding window
    squared = data ** 2
    window = np.ones(window_samples) / window_samples
    envelope = np.sqrt(np.convolve(squared, window, mode='same'))

    return envelope


def bandpass_filter(data, sample_rate, low_freq, high_freq, order=4):
    """Apply a bandpass filter to the data."""
    nyquist = sample_rate / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)


def find_signal_region(data, sample_rate, threshold_factor=2.0, target_freq=40000):
    """
    Auto-detect the region where signal is present.
    Uses bandpass filtering around target frequency for better detection.
    Returns (start_idx, end_idx) of the active signal region.
    """
    # Bandpass filter around target frequency to isolate signal from interference
    filtered = bandpass_filter(data, sample_rate, target_freq - 5000, target_freq + 5000)

    envelope = compute_envelope(filtered, sample_rate, window_ms=20)

    # Use median as baseline (robust to outliers)
    baseline = np.median(envelope)
    threshold = baseline * threshold_factor

    # Find where envelope exceeds threshold
    above_threshold = envelope > threshold

    if not np.any(above_threshold):
        # No signal detected, return full range
        print("  (No distinct signal region detected, using full recording)")
        return 0, len(data)

    # Find first and last points above threshold
    indices = np.where(above_threshold)[0]
    start_idx = max(0, indices[0] - int(0.01 * sample_rate))  # 10ms padding
    end_idx = min(len(data), indices[-1] + int(0.01 * sample_rate))

    return start_idx, end_idx


def compute_spectrum(data, sample_rate, n_fft=None):
    """Compute frequency spectrum of data."""
    if n_fft is None:
        n_fft = min(len(data), 2**16)

    freqs = np.fft.rfftfreq(n_fft, 1/sample_rate)

    # Use Hann window to reduce spectral leakage
    window = np.hanning(n_fft)
    data_windowed = data[:n_fft] * window

    fft_vals = np.fft.rfft(data_windowed)
    magnitude = np.abs(fft_vals)

    # Convert to dB (relative to max possible amplitude)
    magnitude_db = 20 * np.log10(magnitude / (n_fft/2) + 1e-10)

    return freqs, magnitude_db


def find_peak_near(freqs, magnitude_db, target_freq, search_range=2000):
    """Find the peak magnitude near a target frequency."""
    idx_target = np.argmin(np.abs(freqs - target_freq))
    freq_resolution = freqs[1] - freqs[0]
    search_bins = int(search_range / freq_resolution)

    start = max(0, idx_target - search_bins)
    end = min(len(freqs), idx_target + search_bins)

    local_max_idx = start + np.argmax(magnitude_db[start:end])

    return freqs[local_max_idx], magnitude_db[local_max_idx]


def analyze_audio(filename):
    """Main analysis function."""
    print(f"Analyzing: {filename}")
    print("=" * 60)

    # Read the audio file
    try:
        if filename.endswith('.aiff') or filename.endswith('.aif'):
            sample_rate, data = read_aiff(filename)
        else:
            sample_rate, data = wavfile.read(filename)
            if data.dtype == np.int16:
                data = data.astype(np.float64) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float64) / 2147483648.0
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    duration = len(data) / sample_rate

    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {duration:.3f} seconds")
    print(f"Samples: {len(data)}")
    print(f"Nyquist frequency: {sample_rate/2} Hz")
    print()

    # Auto-detect signal region
    start_idx, end_idx = find_signal_region(data, sample_rate)
    signal_data = data[start_idx:end_idx]
    signal_duration = len(signal_data) / sample_rate

    print(f"Signal detected: {start_idx/sample_rate*1000:.1f} ms to {end_idx/sample_rate*1000:.1f} ms")
    print(f"Signal duration: {signal_duration*1000:.1f} ms")
    print()

    # Time vectors
    t_full = np.arange(len(data)) / sample_rate
    t_signal = np.arange(len(signal_data)) / sample_rate

    # Compute envelope for visualization
    envelope = compute_envelope(data, sample_rate, window_ms=10)

    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))

    # --- Plot 1: Full time-domain waveform with detected region ---
    ax1 = axes[0]
    ax1.plot(t_full * 1000, data, 'b-', linewidth=0.3, alpha=0.7, label='Waveform')
    ax1.plot(t_full * 1000, envelope, 'r-', linewidth=1, label='Envelope')
    ax1.axvline(x=start_idx/sample_rate*1000, color='g', linestyle='--', linewidth=2, label='Signal region')
    ax1.axvline(x=end_idx/sample_rate*1000, color='g', linestyle='--', linewidth=2)
    ax1.axvspan(start_idx/sample_rate*1000, end_idx/sample_rate*1000, alpha=0.2, color='green')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Full Waveform with Auto-Detected Signal Region')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, duration * 1000])
    ax1.legend(loc='upper right')

    # --- Plot 2: Zoomed time-domain of signal region (first few cycles at 40kHz) ---
    ax2 = axes[1]
    # Show ~20 cycles at 40kHz = 0.5ms
    zoom_samples = int(0.5e-3 * sample_rate)
    if zoom_samples > len(signal_data):
        zoom_samples = len(signal_data)
    t_zoom = np.arange(zoom_samples) / sample_rate * 1000  # ms
    ax2.plot(t_zoom, signal_data[:zoom_samples], 'b-', linewidth=1)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Zoomed Signal Region (first 0.5 ms of detected signal)')
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Spectrum of SIGNAL REGION ONLY ---
    ax3 = axes[2]

    freqs_sig, mag_sig = compute_spectrum(signal_data, sample_rate)

    ax3.plot(freqs_sig / 1000, mag_sig, 'b-', linewidth=0.5)
    ax3.set_xlabel('Frequency (kHz)')
    ax3.set_ylabel('Magnitude (dBFS)')
    ax3.set_title('Frequency Spectrum (Signal Region Only)')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, min(100, sample_rate/2000)])
    ax3.set_ylim([-80, 0])
    ax3.axvline(x=40, color='r', linestyle='--', alpha=0.7, label='40 kHz (signal)')
    ax3.axvline(x=33, color='orange', linestyle='--', alpha=0.7, label='33 kHz (interference)')
    ax3.legend()

    # --- Plot 4: Comparison - Full file vs Signal region ---
    ax4 = axes[3]

    freqs_full, mag_full = compute_spectrum(data, sample_rate)

    ax4.plot(freqs_full / 1000, mag_full, 'gray', linewidth=0.5, alpha=0.7, label='Full recording')
    ax4.plot(freqs_sig / 1000, mag_sig, 'b-', linewidth=0.8, label='Signal region only')
    ax4.set_xlabel('Frequency (kHz)')
    ax4.set_ylabel('Magnitude (dBFS)')
    ax4.set_title('Spectrum Comparison: Full Recording vs Signal Region')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([25, 50])  # Focus on 25-50 kHz range
    ax4.set_ylim([-60, 0])
    ax4.axvline(x=40, color='r', linestyle='--', alpha=0.5)
    ax4.axvline(x=33, color='orange', linestyle='--', alpha=0.5)
    ax4.legend()

    plt.tight_layout()

    # Save the figure
    output_path = filename.rsplit('.', 1)[0] + '_analysis.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved analysis plot to: {output_path}")

    # --- Signal Analysis ---
    print("\n" + "=" * 60)
    print("SIGNAL REGION ANALYSIS")
    print("=" * 60)

    # Find peaks in signal region spectrum
    print("\nTop spectral peaks (signal region):")
    print("-" * 40)

    peak_indices, properties = signal.find_peaks(mag_sig, height=-50, distance=20)
    if len(peak_indices) > 0:
        sorted_idx = np.argsort(properties['peak_heights'])[::-1]
        for i in sorted_idx[:10]:
            idx = peak_indices[i]
            freq = freqs_sig[idx]
            mag = mag_sig[idx]
            print(f"  {freq/1000:7.2f} kHz : {mag:6.1f} dBFS")

    # Find specific peaks
    freq_40k, level_40k = find_peak_near(freqs_sig, mag_sig, 40000)
    freq_33k, level_33k = find_peak_near(freqs_sig, mag_sig, 33000)

    snr = level_40k - level_33k

    print("\n" + "-" * 40)
    print(f"40 kHz signal: {freq_40k/1000:.2f} kHz @ {level_40k:.1f} dBFS")
    print(f"33 kHz noise:  {freq_33k/1000:.2f} kHz @ {level_33k:.1f} dBFS")
    print(f"Signal-to-Interference Ratio: {snr:.1f} dB")
    print("-" * 40)

    if snr > 0:
        print(f"✓ Good: 40 kHz signal is {snr:.1f} dB above interference")
    else:
        print(f"⚠ Warning: 40 kHz signal is {-snr:.1f} dB below interference")

    # --- Background Analysis ---
    print("\n" + "=" * 60)
    print("BACKGROUND ANALYSIS (before/after signal)")
    print("=" * 60)

    # Analyze background (before signal starts)
    if start_idx > sample_rate * 0.1:  # At least 100ms of background
        bg_data = data[:start_idx]
        freqs_bg, mag_bg = compute_spectrum(bg_data, sample_rate)
        freq_33k_bg, level_33k_bg = find_peak_near(freqs_bg, mag_bg, 33000)
        print(f"33 kHz interference (background): {level_33k_bg:.1f} dBFS")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to the existing file if no argument given
        default_file = "cu_data_mar19_40khz_520mVp2p_audio.aiff"
        if os.path.exists(default_file):
            analyze_audio(default_file)
        else:
            print("Usage: python analyze_signal.py <audio_file.aiff>")
            print("       or run from the batping directory with no arguments")
    else:
        analyze_audio(sys.argv[1])
