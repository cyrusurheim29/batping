#!/usr/bin/env python3
"""Quick spectrum plot for first N seconds of audio files."""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import aifc
import sys


def read_aiff(filename):
    """Read an AIFF audio file."""
    with aifc.open(filename, 'r') as f:
        n_channels = f.getnchannels()
        sample_width = f.getsampwidth()
        sample_rate = f.getframerate()
        n_frames = f.getnframes()
        raw_data = f.readframes(n_frames)

        if sample_width == 2:
            data = np.frombuffer(raw_data, dtype='>i2')
        elif sample_width == 4:
            data = np.frombuffer(raw_data, dtype='>i4')
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        if n_channels == 2:
            data = data[::2]

        max_val = 2 ** (sample_width * 8 - 1)
        data = data.astype(np.float64) / max_val

        return sample_rate, data


def plot_spectrum(filename, duration_sec=2.5, output_suffix="_spectrum"):
    """Plot power spectrum of first N seconds."""
    print(f"Processing: {filename}")

    sample_rate, data = read_aiff(filename)

    # Take first N seconds
    n_samples = int(duration_sec * sample_rate)
    if n_samples > len(data):
        n_samples = len(data)
    data = data[:n_samples]

    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Using first {n_samples/sample_rate:.2f} seconds ({n_samples} samples)")

    # Compute spectrum
    n_fft = min(len(data), 2**16)
    freqs = np.fft.rfftfreq(n_fft, 1/sample_rate)

    window = np.hanning(n_fft)
    data_windowed = data[:n_fft] * window

    fft_vals = np.fft.rfft(data_windowed)
    magnitude = np.abs(fft_vals)
    magnitude_db = 20 * np.log10(magnitude / (n_fft/2) + 1e-10)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(freqs / 1000, magnitude_db, 'b-', linewidth=0.5)
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Magnitude (dBFS)')
    ax.set_title(f'Power Spectrum: {filename}\n(first {duration_sec} seconds)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, min(100, sample_rate/2000)])
    ax.set_ylim([-80, 0])

    # Mark common frequencies
    ax.axvline(x=40, color='r', linestyle='--', alpha=0.5, label='40 kHz')
    ax.axvline(x=33, color='orange', linestyle='--', alpha=0.5, label='33 kHz')
    ax.legend()

    # Save
    output_path = filename.rsplit('.', 1)[0] + output_suffix + '.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  Saved: {output_path}")

    # Find peaks
    from scipy import signal as sig
    peak_indices, props = sig.find_peaks(magnitude_db, height=-50, distance=20)
    if len(peak_indices) > 0:
        sorted_idx = np.argsort(props['peak_heights'])[::-1]
        print(f"  Top peaks:")
        for i in sorted_idx[:5]:
            idx = peak_indices[i]
            print(f"    {freqs[idx]/1000:7.2f} kHz : {magnitude_db[idx]:6.1f} dBFS")

    return output_path


if __name__ == "__main__":
    files = [
        "iu_lab_aligned_mar16.aiff",
        "mar22_hallway_mirror.aiff"
    ]

    for f in files:
        print()
        plot_spectrum(f, duration_sec=2.5)
        print()
