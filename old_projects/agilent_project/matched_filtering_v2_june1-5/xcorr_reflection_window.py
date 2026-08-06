#!/usr/bin/env python3
"""
Cross-correlation on ONLY the 4-8ms reflection window.

Loads the original wav files, computes envelopes, aligns at direct pickup,
then cross-correlates only the reflection window segment.
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, hilbert, correlate, find_peaks
import matplotlib.pyplot as plt


def bandpass_filter(signal, fs, low=35000, high=45000, order=4):
    """Apply bandpass filter around 40kHz."""
    nyq = fs / 2
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal)


def compute_envelope(signal, fs):
    """Bandpass filter and extract Hilbert envelope."""
    filtered = bandpass_filter(signal, fs)
    envelope = np.abs(hilbert(filtered))
    return envelope


def parabolic_interp(y, idx):
    """Sub-sample peak refinement using parabolic interpolation."""
    if idx <= 0 or idx >= len(y) - 1:
        return float(idx)
    y0, y1, y2 = y[idx-1], y[idx], y[idx+1]
    denom = y0 - 2*y1 + y2
    if abs(denom) < 1e-10:
        return float(idx)
    return idx + 0.5 * (y0 - y2) / denom


def find_direct_pickup(envelope, fs, search_ms=3.0):
    """Find the direct pickup peak in the first few ms."""
    search_samples = int(search_ms * fs / 1000)
    segment = envelope[:search_samples]
    peaks, props = find_peaks(segment, height=0.1 * np.max(segment), distance=int(0.5 * fs / 1000))
    if len(peaks) == 0:
        return np.argmax(segment)
    return peaks[np.argmax(props['peak_heights'])]


def main():
    # Load wav files
    fs1, data1 = wavfile.read('gaussian_2m_reflection_50Hz.wav')
    fs2, data2 = wavfile.read('gaussian_1.8m_reflection_50Hz.wav')

    assert fs1 == fs2, "Sample rates must match"
    fs = fs1

    # Convert to float and normalize
    sig_2m = data1.astype(float) / 32768.0
    sig_18m = data2.astype(float) / 32768.0

    print(f"Loaded wav files: fs={fs} Hz")
    print(f"  2m:   {len(sig_2m)} samples ({len(sig_2m)/fs*1000:.1f} ms)")
    print(f"  1.8m: {len(sig_18m)} samples ({len(sig_18m)/fs*1000:.1f} ms)")

    # Compute envelopes
    env_2m = compute_envelope(sig_2m, fs)
    env_18m = compute_envelope(sig_18m, fs)

    # Find direct pickup peaks (for alignment)
    peak_2m = find_direct_pickup(env_2m, fs)
    peak_18m = find_direct_pickup(env_18m, fs)

    print(f"\nDirect pickup peaks:")
    print(f"  2m:   sample {peak_2m} ({peak_2m/fs*1000:.3f} ms)")
    print(f"  1.8m: sample {peak_18m} ({peak_18m/fs*1000:.3f} ms)")

    # Align signals so direct pickup is at t=0
    # Shift both signals so their direct pickup is at index 0
    aligned_2m = env_2m[peak_2m:]
    aligned_18m = env_18m[peak_18m:]

    # Make sure both have same length
    min_len = min(len(aligned_2m), len(aligned_18m))
    aligned_2m = aligned_2m[:min_len]
    aligned_18m = aligned_18m[:min_len]

    print(f"\nAfter alignment: {min_len} samples ({min_len/fs*1000:.1f} ms)")

    # Extract 4-8ms reflection window
    start_ms, end_ms = 4.0, 8.0
    start_idx = int(start_ms * fs / 1000)
    end_idx = int(end_ms * fs / 1000)

    if end_idx > min_len:
        print(f"Warning: window extends beyond signal, truncating")
        end_idx = min_len

    window_2m = aligned_2m[start_idx:end_idx]
    window_18m = aligned_18m[start_idx:end_idx]

    print(f"\nReflection window: {start_ms}-{end_ms} ms")
    print(f"  Window samples: {len(window_2m)}")
    print(f"  Window duration: {len(window_2m)/fs*1000:.2f} ms")

    # Cross-correlate reflection windows
    corr = correlate(window_18m, window_2m, mode='full')
    n = len(window_2m)
    lags = np.arange(-(n-1), n)
    lags_us = lags / fs * 1e6

    # Find peak with sub-sample precision
    peak_idx = np.argmax(corr)
    peak_refined = parabolic_interp(corr, peak_idx)

    shift_samples = peak_refined - (n - 1)
    shift_us = shift_samples / fs * 1e6

    print(f"\n--- Cross-correlation result (4-8ms window) ---")
    print(f"Peak at lag index: {peak_idx} (refined: {peak_refined:.2f})")
    print(f"TOF shift: {shift_us:.2f} μs")
    print(f"  (Positive = 1.8m reflection arrives later relative to its direct pickup)")

    # Expected shift for 10cm displacement (20cm round-trip shorter)
    expected_us = -0.20 / 343 * 1e6  # Negative because path is shorter
    print(f"\nExpected for 10cm closer: {expected_us:.0f} μs")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Full aligned envelopes (like original plot)
    t_ms = np.arange(min_len) / fs * 1000
    axes[0, 0].plot(t_ms, aligned_2m, 'b-', linewidth=0.8, alpha=0.8, label='2m')
    axes[0, 0].plot(t_ms, aligned_18m, 'r-', linewidth=0.8, alpha=0.8, label='1.8m')
    axes[0, 0].axvline(start_ms, color='g', linestyle='--', alpha=0.5, label='Window')
    axes[0, 0].axvline(end_ms, color='g', linestyle='--', alpha=0.5)
    axes[0, 0].axvspan(start_ms, end_ms, alpha=0.1, color='green')
    axes[0, 0].set_xlabel('Time from direct pickup (ms)')
    axes[0, 0].set_ylabel('Envelope')
    axes[0, 0].set_title('Aligned to DIRECT pickup')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(0, 15)

    # Top-right: Zoomed reflection window
    t_window = np.linspace(start_ms, end_ms, len(window_2m))
    axes[0, 1].plot(t_window, window_2m, 'b-', linewidth=1, label='2m')
    axes[0, 1].plot(t_window, window_18m, 'r-', linewidth=1, label='1.8m')
    axes[0, 1].set_xlabel('Time from direct pickup (ms)')
    axes[0, 1].set_ylabel('Envelope')
    axes[0, 1].set_title('Reflection window (4-8ms) - input to xcorr')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom-left: Cross-correlation
    axes[1, 0].plot(lags_us, corr, 'b-', linewidth=1)
    axes[1, 0].axvline(shift_us, color='r', linestyle='--', linewidth=2,
                       label=f'Peak: {shift_us:.1f} μs')
    axes[1, 0].axvline(expected_us, color='g', linestyle=':', linewidth=2,
                       label=f'Expected: {expected_us:.0f} μs')
    axes[1, 0].axvline(0, color='k', linestyle=':', alpha=0.5)
    axes[1, 0].set_xlabel('Lag (μs)')
    axes[1, 0].set_ylabel('Correlation')
    axes[1, 0].set_title('Cross-correlation (1.8m window vs 2m window)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(-2000, 2000)

    # Bottom-right: Summary bar chart
    ax = axes[1, 1]
    bars = ax.bar(['Expected\n(10cm shorter)', 'Measured'],
                  [abs(expected_us), abs(shift_us)],
                  color=['orange', 'steelblue'])
    ax.set_ylabel('|ΔTOF| (μs)')
    ax.set_title(f'Measured: {shift_us:.1f} μs\nExpected: {expected_us:.0f} μs')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, [abs(expected_us), abs(shift_us)]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{val:.0f} μs', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig('xcorr_reflection_window.png', dpi=150)
    print(f"\nSaved: xcorr_reflection_window.png")
    plt.show()


if __name__ == '__main__':
    main()
