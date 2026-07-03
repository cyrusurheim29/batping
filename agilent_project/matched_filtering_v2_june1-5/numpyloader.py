#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 18:14:39 2026

@author: cyrusurheimog
"""

import numpy as np
from scipy.signal import butter, filtfilt, hilbert, correlate
import matplotlib.pyplot as plt


def bandpass_filter(signal, fs, low=35000, high=45000, order=4):
    """Apply bandpass filter around 40kHz."""
    nyq = fs / 2
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal)


def parabolic_interp(y, idx):
    """Sub-sample peak refinement using parabolic interpolation."""
    if idx <= 0 or idx >= len(y) - 1:
        return float(idx)
    y0, y1, y2 = y[idx-1], y[idx], y[idx+1]
    denom = y0 - 2*y1 + y2
    if abs(denom) < 1e-10:
        return float(idx)
    return idx + 0.5 * (y0 - y2) / denom

def xcorr_tof(sig_template, sig_test, fs, window_ms=(5, 15)):
    """
    Compute TOF shift between template and test signals.

    Parameters:
        sig_template: Reference signal (e.g., 2m recording)
        sig_test: Test signal (e.g., 1.8m recording)
        fs: Sample rate (Hz)
        window_ms: (start, end) of reflection window in ms

    Returns:
        shift_us: Time shift in microseconds (positive = test delayed)
        corr: Full cross-correlation array
        lags_us: Lag values in microseconds
    """

    # Extract reflection window
    start_idx = int(window_ms[0] * fs / 1000)
    end_idx = int(window_ms[1] * fs / 1000)

    seg_template = sig_template[start_idx:end_idx]
    seg_test = sig_test[start_idx:end_idx]
    
    

    # Cross-correlate
    corr = correlate(seg_test, seg_template, mode='full')
    n = len(seg_template)
    lags = np.arange(-(n-1), n)
    lags_us = lags / fs * 1e6

    # Find peak with sub-sample precision
    peak_idx = np.argmax(corr)
    peak_refined = parabolic_interp(corr, peak_idx)

    shift_samples = peak_refined - (n - 1)
    shift_us = shift_samples / fs * 1e6

    return shift_us, corr, lags_us


def main():
    window_1 = 10
    window_2 = 13
    
    a = np.load("optimized_signal_1.8m_raw.npz")
    print(a.files)
    signal = a['signal']
    fs = a['fs']
    
    b = np.load("optimized_template_2m_raw.npz")
    print(b.files)
    template = b['template']
    fs = b['fs']
    
    print(f"  2m:   {len(template)} samples ({len(template)/fs*1000:.1f} ms)")
    print(f"  1.8m: {len(signal)} samples ({len(signal)/fs*1000:.1f} ms)")


    shift_us, corr, lags_us = xcorr_tof(template, signal, fs)
    print(f"\n--- Cross-correlation result ({str(window_1)}-{str(window_2)}ms window) ---")
    print(f"TOF shift: {shift_us:.2f} μs")

    # Expected shift for 10cm displacement (20cm round-trip)
    expected_us = 0.20 / 343 * 1e6
    print(f"\nExpected for 10cm displacement: {expected_us:.0f} μs")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    print(len(template)*fs)
    print(len(signal)*fs)
    # Filtered signals in reflection window
    #filt_2m = bandpass_filter(template, fs)
    #filt_18m = bandpass_filter(signal, fs)
    start_idx = int(window_1 * fs / 1000)
    end_idx = int(window_2 * fs / 1000)
    t_ms = np.linspace(window_1, window_2, end_idx - start_idx)
    
    #t_ms_whole = np.linspace(0, 20, end_idx_1)

    """
    axes[0].plot(t_ms, filt_2m[start_idx:end_idx], 'b-', linewidth=0.8, alpha=0.8, label='2m')
    axes[0].plot(t_ms, filt_18m[start_idx:end_idx], 'r-', linewidth=0.8, alpha=0.8, label='1.8m')
    """
    
    axes[0].plot(t_ms, template[start_idx:end_idx], 'b-', linewidth=0.8, alpha=0.8, label='2m')
    axes[0].plot(t_ms, signal[start_idx:end_idx], 'r-', linewidth=0.8, alpha=0.8, label='1.8m')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Filtered signal')
    axes[0].set_title(f'Reflection window ({window_1}-{window_2}ms)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Cross-correlation
    axes[1].plot(lags_us, corr, 'b-', linewidth=1)
    axes[1].axvline(shift_us, color='r', linestyle='--', linewidth=2,
                    label=f'Peak: {shift_us:.1f} μs')
    axes[1].axvline(0, color='k', linestyle=':', alpha=0.5)
    axes[1].set_xlabel('Lag (μs)')
    axes[1].set_ylabel('Correlation')
    axes[1].set_title('Cross-correlation (1.8m vs 2m)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(-2000, 2000)

    plt.tight_layout()
    plt.savefig('xcorr_tof_result_cyrus_optimized.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()












