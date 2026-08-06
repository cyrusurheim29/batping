#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 18:14:39 2026

@author: cyrusurheimog
"""

import numpy as np
from scipy.signal import butter, filtfilt, hilbert, correlate
from scipy.optimize import curve_fit
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


def gaussian(x, amp, mu, sigma, offset):
    """Gaussian function for curve fitting."""
    return amp * np.exp(-0.5 * ((x - mu) / sigma)**2) + offset


def fit_gaussian_to_peak(corr, lags_us, peak_idx):
    """
    Fit a Gaussian to the cross-correlation peak using only points
    between half maximum and full maximum.

    Parameters:
        corr: Cross-correlation array
        lags_us: Lag values in microseconds
        peak_idx: Index of the peak in corr array

    Returns:
        mu: Fitted peak center in microseconds
        sigma: Fitted width (standard deviation) in microseconds
        popt: Full parameter array [amp, mu, sigma, offset]
        pcov: Covariance matrix
    """
    peak_val = corr[peak_idx]
    half_max = peak_val / 2

    # Select only points with amplitude between half max and full max
    mask = corr >= half_max
    x_fit = lags_us[mask]
    y_fit = corr[mask]

    # Initial guesses
    amp_guess = peak_val
    mu_guess = lags_us[peak_idx]
    sigma_guess = 100  # Initial guess for width in μs
    offset_guess = 0

    p0 = [amp_guess, mu_guess, sigma_guess, offset_guess]

    try:
        popt, pcov = curve_fit(gaussian, x_fit, y_fit, p0=p0, maxfev=5000)
        mu = popt[1]
        sigma = abs(popt[2])
        return mu, sigma, popt, pcov, x_fit, y_fit
    except RuntimeError as e:
        print(f"Gaussian fit failed: {e}")
        return lags_us[peak_idx], np.nan, None, None, x_fit, y_fit


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

    return shift_us, corr, lags_us, peak_idx


def main():
    window_1 = 10
    window_2 = 13

    a = np.load("signal_1.8m_envelope.npz")
    print(a.files)
    signal = a['envelope']
    fs = a['fs']

    b = np.load("template_2m_averaged.npz")
    print(b.files)
    template = b['template']
    fs = b['fs']

    print(f"  2m:   {len(template)} samples ({len(template)/fs*1000:.1f} ms)")
    print(f"  1.8m: {len(signal)} samples ({len(signal)/fs*1000:.1f} ms)")

    shift_us, corr, lags_us, peak_idx = xcorr_tof(template, signal, fs, window_ms=(window_1, window_2))

    # Gaussian fit
    mu_gauss, sigma_gauss, popt, pcov, x_fit, y_fit = fit_gaussian_to_peak(corr, lags_us, peak_idx)

    print(f"\n--- Cross-correlation result ({window_1}-{window_2}ms window) ---")
    print(f"TOF shift (parabolic): {shift_us:.2f} μs")
    print(f"TOF shift (Gaussian):  {mu_gauss:.2f} μs")
    if popt is not None:
        print(f"Gaussian sigma:        {sigma_gauss:.2f} μs")
        print(f"Gaussian FWHM:         {2.355 * sigma_gauss:.2f} μs")

    # Expected shift for 10cm displacement (20cm round-trip)
    expected_us = 0.20 / 343 * 1e6
    print(f"\nExpected for 10cm displacement: {expected_us:.0f} μs")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top left: Reflection window signals
    start_idx = int(window_1 * fs / 1000)
    end_idx = int(window_2 * fs / 1000)
    t_ms = np.linspace(window_1, window_2, end_idx - start_idx)

    axes[0, 0].plot(t_ms, template[start_idx:end_idx], 'b-', linewidth=0.8, alpha=0.8, label='2m')
    axes[0, 0].plot(t_ms, signal[start_idx:end_idx], 'r-', linewidth=0.8, alpha=0.8, label='1.8m')
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].set_ylabel('Envelope')
    axes[0, 0].set_title(f'Reflection window ({window_1}-{window_2}ms)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Top right: Full cross-correlation
    axes[0, 1].plot(lags_us, corr, 'b-', linewidth=1)
    axes[0, 1].axvline(shift_us, color='r', linestyle='--', linewidth=2,
                       label=f'Parabolic: {shift_us:.1f} μs')
    axes[0, 1].axvline(mu_gauss, color='g', linestyle='--', linewidth=2,
                       label=f'Gaussian: {mu_gauss:.1f} μs')
    axes[0, 1].axvline(0, color='k', linestyle=':', alpha=0.5)
    axes[0, 1].set_xlabel('Lag (μs)')
    axes[0, 1].set_ylabel('Correlation')
    axes[0, 1].set_title('Cross-correlation (1.8m vs 2m)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(-2000, 2000)

    # Bottom left: Zoomed cross-correlation with Gaussian fit
    zoom_width = 1000  # μs
    zoom_mask = np.abs(lags_us - mu_gauss) < zoom_width
    axes[1, 0].plot(lags_us[zoom_mask], corr[zoom_mask], 'b-', linewidth=1.5, label='Cross-correlation')

    if popt is not None:
        x_dense = np.linspace(x_fit[0], x_fit[-1], 500)
        y_gauss = gaussian(x_dense, *popt)
        axes[1, 0].plot(x_dense, y_gauss, 'g--', linewidth=2, label=f'Gaussian fit (σ={sigma_gauss:.1f} μs)')

    axes[1, 0].axvline(shift_us, color='r', linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'Parabolic: {shift_us:.1f} μs')
    axes[1, 0].axvline(mu_gauss, color='g', linestyle='-', linewidth=1.5, alpha=0.7,
                       label=f'Gaussian: {mu_gauss:.1f} μs')
    axes[1, 0].set_xlabel('Lag (μs)')
    axes[1, 0].set_ylabel('Correlation')
    axes[1, 0].set_title('Zoomed peak with Gaussian fit')
    axes[1, 0].legend(loc='upper right')
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom right: Summary comparison
    ax = axes[1, 1]
    methods = ['Expected\n(10cm)', 'Parabolic', 'Gaussian']
    values = [expected_us, abs(shift_us), abs(mu_gauss)]
    colors = ['orange', 'steelblue', 'green']
    bars = ax.bar(methods, values, color=colors)
    ax.set_ylabel('|ΔTOF| (μs)')
    ax.set_title('Comparison of methods')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{val:.1f} μs', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('xcorr_tof_gaussian_fit.png', dpi=150)
    print(f"\nSaved: xcorr_tof_gaussian_fit.png")
    plt.show()


if __name__ == '__main__':
    main()
