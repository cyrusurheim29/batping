#!/usr/bin/env python3
"""
Regenerate optimized_xcorr_raw.png using properly prepared npz data.
- Template: average of first 10 periods from 2m recording
- Signal: first period from 1.8m recording
- Raw data (no bandpass, no Hilbert on input)
"""

import numpy as np
from scipy.signal import correlate, hilbert
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def parabolic_interp(y, idx):
    if idx <= 0 or idx >= len(y) - 1:
        return float(idx)
    y0, y1, y2 = y[idx-1], y[idx], y[idx+1]
    denom = y0 - 2*y1 + y2
    if abs(denom) < 1e-10:
        return float(idx)
    return idx + 0.5 * (y0 - y2) / denom


def gaussian(x, amp, mu, sigma, offset):
    return amp * np.exp(-0.5 * ((x - mu) / sigma)**2) + offset


def fit_gaussian_to_peak(corr, lags_us, peak_idx):
    peak_val = corr[peak_idx]
    half_max = peak_val / 2
    mask = corr >= half_max
    x_fit = lags_us[mask]
    y_fit = corr[mask]

    p0 = [peak_val, lags_us[peak_idx], 100, 0]
    try:
        popt, pcov = curve_fit(gaussian, x_fit, y_fit, p0=p0, maxfev=5000)
        return popt[1], abs(popt[2]), popt, pcov, x_fit, y_fit
    except:
        return lags_us[peak_idx], np.nan, None, None, x_fit, y_fit


def main():
    data_dir = '/Users/cyrusurheim_1/Downloads/Stubbs_Stuff/Claude_Dir/agilent_project/matched_filtering_v2_june1-5'

    # Load npz files
    t_data = np.load(f'{data_dir}/optimized_template_2m_raw.npz')
    s_data = np.load(f'{data_dir}/optimized_signal_1.8m_raw.npz')

    template = t_data['template']
    signal = s_data['signal']
    fs = int(t_data['fs'])
    period_ms = float(t_data['period_ms'])

    print(f"Loaded: template {template.shape}, signal {signal.shape}, fs={fs}, period={period_ms}ms")

    # Use full period for xcorr
    window_ms = (0, period_ms)

    # Cross-correlation on RAW signals
    corr = correlate(signal, template, mode='full')
    n = len(template)
    lags = np.arange(-(n-1), n)
    lags_us = lags / fs * 1e6

    peak_idx = np.argmax(corr)
    peak_refined = parabolic_interp(corr, peak_idx)
    shift_samples = peak_refined - (n - 1)
    shift_us = shift_samples / fs * 1e6

    # CCF envelope for FWHM measurement
    corr_envelope = np.abs(hilbert(corr))
    fit_peak_idx = np.argmax(corr_envelope)

    mu_gauss, sigma_gauss, popt, pcov, x_fit, y_fit = fit_gaussian_to_peak(
        corr_envelope, lags_us, fit_peak_idx)
    fwhm_us = 2.355 * sigma_gauss if not np.isnan(sigma_gauss) else np.nan

    expected_us = 0.20 / 343 * 1e6  # 10cm displacement = 20cm round-trip

    print(f"Peak lag: {shift_us:.1f} us")
    print(f"Gaussian peak: {mu_gauss:.1f} us")
    print(f"FWHM: {fwhm_us:.0f} us")

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Optimized Waveform - Raw Data (no bandpass, no Hilbert)', fontsize=14, fontweight='bold')

    # Top left: RAW signals
    t_ms = np.linspace(0, period_ms, len(template))

    axes[0, 0].plot(t_ms, template, 'b-', linewidth=0.5, alpha=0.8, label='2m template (raw)')
    axes[0, 0].plot(t_ms, signal, 'r-', linewidth=0.5, alpha=0.8, label='1.8m signal (raw)')
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title(f'Raw Signals (0-{period_ms:.0f}ms, one period)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Top right: Full cross-correlation centered on peak
    peak_lag = lags_us[peak_idx]
    half_range = 1500
    xlim_left = peak_lag - half_range
    xlim_right = peak_lag + half_range

    axes[0, 1].plot(lags_us, corr, 'b-', linewidth=0.8, alpha=0.6, label='Raw CCF')
    axes[0, 1].plot(lags_us, corr_envelope, 'orange', linewidth=1.5,
                    label=f'CCF Envelope (FWHM={fwhm_us:.0f} us)' if not np.isnan(fwhm_us) else 'CCF Envelope')
    axes[0, 1].axvline(shift_us, color='r', linestyle='--', linewidth=2,
                       label=f'Parabolic: {shift_us:.1f} us')
    axes[0, 1].axvline(mu_gauss, color='g', linestyle='--', linewidth=2,
                       label=f'Gaussian: {mu_gauss:.1f} us')
    axes[0, 1].set_xlabel('Lag (us)')
    axes[0, 1].set_ylabel('Correlation')
    axes[0, 1].set_title('Cross-correlation (1.8m vs 2m)')
    axes[0, 1].legend(loc='upper right', fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(xlim_left, xlim_right)

    # Bottom left: Zoomed peak with Gaussian fit
    zoom_half = 800
    zoom_left = mu_gauss - zoom_half
    zoom_right = mu_gauss + zoom_half
    zoom_mask = (lags_us >= zoom_left) & (lags_us <= zoom_right)

    axes[1, 0].plot(lags_us[zoom_mask], corr[zoom_mask], 'b-', linewidth=0.8, alpha=0.6, label='Raw CCF')
    axes[1, 0].plot(lags_us[zoom_mask], corr_envelope[zoom_mask], 'orange', linewidth=1.5, label='CCF Envelope')

    if popt is not None:
        x_dense = np.linspace(zoom_left, zoom_right, 500)
        y_gauss = gaussian(x_dense, *popt)
        axes[1, 0].plot(x_dense, y_gauss, 'g--', linewidth=2,
                        label=f'Gaussian fit (sigma={sigma_gauss:.1f} us)')

    axes[1, 0].axvline(shift_us, color='r', linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'Parabolic: {shift_us:.1f} us')
    axes[1, 0].axvline(mu_gauss, color='g', linestyle='-', linewidth=1.5, alpha=0.7,
                       label=f'Gaussian: {mu_gauss:.1f} us')
    axes[1, 0].set_xlabel('Lag (us)')
    axes[1, 0].set_ylabel('Correlation')
    axes[1, 0].set_title('Zoomed peak with Gaussian fit')
    axes[1, 0].legend(loc='upper right', fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(zoom_left, zoom_right)

    # Bottom right: Summary comparison
    ax = axes[1, 1]
    methods = ['Expected\n(10cm)', 'Parabolic', 'Gaussian']
    values = [expected_us, abs(shift_us), abs(mu_gauss)]
    colors = ['orange', 'steelblue', 'green']
    bars = ax.bar(methods, values, color=colors)
    ax.set_ylabel('|delta TOF| (us)')
    ax.set_title('Comparison of methods')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{val:.1f} us', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    output_path = f'{data_dir}/optimized_xcorr_raw.png'
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved: {output_path}")
    plt.close()


if __name__ == '__main__':
    main()
