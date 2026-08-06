#!/usr/bin/env python3
"""
Generate xcorr plots using RAW data (no bandpass, no Hilbert).
Creates 4 plots for different time windows: 0-20, 5-15, 10-15, 10-13 ms
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


def xcorr_tof(sig_template, sig_test, fs, window_ms=(0, 20)):
    start_idx = int(window_ms[0] * fs / 1000)
    end_idx = int(window_ms[1] * fs / 1000)

    seg_template = sig_template[start_idx:end_idx]
    seg_test = sig_test[start_idx:end_idx]

    corr = correlate(seg_test, seg_template, mode='full')
    n = len(seg_template)
    lags = np.arange(-(n-1), n)
    lags_us = lags / fs * 1e6

    peak_idx = np.argmax(corr)
    peak_refined = parabolic_interp(corr, peak_idx)
    shift_samples = peak_refined - (n - 1)
    shift_us = shift_samples / fs * 1e6

    return shift_us, corr, lags_us, peak_idx


def make_plot(template, signal, fs, window_ms, output_name, title_prefix=""):
    """
    Generate the 4-panel xcorr plot for RAW data (no bandpass, no Hilbert on input).
    CCF will oscillate at 40kHz, so we extract envelope from CCF for FWHM measurement.
    """
    window_1, window_2 = window_ms

    shift_us, corr, lags_us, peak_idx = xcorr_tof(template, signal, fs, window_ms)

    # For raw data, CCF has 40kHz carrier oscillations - extract envelope for fitting
    corr_envelope = np.abs(hilbert(corr))
    fit_peak_idx = np.argmax(corr_envelope)

    # Gaussian fit on envelope
    mu_gauss, sigma_gauss, popt, pcov, x_fit, y_fit = fit_gaussian_to_peak(
        corr_envelope, lags_us, fit_peak_idx)
    fwhm_us = 2.355 * sigma_gauss if not np.isnan(sigma_gauss) else np.nan

    expected_us = 0.20 / 343 * 1e6  # 10cm displacement = 20cm round-trip

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    if title_prefix:
        fig.suptitle(title_prefix, fontsize=14, fontweight='bold')

    # Top left: Signals in window
    start_idx = int(window_1 * fs / 1000)
    end_idx = int(window_2 * fs / 1000)
    t_ms = np.linspace(window_1, window_2, end_idx - start_idx)

    axes[0, 0].plot(t_ms, template[start_idx:end_idx], 'b-', linewidth=0.8, alpha=0.8, label='2m template')
    axes[0, 0].plot(t_ms, signal[start_idx:end_idx], 'r-', linewidth=0.8, alpha=0.8, label='1.8m signal')
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title(f'Raw Signals ({window_1}-{window_2}ms window)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Top right: Full cross-correlation - CENTER ON PEAK
    peak_lag = lags_us[peak_idx]
    half_range = 1500  # us on each side of peak
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

    # Bottom left: Zoomed cross-correlation with Gaussian fit - CENTER ON PEAK
    zoom_half = 800  # us on each side
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
    if title_prefix:
        plt.subplots_adjust(top=0.93)
    plt.savefig(output_name, dpi=150)
    print(f"Saved: {output_name}")
    plt.close()

    return shift_us, mu_gauss, fwhm_us


def main():
    # Load raw data (no bandpass, no Hilbert)
    data_dir = '/Users/cyrusurheim_1/Downloads/Stubbs_Stuff/Claude_Dir/agilent_project/matched_filtering_v2_june1-5'

    t = np.load(f'{data_dir}/template_2m_nofilternohilbert_20ms.npz')
    s = np.load(f'{data_dir}/signal_1.8m_nofilternohilbert_20ms.npz')

    template = t['template']
    signal = s['signal']
    fs = t['fs']

    print(f"Loaded raw data: template {template.shape}, signal {signal.shape}, fs={fs}")

    # Generate plots for 4 time windows
    windows = [
        ((0, 20), 'xcorr_tof_raw_0-20.png', 'Raw Data (no bandpass, no Hilbert) - 0-20ms window'),
        ((5, 15), 'xcorr_tof_raw_5-15.png', 'Raw Data (no bandpass, no Hilbert) - 5-15ms window'),
        ((10, 15), 'xcorr_tof_raw_10-15.png', 'Raw Data (no bandpass, no Hilbert) - 10-15ms window'),
        ((10, 13), 'xcorr_tof_raw_10-13.png', 'Raw Data (no bandpass, no Hilbert) - 10-13ms window'),
    ]

    print("\nGenerating plots...")
    results = []

    for window_ms, filename, title in windows:
        output_path = f'{data_dir}/{filename}'
        shift, mu, fwhm = make_plot(template, signal, fs, window_ms, output_path, title)
        results.append((window_ms, shift, mu, fwhm))
        print(f"  {window_ms}: Parabolic={shift:.1f}us, Gaussian={mu:.1f}us, FWHM={fwhm:.0f}us")

    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print(f"{'Window':<12} {'Parabolic (us)':<16} {'Gaussian (us)':<16} {'FWHM (us)':<12}")
    for window_ms, shift, mu, fwhm in results:
        print(f"{str(window_ms):<12} {shift:<16.1f} {mu:<16.1f} {fwhm:<12.0f}")


if __name__ == '__main__':
    main()
