#!/usr/bin/env python3
"""
GCC-PHAT (Generalized Cross-Correlation with Phase Transform) analysis.

Compares standard cross-correlation with GCC-PHAT for TOF measurement
across different preprocessing levels:
1. Bandpass-filtered envelope (Hilbert) - NOTE: GCC-PHAT not suitable for envelope
2. Bandpass-filtered with 40kHz oscillation (no Hilbert)
3. Raw (no bandpass, no Hilbert)

GCC-PHAT requires phase information (carrier oscillation) to work properly.
For envelope-only data, standard xcorr is more appropriate.
"""

import numpy as np
from scipy.signal import correlate
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def gcc_phat(sig1, sig2, fs, search_range_us=2000):
    """
    Generalized Cross-Correlation with Phase Transform.

    Whitens the cross-spectrum by normalizing by magnitude,
    producing a sharper correlation peak.

    Parameters:
        sig1: Reference signal (template)
        sig2: Test signal
        fs: Sample rate (Hz)
        search_range_us: Range to search for peak (microseconds from center)

    Returns:
        corr: GCC-PHAT correlation
        lags_us: Lag values in microseconds
    """
    n = len(sig1) + len(sig2) - 1
    nfft = 2 ** int(np.ceil(np.log2(n)))

    # FFT of both signals
    S1 = np.fft.fft(sig1, nfft)
    S2 = np.fft.fft(sig2, nfft)

    # Cross-spectrum
    cross_spec = S1.conj() * S2

    # Phase transform: normalize by magnitude (with small epsilon to avoid division by zero)
    eps = 1e-10
    phat = cross_spec / (np.abs(cross_spec) + eps)

    # Inverse FFT to get correlation
    corr_full = np.fft.ifft(phat).real

    # Shift to center (like correlate with mode='full')
    corr = np.fft.fftshift(corr_full)

    # Create lag array
    lags = np.arange(-nfft//2, nfft//2)
    lags_us = lags / fs * 1e6

    return corr, lags_us


def standard_xcorr(sig1, sig2, fs):
    """Standard cross-correlation."""
    corr = correlate(sig2, sig1, mode='full')
    n = len(sig1)
    lags = np.arange(-(n-1), n)
    lags_us = lags / fs * 1e6
    return corr, lags_us


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
    Fit a Gaussian to the correlation peak using points >= half maximum.
    """
    peak_val = corr[peak_idx]
    half_max = peak_val / 2

    mask = corr >= half_max
    x_fit = lags_us[mask]
    y_fit = corr[mask]

    if len(x_fit) < 4:
        return lags_us[peak_idx], np.nan, None, None, x_fit, y_fit

    amp_guess = peak_val
    mu_guess = lags_us[peak_idx]
    sigma_guess = 100
    offset_guess = 0

    p0 = [amp_guess, mu_guess, sigma_guess, offset_guess]

    try:
        popt, pcov = curve_fit(gaussian, x_fit, y_fit, p0=p0, maxfev=5000)
        mu = popt[1]
        sigma = abs(popt[2])
        return mu, sigma, popt, pcov, x_fit, y_fit
    except RuntimeError:
        return lags_us[peak_idx], np.nan, None, None, x_fit, y_fit


def analyze_and_plot(template, signal, fs, window_ms, title_prefix, output_filename,
                     search_range_us=1500):
    """
    Run both standard xcorr and GCC-PHAT, create 4-panel comparison plot.

    Parameters:
        search_range_us: Search for GCC-PHAT peak within this range of expected TOF
    """
    start_idx = int(window_ms[0] * fs / 1000)
    end_idx = int(window_ms[1] * fs / 1000)

    seg_template = template[start_idx:end_idx]
    seg_signal = signal[start_idx:end_idx]

    # Standard cross-correlation
    corr_std, lags_std = standard_xcorr(seg_template, seg_signal, fs)
    peak_idx_std = np.argmax(corr_std)
    peak_refined_std = parabolic_interp(corr_std, peak_idx_std)
    n_std = len(seg_template)
    shift_samples_std = peak_refined_std - (n_std - 1)
    shift_us_std = shift_samples_std / fs * 1e6

    # GCC-PHAT
    corr_phat, lags_phat = gcc_phat(seg_template, seg_signal, fs)

    # For GCC-PHAT, search for peak near the expected TOF (use std result as guide)
    # This avoids spurious peaks at lag=0 for envelope data
    expected_lag_us = shift_us_std  # Use standard xcorr result as reference
    search_mask = np.abs(lags_phat - expected_lag_us) < search_range_us

    if np.any(search_mask):
        search_indices = np.where(search_mask)[0]
        local_peak_idx = search_indices[np.argmax(corr_phat[search_mask])]
        peak_idx_phat = local_peak_idx
    else:
        peak_idx_phat = np.argmax(corr_phat)

    peak_refined_phat = parabolic_interp(corr_phat, peak_idx_phat)
    shift_samples_phat = peak_refined_phat - len(corr_phat)//2
    shift_us_phat = shift_samples_phat / fs * 1e6

    # Gaussian fits
    mu_std, sigma_std, popt_std, _, x_fit_std, y_fit_std = fit_gaussian_to_peak(
        corr_std, lags_std, peak_idx_std)
    mu_phat, sigma_phat, popt_phat, _, x_fit_phat, y_fit_phat = fit_gaussian_to_peak(
        corr_phat, lags_phat, peak_idx_phat)

    # Expected shift for 10cm displacement (20cm round-trip)
    expected_us = 0.20 / 343 * 1e6

    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top left: Reflection window signals
    t_ms = np.linspace(window_ms[0], window_ms[1], len(seg_template))
    axes[0, 0].plot(t_ms, seg_template, 'b-', linewidth=0.8, alpha=0.8, label='2m template')
    axes[0, 0].plot(t_ms, seg_signal, 'r-', linewidth=0.8, alpha=0.8, label='1.8m signal')
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title(f'{title_prefix}\nReflection window ({window_ms[0]}-{window_ms[1]}ms)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Top right: Full correlations comparison
    # Normalize for comparison
    corr_std_norm = corr_std / np.max(corr_std)
    corr_phat_norm = corr_phat / np.max(corr_phat)

    axes[0, 1].plot(lags_std, corr_std_norm, 'b-', linewidth=1, alpha=0.7, label='Standard XCorr')
    axes[0, 1].plot(lags_phat, corr_phat_norm, 'r-', linewidth=1, alpha=0.7, label='GCC-PHAT')
    axes[0, 1].axvline(shift_us_std, color='b', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[0, 1].axvline(shift_us_phat, color='r', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[0, 1].axvline(0, color='k', linestyle=':', alpha=0.5)
    axes[0, 1].set_xlabel('Lag (μs)')
    axes[0, 1].set_ylabel('Normalized Correlation')
    axes[0, 1].set_title(f'Correlation comparison\nStd: {shift_us_std:.1f} μs, PHAT: {shift_us_phat:.1f} μs')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(-2000, 2000)

    # Bottom left: Zoomed GCC-PHAT peak with Gaussian fit
    # Determine zoom window based on peak location and width
    if not np.isnan(sigma_phat):
        zoom_center = mu_phat
        zoom_width = max(500, 6 * sigma_phat)  # At least 500 μs or 6 sigma
    else:
        zoom_center = shift_us_phat
        zoom_width = 500

    zoom_mask = np.abs(lags_phat - zoom_center) < zoom_width
    axes[1, 0].plot(lags_phat[zoom_mask], corr_phat[zoom_mask], 'r-', linewidth=1.5,
                    label='GCC-PHAT')

    if popt_phat is not None:
        x_dense = np.linspace(mu_phat - 3*sigma_phat, mu_phat + 3*sigma_phat, 200)
        y_gauss = gaussian(x_dense, *popt_phat)
        axes[1, 0].plot(x_dense, y_gauss, 'g--', linewidth=2,
                        label=f'Gaussian fit (σ={sigma_phat:.1f} μs)')

    axes[1, 0].axvline(shift_us_phat, color='r', linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'Parabolic: {shift_us_phat:.1f} μs')
    if not np.isnan(mu_phat):
        axes[1, 0].axvline(mu_phat, color='g', linestyle='-', linewidth=1.5, alpha=0.7,
                           label=f'Gaussian: {mu_phat:.1f} μs')
    axes[1, 0].set_xlabel('Lag (μs)')
    axes[1, 0].set_ylabel('Correlation')
    axes[1, 0].set_title('GCC-PHAT zoomed peak with Gaussian fit')
    axes[1, 0].legend(loc='upper right', fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom right: Summary comparison
    ax = axes[1, 1]
    methods = ['Expected\n(10cm)', 'Std XCorr\n(parabolic)', 'Std XCorr\n(Gaussian)',
               'GCC-PHAT\n(parabolic)', 'GCC-PHAT\n(Gaussian)']
    values = [expected_us, abs(shift_us_std), abs(mu_std) if not np.isnan(mu_std) else 0,
              abs(shift_us_phat), abs(mu_phat) if not np.isnan(mu_phat) else 0]
    colors = ['gray', 'steelblue', 'royalblue', 'salmon', 'red']

    bars = ax.bar(methods, values, color=colors)
    ax.set_ylabel('|ΔTOF| (μs)')
    ax.set_title('Method comparison')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    # Add FWHM info
    fwhm_std = 2.355 * sigma_std if not np.isnan(sigma_std) else np.nan
    fwhm_phat = 2.355 * sigma_phat if not np.isnan(sigma_phat) else np.nan
    info_text = f'FWHM Std: {fwhm_std:.1f} μs\nFWHM PHAT: {fwhm_phat:.1f} μs'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'{title_prefix} - Standard XCorr vs GCC-PHAT', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    print(f'Saved: {output_filename}')

    # Print results
    print(f'\n{title_prefix}')
    print(f'  Standard XCorr - Parabolic: {shift_us_std:.2f} μs, Gaussian: {mu_std:.2f} μs, FWHM: {fwhm_std:.1f} μs')
    print(f'  GCC-PHAT       - Parabolic: {shift_us_phat:.2f} μs, Gaussian: {mu_phat:.2f} μs, FWHM: {fwhm_phat:.1f} μs')
    print(f'  Expected: {expected_us:.0f} μs')

    return {
        'std_parabolic': shift_us_std,
        'std_gaussian': mu_std,
        'std_fwhm': fwhm_std,
        'phat_parabolic': shift_us_phat,
        'phat_gaussian': mu_phat,
        'phat_fwhm': fwhm_phat
    }


def main():
    window_ms = (5, 15)  # Reflection window

    print("=" * 60)
    print("GCC-PHAT vs Standard Cross-Correlation Analysis")
    print("=" * 60)

    results = {}

    # 1. Bandpass-filtered envelope (Hilbert)
    print("\n--- Loading envelope data ---")
    template_env = np.load('template_2m_averaged.npz')
    signal_env = np.load('signal_1.8m_envelope.npz')

    t_env = template_env['template']
    s_env = signal_env['envelope']
    fs = int(template_env['fs'])

    print(f"  Template: {len(t_env)} samples, Signal: {len(s_env)} samples, fs={fs} Hz")

    results['envelope'] = analyze_and_plot(
        t_env, s_env, fs, window_ms,
        'Bandpass + Envelope (Hilbert)',
        'gcc_phat_envelope.png'
    )

    # 2. Bandpass-filtered, no Hilbert (40kHz oscillation)
    print("\n--- Loading filtered no-hilbert data ---")
    template_filt = np.load('template_2m_filterednohilbert_20ms.npz')
    signal_filt = np.load('signal_1.8m_filterednohilbert_20ms.npz')

    t_filt = template_filt['template']
    s_filt = signal_filt['signal']
    fs = int(template_filt['fs'])

    print(f"  Template: {len(t_filt)} samples, Signal: {len(s_filt)} samples, fs={fs} Hz")

    results['filtered_nohilbert'] = analyze_and_plot(
        t_filt, s_filt, fs, window_ms,
        'Bandpass Only (40kHz oscillation)',
        'gcc_phat_filterednohilbert.png'
    )

    # 3. No bandpass, no Hilbert (raw)
    print("\n--- Loading raw data ---")
    template_raw = np.load('template_2m_nofilternohilbert_20ms.npz')
    signal_raw = np.load('signal_1.8m_nofilternohilbert_20ms.npz')

    t_raw = template_raw['template']
    s_raw = signal_raw['signal']
    fs = int(template_raw['fs'])

    print(f"  Template: {len(t_raw)} samples, Signal: {len(s_raw)} samples, fs={fs} Hz")

    results['raw'] = analyze_and_plot(
        t_raw, s_raw, fs, window_ms,
        'Raw (no bandpass, no Hilbert)',
        'gcc_phat_nofilternohilbert.png'
    )

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Data Type':<30} {'Std FWHM':<12} {'PHAT FWHM':<12} {'Improvement':<12}")
    print("-" * 66)
    for key, r in results.items():
        if not np.isnan(r['std_fwhm']) and not np.isnan(r['phat_fwhm']):
            improvement = (r['std_fwhm'] - r['phat_fwhm']) / r['std_fwhm'] * 100
            print(f"{key:<30} {r['std_fwhm']:<12.1f} {r['phat_fwhm']:<12.1f} {improvement:>+.1f}%")
        else:
            print(f"{key:<30} {r['std_fwhm']:<12} {r['phat_fwhm']:<12} N/A")

    plt.show()


if __name__ == '__main__':
    main()
