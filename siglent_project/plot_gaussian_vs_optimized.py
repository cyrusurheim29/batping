#!/usr/bin/env python3
"""
Compare Gaussian baseline vs Optimized V3 autocorrelation.
Shows both raw autocorrelation and envelope for each.
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import hilbert, butter, filtfilt
import matplotlib.pyplot as plt

# Files
BASE_DIR = '/Users/cyrusurheim_1/Downloads/Stubbs_Stuff/Claude_Dir/siglent_project/recordings/20260619_114952_distance_prbs/raw/'
GAUSSIAN_WAV = BASE_DIR + 'survey_gaussian.wav'
OPTIMIZED_WAV = BASE_DIR + 'survey_optimized_v3.wav'

OUTPUT_PATH = '/Users/cyrusurheim_1/Downloads/Stubbs_Stuff/Claude_Dir/siglent_project/gaussian_vs_optimized_v3_autocorr.png'

# Analysis parameters
FILTER_LOW = 35000
FILTER_HIGH = 45000
LAG_RANGE_US = 3000  # Plot ± this range around center


def bandpass_filter(sig, fs, low, high, order=4):
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, sig)


def compute_autocorr(wav_path, label):
    print(f"Loading {label}: {wav_path}")
    fs, data = wavfile.read(wav_path)
    data = data.astype(np.float64) / 32768.0

    # Filter
    data_filt = bandpass_filter(data, fs, FILTER_LOW, FILTER_HIGH)

    # Autocorrelation
    autocorr = np.correlate(data_filt, data_filt, mode='full')
    center = len(autocorr) // 2
    lags_us = (np.arange(len(autocorr)) - center) / fs * 1e6

    # Envelope
    envelope = np.abs(hilbert(autocorr))

    # Normalize to peak = 1
    autocorr = autocorr / np.max(np.abs(autocorr))
    envelope = envelope / np.max(envelope)

    # Measure FWHM
    half_max = 0.5
    right_side = envelope[center:]
    cross_idx = np.where(right_side < half_max)[0]
    if len(cross_idx) > 0:
        fwhm_us = cross_idx[0] * 2 / fs * 1e6
    else:
        fwhm_us = None

    print(f"  FWHM: {fwhm_us:.0f} μs" if fwhm_us else "  FWHM: N/A")

    return lags_us, autocorr, envelope, fwhm_us, fs


def main():
    # Load and process both
    lags_g, autocorr_g, env_g, fwhm_g, fs = compute_autocorr(GAUSSIAN_WAV, "Gaussian")
    lags_o, autocorr_o, env_o, fwhm_o, _ = compute_autocorr(OPTIMIZED_WAV, "Optimized V3")

    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Lag mask for zoomed view
    mask_g = np.abs(lags_g) <= LAG_RANGE_US
    mask_o = np.abs(lags_o) <= LAG_RANGE_US

    # Top left: Gaussian raw + envelope
    axes[0, 0].plot(lags_g[mask_g], autocorr_g[mask_g], 'b-', linewidth=0.5, alpha=0.6, label='Raw ACF')
    axes[0, 0].plot(lags_g[mask_g], env_g[mask_g], 'r-', linewidth=1.5, label='Envelope')
    axes[0, 0].axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Half-max')
    axes[0, 0].set_xlabel('Lag (μs)')
    axes[0, 0].set_ylabel('Normalized Amplitude')
    axes[0, 0].set_title(f'Gaussian Baseline (FWHM = {fwhm_g:.0f} μs)')
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(-LAG_RANGE_US, LAG_RANGE_US)

    # Top right: Optimized raw + envelope
    axes[0, 1].plot(lags_o[mask_o], autocorr_o[mask_o], 'b-', linewidth=0.5, alpha=0.6, label='Raw ACF')
    axes[0, 1].plot(lags_o[mask_o], env_o[mask_o], 'r-', linewidth=1.5, label='Envelope')
    axes[0, 1].axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Half-max')
    axes[0, 1].set_xlabel('Lag (μs)')
    axes[0, 1].set_ylabel('Normalized Amplitude')
    axes[0, 1].set_title(f'Optimized V3 (FWHM = {fwhm_o:.0f} μs)')
    axes[0, 1].legend(loc='upper right')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(-LAG_RANGE_US, LAG_RANGE_US)

    # Bottom left: Envelopes overlaid
    axes[1, 0].plot(lags_g[mask_g], env_g[mask_g], 'b-', linewidth=2, label=f'Gaussian ({fwhm_g:.0f} μs)')
    axes[1, 0].plot(lags_o[mask_o], env_o[mask_o], 'r-', linewidth=2, label=f'Optimized V3 ({fwhm_o:.0f} μs)')
    axes[1, 0].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Lag (μs)')
    axes[1, 0].set_ylabel('Normalized Envelope')
    axes[1, 0].set_title('Envelope Comparison')
    axes[1, 0].legend(loc='upper right')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(-LAG_RANGE_US, LAG_RANGE_US)

    # Bottom right: Zoomed comparison (±500 μs)
    zoom_range = 800
    mask_g_zoom = np.abs(lags_g) <= zoom_range
    mask_o_zoom = np.abs(lags_o) <= zoom_range
    axes[1, 1].plot(lags_g[mask_g_zoom], env_g[mask_g_zoom], 'b-', linewidth=2, label=f'Gaussian ({fwhm_g:.0f} μs)')
    axes[1, 1].plot(lags_o[mask_o_zoom], env_o[mask_o_zoom], 'r-', linewidth=2, label=f'Optimized V3 ({fwhm_o:.0f} μs)')
    axes[1, 1].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Lag (μs)')
    axes[1, 1].set_ylabel('Normalized Envelope')
    axes[1, 1].set_title('Zoomed Comparison (±800 μs)')
    axes[1, 1].legend(loc='upper right')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(-zoom_range, zoom_range)

    fig.suptitle('Autocorrelation Comparison: Gaussian vs Optimized V3', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {OUTPUT_PATH}")

    plt.show()


if __name__ == '__main__':
    main()
