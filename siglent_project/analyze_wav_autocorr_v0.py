#!/usr/bin/env python3
"""
Analyze WAV file autocorrelation (v0 - single period analysis).

Configure the variables below, then run:
    python analyze_wav_autocorr_v0.py
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import hilbert, butter, filtfilt
import matplotlib.pyplot as plt



WAV_FILE = '/Users/cyrusurheim_1/Downloads/Stubbs_Stuff/Claude_Dir/siglent_project/recordings/20260619_114952_distance_prbs/raw/chirp_lfm.wav'

# Time range to analyze (seconds). Set to None for full file.
T_START = 0.015
T_END = 0.04

# Bandpass filter settings (Hz). Set both to None to skip filtering.
FILTER_LOW = None
FILTER_HIGH = None

# Autocorrelation lag range to plot (microseconds)
LAG_MIN_US = -25000
LAG_MAX_US = 25000

# TOF search range (microseconds) - for finding peaks
TOF_SEARCH_MIN_US = -500
TOF_SEARCH_MAX_US = 500

# Peak to analyze (0 if lag-0 peak, 1 TOF peak)
PEAK_SELECTION = 0


def bandpass_filter(sig, fs, low, high, order=4):
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, sig)


def find_peak_parabolic(data, fs, search_min_us, search_max_us, center_idx):
    """
    Find peak in search range using argmax + parabolic interpolation.

    Returns:
        peak_us: Sub-sample peak position in microseconds
        peak_val: Interpolated peak value
    """
    min_lag = int(search_min_us * fs / 1e6)
    max_lag = int(search_max_us * fs / 1e6)

    search_region = data[center_idx + min_lag:center_idx + max_lag]
    if len(search_region) < 3:
        return None, None

    # Find argmax in search region
    local_idx = np.argmax(search_region)

    # Parabolic interpolation using 3 points around peak
    if local_idx < 1 or local_idx >= len(search_region) - 1:
        # Edge case: peak at boundary, can't interpolate
        peak_us = (min_lag + local_idx) / fs * 1e6
        return peak_us, search_region[local_idx]

    y_m1 = search_region[local_idx - 1]
    y_0 = search_region[local_idx]
    y_p1 = search_region[local_idx + 1]

    # Parabola vertex offset from center sample
    denom = 2 * (y_m1 - 2 * y_0 + y_p1)
    if abs(denom) < 1e-10:
        peak_us = (min_lag + local_idx) / fs * 1e6
        return peak_us, y_0

    delta = (y_m1 - y_p1) / denom
    delta = np.clip(delta, -0.5, 0.5)

    refined_idx = local_idx + delta
    peak_us = (min_lag + refined_idx) / fs * 1e6
    peak_val = y_0 - (y_m1 - y_p1) * delta / 4

    return peak_us, peak_val


def main():
    # Load WAV file
    print(f"Loading: {WAV_FILE}")
    fs, data = wavfile.read(WAV_FILE)

    # Convert to float
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0

    print(f"Sample rate: {fs} Hz")
    print(f"Duration: {len(data)/fs*1000:.1f} ms ({len(data)} samples)")

    # Extract time range
    if T_START is not None or T_END is not None:
        start_idx = int((T_START or 0) * fs)
        end_idx = int((T_END or len(data)/fs) * fs)
        data = data[start_idx:end_idx]
        print(f"Analyzing: {T_START or 0:.3f}s to {T_END or len(data)/fs:.3f}s ({len(data)} samples)")

    # Apply bandpass filter
    if FILTER_LOW is not None and FILTER_HIGH is not None:
        print(f"Bandpass filter: {FILTER_LOW/1000:.0f}-{FILTER_HIGH/1000:.0f} kHz")
        data = bandpass_filter(data, fs, FILTER_LOW, FILTER_HIGH)

    # Compute autocorrelation
    print("Computing autocorrelation...")
    autocorr = np.correlate(data, data, mode='full')
    center = len(autocorr) // 2
    lags_samples = np.arange(len(autocorr)) - center
    lags_us = lags_samples / fs * 1e6

    # Compute envelope
    envelope = np.abs(hilbert(autocorr))

    # Find peaks in TOF search range using parabolic interpolation
    tof_us, peak_val = find_peak_parabolic(envelope, fs, TOF_SEARCH_MIN_US, TOF_SEARCH_MAX_US, center)
    if tof_us is not None:
        main_peak = envelope[center]
        ratio_db = 20 * np.log10(peak_val / main_peak + 1e-10)
        print(f"\nTOF peak: {tof_us:.1f} μs (ratio to main: {ratio_db:.1f} dB)")

    # Measure FWHM of main peak
    main_peak = envelope[center]
    half_max = main_peak / 2

    # Find where envelope crosses half-max on right side
    right_side = envelope[center:]
    cross_idx = np.where(right_side < half_max)[0]
    if len(cross_idx) > 0:
        fwhm_samples = cross_idx[0] * 2  # symmetric
        fwhm_us = fwhm_samples / fs * 1e6
        print(f"FWHM: {fwhm_us:.0f} μs")
    else:
        fwhm_us = None

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Raw signal
    t_ms = np.arange(len(data)) / fs * 1000
    axes[0].plot(t_ms, data, 'b-', linewidth=0.5)
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'Signal: {WAV_FILE.split("/")[-1]}')
    axes[0].grid(True, alpha=0.3)

    # Full autocorrelation
    lag_mask = (lags_us >= LAG_MIN_US) & (lags_us <= LAG_MAX_US)
    axes[1].plot(lags_us[lag_mask], autocorr[lag_mask], 'b-', linewidth=0.5, alpha=0.5, label='Autocorr')
    axes[1].plot(lags_us[lag_mask], envelope[lag_mask], 'r-', linewidth=1, label='Envelope')
    axes[1].set_xlabel('Lag (μs)')
    axes[1].set_ylabel('Autocorrelation')
    axes[1].set_title('Autocorrelation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Zoom on TOF region
    if PEAK_SELECTION == 0:
        zoom_mask = (lags_us >= TOF_SEARCH_MIN_US*1.5) & (lags_us <= TOF_SEARCH_MAX_US * 1.5)
        peak_var = "lag0"
        plot_title = "Lag-0 Region"
    else:
        zoom_mask = (lags_us >= 0) & (lags_us <= TOF_SEARCH_MAX_US * 1.5)
        peak_var = "tof"
        plot_title = "TOF Region"
    axes[2].plot(lags_us[zoom_mask], envelope[zoom_mask], 'r-', linewidth=1, label="Envelope")
    axes[2].plot(lags_us[zoom_mask], autocorr[zoom_mask], 'b-', linewidth=0.5, alpha=0.5, label="Autocorr")
    axes[2].axvline(TOF_SEARCH_MIN_US, color='g', linestyle='--', alpha=0.5, label='Search range')
    axes[2].axvline(TOF_SEARCH_MAX_US, color='g', linestyle='--', alpha=0.5)
    if tof_us is not None:
        axes[2].axvline(tof_us, color='orange', linewidth=2, label=f'TOF = {tof_us:.1f} μs')
    axes[2].set_xlabel('Lag (μs)')
    axes[2].set_ylabel('Envelope')
    axes[2].set_title(plot_title)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Add info panel
    peak_type = "lag-0" if (TOF_SEARCH_MIN_US <= 0 <= TOF_SEARCH_MAX_US) else "TOF"
    info_text = f"FWHM: {fwhm_us:.0f} μs\nPeak: {peak_type}"
    if tof_us is not None and peak_type == "TOF":
        info_text += f"\nRatio to Main: {ratio_db:.1f} dB"
    fig.text(0.98, 0.98, info_text, transform=fig.transFigure,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Save plot
    if FILTER_LOW is None and FILTER_HIGH is None:
        filter_var = "nobpf"
    else:
        filter_var = f"{FILTER_LOW}-{FILTER_HIGH}kHz"
    new_suffix = f"_autocorr_{T_START}-{T_END}ms_{filter_var}_{peak_var}_{TOF_SEARCH_MIN_US}-{TOF_SEARCH_MAX_US}.png"
    out_path = WAV_FILE.replace('.wav', new_suffix).replace('raw', 'plots')
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot: {out_path}")

    plt.show()


if __name__ == '__main__':
    main()
