#!/usr/bin/env python3
"""
Analyze WAV file autocorrelation.

Configure the variables below, then run:
    python analyze_wav_autocorr.py
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import hilbert, butter, sosfiltfilt 
import matplotlib.pyplot as plt




# Time range to analyze (seconds). Set to None for full file.
T_START = 0.013
T_END = 0.0297

# Bandpass filter settings (Hz). Set both to None to skip filtering.
FILTER_LOW = 30000
FILTER_HIGH = None

# Autocorrelation lag range to plot (microseconds)
LAG_MIN_US = -10000
LAG_MAX_US = 10000

# TOF search range (microseconds) - for finding peaks
TOF_SEARCH_MIN_US = 4000
TOF_SEARCH_MAX_US = 6000

# Peak to analyze (0 if lag-0 peak, 1 TOF peak)
PEAK_SELECTION = 1

# Control for Barker Mode (1 if Barker codes expected, 0 otherwise)
BARKER_MODE = 0

# Rep rate of the waveform generator (Hz). Must match the recording.
# Current setup (barker7_prbs_testing.py, distance_prbs_characterization.py): 40 Hz
# Older Gaussian/optimized recordings (bayesian_optimizer_sdg.py): 62.5 Hz
REP_RATE = 40
AMP_CUTOFF = 0.35


def apply_filter(sig, fs, low, high, order=8):
    nyq = fs / 2
    if low is not None and high is not None:
        sos = butter(order, [low/nyq, high/nyq], btype='band', output='sos')
    elif low is not None:
        sos = butter(order, low/nyq, btype='high', output='sos')
    elif high is not None:
        sos = butter(order, high/nyq, btype='low', output='sos')
    else:
        return sig
    return sosfiltfilt(sos, sig)


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
    #global_idx = center_idx + min_lag + local_idx

    # Parabolic interpolation using 3 points around peak
    # Fit y = a(x - x0)^2 + y0, solve for x0
    # Using samples at idx-1, idx, idx+1
    if local_idx < 1 or local_idx >= len(search_region) - 1:
        # Edge case: peak at boundary, can't interpolate
        peak_us = (min_lag + local_idx) / fs * 1e6
        return peak_us, search_region[local_idx]

    y_m1 = search_region[local_idx - 1]
    y_0 = search_region[local_idx]
    y_p1 = search_region[local_idx + 1]

    # Parabola vertex offset from center sample: delta = (y_m1 - y_p1) / (2 * (y_m1 - 2*y_0 + y_p1))
    denom = 2 * (y_m1 - 2 * y_0 + y_p1)
    if abs(denom) < 1e-10:
        # Flat region, no interpolation
        peak_us = (min_lag + local_idx) / fs * 1e6
        return peak_us, y_0

    delta = (y_m1 - y_p1) / denom
    delta = np.clip(delta, -0.5, 0.5)  # Sanity check

    # Interpolated peak position and value
    refined_idx = local_idx + delta
    peak_us = (min_lag + refined_idx) / fs * 1e6
    peak_val = y_0 - (y_m1 - y_p1) * delta / 4

    return peak_us, peak_val


def match_wg_phase(data, envelope, fs, rep_rate, amp_cutoff):
    """
    This is a crude helper function written by me (Cyrus) to find a time interval
    that captures a single period of the full waveform travel where the piezo responses
    corresponding to the incident and reflected waveforms appearing in the correct
    chronological order. It is crude because it doesn't actually find and account for
    the phase in the signal introduced by the waveform generator (which varies with
    each driving of the wg, hence the necessity of this function). It just locates the approximate
    start time of the reflected peak that comes after at least (period - reflection peak
    duration) seconds.
    """
    period = 1/rep_rate
    period_idx = int(period * fs)
    
    trigger_start_idx = -1
    trigger_end_idx = -1
    t_s = np.arange(len(data)) / fs
    
    for i in range(0, len(envelope)):
        if envelope[i] > amp_cutoff and trigger_start_idx == -1:
            trigger_start_idx = i
            continue
        if envelope[i] < amp_cutoff and trigger_start_idx != -1:
            trigger_end_idx = i
            trigger_start_s = t_s[trigger_start_idx]
            trigger_end_s = t_s[trigger_end_idx]
            #approx_response_duration_s = trigger_end_s - trigger_start_s
            if trigger_start_s - 0.008 < 0:
                trigger_start_idx = -1
                trigger_end_idx = -1
                continue
            break   
    if trigger_start_idx == -1 or trigger_end_idx == -1:
        print ("Invalid amplitude cutoff value for reflection peak detection.")
        return 1
    
    # Back up from reflection to capture incident pulse first
    T_START = trigger_start_s - 0.008  # ~8ms earlier (TOF + margin)
    T_START_IDX = int(T_START * fs)
    T_END = T_START + period
    T_END_IDX = T_START_IDX + period_idx
    
    print(f"T_START: {T_START}, T_END: {T_END}")
    
    j = T_END_IDX
    period_counter = 1
    while j < len(envelope) - period_idx:
        j += period_idx
        period_counter += 1
    
    T_END_LAST_IDX = j
    T_START_LAST_IDX = j - period_idx
    
    phase_corrected_data = data[T_START_IDX:T_END_LAST_IDX]
    phase_corrected_envelope = envelope[T_START_IDX:T_END_LAST_IDX]
    phase_corrected_t_s = t_s[T_START_IDX:T_END_LAST_IDX]
    
    return phase_corrected_data, phase_corrected_envelope, phase_corrected_t_s, T_START_IDX, T_END_IDX, period_counter
    

def main(barker_cyc):
    # Load WAV file
    if barker_cyc is None:
        wav_name = input("What is the name of the .wav file? ")
        wav_folder = input("What is the name of the recording subdir? ")
        WAV_FILE = f"/Users/cyrusurheim_1/Downloads/Stubbs_Stuff/Claude_Dir/siglent_project/recordings/{wav_folder}/raw/{wav_name}"
    else:
        WAV_FILE = f"/Users/cyrusurheim_1/Downloads/Stubbs_Stuff/Claude_Dir/siglent_project/recordings/20260618_174423_barker7_sweep/raw/{barker_cyc}cyc_received.wav"
    print(f"Loading: {WAV_FILE}")
    fs, data = wavfile.read(WAV_FILE)

    # Convert to float
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0

    print(f"Sample rate: {fs} Hz")
    print(f"Duration: {len(data)/fs*1000:.1f} ms ({len(data)} samples)")
    print(f"Waveform Repitition Rate: 1/{REP_RATE} s")

    # Apply filter FIRST — envelope must be computed from filtered data so that
    # match_wg_phase triggers on the actual acoustic signal, not out-of-band noise.
    if FILTER_LOW is not None or FILTER_HIGH is not None:
        if FILTER_LOW is not None and FILTER_HIGH is not None:
            print(f"Bandpass filter: {FILTER_LOW/1000:.0f}-{FILTER_HIGH/1000:.0f} kHz")
        elif FILTER_LOW is not None:
            print(f"Highpass filter: {FILTER_LOW/1000:.0f} kHz")
        else:
            print(f"Lowpass filter: {FILTER_HIGH/1000:.0f} kHz")
        data = apply_filter(data, fs, FILTER_LOW, FILTER_HIGH)

    # Envelope from FILTERED data, normalized to [0,1] so AMP_CUTOFF is relative.
    envelope = np.abs(hilbert(data))
    envelope = envelope / np.max(envelope)

    # Match wg phase and compute autocorrelation for all phase-corrected periods
    period = 1/REP_RATE
    period_idx = int(period * fs)
    data, envelope, t_s, T_START_IDX, T_END_IDX, period_counter = match_wg_phase(data, envelope, fs, REP_RATE, AMP_CUTOFF)
    data_packages = []
    for i in range(0, period_counter-1):
        data_curr = data[i*period_idx:(i+1)*period_idx]
        
        # Compute autocorrelation
        print(f"Computing autocorrelation on period #{i+1} of {period_counter}...")
        autocorr = np.correlate(data_curr, data_curr, mode='full')
        center = len(autocorr) // 2
        lags_samples = np.arange(len(autocorr)) - center
        lags_us = lags_samples / fs * 1e6
        
        # Compute envelope of autocorr
        envelope_autocorr = np.abs(hilbert(autocorr))

        # Find peaks in TOF search range using parabolic interpolation
        tof_us, peak_val = find_peak_parabolic(envelope_autocorr, fs, TOF_SEARCH_MIN_US, TOF_SEARCH_MAX_US, center)
        if tof_us is not None:
            main_peak = envelope_autocorr[center]
            ratio_db = 20 * np.log10(peak_val / main_peak + 1e-10)
            print(f"\nTOF peak: {tof_us:.1f} μs (ratio to main: {ratio_db:.1f} dB)")
    
        # Measure FWHM of main peak
        main_peak = envelope_autocorr[center]
        half_max = main_peak / 2
    
        # Find where envelope_autocorr crosses half-max on right side
        right_side = envelope_autocorr[center:]
        cross_idx = np.where(right_side < half_max)[0]
        if len(cross_idx) > 0:
            fwhm_samples = cross_idx[0] * 2  # symmetric
            fwhm_us = fwhm_samples / fs * 1e6
            print(f"FWHM: {fwhm_us:.0f} μs")
        
        data_curr_package = {
            'autocorr': autocorr,
            'envelope_autocorr': envelope_autocorr,
            'tof_us': tof_us,
            'peak_val': peak_val,
            'ratio_db': ratio_db,
            'fwhm_us': fwhm_us,
        }
        data_packages.append(data_curr_package)

    # Extract metrics and traces
    tof_values = [pkg['tof_us'] for pkg in data_packages if pkg['tof_us'] is not None]
    fwhm_values = [pkg['fwhm_us'] for pkg in data_packages if pkg['fwhm_us'] is not None]
    ratio_values = [pkg['ratio_db'] for pkg in data_packages if pkg['ratio_db'] is not None]
    autocorr_traces = [pkg['autocorr'] for pkg in data_packages]
    envelope_traces = [pkg['envelope_autocorr'] for pkg in data_packages]

    # Report statistics
    mean_tof = np.mean(tof_values) if tof_values else 0
    std_tof = np.std(tof_values) if tof_values else 0
    mean_fwhm = np.mean(fwhm_values) if fwhm_values else 0
    std_fwhm = np.std(fwhm_values) if fwhm_values else 0
    mean_ratio = np.mean(ratio_values) if ratio_values else 0

    print(f"\n--- Averaged over {len(data_packages)} periods ---")
    print(f"TOF: {mean_tof:.1f} ± {std_tof:.1f} μs")
    print(f"FWHM: {mean_fwhm:.0f} ± {std_fwhm:.0f} μs")
    print(f"Ratio to main: {mean_ratio:.1f} dB")

    # Average autocorrelations and envelopes across all periods
    # Averaging envelopes (not envelope of averaged autocorr) is proper incoherent averaging
    avg_autocorr = np.mean(autocorr_traces, axis=0)
    avg_autocorr_envelope = np.mean(envelope_traces, axis=0)

    # First period for raw signal plot
    first_period_data = data[0:period_idx]

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Raw signal (first period only)
    t_ms_first = np.arange(len(first_period_data)) / fs * 1000
    axes[0].plot(t_ms_first, first_period_data, 'b-', linewidth=0.5)
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'Signal (1st period): {WAV_FILE.split("/")[-1]}')
    axes[0].grid(True, alpha=0.3)

    # Full autocorrelation (averaged over all periods)
    lag_mask = (lags_us >= LAG_MIN_US) & (lags_us <= LAG_MAX_US)
    axes[1].plot(lags_us[lag_mask], avg_autocorr[lag_mask], 'b-', linewidth=0.5, alpha=0.5, label='Autocorr')
    axes[1].plot(lags_us[lag_mask], avg_autocorr_envelope[lag_mask], 'r-', linewidth=1, label='Envelope')
    axes[1].set_xlabel('Lag (μs)')
    axes[1].set_ylabel('Autocorrelation')
    axes[1].set_title(f'Autocorrelation (avg of {len(data_packages)} periods)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Zoom on TOF region
    if PEAK_SELECTION == 0:
        zoom_mask = (lags_us >= TOF_SEARCH_MIN_US*1.5) & (lags_us <= TOF_SEARCH_MAX_US * 1.5)
        #peak_region_var = f"-{TOF_SEARCH_MIN_US}-{TOF_SEARCH_MAX_US}"
        peak_var = "lag0"
        plot_title = "Lag-0 Region"
    else:
        zoom_mask = (lags_us >= 0) & (lags_us <= TOF_SEARCH_MAX_US * 1.5)
        #peak_region_var = f""
        peak_var = "tof"
        plot_title = "TOF Region"
    axes[2].plot(lags_us[zoom_mask], avg_autocorr_envelope[zoom_mask], 'r-', linewidth=1, label="Envelope")
    axes[2].plot(lags_us[zoom_mask], avg_autocorr[zoom_mask], 'b-', linewidth=0.5, alpha=0.5, label="Autocorr")
    axes[2].axvline(TOF_SEARCH_MIN_US, color='g', linestyle='--', alpha=0.5, label='Search range')
    axes[2].axvline(TOF_SEARCH_MAX_US, color='g', linestyle='--', alpha=0.5)
    if mean_tof > 0:
        axes[2].axvline(mean_tof, color='orange', linewidth=2, label=f'TOF = {mean_tof:.1f} μs')
    axes[2].set_xlabel('Lag (μs)')
    axes[2].set_ylabel('Envelope')
    axes[2].set_title(f'{plot_title} (averaged)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Add info panel (metrics averaged over all periods)
    peak_type = "lag-0" if (TOF_SEARCH_MIN_US <= 0 <= TOF_SEARCH_MAX_US) else "TOF"
    info_text = f"AVERAGED METRICS\nFWHM: {mean_fwhm:.0f} ± {std_fwhm:.0f} μs\nTOF: {mean_tof:.1f} ± {std_tof:.1f} μs\nPeak: {peak_type}\nRatio: {mean_ratio:.1f} dB\nN periods: {len(data_packages)}"
    fig.text(0.98, 0.12, info_text, transform=fig.transFigure,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Save plot
    if FILTER_LOW is None and FILTER_HIGH is None:
        filter_var = "nofilter"
    elif FILTER_LOW is not None and FILTER_HIGH is not None:
        filter_var = f"bp{FILTER_LOW/1000:.0f}-{FILTER_HIGH/1000:.0f}kHz"
    elif FILTER_LOW is not None:
        filter_var = f"8soshp{FILTER_LOW/1000:.0f}kHz"
    else:
        filter_var = f"lp{FILTER_HIGH/1000:.0f}kHz"
    new_suffix = f"_autocorr_{filter_var}_{peak_var}{TOF_SEARCH_MIN_US}-{TOF_SEARCH_MAX_US}_acorrplot{LAG_MIN_US}-{LAG_MAX_US}_averaging.png"
    out_path = WAV_FILE.replace('.wav', new_suffix).replace('raw', 'plots')
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot: {out_path}")

    plt.show()


if __name__ == '__main__':
    
    if BARKER_MODE == 1:
        barker_list = [16, 20, 24, 28, 32, 36, 40, 44, 48]
    else:
        barker_list = [None]
    for cyc in barker_list:
        main(cyc)
