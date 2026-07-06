#!/usr/bin/env python3
"""
Stubbs Dual-Template Analysis

Implements Prof. Stubbs's suggested protocol:
  1. Average N_TEMPLATE periods to build two co-added templates:
       T_launch  — the incident (direct) pulse window
       T_return  — the reflected pulse window
  2. For each subsequent period, cross-correlate with each template:
       τ_launch  — lag of best match to T_launch
       τ_return  — lag of best match to T_return
  3. Δτ = τ_return - τ_launch  (differential TOF vs. reference conditions)
     If conditions were identical to when the template was built, Δτ = 0.
     A non-zero Δτ indicates a temperature change.
  4. σ(Δτ) over evaluation periods = TOF precision → temperature precision.

Using raw HP-filtered signal for templates (not Hilbert envelope) — gives better
compression since the cross-correlation captures carrier phase information.
The Hilbert envelope is taken of the *cross-correlation output* to locate the peak,
making it robust to carrier phase offsets between template and measurement.

Usage:
    Set WAV_FILE, REP_RATE, and window parameters at the top, then run.
    Or call analyze(wav_path, rep_rate, ...) directly.

@author: cyrusurheimog
Created: July 2026
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, hilbert, correlate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ── Configuration ─────────────────────────────────────────────────────────────

WAV_FILE = None          # Set to path or pass to analyze(); prompts if None

# Rep rate of the recording (Hz). Must match what was used during recording.
REP_RATE = 40.0

# HP filter (same as rest of pipeline)
HP_CUTOFF = 30000
HP_ORDER  = 8

# Number of periods to co-add into templates
N_TEMPLATE = 100  # 100 periods at 40 Hz = 2.5 s of recording for template build

# Incident and reflection windows, relative to the START of each extracted period (ms).
# Set to None to auto-detect from the first period using find_peaks on the envelope.
# Auto-detection centers a window of WIN_HALF_WIDTH_MS around each detected peak.
# Override with explicit (start_ms, end_ms) tuples if auto-detection picks wrong peaks.
INCIDENT_WIN_MS   = None   # auto-detect
REFLECTION_WIN_MS = None   # auto-detect
WIN_HALF_WIDTH_MS = 3.0    # half-width of each window around the detected peak (ms)

# TOF sensitivity at 20°C, 2m round-trip path
TOF_SENS_US_PER_DEG = 10.3

# Output directory for plots
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Core functions ─────────────────────────────────────────────────────────────

def hp_filter(sig, fs):
    sos = butter(HP_ORDER, HP_CUTOFF / (fs / 2), btype='high', output='sos')
    return sosfiltfilt(sos, sig)


def parabolic_peak(env):
    """Sub-sample peak position via parabolic interpolation. Returns (peak_idx_float, peak_val)."""
    idx = int(np.argmax(env))
    if idx < 1 or idx >= len(env) - 1:
        return float(idx), float(env[idx])
    ym, y0, yp = env[idx-1], env[idx], env[idx+1]
    denom = 2 * (ym - 2*y0 + yp)
    delta = np.clip((ym - yp) / denom, -0.5, 0.5) if abs(denom) > 1e-10 else 0.0
    return float(idx) + delta, float(y0 - (ym - yp) * delta / 4)


def xcorr_lag_us(signal_seg, template, fs):
    """
    Cross-correlate signal_seg with template.
    Returns the lag (μs) of the best-matching peak via Hilbert envelope + parabolic interpolation.
    Lag = 0 means perfect alignment (signal identical to template at offset 0).
    """
    xc = correlate(signal_seg, template, mode='full')
    env = np.abs(hilbert(xc))
    env /= np.max(env) + 1e-12
    peak_idx, _ = parabolic_peak(env)
    center = len(template) - 1          # lag=0 index for mode='full'
    lag_us = (peak_idx - center) / fs * 1e6
    return lag_us, xc, env


def detect_windows(period_data, fs, win_half_width_ms=WIN_HALF_WIDTH_MS):
    """
    Auto-detect incident and reflection windows within a single period.

    Uses match_wg_phase's same logic to find the first strong peak, then finds
    the two largest peaks in the envelope and labels them incident (earlier)
    and reflected (later). Returns ((inc_start_ms, inc_end_ms),
    (ref_start_ms, ref_end_ms)) relative to period start.
    """
    from scipy.signal import find_peaks

    env = np.abs(hilbert(period_data))
    env_n = env / np.max(env)
    t_ms = np.arange(len(period_data)) / fs * 1000

    # Minimum separation between peaks: at least 2ms (must be distinct pulses)
    min_dist_samples = int(0.002 * fs)
    peaks, props = find_peaks(env_n, height=0.15, distance=min_dist_samples)

    if len(peaks) < 2:
        raise ValueError(
            f"Auto-detection found only {len(peaks)} peak(s) in the period. "
            "Try lowering amp_cutoff or set INCIDENT_WIN_MS / REFLECTION_WIN_MS manually."
        )

    # Take the two largest peaks
    heights = props['peak_heights']
    top2_idx = np.argsort(heights)[-2:]
    top2_samples = np.sort(peaks[top2_idx])   # sort by time: incident first

    inc_center_ms  = t_ms[top2_samples[0]]
    ref_center_ms  = t_ms[top2_samples[1]]
    hw = win_half_width_ms
    period_ms = len(period_data) / fs * 1000

    inc_win  = (max(0.0,        inc_center_ms - hw),
                min(period_ms, inc_center_ms + hw))
    ref_win  = (max(0.0,        ref_center_ms - hw),
                min(period_ms, ref_center_ms + hw))

    print(f"  Auto-detected incident  peak at {inc_center_ms:.2f} ms  "
          f"→ window [{inc_win[0]:.2f}, {inc_win[1]:.2f}] ms")
    print(f"  Auto-detected reflection peak at {ref_center_ms:.2f} ms  "
          f"→ window [{ref_win[0]:.2f}, {ref_win[1]:.2f}] ms")
    return inc_win, ref_win


def slice_periods(data, fs, rep_rate, amp_cutoff=0.25):
    """
    Find period start index via first strong peak after 8ms, then slice uniformly.
    Returns list of (start_idx, end_idx) tuples.
    """
    period_idx = int(fs / rep_rate)
    env = np.abs(hilbert(data))
    env_n = env / np.max(env)
    t_s = np.arange(len(data)) / fs

    T0 = -1
    in_peak = False
    for i in range(len(env_n)):
        if env_n[i] > amp_cutoff and not in_peak:
            in_peak = True
            if t_s[i] - 0.008 >= 0:
                T0 = int((t_s[i] - 0.008) * fs)
                break
        elif env_n[i] < amp_cutoff:
            in_peak = False

    if T0 < 0:
        T0 = 0

    slices = []
    i = 0
    while T0 + (i + 1) * period_idx <= len(data):
        start = T0 + i * period_idx
        end   = T0 + (i + 1) * period_idx
        slices.append((start, end))
        i += 1
    return slices


def extract_window(period_data, fs, win_ms):
    """Extract a sub-window from a period. win_ms = (start_ms, end_ms) from period start."""
    s = int(win_ms[0] / 1000 * fs)
    e = int(win_ms[1] / 1000 * fs)
    s = max(0, min(s, len(period_data) - 1))
    e = max(s + 1, min(e, len(period_data)))
    return period_data[s:e]


# ── Main analysis ──────────────────────────────────────────────────────────────

def analyze(wav_path, rep_rate=REP_RATE, n_template=N_TEMPLATE,
            incident_win_ms=INCIDENT_WIN_MS, reflection_win_ms=REFLECTION_WIN_MS,
            win_half_width_ms=WIN_HALF_WIDTH_MS, output_dir=OUTPUT_DIR):
    """
    Run Stubbs dual-template analysis on a WAV file.
    Returns dict with templates, per-period lags, and precision metrics.
    """
    fs, raw = wavfile.read(wav_path)
    data = raw.astype(np.float64) / 32768.0 if raw.dtype == np.int16 else raw.astype(np.float64)

    print(f"Loaded: {os.path.basename(wav_path)}")
    print(f"  Duration: {len(data)/fs*1000:.0f} ms  |  FS: {fs} Hz  |  Rep rate: {rep_rate} Hz")

    # Filter
    sig = hp_filter(data, fs)

    # Slice into periods
    slices = slice_periods(sig, fs, rep_rate)
    n_periods = len(slices)
    print(f"  Periods found: {n_periods}")

    if n_periods < n_template + 5:
        print(f"  WARNING: only {n_periods} periods, need at least {n_template + 5}. "
              f"Reducing n_template to {max(1, n_periods - 5)}.")
        n_template = max(1, n_periods - 5)

    # Auto-detect windows from first period if not explicitly provided
    if incident_win_ms is None or reflection_win_ms is None:
        first_period = sig[slices[0][0]:slices[0][1]]
        incident_win_ms, reflection_win_ms = detect_windows(
            first_period, fs, win_half_width_ms)

    # Extract windows from all periods
    inc_segs  = []
    ref_segs  = []
    for (s, e) in slices:
        period = sig[s:e]
        inc_segs.append(extract_window(period, fs, incident_win_ms))
        ref_segs.append(extract_window(period, fs, reflection_win_ms))

    # Pad to common length within each group
    def pad_list(segs):
        L = min(len(s) for s in segs)
        return np.array([s[:L] for s in segs])

    inc_arr = pad_list(inc_segs)
    ref_arr = pad_list(ref_segs)

    # Build templates from first n_template periods
    T_launch = np.mean(inc_arr[:n_template], axis=0)
    T_return = np.mean(ref_arr[:n_template], axis=0)
    print(f"  Templates built from {n_template} periods  "
          f"(incident {incident_win_ms[0]}–{incident_win_ms[1]} ms, "
          f"reflection {reflection_win_ms[0]}–{reflection_win_ms[1]} ms)")

    # Evaluate on remaining periods
    tau_launch_list = []
    tau_return_list = []
    for i in range(n_template, n_periods):
        y_launch = inc_arr[i]
        y_return = ref_arr[i]

        τl, xc_l, env_l = xcorr_lag_us(y_launch, T_launch, fs)
        τr, xc_r, env_r = xcorr_lag_us(y_return, T_return, fs)

        tau_launch_list.append(τl)
        tau_return_list.append(τr)

    tau_launch = np.array(tau_launch_list)
    tau_return = np.array(tau_return_list)
    delta_tau  = tau_return - tau_launch

    n_eval = len(delta_tau)
    mean_dt = np.mean(delta_tau)
    std_dt  = np.std(delta_tau)
    temp_precision = std_dt / TOF_SENS_US_PER_DEG

    print(f"\n  Evaluation periods: {n_eval}")
    print(f"  τ_launch:  mean={np.mean(tau_launch):.2f} μs  σ={np.std(tau_launch):.3f} μs")
    print(f"  τ_return:  mean={np.mean(tau_return):.2f} μs  σ={np.std(tau_return):.3f} μs")
    print(f"  Δτ = τ_return - τ_launch:")
    print(f"    mean = {mean_dt:.3f} μs  (non-zero → temp offset from reference)")
    print(f"    σ    = {std_dt:.3f} μs  →  {temp_precision:.4f} °C per ping")

    # ── Plot ──────────────────────────────────────────────────────────────────
    t_launch_ms = np.arange(len(T_launch)) / fs * 1000
    t_return_ms = np.arange(len(T_return)) / fs * 1000

    # Example xcorr from the last evaluation period
    y_launch_ex = inc_arr[-1]
    y_return_ex = ref_arr[-1]
    τl_ex, xc_l_ex, env_l_ex = xcorr_lag_us(y_launch_ex, T_launch, fs)
    τr_ex, xc_r_ex, env_r_ex = xcorr_lag_us(y_return_ex, T_return, fs)
    lags_l = (np.arange(len(xc_l_ex)) - (len(T_launch) - 1)) / fs * 1e6
    lags_r = (np.arange(len(xc_r_ex)) - (len(T_return) - 1)) / fs * 1e6

    fig, axes = plt.subplots(3, 1, figsize=(12, 11))

    # Panel 1: Templates
    ax = axes[0]
    ax.plot(t_launch_ms, T_launch, 'b-', lw=1.2, label=f'T_launch  ({n_template} avg)')
    ax2 = ax.twinx()
    ax2.plot(t_return_ms, T_return, 'r-', lw=1.2, label=f'T_return  ({n_template} avg)')
    ax.set_xlabel('Time in window (ms)')
    ax.set_ylabel('T_launch (blue)')
    ax2.set_ylabel('T_return (red)')
    ax.set_title(f'Co-added Templates  —  {os.path.basename(wav_path)}')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel 2: Example cross-correlations
    ax = axes[1]
    zoom = 500  # μs either side of peak
    m_l = (lags_l >= τl_ex - zoom) & (lags_l <= τl_ex + zoom)
    m_r = (lags_r >= τr_ex - zoom) & (lags_r <= τr_ex + zoom)
    ax.plot(lags_l[m_l], env_l_ex[m_l], 'b-', lw=1.5, label=f'xcorr launch  τ={τl_ex:.2f}μs')
    ax.plot(lags_r[m_r], env_r_ex[m_r], 'r-', lw=1.5, label=f'xcorr return  τ={τr_ex:.2f}μs')
    ax.axvline(τl_ex, color='b', ls='--', alpha=0.6)
    ax.axvline(τr_ex, color='r', ls='--', alpha=0.6)
    ax.set_xlabel('Lag (μs)')
    ax.set_ylabel('Normalised xcorr envelope')
    ax.set_title(f'Example Cross-Correlations (period {n_periods-1})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Δτ time series
    ax = axes[2]
    ax.plot(delta_tau, 'ko-', ms=3, lw=1, label='Δτ per period')
    ax.axhline(mean_dt, color='b', ls='--', lw=1.5,
               label=f'mean = {mean_dt:.3f} μs')
    ax.axhline(mean_dt + std_dt, color='gray', ls=':', lw=1)
    ax.axhline(mean_dt - std_dt, color='gray', ls=':',  lw=1,
               label=f'σ = {std_dt:.3f} μs → {temp_precision:.4f}°C/ping')
    ax.set_xlabel('Evaluation period index')
    ax.set_ylabel('Δτ = τ_return − τ_launch (μs)')
    ax.set_title('Differential TOF Time Series')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    stem = os.path.splitext(os.path.basename(wav_path))[0]
    out_path = os.path.join(output_dir, f'stubbs_{stem}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved: {out_path}")

    return dict(
        T_launch=T_launch, T_return=T_return,
        tau_launch=tau_launch, tau_return=tau_return, delta_tau=delta_tau,
        mean_delta_tau_us=mean_dt, std_delta_tau_us=std_dt,
        temp_precision_C=temp_precision, n_template=n_template, n_eval=n_eval,
        fs=fs, incident_win_ms=incident_win_ms, reflection_win_ms=reflection_win_ms,
    )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import glob

    if WAV_FILE is not None:
        wav = WAV_FILE
    else:
        wav_name   = input("WAV filename (in recordings/<subdir>/raw/): ").strip()
        subdir     = input("Recording subdir name: ").strip()
        base = os.path.dirname(os.path.abspath(__file__))
        wav = os.path.join(base, 'recordings', subdir, 'raw', wav_name)

    result = analyze(wav)
    print(f"\nSummary: σ(Δτ) = {result['std_delta_tau_us']:.3f} μs "
          f"→ {result['temp_precision_C']:.4f} °C/ping")
