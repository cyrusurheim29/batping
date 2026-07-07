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

# HP filter to remove 25 kHz lab interference
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


def incident_peak_sample(inc_window):
    """Return the sample index of the Hilbert envelope peak in the incident window."""
    env = np.abs(hilbert(inc_window))
    return int(np.argmax(env))


def align_to_ref(window, peak_idx, ref_idx):
    """
    Shift window so that peak_idx moves to ref_idx.
    Regions that shift outside the window are zero-padded.
    The same shift is applied to both incident and reflected windows,
    so the relative TOF between them is preserved.
    """
    shift = ref_idx - peak_idx
    if shift == 0:
        return window.copy()
    out = np.zeros_like(window)
    if shift > 0:
        out[shift:] = window[:len(window) - shift]
    else:
        out[:len(window) + shift] = window[-shift:]
    return out


def xcorr_lag_us(signal_seg, template, fs, search_radius_us=200.0):
    """
    Cross-correlate signal_seg with template.
    Returns the lag (μs) of the best-matching peak via weighted centroid of
    the Hilbert envelope — robust for broad, flat peaks where 3-point parabolic
    interpolation is noise-dominated.

    search_radius_us: half-width of search window around lag=0.
    Should be ~FWHM/2 so the centroid captures the full peak shape.
    Default 200μs works for typical chirp FWHM of 100-400μs.
    """
    xc = correlate(signal_seg, template, mode='full')
    env = np.abs(hilbert(xc))
    env /= np.max(env) + 1e-12
    center = len(template) - 1          # lag=0 index for mode='full'

    # Search window around lag=0
    radius_samples = int(search_radius_us * fs / 1e6)
    lo = max(0, center - radius_samples)
    hi = min(len(env) - 1, center + radius_samples)
    search_env = env[lo:hi+1]
    lags_samples = np.arange(lo, hi+1, dtype=float)

    # Weighted centroid — uses all information in the peak, not just 3 points.
    # Robust for broad peaks where parabolic fit sees <10% curvature variation.
    weights = search_env ** 2   # square-weight to emphasise the peak
    w_sum = np.sum(weights)
    if w_sum > 1e-12:
        peak_idx = float(np.sum(weights * lags_samples) / w_sum)
    else:
        peak_idx = float(center)

    lag_us = (peak_idx - center) / fs * 1e6
    return lag_us, xc, env


def detect_windows(period_data, fs, win_half_width_ms=WIN_HALF_WIDTH_MS,
                   tof_us=5700.0):
    """
    Detect incident and reflection windows within a single period.

    Strategy:
      1. Find the global envelope maximum → this is always the reflected pulse
         (it has the strongest amplitude, having passed through the piezo twice
         in the far field, compared to the weak near-field direct-path incident).
      2. Search for the incident peak at reflected_time - TOF, within ±2ms.
      3. Set windows of ±win_half_width_ms around each detected center.

    tof_us: approximate expected TOF in μs (default 5700 = 2m bench path).
    """
    from scipy.signal import find_peaks

    env = np.abs(hilbert(period_data))
    env_n = env / np.max(env)
    t_ms = np.arange(len(period_data)) / fs * 1000
    period_ms = len(period_data) / fs * 1000
    tof_ms = tof_us / 1000.0

    # Step 1: reflected = global maximum
    ref_sample = int(np.argmax(env_n))
    ref_center_ms = t_ms[ref_sample]

    # Step 2: incident = peak near (reflected_time - TOF)
    inc_expected_ms = ref_center_ms - tof_ms
    search_lo = max(0.0, inc_expected_ms - 2.0)
    search_hi = min(period_ms, inc_expected_ms + 2.0)
    lo_idx = int(search_lo / 1000 * fs)
    hi_idx = int(search_hi / 1000 * fs)
    if hi_idx > lo_idx:
        inc_sample = lo_idx + int(np.argmax(env_n[lo_idx:hi_idx]))
    else:
        inc_sample = int(ref_sample - tof_ms / 1000 * fs)
    inc_center_ms = t_ms[max(0, min(inc_sample, len(t_ms)-1))]

    hw = win_half_width_ms
    inc_win = (max(0.0, inc_center_ms - hw), min(period_ms, inc_center_ms + hw))
    ref_win = (max(0.0, ref_center_ms - hw), min(period_ms, ref_center_ms + hw))

    print(f"  Detected incident  peak at {inc_center_ms:.2f} ms  "
          f"→ window [{inc_win[0]:.2f}, {inc_win[1]:.2f}] ms")
    print(f"  Detected reflection peak at {ref_center_ms:.2f} ms  "
          f"→ window [{ref_win[0]:.2f}, {ref_win[1]:.2f}] ms  "
          f"(sep={ref_center_ms-inc_center_ms:.1f} ms)")
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

    # ── Coherent sub-sample alignment before co-adding ───────────────────────
    # The SDG and UltraMic run on independent clocks (~100 ppm each), causing
    # ~1 μs/period monotonic drift. Over 100 periods that is ~100 μs = 4 carrier
    # cycles of phase spread, causing destructive cancellation in a naive co-add.
    #
    # Fix: cross-correlate each reflected window with period 0 using parabolic
    # sub-sample peak interpolation to get fractional shift s_i (in samples),
    # then apply s_i via FFT phase shift — exact for bandlimited signals.
    # The same shift is applied to both inc and ref (preserving relative TOF).

    def fft_shift(x, s):
        """Shift signal x by s samples (sub-sample precision) via FFT phase ramp."""
        N = len(x)
        X = np.fft.rfft(x, n=N)
        freqs = np.fft.rfftfreq(N)
        X *= np.exp(-2j * np.pi * freqs * s)
        return np.fft.irfft(X, n=N)

    ref0 = ref_arr[0].copy()
    shifts_frac = np.zeros(n_periods)   # sub-sample shifts in samples
    radius = int(0.002 * fs)            # ±2ms search window
    center_idx = len(ref0) - 1          # lag=0 index in 'full' correlation

    for i in range(1, n_periods):
        xc = np.correlate(ref_arr[i], ref0, mode='full')
        lo = max(0, center_idx - radius)
        hi = min(len(xc), center_idx + radius)
        # MUST use Hilbert envelope — xc oscillates at 40 kHz so argmax(|xc|)
        # picks a random carrier cycle peak, not the true envelope maximum.
        env_xc = np.abs(hilbert(xc[lo:hi]))
        local_peak = np.argmax(env_xc)
        local_max = local_peak + lo
        # Parabolic sub-sample refinement on envelope
        if 0 < local_peak < len(env_xc) - 1:
            ym, y0, yp = env_xc[local_peak-1], env_xc[local_peak], env_xc[local_peak+1]
            denom = 2*(ym - 2*y0 + yp)
            delta = np.clip((ym-yp)/denom, -0.5, 0.5) if abs(denom)>1e-10 else 0.0
            shifts_frac[i] = (local_peak + delta + lo) - center_idx
        else:
            shifts_frac[i] = local_max - center_idx

    drift_us = shifts_frac * 1e6 / fs
    print(f"  Clock drift: max={drift_us.max():.1f} μs  "
          f"σ={drift_us.std():.1f} μs over {n_periods} periods  "
          f"(sub-sample FFT alignment)")

    inc_aligned = np.array([fft_shift(row, -shifts_frac[i]) for i, row in enumerate(inc_arr)])
    ref_aligned = np.array([fft_shift(row, -shifts_frac[i]) for i, row in enumerate(ref_arr)])

    # Build templates from first n_template coherently aligned periods
    T_launch = np.mean(inc_aligned[:n_template], axis=0)
    T_return = np.mean(ref_aligned[:n_template], axis=0)
    print(f"  Templates built from {n_template} aligned periods  "
          f"(incident {incident_win_ms[0]:.2f}–{incident_win_ms[1]:.2f} ms, "
          f"reflection {reflection_win_ms[0]:.2f}–{reflection_win_ms[1]:.2f} ms)")
    print(f"  T_launch peak: {np.max(np.abs(T_launch)):.4f}  "
          f"T_return peak: {np.max(np.abs(T_return)):.4f}")

    # Evaluate: cross-correlate each aligned period's reflected window with T_return.
    # τ_return ≈ residual shift after alignment = temperature-driven TOF change.
    # τ_launch ≈ 0 by construction (alignment was done on reflected, not incident,
    # but both shift together since same clock drives both pulses in the period).
    tau_launch_list = []
    tau_return_list = []
    for i in range(n_template, n_periods):
        τl, _, _ = xcorr_lag_us(inc_aligned[i], T_launch, fs)
        τr, xc_r, env_r = xcorr_lag_us(ref_aligned[i], T_return, fs)
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
