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
    """
    11-point least-squares parabolic fit centered on argmax (±5 samples).
    Returns (peak_idx_float, peak_val, sigma_samples):
      peak_idx_float — refined sub-sample peak index
      peak_val       — fitted peak amplitude
      sigma_samples  — 1-sigma fit uncertainty on peak position (in samples)
    Falls back gracefully if array is too short.
    """
    idx = int(np.argmax(env))
    hw = 5
    lo, hi = max(0, idx - hw), min(len(env), idx + hw + 1)
    x = np.arange(lo, hi, dtype=float) - idx
    y = env[lo:hi]
    if len(y) < 5 or np.ptp(y) < 1e-12 or (lo == 0 and idx < 1):
        return float(idx), float(env[idx]), 0.5
    coeffs, residuals, _, _, _ = np.polyfit(x, y, 2, full=True)
    a, b, _ = coeffs
    if abs(a) < 1e-10:
        return float(idx), float(env[idx]), 0.5
    delta = float(np.clip(-b / (2 * a), -hw, hw))
    peak_val = float(np.polyval(coeffs, delta))
    rms = float(np.sqrt(residuals[0] / len(y))) if len(residuals) > 0 else 0.1
    sigma = rms / (abs(a) * float(np.sqrt(np.sum(x ** 2))))
    return float(idx) + delta, peak_val, sigma


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
    Returns (lag_us, xc, env, sigma_us):
      lag_us    — peak lag in μs via 11-point parabolic fit on Hilbert envelope
      xc        — raw cross-correlation array
      env       — normalised Hilbert envelope of xc
      sigma_us  — 1-sigma fit uncertainty on lag_us (μs)

    search_radius_us: half-width of search window around lag=0.
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

    # 11-point parabolic fit on the envelope peak
    idx = int(np.argmax(search_env))
    hw = 5
    sl, sh = max(0, idx - hw), min(len(search_env), idx + hw + 1)
    x = np.arange(sl, sh, dtype=float) - idx
    y = search_env[sl:sh]
    step_us = 1e6 / fs   # μs per sample
    if len(y) >= 5 and np.ptp(y) > 1e-12:
        coeffs, residuals, _, _, _ = np.polyfit(x, y, 2, full=True)
        a, b = coeffs[0], coeffs[1]
        if abs(a) > 1e-10:
            dl = float(np.clip(-b / (2 * a), -hw, hw))
            peak_idx = float(lo + idx) + dl
            rms = float(np.sqrt(residuals[0] / len(y))) if len(residuals) > 0 else step_us * 0.1
            sigma_us = rms / (abs(a) * float(np.sqrt(np.sum(x ** 2)))) * step_us
        else:
            peak_idx = float(lo + idx); sigma_us = step_us
    else:
        peak_idx = float(lo + idx); sigma_us = step_us

    lag_us = (peak_idx - center) / fs * 1e6
    return lag_us, xc, env, sigma_us


def detect_windows(period_data, fs, win_half_width_ms=WIN_HALF_WIDTH_MS,
                   tof_us=5700.0):
    """
    Detect incident and reflection windows within a single period.

    Strategy (robust to room echoes):
      1. Incident = highest peak in the early part of the period
         (first tof_ms from period start) — the direct-path pulse always
         arrives before any round-trip reflection can.
      2. Reflected = highest peak within [incident + tof_ms ± 2ms].
         This prevents room echoes (which arrive later) from being mistaken
         for the direct reflection.
      3. Set windows of ±win_half_width_ms around each detected center.

    tof_us: approximate expected TOF in μs (default 5700 = 2m bench path).
    """
    env = np.abs(hilbert(period_data))
    env_n = env / np.max(env)
    t_ms = np.arange(len(period_data)) / fs * 1000
    period_ms = len(period_data) / fs * 1000
    tof_ms = tof_us / 1000.0
    tol_ms = 2.0   # ±2ms tolerance around expected TOF

    # Step 1: incident = first significant peak in [0, 0.6*tof_ms].
    # Gate is 60% of expected TOF so the reflected burst (arriving at ~TOF)
    # cannot bleed into the search window even for long chirps.
    # Use first-onset detection rather than argmax so a faint incident is
    # preferred over a strong reflected that falls near the gate edge.
    inc_gate_hi  = min(period_ms, 0.6 * tof_ms)
    inc_hi_idx   = max(1, int(inc_gate_hi / 1000 * fs))
    # Find first sample above 5% of global max → marks incident onset
    threshold    = 0.05
    above        = env_n[:inc_hi_idx] > threshold
    if np.any(above):
        onset = int(np.argmax(above))
        # Local max within 5ms of onset
        search_end = min(inc_hi_idx, onset + int(0.005 * fs))
        inc_sample = onset + int(np.argmax(env_n[onset:search_end]))
    else:
        inc_sample = int(np.argmax(env_n[:inc_hi_idx]))
    inc_center_ms = float(t_ms[min(inc_sample, len(t_ms) - 1)])

    # Step 2: reflected = highest peak in [inc + tof - tol, inc + tof + tol]
    ref_expected_ms = inc_center_ms + tof_ms
    ref_lo = max(0.0, ref_expected_ms - tol_ms)
    ref_hi = min(period_ms, ref_expected_ms + tol_ms)
    lo_idx = int(ref_lo / 1000 * fs)
    hi_idx = max(lo_idx + 1, int(ref_hi / 1000 * fs))
    hi_idx = min(hi_idx, len(env_n))
    if hi_idx > lo_idx and np.any(env_n[lo_idx:hi_idx] > 0):
        ref_sample = lo_idx + int(np.argmax(env_n[lo_idx:hi_idx]))
    else:
        ref_sample = int(inc_sample + tof_ms / 1000 * fs)
    ref_center_ms = float(t_ms[min(ref_sample, len(t_ms) - 1)])

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
    period_idx = round(fs / rep_rate)   # round not int — minimises per-period quantisation error
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


def mask_period(period_data, fs, win_ms):
    """Return a full-period copy with everything outside win_ms zeroed.

    This preserves absolute timing: both incident and reflected masks live in the
    same period-length time frame, so xcorr with the co-added template gives a peak
    at lag=0 when perfectly aligned — making tau_launch a clean diagnostic of
    unaccounted jitter rather than an arbitrary offset.
    """
    out = np.zeros_like(period_data)
    s = int(win_ms[0] / 1000 * fs)
    e = int(win_ms[1] / 1000 * fs)
    s = max(0, min(s, len(period_data) - 1))
    e = max(s + 1, min(e, len(period_data)))
    out[s:e] = period_data[s:e]
    return out


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

    # Auto-detect windows from first period
    if incident_win_ms is None or reflection_win_ms is None:
        first_period = sig[slices[0][0]:slices[0][1]]
        incident_win_ms, reflection_win_ms = detect_windows(
            first_period, fs, win_half_width_ms)

    # ── Coherent alignment before co-adding — full-period approach ───────────
    #
    # We follow Stubbs's recommendation: keep incident and reflected as two separate
    # but TIME-SYNCED vectors living in the same full-period frame.  Each period is
    # drift-corrected (LINEAR shift so the burst stays at the period-0 position, then
    # sub-sample FFT shift for the fractional residual), then the relevant window is
    # masked to zero outside it.  Co-adding these full-period masked vectors builds
    # T_launch and T_return.  xcorr of any evaluation period against T_launch/T_return
    # then peaks at lag=0 when alignment is perfect → tau_launch is a clean jitter
    # diagnostic (non-zero tau_launch = unaccounted timing noise).
    #
    # KEY IMPLEMENTATION DETAIL: the FFT-based shift is a CIRCULAR shift, which
    # wraps the burst to the wrong place when the total drift is large (>half the
    # period).  We use a LINEAR shift (zero-padding at boundaries) for the integer
    # part, reserving FFT shift only for the sub-sample residual (always <0.5 samples).
    #
    # Alignment is computed from FULL-PERIOD cross-correlations (not short windows),
    # so the burst is always present in the xcorr regardless of how much it has
    # drifted within the period.  scipy.signal.correlate uses FFT internally so this
    # is fast even for period_idx ~ 7000 samples.

    period_idx = slices[0][1] - slices[0][0]  # all slices are the same length

    def fft_shift(x, s):
        """Sub-sample shift of x by s samples via FFT phase ramp (circular — small s only)."""
        N = len(x)
        X = np.fft.rfft(x, n=N)
        X *= np.exp(-2j * np.pi * np.fft.rfftfreq(N) * s)
        return np.fft.irfft(X, n=N)

    def linear_shift(x, s):
        """Integer-sample LINEAR shift (zero-padding, no wrap). Positive s = shift right."""
        s = int(round(s))
        n = len(x)
        out = np.zeros(n, dtype=x.dtype)
        if s > 0 and s < n:
            out[s:] = x[:n - s]
        elif s < 0 and -s < n:
            out[:n + s] = x[-s:]
        else:
            out[:] = x
        return out

    def apply_correction(period, shift_s):
        """Drift-correct a full period: linear shift for integer part, fft_shift for fraction."""
        s_int  = int(round(shift_s))
        s_frac = shift_s - s_int
        out = linear_shift(period, -s_int)          # move burst left to period-0 position
        if abs(s_frac) > 1e-6:
            out = fft_shift(out, -s_frac)           # sub-sample residual (always <0.5 samples)
        return out

    # Reference = full period 0 (unmasked) — burst is always present for xcorr
    ref0_full = sig[slices[0][0]:slices[0][1]].copy()
    center_idx = period_idx - 1          # lag=0 index in mode='full' correlation
    shifts_frac = np.zeros(n_periods)

    radius = int(0.002 * fs)             # ±2 ms bootstrap search window
    N_BOOT = 50
    narrow = int(0.0003 * fs)            # ±300 μs adaptive search window

    def find_shift(xc, lo, hi):
        env = np.abs(hilbert(xc[lo:hi]))
        lp  = int(np.argmax(env))
        hw = 5
        sl, sh = max(0, lp - hw), min(len(env), lp + hw + 1)
        x = np.arange(sl, sh, dtype=float) - lp
        y = env[sl:sh]
        if len(y) >= 5 and np.ptp(y) > 1e-12:
            coeffs = np.polyfit(x, y, 2)
            a, b = coeffs[0], coeffs[1]
            if abs(a) > 1e-10:
                dl = float(np.clip(-b / (2 * a), -hw, hw))
                return (lp + dl + lo) - center_idx
        return float(lp + lo - center_idx)

    # Phase 1: bootstrap — xcorr full periods (burst always in frame)
    boot_end = min(N_BOOT + 1, n_periods)
    for i in range(1, boot_end):
        period_i = sig[slices[i][0]:slices[i][1]]
        xc = correlate(period_i, ref0_full, mode='full')
        lo = max(0, center_idx - radius)
        hi = min(len(xc), center_idx + radius)
        shifts_frac[i] = find_shift(xc, lo, hi)

    boot_idx  = np.arange(1, boot_end, dtype=float)
    drift_rate = np.polyfit(boot_idx, shifts_frac[1:boot_end], 1)[0]

    # Phase 2: adaptive EMA — still xcorr full periods, narrow search window
    EMA_TAU  = 200.0
    ema_rate = drift_rate
    alpha_ema = 1.0 / EMA_TAU

    for i in range(boot_end, n_periods):
        predicted   = ema_rate * i
        pred_center = center_idx + int(round(predicted))
        period_i    = sig[slices[i][0]:slices[i][1]]
        xc = correlate(period_i, ref0_full, mode='full')
        lo = max(0, pred_center - narrow)
        hi = min(len(xc), pred_center + narrow)
        shifts_frac[i] = find_shift(xc, lo, hi)
        inst_rate = shifts_frac[i] / i
        ema_rate  = (1 - alpha_ema) * ema_rate + alpha_ema * inst_rate

    drift_us = shifts_frac * 1e6 / fs
    print(f"  Clock drift: rate={ema_rate*1e6/fs:.3f} μs/period  "
          f"total={drift_us[-1]:.1f} μs  "
          f"(adaptive + EMA rate update τ={EMA_TAU:.0f}p)")

    # Build templates: drift-correct each period (linear+fft), then mask
    T_launch_acc = np.zeros(period_idx)
    T_return_acc = np.zeros(period_idx)
    for i in range(n_template):
        corrected = apply_correction(sig[slices[i][0]:slices[i][1]], shifts_frac[i])
        T_launch_acc += mask_period(corrected, fs, incident_win_ms)
        T_return_acc += mask_period(corrected, fs, reflection_win_ms)
    T_launch = T_launch_acc / n_template
    T_return = T_return_acc / n_template
    print(f"  Templates built from {n_template} aligned periods  "
          f"(incident {incident_win_ms[0]:.2f}–{incident_win_ms[1]:.2f} ms, "
          f"reflection {reflection_win_ms[0]:.2f}–{reflection_win_ms[1]:.2f} ms)")
    print(f"  T_launch peak: {np.max(np.abs(T_launch)):.4f}  "
          f"T_return peak: {np.max(np.abs(T_return)):.4f}")

    # Evaluate: drift-correct each evaluation period, mask, xcorr with templates.
    # tau_launch should be near 0 if alignment is good — it is the jitter diagnostic.
    # tau_return - tau_launch = differential TOF = temperature signal.
    tau_launch_list = []
    tau_return_list = []
    sigma_launch_list = []
    sigma_return_list = []
    for i in range(n_template, n_periods):
        corrected = apply_correction(sig[slices[i][0]:slices[i][1]], shifts_frac[i])
        inc_m = mask_period(corrected, fs, incident_win_ms)
        ref_m = mask_period(corrected, fs, reflection_win_ms)
        τl, _, _, σl = xcorr_lag_us(inc_m, T_launch, fs)
        τr, xc_r, env_r, σr = xcorr_lag_us(ref_m, T_return, fs)
        tau_launch_list.append(τl)
        tau_return_list.append(τr)
        sigma_launch_list.append(σl)
        sigma_return_list.append(σr)

    tau_launch = np.array(tau_launch_list)
    tau_return = np.array(tau_return_list)
    delta_tau  = tau_return - tau_launch

    n_eval = len(delta_tau)
    mean_dt = np.mean(delta_tau)
    std_dt  = np.std(delta_tau)
    temp_precision = std_dt / TOF_SENS_US_PER_DEG

    print(f"\n  Evaluation periods: {n_eval}")
    mean_fit_sigma = float(np.mean(sigma_launch_list + sigma_return_list)) if sigma_launch_list else None
    print(f"  τ_launch:  mean={np.mean(tau_launch):.2f} μs  σ={np.std(tau_launch):.3f} μs")
    print(f"  τ_return:  mean={np.mean(tau_return):.2f} μs  σ={np.std(tau_return):.3f} μs")
    print(f"  Δτ = τ_return - τ_launch:")
    print(f"    mean = {mean_dt:.3f} μs  (non-zero → temp offset from reference)")
    print(f"    σ    = {std_dt:.3f} μs  →  {temp_precision:.4f} °C per ping")
    if mean_fit_sigma is not None:
        print(f"    fit σ_peak = {mean_fit_sigma*1000:.0f} ns  (11-pt parabola residual)")

    # ── Plot ──────────────────────────────────────────────────────────────────
    t_launch_ms = np.arange(len(T_launch)) / fs * 1000
    t_return_ms = np.arange(len(T_return)) / fs * 1000

    # Example xcorr from the last evaluation period
    last_corrected = apply_correction(sig[slices[-1][0]:slices[-1][1]], shifts_frac[-1])
    y_launch_ex = mask_period(last_corrected, fs, incident_win_ms)
    y_return_ex = mask_period(last_corrected, fs, reflection_win_ms)
    τl_ex, xc_l_ex, env_l_ex, _ = xcorr_lag_us(y_launch_ex, T_launch, fs)
    τr_ex, xc_r_ex, env_r_ex, _ = xcorr_lag_us(y_return_ex, T_return, fs)
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
        mean_fit_sigma_us=mean_fit_sigma,
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
