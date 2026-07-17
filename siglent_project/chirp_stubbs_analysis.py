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


def xcorr_lag_shape_fit(signal_seg, template, fs,
                        R_auto=None, window_half_us=200.0,
                        search_radius_us=200.0):
    """
    Estimate lag by fitting the measured CCF to the template autocorrelation shape.

    Uses the full oscillating CCF peak — not just its argmax — as a vernier ruler.

    Model (linearised Taylor expansion around the coarse integer peak):
        R(τ_k) ≈ A · R_auto(τ_k) + B · R_auto'(τ_k)
    where B/A = −Δτ  (sub-sample correction, in samples).

    Both the carrier oscillations (phase information) and the envelope curvature
    (amplitude information) over the entire ±window_half_us window constrain the
    2-parameter fit simultaneously — far more than the 11-pt parabola on the envelope.

    Uncertainty is the LS covariance propagated through the delta method for Δτ = −B/A.

    Parameters
    ----------
    signal_seg      : masked evaluation period (same length as template)
    template        : co-added template (T_launch or T_return)
    fs              : sample rate (Hz)
    R_auto          : precomputed correlate(template, template, 'full').
                      Pass from analyze() to avoid recomputing each period.
                      Computed internally if None (slow — only for standalone use).
    window_half_us  : half-width of fit window around coarse peak (μs). Default 200 μs
                      covers the main lobe for typical 4–6 kHz effective bandwidth.
    search_radius_us: half-width of coarse-peak search window (μs).

    Returns
    -------
    lag_us    : lag estimate (μs)
    sigma_us  : 1-sigma uncertainty from LS covariance, delta method (μs)
    A         : fitted peak amplitude (SNR proxy)
    rms_resid : RMS fit residual normalised by |A| (goodness-of-fit)
    """
    xc = correlate(signal_seg, template, mode='full')
    if R_auto is None:
        R_auto = correlate(template, template, mode='full')

    center = len(template) - 1          # lag = 0 index (same for xc and R_auto)

    # Coarse peak: argmax of Hilbert envelope in search window
    env    = np.abs(hilbert(xc))
    radius = int(search_radius_us * fs / 1e6)
    lo     = max(0, center - radius)
    hi     = min(len(xc) - 1, center + radius)
    coarse = lo + int(np.argmax(env[lo:hi + 1]))

    # Window of integer sample offsets around coarse peak
    W      = max(int(window_half_us * fs / 1e6), 5)
    k_all  = np.arange(-W, W + 1, dtype=int)

    # Discard offsets that fall outside either array (leave 1-sample margin for deriv)
    valid  = (
        (coarse + k_all >= 0) & (coarse + k_all < len(xc)) &
        (center + k_all >= 1) & (center + k_all < len(R_auto) - 1)
    )
    k = k_all[valid]
    if len(k) < 5:
        lag_us_fb, _, _, sigma_us_fb = xcorr_lag_us(signal_seg, template, fs,
                                                     search_radius_us)
        return lag_us_fb, sigma_us_fb, 1.0, np.nan

    xc_y = xc[coarse + k].astype(float)
    ra   = R_auto[center + k].astype(float)
    # Central-difference derivative of R_auto at each window sample
    ra_d = (R_auto[center + k + 1] - R_auto[center + k - 1]).astype(float) / 2.0

    # Weight by envelope of R_auto: upweights peak region (high SNR),
    # downweights sidelobes (low SNR) without penalising carrier zero-crossings
    env_ra = np.abs(hilbert(ra))
    w      = env_ra / (np.max(env_ra) + 1e-12)

    Phi   = np.column_stack([ra, ra_d])
    Phi_w = Phi * w[:, None]
    y_w   = xc_y * w

    AB, _, _, _ = np.linalg.lstsq(Phi_w, y_w, rcond=None)
    A, B = float(AB[0]), float(AB[1])

    if abs(A) < 1e-10:
        return float(coarse - center) / fs * 1e6, 1e6 / fs, 0.0, np.nan

    delta_samp = -B / A                              # fractional sample correction
    lag_us = (coarse - center + delta_samp) / fs * 1e6

    # Uncertainty via delta method on unweighted LS covariance
    # (unweighted residuals give the true noise model for the covariance)
    residuals    = xc_y - Phi @ AB
    K            = len(xc_y)
    sigma2_noise = float(np.dot(residuals, residuals)) / max(1, K - 2)
    try:
        cov = sigma2_noise * np.linalg.inv(Phi.T @ Phi)
    except np.linalg.LinAlgError:
        cov = sigma2_noise * np.linalg.pinv(Phi.T @ Phi)

    sA2 = cov[0, 0];  sB2 = cov[1, 1];  cAB = cov[0, 1]
    var_delta = (sB2 + delta_samp**2 * sA2 - 2 * delta_samp * cAB) / (A ** 2)
    sigma_us  = float(np.sqrt(max(0.0, var_delta))) * 1e6 / fs

    rms_resid = float(np.sqrt(sigma2_noise)) / (abs(A) + 1e-12)
    return lag_us, sigma_us, float(A), rms_resid


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

    # Robust two-peak detection — works regardless of which peak is stronger.
    #
    # Strategy: find the two best peaks anywhere in the period whose separation
    # is close to tof_ms (within ±tol_ms).  The earlier peak = incident,
    # the later peak = reflected.  This handles:
    #   • New mic positions where incident ≈ reflected amplitude
    #   • Cases where T0 places the incident late in the period frame
    #   • Room echoes (which don't pair with anything at TOF separation)
    #
    # Implementation:
    #   1. Find global maximum → candidate_A
    #   2. Search for candidate_B at candidate_A ± tof_ms (±tol_ms window)
    #      Pick the strongest candidate_B in either direction.
    #   3. If no good candidate_B found, fall back to the first-onset + TOF approach.

    # Step 1: global max as anchor — restricted to interior of period so
    # bursts cut off at the edges ([0,2ms] and [period-2ms, period]) are ignored.
    edge_ms  = 2.0
    lo_edge  = int(edge_ms / 1000 * fs)
    hi_edge  = max(lo_edge + 1, len(env_n) - int(edge_ms / 1000 * fs))
    interior = env_n[lo_edge:hi_edge]
    anchor_sample = lo_edge + int(np.argmax(interior))
    anchor_ms     = float(t_ms[anchor_sample])

    # Step 2: search for pairing peak at anchor ± tof_ms
    def _search_peak(center_ms):
        lo = max(0.0, center_ms - tol_ms)
        hi = min(period_ms, center_ms + tol_ms)
        if lo >= hi or lo >= period_ms or hi <= 0:
            return None, 0.0
        li = int(lo / 1000 * fs)
        hi_i = min(int(hi / 1000 * fs), len(env_n))
        hi_i = max(li + 1, hi_i)
        if li >= len(env_n):
            return None, 0.0
        hi_i = min(hi_i, len(env_n))
        if hi_i <= li:
            return None, 0.0
        idx = li + int(np.argmax(env_n[li:hi_i]))
        return idx, float(env_n[idx])

    # Candidate: reflected is AFTER anchor
    cand_ref_idx, cand_ref_val = _search_peak(anchor_ms + tof_ms)
    # Candidate: incident is BEFORE anchor (anchor is the reflected)
    cand_inc_idx, cand_inc_val = _search_peak(anchor_ms - tof_ms)

    # Amplitude guard (user fix): the mic is positioned for equal incident /
    # reflected pickup, so the pairing peak must be ≥ MIN_PAIR_FRAC × anchor.
    # Without this, a multipath at 5% of the anchor amplitude can "win" the
    # ±TOF search, swapping incident↔reflected assignment and making the xcorr
    # measure (multipath, real_incident) instead of (incident, reflected).
    MIN_PAIR_FRAC = 0.40
    anchor_val    = float(env_n[anchor_sample])
    amp_threshold = MIN_PAIR_FRAC * anchor_val

    cand_ref_ok = cand_ref_idx is not None and cand_ref_val >= amp_threshold
    cand_inc_ok = cand_inc_idx is not None and cand_inc_val >= amp_threshold

    if cand_ref_ok and (cand_ref_val >= cand_inc_val or not cand_inc_ok):
        # anchor = incident, cand_ref = reflected
        inc_sample = anchor_sample
        ref_sample = cand_ref_idx
    elif cand_inc_ok:
        # anchor = reflected, cand_inc = incident
        inc_sample = cand_inc_idx
        ref_sample = anchor_sample
    else:
        # No valid pairing found above amplitude threshold — fall back to
        # onset-based search.  Log when the threshold rejected a candidate.
        if cand_ref_idx is not None and cand_ref_val > 0 and not cand_ref_ok:
            print(f"  FM3 amp guard: fwd candidate {cand_ref_val:.3f} < "
                  f"{amp_threshold:.3f} ({MIN_PAIR_FRAC:.0%} × {anchor_val:.3f}) — rejected")
        if cand_inc_idx is not None and cand_inc_val > 0 and not cand_inc_ok:
            print(f"  FM3 amp guard: bwd candidate {cand_inc_val:.3f} < "
                  f"{amp_threshold:.3f} ({MIN_PAIR_FRAC:.0%} × {anchor_val:.3f}) — rejected")
        threshold = 0.05
        gate_hi   = max(1, int(min(period_ms, 0.6 * tof_ms) / 1000 * fs))
        above     = env_n[:gate_hi] > threshold
        if np.any(above):
            onset = int(np.argmax(above))
            inc_sample = onset + int(np.argmax(env_n[onset:min(gate_hi, onset+int(0.005*fs))]))
        else:
            inc_sample = 0
        ref_sample = min(len(env_n)-1, inc_sample + int(tof_ms / 1000 * fs))

    inc_center_ms = float(t_ms[min(inc_sample, len(t_ms) - 1)])
    ref_center_ms = float(t_ms[min(ref_sample, len(t_ms) - 1)])

    hw = win_half_width_ms
    inc_win = (max(0.0, inc_center_ms - hw), min(period_ms, inc_center_ms + hw))
    ref_win = (max(0.0, ref_center_ms - hw), min(period_ms, ref_center_ms + hw))

    # FM3b fix: enforce minimum gap between incident window end and return window start.
    # ring_down_margin ≈ 3×Q/(π×f₀) ≈ 3×0.18 ms = 0.55 ms.  Without this, the return
    # window can start inside the incident burst's ring-down, contaminating T_return
    # with T_launch energy and causing both xcorr peaks to land at the same wrong lag.
    ring_down_margin_ms = 0.3   # reduced from 0.6: 0.6 was clipping reflected burst
                                # onset for tight-geometry waveforms (e.g. bw_20kHz).
                                # 0.3 ms still clears the Q/(π×f₀)≈0.18 ms ring-down.
    min_ref_start = inc_win[1] + ring_down_margin_ms
    if ref_win[0] < min_ref_start:
        ref_win = (min(min_ref_start, ref_center_ms), ref_win[1])
        print(f"  FM3b: return window start advanced to {ref_win[0]:.2f} ms "
              f"(ring-down separation enforced)")

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


def adaptive_slice_periods(data, fs, rep_rate, tof_us=5700.0,
                           pre_inc_ms=2.0, amp_threshold=0.15):
    """
    FM3a note: pre_inc_ms kept at 2.0 ms (original value).
    Increasing it made incident window wider → FM3b guard clipped reflected
    burst onset for some good waveforms → σ regressions.
    The drift-guard fix (zeroing false shifts_frac) is the main improvement;
    onset clipping for a0.6_lg_4ms is a secondary issue addressed separately.

    Adaptive period slicer: finds actual incident peaks in the continuous
    recording rather than using fixed uniform cuts.

    Each 17ms window is anchored to the detected incident peak, so the
    incident is always at ~pre_inc_ms into the window and the reflected at
    ~pre_inc_ms + TOF — regardless of SDG/UltraMic clock drift.  This
    eliminates the period-boundary drift problem entirely.

    Strategy:
      1. Find all large envelope peaks separated by approximately one SDG
         period (= 1/rep_rate).  Both incident and reflected peaks qualify.
      2. Pair consecutive peaks that are ~TOF apart → identify each pair.
      3. Centre a window of length period_samples on each incident peak.

    Returns list of (start_idx, end_idx) tuples, same interface as slice_periods.
    """
    from scipy.signal import find_peaks

    period_samples = round(fs / rep_rate)
    window_samples = period_samples          # output window length = 1 period
    tof_samples    = int(tof_us / 1e6 * fs)
    pre_samples    = int(pre_inc_ms / 1000 * fs)
    tol_samples    = int(0.002 * fs)         # ±2ms tolerance on TOF matching

    env   = np.abs(hilbert(data))
    env_n = env / np.max(env)

    # Minimum spacing between detected peaks.
    # When TOF < period/3 (e.g. 4ms chirp at 52 Hz: period/3=6.3ms, TOF=5.7ms),
    # period//3 merges incident and reflected into one peak. Cap min_dist at
    # 80% of expected TOF so both bursts are always resolvable.
    min_dist = max(1, min(period_samples // 3, int(tof_samples * 0.8)))
    peaks, props = find_peaks(env_n, height=amp_threshold, distance=min_dist)

    # Pair peaks into (incident, reflected) by TOF separation + amplitude guard.
    #
    # Amplitude guard: mic is positioned for roughly equal incident/reflected
    # pickup, so the paired reflection must be within MIN_PAIR_FRAC–(1/MIN_PAIR_FRAC)
    # of the candidate incident amplitude.
    #
    # Without this, a multipath at 5–20% of the main burst amplitude can form
    # a "valid" pair with the real incident burst (which arrives ~TOF later and
    # is 4–20× stronger), swapping the incident↔reflected assignment and making
    # the xcorr measure (multipath, real_incident) instead of (incident, reflected).
    # This was the root cause of the spurious ±50 000 μs outliers seen across ~20
    # of the 45 ECM999 waveforms.
    #
    # When the (multipath, real_incident) pair is rejected, the loop finds
    # (real_incident, real_reflected) on the next iteration.
    MIN_PAIR_FRAC = 0.40   # reflected must be ≥ 40% of incident amplitude
    MAX_PAIR_FRAC = 1.0 / MIN_PAIR_FRAC   # and ≤ 2.5× (catches swapped pairs)

    incident_samples = []
    used = set()
    for i, p in enumerate(peaks):
        if i in used:
            continue
        # Look for a partner ~TOF after this peak
        for j in range(i + 1, len(peaks)):
            sep = peaks[j] - p
            if sep > tof_samples + tol_samples:
                break
            if abs(sep - tof_samples) <= tol_samples:
                amp_p = float(env_n[p])
                amp_j = float(env_n[peaks[j]])
                ratio = amp_j / (amp_p + 1e-12)
                if ratio < MIN_PAIR_FRAC or ratio > MAX_PAIR_FRAC:
                    # Amplitude mismatch — likely multipath or swapped assignment.
                    # Skip this candidate; the loop will try peaks[j] as anchor
                    # on the next outer iteration.
                    continue
                # Valid pair: p=incident, peaks[j]=reflected
                incident_samples.append(p)
                used.add(i); used.add(j)
                break

    if not incident_samples:
        print("  adaptive_slice_periods: no incident-reflected pairs found, "
              "falling back to slice_periods")
        return slice_periods(data, fs, rep_rate)

    # Build windows anchored pre_inc_ms before each incident peak
    slices = []
    for inc in incident_samples:
        start = inc - pre_samples
        end   = start + window_samples
        if start >= 0 and end <= len(data):
            slices.append((start, end))

    print(f"  adaptive_slice_periods: {len(slices)} windows from "
          f"{len(incident_samples)} detected incident peaks")
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
            win_half_width_ms=WIN_HALF_WIDTH_MS, output_dir=OUTPUT_DIR,
            save_plot=True):
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
    slices = adaptive_slice_periods(sig, fs, rep_rate)
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

    # FM3 drift fix: build a windowed reference that is zero outside the incident
    # burst window.  Full-period xcorr is dominated by broadband noise for
    # low-SNR waveforms (e.g. ECM999 a0.6_lg_4ms: T_launch~0.09), giving a
    # false drift of ±6 μs/period.  Restricting to the burst window ensures
    # SNR ≫ 1 regardless of waveform amplitude.
    iw_s_d = int(incident_win_ms[0] / 1000 * fs)
    iw_e_d = min(int(incident_win_ms[1] / 1000 * fs), period_idx)
    ref0_drift = np.zeros(period_idx)
    ref0_drift[iw_s_d:iw_e_d] = ref0_full[iw_s_d:iw_e_d]

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

    # Phase 1: bootstrap — windowed xcorr (burst region only)
    boot_end = min(N_BOOT + 1, n_periods)
    for i in range(1, boot_end):
        period_i = sig[slices[i][0]:slices[i][1]]
        xc = correlate(period_i, ref0_drift, mode='full')
        lo = max(0, center_idx - radius)
        hi = min(len(xc), center_idx + radius)
        shifts_frac[i] = find_shift(xc, lo, hi)

    boot_idx  = np.arange(1, boot_end, dtype=float)
    drift_rate = np.polyfit(boot_idx, shifts_frac[1:boot_end], 1)[0]

    # Phase 2: adaptive EMA — windowed xcorr, narrow search window
    EMA_TAU  = 200.0
    ema_rate = drift_rate
    alpha_ema = 1.0 / EMA_TAU

    for i in range(boot_end, n_periods):
        predicted   = ema_rate * i
        pred_center = center_idx + int(round(predicted))
        period_i    = sig[slices[i][0]:slices[i][1]]
        xc = correlate(period_i, ref0_drift, mode='full')
        lo = max(0, pred_center - narrow)
        hi = min(len(xc), pred_center + narrow)
        if lo >= hi:                       # pred_center out of bounds — fallback
            lo = max(0, center_idx - narrow)
            hi = min(len(xc), center_idx + narrow)
        if lo >= hi:                       # xcorr too short — skip
            shifts_frac[i] = shifts_frac[i - 1] if i > 0 else 0.0
            continue
        shifts_frac[i] = find_shift(xc, lo, hi)
        inst_rate = shifts_frac[i] / i
        ema_rate  = (1 - alpha_ema) * ema_rate + alpha_ema * inst_rate

    drift_us = shifts_frac * 1e6 / fs
    print(f"  Clock drift: rate={ema_rate*1e6/fs:.3f} μs/period  "
          f"total={drift_us[-1]:.1f} μs  "
          f"(adaptive + EMA rate update τ={EMA_TAU:.0f}p)")

    # FM3 drift guard: adaptive_slice_periods anchors every period to its
    # detected incident peak, so true residual drift should be tiny
    # (< ~0.5 μs/period from genuine SDG/PicoScope clock mismatch).
    # Large EMA rates (±3–7 μs/period seen for ECM999 a0.6 waveforms) are
    # detection noise.  Applying them over 1000+ periods accumulates to ±5 ms,
    # displacing bursts outside windows and biasing templates vs eval periods.
    # Threshold: if |rate| > 0.5 μs/period, zero shifts_frac entirely.
    # Genuine small drifts (UltraMic, ECM999 good waveforms) are ≤ 0.1 μs/period
    # and are kept.
    _DRIFT_THRESHOLD_US_PER_PERIOD = 0.5
    drift_rate_us = ema_rate * 1e6 / fs
    if abs(drift_rate_us) > _DRIFT_THRESHOLD_US_PER_PERIOD:
        shifts_frac_max_us = float(np.max(np.abs(shifts_frac))) * 1e6 / fs
        shifts_frac[:] = 0.0
        print(f"  FM3 drift zeroed: EMA rate {drift_rate_us:.2f} μs/period > "
              f"{_DRIFT_THRESHOLD_US_PER_PERIOD} threshold (max was "
              f"{shifts_frac_max_us:.0f} μs — detection noise, slicer handles alignment)")
    else:
        print(f"  FM3 drift kept: EMA rate {drift_rate_us:.3f} μs/period "
              f"≤ {_DRIFT_THRESHOLD_US_PER_PERIOD} threshold (genuine small drift)")

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

    # Precompute template autocorrelations for the shape-fit estimator (done once)
    R_auto_L = correlate(T_launch, T_launch, mode='full')
    R_auto_R = correlate(T_return, T_return, mode='full')

    # Evaluate: drift-correct each evaluation period, mask, xcorr with templates.
    # tau_launch should be near 0 if alignment is good — it is the jitter diagnostic.
    # tau_return - tau_launch = differential TOF = temperature signal.
    tau_launch_list    = [];  tau_return_list    = []
    sigma_launch_list  = [];  sigma_return_list  = []
    tau_launch_sf_list = [];  tau_return_sf_list = []
    sigma_launch_sf_list=[];  sigma_return_sf_list=[]
    for i in range(n_template, n_periods):
        corrected = apply_correction(sig[slices[i][0]:slices[i][1]], shifts_frac[i])
        inc_m = mask_period(corrected, fs, incident_win_ms)
        ref_m = mask_period(corrected, fs, reflection_win_ms)
        τl,  _, _, σl  = xcorr_lag_us(inc_m, T_launch, fs)
        τr, xc_r, env_r, σr = xcorr_lag_us(ref_m, T_return, fs)
        τl_sf, σl_sf, _, _ = xcorr_lag_shape_fit(inc_m, T_launch, fs, R_auto=R_auto_L)
        τr_sf, σr_sf, _, _ = xcorr_lag_shape_fit(ref_m, T_return, fs, R_auto=R_auto_R)
        tau_launch_list.append(τl);     tau_return_list.append(τr)
        sigma_launch_list.append(σl);   sigma_return_list.append(σr)
        tau_launch_sf_list.append(τl_sf); tau_return_sf_list.append(τr_sf)
        sigma_launch_sf_list.append(σl_sf); sigma_return_sf_list.append(σr_sf)

    tau_launch = np.array(tau_launch_list)
    tau_return = np.array(tau_return_list)
    delta_tau  = tau_return - tau_launch

    tau_launch_sf = np.array(tau_launch_sf_list)
    tau_return_sf = np.array(tau_return_sf_list)
    delta_tau_sf  = tau_return_sf - tau_launch_sf

    n_eval         = len(delta_tau)
    mean_dt        = float(np.mean(delta_tau))
    std_dt         = float(np.std(delta_tau))
    temp_precision = std_dt / TOF_SENS_US_PER_DEG
    mean_dt_sf     = float(np.mean(delta_tau_sf))
    std_dt_sf      = float(np.std(delta_tau_sf))
    temp_prec_sf   = std_dt_sf / TOF_SENS_US_PER_DEG

    # Per-period δ(Δτ) = √(δ_L² + δ_R²) from LS covariance — the true estimator uncertainty
    sigma_dt_para = np.sqrt(np.array(sigma_launch_list)**2    + np.array(sigma_return_list)**2)
    sigma_dt_sf   = np.sqrt(np.array(sigma_launch_sf_list)**2 + np.array(sigma_return_sf_list)**2)

    print(f"\n  Evaluation periods: {n_eval}")
    print(f"  τ_launch:  mean={np.mean(tau_launch):.2f} μs  σ={np.std(tau_launch):.3f} μs")
    print(f"  τ_return:  mean={np.mean(tau_return):.2f} μs  σ={np.std(tau_return):.3f} μs")
    print(f"  Δτ = τ_return - τ_launch:")
    print(f"    [parabola]  mean={mean_dt:.3f} μs  between-period σ={std_dt:.4f} μs")
    print(f"    [shape-fit] mean={mean_dt_sf:.3f} μs  between-period σ={std_dt_sf:.4f} μs")
    print(f"\n  Per-period δ(Δτ) = √(δ_L² + δ_R²)  [from LS covariance — estimator uncertainty]:")
    print(f"    parabola:  median={np.median(sigma_dt_para)*1e3:.1f} ns  mean={np.mean(sigma_dt_para)*1e3:.1f} ns")
    print(f"    shape-fit: median={np.median(sigma_dt_sf)*1e3:.1f} ns  mean={np.mean(sigma_dt_sf)*1e3:.1f} ns")
    print(f"    improvement: {np.median(sigma_dt_para)/np.median(sigma_dt_sf):.2f}×  (median parabola / median shape-fit)")

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

    # Panel 2: Example cross-correlations (raw with carrier + envelope overlay)
    ax = axes[1]
    zoom = 500  # μs either side of peak
    m_l = (lags_l >= τl_ex - zoom) & (lags_l <= τl_ex + zoom)
    m_r = (lags_r >= τr_ex - zoom) & (lags_r <= τr_ex + zoom)
    xc_l_norm = xc_l_ex / (np.max(np.abs(xc_l_ex)) + 1e-12)
    xc_r_norm = xc_r_ex / (np.max(np.abs(xc_r_ex)) + 1e-12)
    # raw xcorr (thin, shows carrier oscillations)
    ax.plot(lags_l[m_l], xc_l_norm[m_l], 'b-', lw=0.6, alpha=0.45, label=f'xcorr launch  τ={τl_ex:.2f}μs')
    ax.plot(lags_r[m_r], xc_r_norm[m_r], 'r-', lw=0.6, alpha=0.45, label=f'xcorr return  τ={τr_ex:.2f}μs')
    # envelope overlay (bold, top only)
    ax.plot(lags_l[m_l], env_l_ex[m_l], 'b-', lw=2.0, alpha=0.9)
    ax.plot(lags_r[m_r], env_r_ex[m_r], 'r-', lw=2.0, alpha=0.9)
    ax.axvline(τl_ex, color='b', ls='--', alpha=0.6)
    ax.axvline(τr_ex, color='r', ls='--', alpha=0.6)
    ax.set_xlabel('Lag (μs)')
    ax.set_ylabel('Normalised xcorr')
    ax.set_title(f'Example Cross-Correlations (period {n_periods-1})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Δτ time series — hollow black circles + error bars + blue thread
    ax = axes[2]
    n_template_used = n_template
    t_sec = (np.arange(n_eval) + n_template_used) / rep_rate
    err_us = sigma_dt_sf * 1e6     # shape-fit propagated uncertainty, μs

    # Blue thread
    ax.plot(t_sec, delta_tau_sf, '-', color='#4477cc', lw=0.7, alpha=0.6, zorder=1)
    # Error bars + hollow black circles (plot every point; errorevery thins if dense)
    step = max(1, n_eval // 600)   # cap to ~600 visible error bars
    ax.errorbar(t_sec[::step], delta_tau_sf[::step], yerr=err_us[::step],
                fmt='o', ms=2.5, mfc='none', mec='black', mew=0.7,
                ecolor='black', elinewidth=0.5, capsize=0, zorder=2, alpha=0.7)
    ax.axhline(mean_dt_sf, color='navy', ls='--', lw=1.2, alpha=0.7,
               label=f'mean = {mean_dt_sf:.3f} μs')
    ax.axhline(mean_dt_sf + std_dt_sf, color='gray', ls=':', lw=0.9)
    ax.axhline(mean_dt_sf - std_dt_sf, color='gray', ls=':', lw=0.9,
               label=f'σ = {std_dt_sf:.3f} μs  ({std_dt_sf/TOF_SENS_US_PER_DEG:.4f}°C/ping)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Δτ (μs)')
    ax.set_title(f'Δτ time series  (shape-fit)  '
                 f'N={n_eval}  σ={std_dt_sf:.3f} μs  '
                 f'δ_prop med={np.median(sigma_dt_sf)*1e6:.0f} ns')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    stem = os.path.splitext(os.path.basename(wav_path))[0]
    # Always save into stubbs_pipeline_v1/ next to the WAV file
    pipeline_dir = os.path.join(os.path.dirname(wav_path), '..', 'stubbs_pipeline_v1')
    pipeline_dir = os.path.normpath(pipeline_dir)
    os.makedirs(pipeline_dir, exist_ok=True)
    if save_plot:
        out_path = os.path.join(pipeline_dir, f'stubbs_{stem}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"\n  Plot saved: {out_path}")
    plt.close()
    # Save NPZ
    npz_path = os.path.join(pipeline_dir, f'stubbs_{stem}.npz')
    np.savez(npz_path,
             delta_tau=delta_tau_sf, sigma_launch=np.array(sigma_launch_sf_list),
             sigma_return=np.array(sigma_return_sf_list),
             tau_launch=tau_launch_sf, tau_return=tau_return_sf,
             rep_rate=rep_rate, n_template=n_template)
    print(f"  NPZ  saved: {npz_path}")

    return dict(
        T_launch=T_launch, T_return=T_return,
        tau_launch=tau_launch, tau_return=tau_return, delta_tau=delta_tau,
        mean_delta_tau_us=mean_dt, std_delta_tau_us=std_dt,
        temp_precision_C=temp_precision, n_template=n_template, n_eval=n_eval,
        fs=fs, incident_win_ms=incident_win_ms, reflection_win_ms=reflection_win_ms,
        mean_fit_sigma_us=None,
        sigma_launch=np.array(sigma_launch_list),
        sigma_return=np.array(sigma_return_list),
        # Shape-fit estimator (vernier CCF fit)
        tau_launch_sf=tau_launch_sf, tau_return_sf=tau_return_sf,
        delta_tau_sf=delta_tau_sf,
        mean_delta_tau_sf_us=mean_dt_sf, std_delta_tau_sf_us=std_dt_sf,
        temp_precision_sf_C=temp_prec_sf,
        mean_fit_sigma_sf_us=None,
        sigma_launch_sf=np.array(sigma_launch_sf_list),
        sigma_return_sf=np.array(sigma_return_sf_list),
        sigma_dt_para=sigma_dt_para,   # per-period δ(Δτ) = √(δ_L²+δ_R²), parabola
        sigma_dt_sf=sigma_dt_sf,       # per-period δ(Δτ) = √(δ_L²+δ_R²), shape-fit
        # Example xcorr (last eval period) for panel-2 plotting in batch scripts
        ex_xc_l=xc_l_ex, ex_xc_r=xc_r_ex,
        ex_env_l=env_l_ex, ex_env_r=env_r_ex,
        ex_lags_l=lags_l, ex_lags_r=lags_r,
        ex_tau_l=τl_ex, ex_tau_r=τr_ex,
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
