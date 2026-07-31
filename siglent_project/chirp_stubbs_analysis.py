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
HP_CUTOFF  = 28000   # bandpass lower edge (rejects 25 kHz secondary resonance + below)
BP_HI_CUTOFF = 65000 # bandpass upper edge (rejects broadband noise above piezo band)
HP_ORDER   = 8

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
    """Bandpass 28–65 kHz: rejects sub-piezo noise and broadband high-freq noise."""
    nyq = fs / 2
    sos = butter(HP_ORDER, [HP_CUTOFF / nyq, BP_HI_CUTOFF / nyq],
                 btype='band', output='sos')
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


def xcorr_lag_carrier_phase(signal_seg, template, fs,
                             search_radius_us=300.0, f_carrier_hz=41000.0,
                             coarse_lag_us=None):
    """
    Sub-sample lag via carrier-phase extraction.  Designed for narrowband chirps
    (BW << f_carrier) where the Hilbert envelope is nearly flat within the search
    window and xcorr_lag_us 'defaults to 0' (argmax lands on the noise-dominated
    flat envelope rather than the true peak).

    The CCF of a linear FM chirp is:
        R(τ) ≈ A × sinc(BW × τ) × exp(j 2π f_c τ)

    Coarse estimate — envelope-free, works for any BW:
        τ_true ≈ −∠( Σ_τ R(τ) × exp(−j 2π f_c τ/fs) × |R(τ)| ) / (2π f_c/fs)

        Detrending the carrier from each CCF sample and taking the amplitude-
        weighted complex sum gives the mean phase offset ≡ −2π f_c τ_true/fs,
        regardless of how flat the envelope is.  No argmax needed.

    Fine correction — single-sample carrier phase:
        Once τ_coarse is known (within ±T_c/2 of the truth), the carrier phase
        at that sample gives a sub-cycle residual:
        δ = −∠(R_analytic[τ_coarse]) / (2π f_c/fs)   mod ±T_c/2

    Cramér-Rao bound:  σ ≈ T_carrier / (2π × SNR_amplitude)
    At 41 kHz, SNR=100 → σ ≈ 39 ns;  SNR=1000 → σ ≈ 3.9 ns.
    """
    xc = correlate(signal_seg, template, mode='full')
    center    = len(template) - 1
    xc_anal   = hilbert(xc)
    amplitude = np.abs(xc_anal)

    radius = int(search_radius_us * fs / 1e6)
    lo = max(0, center - radius)
    hi = min(len(xc) - 1, center + radius)

    T_c_samp    = fs / f_carrier_hz
    from_slope  = 2.0 * np.pi * f_carrier_hz / fs   # rad / sample

    if coarse_lag_us is not None:
        coarse = int(round(center + coarse_lag_us * fs / 1e6))
        coarse = max(lo, min(hi, coarse))
    else:
        # Carrier-detrended weighted complex sum — works even for flat envelopes
        lags      = np.arange(lo, hi + 1, dtype=float)
        detrended = xc_anal[lo:hi + 1] * np.exp(-1j * from_slope * (lags - center))
        weights   = amplitude[lo:hi + 1]
        wsum      = float(np.sum(weights))
        if wsum < 1e-12:
            return 0.0, 1e6 / fs, 0.0, 0.0
        weighted_mean_phase = np.angle(np.dot(weights, detrended) / wsum)
        coarse_frac = -weighted_mean_phase / from_slope
        coarse = int(round(center + coarse_frac))
        coarse = max(lo, min(hi, coarse))

    # Fine correction: carrier phase at the coarse sample → sub-cycle residual
    phase_at_coarse = float(np.angle(xc_anal[coarse]))
    delta_samp = -phase_at_coarse / (2.0 * np.pi) * T_c_samp
    delta_samp = ((delta_samp + T_c_samp / 2) % T_c_samp) - T_c_samp / 2

    lag_us = (coarse - center + delta_samp) / fs * 1e6

    # Uncertainty
    noise_lo = max(0, center - int(0.020 * fs))
    noise_hi = max(1, center - int(0.015 * fs))
    sigma_noise = float(np.sqrt(np.mean(amplitude[noise_lo:noise_hi] ** 2))) + 1e-12
    snr_amp     = float(amplitude[coarse]) / sigma_noise
    sigma_us    = (T_c_samp / (2.0 * np.pi * snr_amp)) / fs * 1e6

    return lag_us, sigma_us, float(amplitude[coarse]), snr_amp


def xcorr_lag_shape_fit_dechirp(signal_seg, template, fs,
                                 f_lo_hz, chirp_rate_hz_per_s,
                                 R_auto=None, window_half_us=200.0,
                                 search_radius_us=700.0,
                                 coarse_lag_us=None):
    """
    Dechirped LS shape-fit — works at ANY lag position (Category 1 fix).

    For LFM chirps, the CCF is itself a chirp: xc(τ) ≈ sinc(BW·τ) × exp(j·chirp_phase(τ)).
    Multiplying by exp(-j·chirp_phase(τ)) dechirps it into an approximately real sinc
    regardless of where the main lobe sits on the lag axis.  The same dechirp applied to
    R_auto gives a real sinc centred at lag=0.  The LS fit of Re(xc_d) vs Re(R_auto_d)
    is then well-conditioned at any lag — eliminating the R_auto model-mismatch that
    causes the original shape-fit to fail for Category 1 shifted CCFs.

    After dechirp: Re(xc_d) ≈ A · sinc(BW·ξ),  Re(R_auto_d) ≈ sinc(BW·ξ)
    → LS recovers sub-sample δ = -B/A correctly at any peak position.

    Parameters
    ----------
    f_lo_hz             : chirp start frequency (Hz)  — 38450 for a0.3_sm_4ms
    chirp_rate_hz_per_s : BW/T (Hz/s)                — 1.5e6 for a0.3_sm_4ms (6kHz/4ms)
    """
    xc = correlate(signal_seg, template, mode='full')
    if R_auto is None:
        R_auto = correlate(template, template, mode='full')
    center = len(template) - 1

    # Analytic signals — needed so that dechirp removes the carrier cleanly.
    # For real CCF xc(τ) ≈ A·sinc(BW·τ)·cos(chirp_phase(τ)), the Hilbert transform
    # gives xc_anal ≈ A·sinc(BW·τ)·exp(j·chirp_phase(τ)).  Multiplying by
    # D = exp(-j·chirp_phase) then yields ≈ A·sinc — a real sinc regardless of lag.
    # Using xc.astype(complex) instead would leave a double-frequency artifact.
    xc_anal  = hilbert(xc)
    Ra_anal  = hilbert(R_auto)

    # Build dechirp factors D(k) = exp(-j2π(f_lo·k/fs + rate·(k/fs)²/2))
    k_arr = np.arange(len(xc)) - center          # lag in samples
    lags_s = k_arr / fs
    chirp_phase = 2*np.pi*(f_lo_hz*lags_s + chirp_rate_hz_per_s*lags_s**2/2)
    D = np.exp(-1j * chirp_phase)

    xc_d  = xc_anal * D                        # ≈ A·sinc (complex, constant phase)
    Rau_d = (Ra_anal * D).real                  # ≈ sinc envelope of R_auto (real)

    # Coarse peak from analytic envelope
    env = np.abs(xc_anal)
    if coarse_lag_us is not None:
        coarse = center + int(round(coarse_lag_us * fs / 1e6))
        coarse = max(0, min(coarse, len(xc)-1))
    else:
        radius = int(search_radius_us * fs / 1e6)
        lo = max(0, center - radius)
        hi = min(len(xc)-1, center + radius)
        coarse = lo + int(np.argmax(env[lo:hi+1]))

    # Rotate xc_d so the peak is real positive — removes residual carrier phase
    phase_at_coarse = np.angle(xc_d[coarse])
    rotation = np.exp(-1j * phase_at_coarse)
    xc_d_rot = xc_d * rotation                  # peak is now real positive

    # LS fit window
    W = max(int(window_half_us * fs / 1e6), 5)
    k_all = np.arange(-W, W+1, dtype=int)
    valid = (
        (coarse + k_all >= 0) & (coarse + k_all < len(xc)) &
        (center + k_all >= 1) & (center + k_all < len(Rau_d)-1)
    )
    k = k_all[valid]
    if len(k) < 5:
        return xcorr_lag_us(signal_seg, template, fs, search_radius_us)

    xc_fit = xc_d_rot[coarse + k].real         # ≈ A·sinc(BW·k/fs)
    ra     = Rau_d[center + k]                  # ≈ sinc(BW·k/fs)
    ra_d   = (Rau_d[center + k + 1] - Rau_d[center + k - 1]) / 2.0

    # Weight by dechirped R_auto envelope
    w = Rau_d[center + k].copy()
    w = np.maximum(w, 0.0); w /= (w.max() + 1e-12)

    Phi   = np.column_stack([ra, ra_d])
    Phi_w = Phi * w[:, None]; y_w = xc_fit * w

    AB, _, _, _ = np.linalg.lstsq(Phi_w, y_w, rcond=None)
    A, B = float(AB[0]), float(AB[1])

    if abs(A) < 1e-10:
        return float(coarse - center) / fs * 1e6, 1e6/fs, 0.0, np.nan

    delta_samp = -B / A
    lag_us = (coarse - center + delta_samp) / fs * 1e6

    residuals    = xc_fit - Phi @ AB
    sigma2_noise = float(np.dot(residuals, residuals)) / max(1, len(xc_fit)-2)
    try:
        cov = sigma2_noise * np.linalg.inv(Phi.T @ Phi)
    except np.linalg.LinAlgError:
        cov = sigma2_noise * np.linalg.pinv(Phi.T @ Phi)
    sA2 = cov[0,0]; sB2 = cov[1,1]; cAB = cov[0,1]
    var_delta = (sB2 + delta_samp**2*sA2 - 2*delta_samp*cAB) / (A**2)
    sigma_us  = float(np.sqrt(max(0.0, var_delta))) * 1e6 / fs
    rms_resid = float(np.sqrt(sigma2_noise)) / (abs(A) + 1e-12)
    return lag_us, sigma_us, float(A), rms_resid


def xcorr_lag_shape_fit(signal_seg, template, fs,
                        R_auto=None, window_half_us=200.0,
                        search_radius_us=200.0,
                        coarse_lag_us=None):
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

    # Coarse peak: either from caller (parabola seed) or envelope argmax.
    # Passing coarse_lag_us eliminates carrier-cycle slips: the parabola estimator
    # (xcorr_lag_us) is immune to carrier ambiguity because it operates on the
    # smoothed Hilbert envelope with a tight search radius; seeding the shape-fit
    # from it means the LS window is always centred on the correct peak.
    if coarse_lag_us is not None:
        coarse = center + int(round(coarse_lag_us * fs / 1e6))
        coarse = max(0, min(coarse, len(xc) - 1))
    else:
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


def learn_burst_extent(period_data, fs, peak_sample,
                       threshold_db=-20.0, margin_ms=0.3):
    """
    Infer burst onset and offset from the Hilbert envelope of per0.

    Uses a 0.5 ms smoothed envelope so individual carrier-oscillation dips in
    the raw Hilbert envelope don't terminate the window prematurely — a common
    failure for upchirps driving resonant piezos where the envelope has ripple
    from dispersive ring-up/down.

    threshold_db : level below peak treated as noise boundary (default -20 dB = 10%).
    margin_ms    : safety margin added outside the learned extent (ms).
    """
    env_raw = np.abs(hilbert(period_data))
    n       = len(period_data)
    t_ms_arr  = np.arange(n) / fs * 1000.0
    period_ms = t_ms_arr[-1]

    # Smooth envelope: 0.5 ms moving average suppresses carrier ripple (~40 cycles)
    smooth_n = max(3, int(0.0005 * fs))
    env      = np.convolve(env_raw, np.ones(smooth_n) / smooth_n, mode='same')

    # Noise floor: RMS of first 0.5 ms (pre-burst silence with pre_inc_ms=4ms)
    noise_n   = max(2, min(int(0.0005 * fs), n // 10))
    noise_rms = float(np.sqrt(np.mean(env[:noise_n] ** 2))) + 1e-12

    peak_amp  = float(env[min(peak_sample, n - 1)])
    threshold = max(noise_rms * 10.0,
                    peak_amp * (10.0 ** (threshold_db / 20.0)))

    # Onset: walk backward from peak until below threshold
    onset_samp = 0
    for k in range(min(peak_sample, n - 1), -1, -1):
        if env[k] < threshold:
            onset_samp = k
            break

    # Offset: walk forward from peak until below threshold
    offset_samp = n - 1
    for k in range(min(peak_sample, n - 1), n):
        if env[k] < threshold:
            offset_samp = k
            break

    onset_ms  = max(0.0,        t_ms_arr[onset_samp]  - margin_ms)
    offset_ms = min(period_ms,  t_ms_arr[offset_samp] + margin_ms)

    return onset_ms, offset_ms


def detect_windows(period_data, fs, win_half_width_ms=WIN_HALF_WIDTH_MS,
                   tof_us=5700.0, tol_ms=2.0):
    """
    Detect incident and reflection windows within a single period.

    Strategy: data-driven, no hard-coded TOF prior.
      1. Find anchor = global envelope max in period interior.
      2. Learn anchor burst extent via learn_burst_extent → exclusion zone.
      3. Find second-largest peak ANYWHERE outside that zone (full-interior
         search) — no tof_us constraint.  Amplitude guard (≥40% of anchor)
         rejects weak multipath.
      4. earlier peak = incident, later = reflected.
      5. Learn each burst's window via learn_burst_extent.

    tof_us is kept for backward compatibility but is no longer used in the
    primary search path.  It is only used in the fallback (step 3 failure).
    """
    env_raw = np.abs(hilbert(period_data))
    # Smooth envelope: 0.5 ms moving average removes carrier ripple that creates
    # false peaks in the raw Hilbert envelope (the real failure mode here).
    smooth_n  = max(3, int(0.0005 * fs))
    env_sm    = np.convolve(env_raw, np.ones(smooth_n) / smooth_n, mode='same')
    env_n     = env_sm / (np.max(env_sm) + 1e-12)
    t_ms = np.arange(len(period_data)) / fs * 1000
    period_ms = len(period_data) / fs * 1000

    # Step 1: anchor = global max in interior [2ms, period-2ms]
    edge_ms  = 2.0
    lo_edge  = int(edge_ms / 1000 * fs)
    hi_edge  = max(lo_edge + 1, len(env_n) - int(edge_ms / 1000 * fs))
    interior = env_n[lo_edge:hi_edge]
    anchor_sample = lo_edge + int(np.argmax(interior))
    anchor_ms     = float(t_ms[anchor_sample])

    # Step 2: learn anchor burst extent → exclusion zone
    anc_onset_ms, anc_offset_ms = learn_burst_extent(period_data, fs, anchor_sample)
    excl_margin_ms = 1.0   # extra gap outside the learned burst edges
    excl_lo_i = max(0,          int((anc_onset_ms  - excl_margin_ms) / 1000 * fs))
    excl_hi_i = min(len(env_n), int((anc_offset_ms + excl_margin_ms) / 1000 * fs))

    # Step 3: full-interior search for second-largest peak outside exclusion zone
    # Uses smoothed envelope throughout — no false spikes from carrier ripple.
    search_mask = np.zeros(len(env_n), dtype=bool)
    search_mask[lo_edge:hi_edge] = True
    search_mask[excl_lo_i:excl_hi_i] = False
    masked_env = np.where(search_mask, env_n, 0.0)
    second_sample = int(np.argmax(masked_env))
    second_val    = float(masked_env[second_sample])

    anchor_val    = float(env_n[anchor_sample])
    MIN_PAIR_FRAC = 0.40

    if second_val >= MIN_PAIR_FRAC * anchor_val and second_sample != anchor_sample:
        # Valid second peak found
        if second_sample < anchor_sample:
            inc_sample, ref_sample = second_sample, anchor_sample
        else:
            inc_sample, ref_sample = anchor_sample, second_sample
        tof_detected_ms = abs(float(t_ms[ref_sample]) - float(t_ms[inc_sample]))
        print(f"  [detect_windows] peaks at {float(t_ms[inc_sample]):.2f} ms "
              f"and {float(t_ms[ref_sample]):.2f} ms  "
              f"(TOF={tof_detected_ms:.2f} ms, R/I={second_val/anchor_val:.2f})")
    else:
        # Fallback: tof_us-constrained search
        print(f"  [detect_windows] full-interior search failed "
              f"(best second={second_val:.3f} < {MIN_PAIR_FRAC:.0%}×{anchor_val:.3f}), "
              f"falling back to tof_us={tof_us:.0f} μs")
        tof_ms   = tof_us / 1000.0
        fb_lo    = max(lo_edge, int((anchor_ms + tof_ms - tol_ms) / 1000 * fs))
        fb_hi    = min(hi_edge, int((anchor_ms + tof_ms + tol_ms) / 1000 * fs))
        if fb_hi > fb_lo:
            ref_sample = fb_lo + int(np.argmax(env_n[fb_lo:fb_hi]))
            inc_sample = anchor_sample
        else:
            inc_sample = anchor_sample
            ref_sample = min(len(env_n)-1,
                             anchor_sample + int(tof_ms / 1000 * fs))

    inc_center_ms = float(t_ms[min(inc_sample, len(t_ms) - 1)])
    ref_center_ms = float(t_ms[min(ref_sample, len(t_ms) - 1)])

    # Learn window extent from per0 envelope — replaces fixed ±win_half_width_ms.
    # win_half_width_ms is kept as a parameter for backward compat but is now only
    # used as a fallback if the learned window is degenerate.
    inc_onset_ms, inc_offset_ms = learn_burst_extent(period_data, fs, inc_sample)
    ref_onset_ms, ref_offset_ms = learn_burst_extent(period_data, fs, ref_sample)

    # Fallback: if learned window is too narrow (<1 ms), use safe defaults.
    # Incident left edge → 0 ms (period start) to capture full burst onset.
    # Incident right edge → peak + win_half_width_ms (capped below).
    if (inc_offset_ms - inc_onset_ms) < 1.0:
        inc_onset_ms  = 0.0
        inc_offset_ms = min(period_ms, inc_center_ms + win_half_width_ms)
        print(f"  learn_burst_extent: incident window degenerate, "
              f"using [0, peak+{win_half_width_ms:.1f}ms] fallback")
    if (ref_offset_ms - ref_onset_ms) < 1.0:
        ref_onset_ms  = max(0.0,       ref_center_ms - win_half_width_ms)
        ref_offset_ms = min(period_ms, ref_center_ms + win_half_width_ms)
        print(f"  learn_burst_extent: reflected window degenerate, "
              f"using ±{win_half_width_ms:.1f} ms fallback")

    # Cap incident window right edge at inc + 0.4×TOF.
    # 40% of inter-peak separation keeps the right edge well clear of the
    # reflected burst onset (~burst_duration/2 before its peak) while still
    # capturing the full incident ring-down.
    tof_detected_ms = ref_center_ms - inc_center_ms
    max_inc_right   = inc_center_ms + tof_detected_ms * 0.4
    if inc_offset_ms > max_inc_right:
        print(f"  detect_windows: incident right edge capped to "
              f"inc+0.4×TOF = {max_inc_right:.2f} ms  (was {inc_offset_ms:.2f} ms)")
        inc_offset_ms = max_inc_right

    inc_win = (inc_onset_ms,  inc_offset_ms)
    ref_win = (ref_onset_ms,  ref_offset_ms)

    # Reflection window must not start before incident window ends — no overlap.
    # No additional ring-down gap: piezo τ_ringdown ≈ 172 μs, so by inc+0.4×TOF
    # (≥1 ms past the incident peak) the ring-down is negligible (<e^{-6}).
    if ref_win[0] < inc_win[1]:
        ref_win = (inc_win[1], ref_win[1])
        print(f"  detect_windows: reflection start set to {inc_win[1]:.2f} ms "
              f"(no-overlap, no dead zone)")

    print(f"  Detected incident  peak at {inc_center_ms:.2f} ms  "
          f"→ window [{inc_win[0]:.2f}, {inc_win[1]:.2f}] ms  "
          f"(width={inc_win[1]-inc_win[0]:.2f} ms, envelope-learned)")
    print(f"  Detected reflection peak at {ref_center_ms:.2f} ms  "
          f"→ window [{ref_win[0]:.2f}, {ref_win[1]:.2f}] ms  "
          f"(width={ref_win[1]-ref_win[0]:.2f} ms, sep={ref_center_ms-inc_center_ms:.1f} ms)")
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
                           pre_inc_ms=4.0, amp_threshold=0.15):
    """
    pre_inc_ms raised from 2.0 → 4.0 ms.
    At 2.0 ms, high-α waveforms (e.g. a0.8_sm_4ms) have their Hilbert peak
    at ~3 ms into the 4 ms burst, placing the burst onset at 2−3 = −1 ms
    (clipped by the period boundary).  4.0 ms puts the onset at ~1 ms inside
    the window.  FM3b impact: inc_win right edge moves from 5→7 ms, forcing
    ref_win start to 7.3 ms; at TOF=5.8 ms this clips only ~0.5 ms of the
    reflected Tukey onset (negligible amplitude) — verified safe for our geometry.

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
    # Normalise by the maximum in the first half of the recording so that
    # reverberant tails (which can exceed incident amplitude in enclosed spaces)
    # don't inflate the reference and hide the reflected burst below threshold.
    env_ref = float(np.max(env[:len(env) // 2])) + 1e-12
    env_n = env / env_ref

    # Minimum spacing between detected peaks.
    min_dist = max(1, min(period_samples // 3, int(tof_samples * 0.8)))
    peaks, props = find_peaks(env_n, height=amp_threshold, distance=min_dist)

    # Pair peaks into (incident, reflected) by TOF separation.
    #
    # KEY CHANGE: search for the reflected directly in the raw envelope at
    # [p + tof ± tol] rather than requiring it to appear in find_peaks.
    # In enclosed/reverberant environments the reflected burst can be weaker
    # than a later echo, falling below find_peaks height threshold even though
    # it is clearly visible on the scope.
    # MIN_PAIR_FRAC is applied relative to the LOCAL incident amplitude (env[p]),
    # not the global max, so it works even when a reverberant tail dominates.
    MIN_PAIR_FRAC = 0.40   # restored to 0.40 — 0.15 caused false pairings with noise in direct-drive setups
    MAX_PAIR_FRAC = 1.0 / MIN_PAIR_FRAC

    incident_samples = []
    used = set()
    for i, p in enumerate(peaks):
        if i in used:
            continue
        # Direct envelope search at expected TOF position (no find_peaks dependency)
        lo = max(0, p + tof_samples - tol_samples)
        hi = min(len(env), p + tof_samples + tol_samples)
        if lo >= hi:
            continue
        r_idx = lo + int(np.argmax(env[lo:hi]))
        amp_p = float(env[p])
        amp_r = float(env[r_idx])
        ratio = amp_r / (amp_p + 1e-12)
        if ratio >= MIN_PAIR_FRAC and ratio <= MAX_PAIR_FRAC:
            incident_samples.append(p)
            used.add(i)
            continue
        # Legacy: also try find_peaks partners (backward compat for open setups)
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

    # Deduplicate: keep only the EARLIEST incident within each period window.
    # In reverberant setups (enclosures), wall echoes arrive 1-3 ms after the
    # direct burst and can be detected as a second valid incident candidate.
    # Keeping the earliest ensures the direct burst is always the anchor,
    # preventing bimodal templates from alternating between direct and echo.
    half_period = window_samples // 2
    incident_dedup = []
    for inc in sorted(incident_samples):
        if not incident_dedup or inc - incident_dedup[-1] > half_period:
            incident_dedup.append(inc)
    incident_samples = incident_dedup

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


# ── Period-correction helpers (module-level so GUI can import them) ────────────

def fft_shift(x, s):
    """Sub-sample shift by s samples via FFT phase ramp (circular — |s| < 0.5 only)."""
    N = len(x)
    X = np.fft.rfft(x, n=N)
    X *= np.exp(-2j * np.pi * np.fft.rfftfreq(N) * s)
    return np.fft.irfft(X, n=N)


def linear_shift(x, s):
    """Integer-sample shift with zero-padding (no circular wrap). Positive s = right."""
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
    """Drift-correct a period: integer part via linear_shift, fraction via fft_shift."""
    s_int  = int(round(shift_s))
    s_frac = shift_s - s_int
    out    = linear_shift(period, -s_int)
    if abs(s_frac) > 1e-6:
        out = fft_shift(out, -s_frac)
    return out


# ── Main analysis ──────────────────────────────────────────────────────────────

def analyze(wav_path, rep_rate=REP_RATE, n_template=N_TEMPLATE,
            incident_win_ms=INCIDENT_WIN_MS, reflection_win_ms=REFLECTION_WIN_MS,
            win_half_width_ms=WIN_HALF_WIDTH_MS, output_dir=OUTPUT_DIR,
            save_plot=True, _return_internals=False,
            _plot_parabola=False, _pipeline_dir=None,
            _search_radius_us=200.0,
            _dechirp_f_lo=None, _dechirp_rate=None):
    """
    Run Stubbs dual-template analysis on a WAV file.
    Returns dict with templates, per-period lags, and precision metrics.
    """
    # Use soundfile to read — handles both standard WAV and RF64 WAV (>4 GB)
    import soundfile as _sf
    raw, fs = _sf.read(wav_path, dtype='int16', always_2d=False)
    data = raw.astype(np.float64) / 32768.0

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

    # fft_shift / linear_shift / apply_correction are module-level (see below analyze)

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

    _DRIFT_THRESHOLD_US_PER_PERIOD = 0.5
    drift_rate_us = ema_rate * 1e6 / fs
    _fm3_zeroed = abs(drift_rate_us) > _DRIFT_THRESHOLD_US_PER_PERIOD
    if _fm3_zeroed:
        shifts_frac_max_us = float(np.max(np.abs(shifts_frac))) * 1e6 / fs
        shifts_frac[:] = 0.0
        print(f"  FM3 drift zeroed: EMA rate {drift_rate_us:.2f} μs/period > "
              f"{_DRIFT_THRESHOLD_US_PER_PERIOD} threshold (max was "
              f"{shifts_frac_max_us:.0f} μs — detection noise, slicer handles alignment)")
    else:
        print(f"  FM3 drift kept: EMA rate {drift_rate_us:.3f} μs/period "
              f"≤ {_DRIFT_THRESHOLD_US_PER_PERIOD} threshold (genuine small drift)")

    # Build templates: adaptive coherent accumulation.
    #
    # Standard approach (shifts_frac only): robust when FM3 keeps the EMA, but
    # fails when FM3 zeroes it — template periods then accumulate with carrier-
    # max jitter from the raw-envelope slicer, producing multi-hump templates.
    #
    # Adaptive fix: after applying shifts_frac, additionally align each period to
    # the CURRENT RUNNING TEMPLATE via direct xcorr_lag_us.  The direct measurement
    # is always correct regardless of FM3 status and costs ~1 ms per period.
    # Search radius for adaptive xcorr:
    #   FM3 not triggered → burst is at ~4.0ms ± tiny shifts_frac, correction is small.
    #     ±50 μs is enough and avoids locking onto CCF sidelobes at ±200 μs that
    #     can appear for narrow-BW waveforms (flat3db: BW=2.3 kHz, sidelobe at ~±435 μs).
    #   FM3 triggered → bimodal slicer shifts burst by ~200 μs; need wider search.
    _ADAPTIVE_SEARCH_US = 200.0 if not _fm3_zeroed else 500.0
    _MAX_TMPL_CORRECT_SAMP = int(0.002 * fs)   # 2 ms hard cap (pathological rejection)
    T_launch_acc = np.zeros(period_idx)
    T_return_acc = np.zeros(period_idx)
    n_tmpl_corrected = 0

    for i in range(n_template):
        corrected = apply_correction(sig[slices[i][0]:slices[i][1]], shifts_frac[i])
        inc_i = mask_period(corrected, fs, incident_win_ms)
        ref_i = mask_period(corrected, fs, reflection_win_ms)

        if i > 10:   # wait for 10-period accumulation before adaptive xcorr has enough SNR
            T_L_so_far = T_launch_acc / i
            # Wide search radius: shifts_frac[i] can reach ±4μs/period × 99 = ±400μs,
            # and xcorr_lag_us default ±300μs would miss the peak beyond period ~73.
            τ_direct, _, _, _ = xcorr_lag_us(inc_i, T_L_so_far, fs,
                                              search_radius_us=_ADAPTIVE_SEARCH_US)
            extra = τ_direct * fs / 1e6          # samples
            if 0 < abs(extra) <= _MAX_TMPL_CORRECT_SAMP:
                corrected = apply_correction(corrected, extra)
                inc_i = mask_period(corrected, fs, incident_win_ms)
                ref_i = mask_period(corrected, fs, reflection_win_ms)
                n_tmpl_corrected += 1

        T_launch_acc += inc_i
        T_return_acc += ref_i

    T_launch = T_launch_acc / n_template
    T_return = T_return_acc / n_template
    print(f"  Template adaptive alignment: {n_tmpl_corrected}/{n_template-1} periods "
          f"received extra correction")

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
        τl,  _, _, σl  = xcorr_lag_us(inc_m, T_launch, fs,
                                       search_radius_us=_search_radius_us)
        τr, xc_r, env_r, σr = xcorr_lag_us(ref_m, T_return, fs,
                                             search_radius_us=_search_radius_us)
        if _dechirp_f_lo is not None and _dechirp_rate is not None:
            τl_sf, σl_sf, _, _ = xcorr_lag_shape_fit_dechirp(
                inc_m, T_launch, fs,
                f_lo_hz=_dechirp_f_lo, chirp_rate_hz_per_s=_dechirp_rate,
                R_auto=R_auto_L, search_radius_us=_search_radius_us,
                coarse_lag_us=τl)
            τr_sf, σr_sf, _, _ = xcorr_lag_shape_fit_dechirp(
                ref_m, T_return, fs,
                f_lo_hz=_dechirp_f_lo, chirp_rate_hz_per_s=_dechirp_rate,
                R_auto=R_auto_R, search_radius_us=_search_radius_us,
                coarse_lag_us=τr)
        else:
            τl_sf, σl_sf, _, _ = xcorr_lag_shape_fit(inc_m, T_launch, fs, R_auto=R_auto_L,
                                                       coarse_lag_us=τl)
            τr_sf, σr_sf, _, _ = xcorr_lag_shape_fit(ref_m, T_return, fs, R_auto=R_auto_R,
                                                       coarse_lag_us=τr)
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

    # Panel 3: Δτ time series — use shape-fit unless it's demonstrably worse than
    # parabola (near-singular LS for narrow-BW waveforms like flat3db BW=2.3kHz).
    # Heuristic: if shape-fit median δ > 3× parabola median δ, use parabola.
    ax = axes[2]
    n_template_used = n_template
    t_sec = (np.arange(n_eval) + n_template_used) / rep_rate
    # Use the estimator that the caller is actually using for its measurements.
    # Default: shape-fit (original behaviour). Pass _plot_parabola=True when
    # the pipeline is using parabola (e.g. --use-parabola for narrow-BW waveforms).
    _estimator_label = 'parabola' if _plot_parabola else 'shape-fit'
    _dt_plot = delta_tau        if _plot_parabola else delta_tau_sf
    err_us   = sigma_dt_para    if _plot_parabola else sigma_dt_sf

    # MAD-based outlier removal, then fit axis tightly to inlier data range
    _med      = float(np.median(_dt_plot))
    _mad      = float(np.median(np.abs(_dt_plot - _med))) * 1.4826
    _clip_hw  = 5.0 * _mad
    _inlier   = np.abs(_dt_plot - _med) <= _clip_hw
    n_hidden  = int(np.sum(~_inlier))

    # Inlier-only stats for hlines and legend
    _mean_plot = float(np.mean(_dt_plot[_inlier]))
    _std_plot  = float(np.std(_dt_plot[_inlier]))

    dt_plot = _dt_plot.astype(float).copy()
    dt_plot[~_inlier] = np.nan

    ax.plot(t_sec, dt_plot, '-', color='#4477cc', lw=0.7, alpha=0.6, zorder=1)
    step = max(1, n_eval // 600)
    _sel = _inlier[::step]
    ax.errorbar(t_sec[::step][_sel], _dt_plot[::step][_sel], yerr=err_us[::step][_sel],
                fmt='o', ms=2.5, mfc='none', mec='black', mew=0.7,
                ecolor='black', elinewidth=0.5, capsize=2, zorder=2, alpha=0.7)

    ax.axhline(_mean_plot, color='navy', ls='--', lw=1.2, alpha=0.7,
               label=f'mean = {_mean_plot:.3f} μs  (inliers)')
    ax.axhline(_mean_plot + _std_plot, color='gray', ls=':', lw=0.9)
    ax.axhline(_mean_plot - _std_plot, color='gray', ls=':', lw=0.9,
               label=f'σ = {_std_plot:.3f} μs  ({_std_plot/TOF_SENS_US_PER_DEG:.4f}°C/ping)')

    _ylo_data = float(_dt_plot[_inlier].min())
    _yhi_data = float(_dt_plot[_inlier].max())
    _yrange   = max(_yhi_data - _ylo_data, 2 * _mad, 0.2)
    ax.set_ylim(_ylo_data - 0.12*_yrange, _yhi_data + 0.12*_yrange)

    if n_hidden:
        ax.text(0.99, 0.03,
                f'{n_hidden} outlier{"s" if n_hidden > 1 else ""} hidden  '
                f'(|Δτ−med| > {_clip_hw:.2f} μs = 5×MAD)',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=7.5, color='#cc4444',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#cc4444',
                          alpha=0.85, lw=0.8))

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Δτ = τ_return − τ_launch  (μs)')
    ax.set_title(f'Δτ time series  ({_estimator_label})  '
                 f'N={n_eval} ({n_eval - n_hidden} shown)  '
                 f'σ={_std_plot:.3f} μs  '
                 f'δ_prop med={np.median(err_us)*1e3:.1f} ns')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    stem = os.path.splitext(os.path.basename(wav_path))[0]
    # v2: parabola-seeded shape-fit (eliminates carrier-cycle slips)
    if _dechirp_f_lo is not None and _dechirp_rate is not None:
        _default_pipeline = 'stubbs_pipeline_vtest_dechirp'
    elif _search_radius_us != 200.0:
        _default_pipeline = 'stubbs_pipeline_vtest'
    else:
        _default_pipeline = 'stubbs_pipeline_v3'
    pipeline_dir = _pipeline_dir if _pipeline_dir is not None else \
        os.path.join(os.path.dirname(wav_path), '..', _default_pipeline)
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
             rep_rate=rep_rate, n_template=n_template,
             T_launch=T_launch, T_return=T_return,
             incident_win_ms=np.array(incident_win_ms),
             reflection_win_ms=np.array(reflection_win_ms))
    print(f"  NPZ  saved: {npz_path}")

    _result = dict(
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
    if _return_internals:
        _result['_sig']         = sig
        _result['_slices']      = slices
        _result['_shifts_frac'] = shifts_frac
    return _result


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
