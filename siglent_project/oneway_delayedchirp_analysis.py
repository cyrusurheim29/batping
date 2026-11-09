#!/usr/bin/env python3
"""
Analysis pipeline for delayed-chirp-on-CH2 recordings — FULL FEATURE PARITY
with oneway_sync_analysis.py.

Instead of a sync PULSE on Ch B (handled by oneway_sync_analysis.py), Ch B
has a time-delayed version of the chirp.  τ_L is the sub-sample arrival
time of the chirp on Ch B via matched filter against a Ch B template.

Per-ping procedure:
  1. Detect approximate Ch B chirp arrival (envelope peak)
  2. Refine via matched filter against Ch B template → sub-sample τ_B
  3. Extract Ch A window at expected acoustic offset → matched filter → τ_A
  4. TOF_i = τ_A_i − τ_B_i   per ping
  5. Δτ = TOF − <TOF over first N_BASELINE pings>

Outputs:
  tof_summary.png             4-panel: raw TOF, HF residual, histogram, rolling σ
  uncertainty_histograms.png  3-panel: σ(τ_B), σ(τ_A), σ(Δτ) histograms
  chunk_plots/chunk_XXX.png   per-30s 3-panel diagnostic (templates, CCF, Δτ)
  per_ping.npz                full per-ping arrays
  templates.npz               Ch A + Ch B templates

Usage:
    python3 oneway_delayedchirp_analysis.py <wav_path> \\
        [--per-chunk-plots] [--rep-rate HZ] [--search-ms LO HI] [--chb-delay-ms N]
"""
import os, sys, argparse
import numpy as np
import soundfile as sf
import pandas as pd
from scipy.signal import correlate, hilbert, find_peaks
from scipy.ndimage import uniform_filter1d
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chirp_stubbs_analysis import (
    hp_filter, xcorr_lag_us, xcorr_lag_shape_fit_dechirp, xcorr_lag_shape_fit,
)

# Which shape-fit method per-ping-analysis uses. Set from CLI.  Pipeline keeps
# calling 'xcorr_lag_shape_fit_dechirp'-style signature (same return tuple).
SHAPE_METHOD = 'dechirped'   # 'dechirped' or 'classic'

# ── Configuration ─────────────────────────────────────────────────────────
REP_RATE_DEFAULT     = 41.667
BURST_MS             = 4.0
F_LO                 = 37000.0
F_HI                 = 43000.0
CHIRP_RATE_HZ_S      = (F_HI - F_LO) / (BURST_MS / 1000.0)
CH_A_FS_V_DEFAULT    = 10.0
CH_B_FS_V_DEFAULT    = 10.0
ARRIVAL_WIN_HW_MS    = 3.0
N_TEMPLATE           = 100
N_BASELINE           = 100
CHUNK_S              = 30.0
SEARCH_MS_DEFAULT    = (8.0, 20.0)
CHB_DELAY_MS_DEFAULT = 2.0
SOUND_SPEED_M_S      = 343.0
DC_DT_M_S_PER_C      = 0.606


# ── Ch B chirp detection via envelope peak-finding ────────────────────────
def find_chb_peaks_global(ch_b_hp, fs, rep_rate):
    env = np.abs(hilbert(ch_b_hp))
    smooth_n = max(1, int(0.3e-3 * fs))
    env_smooth = uniform_filter1d(env, size=smooth_n)
    min_dist = int(0.8 / rep_rate * fs)
    env_prom_thresh = np.median(env_smooth) * 3.0
    peaks, _ = find_peaks(env_smooth, distance=min_dist, prominence=env_prom_thresh)
    return peaks


def _shift_fft(x, delta_samp):
    """Sub-sample circular shift via FFT phase rotation."""
    N = len(x)
    X = np.fft.rfft(x)
    k = np.fft.rfftfreq(N, d=1.0) * N
    return np.fft.irfft(X * np.exp(-2j * np.pi * k * delta_samp / N), n=N)


def build_template(ch_hp, arrival_times_samp, n_template, fs, n_iter=2):
    """Build a template by averaging n_template arrival windows.

    Pass 0 uses integer-sample alignment from `arrival_times_samp`.
    Passes 1..n_iter re-align each window to the running template via
    sub-sample xcorr lag + FFT phase shift, then re-average.  This
    recovers carrier phase coherence when the physical arrival time
    jitters at μs scale (airflow turbulence over the acoustic path).
    """
    win_samples = int(2 * ARRIVAL_WIN_HW_MS / 1000 * fs)
    if win_samples % 2 == 0:
        win_samples += 1
    half = win_samples // 2
    windows = []
    for t in arrival_times_samp[:n_template]:
        p = int(round(t))
        lo, hi = p - half, p + half + 1
        if lo < 0 or hi > len(ch_hp):
            continue
        windows.append(ch_hp[lo:hi].copy())
    if len(windows) < 10:
        raise RuntimeError(f"only {len(windows)} usable template windows (need >=10)")
    stack = np.stack(windows)
    tpl = stack.mean(axis=0)
    for _ in range(int(n_iter)):
        aligned = np.empty_like(stack)
        for i, w in enumerate(stack):
            lag_us, _, _, _ = xcorr_lag_us(w, tpl, fs, search_radius_us=500.0)
            aligned[i] = _shift_fft(w, -lag_us * fs / 1e6)
        tpl = aligned.mean(axis=0)
    return tpl, win_samples


def find_cha_initial_arrivals(ch_a_hp, chb_times_samp, search_ms, fs, n_use=100):
    search_lo_samp = int(search_ms[0] / 1000 * fs)
    search_hi_samp = int(search_ms[1] / 1000 * fs)
    cha_times = []
    for t_b in chb_times_samp[:n_use]:
        p_b = int(round(t_b))
        s_lo, s_hi = p_b + search_lo_samp, p_b + search_hi_samp
        if s_hi > len(ch_a_hp):
            break
        search_seg = ch_a_hp[s_lo:s_hi]
        env = np.abs(hilbert(search_seg))
        peak_local = int(np.argmax(env))
        cha_times.append(s_lo + peak_local)
    return cha_times


# ── Per-ping analysis (optionally returns CCFs for diagnostics) ───────────
def analyze_ping_pair(ch_a_hp, ch_b_hp, nominal_chb_time_samp,
                      ch_a_template, ch_b_template, R_auto_a, R_auto_b,
                      fs, win_samples, cha_offset_samp, return_ccf=False):
    half = win_samples // 2

    # Ch B window
    p_b = int(round(nominal_chb_time_samp))
    lo_b, hi_b = p_b - half, p_b + half + 1
    if lo_b < 0 or hi_b > len(ch_b_hp):
        return None
    seg_b = ch_b_hp[lo_b:hi_b]
    if len(seg_b) != win_samples:
        return None
    lag_b_us, xc_b, env_b, sigma_b = xcorr_lag_us(seg_b, ch_b_template, fs,
                                                    search_radius_us=200.0)
    if SHAPE_METHOD == 'classic':
        lag_b_shape_us, sigma_b_shape, _, _ = xcorr_lag_shape_fit(
            seg_b, ch_b_template, fs,
            R_auto=R_auto_b, window_half_us=200.0, search_radius_us=200.0,
            coarse_lag_us=lag_b_us,
        )
    else:
        lag_b_shape_us, sigma_b_shape, _, _ = xcorr_lag_shape_fit_dechirp(
            seg_b, ch_b_template, fs,
            f_lo_hz=F_LO, chirp_rate_hz_per_s=CHIRP_RATE_HZ_S,
            R_auto=R_auto_b, window_half_us=200.0, search_radius_us=200.0,
            coarse_lag_us=lag_b_us,
        )
    tau_b_samp_para  = p_b + lag_b_us        * fs / 1e6
    tau_b_samp_shape = p_b + lag_b_shape_us  * fs / 1e6

    # Ch A window
    p_a = int(round(nominal_chb_time_samp + cha_offset_samp))
    lo_a, hi_a = p_a - half, p_a + half + 1
    if lo_a < 0 or hi_a > len(ch_a_hp):
        return None
    seg_a = ch_a_hp[lo_a:hi_a]
    if len(seg_a) != win_samples:
        return None
    lag_a_us, xc_a, env_a, sigma_a = xcorr_lag_us(seg_a, ch_a_template, fs,
                                                    search_radius_us=200.0)
    if SHAPE_METHOD == 'classic':
        lag_a_shape_us, sigma_a_shape, A_shape, rms_shape = xcorr_lag_shape_fit(
            seg_a, ch_a_template, fs,
            R_auto=R_auto_a, window_half_us=200.0, search_radius_us=200.0,
            coarse_lag_us=lag_a_us,
        )
    else:
        lag_a_shape_us, sigma_a_shape, A_shape, rms_shape = xcorr_lag_shape_fit_dechirp(
            seg_a, ch_a_template, fs,
            f_lo_hz=F_LO, chirp_rate_hz_per_s=CHIRP_RATE_HZ_S,
            R_auto=R_auto_a, window_half_us=200.0, search_radius_us=200.0,
            coarse_lag_us=lag_a_us,
        )
    tau_a_samp_para  = p_a + lag_a_us        * fs / 1e6
    tau_a_samp_shape = p_a + lag_a_shape_us  * fs / 1e6

    tof_para_us  = (tau_a_samp_para  - tau_b_samp_para)  / fs * 1e6
    tof_shape_us = (tau_a_samp_shape - tau_b_samp_shape) / fs * 1e6
    sigma_diff_para_us  = float(np.sqrt(sigma_a**2       + sigma_b**2))
    sigma_diff_shape_us = float(np.sqrt(sigma_a_shape**2 + sigma_b_shape**2))

    result = dict(
        tof_para_us=tof_para_us, tof_shape_us=tof_shape_us,
        sigma_para_us=sigma_a, sigma_shape_us=sigma_a_shape,
        sigma_B_para_us=sigma_b, sigma_B_shape_us=sigma_b_shape,
        sigma_diff_para_us=sigma_diff_para_us,
        sigma_diff_shape_us=sigma_diff_shape_us,
        A=A_shape, rms=rms_shape,
    )
    if return_ccf:
        result['ccf_a']  = xc_a
        result['env_a']  = env_a
        result['ccf_b']  = xc_b
        result['env_b']  = env_b
    return result


# ── 4-panel summary plot (parity with oneway_sync_analysis.plot_summary) ──
def plot_summary(t_pulse, tof_para, tof_shape, dtau_para, dtau_shape,
                  sig_para, sig_shape, inl_para, inl_shape,
                  sigma_full_para, sigma_full_shape, tof_sens,
                  out_dir, wav_basename, rep_rate):
    t_min = t_pulse / 60.0
    fig, axes = plt.subplots(2, 2, figsize=(17, 10))

    # (1) Raw TOF
    ax = axes[0, 0]
    ax.plot(t_min[inl_para],  tof_para[inl_para],  '.', ms=1.0, alpha=0.5,
             color='#1f77b4', label=f'parabolic  σ_full={sigma_full_para:.2f} μs')
    ax.plot(t_min[inl_shape], tof_shape[inl_shape], '.', ms=1.0, alpha=0.5,
             color='#e6444d', label=f'shape-fit  σ_full={sigma_full_shape:.2f} μs')
    ax.set_xlabel('Time (min)'); ax.set_ylabel('TOF (μs)')
    ax.set_title('Raw diff-TOF (τ_A − τ_B)   both methods overlaid')
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='upper right')

    # (2) HF residual
    ax = axes[0, 1]
    win = max(3, int(rep_rate) | 1)
    r_para  = pd.Series(np.where(inl_para,  dtau_para,  np.nan)).rolling(win, center=True, min_periods=win//2).mean().values
    r_shape = pd.Series(np.where(inl_shape, dtau_shape, np.nan)).rolling(win, center=True, min_periods=win//2).mean().values
    ax.plot(t_min[inl_para],  dtau_para[inl_para]  - r_para[inl_para],  '.', ms=1.0, alpha=0.4,
             color='#1f77b4', label='parabolic (HF only)')
    ax.plot(t_min[inl_shape], dtau_shape[inl_shape] - r_shape[inl_shape], '.', ms=1.0, alpha=0.4,
             color='#e6444d', label='shape-fit (HF only)')
    ax.axhline(0, color='k', lw=0.4)
    ax.set_xlabel('Time (min)'); ax.set_ylabel('Δτ − rolling_mean (μs)')
    ax.set_title(f'HF residual (thermal drift removed via {win/rep_rate:.1f}s rolling mean)')
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='upper right')

    # (3) Histograms
    ax = axes[1, 0]
    hf_para  = (dtau_para  - r_para )[inl_para  & np.isfinite(r_para )]
    hf_shape = (dtau_shape - r_shape)[inl_shape & np.isfinite(r_shape)]
    sig_hf_para  = float(np.std(hf_para))
    sig_hf_shape = float(np.std(hf_shape))
    bins = 100
    lo = min(np.percentile(hf_para, 1), np.percentile(hf_shape, 1))
    hi = max(np.percentile(hf_para, 99), np.percentile(hf_shape, 99))
    ax.hist(hf_para,  bins=np.linspace(lo, hi, bins), alpha=0.55, color='#1f77b4',
             label=f'parabolic σ_HF={sig_hf_para:.3f} μs')
    ax.hist(hf_shape, bins=np.linspace(lo, hi, bins), alpha=0.55, color='#e6444d',
             label=f'shape-fit σ_HF={sig_hf_shape:.3f} μs')
    ax.set_xlabel('HF residual (μs)'); ax.set_ylabel('Count')
    ax.set_title('HF residual histogram (drift-subtracted)')
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # (4) Rolling σ
    ax = axes[1, 1]
    win_1m = max(3, int(60 * rep_rate) | 1)
    r_std_p = pd.Series(np.where(inl_para,  dtau_para,  np.nan)).rolling(win_1m, center=True, min_periods=win_1m//3).std().values
    r_std_s = pd.Series(np.where(inl_shape, dtau_shape, np.nan)).rolling(win_1m, center=True, min_periods=win_1m//3).std().values
    ax.plot(t_min, r_std_p, '-', color='#1f77b4', lw=1.0, label='parabolic')
    ax.plot(t_min, r_std_s, '-', color='#e6444d', lw=1.0, label='shape-fit')
    ax.set_xlabel('Time (min)'); ax.set_ylabel('Rolling-1-min σ(Δτ) (μs)')
    ax.set_title('σ(Δτ) rolling window (1-min)  — precision stability over recording')
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    fig.suptitle(f'Delayed-chirp diff-TOF — {wav_basename}   '
                  f'({len(t_pulse):,} pings, {(t_pulse[-1]-t_pulse[0])/60:.1f} min)',
                  fontsize=12, fontweight='bold')
    fig.tight_layout()
    out_png = os.path.join(out_dir, 'tof_summary.png')
    fig.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_png}")


# ── 3-panel uncertainty histograms (σ_B, σ_A, σ_diff) ────────────────────
def plot_uncertainty_histograms(sig_B_para, sig_B_shape, sig_A_para, sig_A_shape,
                                  sig_diff_para, sig_diff_shape, inl_para, inl_shape,
                                  out_dir, wav_basename):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    # (1) σ(τ_B) — matched filter uncertainty on Ch B chirp arrival
    ax = axes[0]
    sig_B_p = sig_B_para[inl_para & np.isfinite(sig_B_para) & (sig_B_para > 0)]
    sig_B_s = sig_B_shape[inl_shape & np.isfinite(sig_B_shape) & (sig_B_shape > 0)]
    med_p = float(np.median(sig_B_p)) if len(sig_B_p) else 0.0
    med_s = float(np.median(sig_B_s)) if len(sig_B_s) else 0.0
    lo = max(min(np.percentile(sig_B_p, 0.1), np.percentile(sig_B_s, 0.1)), 1e-5)
    hi = max(np.percentile(sig_B_p, 99.9),    np.percentile(sig_B_s, 99.9))
    bins = np.logspace(np.log10(lo), np.log10(hi), 60)
    ax.hist(sig_B_p, bins=bins, alpha=0.6, color='#4477cc', edgecolor='k', lw=0.3,
             label=f'parabolic  median = {med_p*1000:.2f} ns')
    ax.hist(sig_B_s, bins=bins, alpha=0.6, color='#88aa44', edgecolor='k', lw=0.3,
             label=f'shape-fit  median = {med_s*1000:.2f} ns')
    ax.axvline(med_p, color='#4477cc', ls='--', lw=1.0, alpha=0.7)
    ax.axvline(med_s, color='#88aa44', ls='--', lw=1.0, alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel('σ(τ_B)  (μs)   [Ch B chirp arrival]')
    ax.set_ylabel('Count')
    ax.set_title('(1) σ(τ_B) — matched-filter uncertainty on Ch B chirp\n'
                  '(replaces σ(τ_L) from sync-pulse pipeline)', fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    # (2) σ(τ_A) — matched filter uncertainty on Ch A arrival
    ax = axes[1]
    sig_A_p = sig_A_para[inl_para & np.isfinite(sig_A_para) & (sig_A_para > 0)]
    sig_A_s = sig_A_shape[inl_shape & np.isfinite(sig_A_shape) & (sig_A_shape > 0)]
    med_p = float(np.median(sig_A_p)) if len(sig_A_p) else 0.0
    med_s = float(np.median(sig_A_s)) if len(sig_A_s) else 0.0
    lo = max(min(np.percentile(sig_A_p, 0.1), np.percentile(sig_A_s, 0.1)), 1e-5)
    hi = max(np.percentile(sig_A_p, 99.9),    np.percentile(sig_A_s, 99.9))
    bins = np.logspace(np.log10(lo), np.log10(hi), 60)
    ax.hist(sig_A_p, bins=bins, alpha=0.6, color='#1f77b4', edgecolor='k', lw=0.3,
             label=f'parabolic  median = {med_p*1000:.2f} ns')
    ax.hist(sig_A_s, bins=bins, alpha=0.6, color='#e6444d', edgecolor='k', lw=0.3,
             label=f'shape-fit  median = {med_s*1000:.2f} ns')
    ax.axvline(med_p, color='#1f77b4', ls='--', lw=1.0, alpha=0.7)
    ax.axvline(med_s, color='#e6444d', ls='--', lw=1.0, alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel('σ(τ_A)  (μs)   [Ch A arrival]')
    ax.set_ylabel('Count')
    ax.set_title('(2) σ(τ_A) — Ch A arrival time uncertainty\n'
                  'both matched-filter methods overlaid', fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    # (3) σ(diff) propagated
    ax = axes[2]
    sig_d_p = sig_diff_para[inl_para & np.isfinite(sig_diff_para) & (sig_diff_para > 0)]
    sig_d_s = sig_diff_shape[inl_shape & np.isfinite(sig_diff_shape) & (sig_diff_shape > 0)]
    med_p = float(np.median(sig_d_p)) if len(sig_d_p) else 0.0
    med_s = float(np.median(sig_d_s)) if len(sig_d_s) else 0.0
    lo = max(min(np.percentile(sig_d_p, 0.1), np.percentile(sig_d_s, 0.1)), 1e-5)
    hi = max(np.percentile(sig_d_p, 99.9),    np.percentile(sig_d_s, 99.9))
    bins = np.logspace(np.log10(lo), np.log10(hi), 60)
    ax.hist(sig_d_p, bins=bins, alpha=0.6, color='#1f77b4', edgecolor='k', lw=0.3,
             label=f'parabolic  median = {med_p*1000:.2f} ns')
    ax.hist(sig_d_s, bins=bins, alpha=0.6, color='#e6444d', edgecolor='k', lw=0.3,
             label=f'shape-fit  median = {med_s*1000:.2f} ns')
    ax.axvline(med_p, color='#1f77b4', ls='--', lw=1.0, alpha=0.7)
    ax.axvline(med_s, color='#e6444d', ls='--', lw=1.0, alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel('σ(Δτ)  (μs)   [propagated]')
    ax.set_ylabel('Count')
    ax.set_title('(3) σ(Δτ) — propagated per-ping uncertainty\n'
                  '= √(σ(τ_A)² + σ(τ_B)²)', fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    fig.suptitle(f'Per-ping uncertainty budget — {wav_basename}',
                  fontsize=12, fontweight='bold')
    fig.tight_layout()
    out_png = os.path.join(out_dir, 'uncertainty_histograms.png')
    fig.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_png}")


# ── Per-chunk 3-panel diagnostic plots ────────────────────────────────────
def plot_per_chunk(cd, ch_a_template, ch_b_template, t_pulse,
                    tof_para, tof_shape, dtau_para, dtau_shape,
                    sig_para, sig_shape, inl_para, inl_shape, fs,
                    chunk_plot_dir, wav_basename, tof_sens):
    """3-panel per-chunk plot: templates (top), example CCFs (middle),
    Δτ time series (bottom with blue thread + black open circles + error bars)."""
    i0, i1 = cd['ping_i_start'], cd['ping_i_end']
    t_chunk       = t_pulse[i0:i1]
    dtau_p_chunk  = dtau_para[i0:i1]
    dtau_s_chunk  = dtau_shape[i0:i1]
    sig_p_chunk   = sig_para[i0:i1]
    inl_p_chunk   = inl_para[i0:i1]

    fig, axes = plt.subplots(3, 1, figsize=(13, 11),
                              gridspec_kw={'height_ratios': [1, 1, 1.4]})

    # (1) Both templates stacked
    ax = axes[0]
    t_tpl_ms = (np.arange(len(ch_a_template)) - len(ch_a_template)//2) / fs * 1000
    ax.plot(t_tpl_ms, ch_a_template, '-', color='#1f77b4', lw=0.8, label='Ch A template (acoustic)')
    ax.plot(t_tpl_ms, ch_b_template, '-', color='#88aa44', lw=0.8, alpha=0.7,
             label='Ch B template (electrical delayed chirp)')
    ax.axvline(0, color='k', ls='--', lw=0.4, alpha=0.5)
    ax.set_xlabel('Time relative to template center (ms)')
    ax.set_ylabel('Amplitude (HP-filtered V)')
    ax.set_title(f'Both arrival templates ({N_TEMPLATE}-ping co-add, ±{ARRIVAL_WIN_HW_MS:.1f} ms half-width)',
                 fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='upper right')

    # (2) Example CCFs (Ch A and Ch B)
    ax = axes[1]
    if cd.get('ccf_a') is not None:
        ccf_a = cd['ccf_a']; env_a = cd['env_a']
        center = len(ch_a_template) - 1
        lag_us = (np.arange(len(ccf_a)) - center) / fs * 1e6
        zoom_us = 300.0
        mask = np.abs(lag_us) <= zoom_us
        ax.plot(lag_us[mask], ccf_a[mask] / (np.max(np.abs(ccf_a)) + 1e-12),
                 '-', color='#4477cc', lw=0.6, alpha=0.7, label='Ch A CCF (normalized)')
        ax.plot(lag_us[mask], env_a[mask], '-', color='#e6444d', lw=1.0,
                 label='Ch A Hilbert envelope')
        if cd.get('ccf_b') is not None:
            ccf_b = cd['ccf_b']; env_b = cd['env_b']
            ax.plot(lag_us[mask], env_b[mask], '--', color='#88aa44', lw=1.0,
                     alpha=0.7, label='Ch B Hilbert envelope')
        ax.axvline(0, color='k', ls='--', lw=0.4, alpha=0.5, label='lag=0')
        ax.set_xlabel('Lag (μs)')
        ax.set_ylabel('CCF (normalized)')
        ax.set_title(f'Example CCFs (middle ping of chunk)  ping index {(i0+i1)//2}',
                      fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'no CCF saved for this chunk', ha='center', va='center',
                transform=ax.transAxes, fontsize=11)
        ax.set_axis_off()

    # (3) Per-chunk Δτ time series (blue thread + black open circles + error bars)
    ax = axes[2]
    t_s_chunk = (t_chunk - t_chunk[0])
    inl = inl_p_chunk
    ax.plot(t_s_chunk[inl], dtau_p_chunk[inl],
             '-', color='#1f77b4', lw=0.6, alpha=0.6, zorder=1, label='parabolic thread')
    ax.errorbar(t_s_chunk[inl], dtau_p_chunk[inl], yerr=sig_p_chunk[inl],
                 fmt='o', mfc='none', mec='k', ecolor='k', ms=3.5, capsize=1.0,
                 lw=0.4, elinewidth=0.4, alpha=0.8, zorder=2,
                 label='parabolic (open circles + δ_fit error bars)')
    ax.plot(t_s_chunk[inl_shape[i0:i1]], dtau_s_chunk[inl_shape[i0:i1]],
             '.', color='#e6444d', ms=1.5, alpha=0.4, zorder=0,
             label='shape-fit (overlay)')
    ax.axhline(0, color='k', lw=0.3, alpha=0.4)
    ax.set_xlabel(f'Time within chunk (s)   [absolute wall time {cd["t_start_s"]:.1f}–{cd["t_end_s"]:.1f} s]')
    ax.set_ylabel('Δτ = TOF − baseline (μs)')
    sig_chunk_hf = float(np.std(dtau_p_chunk[inl])) if inl.sum() > 1 else 0.0
    ax.set_title(f'Chunk {cd["chunk_idx"]+1:03d} Δτ time series   '
                  f'N={inl.sum()} pings   σ_chunk={sig_chunk_hf:.3f} μs '
                  f'({sig_chunk_hf/tof_sens*1000:.1f} mK)',
                  fontsize=10)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f'Delayed-chirp per-chunk diagnostic — {wav_basename}   '
                 f'chunk {cd["chunk_idx"]+1:03d}',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    out_png = os.path.join(chunk_plot_dir, f'chunk_{cd["chunk_idx"]+1:03d}.png')
    fig.savefig(out_png, dpi=110, bbox_inches='tight')
    plt.close(fig)


# ── Main pipeline ─────────────────────────────────────────────────────────
def analyze_delayedchirp(wav_path, out_dir=None,
                          ch_a_scale_v=CH_A_FS_V_DEFAULT,
                          ch_b_scale_v=CH_B_FS_V_DEFAULT,
                          rep_rate=REP_RATE_DEFAULT,
                          search_ms=SEARCH_MS_DEFAULT,
                          chb_delay_ms=CHB_DELAY_MS_DEFAULT,
                          per_chunk_plots=False,
                          shape_method='dechirped'):
    global SHAPE_METHOD
    SHAPE_METHOD = shape_method
    info = sf.info(wav_path)
    fs = info.samplerate
    n_total = info.frames
    print(f"\n{'='*72}")
    print(f"Delayed-chirp Analysis: {os.path.basename(wav_path)}   [shape={shape_method}]")
    print(f"{'='*72}")
    print(f"  fs = {fs} Hz    duration = {n_total/fs/60:.2f} min    channels = {info.channels}")
    print(f"  rep rate: {rep_rate:.3f} Hz    search window (Ch A rel Ch B): [{search_ms[0]:.1f}, {search_ms[1]:.1f}] ms")
    print(f"  CH2 chirp delay (within arb loop): {chb_delay_ms:.2f} ms")
    if info.channels != 2:
        raise RuntimeError("Expected stereo WAV")

    if out_dir is None:
        wav_stem = os.path.splitext(os.path.basename(wav_path))[0]
        # Append a shape-method tag so classic/dechirped runs don't collide
        subdir = f'oneway_delayedchirp_analysis_{shape_method}'
        out_dir = os.path.join(os.path.dirname(wav_path), subdir, wav_stem)
    os.makedirs(out_dir, exist_ok=True)
    print(f"  out_dir = {out_dir}")

    # ── Phase 1: build templates from first ~5 s ──────────────────────────
    seconds_needed = max(5.0, (N_TEMPLATE + 20) / rep_rate + 1.0)
    n_read = int(seconds_needed * fs)
    print(f"\n  Phase 1: building templates from first {seconds_needed:.1f} s...")
    with sf.SoundFile(wav_path, 'r') as f:
        data = f.read(n_read, dtype='int16')
    ch_a_v = data[:, 0].astype(np.float32) / 32768.0 * ch_a_scale_v
    ch_b_v = data[:, 1].astype(np.float32) / 32768.0 * ch_b_scale_v
    ch_a_hp_tpl = hp_filter(ch_a_v, fs)
    ch_b_hp_tpl = hp_filter(ch_b_v, fs)

    chb_peaks = find_chb_peaks_global(ch_b_hp_tpl, fs, rep_rate)
    if len(chb_peaks) < N_TEMPLATE:
        raise RuntimeError(f"insufficient Ch B peaks: {len(chb_peaks)} < {N_TEMPLATE}")
    print(f"    detected {len(chb_peaks)} Ch B chirp peaks")

    ch_b_template, win_samples = build_template(ch_b_hp_tpl, chb_peaks, N_TEMPLATE, fs)
    cha_times = find_cha_initial_arrivals(ch_a_hp_tpl, chb_peaks, search_ms, fs, N_TEMPLATE)
    ch_a_template, _ = build_template(ch_a_hp_tpl, cha_times, N_TEMPLATE, fs)
    cha_offset_samp = int(np.median(
        [a - b for a, b in zip(cha_times[:N_TEMPLATE], chb_peaks[:N_TEMPLATE])]
    ))
    print(f"    templates built — Ch A offset from Ch B: {cha_offset_samp/fs*1000:.3f} ms")

    R_auto_a = correlate(ch_a_template, ch_a_template, mode='full')
    R_auto_b = correlate(ch_b_template, ch_b_template, mode='full')

    # ── Phase 2: chunked processing ───────────────────────────────────────
    chunk_samples = int(CHUNK_S * fs)
    overlap_samples = int(0.5 * fs)
    per_ping = []
    per_chunk_diag = []
    total_pings = 0
    prev_peak_abs = -1e18

    print(f"\n  Phase 2: processing {n_total/fs:.1f} s in {CHUNK_S:.0f}-s chunks...")
    with sf.SoundFile(wav_path, 'r') as f:
        abs_offset = 0
        chunk_idx = 0
        carry = None
        carry_abs_offset = 0

        while abs_offset < n_total:
            f.seek(max(0, abs_offset))
            want = min(chunk_samples, n_total - abs_offset)
            data = f.read(want, dtype='int16')
            if len(data) == 0:
                break
            ch_a_v = data[:, 0].astype(np.float32) / 32768.0 * ch_a_scale_v
            ch_b_v = data[:, 1].astype(np.float32) / 32768.0 * ch_b_scale_v
            if carry is not None:
                ch_a_v = np.concatenate([carry[0], ch_a_v])
                ch_b_v = np.concatenate([carry[1], ch_b_v])
                chunk_abs_start = carry_abs_offset
            else:
                chunk_abs_start = abs_offset

            ch_a_hp = hp_filter(ch_a_v, fs)
            ch_b_hp = hp_filter(ch_b_v, fs)
            chb_peaks_local = find_chb_peaks_global(ch_b_hp, fs, rep_rate)
            chb_peaks_abs = chb_peaks_local + chunk_abs_start
            new_mask = chb_peaks_abs > prev_peak_abs + 0.5 * fs / rep_rate
            chb_peaks_new = chb_peaks_local[new_mask]
            chb_peaks_abs_new = chb_peaks_abs[new_mask]

            n_this_chunk = 0
            ping_i_start = len(per_ping)
            middle_ping_i = len(chb_peaks_new) // 2
            example_ccf_a = None; example_env_a = None
            example_ccf_b = None; example_env_b = None
            for i, (peak_loc, peak_abs) in enumerate(zip(chb_peaks_new, chb_peaks_abs_new)):
                want_ccf = per_chunk_plots and (i == middle_ping_i)
                res = analyze_ping_pair(ch_a_hp, ch_b_hp, peak_loc,
                                         ch_a_template, ch_b_template,
                                         R_auto_a, R_auto_b, fs, win_samples,
                                         cha_offset_samp, return_ccf=want_ccf)
                if res is None:
                    continue
                res['t_pulse_s'] = peak_abs / fs
                if want_ccf and 'ccf_a' in res:
                    example_ccf_a = res.pop('ccf_a')
                    example_env_a = res.pop('env_a')
                    example_ccf_b = res.pop('ccf_b')
                    example_env_b = res.pop('env_b')
                per_ping.append(res)
                prev_peak_abs = peak_abs
                n_this_chunk += 1

            if per_chunk_plots and n_this_chunk > 0:
                per_chunk_diag.append(dict(
                    chunk_idx=chunk_idx,
                    t_start_s=abs_offset / fs,
                    t_end_s=(abs_offset + want) / fs,
                    ping_i_start=ping_i_start,
                    ping_i_end=len(per_ping),
                    ccf_a=example_ccf_a, env_a=example_env_a,
                    ccf_b=example_ccf_b, env_b=example_env_b,
                ))

            total_pings += n_this_chunk
            print(f"    chunk {chunk_idx+1:3d}  [{abs_offset/fs:6.1f}-{(abs_offset+want)/fs:6.1f} s]  "
                  f"pings={n_this_chunk:>5}  total={total_pings:>6,}", flush=True)

            carry = (ch_a_v[-overlap_samples:], ch_b_v[-overlap_samples:])
            carry_abs_offset = chunk_abs_start + len(ch_a_v) - overlap_samples
            abs_offset += want
            chunk_idx += 1

    print(f"\n  processed {total_pings:,} pings")
    if total_pings == 0:
        raise RuntimeError("no pings analyzed")

    # ── Aggregate per-ping arrays ─────────────────────────────────────────
    t_pulse   = np.array([p['t_pulse_s']           for p in per_ping])
    tof_para  = np.array([p['tof_para_us']         for p in per_ping])
    tof_shape = np.array([p['tof_shape_us']        for p in per_ping])
    sig_para  = np.array([p['sigma_para_us']       for p in per_ping])
    sig_shape = np.array([p['sigma_shape_us']      for p in per_ping])
    sig_B_para  = np.array([p['sigma_B_para_us']   for p in per_ping])
    sig_B_shape = np.array([p['sigma_B_shape_us']  for p in per_ping])
    sig_diff_para  = np.array([p['sigma_diff_para_us']  for p in per_ping])
    sig_diff_shape = np.array([p['sigma_diff_shape_us'] for p in per_ping])

    baseline_para  = float(np.mean(tof_para[:N_BASELINE]))
    baseline_shape = float(np.mean(tof_shape[:N_BASELINE]))
    dtau_para  = tof_para  - baseline_para
    dtau_shape = tof_shape - baseline_shape

    # Infer L and tof_sens from acoustic TOF estimate (diff-TOF + known CH B delay)
    acoustic_tof_est_us = baseline_para + chb_delay_ms * 1000  # tof + 2ms delay offset
    L_m = acoustic_tof_est_us * 1e-6 * SOUND_SPEED_M_S
    tof_sens = L_m * DC_DT_M_S_PER_C / (SOUND_SPEED_M_S**2) * 1e6

    # MAD inliers
    def mad_inl(x, k=5.0):
        m = np.median(x); mad = 1.4826 * np.median(np.abs(x - m))
        return np.abs(x - m) < k * mad
    inl_para  = mad_inl(tof_para)
    inl_shape = mad_inl(tof_shape)

    sigma_full_para  = float(np.std(dtau_para[inl_para]))
    sigma_full_shape = float(np.std(dtau_shape[inl_shape]))
    win_1s = max(3, int(rep_rate) | 1)
    r_para_1s  = pd.Series(np.where(inl_para,  dtau_para,  np.nan)).rolling(win_1s, center=True, min_periods=win_1s//2).mean().values
    r_shape_1s = pd.Series(np.where(inl_shape, dtau_shape, np.nan)).rolling(win_1s, center=True, min_periods=win_1s//2).mean().values
    hf_p = (dtau_para  - r_para_1s )[inl_para  & np.isfinite(r_para_1s )]
    hf_s = (dtau_shape - r_shape_1s)[inl_shape & np.isfinite(r_shape_1s)]
    sigma_hf_para  = float(np.std(hf_p))
    sigma_hf_shape = float(np.std(hf_s))

    sl_para,  _, r_para_corr,  _, _ = stats.linregress(t_pulse[inl_para],  tof_para[inl_para])
    sl_shape, _, r_shape_corr, _, _ = stats.linregress(t_pulse[inl_shape], tof_shape[inl_shape])

    print(f"\n  ═══ SUMMARY ═══")
    print(f"    N pings           : {total_pings:,}")
    print(f"    duration          : {(t_pulse[-1]-t_pulse[0])/60:.2f} min")
    print(f"    baseline diff-TOF : para {baseline_para:.3f} μs  shape {baseline_shape:.3f} μs")
    print(f"    acoustic TOF est  : {acoustic_tof_est_us:.3f} μs → L ≈ {L_m*100:.1f} cm → tof_sens {tof_sens:.2f} μs/°C")
    print(f"    σ(Δτ) full-run    : para {sigma_full_para:.3f} μs ({sigma_full_para/tof_sens*1000:.1f} mK)  "
          f"shape {sigma_full_shape:.3f} μs ({sigma_full_shape/tof_sens*1000:.1f} mK)")
    print(f"    σ(Δτ) HF-only ★   : para {sigma_hf_para:.3f} μs ({sigma_hf_para/tof_sens*1000:.1f} mK/ping)  "
          f"shape {sigma_hf_shape:.3f} μs ({sigma_hf_shape/tof_sens*1000:.1f} mK/ping)")
    print(f"    LINEAR DRIFT      : para {sl_para*60:+.4f} μs/min (r²={r_para_corr**2:.4f})  "
          f"shape {sl_shape*60:+.4f} μs/min (r²={r_shape_corr**2:.4f})")

    # ── Save NPZ ──────────────────────────────────────────────────────────
    npz_path = os.path.join(out_dir, 'per_ping.npz')
    np.savez(npz_path,
              t_pulse_s=t_pulse,
              tof_para_us=tof_para, tof_shape_us=tof_shape,
              sigma_para_us=sig_para, sigma_shape_us=sig_shape,
              sigma_B_para_us=sig_B_para, sigma_B_shape_us=sig_B_shape,
              sigma_diff_para_us=sig_diff_para, sigma_diff_shape_us=sig_diff_shape,
              dtau_para_us=dtau_para, dtau_shape_us=dtau_shape,
              inl_para=inl_para, inl_shape=inl_shape,
              baseline_para_us=baseline_para, baseline_shape_us=baseline_shape,
              rep_rate=rep_rate, tof_sens_us_per_deg=tof_sens,
              drift_para_us_per_min=sl_para*60, drift_shape_us_per_min=sl_shape*60,
              ch_a_template=ch_a_template, ch_b_template=ch_b_template,
              cha_offset_samp=cha_offset_samp)
    print(f"  wrote {npz_path}")
    np.savez(os.path.join(out_dir, 'templates.npz'),
              ch_a_template=ch_a_template, ch_b_template=ch_b_template,
              cha_offset_samp=cha_offset_samp, win_samples=win_samples,
              fs=fs, rep_rate=rep_rate)

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_summary(t_pulse, tof_para, tof_shape, dtau_para, dtau_shape,
                  sig_para, sig_shape, inl_para, inl_shape,
                  sigma_full_para, sigma_full_shape, tof_sens,
                  out_dir, os.path.basename(wav_path), rep_rate)
    plot_uncertainty_histograms(sig_B_para, sig_B_shape, sig_para, sig_shape,
                                  sig_diff_para, sig_diff_shape, inl_para, inl_shape,
                                  out_dir, os.path.basename(wav_path))

    if per_chunk_plots and len(per_chunk_diag) > 0:
        chunk_plot_dir = os.path.join(out_dir, 'chunk_plots')
        os.makedirs(chunk_plot_dir, exist_ok=True)
        print(f"\n  generating {len(per_chunk_diag)} per-chunk 3-panel plots → {chunk_plot_dir}")
        for cd in per_chunk_diag:
            plot_per_chunk(cd, ch_a_template, ch_b_template, t_pulse,
                            tof_para, tof_shape, dtau_para, dtau_shape,
                            sig_para, sig_shape, inl_para, inl_shape, fs,
                            chunk_plot_dir, os.path.basename(wav_path), tof_sens)
        print(f"  wrote {len(per_chunk_diag)} per-chunk PNGs")

    return dict(sigma_hf_para=sigma_hf_para, sigma_hf_shape=sigma_hf_shape,
                 drift_para=sl_para*60, drift_shape=sl_shape*60,
                 out_dir=out_dir)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('wav', help='stereo WAV with chirp on Ch A (acoustic) + delayed chirp on Ch B')
    p.add_argument('--out-dir', default=None)
    p.add_argument('--ch-a-fs-v', type=float, default=CH_A_FS_V_DEFAULT)
    p.add_argument('--ch-b-fs-v', type=float, default=CH_B_FS_V_DEFAULT)
    p.add_argument('--rep-rate', type=float, default=REP_RATE_DEFAULT)
    p.add_argument('--search-ms', type=float, nargs=2, default=list(SEARCH_MS_DEFAULT),
                    metavar=('LO', 'HI'))
    p.add_argument('--chb-delay-ms', type=float, default=CHB_DELAY_MS_DEFAULT)
    p.add_argument('--per-chunk-plots', action='store_true', default=False)
    p.add_argument('--shape-method', choices=['dechirped', 'classic'], default='dechirped',
                    help='LS shape-fit method: dechirped (default; narrowband baseband fit) '
                         'or classic (full oscillating CCF fit against template autocorrelation)')
    args = p.parse_args()
    SHAPE_METHOD = args.shape_method   # module-global toggle read by analyze_ping_pair
    print(f"  shape-fit method: {SHAPE_METHOD}")
    analyze_delayedchirp(args.wav, out_dir=args.out_dir,
                          ch_a_scale_v=args.ch_a_fs_v,
                          ch_b_scale_v=args.ch_b_fs_v,
                          rep_rate=args.rep_rate,
                          search_ms=tuple(args.search_ms),
                          chb_delay_ms=args.chb_delay_ms,
                          per_chunk_plots=args.per_chunk_plots,
                          shape_method=args.shape_method)
