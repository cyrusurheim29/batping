#!/usr/bin/env python3
"""Δτ residual waterfall analysis for one-way sync-pulse recordings.

Draws on the heat/fan waterfall characterization code pattern:
  - load per-ping Δτ time series from per_ping(_combined).npz
  - MAD-clip outliers, remove slow drift via 1-s rolling mean
  - compute PSD in 5-s windows (Welch's method), stack as time×freq image
  - also compute median PSD across all windows and HF-band (15-25 Hz)
    power time series

Produces a 2×2 panel figure:
  (1) PSD waterfall (time vs frequency, colored by PSD dB)
  (2) Median PSD spectrum across all windows
  (3) HF-band (0.5-20 Hz broadband turbulence) rolling power
  (4) 0.3-5 Hz thermal-convection-band rolling power

Usage:
    python3 oneway_delta_tau_waterfall.py <recording_dir>
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch

# ── Configuration ────────────────────────────────────────────────────────
WIN_S         = 5.0      # window duration per PSD estimate (seconds)
HF_BAND_HZ    = (0.5, 20.0)   # broadband acoustic turbulence
CONV_BAND_HZ  = (0.3, 5.0)    # slow thermal convection
MAD_CLIP_K    = 5.0      # inlier gate
SOUND_SPEED_M_S = 343.0
DC_DT_M_S_PER_C = 0.606


def load_npz(rec_dir, subdir='oneway_analysis'):
    """Load either per_ping_combined.npz or the single-segment per_ping.npz."""
    combined = os.path.join(rec_dir, subdir, 'per_ping_combined.npz')
    if os.path.exists(combined):
        print(f"  loading {combined}")
        return np.load(combined), combined
    import glob
    cands = sorted(glob.glob(os.path.join(rec_dir, subdir, '*', 'per_ping.npz')))
    if not cands:
        sys.exit(f"no per_ping NPZ found in {rec_dir}/{subdir}/")
    print(f"  loading {cands[0]}  (combined NPZ not present; using single-segment)")
    return np.load(cands[0]), cands[0]


def analyze(rec_dir, subdir='oneway_analysis', npz_path=None, out_dir_override=None):
    if npz_path is not None:
        d = np.load(npz_path)
        print(f"  loading {npz_path}")
    else:
        d, npz_path = load_npz(rec_dir, subdir)
    t_s        = d['t_pulse_s']
    tof        = d['tof_para_us']
    inl        = d['inl_para']
    dtau_us    = tof - float(np.mean(tof[:100]))
    if 'rep_rate' in d.files:
        rep_rate = float(d['rep_rate'])
    else:
        # auto-infer from median per-ping spacing
        rep_rate = 1.0 / float(np.median(np.diff(t_s)))
    print(f"  N pings = {len(t_s):,}   rep rate = {rep_rate:.3f} Hz   "
          f"duration = {(t_s[-1]-t_s[0])/60:.2f} min")

    # TOF sensitivity (L-dependent)
    L_m = float(np.mean(tof[:100])) * 1e-6 * SOUND_SPEED_M_S
    tof_sens = L_m * DC_DT_M_S_PER_C / (SOUND_SPEED_M_S**2) * 1e6
    print(f"  inferred one-way L = {L_m*100:.1f} cm   dτ/dT = {tof_sens:.2f} μs/°C")

    # Resample Δτ onto a uniform grid at rep_rate (handles any small gaps
    # from dropped pings or non-stationary rep rate).  Use linear interp.
    t_uniform = np.arange(t_s[0], t_s[-1], 1.0 / rep_rate)
    # MAD-clip the Δτ, interpolate over outliers for PSD estimation
    med = float(np.median(dtau_us))
    mad = float(np.median(np.abs(dtau_us - med))) * 1.4826
    inlier_mask = inl & (np.abs(dtau_us - med) <= MAD_CLIP_K * mad)
    dtau_uniform = np.interp(t_uniform, t_s[inlier_mask], dtau_us[inlier_mask])
    # Remove slow drift via 1-s rolling mean so PSD is of HF residual
    win_1s = max(3, int(rep_rate) | 1)
    dtau_ser = pd.Series(dtau_uniform)
    hf_series = (dtau_ser - dtau_ser.rolling(win_1s, center=True,
                                              min_periods=win_1s//2).mean()).values

    # ── PSD waterfall: 5-s windows ────────────────────────────────────────
    win_n = int(WIN_S * rep_rate)
    n_windows = len(hf_series) // win_n
    psds, win_t_centers = [], []
    for k in range(n_windows):
        s0, s1 = k * win_n, (k + 1) * win_n
        seg = hf_series[s0:s1]
        if not np.all(np.isfinite(seg)):
            continue
        # Welch with 256-sample sub-windows → freq res ~rep_rate/256
        f, pxx = welch(seg, fs=rep_rate, nperseg=min(256, len(seg)),
                        detrend='linear')
        psds.append(pxx)
        win_t_centers.append((s0 + s1) / 2 / rep_rate)
    freqs = f
    psds  = np.stack(psds, axis=0)
    win_t_min = np.array(win_t_centers) / 60.0

    # Effective bands and plot limit: clamp to Nyquist = rep_rate/2
    nyquist_hz = rep_rate / 2.0
    hf_hi = min(HF_BAND_HZ[1],   nyquist_hz)
    hf_lo = min(HF_BAND_HZ[0],   nyquist_hz)
    cv_hi = min(CONV_BAND_HZ[1], nyquist_hz)
    cv_lo = min(CONV_BAND_HZ[0], nyquist_hz)
    freq_xlim_hi = nyquist_hz

    psd_db = 10 * np.log10(psds + 1e-20)
    median_psd    = np.median(psds, axis=0)
    median_psd_db = 10 * np.log10(median_psd + 1e-20)

    # Band-integrated power time series
    hf_mask   = (freqs >= hf_lo) & (freqs <= hf_hi)
    conv_mask = (freqs >= cv_lo) & (freqs <= cv_hi)
    hf_pow_t   = np.trapz(psds[:, hf_mask],   freqs[hf_mask],   axis=1)
    conv_pow_t = np.trapz(psds[:, conv_mask], freqs[conv_mask], axis=1)

    # ── Plot ──────────────────────────────────────────────────────────────
    out_dir = out_dir_override if out_dir_override else os.path.join(rec_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, 'delta_tau_waterfall.png')

    fig, axes = plt.subplots(2, 2, figsize=(17, 11))

    # (1) Waterfall PSD(time, freq) in dB — capped at Nyquist
    ax = axes[0, 0]
    p5, p95 = np.percentile(psd_db, [5, 95])
    im = ax.imshow(psd_db, aspect='auto', origin='lower',
                    extent=[freqs[0], freq_xlim_hi, win_t_min[0], win_t_min[-1]],
                    cmap='inferno', vmin=p5, vmax=p95)
    for b_lo, b_hi, c, lbl in [(hf_lo, hf_hi, 'cyan',   f'HF turb ({hf_lo:.1f}-{hf_hi:.1f} Hz)'),
                                (cv_lo, cv_hi, 'yellow', f'convection ({cv_lo:.1f}-{cv_hi:.1f} Hz)')]:
        for x in (b_lo, b_hi):
            ax.axvline(x, color=c, ls='--', lw=0.6, alpha=0.6)
    ax.set_xlim(freqs[0], freq_xlim_hi)
    ax.set_xlabel(f'Frequency (Hz)   [Nyquist = {nyquist_hz:.2f} Hz at rep {rep_rate:.2f} Hz]')
    ax.set_ylabel('Time (min)')
    ax.set_title(f'Δτ residual PSD waterfall  ({WIN_S:.0f}-s windows, 1-s-detrended)')
    plt.colorbar(im, ax=ax, label='PSD (dB μs²/Hz)')

    # (2) Median PSD spectrum — capped at Nyquist
    ax = axes[0, 1]
    ax.loglog(freqs, median_psd, '-', color='#1f77b4', lw=1.4)
    ax.axvspan(hf_lo, hf_hi, alpha=0.08, color='#4477cc',
                label=f'HF turb {hf_lo:.1f}-{hf_hi:.1f} Hz')
    ax.axvspan(cv_lo, cv_hi, alpha=0.12, color='#e6444d',
                label=f'convection {cv_lo:.1f}-{cv_hi:.1f} Hz')
    ax.set_xlim(freqs[1], freq_xlim_hi)      # drop DC bin
    ax.set_xlabel(f'Frequency (Hz)   [Nyquist = {nyquist_hz:.2f} Hz]')
    ax.set_ylabel('median PSD (μs²/Hz)')
    ax.set_title('Median PSD across all windows')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9, loc='upper right')

    # (3) HF-band power vs time
    ax = axes[1, 0]
    ax.plot(win_t_min, hf_pow_t, '-', color='#4477cc', lw=1.2)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel(f'HF band power ({hf_lo:.1f}-{hf_hi:.1f} Hz) (μs²)')
    ax.set_title(f'Broadband acoustic turbulence vs time  '
                  f'(mean {np.mean(hf_pow_t):.4f} μs²)')
    ax.grid(alpha=0.3)

    # (4) Convection-band power vs time
    ax = axes[1, 1]
    ax.plot(win_t_min, conv_pow_t, '-', color='#e6444d', lw=1.2)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel(f'Convection band power ({cv_lo:.1f}-{cv_hi:.1f} Hz) (μs²)')
    ax.set_title(f'Thermal convection signature vs time  '
                  f'(mean {np.mean(conv_pow_t):.4f} μs²)')
    ax.grid(alpha=0.3)

    fig.suptitle(f'Δτ waterfall — {os.path.basename(rec_dir)}   '
                  f'L = {L_m*100:.1f} cm,  dτ/dT = {tof_sens:.2f} μs/°C,  '
                  f'rep = {rep_rate:.2f} Hz,  N = {len(t_s):,} pings',
                  fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_png}")

    # Summary print
    print(f"\n═══ waterfall summary ═══")
    print(f"  windows analyzed      : {len(psds)} × {WIN_S:.0f}s = {len(psds)*WIN_S:.0f}s audio")
    print(f"  median PSD peak at    : {freqs[np.argmax(median_psd)]:.2f} Hz "
          f"({10*np.log10(np.max(median_psd)):.1f} dB μs²/Hz)")
    print(f"  HF band ({HF_BAND_HZ[0]}-{HF_BAND_HZ[1]} Hz) power: "
          f"median {np.median(hf_pow_t):.4f} μs²   "
          f"({np.median(hf_pow_t)**0.5/tof_sens*1000:.1f} mK rms-equivalent)")
    print(f"  conv band ({CONV_BAND_HZ[0]}-{CONV_BAND_HZ[1]} Hz) power: "
          f"median {np.median(conv_pow_t):.4f} μs²   "
          f"({np.median(conv_pow_t)**0.5/tof_sens*1000:.1f} mK rms-equivalent)")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit("usage: oneway_delta_tau_waterfall.py <recording_dir> "
                 "[--subdir oneway_delayedchirp_analysis] [--per-segment]")
    rec = sys.argv[1]
    sub = 'oneway_analysis'
    if '--subdir' in sys.argv:
        sub = sys.argv[sys.argv.index('--subdir') + 1]
    per_segment = '--per-segment' in sys.argv

    # Combined waterfall (always, if combined NPZ is present)
    analyze(rec, sub)

    # Per-segment waterfalls — write PNG into each per-segment subdir
    if per_segment:
        import glob as _g
        seg_npzs = sorted(_g.glob(os.path.join(rec, sub, '*_seg*', 'per_ping.npz')))
        if not seg_npzs:
            print("  no per-segment NPZs for --per-segment waterfalls")
        for p in seg_npzs:
            seg_dir_abs = os.path.dirname(p)
            seg_tag = os.path.basename(seg_dir_abs)
            seg_num = seg_tag.split('_seg')[-1][:2] if '_seg' in seg_tag else '??'
            print(f"\n═══ Per-segment waterfall — seg{seg_num} ═══")
            analyze(rec, sub, npz_path=p, out_dir_override=seg_dir_abs)
