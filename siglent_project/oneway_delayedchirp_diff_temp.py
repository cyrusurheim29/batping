#!/usr/bin/env python3
"""Diff-temperature + diff-TOF time series.

Default: plots the combined-segments NPZ (one 150-min figure).
With --per-segment: ALSO plots each 30-min segment NPZ individually.

Usage:
    python3 oneway_delayedchirp_diff_temp.py <recording_dir> [--per-segment]
"""
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

if len(sys.argv) < 2:
    sys.exit("usage: oneway_delayedchirp_diff_temp.py <recording_dir> "
             "[--per-segment] [--subdir oneway_delayedchirp_analysis_classic]")
REC = sys.argv[1]
SUBDIR = 'oneway_delayedchirp_analysis'
if '--subdir' in sys.argv:
    SUBDIR = sys.argv[sys.argv.index('--subdir') + 1]
PER_SEGMENT = '--per-segment' in sys.argv

# ── Discover NPZs to plot ─────────────────────────────────────────────────
jobs = []
combined = os.path.join(REC, SUBDIR, 'per_ping_combined.npz')
if os.path.exists(combined):
    jobs.append(('combined 150-min', combined, 'diff_temp_timeseries.png'))
seg_npzs = sorted(glob.glob(os.path.join(REC, SUBDIR, '*_seg*', 'per_ping.npz')))
# Fallback: any per-WAV subdir, not just _seg* ones (handles single-WAV recordings)
all_npzs = sorted(glob.glob(os.path.join(REC, SUBDIR, '*', 'per_ping.npz')))
if PER_SEGMENT and seg_npzs:
    for p in seg_npzs:
        tag = os.path.basename(os.path.dirname(p))
        seg_num = tag.split('_seg')[-1][:2] if '_seg' in tag else '00'
        jobs.append((f'segment {seg_num}', p, f'diff_temp_timeseries_seg{seg_num}.png'))
elif not jobs:
    if all_npzs:
        jobs.append(('single', all_npzs[0], 'diff_temp_timeseries.png'))
    else:
        sys.exit(f"no per_ping NPZ in {REC}/{SUBDIR}/")

print(f"will plot {len(jobs)} figure(s):")
for tag, p, _ in jobs:
    print(f"  {tag}: {p}")


def make_plot(NPZ, title_tag, out_name):
    print(f"\n──── {title_tag} ────")
    d = np.load(NPZ)
    t_s        = d['t_pulse_s']
    tof_para   = d['tof_para_us']
    tof_shape  = d['tof_shape_us']
    sig_para   = d['sigma_para_us']
    sig_shape  = d['sigma_shape_us']
    inl_para   = d['inl_para']
    inl_shape  = d['inl_shape']
    tof_sens   = float(d['tof_sens_us_per_deg']) if 'tof_sens_us_per_deg' in d.files else 48.0

    baseline_para  = float(np.mean(tof_para [inl_para ][:100]))
    baseline_shape = float(np.mean(tof_shape[inl_shape][:100]))
    dtau_para_us   = tof_para  - baseline_para
    dtau_shape_us  = tof_shape - baseline_shape
    dT_para_mK     = -dtau_para_us  / tof_sens * 1000.0
    dT_shape_mK    = -dtau_shape_us / tof_sens * 1000.0
    sig_para_mK    = sig_para  / tof_sens * 1000.0
    sig_shape_mK   = sig_shape / tof_sens * 1000.0
    t_min = t_s / 60.0

    sl_p, i_p, r_p, _, _ = stats.linregress(t_s[inl_para],  dT_para_mK [inl_para ])
    sl_s, i_s, r_s, _, _ = stats.linregress(t_s[inl_shape], dT_shape_mK[inl_shape])
    drift_p = sl_p * 60.0
    drift_s = sl_s * 60.0

    rep = 1.0 / float(np.median(np.diff(t_s)))
    win_1s = max(3, int(rep) | 1)
    r_para  = pd.Series(np.where(inl_para,  dT_para_mK,  np.nan)).rolling(win_1s, center=True, min_periods=win_1s//2).mean().values
    r_shape = pd.Series(np.where(inl_shape, dT_shape_mK, np.nan)).rolling(win_1s, center=True, min_periods=win_1s//2).mean().values
    sigma_hf_para  = float(np.nanstd(dT_para_mK [inl_para ] - r_para [inl_para ]))
    sigma_hf_shape = float(np.nanstd(dT_shape_mK[inl_shape] - r_shape[inl_shape]))
    sigma_full_para  = float(np.nanstd(dT_para_mK [inl_para ]))
    sigma_full_shape = float(np.nanstd(dT_shape_mK[inl_shape]))

    fig, axes = plt.subplots(2, 1, figsize=(18, 11), sharex=True)
    for ax, label, dT_mK, sig_mK, inl, drift, sigma_hf, sigma_full, sl, ic, r2, color in [
        (axes[0], 'parabolic', dT_para_mK,  sig_para_mK,  inl_para,  drift_p, sigma_hf_para,  sigma_full_para,  sl_p, i_p, r_p**2, '#1f77b4'),
        (axes[1], 'shape-fit', dT_shape_mK, sig_shape_mK, inl_shape, drift_s, sigma_hf_shape, sigma_full_shape, sl_s, i_s, r_s**2, '#e6444d'),
    ]:
        ax.plot(t_min[inl], dT_mK[inl], '-', color=color, lw=0.3, alpha=0.5,
                zorder=1, label=f'{label} thread')
        ax.errorbar(t_min[inl], dT_mK[inl], yerr=sig_mK[inl],
                    fmt='o', mfc='none', mec='k', ecolor='k',
                    ms=1.0, mew=0.2, elinewidth=0.15, alpha=0.5, zorder=2,
                    label=f'per-ping ({inl.sum():,} pings, δ_fit error bars)')
        rolling = pd.Series(np.where(inl, dT_mK, np.nan)).rolling(win_1s, center=True, min_periods=win_1s//2).mean().values
        ax.plot(t_min[inl], rolling[inl], color='k', lw=0.7, alpha=0.7, zorder=3,
                label='1-s rolling mean')
        tfit = np.array([t_s.min(), t_s.max()])
        ax.plot(tfit/60, sl*tfit + ic, '--', color='k', lw=0.8, alpha=0.5,
                label=f'linear fit: {drift:+.3f} mK/min (r²={r2:.3f})')
        ax.axhline(0, color='k', lw=0.4, alpha=0.4)
        ax.set_ylabel(f'ΔT (mK)  —  {label}', fontsize=11)
        net_mK = dT_mK[inl][-1] - dT_mK[inl][0]
        ax.set_title(f'{label}: σ_HF = {sigma_hf:.2f} mK/ping,  σ_full-run = {sigma_full:.1f} mK,  '
                     f'drift = {drift:+.3f} mK/min,  net = {net_mK:+.1f} mK',
                     fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        ax2 = ax.twinx()
        ylo, yhi = ax.get_ylim()
        ax2.set_ylim(ylo/1000 * -tof_sens, yhi/1000 * -tof_sens)
        ax2.set_ylabel('Δτ (μs)', fontsize=9, color='#666')
        ax2.tick_params(colors='#666')

    axes[-1].set_xlabel('Time (min)', fontsize=12)
    fig.suptitle(f'Round-trip 8 m delayed-chirp — ΔT from diff-TOF   [{title_tag}]\n'
                 f'baseline TOF = {baseline_para:.1f} μs,  tof_sens = {tof_sens:.2f} μs/°C   '
                 f'(iterative-xcorr template)   —   {os.path.basename(REC)}',
                 fontsize=12, fontweight='bold', y=0.995)
    fig.tight_layout()
    out_png = os.path.join(REC, SUBDIR, out_name)
    fig.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_png}")
    print(f"  N pings    : para {inl_para.sum():,}  shape {inl_shape.sum():,}")
    print(f"  duration   : {(t_s[-1]-t_s[0])/60:.2f} min")
    print(f"  tof_sens   : {tof_sens:.2f} μs/°C")
    print(f"  σ_HF para  : {sigma_hf_para :.2f} mK/ping   σ_full: {sigma_full_para :.1f} mK   drift: {drift_p:+.3f} mK/min (r²={r_p**2:.3f})")
    print(f"  σ_HF shape : {sigma_hf_shape:.2f} mK/ping   σ_full: {sigma_full_shape:.1f} mK   drift: {drift_s:+.3f} mK/min (r²={r_s**2:.3f})")


for tag, npz, out in jobs:
    make_plot(npz, tag, out)
