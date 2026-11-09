#!/usr/bin/env python3
"""Compare batping-inferred ΔT (from diff-TOF drift) against the independent
TMP36-on-Ch-C temperature reading over the same recording window.

Both ΔT traces are relative to their own first-100-samples baseline, so we're
comparing SHAPES of thermal drift, not absolute calibration.

Usage:
    python3 oneway_batping_vs_tmp36.py <recording_dir>
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TOF_SENS_US_PER_DEG = 23.2   # dτ/dT at 4.5 m one-way (NEGATIVE — warmer air = shorter τ)
# Sign convention: TOF DECREASES when temp INCREASES.  So ΔT_batping = -Δτ / sensitivity.

if len(sys.argv) < 2:
    sys.exit("usage: oneway_batping_vs_tmp36.py <recording_dir> [--subdir oneway_delayedchirp_analysis]")
REC = sys.argv[1]
SUBDIR = 'oneway_analysis'
if '--subdir' in sys.argv:
    SUBDIR = sys.argv[sys.argv.index('--subdir') + 1]

npz = os.path.join(REC, SUBDIR, 'per_ping_combined.npz')
if not os.path.exists(npz):
    import glob
    cands = sorted(glob.glob(os.path.join(REC, SUBDIR, '*', 'per_ping.npz')))
    if cands:
        npz = cands[0]
        print(f"per_ping_combined.npz not found; falling back to {npz}")
    else:
        sys.exit(f"no per_ping NPZ found in {REC}/{SUBDIR}/")
tmp_csv = None
for fn in os.listdir(REC):
    if fn.endswith('_tmp36c_perping.csv'):
        tmp_csv = os.path.join(REC, fn); break
if tmp_csv is None:
    sys.exit("no *_tmp36c_perping.csv found — did you record with --ch-c-mode tmp36?")

d = np.load(npz)
t_bat  = d['t_pulse_s']
tof    = d['tof_para_us']
sig_p  = d['sigma_para_us']
inl    = d['inl_para']
tof_sens = float(d['tof_sens_us_per_deg']) if 'tof_sens_us_per_deg' in d.files else TOF_SENS_US_PER_DEG

# Batping ΔT: baseline = mean of first 100 pings; ΔT = -(TOF - baseline) / sensitivity
baseline_tof = float(np.mean(tof[:100]))
dT_bat = -(tof - baseline_tof) / tof_sens

tdf = pd.read_csv(tmp_csv)
print(f"TMP36 CSV columns: {list(tdf.columns)}")
print(f"TMP36 rows: {len(tdf):,}")
t_tmp = tdf['audio_s'].values
T_tmp = tdf['tmp36c_temp_C'].values

# Batping ΔT
baseline_tmp = float(np.mean(T_tmp[:100]))
dT_tmp = T_tmp - baseline_tmp

# ── Stats ────────────────────────────────────────────────────────────────
print(f"\n═══ Batping vs TMP36 (Ch C) comparison ═══")
print(f"  duration           : batping {(t_bat[-1]-t_bat[0])/60:.2f} min  /  "
      f"tmp36 {(t_tmp[-1]-t_tmp[0])/60:.2f} min")
print(f"  N points           : batping {len(t_bat):,}  /  tmp36 {len(t_tmp):,}")
print(f"  batping ΔT range   : {dT_bat.min():+.3f} to {dT_bat.max():+.3f} °C  "
      f"(net {dT_bat[-1]-dT_bat[0]:+.3f} °C)")
print(f"  tmp36   ΔT range   : {dT_tmp.min():+.3f} to {dT_tmp.max():+.3f} °C  "
      f"(net {dT_tmp[-1]-dT_tmp[0]:+.3f} °C)")

# Interpolate TMP36 onto batping time base for direct comparison
dT_tmp_on_bat = np.interp(t_bat, t_tmp, dT_tmp, left=np.nan, right=np.nan)
valid = np.isfinite(dT_tmp_on_bat) & inl
if valid.sum() > 100:
    # Pearson correlation between batping ΔT and TMP36 ΔT
    corr = float(np.corrcoef(dT_bat[valid], dT_tmp_on_bat[valid])[0, 1])
    # Residual after linear scale + offset regression:
    #     dT_bat ≈ a * dT_tmp + b
    A = np.column_stack([dT_tmp_on_bat[valid], np.ones(valid.sum())])
    coef, *_ = np.linalg.lstsq(A, dT_bat[valid], rcond=None)
    a, b = float(coef[0]), float(coef[1])
    resid = dT_bat[valid] - (a * dT_tmp_on_bat[valid] + b)
    resid_rms = float(np.sqrt(np.mean(resid**2)))
    print(f"  Pearson r          : {corr:.4f}")
    print(f"  LS scale factor    : batping = {a:.3f} × tmp36 + {b:+.3f}   "
          f"(if perfectly matched: a=1.000, b=0.000)")
    print(f"  residual RMS       : {resid_rms:.4f} °C = {resid_rms*1000:.0f} mK")
else:
    corr = np.nan; a = np.nan; b = np.nan; resid_rms = np.nan

# ── Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(17, 12), gridspec_kw={'height_ratios':[1.2, 1, 1]})

# (1) Overlay: batping ΔT (blue) + TMP36 ΔT (red)
ax = axes[0]
t_bat_min = t_bat / 60.0
t_tmp_min = t_tmp / 60.0
ax.plot(t_bat_min[inl], dT_bat[inl], '.', ms=1.0, alpha=0.5, color='#1f77b4',
         label=f'batping (from TOF, dτ/dT = {tof_sens} μs/°C at 4.5m one-way)')
ax.plot(t_tmp_min, dT_tmp, '-', lw=1.8, color='#e6444d',
         label=f'TMP36 (Ch C, mid-path sensor)')
ax.axhline(0, color='k', lw=0.4, alpha=0.4)
ax.set_xlabel('Time (min)'); ax.set_ylabel('ΔT (°C)   [relative to first 100 samples]')
ax.set_title(f'Overlay: batping-inferred ΔT vs. TMP36-measured ΔT   '
             f'Pearson r = {corr:.4f}   scale = {a:.3f}   RMS residual = {resid_rms*1000:.0f} mK',
             fontsize=11, fontweight='bold')
ax.grid(alpha=0.3); ax.legend(loc='lower left', fontsize=10)

# (2) Residual (batping - a*tmp36 - b)
ax = axes[1]
if valid.sum() > 100:
    ax.plot(t_bat_min[valid], resid, '.', ms=1.0, alpha=0.4, color='#666')
    ax.axhline(0, color='k', lw=0.4, alpha=0.4)
    ax.axhline(+resid_rms, color='r', lw=0.8, ls='--', alpha=0.6,
                label=f'±RMS = {resid_rms*1000:.0f} mK')
    ax.axhline(-resid_rms, color='r', lw=0.8, ls='--', alpha=0.6)
    ax.set_xlabel('Time (min)'); ax.set_ylabel('Residual: batping ΔT − a·TMP36 − b   (°C)')
    ax.set_title(f'Residual after best linear scaling  (a={a:.3f}, b={b:+.3f} °C)', fontsize=10)
    ax.grid(alpha=0.3); ax.legend(loc='upper right', fontsize=9)
else:
    ax.text(0.5, 0.5, 'insufficient overlap for residual analysis',
             ha='center', va='center', transform=ax.transAxes)
    ax.set_axis_off()

# (3) Batping ΔT vs TMP36 ΔT scatter
ax = axes[2]
if valid.sum() > 100:
    # Subsample scatter for readability
    step = max(1, int(valid.sum() // 20000))
    idx = np.where(valid)[0][::step]
    ax.scatter(dT_tmp_on_bat[idx], dT_bat[idx], s=1.0, alpha=0.4, color='#4477cc')
    # Best-fit line + identity line
    xs = np.linspace(dT_tmp_on_bat[valid].min(), dT_tmp_on_bat[valid].max(), 100)
    ax.plot(xs, a * xs + b, '-', color='#e6444d', lw=1.5,
             label=f'best fit: y = {a:.3f}x + {b:+.3f}')
    ax.plot(xs, xs, '--', color='k', lw=0.8, alpha=0.6, label='identity (y = x)')
    ax.set_xlabel('TMP36 ΔT (°C)')
    ax.set_ylabel('Batping ΔT (°C)')
    ax.set_title(f'Scatter: batping ΔT vs. TMP36 ΔT   (Pearson r = {corr:.4f})', fontsize=10)
    ax.grid(alpha=0.3); ax.legend(loc='lower right', fontsize=9)
    ax.set_aspect('equal', 'box')
else:
    ax.set_axis_off()

fig.suptitle(f'BATPING-inferred temperature vs. TMP36 reference — '
             f'{os.path.basename(REC)}',
             fontsize=13, fontweight='bold')
fig.tight_layout()
out_png = os.path.join(REC, SUBDIR, 'batping_vs_tmp36.png')
fig.savefig(out_png, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f"\nwrote {out_png}")
