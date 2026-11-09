#!/usr/bin/env python3
"""Salvage the noisy chain_tconnector_careful NPZ by assuming a two-state
mixture model at each frequency:
    k successful passes at V_real,  (n-k) failed passes at 0
    mean = k*V/n,  std = V * sqrt(k*(n-k))/n
    →  V_real = mean + std**2 / mean
This recovers the true amplitude from each (mean, std) pair without needing
the raw per-pass measurements.

Usage:
    python3 salvage_chain_tconnector.py <path_to_noisy_npz> \\
        [--out <path_to_salvaged_npz>]
"""
import os, sys, argparse
import numpy as np
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(description=__doc__.strip().split('\n')[0])
ap.add_argument('src_npz', help='noisy characterization NPZ')
ap.add_argument('--out', default=None, help='output NPZ path (default: src with _salvaged suffix)')
args = ap.parse_args()

d = np.load(args.src_npz)
f    = d['freqs'].astype(float)
mean = d['amp_mV'].astype(float)
std  = d['amp_std_mV'].astype(float) if 'amp_std_mV' in d.files else np.zeros_like(mean)
hits = d['n_hits_per_freq'] if 'n_hits_per_freq' in d.files else np.full(len(f), 4)

# Reconstruct V_real.  When std == 0 (only 1 pass, or all identical), mean itself
# is already the estimate (no deficit to correct).
V_real = mean + np.where(mean > 0, std**2 / np.clip(mean, 1e-9, None), 0.0)
# Fraction of passes that succeeded, inferred from std/mean
#   std/mean = sqrt((n-k)/k)  →  k/n = 1 / (1 + (std/mean)^2)
rel = np.clip(std / np.clip(mean, 1e-9, None), 0, 10)
frac_success = 1.0 / (1.0 + rel**2)

# Recompute H_mag + smooth
amp_dB = 20 * np.log10(V_real + 1e-9)
mask = f > 25_000
peak_ref = float(np.max(amp_dB[mask]))
H_mag = amp_dB - peak_ref
win = min(71, len(H_mag) | 1)
H_smooth = savgol_filter(H_mag, window_length=win, polyorder=3)

# Peak + BW
peak_i = int(np.argmax(H_smooth))
peak_f = f[peak_i]
H_norm = H_smooth - H_smooth[peak_i]
def _bw(thr):
    sel = f[(f > 35_000) & (f < 50_000) & (H_norm >= thr)]
    return (sel.min(), sel.max()) if len(sel) else (None, None)
f3lo, f3hi = _bw(-3); f6lo, f6hi = _bw(-6)

print(f"Loaded: {args.src_npz}")
print(f"  N freqs           : {len(f)}")
print(f"  mean n_hits/freq  : {np.mean(hits):.1f}")
print(f"  median frac_success: {np.median(frac_success):.2f}  (0.5 = half the passes fired at a bad phase)")
print()
print(f"After salvage (V_real = mean + std²/mean):")
print(f"  Peak        : {peak_f/1e3:.2f} kHz")
if f3lo: print(f"  -3 dB BW    : {f3lo/1e3:.2f}–{f3hi/1e3:.2f} kHz   ({int(f3hi-f3lo)} Hz)")
if f6lo: print(f"  -6 dB BW    : {f6lo/1e3:.2f}–{f6hi/1e3:.2f} kHz   ({int(f6hi-f6lo)} Hz)")

out_npz = args.out or args.src_npz.replace('.npz', '_salvaged.npz')
np.savez(out_npz,
         freqs=f, amp_mV=V_real, H_mag=H_mag, H_smooth=H_smooth,
         frac_success=frac_success, n_hits_per_freq=hits,
         src=args.src_npz, salvage_method='V_real = mean + std^2/mean')
print(f"\nwrote {out_npz}")

# Side-by-side plot: raw vs salvaged
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# (1) amplitudes
ax = axes[0]
ax.plot(f/1e3, mean, '.', color='#888', ms=3, alpha=0.5, label='raw (noisy mean)')
ax.plot(f/1e3, V_real, '.', color='#e6444d', ms=3, alpha=0.7, label='salvaged V_real')
ax.set_ylabel('Amplitude (mV)')
ax.set_title(f'Chain salvage — {os.path.basename(args.src_npz)}')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# (2) H(f) in dB
ax = axes[1]
raw_H = 20*np.log10(mean+1e-9) - peak_ref
ax.plot(f/1e3, raw_H,    '-', color='#888',    lw=0.4, alpha=0.5, label='raw H (noisy)')
ax.plot(f/1e3, H_mag,    '.', color='#4477cc', ms=2, alpha=0.5, label='salvaged H_mag')
ax.plot(f/1e3, H_smooth, '-', color='#e6444d', lw=1.5,          label=f'SG-smoothed (w={win})')
ax.axvspan(37, 43, color='green', alpha=0.08, label='chirp band 37-43 kHz')
ax.axhline(-3, color='k', ls=':', lw=0.5)
ax.axhline(-6, color='k', ls=':', lw=0.5)
ax.set_xlabel('Frequency (kHz)')
ax.set_ylabel('H (dB)')
ax.set_ylim(-30, 5)
ax.set_xlim(f.min()/1e3, f.max()/1e3)
ax.legend(fontsize=9, loc='lower center'); ax.grid(alpha=0.3)

out_png = out_npz.replace('.npz', '.png')
fig.tight_layout()
fig.savefig(out_png, dpi=130, bbox_inches='tight')
plt.close(fig)
print(f"wrote {out_png}")
