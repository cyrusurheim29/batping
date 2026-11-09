#!/usr/bin/env python3
"""Plot a chain H(f) characterization NPZ properly (freq-sorted, with
pass-to-pass dispersion shown as a shaded band).

Usage:
    python3 plot_chain_hf.py <path_to_averaged_npz>
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

if len(sys.argv) < 2:
    sys.exit("usage: plot_chain_hf.py <npz>")
NPZ = sys.argv[1]

d = np.load(NPZ)
f    = d['freqs'].astype(float)
amp  = d['amp_mV'].astype(float)
std  = d['amp_std_mV'].astype(float) if 'amp_std_mV' in d.files else np.zeros_like(amp)
snr  = d['snr_dB'].astype(float)     if 'snr_dB'     in d.files else np.full_like(amp, np.nan)
hits = d['n_hits_per_freq'].astype(int) if 'n_hits_per_freq' in d.files else np.full(len(f), 1)

# Freq-sort (guard in case someone passed random-order data)
order = np.argsort(f)
f, amp, std, snr, hits = f[order], amp[order], std[order], snr[order], hits[order]

# H(f) in dB, normalised to peak in the band
amp_dB = 20 * np.log10(amp + 1e-9)
mask = f > 25_000
peak_ref = float(np.max(amp_dB[mask]))
H_mag = amp_dB - peak_ref

# ±1σ envelope in dB (asymmetric in linear → symmetric in dB works OK for small std/amp)
amp_lo = np.clip(amp - std, 1e-9, None)
amp_hi = amp + std
H_lo = 20*np.log10(amp_lo) - peak_ref
H_hi = 20*np.log10(amp_hi) - peak_ref

# Savitzky-Golay smoothed (same window as the pipeline)
wlen = min(71, len(H_mag) | 1)
H_smooth = savgol_filter(H_mag, window_length=wlen, polyorder=3)

# Peak + −3/−6 dB bandwidth
peak_i = int(np.argmax(H_smooth))
peak_f = f[peak_i]
H_norm = H_smooth - H_smooth[peak_i]
def _bw(thr):
    sel = f[(f > 35_000) & (f < 50_000) & (H_norm >= thr)]
    return (sel.min(), sel.max()) if len(sel) else (None, None)
f3lo, f3hi = _bw(-3); f6lo, f6hi = _bw(-6)

# Reproducibility stats (multi-hit freqs only)
multi = (hits > 1) & (amp > 0)
rel = 100 * std[multi] / amp[multi] if multi.any() else np.zeros(1)
med_rel = float(np.median(rel))
p90_rel = float(np.percentile(rel, 90))

# ── Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True,
                          gridspec_kw={'height_ratios': [2, 1, 1]})

# (1) H(f) with ±σ band + smoothed curve
ax = axes[0]
ax.fill_between(f/1e3, H_lo, H_hi, color='#888', alpha=0.25, label='±1σ across passes')
ax.plot(f/1e3, H_mag,    '.', color='#4477cc', ms=3, alpha=0.6, label='averaged H(f)')
ax.plot(f/1e3, H_smooth, '-', color='#e6444d', lw=1.7, label=f'SG-smoothed (w={wlen})')
ax.axvspan(37, 43, color='green', alpha=0.08, label='chirp band 37-43 kHz')
ax.axhline(-3, color='k', ls=':', lw=0.5)
ax.axhline(-6, color='k', ls=':', lw=0.5)
if f3lo: ax.axvspan(f3lo/1e3, f3hi/1e3, color='yellow', alpha=0.05)
ax.set_ylabel('H (dB), normalised to peak')
ax.set_ylim(-30, 5)
ax.set_title(f'{os.path.basename(NPZ)}   —   peak {peak_f/1e3:.2f} kHz   '
             f'−3 dB: {f3lo/1e3:.2f}-{f3hi/1e3:.2f} kHz  '
             f'BW={int(f3hi-f3lo) if f3lo else 0} Hz   '
             f'(reproducibility: median σ/μ {med_rel:.1f}%, p90 {p90_rel:.1f}%)',
             fontsize=10, fontweight='bold')
ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc='lower center')

# (2) SNR per freq
ax = axes[1]
ax.plot(f/1e3, snr, '-', color='#88aa44', lw=1.0)
ax.axhline(10, color='k', ls=':', lw=0.5, alpha=0.5)
ax.axhline(20, color='k', ls=':', lw=0.5, alpha=0.5)
ax.set_ylabel('SNR (dB)')
ax.grid(alpha=0.3)
ax.set_title(f'Per-freq SNR   (median {np.nanmedian(snr):.1f} dB)', fontsize=10)

# (3) Reproducibility
ax = axes[2]
ax.plot(f/1e3, 100*std/(amp+1e-9), '-', color='#1f77b4', lw=1.0)
ax.axhline(5,  color='k', ls=':', lw=0.5, alpha=0.5)
ax.axhline(15, color='k', ls=':', lw=0.5, alpha=0.5)
ax.axhline(50, color='r', ls=':', lw=0.5, alpha=0.5)
ax.set_xlabel('Frequency (kHz)')
ax.set_ylabel('σ/μ across passes (%)')
ax.set_ylim(0, min(150, np.percentile(100*std/(amp+1e-9), 95) * 1.3))
ax.grid(alpha=0.3)
ax.set_title(f'Pass-to-pass dispersion (lower = cleaner, 100% = trigger-miss pattern)',
             fontsize=10)

fig.tight_layout()
out_png = NPZ.replace('.npz', '_plot.png')
fig.savefig(out_png, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f"wrote {out_png}")
print(f"  peak {peak_f/1e3:.2f} kHz, −3 dB {f3lo/1e3:.2f}–{f3hi/1e3:.2f} kHz "
      f"(BW {int(f3hi-f3lo) if f3lo else 0} Hz)")
print(f"  reproducibility median {med_rel:.1f}%, p90 {p90_rel:.1f}%")
