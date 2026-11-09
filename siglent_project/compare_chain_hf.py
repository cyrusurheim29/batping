#!/usr/bin/env python3
"""Overlay two or three chain_tconnector_careful NPZs to test whether the
ripples in H(f) are real piezo structure or multipath interference.

Usage:
    python3 compare_chain_hf.py <A.npz> <B.npz> [<C.npz>]

Interpretation:
  A = original geometry
  B = mic shifted 5-10 mm laterally (tests MULTIPATH vs piezo)
  C = restored to original geometry (tests TIME drift)

  • A vs B differ at ripple locations BUT share smooth envelope → multipath
  • A vs C agree point-for-point → setup is stable in time
  • A vs B agree everywhere → multipath hypothesis FAILS (likely real piezo)
"""
import os, sys
import numpy as np
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    sys.exit(__doc__)

paths = sys.argv[1:]
labels = ['A (original)', 'B (shifted)', 'C (restored)'][:len(paths)]
colors = ['#1f77b4', '#e6444d', '#88aa44'][:len(paths)]

Hs = []
for p in paths:
    d = np.load(p)
    f    = d['freqs'].astype(float)
    amp  = d['amp_mV'].astype(float)
    # freq-sort
    order = np.argsort(f)
    f, amp = f[order], amp[order]
    amp_dB = 20*np.log10(amp + 1e-9)
    H_raw = amp_dB - np.max(amp_dB[f > 25_000])
    H_smooth = savgol_filter(H_raw, window_length=min(71, len(H_raw)|1), polyorder=3)
    Hs.append(dict(f=f, H_raw=H_raw, H_smooth=H_smooth, name=os.path.basename(p)))

fig, axes = plt.subplots(3, 1, figsize=(15, 11),
                          gridspec_kw={'height_ratios': [2, 1.2, 1.2]})

# (1) Overlay raw H(f)
ax = axes[0]
for H, lbl, c in zip(Hs, labels, colors):
    ax.plot(H['f']/1e3, H['H_raw'],    '.', color=c, ms=2, alpha=0.4)
    ax.plot(H['f']/1e3, H['H_smooth'], '-', color=c, lw=1.5, label=f'{lbl}  ({H["name"]})')
ax.axvspan(37, 43, color='green', alpha=0.08, label='chirp band 37-43 kHz')
ax.axhline(-3, color='k', ls=':', lw=0.5)
ax.axhline(-6, color='k', ls=':', lw=0.5)
ax.set_ylim(-30, 5)
ax.set_ylabel('H (dB)')
ax.set_title('Raw H(f) + SG-smoothed overlay', fontsize=11, fontweight='bold')
ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc='lower center')

# (2) Smoothed-only overlay — do the envelopes agree?
ax = axes[1]
for H, lbl, c in zip(Hs, labels, colors):
    ax.plot(H['f']/1e3, H['H_smooth'], '-', color=c, lw=1.5, label=lbl)
ax.axvspan(37, 43, color='green', alpha=0.08)
ax.set_ylim(-20, 2)
ax.set_ylabel('H_smooth (dB)')
ax.set_title('Smoothed envelopes — if these agree, the SMOOTH shape is reproducible (piezo chain)',
             fontsize=10)
ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='lower center')

# (3) Differences from A, to see where ripples disagree
ax = axes[2]
f_common = Hs[0]['f']
H_A = Hs[0]['H_raw']
for H, lbl, c in zip(Hs[1:], labels[1:], colors[1:]):
    diff = H['H_raw'] - np.interp(H['f'], f_common, H_A)
    rms = np.sqrt(np.mean(diff**2))
    ax.plot(H['f']/1e3, diff, '-', color=c, lw=0.8, alpha=0.7,
            label=f'{lbl} − A   RMS = {rms:.2f} dB')
ax.axhline(0, color='k', lw=0.5)
ax.axvspan(37, 43, color='green', alpha=0.08)
ax.set_xlabel('Frequency (kHz)')
ax.set_ylabel('ΔH from run A (dB)')
ax.set_ylim(-10, 10)
ax.set_title('Ripple-location disagreement:  A vs B (geometry test),  A vs C (time test).\n'
             'Multipath →  A-B RMS >> A-C RMS, and large swings precisely at ripple freqs',
             fontsize=10)
ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='upper right')

fig.tight_layout()
out = 'compare_chain_hf.png'
fig.savefig(out, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f'wrote {out}')

# Quantitative signature print
print(f'\nQuantitative signatures:')
for H, lbl, c in zip(Hs[1:], labels[1:], colors[1:]):
    diff = H['H_raw'] - np.interp(H['f'], f_common, H_A)
    diff_smooth = H['H_smooth'] - np.interp(H['f'], f_common, Hs[0]['H_smooth'])
    print(f'  {lbl} - A:')
    print(f'    raw      RMS = {np.sqrt(np.mean(diff**2)):.2f} dB  (ripple-position difference)')
    print(f'    smoothed RMS = {np.sqrt(np.mean(diff_smooth**2)):.2f} dB  (envelope difference)')
