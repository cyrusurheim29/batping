#!/usr/bin/env python3
"""
Interactive viewer for the 1-hour Δτ longterm time series.
Uses matplotlib's standard toolbar: zoom (magnifier), pan (hand), home (reset).
Scroll wheel also zooms on both axes.

Usage:
    python view_longterm.py
"""
import numpy as np
import matplotlib
matplotlib.use('MacOSX')          # native macOS window with toolbar
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.ndimage import uniform_filter1d
import os

NPZ  = os.path.join(os.path.dirname(__file__),
       'recordings/20260710_194120_longterm_a0.3_sm_4ms/'
       'stubbs_pipeline_v1/stubbs_lsfit.npz')

# ── Load + clean ───────────────────────────────────────────────────────────
d = np.load(NPZ)
dt_raw   = d['delta_tau']
sl_raw   = d['sigma_launch'] * 1000   # → ns
sr_raw   = d['sigma_return'] * 1000
sp_raw   = np.sqrt(sl_raw**2 + sr_raw**2)
eval_idx = d['eval_idx']
rep_rate = float(d['rep_rate'])

good_dt      = np.abs(dt_raw) <= 5.0
sp_med_ref   = np.median(sp_raw[good_dt])
good         = good_dt & (sp_raw <= 5.0 * sp_med_ref)

dt    = dt_raw[good]
sp    = sp_raw[good]
t_min = eval_idx[good] / rep_rate / 60.0

sort_i = np.argsort(t_min)
t_s  = t_min[sort_i]
dt_s = dt[sort_i]
sp_s = sp[sort_i]

# Rolling means at different window sizes
def rolling(data, window_periods):
    return uniform_filter1d(data, size=max(1, window_periods))

smooth_1min  = rolling(dt_s, int(rep_rate * 60))
smooth_5min  = rolling(dt_s, int(rep_rate * 300))

mean_dt = dt_s.mean()
std_dt  = dt_s.std()

# ── Figure layout ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 8), num='BATPING — 1-hr Δτ viewer')
fig.patch.set_facecolor('#f5f5f5')

# Main axes (top, large)
ax_main = fig.add_axes([0.06, 0.32, 0.91, 0.60])
# Rolling-mean-only axes (bottom, smaller)
ax_roll = fig.add_axes([0.06, 0.10, 0.91, 0.18], sharex=ax_main)

# ── Main panel ────────────────────────────────────────────────────────────
ax_main.plot(t_s, dt_s, '.', ms=0.8, alpha=0.15, color='#333333',
             rasterized=True, label='Δτ per period')
line_1min, = ax_main.plot(t_s, smooth_1min, '-', lw=1.8, color='#1f77b4',
                          label='1-min rolling mean', zorder=4)
line_5min, = ax_main.plot(t_s, smooth_5min, '-', lw=2.5, color='#d62728',
                          alpha=0.85, label='5-min rolling mean', zorder=5)
ax_main.axhline(mean_dt, color='navy', ls='--', lw=1.2, alpha=0.6,
                label=f'mean = {mean_dt:.3f} μs')
ax_main.axhline(mean_dt + std_dt, color='gray', ls=':', lw=1.0)
ax_main.axhline(mean_dt - std_dt, color='gray', ls=':', lw=1.0,
                label=f'σ = {std_dt:.3f} μs ({std_dt/10.3:.4f}°C)')
ax_main.set_ylabel('Δτ = τ_return − τ_launch (μs)', fontsize=11)
ax_main.set_title(
    f'a0.3_sm_4ms  —  1-hour Δτ time series  '
    f'(N={len(dt_s):,} clean periods  |  '
    f'δ_prop median = {np.median(sp_s):.0f} ns  |  '
    f'σ(Δτ) = {std_dt:.3f} μs → {std_dt/10.3:.4f}°C/ping)',
    fontsize=10, fontweight='bold')
ax_main.legend(fontsize=9, loc='upper right', framealpha=0.9)
ax_main.grid(True, alpha=0.3, ls='--')
ax_main.tick_params(labelbottom=False)

# ── Rolling-mean-only panel ───────────────────────────────────────────────
ax_roll.plot(t_s, smooth_1min, '-', lw=1.2, color='#1f77b4', alpha=0.6)
ax_roll.plot(t_s, smooth_5min, '-', lw=2.0, color='#d62728',
             label='5-min rolling mean')
ax_roll.axhline(mean_dt, color='navy', ls='--', lw=1.0, alpha=0.5)
ax_roll.set_xlabel('Time (min)', fontsize=11)
ax_roll.set_ylabel('Δτ (μs)', fontsize=9)
ax_roll.set_title('Rolling mean (zoom-linked)', fontsize=9)
ax_roll.grid(True, alpha=0.3, ls='--')

# ── Instructions text ─────────────────────────────────────────────────────
fig.text(0.06, 0.01,
         '  Zoom: toolbar magnifier or scroll wheel  |  '
         'Pan: toolbar hand or middle-click drag  |  '
         'Reset: Home button or double-click',
         fontsize=8.5, color='#555555',
         bbox=dict(boxstyle='round', fc='white', alpha=0.7, ec='#cccccc'))

# ── Scroll-wheel zoom ─────────────────────────────────────────────────────
def on_scroll(event):
    if event.inaxes not in (ax_main, ax_roll):
        return
    factor = 1.15 if event.button == 'down' else 1/1.15
    ax = event.inaxes
    xdata = event.xdata
    xlo, xhi = ax.get_xlim()
    xrange = (xhi - xlo) * factor
    ax.set_xlim(xdata - xrange * (xdata - xlo)/(xhi - xlo),
                xdata + xrange * (xhi - xdata)/(xhi - xlo))
    # y-zoom only on main panel
    if ax is ax_main:
        ydata = event.ydata
        ylo, yhi = ax.get_ylim()
        yr = (yhi - ylo) * factor
        ax.set_ylim(ydata - yr*(ydata-ylo)/(yhi-ylo),
                    ydata + yr*(yhi-ydata)/(yhi-ylo))
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('scroll_event', on_scroll)

plt.show()
