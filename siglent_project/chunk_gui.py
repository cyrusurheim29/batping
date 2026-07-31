#!/usr/bin/env python3
"""
chunk_gui.py — Interactive 3-panel hover viewer for a BATPING analysis chunk.

Recreates the exact stubbs_chunk_NNN.png format as a live GUI.
Hover over any eval period in the bottom Δτ panel to update the top two panels.
Mouse leaves the panel → panels revert to templates / last-period CCF default.

Usage:
    python3 chunk_gui.py CHUNK_WAV
        [--rep-rate R]    default: 52.632 Hz
        [--n-template N]  default: 100
        [--inc-win L H]   incident window ms (auto-detected if omitted)
        [--ref-win L H]   reflection window ms (auto-detected if omitted)
        [--dechirp]       use dechirped LS fit; shows dechirped CCF/model in panel 2
        [--f-lo F]        chirp start frequency Hz (default 38450, for a0.3_sm_4ms)
        [--chirp-rate R]  BW/T Hz/s (default 1.5e6, for a0.3_sm_4ms 6kHz/4ms)

Example (chunk 28 of take4):
    python3 chunk_gui.py \\
      recordings/20260728_032243_batping_heattest_30V_take4/analysis_xcorr/chunks/chunk_028.wav \\
      --inc-win 0.0 5.63 --ref-win 5.63 11.28

Example (enclosure stats recording with dechirp):
    python3 chunk_gui.py \\
      recordings/20260730_184408_batping_stats_enclosure/a0.3_sm_4ms_received.wav \\
      --dechirp
"""
import sys, os, argparse
import numpy as np

import matplotlib.pyplot as plt   # imported here; backend switched in main() after csa import
from scipy.signal import correlate as _correlate, hilbert as _hilbert

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chirp_stubbs_analysis as csa

TOF_SENS_US = 10.3   # μs/°C
ZOOM_US     = 500    # μs half-width for CCF panel zoom


# ── Per-period computation ─────────────────────────────────────────────────────

def _dechirp_ccf(xc, R_auto, fs, f_lo_hz, chirp_rate, coarse_lag_us):
    """
    Compute dechirped CCF and model for display in panel 2.
    Returns (xc_d_rot_real, Rau_d, lags_d_us, coarse_idx) — all arrays.
    """
    center = len(R_auto) // 2      # = len(T) - 1
    xc_anal  = _hilbert(xc)
    Ra_anal  = _hilbert(R_auto)
    k_arr    = np.arange(len(xc)) - center
    lags_s   = k_arr / fs
    phi      = 2*np.pi*(f_lo_hz*lags_s + chirp_rate*lags_s**2/2)
    D        = np.exp(-1j * phi)
    xc_d     = xc_anal * D
    Rau_d    = (Ra_anal * D).real

    coarse   = center + int(round(coarse_lag_us * fs / 1e6))
    coarse   = max(0, min(coarse, len(xc)-1))
    rotation = np.exp(-1j * np.angle(xc_d[coarse]))
    xc_d_rot = (xc_d * rotation).real

    lags_us  = k_arr / fs * 1e6
    return xc_d_rot, Rau_d, lags_us, coarse


def _get_period(r, eval_idx, R_auto_L, R_auto_R,
                dechirp_f_lo=None, dechirp_rate=None):
    """
    Extract, drift-correct and mask eval period eval_idx (0-indexed).
    Computes BOTH parabola (xcorr_lag_us) and shape-fit (xcorr_lag_shape_fit) τ values.
    If dechirp_f_lo/rate are provided, also computes dechirped LS fit and CCF.
    Returns a dict with all quantities needed by the panel-drawing functions.
    """
    sig    = r['_sig']
    slices = r['_slices']
    shifts = r['_shifts_frac']
    n_tmpl = r['n_template']
    T_L    = r['T_launch']
    T_R    = r['T_return']
    fs     = r['fs']
    inc_ms = r['incident_win_ms']
    ref_ms = r['reflection_win_ms']

    pidx       = n_tmpl + eval_idx
    raw_period = sig[slices[pidx][0]:slices[pidx][1]]
    corrected  = csa.apply_correction(raw_period, shifts[pidx])
    inc_m = csa.mask_period(corrected, fs, inc_ms)
    ref_m = csa.mask_period(corrected, fs, ref_ms)

    # Parabola (envelope argmax + 11-pt parabola) — robust, no carrier ambiguity
    τl, xc_l, env_l, _ = csa.xcorr_lag_us(inc_m, T_L, fs)
    τr, xc_r, env_r, _ = csa.xcorr_lag_us(ref_m, T_R, fs)

    lags_l = (np.arange(len(xc_l)) - (len(T_L) - 1)) / fs * 1e6
    lags_r = (np.arange(len(xc_r)) - (len(T_R) - 1)) / fs * 1e6

    out = dict(
        corrected=corrected, inc_m=inc_m, ref_m=ref_m,
        τl=τl, τr=τr, xc_l=xc_l, env_l=env_l, xc_r=xc_r, env_r=env_r,
        lags_l=lags_l, lags_r=lags_r,
        dechirp=False,
    )

    if dechirp_f_lo is not None and dechirp_rate is not None:
        # Dechirped LS fit (Category 1 fix)
        τl_sf, σl_sf, _, _ = csa.xcorr_lag_shape_fit_dechirp(
            inc_m, T_L, fs, dechirp_f_lo, dechirp_rate,
            R_auto=R_auto_L, coarse_lag_us=τl)
        τr_sf, σr_sf, _, _ = csa.xcorr_lag_shape_fit_dechirp(
            ref_m, T_R, fs, dechirp_f_lo, dechirp_rate,
            R_auto=R_auto_R, coarse_lag_us=τr)
        # Dechirped CCF and model arrays for visualization
        xc_dl, Rau_dl, lags_dl, coarse_l = _dechirp_ccf(
            xc_l, R_auto_L, fs, dechirp_f_lo, dechirp_rate, τl)
        xc_dr, Rau_dr, lags_dr, coarse_r = _dechirp_ccf(
            xc_r, R_auto_R, fs, dechirp_f_lo, dechirp_rate, τr)
        out.update(dict(
            τl_sf=τl_sf, τr_sf=τr_sf, σl_sf=σl_sf, σr_sf=σr_sf,
            xc_dl=xc_dl, Rau_dl=Rau_dl, lags_dl=lags_dl,
            xc_dr=xc_dr, Rau_dr=Rau_dr, lags_dr=lags_dr,
            coarse_l_us=τl, coarse_r_us=τr,
            dechirp=True,
        ))
    else:
        τl_sf, σl_sf, _, _ = csa.xcorr_lag_shape_fit(inc_m, T_L, fs, R_auto=R_auto_L)
        τr_sf, σr_sf, _, _ = csa.xcorr_lag_shape_fit(ref_m, T_R, fs, R_auto=R_auto_R)
        out.update(dict(
            τl_sf=τl_sf, τr_sf=τr_sf, σl_sf=σl_sf, σr_sf=σr_sf,
        ))

    return out


# ── Panel-drawing helpers (exact match to analyze() formatting) ────────────────

def _panel1_templates(ax, ax_tw, r, wav_name):
    n_tmpl = r['n_template']
    T_L    = r['T_launch']
    T_R    = r['T_return']
    fs     = r['fs']
    t_ms   = np.arange(len(T_L)) / fs * 1000.0

    ax.cla(); ax_tw.cla()
    ax.plot(t_ms, T_L, 'b-', lw=1.2, label=f'T_launch  ({n_tmpl} avg)')
    ax_tw.plot(t_ms, T_R, 'r-', lw=1.2, label=f'T_return  ({n_tmpl} avg)')
    ax.set_xlabel('Time in window (ms)')
    ax.set_ylabel('T_launch (blue)')
    ax_tw.set_ylabel('T_return (red)')
    ax.set_title(f'Co-added Templates  —  {wav_name}')
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax_tw.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)


def _panel1_period(ax, ax_tw, pd, eval_idx, abs_idx, r, wav_name):
    fs   = r['fs']
    t_ms = np.arange(len(pd['corrected'])) / fs * 1000.0
    τl, τr, τl_sf, τr_sf = pd['τl'], pd['τr'], pd['τl_sf'], pd['τr_sf']

    ax.cla(); ax_tw.cla()
    ax.plot(t_ms, pd['inc_m'], 'b-', lw=1.2,
            label=f'T_launch  (period {abs_idx},  τ_L={τl:.2f} μs  SF={τl_sf:.2f} μs)')
    ax_tw.plot(t_ms, pd['ref_m'], 'r-', lw=1.2,
               label=f'T_return  (period {abs_idx},  τ_R={τr:.2f} μs  SF={τr_sf:.2f} μs)')
    ax.set_xlabel('Time in window (ms)')
    ax.set_ylabel('T_launch (blue)')
    ax_tw.set_ylabel('T_return (red)')
    ax.set_title(f'Period {abs_idx}  (eval {eval_idx})  —  {wav_name}')
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax_tw.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)


def _panel2(ax, pd, abs_idx):
    τl, τr         = pd['τl'],    pd['τr']
    τl_sf, τr_sf   = pd['τl_sf'], pd['τr_sf']
    xc_l, env_l    = pd['xc_l'], pd['env_l']
    xc_r, env_r    = pd['xc_r'], pd['env_r']
    lags_l, lags_r = pd['lags_l'], pd['lags_r']
    dechirp        = pd.get('dechirp', False)

    Δτ_sf   = τr_sf - τl_sf
    Δτ_para = τr    - τl
    slip_l  = abs(τl_sf - τl) > 10.0
    slip_r  = abs(τr_sf - τr) > 10.0

    # Zoom: centre on SF τ (primary estimator); widen if slip to keep parabola visible
    extra   = max(0.0, abs(τr_sf - τr), abs(τl_sf - τl)) * 1.1
    zoom    = ZOOM_US + extra
    m_l = (lags_l >= τl_sf - zoom) & (lags_l <= τl_sf + zoom)
    m_r = (lags_r >= τr_sf - zoom) & (lags_r <= τr_sf + zoom)

    # Normalise raw CCF relative to its peak in the zoom window
    peak_l = np.max(np.abs(xc_l[m_l])) + 1e-12
    peak_r = np.max(np.abs(xc_r[m_r])) + 1e-12
    xc_l_n = xc_l / peak_l
    xc_r_n = xc_r / peak_r
    env_l_n = env_l / peak_l
    env_r_n = env_r / peak_r

    ax.cla()
    ax.plot(lags_l[m_l], xc_l_n[m_l], 'b-', lw=0.6, alpha=0.45)
    ax.plot(lags_r[m_r], xc_r_n[m_r], 'r-', lw=0.6, alpha=0.45)
    ax.plot(lags_l[m_l], env_l_n[m_l], 'b-', lw=2.0, alpha=0.9, label='envelope launch')
    ax.plot(lags_r[m_r], env_r_n[m_r], 'r-', lw=2.0, alpha=0.9, label='envelope return')

    if dechirp:
        # Dechirped CCF (Re(xc_d_rot)) and dechirped template model (Rau_d) ─────
        # Both are real sincs after dechirping; overlap shows fit quality.
        xc_dl, Rau_dl = pd['xc_dl'], pd['Rau_dl']
        xc_dr, Rau_dr = pd['xc_dr'], pd['Rau_dr']
        lags_dl, lags_dr = pd['lags_dl'], pd['lags_dr']
        coarse_l = pd['coarse_l_us']; coarse_r = pd['coarse_r_us']

        # Zoom around coarse peak (±500 μs)
        zoom_d = ZOOM_US
        ml_d = (lags_dl >= coarse_l - zoom_d) & (lags_dl <= coarse_l + zoom_d)
        mr_d = (lags_dr >= coarse_r - zoom_d) & (lags_dr <= coarse_r + zoom_d)

        # Normalise by peak of dechirped CCF in zoom window
        pk_dl = np.max(np.abs(xc_dl[ml_d])) + 1e-12
        pk_dr = np.max(np.abs(xc_dr[mr_d])) + 1e-12
        xc_dl_n  = xc_dl  / pk_dl
        Rau_dl_n = Rau_dl / (np.max(Rau_dl[ml_d]) + 1e-12)
        xc_dr_n  = xc_dr  / pk_dr
        Rau_dr_n = Rau_dr / (np.max(Rau_dr[mr_d]) + 1e-12)

        ax.plot(lags_dl[ml_d], xc_dl_n[ml_d], 'b--', lw=1.5, alpha=0.9,
                label=f'dechirped CCF launch (coarse {coarse_l:.1f} μs)')
        ax.plot(lags_dr[mr_d], xc_dr_n[mr_d], 'r--', lw=1.5, alpha=0.9,
                label=f'dechirped CCF return (coarse {coarse_r:.1f} μs)')
        ax.plot(lags_dl[ml_d], Rau_dl_n[ml_d], 'b:', lw=1.2, alpha=0.55,
                label='Rau_d model (launch)')
        ax.plot(lags_dr[mr_d], Rau_dr_n[mr_d], 'r:', lw=1.2, alpha=0.55,
                label='Rau_d model (return)')

        # Dechirped SF τ marks
        ax.axvline(τl_sf, color='b', ls='--', lw=1.8, alpha=0.9,
                   label=f'τ_L dechirp = {τl_sf:.4f} μs  (Δτ={Δτ_sf:.4f})')
        ax.axvline(τr_sf, color='r', ls='--', lw=1.8, alpha=0.9,
                   label=f'τ_R dechirp = {τr_sf:.4f} μs')
        # Coarse (parabola) τ for reference
        ax.axvline(τl, color='#88aadd', ls=':', lw=1.4, alpha=0.8,
                   label=f'τ_L coarse = {τl:.4f} μs')
        ax.axvline(τr, color='#dd8888', ls=':', lw=1.4, alpha=0.8,
                   label=f'τ_R coarse = {τr:.4f} μs')
    else:
        # SF τ — PRIMARY: solid dashed, boldly labelled
        ax.axvline(τl_sf, color='b', ls='--', lw=1.8, alpha=0.9,
                   label=f'τ_L SF = {τl_sf:.2f} μs  (Δτ_SF={Δτ_sf:.2f})')
        ax.axvline(τr_sf, color='r', ls='--', lw=1.8, alpha=0.9,
                   label=f'τ_R SF = {τr_sf:.2f} μs')
        # Parabola τ — REFERENCE: thinner dotted, orange if disagreeing with SF
        col_l = 'darkorange' if slip_l else '#88aadd'
        col_r = 'darkorange' if slip_r else '#dd8888'
        ax.axvline(τl, color=col_l, ls=':', lw=1.4, alpha=0.8,
                   label=f'τ_L para = {τl:.2f} μs{"  ⚠ SLIP" if slip_l else ""}')
        ax.axvline(τr, color=col_r, ls=':', lw=1.4, alpha=0.8,
                   label=f'τ_R para = {τr:.2f} μs{"  ⚠ SLIP" if slip_r else ""}')

    ax.set_xlabel('Lag (μs)')
    ax.set_ylabel('Normalised xcorr / dechirped')
    slip_warn = '  ⚠ CARRIER-CYCLE SLIP' if (slip_l or slip_r) else ''
    mode_str  = 'dechirped LS' if dechirp else 'shape-fit'
    ax.set_title(
        f'Cross-Corr + dechirped fit (period {abs_idx}){slip_warn}\n'
        f'Δτ_{mode_str[:2]}={Δτ_sf:.4f} μs  |  Δτ_para={Δτ_para:.4f} μs')
    ax.legend(fontsize=7.0, loc='upper left')
    ax.grid(True, alpha=0.3)


def _panel3(ax, t_sec, delta_tau_sf, sigma_dt_sf, n_eval, n_tmpl, wav_name):
    """Exact replica of analyze() panel 3. Returns (inlier_mask,)."""
    _med = float(np.median(delta_tau_sf))
    _mad = float(np.median(np.abs(delta_tau_sf - _med))) * 1.4826
    _clip_hw = 5.0 * _mad
    _inlier  = np.abs(delta_tau_sf - _med) <= _clip_hw
    n_hidden  = int(np.sum(~_inlier))
    _mean_p   = float(np.mean(delta_tau_sf[_inlier]))
    _std_p    = float(np.std(delta_tau_sf[_inlier]))

    dt_plot = delta_tau_sf.astype(float).copy()
    dt_plot[~_inlier] = np.nan

    ax.plot(t_sec, dt_plot, '-', color='#4477cc', lw=0.7, alpha=0.6, zorder=1)

    step = max(1, n_eval // 600)
    _sel = _inlier[::step]
    ax.errorbar(t_sec[::step][_sel], delta_tau_sf[::step][_sel],
                yerr=sigma_dt_sf[::step][_sel],
                fmt='o', ms=2.5, mfc='none', mec='black', mew=0.7,
                ecolor='black', elinewidth=0.5, capsize=2, zorder=2, alpha=0.7,
                label='Δτ per period' + (f'  ({n_hidden} outliers hidden)' if n_hidden else ''))
    ax.axhline(_mean_p, color='navy', ls='--', lw=1.2, alpha=0.7,
               label=f'mean = {_mean_p:.3f} μs  (inliers)')
    ax.axhline(_mean_p + _std_p, color='gray', ls=':', lw=0.9)
    ax.axhline(_mean_p - _std_p, color='gray', ls=':', lw=0.9,
               label=f'σ = {_std_p:.3f} μs  ({_std_p/TOF_SENS_US:.4f}°C/ping)')

    _ylo = float(delta_tau_sf[_inlier].min())
    _yhi = float(delta_tau_sf[_inlier].max())
    _yr  = max(_yhi - _ylo, 2 * _mad, 0.2)
    ax.set_ylim(_ylo - 0.12 * _yr, _yhi + 0.12 * _yr)

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
    ax.set_title(
        f'Δτ time series  (shape-fit)  N={n_eval} ({n_eval - n_hidden} shown)  '
        f'σ={_std_p:.3f} μs  δ_prop med={np.median(sigma_dt_sf)*1e3:.1f} ns')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    return _inlier


# ── Interactive viewer ─────────────────────────────────────────────────────────

class ChunkViewer:
    def __init__(self, r, wav_name, rep_rate,
                 dechirp_f_lo=None, dechirp_rate=None):
        self.r           = r
        self.wav_name    = wav_name
        self.rep_rate    = rep_rate
        self.n_eval      = r['n_eval']
        self.n_tmpl      = r['n_template']
        self.dechirp_f_lo  = dechirp_f_lo
        self.dechirp_rate  = dechirp_rate

        dt_sf  = r['delta_tau_sf']
        sig_sf = r['sigma_dt_sf']
        t_sec  = (np.arange(self.n_eval) + self.n_tmpl) / rep_rate
        self.t_sec = t_sec
        self.dt_sf = dt_sf
        self.sig_sf = sig_sf

        # ── Figure — same size as analyze() ──────────────────────────────────
        self.fig, axes = plt.subplots(3, 1, figsize=(12, 11))
        self.ax1    = axes[0]
        self.ax1_tw = self.ax1.twinx()
        self.ax2    = axes[1]
        self.ax3    = axes[2]

        # Precompute template autocorrelations for shape-fit (done once)
        self.R_auto_L = _correlate(r['T_launch'], r['T_launch'], mode='full')
        self.R_auto_R = _correlate(r['T_return'], r['T_return'], mode='full')

        # ── Default panels ────────────────────────────────────────────────────
        _panel1_templates(self.ax1, self.ax1_tw, r, wav_name)

        # Panel 2 default: last eval period (matches analyze() behavior)
        last = self.n_eval - 1
        pd = _get_period(r, last, self.R_auto_L, self.R_auto_R,
                         dechirp_f_lo=dechirp_f_lo, dechirp_rate=dechirp_rate)
        _panel2(self.ax2, pd, self.n_tmpl + last)

        # Panel 3: static Δτ series
        _panel3(self.ax3, t_sec, dt_sf, sig_sf,
                self.n_eval, self.n_tmpl, wav_name)

        # Hover cursor on panel 3
        self._cursor = self.ax3.axvline(
            t_sec[0], color='darkorange', lw=1.2, alpha=0.0, zorder=10)

        self._last_idx = None
        self.fig.tight_layout()

        # ── Events ───────────────────────────────────────────────────────────
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.fig.canvas.mpl_connect('axes_leave_event',    self._on_leave)

        plt.show()

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_motion(self, event):
        if event.inaxes is not self.ax3 or event.xdata is None:
            return
        idx = int(np.argmin(np.abs(self.t_sec - event.xdata)))
        idx = max(0, min(idx, self.n_eval - 1))
        if idx == self._last_idx:
            return
        self._last_idx = idx
        self._update(idx)

    def _on_leave(self, event):
        if event.inaxes is self.ax3:
            self._last_idx = None
            self._restore_defaults()

    # ── Update / restore ──────────────────────────────────────────────────────

    def _update(self, eval_idx):
        self._cursor.set_xdata([self.t_sec[eval_idx], self.t_sec[eval_idx]])
        self._cursor.set_alpha(0.75)

        pd      = _get_period(self.r, eval_idx, self.R_auto_L, self.R_auto_R,
                              dechirp_f_lo=self.dechirp_f_lo,
                              dechirp_rate=self.dechirp_rate)
        abs_idx = self.n_tmpl + eval_idx

        _panel1_period(self.ax1, self.ax1_tw, pd, eval_idx, abs_idx,
                       self.r, self.wav_name)
        _panel2(self.ax2, pd, abs_idx)
        self.fig.canvas.draw_idle()

    def _restore_defaults(self):
        self._cursor.set_alpha(0.0)
        _panel1_templates(self.ax1, self.ax1_tw, self.r, self.wav_name)
        last = self.n_eval - 1
        pd = _get_period(self.r, last, self.R_auto_L, self.R_auto_R,
                         dechirp_f_lo=self.dechirp_f_lo,
                         dechirp_rate=self.dechirp_rate)
        _panel2(self.ax2, pd, self.n_tmpl + last)
        self.fig.canvas.draw_idle()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    # Switch to an interactive backend NOW — after all imports (including csa which
    # forces Agg).  switch_backend() overrides use() even after pyplot is imported.
    for _b in ('MacOSX', 'TkAgg', 'Qt5Agg', 'GTK3Agg'):
        try:
            plt.switch_backend(_b)
            break
        except Exception:
            pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('wav')
    ap.add_argument('--rep-rate',   type=float, default=52.631578947368425)
    ap.add_argument('--n-template', type=int,   default=100)
    ap.add_argument('--inc-win',    type=float, nargs=2, default=None,
                    metavar=('L', 'H'))
    ap.add_argument('--ref-win',    type=float, nargs=2, default=None,
                    metavar=('L', 'H'))
    ap.add_argument('--dechirp', action='store_true',
                    help='Use dechirped LS fit; shows dechirped CCF and model in panel 2')
    ap.add_argument('--f-lo',    type=float, default=38450.0,
                    help='Chirp start frequency Hz (default 38450, a0.3_sm_4ms)')
    ap.add_argument('--chirp-rate', type=float, default=1_500_000.0,
                    help='Chirp rate BW/T Hz/s (default 1.5e6, 6kHz/4ms)')
    args = ap.parse_args()

    dechirp_f_lo  = args.f_lo        if args.dechirp else None
    dechirp_rate  = args.chirp_rate  if args.dechirp else None

    print(f"Loading: {args.wav}")
    print("Running analysis (~10 s)…")
    r = csa.analyze(
        args.wav,
        rep_rate=args.rep_rate,
        n_template=args.n_template,
        incident_win_ms=tuple(args.inc_win) if args.inc_win else None,
        reflection_win_ms=tuple(args.ref_win) if args.ref_win else None,
        save_plot=False,
        _return_internals=True,
        _search_radius_us=700.0,
        _dechirp_f_lo=dechirp_f_lo,
        _dechirp_rate=dechirp_rate,
    )
    wav_name = os.path.splitext(os.path.basename(args.wav))[0]
    print(f"Done — n_eval={r['n_eval']}, fs={r['fs']} Hz, "
          f"inc={r['incident_win_ms'][0]:.2f}–{r['incident_win_ms'][1]:.2f} ms  "
          f"ref={r['reflection_win_ms'][0]:.2f}–{r['reflection_win_ms'][1]:.2f} ms")
    if args.dechirp:
        print(f"Dechirp mode: f_lo={args.f_lo:.0f} Hz, rate={args.chirp_rate:.0e} Hz/s")
    print("Hover over the bottom panel to inspect individual periods.")

    ChunkViewer(r, wav_name, args.rep_rate,
                dechirp_f_lo=dechirp_f_lo, dechirp_rate=dechirp_rate)


if __name__ == '__main__':
    main()
