#!/usr/bin/env python3
"""
Test FM3a + FM3b fixes on the two characteristic failure waveforms.
Outputs → stubbs_pipeline_lsfit_v1_fm3fix/ inside each waveform's plots dir.
Original files are never touched.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import chirp_stubbs_analysis_fm3fix as st

# ── inline plot/save helpers (copied from run_lsfit_batch.py) ──────────────

def save_3panel(result, wf_name, rep_rate, out_path):
    T_L = result['T_launch']; T_R = result['T_return']
    fs = result['fs']; n_tmpl = result['n_template']
    dts = result['delta_tau_sf']; sigs = result['sigma_dt_sf']
    n_eval = result['n_eval']
    xc_l = result['ex_xc_l']; xc_r = result['ex_xc_r']
    env_l = result['ex_env_l']; env_r = result['ex_env_r']
    lags_l = result['ex_lags_l']; lags_r = result['ex_lags_r']
    tau_l = result['ex_tau_l']; tau_r = result['ex_tau_r']

    fig, axes = plt.subplots(3, 1, figsize=(11, 10))
    fig.suptitle(f'Stubbs LS shape-fit  —  {wf_name}  [FM3fix]',
                 fontsize=12, fontweight='bold')

    ax = axes[0]
    t_ms = np.arange(len(T_L)) / fs * 1000
    ax.plot(t_ms, T_L, 'b-', lw=1.2, label=f'T_launch (N={n_tmpl})')
    ax2 = ax.twinx()
    ax2.plot(t_ms, T_R, 'r-', lw=1.2, label=f'T_return (N={n_tmpl})')
    ax.set_xlabel('Time in period (ms)')
    ax.set_ylabel('T_launch', color='b'); ax2.set_ylabel('T_return', color='r')
    ax.set_title('Co-added Templates')
    lns = ax.get_lines() + ax2.get_lines()
    ax.legend(lns, [l.get_label() for l in lns], fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    zoom = 500
    ml = (lags_l >= tau_l - zoom) & (lags_l <= tau_l + zoom)
    mr = (lags_r >= tau_r - zoom) & (lags_r <= tau_r + zoom)
    xc_l_n = xc_l / (np.max(np.abs(xc_l)) + 1e-12)
    xc_r_n = xc_r / (np.max(np.abs(xc_r)) + 1e-12)
    ax.plot(lags_l[ml], xc_l_n[ml], 'b-', lw=0.5, alpha=0.35,
            label=f'launch xcorr  τ={tau_l:.2f} μs')
    ax.plot(lags_r[mr], xc_r_n[mr], 'r-', lw=0.5, alpha=0.35,
            label=f'return xcorr  τ={tau_r:.2f} μs')
    ax.plot(lags_l[ml], env_l[ml], 'b-', lw=2.0, alpha=0.9)
    ax.plot(lags_r[mr], env_r[mr], 'r-', lw=2.0, alpha=0.9)
    ax.axvline(tau_l, color='b', ls='--', lw=0.9, alpha=0.7)
    ax.axvline(tau_r, color='r', ls='--', lw=0.9, alpha=0.7)
    ax.set_xlabel('Lag (μs)'); ax.set_ylabel('Normalised xcorr')
    ax.set_title('Example Cross-Correlations (last eval period)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2]
    t_sec = (np.arange(n_eval) + n_tmpl) / rep_rate
    # Filter: only show clean periods (|Δτ|≤5μs) in the plot.
    # Burst-absent/transient periods (flat xcorr → wrong peak → |Δτ|>>5μs)
    # are already rejected from the σ calculation; showing them raw just
    # blows up the y-axis scale.
    clean = np.abs(dts) <= 5.0
    dts_plot  = np.where(clean, dts,  np.nan)
    sigs_plot = np.where(clean, sigs, np.nan)
    n_out    = int((~clean).sum())
    mean_dt  = float(np.nanmean(dts_plot))
    std_dt   = float(np.nanstd(dts_plot))
    med_sig  = float(np.median(sigs * 1e3))
    ax.errorbar(t_sec, dts_plot, yerr=sigs_plot,
                fmt='o-', ms=3, lw=1.0, elinewidth=0.9, capsize=2,
                mfc='none', mec='k', mew=0.7, ecolor='black', alpha=0.85,
                label=f'Δτ per period  ({n_out} outlier{"s" if n_out!=1 else ""} hidden)')
    ax.axhline(mean_dt, color='b', ls='--', lw=1.5,
               label=f'mean = {mean_dt:.3f} μs')
    ax.axhline(mean_dt + std_dt, color='gray', ls=':', lw=1)
    ax.axhline(mean_dt - std_dt, color='gray', ls=':', lw=1,
               label=f'σ = {std_dt:.3f} μs → {std_dt/10.3:.4f}°C/ping')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Δτ = τ_return − τ_launch  (μs)')
    ax.set_title(f'Differential TOF Time Series  (δ(Δτ) LS median = {med_sig:.1f} ns)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_hist_delta_tau(dts, wf_name, out_path):
    clean = dts[np.abs(dts) <= 5.0]
    n_out = len(dts) - len(clean)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(clean, bins=40, color='steelblue', alpha=0.75, edgecolor='white', lw=0.4)
    mu, s = float(np.mean(clean)), float(np.std(clean))
    ax.axvline(mu,     color='r', ls='-',  lw=1.5, label=f'mean = {mu:.3f} μs')
    ax.axvline(mu + s, color='r', ls='--', lw=1.2)
    ax.axvline(mu - s, color='r', ls='--', lw=1.2,
               label=f'±1σ = {s:.3f} μs  →  {s/10.3:.4f} °C/ping')
    ax.set_xlabel('Δτ  (μs)'); ax.set_ylabel('Count')
    title = f'{wf_name}  [FM3fix]  —  Δτ between-measurement spread  (N={len(clean)}'
    if n_out: title += f',  {n_out} outlier{"s" if n_out!=1 else ""} hidden'
    ax.set_title(title + ')')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_hist_uncertainties(result, wf_name, out_path):
    sig_l  = result['sigma_launch_sf'] * 1e3
    sig_r  = result['sigma_return_sf']  * 1e3
    sig_dt = result['sigma_dt_sf']      * 1e3
    N      = result['n_eval']
    meas_spread_ns = float(result['std_delta_tau_sf_us']) * 1e3

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        f'{wf_name}  [FM3fix]  —  Per-period fit uncertainty propagation\n'
        r'$\delta_{\Delta\tau} = \sqrt{\delta_L^2 + \delta_R^2}$'
        '  (LS shape-fit, ~150 pts each)',
        fontsize=11, fontweight='bold')

    panels = [
        (axes[0], sig_l,  r'$\delta_{launch}$  (xcorr fit uncertainty)',
         r'$\delta_{launch}$ (ns)', '#5b8db8'),
        (axes[1], sig_r,  r'$\delta_{return}$  (xcorr fit uncertainty)',
         r'$\delta_{return}$ (ns)', '#5aab6e'),
        (axes[2], sig_dt, r'$\sqrt{\delta_L^2+\delta_R^2}$  (propagated $\Delta\tau$)',
         r'$\sqrt{\delta_L^2+\delta_R^2}$ (ns)', '#9b72b0'),
    ]
    for ax, data, title, xlabel, color in panels:
        p99  = float(np.percentile(data, 99))
        p95  = float(np.percentile(data, 95))
        clip = data[data <= p99 * 1.2]
        mu   = float(np.mean(clip)); sigma = float(np.std(clip))
        xlim_max = max(p95 * 1.5, mu + 4 * sigma, 0.5)
        ax.hist(clip, bins=40, color=color, alpha=0.75, edgecolor='white', lw=0.3)
        ax.set_xlim([0, xlim_max])
        ax.axvline(mu,         color='r', ls='-',  lw=1.5, label=f'mean = {mu:.1f} ns')
        ax.axvline(mu + sigma, color='r', ls='--', lw=1.2, label=f'±1σ = {sigma:.1f} ns')
        ax.axvline(mu - sigma, color='r', ls='--', lw=1.2)
        ax.set_title(f'{title}  (N={N})', fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel('Count', fontsize=9)
        ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_excel(result, rep_rate, out_path):
    n_eval = result['n_eval']; n_tmpl = result['n_template']
    df = pd.DataFrame({
        'period_idx':         np.arange(n_eval) + n_tmpl,
        'time_s':             (np.arange(n_eval) + n_tmpl) / rep_rate,
        'tau_L_us':           result['tau_launch_sf'],
        'tau_R_us':           result['tau_return_sf'],
        'delta_tau_us':       result['delta_tau_sf'],
        'sigma_L_ns':         result['sigma_launch_sf'] * 1e3,
        'sigma_R_ns':         result['sigma_return_sf'] * 1e3,
        'sigma_delta_tau_ns': result['sigma_dt_sf']     * 1e3,
    })
    df.to_excel(out_path, index=False)


# ── test jobs ─────────────────────────────────────────────────────────────────

BASE    = os.path.dirname(os.path.abspath(__file__))
SESSION = 'recordings/20260716_015251_ecm999_full_sweep_v2_ecmHf'
SUBDIR  = 'stubbs_pipeline_lsfit_v1_fm3fix'
SDG_SR, SDG_MAX = 400_000, 16_384

JOBS = [
    ('a0.6_lg_4ms_received.wav', 4.0, 15.0),   # FM3b: -72% reject
    ('a0.6_sm_2ms_received.wav', 2.0, 15.0),   # FM3b + drift: -57% reject
    ('a0.8_sm_4ms_received.wav', 4.0, 15.0),   # crash fix: -76% reject (was N/A)
    ('a0.3_sm_2ms_received.wav', 2.0, 15.0),   # drift: -6% reject
    ('a0.4_md_2ms_received.wav', 2.0, 15.0),   # control: best performer
]

def rr(dur_ms, dead_ms=15.0):
    buf = min(int((dur_ms + dead_ms) / 1000 * SDG_SR), SDG_MAX)
    return SDG_SR / buf

raw_dir   = os.path.join(BASE, SESSION, 'raw')
plots_dir = os.path.join(BASE, SESSION, 'plots')

print("=" * 65)
print("FM3 fix test — chirp_stubbs_analysis_fm3fix.py")
print("  FM3a: pre_inc_ms 2→5 ms")
print("  FM3b: return window start ≥ inc_win_end + 0.6 ms")
print("=" * 65)

for wav_fname, dur_ms, dead_ms in JOBS:
    wf_name  = wav_fname.replace('_received.wav', '')
    wav_path = os.path.join(raw_dir, wav_fname)
    rep      = rr(dur_ms, dead_ms)
    out_dir  = os.path.join(plots_dir, wf_name, SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'─'*65}")
    print(f"  {wf_name}   rep={rep:.2f} Hz")
    print(f"{'─'*65}")

    r = st.analyze(wav_path, rep_rate=rep, n_template=100,
                   output_dir=out_dir, save_plot=False)

    save_3panel(r, wf_name, rep,
                os.path.join(out_dir, 'stubbs_lsfit.png'))
    save_hist_delta_tau(r['delta_tau_sf'], wf_name,
                        os.path.join(out_dir, 'hist_delta_tau.png'))
    save_hist_uncertainties(r, wf_name,
                            os.path.join(out_dir, 'hist_uncertainties.png'))
    save_excel(r, rep, os.path.join(out_dir, 'per_period.xlsx'))

    dt    = r['delta_tau_sf']
    sp    = r['sigma_dt_sf'] * 1e3
    clean = np.abs(dt) <= 5.0
    sigma = float(np.std(dt[clean])) if clean.sum() > 3 else None
    med_sp = float(np.median(sp[clean])) if clean.sum() > 3 else None
    n_c, n_t = int(clean.sum()), len(dt)

    print(f"\n  RESULT (FM3fix):")
    if sigma:
        print(f"    σ(Δτ)  = {sigma:.3f} μs  →  {sigma/10.3:.4f} °C/ping")
        print(f"    δ_prop = {med_sp:.1f} ns  (median shape-fit)")
    print(f"    N clean = {n_c}/{n_t}  (reject {(n_t-n_c)/n_t*100:.0f}%)")

    orig_xlsx = os.path.join(plots_dir, wf_name,
                             'stubbs_pipeline_lsfit_v1', 'per_period.xlsx')
    if os.path.exists(orig_xlsx):
        df_o = pd.read_excel(orig_xlsx)
        dt_o = df_o['delta_tau_us'].values
        sp_o = df_o['sigma_delta_tau_ns'].values
        cl_o = np.abs(dt_o) <= 5.0
        sig_o = float(np.std(dt_o[cl_o])) if cl_o.sum() > 3 else None
        print(f"\n  ORIGINAL (baseline):")
        if sig_o:
            print(f"    σ(Δτ)  = {sig_o:.3f} μs  →  {sig_o/10.3:.4f} °C/ping")
            print(f"    δ_prop = {float(np.median(sp_o[cl_o])):.0f} ns")
        print(f"    N clean = {cl_o.sum()}/{len(dt_o)}  "
              f"(reject {(len(dt_o)-cl_o.sum())/len(dt_o)*100:.0f}%)")

print(f"\nPlots → {SUBDIR}/ inside each waveform's plots dir.")
