#!/usr/bin/env python3
"""
run_lsfit_batch.py

Batch Stubbs pipeline with LS shape-fit estimator on all chirp and alpha sweep waveforms.

For each waveform creates  <per-wf-plots-dir>/stubbs_pipeline_lsfit_v1/  containing:
  stubbs_lsfit.png          3-panel plot: templates | example CCF | Δτ with error bars
  hist_delta_tau.png        histogram of Δτ across eval periods
  hist_sigma_delta_tau.png  histogram of per-period σ(Δτ) = √(σ_L² + σ_R²)
  per_period.xlsx           table: period_idx, time_s, tau_L, tau_R, Δτ, σ_L, σ_R, σ(Δτ)
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import chirp_stubbs_analysis as st

BASE    = os.path.dirname(os.path.abspath(__file__))
SDG_SR, SDG_MAX = 400000, 16384
SUBDIR  = 'stubbs_pipeline_lsfit_v1'
N_TMPL  = 100


def rr(dur_ms, dead_ms=15.0):
    buf = min(int((dur_ms + dead_ms) / 1000 * SDG_SR), SDG_MAX)
    return SDG_SR / buf


# ── Sessions ──────────────────────────────────────────────────────────────────
# chirp sweep: explicit dead times (bw_6kHz has 23ms, all others 15ms)
CHIRP_CFG = {
    'bw_20kHz_wide_4ms_comp':       (4.0, 15.0),
    'bw_20kHz_wide_4ms_comp_light': (4.0, 15.0),
    'bw_20kHz_wide_4ms_diff':       (4.0, 15.0),
    'bw_25kHz_wide_2ms_comp':       (2.0, 15.0),
    'bw_6kHz_sym_2ms_flat':         (2.0, 23.0),
    'hf_m15dB_asym_2ms_comp_light': (2.0, 15.0),
    'hf_m15dB_asym_2ms_comp':       (2.0, 15.0),
    'hf_m15dB_asym_2ms_comp_rect':  (2.0, 15.0),
    'hf_m15dB_asym_2ms_diff':       (2.0, 15.0),
}

SESSIONS = [
    ('recordings/20260710_030748_chirp_sweep',                        CHIRP_CFG),
    ('recordings/20260710_031308_alpha_sweep_voltage_compensated',     None),   # auto
]


def dur_from_name(name):
    """Extract dur_ms from filenames like a0.3_sm_2ms or bw_6kHz_sym_2ms_flat."""
    m = re.search(r'_(\d+)ms', name)
    return float(m.group(1)) if m else 4.0


# ── Output generators ─────────────────────────────────────────────────────────

def save_3panel(result, wf_name, rep_rate, out_path):
    T_L    = result['T_launch'];    T_R    = result['T_return']
    fs     = result['fs'];          n_tmpl = result['n_template']
    dts    = result['delta_tau_sf']; sigs  = result['sigma_dt_sf']
    n_eval = result['n_eval']
    xc_l   = result['ex_xc_l'];    xc_r   = result['ex_xc_r']
    env_l  = result['ex_env_l'];   env_r  = result['ex_env_r']
    lags_l = result['ex_lags_l'];  lags_r = result['ex_lags_r']
    tau_l  = result['ex_tau_l'];   tau_r  = result['ex_tau_r']

    fig, axes = plt.subplots(3, 1, figsize=(11, 10))
    fig.suptitle(f'Stubbs LS shape-fit  —  {wf_name}', fontsize=12, fontweight='bold')

    # Panel 1: co-added templates
    ax = axes[0]
    t_ms = np.arange(len(T_L)) / fs * 1000
    ax.plot(t_ms, T_L, 'b-', lw=1.2, label=f'T_launch  (N={n_tmpl})')
    ax2 = ax.twinx()
    ax2.plot(t_ms, T_R, 'r-', lw=1.2, label=f'T_return  (N={n_tmpl})')
    ax.set_xlabel('Time in period (ms)')
    ax.set_ylabel('T_launch', color='b');  ax2.set_ylabel('T_return', color='r')
    ax.set_title('Co-added Templates')
    lns = ax.get_lines() + ax2.get_lines()
    ax.legend(lns, [l.get_label() for l in lns], fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel 2: example CCF (raw carrier + envelope)
    ax = axes[1]
    zoom = 500   # μs
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
    ax.set_xlabel('Lag (μs)');  ax.set_ylabel('Normalised xcorr')
    ax.set_title('Example Cross-Correlations (last eval period)')
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)

    # Panel 3: Δτ time series — black points + connecting line + LS error bars
    ax = axes[2]
    t_sec   = (np.arange(n_eval) + n_tmpl) / rep_rate
    mean_dt = float(np.mean(dts))
    std_dt  = float(np.std(dts))
    med_sig = float(np.median(sigs * 1e3))
    ax.errorbar(t_sec, dts, yerr=sigs,
                fmt='o-', ms=3, lw=1.0, elinewidth=0.9, capsize=2,
                mfc='none', mec='k', mew=0.7, ecolor='black', alpha=0.85,
                label=f'Δτ per period')
    ax.axhline(mean_dt, color='b', ls='--', lw=1.5,
               label=f'mean = {mean_dt:.3f} μs')
    ax.axhline(mean_dt + std_dt, color='gray', ls=':', lw=1)
    ax.axhline(mean_dt - std_dt, color='gray', ls=':', lw=1,
               label=f'σ = {std_dt:.3f} μs → {std_dt/10.3:.4f}°C/ping')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Δτ = τ_return − τ_launch  (μs)')
    ax.set_title(f'Differential TOF Time Series  (δ(Δτ) LS median = {med_sig:.1f} ns)')
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_hist_delta_tau(dts, wf_name, out_path):
    """Standalone histogram of Δτ — the between-measurement variance."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(dts, bins=40, color='steelblue', alpha=0.75, edgecolor='white', lw=0.4)
    mu, s = float(np.mean(dts)), float(np.std(dts))
    ax.axvline(mu,     color='r', ls='-',  lw=1.5, label=f'mean = {mu:.3f} μs')
    ax.axvline(mu + s, color='r', ls='--', lw=1.2)
    ax.axvline(mu - s, color='r', ls='--', lw=1.2,
               label=f'±1σ = {s:.3f} μs  →  {s/10.3:.4f} °C/ping')
    ax.set_xlabel('Δτ  (μs)');  ax.set_ylabel('Count')
    ax.set_title(f'{wf_name}  —  Δτ between-measurement spread  (N={len(dts)})')
    ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_hist_uncertainties(result, wf_name, out_path):
    sig_l   = result['sigma_launch_sf'] * 1e3   # ns
    sig_r   = result['sigma_return_sf']  * 1e3   # ns
    sig_dt  = result['sigma_dt_sf']      * 1e3   # ns
    N       = result['n_eval']
    # between-period scatter of shape-fit Δτ, for reference dotted line in right panel
    meas_spread_ns = float(result['std_delta_tau_sf_us']) * 1e3

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        f'{wf_name}  —  Per-period fit uncertainty propagation\n'
        r'$\delta_{\Delta\tau} = \sqrt{\delta_L^2 + \delta_R^2}$'
        '  (LS shape-fit, ~150 pts each)',
        fontsize=11, fontweight='bold'
    )

    panels = [
        (axes[0], sig_l,  r'$\delta_{launch}$  (xcorr fit uncertainty)',
         r'$\delta_{launch}$ (ns)', '#5b8db8'),
        (axes[1], sig_r,  r'$\delta_{return}$  (xcorr fit uncertainty)',
         r'$\delta_{return}$ (ns)', '#5aab6e'),
        (axes[2], sig_dt, r'$\sqrt{\delta_L^2+\delta_R^2}$  (propagated $\Delta\tau$ fit uncertainty)',
         r'$\sqrt{\delta_L^2+\delta_R^2}$ (ns)', '#9b72b0'),
    ]

    for i, (ax, data, title, xlabel, color) in enumerate(panels):
        # Clip at 99th percentile to remove blown-up outliers, then set x-axis
        # tightly around the bulk (95th pct) so the histogram is never squished
        p99   = float(np.percentile(data, 99))
        p95   = float(np.percentile(data, 95))
        clip  = data[data <= p99 * 1.2]
        mu    = float(np.mean(clip))
        sigma = float(np.std(clip))
        xlim_max = max(p95 * 1.5, mu + 4 * sigma, 0.5)  # always show ≥4σ above mean

        ax.hist(clip, bins=40, color=color, alpha=0.75, edgecolor='white', lw=0.3)
        ax.set_xlim([0, xlim_max])
        ax.axvline(mu,         color='r', ls='-',  lw=1.5,
                   label=f'mean = {mu:.1f} ns')
        ax.axvline(mu + sigma, color='r', ls='--', lw=1.2,
                   label=f'±1σ = {sigma:.1f} ns')
        ax.axvline(mu - sigma, color='r', ls='--', lw=1.2)

        # (σ(Δτ) measured line intentionally omitted: it's far off-axis for most waveforms)

        ax.set_title(f'{title}  (N={N})', fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_excel(result, rep_rate, out_path):
    n_eval = result['n_eval'];  n_tmpl = result['n_template']
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


# ── Main loop ─────────────────────────────────────────────────────────────────
total_wf = 0
for session_rel, cfg in SESSIONS:
    session_dir = os.path.join(BASE, session_rel)
    raw_dir     = os.path.join(session_dir, 'raw')
    plots_dir   = os.path.join(session_dir, 'plots')

    if not os.path.isdir(raw_dir):
        print(f'Skipping (no raw/): {session_rel}'); continue

    wavs = sorted(f for f in os.listdir(raw_dir) if f.endswith('_received.wav'))
    print(f'\n{"="*62}')
    print(f'Session: {os.path.basename(session_rel)}   ({len(wavs)} waveforms)')
    print(f'{"="*62}')

    for wav_fname in wavs:
        wf_name  = wav_fname.replace('_received.wav', '')
        wav_path = os.path.join(raw_dir, wav_fname)

        if cfg and wf_name in cfg:
            dur_ms, dead_ms = cfg[wf_name]
        else:
            dur_ms, dead_ms = dur_from_name(wf_name), 15.0

        rep = rr(dur_ms, dead_ms)

        wf_plots_dir = os.path.join(plots_dir, wf_name)
        out_dir      = os.path.join(wf_plots_dir, SUBDIR)
        os.makedirs(out_dir, exist_ok=True)

        print(f'\n  [{total_wf+1}] {wf_name}  (rep={rep:.2f} Hz)', flush=True)

        # Remove stale files from earlier batch runs
        for stale in ('hist_delta_tau.png', 'hist_sigma_delta_tau.png'):
            p = os.path.join(out_dir, stale)
            if os.path.exists(p):
                os.remove(p)

        try:
            r = st.analyze(wav_path, rep_rate=rep, n_template=N_TMPL,
                           output_dir=out_dir, save_plot=False)

            save_3panel(r, wf_name, rep,
                        os.path.join(out_dir, 'stubbs_lsfit.png'))
            save_hist_delta_tau(r['delta_tau_sf'], wf_name,
                                os.path.join(out_dir, 'hist_delta_tau.png'))
            save_hist_uncertainties(r, wf_name,
                                    os.path.join(out_dir, 'hist_uncertainties.png'))
            save_excel(r, rep, os.path.join(out_dir, 'per_period.xlsx'))

            med_sig  = float(np.median(r['sigma_dt_sf'])) * 1e3
            spread   = float(np.std(r['delta_tau_sf']))
            print(f'       N={r["n_eval"]}  σ(Δτ) median={med_sig:.1f} ns  '
                  f'Δτ spread={spread:.4f} μs  → {SUBDIR}/')
            total_wf += 1

        except Exception as exc:
            import traceback
            print(f'       ERROR: {exc}')
            traceback.print_exc()

print(f'\nDone — processed {total_wf} waveforms.')
