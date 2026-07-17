#!/usr/bin/env python3
"""
For each ECM999 waveform, find outlier periods (|Δτ|>5μs) and generate
a 3-column inspection plot (launch xcorr | return xcorr | period waveform
+ windows) identical in format to xcorr_outlier_inspection_v2.png.

One good period is always shown as row 0 for comparison.
Up to MAX_BAD_ROWS outlier periods are shown (worst first by |Δτ|).
"""
import sys, os, re, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.io import wavfile
from scipy.signal import hilbert, correlate
import chirp_stubbs_analysis_fm3fix as st

BASE    = os.path.dirname(os.path.abspath(__file__))
SESSION = 'recordings/20260716_015251_ecm999_full_sweep_v2_ecmHf'
SUBDIR  = 'stubbs_pipeline_lsfit_v1_fm3fix'
SDG_SR, SDG_MAX = 400_000, 16_384
MAX_BAD_ROWS    = 3    # max outlier periods to show per waveform

def rr(dur_ms, dead_ms=15.0):
    buf = min(int((dur_ms + dead_ms) / 1000 * SDG_SR), SDG_MAX)
    return SDG_SR / buf

def dur_from(name):
    m = re.search(r'_(\d+)ms', name)
    return float(m.group(1)) if m else 4.0


def make_inspection_plot(wf, wav_path, rep, out_dir):
    xlsx = os.path.join(out_dir, 'per_period.xlsx')
    if not os.path.exists(xlsx):
        # Run FM3fix analysis to generate per_period.xlsx
        print(f'  {wf}: running analyze() to generate per_period.xlsx...')
        r = st.analyze(wav_path, rep_rate=rep, n_template=100,
                       output_dir=out_dir, save_plot=False)
        dt  = r['delta_tau_sf']
        sp  = r['sigma_dt_sf'] * 1e3
        n_t = r['n_template']
        n_e = r['n_eval']
        df_out = pd.DataFrame({
            'period_idx':         np.arange(n_e) + n_t,
            'time_s':             (np.arange(n_e) + n_t) / rep,
            'tau_L_us':           r['tau_launch_sf'],
            'tau_R_us':           r['tau_return_sf'],
            'delta_tau_us':       dt,
            'sigma_L_ns':         r['sigma_launch_sf'] * 1e3,
            'sigma_R_ns':         r['sigma_return_sf'] * 1e3,
            'sigma_delta_tau_ns': sp,
        })
        df_out.to_excel(xlsx, index=False)

    df   = pd.read_excel(xlsx)
    bad  = df[np.abs(df['delta_tau_us']) > 5.0].sort_values('delta_tau_us',
               key=abs, ascending=False).head(MAX_BAD_ROWS)

    if len(bad) == 0:
        print(f'  {wf}: no outliers, skipping inspection plot')
        return

    print(f'  {wf}: {len(bad)} outlier periods → generating inspection plot')

    fs, raw = wavfile.read(wav_path)
    data    = raw.astype(np.float64) / 32768.0
    sig     = st.hp_filter(data, fs)
    slices  = st.adaptive_slice_periods(sig, fs, rep)
    iw, rw  = st.detect_windows(sig[slices[0][0]:slices[0][1]], fs)

    period_idx = slices[0][1] - slices[0][0]
    TL_acc = np.zeros(period_idx); TR_acc = np.zeros(period_idx)
    for i in range(min(100, len(slices))):
        p = sig[slices[i][0]:slices[i][1]]
        TL_acc += st.mask_period(p, fs, iw)
        TR_acc += st.mask_period(p, fs, rw)
    n_tmpl = min(100, len(slices))
    TL = TL_acc / n_tmpl; TR = TR_acc / n_tmpl

    iw_s = int(iw[0]/1000*fs); iw_e = int(iw[1]/1000*fs)
    rw_s = int(rw[0]/1000*fs); rw_e = int(rw[1]/1000*fs)
    rms_TL = np.sqrt(np.mean(TL[iw_s:iw_e]**2))
    rms_TR = np.sqrt(np.mean(TR[rw_s:rw_e]**2))

    # Pick a representative good period (near median clean Δτ)
    good_df  = df[np.abs(df['delta_tau_us']) <= 5.0]
    med_dt   = float(np.median(good_df['delta_tau_us']))
    good_row = good_df.iloc[(good_df['delta_tau_us'] - med_dt).abs().argsort()[:1]]
    good_pidx = int(good_row['period_idx'].iloc[0])

    target_info = [(good_pidx, f'Period {good_pidx} (t={good_row["time_s"].iloc[0]:.1f}s)  GOOD', '#1a4a8a')]
    for _, row in bad.iterrows():
        pi  = int(row['period_idx'])
        dt  = float(row['delta_tau_us'])
        ts  = float(row['time_s'])
        target_info.append((pi, f'Period {pi} (t={ts:.1f}s)  BAD  Δτ={dt:.1f}μs', '#8a1a1a'))

    n_rows = len(target_info)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 3.5*n_rows))
    if n_rows == 1: axes = axes[np.newaxis, :]
    fig.suptitle(f'{wf} — xcorr + period windowing: outlier inspection\n'
                 f'iw=[{iw[0]:.1f},{iw[1]:.1f}ms]  rw=[{rw[0]:.1f},{rw[1]:.1f}ms]',
                 fontsize=11, fontweight='bold')

    zoom = 600
    for row_i, (pi, lbl, rc) in enumerate(target_info):
        p    = sig[slices[pi][0]:slices[pi][1]]
        t_ms = np.arange(len(p)) / fs * 1000
        iw_m = st.mask_period(p, fs, iw)
        rw_m = st.mask_period(p, fs, rw)

        xc_l = correlate(iw_m, TL, mode='full')
        xc_r = correlate(rw_m, TR, mode='full')
        ctr  = period_idx - 1
        lags = (np.arange(len(xc_l)) - ctr) / fs * 1e6
        xc_l /= max(np.max(np.abs(xc_l)), 1e-12)
        xc_r /= max(np.max(np.abs(xc_r)), 1e-12)
        env_l = np.abs(hilbert(xc_l)); env_r = np.abs(hilbert(xc_r))
        m = (lags >= -zoom) & (lags <= zoom)

        for col, (xc, env, color, side) in enumerate([
                (xc_l, env_l, rc, 'Launch xcorr'),
                (xc_r, env_r, rc, 'Return xcorr')]):
            ax = axes[row_i, col]
            ax.plot(lags[m], xc[m], '-', lw=0.4, alpha=0.3, color=color)
            ax.plot(lags[m], env[m], '-', lw=2,   color=color)
            pk = lags[m][np.argmax(env[m])]
            ax.axvline(pk, color='k', ls='--', lw=1.2, label=f'τ={pk:.1f} μs')
            ax.axvline(0, color='gray', ls=':', lw=0.8)
            ax.set_title(f'{lbl}\n{side}', fontsize=8,
                         fontweight='bold' if row_i==0 else 'normal', color=color)
            ax.set_xlabel('Lag (μs)', fontsize=8); ax.legend(fontsize=7)
            if col == 0: ax.set_ylabel('Norm xcorr', fontsize=8)
            ax.grid(True, alpha=0.3)

        # Col 2: period waveform + windows
        ax = axes[row_i, 2]
        ax.plot(t_ms, p, 'k-', lw=0.5, alpha=0.6, label='HP-filtered period')
        ax.axvspan(iw[0], iw[1], alpha=0.18, color='royalblue',
                   label=f'incident [{iw[0]:.1f}–{iw[1]:.1f}ms]')
        ax.axvspan(rw[0], rw[1], alpha=0.18, color='crimson',
                   label=f'return [{rw[0]:.1f}–{rw[1]:.1f}ms]')
        for x in [iw[0], iw[1]]: ax.axvline(x, color='royalblue', lw=1, ls='--', alpha=0.8)
        for x in [rw[0], rw[1]]: ax.axvline(x, color='crimson',   lw=1, ls='--', alpha=0.8)

        rms_inc = np.sqrt(np.mean(p[iw_s:iw_e]**2))
        rms_ret = np.sqrt(np.mean(p[rw_s:rw_e]**2))
        note = (f'RMS inc = {rms_inc:.4f}  ({rms_inc/rms_TL*100:.0f}% of T_launch)\n'
                f'RMS ret = {rms_ret:.4f}  ({rms_ret/rms_TR*100:.0f}% of T_return)')
        ax.text(0.02, 0.97, note, transform=ax.transAxes, fontsize=7.5,
                va='top', family='monospace',
                bbox=dict(boxstyle='round', fc='white', alpha=0.85, ec='#aaa'))
        ax.set_title('Period waveform + windows', fontsize=8)
        ax.set_xlabel('Time in period (ms)', fontsize=8)
        ax.set_ylabel('Amplitude', fontsize=8)
        ax.legend(fontsize=6.5, loc='upper right'); ax.grid(True, alpha=0.3)
        ax.set_xlim(0, len(p)/fs*1000)

    plt.tight_layout()
    out_png = os.path.join(out_dir, 'xcorr_outlier_inspection.png')
    plt.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'    saved: {out_png}')


raw_dir   = os.path.join(BASE, SESSION, 'raw')
plots_dir = os.path.join(BASE, SESSION, 'plots')

wavs = sorted(f for f in os.listdir(raw_dir) if f.endswith('_received.wav'))
print(f'Processing {len(wavs)} waveforms...\n')

for wav_fname in wavs:
    wf      = wav_fname.replace('_received.wav', '')
    rep     = rr(dur_from(wf))
    out_dir = os.path.join(plots_dir, wf, SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    try:
        make_inspection_plot(wf, os.path.join(raw_dir, wav_fname), rep, out_dir)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f'  ERROR {wf}: {e}')

print('\nDone.')
