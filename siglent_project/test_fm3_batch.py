#!/usr/bin/env python3
"""
Full regression test: run chirp_stubbs_analysis_fm3fix on ALL ECM999 waveforms
and compare to original results from per_period.xlsx.
"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use('Agg')

import chirp_stubbs_analysis_fm3fix as st_fix

BASE    = os.path.dirname(os.path.abspath(__file__))
SESSION = 'recordings/20260716_015251_ecm999_full_sweep_v2_ecmHf'
SUBDIR  = 'stubbs_pipeline_lsfit_v1_fm3fix'
SDG_SR, SDG_MAX = 400_000, 16_384

def rr(dur_ms, dead_ms=15.0):
    buf = min(int((dur_ms + dead_ms) / 1000 * SDG_SR), SDG_MAX)
    return SDG_SR / buf

import re
def dur_from(name):
    m = re.search(r'_(\d+)ms', name)
    return float(m.group(1)) if m else 4.0

raw_dir   = os.path.join(BASE, SESSION, 'raw')
plots_dir = os.path.join(BASE, SESSION, 'plots')

wavs = sorted(f for f in os.listdir(raw_dir) if f.endswith('_received.wav'))
print(f"Testing {len(wavs)} waveforms with FM3fix...\n")

results = []
for wav_fname in wavs:
    wf  = wav_fname.replace('_received.wav', '')
    rep = rr(dur_from(wf))
    out = os.path.join(plots_dir, wf, SUBDIR)
    os.makedirs(out, exist_ok=True)

    try:
        r = st_fix.analyze(os.path.join(raw_dir, wav_fname),
                           rep_rate=rep, n_template=100,
                           output_dir=out, save_plot=False)
        dt  = r['delta_tau_sf']
        sp  = r['sigma_dt_sf'] * 1e3
        cl  = np.abs(dt) <= 5.0
        sig = float(np.std(dt[cl])) if cl.sum() > 3 else None
        med = float(np.median(sp[cl])) if cl.sum() > 3 else None
        results.append(dict(label=wf, sigma=sig, med_sp=med,
                            n_clean=int(cl.sum()), n_total=len(dt)))
    except Exception as e:
        import traceback; traceback.print_exc()
        results.append(dict(label=wf, sigma=None, med_sp=None,
                            n_clean=0, n_total=0))

# Load originals from per_period.xlsx
orig = {}
for wf_dir in os.listdir(plots_dir):
    x = os.path.join(plots_dir, wf_dir, 'stubbs_pipeline_lsfit_v1', 'per_period.xlsx')
    if not os.path.exists(x): continue
    df = pd.read_excel(x)
    dt = df['delta_tau_us'].values
    sp = df['sigma_delta_tau_ns'].values
    cl = np.abs(dt) <= 5.0
    orig[wf_dir] = dict(sigma=float(np.std(dt[cl])) if cl.sum()>3 else None,
                        n_clean=int(cl.sum()), n_total=len(dt))

results.sort(key=lambda x: x['sigma'] if x['sigma'] else 999)

print(f"\n{'Waveform':<34} {'FM3 σ':>8} {'Orig σ':>8} {'FM3 N':>7} {'Orig N':>7} {'Δrej':>7} {'FM3 δp':>8}")
print('-'*90)
for r in results:
    o = orig.get(r['label'], {})
    sig_f = r['sigma']; sig_o = o.get('sigma')
    n_f   = r['n_clean']; n_t = r['n_total']
    n_o   = o.get('n_clean', 0); n_ot = o.get('n_total', 1)
    rej_f = (n_t - n_f) / n_t * 100 if n_t else 0
    rej_o = (n_ot - n_o) / n_ot * 100 if n_ot else 0
    d_rej = rej_f - rej_o
    dp    = r['med_sp']
    sig_f_s = f"{sig_f:.3f}" if sig_f else "N/A"
    sig_o_s = f"{sig_o:.3f}" if sig_o else "N/A"
    dp_s    = f"{dp:.0f}" if dp else "N/A"
    flag = '▼' if d_rej < -5 else ('▲' if d_rej > 5 else ' ')
    print(f"{r['label']:<34} {sig_f_s:>8} {sig_o_s:>8} {n_f:>7} {n_o:>7} {d_rej:>+6.0f}% {flag} {dp_s:>7}")

print(f"\nDone. Results in {SUBDIR}/ dirs.")
