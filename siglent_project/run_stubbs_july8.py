#!/usr/bin/env python3
import sys, os, traceback
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import chirp_stubbs_analysis as st

SDG_SR = 400000; SDG_MAX = 16384
def rr(dur): return SDG_SR / min(int((dur+15)/1000*SDG_SR), SDG_MAX)

BASE = 'recordings/20260708_154521_chirp_sweep/raw'
OUT  = 'stubbs_july8'
os.makedirs(OUT, exist_ok=True)

configs = [
    (f'{BASE}/bw_6kHz_sym_2ms_flat_received.wav',          rr(2.0), 'bw_6kHz_flat'),
    (f'{BASE}/bw_20kHz_wide_4ms_comp_light_received.wav',  rr(4.0), 'bw_20kHz_comp_light'),
    (f'{BASE}/bw_20kHz_wide_4ms_comp_received.wav',        rr(4.0), 'bw_20kHz_comp'),
    (f'{BASE}/bw_20kHz_wide_4ms_diff_received.wav',        rr(4.0), 'bw_20kHz_diff'),
    (f'{BASE}/bw_25kHz_wide_2ms_comp_received.wav',        rr(2.0), 'bw_25kHz_comp'),
    (f'{BASE}/hf_m15dB_asym_2ms_comp_light_received.wav',  rr(2.0), 'hf_comp_light'),
    (f'{BASE}/hf_m15dB_asym_2ms_comp_received.wav',        rr(2.0), 'hf_comp'),
    (f'{BASE}/hf_m15dB_asym_2ms_comp_rect_received.wav',   rr(2.0), 'hf_comp_rect'),
    (f'{BASE}/hf_m15dB_asym_2ms_diff_received.wav',        rr(2.0), 'hf_diff'),
]

results = []
for wav, rep, tag in configs:
    print(f'\n{"="*55}')
    print(f'  {tag}')
    print(f'{"="*55}')
    try:
        r = st.analyze(wav, rep_rate=rep, output_dir=OUT)
        r['tag'] = tag
        results.append(r)
    except Exception as e:
        traceback.print_exc()
        results.append({'tag': tag, 'std_delta_tau_us': None,
                        'tau_launch': np.array([0.]), 'tau_return': np.array([0.]),
                        'T_launch': np.array([0.]), 'T_return': np.array([0.]),
                        'fs': 384000})

print(f'\n{"="*90}')
print(f'{"Tag":<30} {"σ(Δτ) μs":>10} {"°C/ping":>9} {"σ(τL)":>8} {"σ(τR)":>8} {"T_L pk":>8} {"T_R pk":>8}')
print('-'*90)
rows = sorted(results, key=lambda r: r.get('std_delta_tau_us') or 999)
for r in rows:
    s = r.get('std_delta_tau_us')
    if s:
        tl = r['tau_launch']; tr = r['tau_return']
        print(f"  {r['tag']:<28} {s:>10.3f} {s/10.3:>9.4f} "
              f"{tl.std():>8.3f} {tr.std():>8.3f} "
              f"{np.max(np.abs(r['T_launch'])):>8.4f} {np.max(np.abs(r['T_return'])):>8.4f}")
    else:
        print(f"  {r['tag']:<28} {'FAILED':>10}")
