#!/usr/bin/env python3
"""Batch Stubbs analysis on all July 7 recordings. Saves plots + summary table."""
import sys, os, traceback
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import chirp_stubbs_analysis as st

SDG_SR = 400000; SDG_MAX = 16384
def rep_rate(dur_ms):
    return SDG_SR / min(int((dur_ms + 15) / 1000 * SDG_SR), SDG_MAX)

OUT = 'stubbs_batch_july7'
os.makedirs(OUT, exist_ok=True)

RECORDINGS = [
    # (wav_path, rep_rate_hz, session_tag)
    # ── July 7 morning (old mic position) ──────────────────────────────────
    ('recordings/20260707_025425_chirp_sweep/raw/bw_20kHz_wide_4ms_comp_light_received.wav',
     rep_rate(4.0), 'morn_comp_light_20kHz_4ms'),
    ('recordings/20260707_025425_chirp_sweep/raw/bw_20kHz_wide_4ms_comp_received.wav',
     rep_rate(4.0), 'morn_comp_20kHz_4ms'),
    ('recordings/20260707_025425_chirp_sweep/raw/bw_25kHz_wide_2ms_comp_received.wav',
     rep_rate(2.0), 'morn_comp_25kHz_2ms'),
    ('recordings/20260707_025425_chirp_sweep/raw/bw_6kHz_sym_2ms_flat_received.wav',
     rep_rate(2.0), 'morn_flat_6kHz_2ms'),
    ('recordings/20260707_025425_chirp_sweep/raw/hf_m15dB_asym_2ms_comp_light_received.wav',
     rep_rate(2.0), 'morn_comp_light_hf_2ms'),
    ('recordings/20260707_025425_chirp_sweep/raw/hf_m15dB_asym_2ms_comp_received.wav',
     rep_rate(2.0), 'morn_comp_hf_2ms'),
    ('recordings/20260707_025425_chirp_sweep/raw/hf_m15dB_asym_2ms_comp_rect_received.wav',
     rep_rate(2.0), 'morn_comp_rect_hf_2ms'),
    # ── July 7 evening (new mic position) ──────────────────────────────────
    ('recordings/20260707_214657_chirp_sweep/raw/bw_20kHz_wide_4ms_comp_light_received.wav',
     rep_rate(4.0), 'eve_comp_light_20kHz_4ms'),
    ('recordings/20260707_214657_chirp_sweep/raw/bw_20kHz_wide_4ms_comp_received.wav',
     rep_rate(4.0), 'eve_comp_20kHz_4ms'),
    ('recordings/20260707_214657_chirp_sweep/raw/bw_20kHz_wide_4ms_diff_received.wav',
     rep_rate(4.0), 'eve_diff_20kHz_4ms'),
    ('recordings/20260707_214657_chirp_sweep/raw/bw_25kHz_wide_2ms_comp_received.wav',
     rep_rate(2.0), 'eve_comp_25kHz_2ms'),
    ('recordings/20260707_214657_chirp_sweep/raw/bw_6kHz_sym_2ms_flat_received.wav',
     rep_rate(2.0), 'eve_flat_6kHz_2ms'),
    ('recordings/20260707_214657_chirp_sweep/raw/hf_m15dB_asym_2ms_comp_light_received.wav',
     rep_rate(2.0), 'eve_comp_light_hf_2ms'),
    ('recordings/20260707_214657_chirp_sweep/raw/hf_m15dB_asym_2ms_comp_received.wav',
     rep_rate(2.0), 'eve_comp_hf_2ms'),
    ('recordings/20260707_214657_chirp_sweep/raw/hf_m15dB_asym_2ms_comp_rect_received.wav',
     rep_rate(2.0), 'eve_comp_rect_hf_2ms'),
    ('recordings/20260707_214657_chirp_sweep/raw/hf_m15dB_asym_2ms_diff_received.wav',
     rep_rate(2.0), 'eve_diff_hf_2ms'),
    # ── Bayesian optimizer best waveforms ───────────────────────────────────
    ('best_nlfm_20260707_095536.wav',        rep_rate(0.5),  'bayes_nlfm'),
    ('best_lfm_wiener_20260707_095536.wav',  rep_rate(2.275),'bayes_lfm_wiener'),
]

results = []
for wav, rr, tag in RECORDINGS:
    print(f'\n{"="*60}')
    print(f'  {tag}  (rep={rr:.2f} Hz)')
    print(f'{"="*60}')
    try:
        r = st.analyze(wav, rep_rate=rr, output_dir=OUT)
        r['tag'] = tag; r['wav'] = wav; r['rr'] = rr
        results.append(r)

        # Template quality plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        t = np.arange(len(r['T_launch'])) / r['fs'] * 1000
        axes[0].plot(t, r['T_launch'], lw=0.8)
        axes[0].set_title(f'T_launch — {tag}  (peak={np.max(np.abs(r["T_launch"])):.4f})')
        axes[0].set_xlabel('Time (ms)'); axes[0].grid(True, alpha=0.3)
        axes[1].plot(t, r['T_return'], lw=0.8, color='C3')
        axes[1].set_title(f'T_return — {tag}  (peak={np.max(np.abs(r["T_return"])):.4f})')
        axes[1].set_xlabel('Time (ms)'); axes[1].grid(True, alpha=0.3)
        plt.suptitle(f'Templates — {tag}', fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{OUT}/templates_{tag}.png', dpi=130, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f'  FAILED: {e}')
        traceback.print_exc()
        results.append({'tag': tag, 'wav': wav, 'std_delta_tau_us': None,
                        'tau_launch': np.array([0.]), 'tau_return': np.array([0.]),
                        'T_launch': np.array([0.]), 'T_return': np.array([0.]),
                        'fs': 384000, 'rr': rr})

# Summary table
print(f'\n\n{"="*100}')
print(f'{"Tag":<35} {"σ(Δτ) μs":>10} {"°C/ping":>9} {"σ(τL) μs":>10} {"σ(τR) μs":>10} '
      f'{"T_L peak":>9} {"T_R peak":>9}')
print('-'*100)
rows = []
for r in results:
    s = r.get('std_delta_tau_us')
    rows.append((s or 999, r))
rows.sort(key=lambda x: x[0])

for _, r in rows:
    s = r.get('std_delta_tau_us')
    tl = r['tau_launch']; tr = r['tau_return']
    Tl_pk = np.max(np.abs(r['T_launch'])); Tr_pk = np.max(np.abs(r['T_return']))
    if s:
        print(f"  {r['tag']:<33} {s:>10.3f} {s/10.3:>9.4f} "
              f"{tl.std():>10.3f} {tr.std():>10.3f} "
              f"{Tl_pk:>9.4f} {Tr_pk:>9.4f}")
    else:
        print(f"  {r['tag']:<33} {'FAILED':>10}")

# Summary σ(Δτ) bar chart
valid = [(r['tag'], r['std_delta_tau_us']) for _, r in rows if r.get('std_delta_tau_us')]
if valid:
    tags, sigmas = zip(*valid)
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ['C2' if s < 0.5 else 'C1' if s < 1.0 else 'C3' for s in sigmas]
    ax.bar(range(len(tags)), sigmas, color=colors)
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=45, ha='right', fontsize=8)
    ax.axhline(0.5, color='g', lw=1, ls='--', label='0.5 μs = 0.049°C')
    ax.axhline(1.0, color='orange', lw=1, ls='--', label='1.0 μs = 0.097°C')
    ax.set_ylabel('σ(Δτ) μs'); ax.set_title('Stubbs σ(Δτ) — all July 7 recordings')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{OUT}/summary_sigma.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {OUT}/summary_sigma.png')
