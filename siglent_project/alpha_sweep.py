#!/usr/bin/env python3
"""
Alpha-sweep experiment: H(f)^{-alpha} amplitude compensation.
36 chirps: 6 alpha × 3 BW × 2 durations. 10s recording each. ~6.6 min total.

Alpha:    0.0 (flat) → 0.2 → 0.4 → 0.6 → 0.8 → 1.0 (full 1/H)
BW:       small=6kHz, medium=12kHz, large=20kHz  (all centered near 42kHz)
Duration: 2ms, 4ms
"""
import os, sys, time, json
import numpy as np
from datetime import datetime
from scipy.io import wavfile
sys.path.insert(0, os.path.dirname(__file__))
import chirp_sweep as cs

RECORD_S    = 30.0   # extended to 30s for proper Stubbs analysis (was 10s)
ALPHAS      = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
# Recentered on new burst-characterization peak (41450 Hz, Jul 15 2026)
# Old center was ~42000 Hz (from 300ms continuous — standing-wave biased)
BANDWIDTHS  = [
    ('sm',  6_000,  38_450, 44_450),   # small:  6 kHz,  38.45–44.45 kHz  (spans -6dB band)
    ('md', 12_000,  35_450, 47_450),   # medium: 12 kHz, 35.45–47.45 kHz  (spans -10dB band)
    ('lg', 20_000,  31_450, 51_450),   # large:  20 kHz, 31.45–51.45 kHz  (wide)
]
DURATIONS   = [2.0, 4.0]              # ms

# Per-waveform Vpp calibrated to 90% UltraMic FS.
# Measured July 9 from existing 2V recordings; new = 2.0 × (0.90 / observed_max).
VOLTAGE_MAP = {
    'a0.2_sm_2ms': 2.27, 'a0.2_sm_4ms': 2.38,
    'a0.2_md_2ms': 3.17, 'a0.2_md_4ms': 2.92,
    'a0.2_lg_2ms': 3.42, 'a0.2_lg_4ms': 3.25,
    'a0.3_sm_2ms': 2.64, 'a0.3_sm_4ms': 2.60,
    'a0.3_md_2ms': 3.57, 'a0.3_md_4ms': 3.93,
    'a0.3_lg_2ms': 4.07, 'a0.3_lg_4ms': 3.82,
    'a0.4_sm_2ms': 2.58, 'a0.4_sm_4ms': 2.80,
    'a0.4_md_2ms': 4.38, 'a0.4_md_4ms': 4.27,
    'a0.4_lg_2ms': 5.15, 'a0.4_lg_4ms': 5.18,
    'a0.5_sm_2ms': 2.95, 'a0.5_sm_4ms': 2.59,
    'a0.5_md_2ms': 4.59, 'a0.5_md_4ms': 4.24,
    'a0.5_lg_2ms': 6.19, 'a0.5_lg_4ms': 4.67,
    'a0.6_sm_2ms': 3.09, 'a0.6_sm_4ms': 2.96,
    'a0.6_md_2ms': 4.29, 'a0.6_md_4ms': 4.25,
    'a0.6_lg_2ms': 6.27, 'a0.6_lg_4ms': 5.14,
    'a0.8_sm_2ms': 3.36, 'a0.8_sm_4ms': 3.26,
    'a0.8_md_2ms': 5.12, 'a0.8_md_4ms': 4.34,
    'a0.8_lg_2ms': 5.72, 'a0.8_lg_4ms': 5.35,
}

SDG_SR, SDG_MAX = 400000, 16384

def rep_rate(dur_ms):
    buf = min(int((dur_ms + 15) / 1000 * SDG_SR), SDG_MAX)
    return SDG_SR / buf


def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(cs.OUTPUT_DIR, 'recordings', f'{ts}_alpha_sweep_v2_newHf')
    raw_dir = os.path.join(run_dir, 'raw')
    plt_dir = os.path.join(run_dir, 'plots')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(plt_dir, exist_ok=True)

    piezo_response = cs.load_piezo_response()
    if piezo_response is None:
        print("ERROR: piezo response not found"); return 1

    n_chirps = len(ALPHAS) * len(BANDWIDTHS) * len(DURATIONS)
    t_est    = n_chirps * (RECORD_S + 1.0)
    print(f"\nAlpha sweep: {n_chirps} chirps, ~{t_est/60:.1f} min")
    print(f"Alphas:   {ALPHAS}")
    print(f"BWs:      {[b[0] for b in BANDWIDTHS]}")
    print(f"Durations:{DURATIONS} ms\n")

    pa, mic_index = cs.find_ultramic()
    if pa is None: return 1

    results = []
    sdg = None
    try:
        sdg = cs.SDG1062X_LAN()
        sdg.connect()
        print(f"SDG: {sdg.query('*IDN?')}")
        sdg.set_arb_mode(channel=1)

        chirp_n = 0
        for dur_ms in DURATIONS:
            for bw_tag, bw_hz, f_lo, f_hi in BANDWIDTHS:
                for alpha in ALPHAS:
                    chirp_n += 1
                    label    = f'a{alpha:.1f}_{bw_tag}_{dur_ms:.0f}ms'
                    rr       = rep_rate(dur_ms)
                    ampl_vpp = VOLTAGE_MAP.get(label, 2.0)
                    print(f"\n[{chirp_n:02d}/{n_chirps}] {label}  "
                          f"({f_lo/1000:.0f}–{f_hi/1000:.0f}kHz  "
                          f"BW={bw_hz/1000:.0f}kHz  α={alpha}  {ampl_vpp:.2f}Vpp)")

                    # Generate chirp
                    _, chirp, _ = cs.generate_lfm_chirp(
                        f_lo, f_hi, dur_ms, cs.DODOMIC_SAMPLE_RATE,
                        amplitude_shape='alpha_comp',
                        piezo_response=piezo_response,
                        window='tukey', alpha=alpha)

                    sdg_buf, _ = cs.create_sdg_buffer(chirp, dur_ms)
                    wf_name = ('a' + label.replace('.', 'p').replace('-', 'm'))[:15]
                    sdg.upload_waveform(wf_name, sdg_buf, rr, ampl_vpp=ampl_vpp)
                    sdg.output_on()
                    time.sleep(0.3)
                    received = cs.record_audio(pa, mic_index, RECORD_S)
                    sdg.output_off()

                    # Save WAV
                    wav_path = os.path.join(raw_dir, f'{label}_received.wav')
                    wavfile.write(wav_path, cs.DODOMIC_SAMPLE_RATE,
                                  (received * 32767).astype(np.int16))

                    # Per-period σ_TOF
                    tof_mean, tof_std, n_pp, fit_sig = cs.per_period_sigma_tof(
                        received, rr, analysis_duration_s=3.0)
                    if tof_std is not None:
                        fit_s = f"  fit_σ={fit_sig*1000:.0f}ns" if fit_sig else ""
                        print(f"  σ_TOF={tof_std:.3f}μs  {tof_std/10.3:.4f}°C/ping"
                              f"  N={n_pp}{fit_s}")
                    else:
                        print(f"  σ_TOF: not enough periods (N={n_pp})")

                    # Quick autocorr plot
                    mf = cs.matched_filter_analysis(received, chirp, f_lo, f_hi)
                    n50 = int(0.05 * cs.DODOMIC_SAMPLE_RATE)
                    result = dict(
                        f_start=f_lo, f_stop=f_hi, bw_hz=bw_hz,
                        duration_ms=dur_ms, label=label, amp_shape='alpha_comp',
                        alpha=alpha, bw_tag=bw_tag,
                        theo_fwhm_us=cs.theoretical_fwhm_us(f_lo, f_hi),
                        chirp=chirp, received=received[:n50].copy(), mf=mf,
                        sigma_tof_us=tof_std, n_periods=n_pp,
                    )
                    results.append(result)
                    plot_path = os.path.join(plt_dir, f'{label}.png')
                    cs.plot_chirp_result(result, (f_lo, f_hi, dur_ms, label, 'alpha_comp'),
                                         plot_path)
                    del received
                    time.sleep(0.3)

    except Exception as e:
        import traceback; traceback.print_exc()
        return 1
    finally:
        if sdg: sdg.output_off(); sdg.disconnect()
        if pa:  pa.terminate()

    # Summary table
    print(f"\n{'='*90}")
    print(f"{'Label':<28} {'α':>5} {'BW':>6} {'dur':>5} {'σ_TOF μs':>10} {'°C/ping':>9} {'FWHM μs':>9}")
    print('-'*90)
    rows = []
    for r in results:
        s = r['sigma_tof_us']
        f = r['mf']['fwhm_us']
        rows.append((s or 999, r))
    rows.sort(key=lambda x: x[0])
    for _, r in rows:
        s = r['sigma_tof_us']; f = r['mf']['fwhm_us']
        print(f"  {r['label']:<26} {r['alpha']:>5.1f} "
              f"{r['bw_hz']/1000:>5.0f}k {r['duration_ms']:>4.0f}ms "
              f"{s:>10.3f} {s/10.3:>9.4f} {f:>9.0f}" if s else
              f"  {r['label']:<26}  N/A")

    # Summary plot (sigma_tof vs alpha, one curve per BW×duration)
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, dur_ms in zip(axes, DURATIONS):
        for bw_tag, bw_hz, f_lo, f_hi in BANDWIDTHS:
            alphas_plot, sigmas = [], []
            for r in results:
                if r['duration_ms'] == dur_ms and r['bw_tag'] == bw_tag and r['sigma_tof_us']:
                    alphas_plot.append(r['alpha'])
                    sigmas.append(r['sigma_tof_us'])
            if alphas_plot:
                ax.plot(alphas_plot, sigmas, 'o-', lw=1.5, ms=6,
                        label=f'{bw_hz/1000:.0f} kHz BW')
        ax.set_xlabel('Alpha (compensation exponent)')
        ax.set_ylabel('σ_TOF (μs)')
        ax.set_title(f'Duration = {dur_ms:.0f} ms')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_xticks(ALPHAS)
    fig.suptitle('Alpha sweep: H(f)^{-α} compensation vs σ_TOF', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plt_dir, 'alpha_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f'\nResults in: {run_dir}')
    return 0

if __name__ == '__main__':
    sys.exit(main())
