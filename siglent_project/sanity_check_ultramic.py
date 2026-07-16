#!/usr/bin/env python3
"""
Sanity check: re-record a small subset of the best alpha/chirp waveforms with
the UltraMic to verify we're still seeing ~2 ns per-period uncertainty and smooth
between-measurement Δτ. Should take ~3 min.

Edit SANITY_CONFIGS below to swap in different waveforms.
"""
import os, sys, time, json
import numpy as np
from datetime import datetime
from scipy.io import wavfile
sys.path.insert(0, os.path.dirname(__file__))
import chirp_sweep as cs

RECORD_S = 30.0

# ── Waveforms to test ─────────────────────────────────────────────────────────
# Each entry: (style, f_lo, f_hi, dur_ms, label, amp_shape, window, ampl_vpp, alpha)
#   style='alpha' uses amplitude_shape='alpha_comp' with given alpha
#   style='chirp' uses the given amp_shape directly
# ampl_vpp: UltraMic-calibrated values from previous sweep (90% FS target).
# ─────────────────────────────────────────────────────────────────────────────
SANITY_CONFIGS = [
    # Best performer from Jul 10 alpha sweep — recentered sm band (38450–44450)
    ('alpha', 38_450, 44_450, 4.0, 'a0.3_sm_4ms', 'alpha_comp', 'tukey', 2.60, 0.3),
    ('alpha', 38_450, 44_450, 2.0, 'a0.3_sm_2ms', 'alpha_comp', 'tukey', 2.64, 0.3),
    # Best wide-BW chirp
    ('chirp', 31_475, 51_475, 4.0, 'bw_20kHz_wide_4ms_comp', 'piezo_compensated', 'tukey', 8.64, None),
    # Asymmetric chirp — updated to new -10dB crossings (37200–45000)
    ('chirp', 37_200, 45_000, 2.0, 'bw_10dB_asym_2ms_comp',  'piezo_compensated', 'tukey', 6.60, None),
]

SDG_SR, SDG_MAX = 400_000, 16_384

def rep_rate(dur_ms):
    buf = min(int((dur_ms + 15) / 1000 * SDG_SR), SDG_MAX)
    return SDG_SR / buf


def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(cs.OUTPUT_DIR, 'recordings', f'{ts}_sanity_check_ultramic_v2_newHf')
    raw_dir = os.path.join(run_dir, 'raw')
    plt_dir = os.path.join(run_dir, 'plots')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(plt_dir, exist_ok=True)

    piezo_response = cs.load_piezo_response()
    if piezo_response is None:
        print("ERROR: piezo response not found"); return 1

    n = len(SANITY_CONFIGS)
    print(f"\nUltraMic sanity check — {n} waveforms, {RECORD_S:.0f} s each")
    print(f"Estimated time: ~{n * (RECORD_S + 2) / 60:.1f} min\n")

    pa, mic_index = cs.find_ultramic()
    if pa is None: return 1

    manifest = {
        'timestamp': datetime.now().isoformat(),
        'description': 'ultramic_sanity_check',
        'hardware': {
            'sdg': f'SDG1062X @ {cs.SDG_IP}',
            'mic': 'Dodotronic UltraMic 384k',
            'mic_sample_rate_hz': cs.DODOMIC_SAMPLE_RATE,
        },
        'measurements': {},
    }

    sdg = None
    try:
        sdg = cs.SDG1062X_LAN(); sdg.connect()
        print(f"SDG: {sdg.query('*IDN?')}")
        sdg.set_arb_mode(channel=1)

        for i, cfg in enumerate(SANITY_CONFIGS):
            style, f_lo, f_hi, dur_ms, label, amp_shape, window, ampl_vpp, alpha = cfg
            rr = rep_rate(dur_ms)
            print(f"\n[{i+1}/{n}] {label}  "
                  f"({f_lo/1000:.0f}–{f_hi/1000:.0f} kHz  {dur_ms:.0f} ms  {ampl_vpp:.2f} Vpp)")

            # Generate chirp
            kw = dict(alpha=alpha) if alpha is not None else {}
            _, chirp, _ = cs.generate_lfm_chirp(
                f_lo, f_hi, dur_ms, cs.DODOMIC_SAMPLE_RATE,
                amplitude_shape=amp_shape, piezo_response=piezo_response,
                window=window, **kw)

            sdg_buf, _ = cs.create_sdg_buffer(chirp, dur_ms)
            wf_name = label.replace('.', 'p').replace('-', 'm')[:15]
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
                fit_s = f"  fit_σ={fit_sig*1000:.0f} ns" if fit_sig else ""
                print(f"  σ_TOF = {tof_std:.3f} μs  →  {tof_std/10.3:.4f}°C/ping"
                      f"  (N={n_pp}){fit_s}")
            else:
                print(f"  σ_TOF: not enough periods (N={n_pp})")

            # Quick autocorr analysis + plot
            mf = cs.matched_filter_analysis(received, chirp, f_lo, f_hi)
            n50 = int(0.05 * cs.DODOMIC_SAMPLE_RATE)
            result = dict(
                f_start=f_lo, f_stop=f_hi, bw_hz=abs(f_hi-f_lo),
                duration_ms=dur_ms, label=label, amp_shape=amp_shape,
                theo_fwhm_us=cs.theoretical_fwhm_us(f_lo, f_hi),
                chirp=chirp, received=received[:n50].copy(), mf=mf,
                sigma_tof_us=tof_std, n_periods=n_pp,
            )
            cs.plot_chirp_result(result, (f_lo, f_hi, dur_ms, label, amp_shape, window),
                                 os.path.join(plt_dir, f'{label}.png'))

            manifest['measurements'][label] = {
                'sigma_tof_us': tof_std,
                'n_periods': n_pp,
                'tof_us': mf['tof_us'],
                'fwhm_us': mf['fwhm_us'],
                'snr_db': mf.get('snr_db'),
                'wav': f'raw/{label}_received.wav',
            }
            del received
            time.sleep(0.3)

    except Exception as e:
        import traceback; traceback.print_exc(); return 1
    finally:
        if sdg: sdg.output_off(); sdg.disconnect()
        if pa:  pa.terminate()

    with open(os.path.join(run_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"{'Label':<28} {'σ_TOF μs':>10} {'°C/ping':>9} {'N':>6}")
    print('-'*60)
    for cfg in SANITY_CONFIGS:
        label = cfg[4]
        m = manifest['measurements'].get(label, {})
        s = m.get('sigma_tof_us')
        n_pp = m.get('n_periods', 0)
        if s is not None:
            print(f"  {label:<26} {s:>10.3f} {s/10.3:>9.4f} {n_pp:>6}")
        else:
            print(f"  {label:<26}  N/A")
    print(f"\nResults in: {run_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
