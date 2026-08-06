#!/usr/bin/env python3
"""
Full alpha + chirp sweep recorded via ECM999 + SR560 (×100) + PicoScope 5444D.

Signal chain:
  SDG Ch1 → piezo array → air → reflector → air → ECM999 → phantom box
  → SR560 (×100 gain) → BNC → PicoScope Ch A

Recording: PicoScope block capture at ~384 kHz, 30 s per waveform.
Output WAV files are saved at DODOMIC_SAMPLE_RATE (384000 Hz) so all existing
chirp_stubbs_analysis.py pipelines work without modification.

VOLTAGE MAP NOTE:
  Vpp values below were calibrated for UltraMic (90% FS target).  The ECM999 +
  SR560 chain has different sensitivity — the PicoScope overflow warning will
  flag clipping.  If overflow is reported, reduce ampl_vpp for that waveform
  and re-run.  If signal is very weak, increase it.  Ch A range is ±2 V.

Usage:
  DYLD_LIBRARY_PATH=/Library/Frameworks/PicoSDK.framework/Libraries/libps5000a \
  python ecm999_full_sweep.py
"""
import os, sys, time, json, ctypes
import numpy as np
from datetime import datetime
from scipy.io import wavfile
sys.path.insert(0, os.path.dirname(__file__))
import chirp_sweep as cs

from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import adc2mV, assert_pico_ok

# ═══════════════════════════════════════════════════════════════════════════════
# Settings
# ═══════════════════════════════════════════════════════════════════════════════

RECORD_S       = 30.0

# PicoScope — 12-bit, single channel A
# TIMEBASE=64 → 1.025 MSa/s (verified by ps5000aGetTimebase2 on PicoScope 5444D,
# 1-channel 12-bit). Gives 25.6 samples/cycle at 40 kHz (vs ~9.6 at old 384 kHz).
RESOLUTION     = ps.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_12BIT"]
TIMEBASE       = 64             # → 1.025 MSa/s  (was 165 → 384 kHz, now corrected)
CH_A_RANGE     = ps.PS5000A_RANGE["PS5000A_2V"]       # ±2 V  (SR560 clips at ±1 V output)
CH_A_COUPLING  = ps.PS5000A_COUPLING["PS5000A_AC"]    # block DC offset from SR560

SDG_SR, SDG_MAX = 400_000, 16_384

# ═══════════════════════════════════════════════════════════════════════════════
# Alpha sweep configs  (mirrors alpha_sweep.py exactly)
# ═══════════════════════════════════════════════════════════════════════════════

ALPHAS     = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
# Recentered on new burst-characterization peak (41450 Hz, Jul 15 2026)
BANDWIDTHS = [
    ('sm',  6_000,  38_450, 44_450),
    ('md', 12_000,  35_450, 47_450),
    ('lg', 20_000,  31_450, 51_450),
]
DURATIONS  = [2.0, 4.0]

# UltraMic-calibrated Vpp values — may need adjustment for ECM999 sensitivity.
# Reduce if PicoScope reports overflow; increase if signal is too weak.
VOLTAGE_MAP = {
    # Recalibrated 2026-07-28 BSWV H(f), direct SDG (no SPA1010), burst-targeted metric, max 20V
    'a0.2_sm_2ms': 20.00, 'a0.2_sm_4ms': 20.00,
    'a0.2_md_2ms': 20.00, 'a0.2_md_4ms': 20.00,
    'a0.2_lg_2ms': 20.00, 'a0.2_lg_4ms': 20.00,
    'a0.3_sm_2ms': 20.00, 'a0.3_sm_4ms': 20.00,
    'a0.3_md_2ms': 20.00, 'a0.3_md_4ms': 20.00,
    'a0.3_lg_2ms': 20.00, 'a0.3_lg_4ms': 20.00,
    'a0.4_sm_2ms': 20.00, 'a0.4_sm_4ms': 20.00,
    'a0.4_md_2ms': 20.00, 'a0.4_md_4ms': 20.00,
    'a0.4_lg_2ms': 20.00, 'a0.4_lg_4ms': 20.00,
    'a0.5_sm_2ms': 20.00, 'a0.5_sm_4ms': 20.00,
    'a0.5_md_2ms': 20.00, 'a0.5_md_4ms': 20.00,
    'a0.5_lg_2ms': 20.00, 'a0.5_lg_4ms': 20.00,
    'a0.6_sm_2ms': 20.00, 'a0.6_sm_4ms': 18.34,
    'a0.6_md_2ms': 17.43, 'a0.6_md_4ms': 20.00,
    'a0.6_lg_2ms': 20.00, 'a0.6_lg_4ms': 20.00,
    'a0.8_sm_2ms': 20.00, 'a0.8_sm_4ms': 20.00,
    'a0.8_md_2ms': 20.00, 'a0.8_md_4ms': 20.00,
    'a0.8_lg_2ms': 20.00, 'a0.8_lg_4ms': 20.00,
}

# ═══════════════════════════════════════════════════════════════════════════════
# PicoScope helpers
# ═══════════════════════════════════════════════════════════════════════════════

def open_picoscope():
    """
    Open PicoScope 5444D, enable Ch A only (ECM999 signal), return (chandle, actual_fs).
    actual_fs is the true sample rate in Hz — used to set WAV file sample rate.
    """
    status  = {}
    chandle = ctypes.c_int16()

    status["open"] = ps.ps5000aOpenUnit(ctypes.byref(chandle), None, RESOLUTION)
    assert_pico_ok(status["open"])

    # Ch A — mic via SR560
    chA = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
    status["chA"] = ps.ps5000aSetChannel(chandle, chA, 1,
                                          CH_A_COUPLING, CH_A_RANGE, 0.0)
    assert_pico_ok(status["chA"])

    # Disable B, C, D
    for ch_name in ["PS5000A_CHANNEL_B", "PS5000A_CHANNEL_C", "PS5000A_CHANNEL_D"]:
        ps.ps5000aSetChannel(chandle,
                             ps.PS5000A_CHANNEL[ch_name], 0,
                             ps.PS5000A_COUPLING["PS5000A_DC"],
                             ps.PS5000A_RANGE["PS5000A_1V"], 0.0)

    # Verify actual sample interval
    max_samples = ctypes.c_int32(1)
    dt_ns       = ctypes.c_float()
    status["tb"] = ps.ps5000aGetTimebase2(chandle, TIMEBASE, 1,
                                           ctypes.byref(dt_ns),
                                           ctypes.byref(max_samples), 0)
    assert_pico_ok(status["tb"])
    actual_dt_ns = float(dt_ns.value)
    actual_fs    = 1e9 / actual_dt_ns
    print(f"  PicoScope: {actual_dt_ns:.1f} ns/sample = {actual_fs/1000:.1f} kHz")
    return chandle, actual_fs


def record_picoscope(chandle, actual_fs, duration_s):
    """
    Block capture of duration_s seconds from Ch A.
    Returns float64 array normalised to ±1 FS.
    """
    n_samples = int(round(actual_fs * duration_s))
    chA = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]

    # Auto-trigger: disabled (trigger source = ch A, threshold direction = NONE,
    # auto-trigger delay = 1 ms so it fires immediately)
    ps.ps5000aSetSimpleTrigger(chandle, 0, chA, 0,
                               ps.PS5000A_THRESHOLD_DIRECTION["PS5000A_NONE"],
                               0, 1)

    ps.ps5000aRunBlock(chandle, 0, n_samples, TIMEBASE, None, 0, None, None)

    # Poll until ready — capture takes duration_s seconds
    ready = ctypes.c_int16(0)
    while ready.value == 0:
        time.sleep(0.2)
        ps.ps5000aIsReady(chandle, ctypes.byref(ready))

    buf_A = (ctypes.c_int16 * n_samples)()
    ps.ps5000aSetDataBuffer(chandle, chA, ctypes.byref(buf_A),
                            n_samples, 0,
                            ps.PS5000A_RATIO_MODE["PS5000A_RATIO_MODE_NONE"])

    overflow  = ctypes.c_int16()
    n_ret     = ctypes.c_uint32(n_samples)
    ps.ps5000aGetValues(chandle, 0, ctypes.byref(n_ret), 1, 0, 0,
                        ctypes.byref(overflow))

    if overflow.value:
        print("  WARNING: Ch A overflow — SR560 or PicoScope clipping. "
              "Reduce ampl_vpp for this waveform.")

    maxADC = ctypes.c_int16()
    ps.ps5000aMaximumValue(chandle, ctypes.byref(maxADC))
    mV = np.array(adc2mV(buf_A, CH_A_RANGE, maxADC), dtype=np.float32)

    # Normalise to ±1 FS (matching UltraMic int16 convention)
    # ±2 V range → 2000 mV full scale
    return (mV / 2000.0).astype(np.float64)


def rep_rate(dur_ms):
    buf = min(int((dur_ms + 15) / 1000 * SDG_SR), SDG_MAX)
    return SDG_SR / buf


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--hf', default=None,
                    help='Path to H(f) NPZ for pre-compensation (overrides auto-select)')
    ap.add_argument('--suffix', default='',
                    help='Extra label appended to run directory (e.g. "enclosure")')
    args = ap.parse_args()

    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    sfx     = f'_{args.suffix}' if args.suffix else ''
    run_dir = os.path.join(cs.OUTPUT_DIR, 'recordings', f'{ts}_ecm999_full_sweep_v3_ecmHf{sfx}')
    raw_dir = os.path.join(run_dir, 'raw')
    plt_dir = os.path.join(run_dir, 'plots')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(plt_dir, exist_ok=True)

    if args.hf:
        cs.PIEZO_CHAR_FILE = args.hf
    else:
        import glob as _glob
        _burst_hf = sorted(_glob.glob(
            os.path.join(cs.OUTPUT_DIR, 'ecm999_stepped_sine_burst8ms_*.npz')))
        _bswv_hf  = os.path.join(cs.OUTPUT_DIR,
                                 'ecm999_stepped_sine_bswv_20260716_013405.npz')
        # Always prefer the BSWV (continuous sine) measurement — it has far better
        # SNR than the burst measurement and gives the correct narrow-band H(f).
        cs.PIEZO_CHAR_FILE = _bswv_hf if os.path.isfile(_bswv_hf) else (_burst_hf[-1] if _burst_hf else None)
    print(f"  Using H(f): {os.path.basename(cs.PIEZO_CHAR_FILE)}")
    piezo_response = cs.load_piezo_response()
    if piezo_response is None:
        print("ERROR: piezo response not found"); return 1

    # Build full waveform list: all alpha × BW × duration, then all chirp configs
    alpha_jobs = []
    for dur_ms in DURATIONS:
        for bw_tag, bw_hz, f_lo, f_hi in BANDWIDTHS:
            for alpha in ALPHAS:
                label    = f'a{alpha:.1f}_{bw_tag}_{dur_ms:.0f}ms'
                ampl_vpp = VOLTAGE_MAP.get(label, 2.0)
                alpha_jobs.append({
                    'style': 'alpha', 'f_lo': f_lo, 'f_hi': f_hi,
                    'dur_ms': dur_ms, 'label': label,
                    'amp_shape': 'alpha_comp', 'window': 'tukey',
                    'ampl_vpp': ampl_vpp, 'alpha': alpha,
                })

    chirp_jobs = []
    for cfg in cs.CHIRP_CONFIGS:
        f_start, f_stop, dur_ms, label, amp_shape = cfg[:5]
        window   = cfg[5] if len(cfg) > 5 else 'tukey'
        ampl_vpp = cfg[6] if len(cfg) > 6 else 2.0
        dead_ms  = cfg[7] if len(cfg) > 7 else 15.0
        chirp_jobs.append({
            'style': 'chirp', 'f_lo': f_start, 'f_hi': f_stop,
            'dur_ms': dur_ms, 'label': label,
            'amp_shape': amp_shape, 'window': window,
            'ampl_vpp': ampl_vpp, 'alpha': None, 'dead_ms': dead_ms,
        })

    all_jobs = alpha_jobs + chirp_jobs
    n_total  = len(all_jobs)
    t_est    = n_total * (RECORD_S + 3.0)

    print(f"\nECM999 full sweep — {len(alpha_jobs)} alpha + {len(chirp_jobs)} chirp "
          f"= {n_total} waveforms")
    print(f"Estimated time: ~{t_est/60:.0f} min\n")
    print("Ensure: ECM999 phantom box ON, SR560 gain = ×100, BNC → PicoScope Ch A\n")

    print("Opening PicoScope...")
    chandle, actual_fs = open_picoscope()
    wav_fs = int(round(actual_fs))  # e.g. 383977 → 383977  (close enough to 384000)

    manifest = {
        'timestamp':  datetime.now().isoformat(),
        'description': 'ecm999_full_sweep',
        'hardware': {
            'sdg':           f'SDG1062X @ {cs.SDG_IP}',
            'mic':           'Superlux ECM999 + SR560 (×100) + PicoScope 5444D',
            'mic_sample_rate_hz': wav_fs,
            'picoscope_timebase': TIMEBASE,
            'picoscope_actual_fs_hz': actual_fs,
            'ch_a_range':    '±2 V AC',
        },
        'measurements': {},
    }

    sdg = None
    try:
        sdg = cs.SDG1062X_LAN(); sdg.connect()
        print(f"SDG: {sdg.query('*IDN?')}")
        sdg.set_arb_mode(channel=1)

        for job_n, job in enumerate(all_jobs):
            f_lo, f_hi   = job['f_lo'], job['f_hi']
            dur_ms        = job['dur_ms']
            label         = job['label']
            amp_shape     = job['amp_shape']
            window        = job['window']
            ampl_vpp      = job['ampl_vpp']
            alpha         = job['alpha']
            dead_ms       = job.get('dead_ms', 15.0)

            if amp_shape in ('piezo_compensated', 'piezo_compensated_light',
                             'piezo_compensated_diffraction') and piezo_response is None:
                print(f"  Skipping {label} (no piezo response)")
                continue

            rr = rep_rate(dur_ms)
            print(f"\n[{job_n+1:02d}/{n_total}] {label}  "
                  f"({f_lo/1000:.0f}–{f_hi/1000:.0f} kHz  "
                  f"{dur_ms:.0f} ms  {ampl_vpp:.2f} Vpp)")

            # Generate chirp at SDG/analysis sample rates
            kw = dict(alpha=alpha) if alpha is not None else {}
            # SDG-rate chirp (384 kHz) — for waveform upload and SDG buffer
            _, chirp_sdg, _ = cs.generate_lfm_chirp(
                f_lo, f_hi, dur_ms, cs.DODOMIC_SAMPLE_RATE,
                amplitude_shape=amp_shape, piezo_response=piezo_response,
                window=window, **kw)
            chirp = chirp_sdg   # alias kept for SDG upload below
            # PicoScope-rate chirp (actual_fs) — for matched filter analysis
            _, chirp_rx, _ = cs.generate_lfm_chirp(
                f_lo, f_hi, dur_ms, actual_fs,
                amplitude_shape=amp_shape, piezo_response=piezo_response,
                window=window, **kw)

            sdg_buf, _ = cs.create_sdg_buffer(chirp, dur_ms, dead_ms=dead_ms)
            wf_name = label.replace('.', 'p').replace('-', 'm')[:15]
            sdg.upload_waveform(wf_name, sdg_buf, rr, ampl_vpp=ampl_vpp)
            sdg.output_on()
            time.sleep(0.5)          # slightly longer settle for ECM999 phantom box

            print(f"  Recording {RECORD_S:.0f} s ...", end=' ', flush=True)
            received = record_picoscope(chandle, actual_fs, RECORD_S)
            print("done")
            sdg.output_off()
            sdg.delete_waveform(wf_name)   # free SDG Flash storage slot

            # Save WAV — use actual PicoScope sample rate so analysis tools are exact
            wav_path = os.path.join(raw_dir, f'{label}_received.wav')
            wavfile.write(wav_path, wav_fs,
                          (received * 32767).astype(np.int16))

            # Per-period σ_TOF (uses DODOMIC_SAMPLE_RATE internally; actual_fs ≈ 384 kHz
            # so the 0.006% difference is negligible for σ_TOF estimation)
            tof_mean, tof_std, n_pp, fit_sig = cs.per_period_sigma_tof(
                received, rr, analysis_duration_s=3.0, fs=actual_fs)
            if tof_std is not None:
                fit_s = f"  fit_σ={fit_sig*1000:.0f} ns" if fit_sig else ""
                print(f"  σ_TOF = {tof_std:.3f} μs  →  {tof_std/10.3:.4f}°C/ping"
                      f"  (N={n_pp}){fit_s}")
            else:
                print(f"  σ_TOF: not enough periods (N={n_pp})")

            # Quick autocorr plot
            mf = cs.matched_filter_analysis(received, chirp_rx, f_lo, f_hi,
                                            fs=actual_fs)
            n50 = int(0.05 * actual_fs)   # 50 ms at actual PicoScope rate
            result = dict(
                f_start=f_lo, f_stop=f_hi, bw_hz=abs(f_hi-f_lo),
                duration_ms=dur_ms, label=label, amp_shape=amp_shape,
                theo_fwhm_us=cs.theoretical_fwhm_us(f_lo, f_hi),
                chirp=chirp_rx, received=received[:n50].copy(), mf=mf,
                sigma_tof_us=tof_std, n_periods=n_pp,
            )
            cs.plot_chirp_result(result, (f_lo, f_hi, dur_ms, label, amp_shape, window),
                                 os.path.join(plt_dir, f'{label}.png'))

            manifest['measurements'][label] = {
                'style':        job['style'],
                'sigma_tof_us': tof_std,
                'n_periods':    n_pp,
                'tof_us':       mf['tof_us'],
                'fwhm_us':      mf['fwhm_us'],
                'snr_db':       mf.get('snr_db'),
                'ampl_vpp':     ampl_vpp,
                'wav':          f'raw/{label}_received.wav',
            }
            del received
            time.sleep(0.3)

    except Exception as e:
        import traceback; traceback.print_exc()
        return 1
    finally:
        if sdg: sdg.output_off(); sdg.disconnect()
        try:
            ps.ps5000aStop(chandle)
            ps.ps5000aCloseUnit(chandle)
        except Exception:
            pass

    with open(os.path.join(run_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    # Summary table
    print(f"\n{'='*75}")
    print(f"{'Label':<28} {'σ_TOF μs':>10} {'°C/ping':>9} {'N':>6} {'Vpp':>6}")
    print('-'*75)
    jobs_by_sigma = sorted(
        [(j['label'], manifest['measurements'].get(j['label'], {})) for j in all_jobs],
        key=lambda x: x[1].get('sigma_tof_us') or 999)
    for label, m in jobs_by_sigma:
        s = m.get('sigma_tof_us')
        n_pp = m.get('n_periods', 0)
        vpp  = m.get('ampl_vpp', 0)
        if s is not None:
            print(f"  {label:<26} {s:>10.3f} {s/10.3:>9.4f} {n_pp:>6} {vpp:>6.2f}")
        else:
            print(f"  {label:<26}  N/A")

    print(f"\nResults in: {run_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
