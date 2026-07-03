#!/usr/bin/env python3
"""
Reanalyze saved chirp WAV files using full-signal autocorrelation.

No period slicing — avoids the REP_RATE mismatch and envelope/trigger bugs
in analyze_wav_autocorr_v1.py. The full-signal autocorrelation is robust:
all 20-40 repetitions contribute to the TOF peak, and no per-period
phase alignment is needed.

Filter: HP 30 kHz order-8 SOS (same as optimized Gaussian/Barker analysis)
TOF window: 5000-6500 μs (tight around expected ~5700 μs)
"""

import numpy as np
import json
import os
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, hilbert
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUN_DIR   = '/Users/cyrusurheim_1/Downloads/Stubbs_Stuff/Claude_Dir/siglent_project/recordings/20260702_211222_chirp_sweep'
OUT_DIR   = '/Users/cyrusurheim_1/Downloads/Stubbs_Stuff/Claude_Dir/siglent_project'
FS        = 384000
HP_CUTOFF = 30000
HP_ORDER  = 8
TOF_MIN   = 5000   # μs
TOF_MAX   = 6500   # μs
LAG_PLOT  = 10000  # μs half-width for autocorr overview plot


def hp_filter(sig, fs=FS, cutoff=HP_CUTOFF, order=HP_ORDER):
    sos = butter(order, cutoff / (fs / 2), btype='high', output='sos')
    return sosfiltfilt(sos, sig)


def parabolic_peak(env, lags_us, lo, hi):
    """Argmax + parabolic sub-sample in [lo, hi] μs. Returns (peak_us, peak_val)."""
    mask = (lags_us >= lo) & (lags_us <= hi)
    if not np.any(mask):
        return None, None
    region = env[mask]
    lags_r = lags_us[mask]
    i = np.argmax(region)
    if i < 1 or i >= len(region) - 1:
        return float(lags_r[i]), float(region[i])
    ym, y0, yp = region[i-1], region[i], region[i+1]
    denom = 2 * (ym - 2*y0 + yp)
    delta = np.clip((ym - yp) / denom, -0.5, 0.5) if abs(denom) > 1e-10 else 0.0
    step  = float(lags_r[1] - lags_r[0])
    return float(lags_r[i]) + delta * step, float(y0 - (ym - yp) * delta / 4)


def fwhm_at(env, lags_us, peak_us, peak_val):
    """Half-max FWHM centred on peak_us."""
    idx  = int(np.argmin(np.abs(lags_us - peak_us)))
    half = peak_val * 0.5
    li = idx
    while li > 0 and env[li] > half:
        li -= 1
    ri = idx
    while ri < len(env) - 1 and env[ri] > half:
        ri += 1
    return (ri - li) / FS * 1e6


def analyze_wav(wav_path, label, f_start, f_stop, dur_ms, bw_hz, amp_shape):
    fs, raw = wavfile.read(wav_path)
    assert fs == FS, f"Expected {FS} Hz, got {fs}"
    data = raw.astype(np.float64) / 32768.0 if raw.dtype == np.int16 else raw.astype(np.float64)

    sig   = hp_filter(data)
    acorr = np.correlate(sig, sig, mode='full')
    ctr   = len(acorr) // 2
    lags  = (np.arange(len(acorr)) - ctr) / FS * 1e6   # μs

    acorr /= np.max(np.abs(acorr))
    env    = np.abs(hilbert(acorr))
    env   /= np.max(env)

    # Lag-0 FWHM
    half0  = 0.5
    li0 = ctr
    while li0 > 0 and env[li0] > half0: li0 -= 1
    ri0 = ctr
    while ri0 < len(env)-1 and env[ri0] > half0: ri0 += 1
    lag0_fwhm = (ri0 - li0) / FS * 1e6

    # TOF peak
    tof_us, tof_val = parabolic_peak(env, lags, TOF_MIN, TOF_MAX)
    tof_fwhm = fwhm_at(env, lags, tof_us, tof_val) if tof_us is not None else None

    # TOF/lag-0 ratio
    ratio_db = 20 * np.log10(tof_val / env[ctr] + 1e-12) if tof_val is not None else None

    print(f"  {label:<26} TOF={tof_us or 'N/F':>8.1f} μs  "
          f"TOF-FWHM={tof_fwhm or 'N/F':>7.0f} μs  "
          f"Lag0-FWHM={lag0_fwhm:>7.0f} μs  "
          f"Ratio={ratio_db or 'N/F':>6.1f} dB  "
          f"1/BW={1/bw_hz*1e6:>6.0f} μs")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    ax = axes[0]
    om = (lags >= -LAG_PLOT) & (lags <= LAG_PLOT)
    ax.plot(lags[om]/1000, acorr[om], 'b-', lw=0.4, alpha=0.5, label='Raw')
    ax.plot(lags[om]/1000, env[om],   'r-', lw=1.2, label='Envelope')
    ax.set_xlabel('Lag (ms)'); ax.set_ylabel('Autocorr')
    ax.set_title(f'{label}  |  Overview ±{LAG_PLOT/1000:.0f} ms')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    zm = (lags >= TOF_MIN-500) & (lags <= TOF_MAX+500)
    ax.plot(lags[zm], acorr[zm], 'b-', lw=0.4, alpha=0.5)
    ax.plot(lags[zm], env[zm],   'r-', lw=1.5)
    if tof_us is not None:
        ax.axvline(tof_us, color='g', ls='--', lw=1.5, label=f'TOF={tof_us:.1f} μs')
    ax.axvline(TOF_MIN, color='gray', ls=':', lw=1)
    ax.axvline(TOF_MAX, color='gray', ls=':', lw=1)
    ax.set_xlabel('Lag (μs)'); ax.set_ylabel('Envelope')
    info = (f"BW={bw_hz/1000:.1f} kHz  dur={dur_ms} ms  {amp_shape}\n"
            f"TOF={tof_us:.1f} μs  TOF-FWHM={tof_fwhm:.0f} μs  "
            f"Lag0-FWHM={lag0_fwhm:.0f} μs  1/BW={1/bw_hz*1e6:.0f} μs")
    ax.set_title(f'TOF region  |  {info}')
    if tof_us is not None:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    safe = label.replace('.', 'p').replace(' ', '_')
    out = os.path.join(OUT_DIR, f'chirp_reanalyzed_{safe}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()

    return dict(label=label, f_start=f_start, f_stop=f_stop,
                bw_hz=bw_hz, dur_ms=dur_ms, amp_shape=amp_shape,
                theo_fwhm_us=1/bw_hz*1e6,
                tof_us=tof_us, tof_fwhm_us=tof_fwhm,
                lag0_fwhm_us=lag0_fwhm, ratio_db=ratio_db)


def main():
    with open(os.path.join(RUN_DIR, 'manifest.json')) as f:
        manifest = json.load(f)

    print(f"Reanalyzing {len(manifest['measurements'])} chirp configs")
    print(f"Filter: HP {HP_CUTOFF/1000:.0f} kHz order {HP_ORDER}")
    print(f"TOF search: {TOF_MIN}–{TOF_MAX} μs\n")

    results = []
    for label, m in manifest['measurements'].items():
        wav_path = os.path.join(RUN_DIR, m['wav'])
        r = analyze_wav(wav_path, label,
                        m['f_start_hz'], m['f_stop_hz'],
                        m['duration_ms'], m['bandwidth_hz'], m['amp_shape'])
        results.append(r)

    print(f"\n{'='*120}")
    print("REANALYSIS SUMMARY  (full-signal autocorrelation, HP30kHz, TOF window 5000-6500 μs)")
    print(f"{'='*120}")
    print(f"{'Label':<28} {'BW':<7} {'ms':<5} {'Shape':<14} "
          f"{'Theo μs':<9} {'TOF μs':<9} {'TOF FWHM':<10} {'Lag0 FWHM':<11} {'Ratio dB'}")
    print('-'*120)
    for r in results:
        print(f"{r['label']:<28} {r['bw_hz']/1000:<7.2f} {r['dur_ms']:<5} {r['amp_shape']:<14} "
              f"{r['theo_fwhm_us']:<9.0f} "
              f"{r['tof_us'] or 'N/F':<9} "
              f"{r['tof_fwhm_us'] or 'N/F':<10} "
              f"{r['lag0_fwhm_us']:<11.0f} "
              f"{r['ratio_db'] or 'N/F'}")

    np.savez(os.path.join(OUT_DIR, 'chirp_reanalysis_results.npz'),
             labels=[r['label'] for r in results],
             bw_hz=[r['bw_hz'] for r in results],
             dur_ms=[r['dur_ms'] for r in results],
             theo_fwhm_us=[r['theo_fwhm_us'] for r in results],
             tof_us=[r['tof_us'] or 0 for r in results],
             tof_fwhm_us=[r['tof_fwhm_us'] or 0 for r in results],
             lag0_fwhm_us=[r['lag0_fwhm_us'] for r in results],
             ratio_db=[r['ratio_db'] or 0 for r in results])

    print(f"\nPlots saved to {OUT_DIR}/chirp_reanalyzed_*.png")
    print("Done!")


if __name__ == '__main__':
    main()
