#!/usr/bin/env python3
"""Careful (hysteresis-resistant) chain transfer function H(f) for the T-connector setup.

Wraps ecm999_stepped_sine.py with:
  - 32–48 kHz sweep, 100 Hz step  → 161 points
  - **Randomized order** over TWO full passes → 322 measurements, averaged.
    Randomization decorrelates piezo ringdown from freq-ordered bias.
  - Longer burst (30 ms)  → narrower FFT bin (33 Hz), tighter rejection of leftover
    energy from the previously-driven tone.
  - Longer dead-time (50 ms)  → ≥50× τ_ringdown at Q≈50 — piezo fully quiet
    before the next burst.
  - Longer SDG-command settle (100 ms)  → SDG reaches steady-state before capture.
  - Discard the first 50% of each burst (BURST_SETTLE_FRAC) — avoids capturing
    ring-up + any residual from previous freq.

Wall-clock ≈ 322 × (100 ms settle + 110 ms capture + 100 ms SDG query) / 60 ≈ 2 min
per pass, ~5 min total.  More than fast enough and much cleaner than a single pass.

Pico ranges:
  Ch A ±5V FS (preamp G=100 + power amp ×10 → strong signal)
  Ch B ±10V FS (T-connector feeds raw SDG drive to Ch B)

Usage:
    DYLD_LIBRARY_PATH=/Library/Frameworks/PicoSDK.framework/Libraries/libps5000a \\
    python3 characterize_chain_tconnector_careful.py
"""
import os, glob, random, argparse
import numpy as np
from datetime import datetime
import ecm999_stepped_sine as _css
from picosdk.ps5000a import ps5000a as _ps

_ap = argparse.ArgumentParser(description=__doc__.strip().split('\n')[0])
_ap.add_argument('--n-passes',  type=int,   default=4,
                 help='Number of randomized passes to average (default 4)')
_ap.add_argument('--freq-step', type=int,   default=50,
                 help='Frequency step (Hz).  50 → 321 freqs; 100 → 161 freqs')
_ap.add_argument('--f-lo',      type=float, default=32_000.0, help='Sweep low (Hz)')
_ap.add_argument('--f-hi',      type=float, default=48_000.0, help='Sweep high (Hz)')
_ap.add_argument('--ampl',      type=float, default=0.3, help='SDG burst amplitude (Vpp)')
_ap.add_argument('--settle-frac', type=float, default=0.5,
                 help='Fraction of burst to drop (ring-up); default 0.5')
_args, _ = _ap.parse_known_args()

# ── Frequency grid (from CLI; defaults 32-48 kHz @ 50 Hz = 321 freqs) ────
_css.FREQ_START = int(_args.f_lo)
_css.FREQ_END   = int(_args.f_hi)
_css.FREQ_STEP  = int(_args.freq_step)

# ── Hysteresis-mitigation knobs ──────────────────────────────────────────
_css.BURST_MS           = 30.0    # longer burst → finer FFT bin (33 Hz)
_css.DEAD_MS            = 50.0    # ≥50× piezo ringdown time constant
_css.SETTLE_MS          = 100     # SDG settling after frequency command
_css.BURST_SETTLE_FRAC  = _args.settle_frac
# capture window = DEAD_MS + 2 * BURST_MS = 110 ms — one full burst guaranteed
_css.CAPTURE_MS         = _css.DEAD_MS + 2 * _css.BURST_MS

# ── PicoScope ranges for the current chain ──────────────────────────────
# Ch B: KEEP at ±1V FS.  The characterization drives at only 0.3 Vpp (150 mV peak)
# which fits in ±1V comfortably, AND the inherited trigger threshold logic
# (TRIG_MV/1000 * maxADC) assumes ±1V FS — overriding to ±10V breaks the
# hardware trigger (threshold becomes 1 V but signal is 150 mV peak → never
# fires → sweep falls back to 500 ms auto-trigger, random phase → per-freq
# scatter).  Recording setup at 10 Vpp is a separate concern handled in
# picoscope_batping_stream.py's own Ch B range flag.
_css.CH_B_RANGE = _ps.PS5000A_RANGE["PS5000A_1V"]
_css.CH_A_RANGE = _ps.PS5000A_RANGE["PS5000A_5V"]    # preamp×100 + ampX10

# ── Conservative drive (power amp ×10 is in the chain) ──────────────────
_css.BURST_AMPL_VPP = _args.ampl

# ── Patch the sweep to use randomized two-pass order ─────────────────────
# ecm999_stepped_sine.main() iterates in linear order.  We override by
# monkey-patching np.arange (which the main() uses to build freq_list) so
# it returns our custom [randomized pass1] + [randomized pass2] list.
_orig_arange = np.arange
_SWEEP_LIST = None        # populated below

def _sweep_arange(start, stop=None, step=None, dtype=None):
    """Replacement np.arange — returns our precomputed sweep list only when
    main() builds its freq_list (triggered by start=_css.FREQ_START)."""
    # main() calls np.arange(FREQ_START, FREQ_END + FREQ_STEP, FREQ_STEP)
    if (start == _css.FREQ_START
            and stop == _css.FREQ_END + _css.FREQ_STEP
            and step == _css.FREQ_STEP):
        print(f"  ✓ sentinel matched — injecting {len(_SWEEP_LIST)}-point randomized sweep")
        return np.array(_SWEEP_LIST)
    return _orig_arange(start, stop, step, dtype=dtype) if dtype is not None \
           else _orig_arange(start, stop, step)


def _build_randomized_passes(n_passes):
    rng = random.Random(0xB47910)  # deterministic — swap for None for true random
    f_base = list(range(_css.FREQ_START, _css.FREQ_END + 1, _css.FREQ_STEP))
    seq = []
    for _ in range(n_passes):
        p = f_base.copy(); rng.shuffle(p); seq.extend(p)
    return seq, f_base


if __name__ == '__main__':
    stamp     = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_base  = os.path.join(_css.OUTPUT_DIR, f'chain_tconnector_careful_{stamp}')
    print(f"Chain characterization (careful, randomized two-pass)")
    print(f"  Sweep       : {_css.FREQ_START/1e3:.1f} → {_css.FREQ_END/1e3:.1f} kHz @ {_css.FREQ_STEP} Hz step")
    print(f"  Burst       : {_css.BURST_MS:.0f} ms   dead: {_css.DEAD_MS:.0f} ms   settle: {_css.SETTLE_MS} ms")
    print(f"  SDG drive   : {_css.BURST_AMPL_VPP} Vpp (×10 power amp → ~3 Vpp at piezo)")
    print(f"  Pico ranges : Ch A ±5V,  Ch B ±10V")
    print(f"  Output base : {out_base}")

    _SWEEP_LIST, f_base = _build_randomized_passes(_args.n_passes)
    print(f"\n  {len(f_base)} unique frequencies × {_args.n_passes} passes "
          f"→ {len(_SWEEP_LIST)} measurements")

    np.arange = _sweep_arange
    try:
        _css.main()
    finally:
        np.arange = _orig_arange

    # ── Collapse two passes: average amp_mV at each unique frequency ─────
    # The last-written NPZ is what main() just dumped; load it, average, resave.
    cands = sorted(glob.glob(os.path.join(_css.OUTPUT_DIR,
                              'ecm999_stepped_sine_burst8ms_*.npz')),
                    key=os.path.getmtime)
    if not cands:
        raise RuntimeError("main() did not produce an output NPZ")
    src = cands[-1]
    d_full = np.load(src)
    freqs_measured = d_full['freqs'].astype(int)
    amp_measured   = d_full['amp_mV']
    snr_measured   = d_full['snr_dB']

    # Average across the two passes
    freqs_unique = np.array(sorted(set(freqs_measured.tolist())), dtype=float)
    amp_avg  = np.zeros(len(freqs_unique))
    snr_avg  = np.zeros(len(freqs_unique))
    amp_std  = np.zeros(len(freqs_unique))   # std across passes per freq
    n_hits   = np.zeros(len(freqs_unique), dtype=int)
    for i, f in enumerate(freqs_unique):
        hits = np.where(freqs_measured == int(f))[0]
        amp_avg[i] = np.mean(amp_measured[hits])
        snr_avg[i] = np.mean(snr_measured[hits])
        amp_std[i] = np.std(amp_measured[hits]) if len(hits) > 1 else 0.0
        n_hits[i]  = len(hits)

    # Recompute H_mag + H_smooth from the averaged amplitudes
    from scipy.signal import savgol_filter
    amp_dB = 20 * np.log10(amp_avg + 1e-12)
    peak_raw = float(np.max(amp_dB[freqs_unique > 25000]))
    H_mag = amp_dB - peak_raw
    H_smooth = savgol_filter(H_mag, window_length=min(71, len(H_mag) | 1), polyorder=3)

    # Save under the careful name — upload_variable_chirp.py reads 'freqs' + 'H_mag'
    np.savez(out_base + '.npz',
             freqs=freqs_unique, amp_mV=amp_avg, H_mag=H_mag, H_smooth=H_smooth,
             snr_dB=snr_avg, amp_std_mV=amp_std, n_hits_per_freq=n_hits,
             SDG_AMPL_VPP=_css.BURST_AMPL_VPP, burst_ms=_css.BURST_MS,
             dead_ms=_css.DEAD_MS, randomized=True, n_passes=_args.n_passes)
    # Rename the auto-dumped PNG so it carries the "careful" name too
    png_src = src.replace('.npz', '.png')
    if os.path.exists(png_src):
        os.rename(png_src, out_base + '_single_pass.png')
    # Remove the original auto-dumped NPZ (we already have a cleaner averaged version)
    os.remove(src)

    # Reproducibility report — per-freq std / mean across the passes
    rel_std_pct = 100 * amp_std / (amp_avg + 1e-12)
    pct_med = float(np.median(rel_std_pct[n_hits > 1]))
    pct_90  = float(np.percentile(rel_std_pct[n_hits > 1], 90))
    multi_pct = 100 * np.mean(n_hits > 1)
    print(f"\n  reproducibility across {_args.n_passes} passes:")
    print(f"    freqs with ≥2 passes    : {multi_pct:5.1f}%  ({int(np.mean(n_hits)):.1f} passes/freq avg)")
    print(f"    within-freq σ/μ (median): {pct_med:5.2f}%")
    print(f"    within-freq σ/μ (p90)   : {pct_90:5.2f}%")
    print(f"\n  wrote {out_base}.npz")
    print(f"\n  Use with upload: --h-file '{out_base}.npz'")
