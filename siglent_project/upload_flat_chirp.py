#!/usr/bin/env python3
"""Upload a FLAT (no H(f) compensation) LFM chirp to SDG CH1 only.

For the T-connector setup where SDG CH1 drives both the piezo array
(acoustic launch via Ch A mic) AND PicoScope Ch B (electrical reference).

Just a plain LFM chirp with a 5% Tukey window — no amplitude modulation.

Usage:
    python3 upload_flat_chirp.py --burst-ms 4 --dead-ms 30 --ampl 10
"""
import os, sys, time, argparse
import numpy as np
from scipy.signal import windows as _win

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from picoscope_batping_record import SDG

SDG_SR     = 400_000
SDG_MAX    = 16_384
F_LO, F_HI = 37_000.0, 43_000.0


def _build_chirp(burst_ms, dead_ms, ampl_vpp):
    buf_n = int((burst_ms + dead_ms) / 1000 * SDG_SR)
    if buf_n > SDG_MAX:
        max_total_ms = SDG_MAX / SDG_SR * 1000
        sys.exit(f"ERROR: burst+dead = {burst_ms+dead_ms:.1f} ms exceeds SDG arb buffer "
                 f"({max_total_ms:.2f} ms). Reduce burst-ms or dead-ms.")
    rep_hz  = SDG_SR / buf_n
    burst_n = int(burst_ms / 1000 * SDG_SR)
    t_b     = np.arange(burst_n) / SDG_SR

    # LFM chirp phase
    phase = 2*np.pi*(F_LO*t_b + (F_HI-F_LO)/(2*burst_ms/1000)*t_b**2)
    burst = np.sin(phase) * _win.tukey(burst_n, alpha=0.05)

    buf = np.zeros(buf_n, dtype=np.float64)
    buf[:burst_n] = burst
    mx = np.max(np.abs(buf))
    if mx > 0: buf /= mx
    buf_i16 = (np.clip(buf, -1, 1) * 32767).astype('<i2')
    return buf_i16, rep_hz, buf_n


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split('\n')[0])
    ap.add_argument('--burst-ms', type=float, default=4.0, help='Chirp duration (ms).')
    ap.add_argument('--dead-ms',  type=float, default=30.0, help='Dead-time between chirps (ms).')
    ap.add_argument('--ampl',     type=float, default=10.0, help='SDG CH1 amplitude (Vpp).')
    args = ap.parse_args()

    buf_i16, rep_hz, buf_n = _build_chirp(args.burst_ms, args.dead_ms, args.ampl)
    period_ms = 1000.0 / rep_hz
    print(f"  Flat chirp (no H(f) compensation)")
    print(f"  Chirp duration : {args.burst_ms:5.2f} ms  ({F_LO/1e3:.0f}-{F_HI/1e3:.0f} kHz)")
    print(f"  Dead time      : {args.dead_ms:5.2f} ms")
    print(f"  Arb period     : {period_ms:5.2f} ms   rep rate: {rep_hz:.3f} Hz   Nyquist: {rep_hz/2:.3f} Hz")
    print(f"  Arb buffer     : {buf_n} / {SDG_MAX} samples   ({100*buf_n/SDG_MAX:.1f}%)")

    name = f'BATPING_C1_flat_{int(round(args.burst_ms))}ms_dead{int(round(args.dead_ms))}ms'
    print(f"\nConnecting to SDG...")
    sdg = SDG()
    try:
        print(f"  {sdg.query('*IDN?')[:60]}")
        prefix = (f"C1:WVDT WVNM,{name},FREQ,{rep_hz:.6f},"
                  f"AMPL,{args.ampl},OFST,0,PHASE,0,WAVEDATA,").encode()
        sdg.write_raw(prefix + buf_i16.tobytes() + b'\n')
        time.sleep(0.8)
        sdg.write(f"C1:ARWV NAME,{name}")
        time.sleep(0.3)
        sdg.write(f"C1:BSWV WVTP,ARB")
        time.sleep(0.2)
        sdg.write('C1:OUTP ON')
        sdg.write('C2:OUTP OFF')
        time.sleep(0.3)
        print(f"\n  CH1 loaded + ON (amplitude {args.ampl:.2f} Vpp).  CH2 OFF.")
    finally:
        sdg.close()


if __name__ == '__main__':
    main()
