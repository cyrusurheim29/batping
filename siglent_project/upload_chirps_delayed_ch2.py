#!/usr/bin/env python3
"""
Upload chirp to BOTH SDG channels:
    CH1: standard a0.3_sm_4ms chirp at t=0 of each arb loop
    CH2: SAME chirp, delayed by --delay-ms within the arb loop

Both channels at same arb loop period (24 ms = 41.67 Hz for dead-ms=20).

Usage:
    python3 upload_chirps_delayed_ch2.py --ampl-c1 10 --ampl-c2 3.3 --delay-ms 2 --dead-ms 20
"""
import numpy as np
import time, argparse, os, sys
import scipy.signal.windows as _win
from scipy.signal import savgol_filter as _sg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from picoscope_batping_record import SDG

SDG_SR  = 400_000
SDG_MAX = 16_384
BURST_MS = 4.0
F_LO, F_HI = 37_000.0, 43_000.0
ALPHA   = 0.3

# Same H(f) file as upload_waveform_only.py uses
PIEZO_CHAR_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    'ecm999_stepped_sine_bswv_20260716_013405.npz')


def _make_chirp(dead_ms, delay_ms, ampl_vpp):
    """Generate buffer with a chirp placed at `delay_ms` offset."""
    buf_n = min(int((BURST_MS + dead_ms) / 1000 * SDG_SR), SDG_MAX)
    rep_hz = SDG_SR / buf_n
    burst_n = int(BURST_MS / 1000 * SDG_SR)
    delay_n = int(delay_ms / 1000 * SDG_SR)
    if delay_n + burst_n > buf_n:
        raise RuntimeError(f"delay={delay_ms} + chirp doesn't fit in {buf_n/SDG_SR*1000:.1f} ms period")

    t_b = np.arange(burst_n) / SDG_SR
    phase  = 2*np.pi*(F_LO*t_b + (F_HI-F_LO)/(2*BURST_MS/1000)*t_b**2)
    f_inst = F_LO + (F_HI-F_LO)/(BURST_MS/1000)*t_b

    # Load measured chain H(f) for alpha compensation (same as upload_waveform_only.py)
    _d = np.load(PIEZO_CHAR_FILE)
    _freqs = _d['freqs'].astype(float)
    _mag = 10**(_sg(_d['H_mag'], window_length=71, polyorder=3) / 20.0)
    _mag_f = np.clip(np.interp(f_inst, _freqs, _mag), 0.05, None)
    amp = _mag_f ** (-ALPHA)
    amp = amp / np.max(amp)

    burst = amp * np.sin(phase) * _win.tukey(burst_n, alpha=0.05)

    # Place in buffer at delay offset
    buf = np.zeros(buf_n)
    buf[delay_n:delay_n+burst_n] = burst
    # Normalise and convert to int16
    mx = np.max(np.abs(buf))
    if mx > 0: buf /= mx
    buf_i16 = (np.clip(buf, -1, 1) * 32767).astype('<i2')
    return buf_i16, rep_hz, buf_n, delay_n, burst_n


def _upload_to_channel(sdg, channel, buf_i16, rep_hz, ampl_vpp):
    name = f'BATPING_C{channel}'
    prefix = (f"C{channel}:WVDT WVNM,{name},FREQ,{rep_hz:.6f},"
              f"AMPL,{ampl_vpp},OFST,0,PHASE,0,WAVEDATA,").encode()
    sdg.write_raw(prefix + buf_i16.tobytes() + b'\n')
    time.sleep(0.8)
    sdg.write(f"C{channel}:ARWV NAME,{name}")
    time.sleep(0.3)
    sdg.write(f"C{channel}:BSWV WVTP,ARB")
    time.sleep(0.2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split('\n')[0])
    ap.add_argument('--ampl-c1', type=float, default=10.0, help='SDG C1 amplitude (Vpp) — the acoustic drive')
    ap.add_argument('--ampl-c2', type=float, default=3.3,  help='SDG C2 amplitude (Vpp) — the delayed-chirp reference signal')
    ap.add_argument('--delay-ms', type=float, default=2.0, help='time delay of CH2 chirp within each arb loop (ms)')
    ap.add_argument('--dead-ms',  type=float, default=20.0, help='dead time (ms) → arb loop period = burst + dead')
    args = ap.parse_args()

    print(f"Connecting to SDG...")
    sdg = SDG()
    try:
        print(f"  {sdg.query('*IDN?')[:60]}")

        # Build CH1 buffer (chirp at t=0)
        buf_c1, rep_hz, buf_n, _, burst_n = _make_chirp(args.dead_ms, 0.0, args.ampl_c1)
        # Build CH2 buffer (chirp at t=delay_ms)
        buf_c2, _, _, delay_n, _ = _make_chirp(args.dead_ms, args.delay_ms, args.ampl_c2)

        print(f"\n  Arb loop period: {1000/rep_hz:.3f} ms   rep rate: {rep_hz:.3f} Hz")
        print(f"  CH1 chirp fires at t = 0.00 ms within loop")
        print(f"  CH2 chirp fires at t = {args.delay_ms:.2f} ms within loop")
        print(f"  Separation: {args.delay_ms:.2f} ms (electrical delay)")

        print(f"\n  Uploading CH1 ({args.ampl_c1:.2f} Vpp)...")
        _upload_to_channel(sdg, 1, buf_c1, rep_hz, args.ampl_c1)
        sdg.output_on_channel(1) if hasattr(sdg, 'output_on_channel') else sdg.write('C1:OUTP ON')
        print(f"  Uploading CH2 ({args.ampl_c2:.2f} Vpp)...")
        _upload_to_channel(sdg, 2, buf_c2, rep_hz, args.ampl_c2)
        sdg.write('C2:OUTP ON')

        time.sleep(0.5)
        print(f"\n  Both channels loaded + ON.")
        print(f"  Verify with scope: Ch1 shows chirp starting at t=0,")
        print(f"  Ch2 shows chirp starting at t={args.delay_ms} ms.")
    finally:
        sdg.close()


if __name__ == '__main__':
    main()
