#!/usr/bin/env python3
"""
tmp36_calibration.py

Quick TMP36 verification on PicoScope Ch B.
Reads 10 blocks (~38 seconds total) and reports mean voltage and temperature.
No SDG needed — just the PicoScope with TMP36 on Ch B.

Usage:
    python3 tmp36_calibration.py
    python3 tmp36_calibration.py --n-blocks 5
"""
import sys, os, argparse, ctypes
import numpy as np

_pico_lib = '/Library/Frameworks/PicoSDK.framework/Libraries/libps5000a'
if os.path.isdir(_pico_lib):
    os.environ['DYLD_LIBRARY_PATH'] = _pico_lib + ':' + os.environ.get('DYLD_LIBRARY_PATH','')

from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import adc2mV, assert_pico_ok

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import picoscope_batping_record as pbr

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n-blocks', type=int, default=10,
                    help='Number of blocks to average (default 10, ~38 s)')
    args = ap.parse_args()

    print("TMP36 calibration check — PicoScope Ch B (DC, ±2V)")
    print("TMP36 conversion: T(°C) = (V_mV − 500) / 10")
    print(f"  Expected at 25°C: 750 mV")
    print(f"  Expected at 20°C: 700 mV")
    print(f"  Expected at 30°C: 800 mV\n")

    print("Opening PicoScope...")
    chandle, actual_fs = pbr.open_scope()
    period_n   = round(actual_fs / 52.631578947368425)
    n_samples  = pbr.BLOCK_PERIODS * period_n
    print(f"  Block size: {pbr.BLOCK_PERIODS} periods = {n_samples:,} samples = {n_samples/actual_fs:.2f} s\n")

    temps = []
    voltages = []

    print(f"Reading {args.n_blocks} blocks...\n")
    print(f"  {'Block':>6}  {'V_mean (mV)':>12}  {'T (°C)':>8}  {'V_std (mV)':>11}  {'V_min':>7}  {'V_max':>7}")
    print("  " + "-"*60)

    for i in range(args.n_blocks):
        _, temp_C = pbr.capture_block(chandle, n_samples)

        # Re-read Ch B raw to get voltage stats
        # (temp_C is already the mean — recompute from raw for stats)
        # We'll use the temp_C and back-calculate V
        v_mean_mv = temp_C * pbr.TMP36_SCALE + pbr.TMP36_V0_MV
        temps.append(temp_C)
        voltages.append(v_mean_mv)
        print(f"  {i+1:>6}  {v_mean_mv:>12.2f}  {temp_C:>8.3f}  (block mean)", flush=True)

    ps.ps5000aStop(chandle)
    ps.ps5000aCloseUnit(chandle)

    temps = np.array(temps)
    volts = np.array(voltages)

    print(f"\n{'='*50}")
    print(f"  Mean voltage : {volts.mean():.2f} mV")
    print(f"  Voltage std  : {volts.std():.3f} mV  (block-to-block)")
    print(f"  Mean temp    : {temps.mean():.3f} °C")
    print(f"  Temp std     : {temps.std():.4f} °C  (block-to-block)")
    print(f"  Temp range   : {temps.min():.3f} – {temps.max():.3f} °C")

    t = temps.mean()
    if 10 <= t <= 45:
        print(f"\n  ✓  LOOKS GOOD — {t:.2f}°C is a plausible lab temperature")
    elif t < 10:
        print(f"\n  ✗  TOO COLD ({t:.2f}°C) — check wiring, likely Ch B seeing wrong signal")
    elif t > 45:
        print(f"\n  ✗  TOO HOT ({t:.2f}°C) — check wiring or supply voltage")

    if volts.mean() < 50:
        print(f"  ✗  VOLTAGE NEAR ZERO — TMP36 likely not connected or wired wrong")


if __name__ == '__main__':
    main()
