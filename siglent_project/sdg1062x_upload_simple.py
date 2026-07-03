#!/usr/bin/env python3
"""
SDG1062X Arbitrary Waveform Upload - Write-only (no queries)
Skips all queries since Siglent USB reads are problematic.

Usage: sudo python sdg1062x_upload_simple.py
"""

import pyvisa
import numpy as np
import time
import binascii


def wave_to_binary(wave_data):
    """Convert float waveform [-1, 1] to binary format for SDG."""
    wave_clipped = np.clip(wave_data, -1.0, 1.0)
    wave_int16 = (wave_clipped * 32767).astype(np.int16)

    hex_data = ''
    for val in wave_int16:
        if val < 0:
            val = val + 65536
        hex_data += format(val, '04x')

    return binascii.a2b_hex(hex_data)


def main():
    print("=" * 60)
    print("SDG1062X Upload (Write-Only Mode)")
    print("=" * 60)

    rm = pyvisa.ResourceManager()
    resources = rm.list_resources()
    print(f"Found: {resources}")

    sdg_resource = None
    for r in resources:
        if 'USB' in r and ('F4EC' in r or '62700' in r or 'SDG' in r.upper()):
            sdg_resource = r
            break

    if not sdg_resource:
        print("SDG1062X not found!")
        return 1

    print(f"\nConnecting to: {sdg_resource}")
    dev = rm.open_resource(sdg_resource)
    dev.timeout = 10000
    dev.write_termination = '\n'

    # Create 256-point sine wave
    print("\n--- Creating Waveform ---")
    n_points = 256
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    wave = np.sin(t)
    bin_data = wave_to_binary(wave)

    name = "sine256"
    print(f"Waveform: {name}, {n_points} points, {len(bin_data)} bytes")

    response = input("\nUpload and enable output? (y/N): ").strip().lower()
    if response != 'y':
        dev.close()
        return 0

    print("\n1. Clearing status...")
    dev.write("*CLS")
    time.sleep(0.2)

    print("2. Uploading waveform data...")
    cmd_header = f"C1:WVDT WVNM,{name},FREQ,1000.0,AMPL,2.0,OFST,0.0,PHASE,0.0,WAVEDATA,"
    full_cmd = cmd_header.encode('utf-8') + bin_data
    dev.write_raw(full_cmd)
    time.sleep(1.5)

    print("3. Setting ARB mode...")
    dev.write("C1:BSWV WVTP,ARB")
    time.sleep(0.3)

    print("4. Selecting waveform...")
    dev.write(f"C1:ARWV NAME,{name}")
    time.sleep(0.5)

    print("5. Enabling CH1 output...")
    dev.write("C1:OUTP ON")
    time.sleep(0.2)

    print("\n>>> Commands sent! <<<")
    print("\nCheck the SDG1062X front panel:")
    print("  - Should show ARB waveform type")
    print("  - CH1 output should be ON")
    print("  - Press [Arb] button to see stored waveforms")

    dev.close()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
