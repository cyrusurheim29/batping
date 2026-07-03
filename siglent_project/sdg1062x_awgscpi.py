#!/usr/bin/env python3
"""
SDG1062X arbitrary waveform upload using awg-scpi library.
This library knows the correct Siglent SCPI dialect.
"""

import numpy as np
import sys

# awg-scpi handles the SCPI commands properly
from awg_scpi import Siglent


def main():
    print("=" * 60)
    print("SDG1062X: Using awg-scpi Library")
    print("=" * 60)

    # Find the SDG resource string
    import pyvisa
    rm = pyvisa.ResourceManager('@py')
    resources = rm.list_resources()
    print(f"Found resources: {resources}")

    sdg_resource = None
    for r in resources:
        if 'USB' in r:
            sdg_resource = r
            break

    if not sdg_resource:
        print("No USB instrument found!")
        return 1

    print(f"\nConnecting to: {sdg_resource}")

    # Create Siglent AWG instance
    # SDG1062X has 2 channels
    awg = Siglent(sdg_resource, maxChannel=2, wait=0.1)
    awg.open()

    # Get device ID
    idn = awg.idn()
    print(f"Device: {idn}")

    # Test 1: Simple sine wave using built-in
    print("\n--- Test 1: Built-in sine wave ---")
    response = input("Set CH1 to 40kHz sine, 2Vpp? (y/N): ").strip().lower()

    if response == 'y':
        awg.setWaveType('SINE', channel=1)
        awg.setFrequency(40000, channel=1)
        awg.setAmplitude(2.0, channel=1)
        awg.setOffset(0, channel=1)
        awg.outputOn(channel=1)
        print("Done! Check output.")

    # Test 2: Simple arbitrary waveform
    print("\n--- Test 2: Arbitrary waveform (square pulse) ---")
    response = input("Upload simple square pulse? (y/N): ").strip().lower()

    if response == 'y':
        # Simple square pulse: low for half, high for half
        n_points = 100
        data = np.zeros(n_points, dtype=np.int16)
        data[:50] = -32768  # Low
        data[50:] = 32767   # High

        print(f"Uploading {n_points} points...")
        awg.setArbWaveData(
            data=data,
            name='PULSE',
            freq=1000,      # 1kHz repetition
            amp=2.0,        # 2Vpp
            offset=0,
            phase=0,
            channel=1,
            wait=2
        )

        # Select it and set to ARB mode
        awg.setArbWaveByName('PULSE', channel=1)
        awg.setWaveType('ARB', channel=1)
        awg.outputOn(channel=1)
        print("Done! Check output for square pulse at 1kHz.")

    # Test 3: Gaussian burst (what we actually want)
    print("\n--- Test 3: Gaussian burst for ultrasonic ---")
    response = input("Upload Gaussian burst? (y/N): ").strip().lower()

    if response == 'y':
        # Create Gaussian-windowed 40kHz burst
        n_points = 4000
        t = np.linspace(0, 1, n_points, endpoint=False)

        # 5-cycle 40kHz carrier with Gaussian envelope
        n_cycles = 5
        carrier = np.sin(2 * np.pi * n_cycles * t)
        sigma = 0.15
        envelope = np.exp(-0.5 * ((t - 0.5) / sigma)**2)
        waveform = carrier * envelope

        # Normalize and convert to int16
        waveform = waveform / np.max(np.abs(waveform))
        data = (waveform * 32767).astype(np.int16)

        print(f"Uploading {n_points} points...")
        print(f"Data range: {data.min()} to {data.max()}")

        # For 40kHz output with 4000 points per cycle:
        # freq = 40000 / n_cycles = 8000 Hz (waveform repeat rate)
        # This gives us 40kHz carrier within each repetition
        rep_freq = 40000 / n_cycles  # 8000 Hz

        awg.setArbWaveData(
            data=data,
            name='BURST',
            freq=rep_freq,
            amp=2.0,
            offset=0,
            phase=0,
            channel=1,
            wait=3
        )

        awg.setArbWaveByName('BURST', channel=1)
        awg.setWaveType('ARB', channel=1)
        awg.outputOn(channel=1)
        print(f"Done! 40kHz burst at {rep_freq}Hz rep rate.")

    # Query what's loaded
    print("\n--- Current state ---")
    try:
        names = awg.queryArbWaveNamesUser(channel=1)
        print(f"User waveforms: {names}")
    except Exception as e:
        print(f"Could not query waveforms: {e}")

    awg.close()
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
