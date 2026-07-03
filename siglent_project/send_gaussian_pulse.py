#!/usr/bin/env python3
"""
Send a Gaussian-enveloped 40kHz pulse to the Siglent SDG1000X.
"""

import numpy as np
import sys
sys.path.append('/Users/cyrusurheim_1/Downloads/Stubbs_Stuff/Claude_Dir/agilent_project')

from siglent_sdg1000x import SiglentSDG1000X, generate_gaussian_pulse


def main():
    # Parameters
    center_freq_hz = 40000      # 40 kHz carrier
    sigma_cycles = 2            # 2-cycle Gaussian envelope (narrow pulse)
    rep_rate_hz = 50            # 50 Hz repetition rate (20 ms period)
    vpp = 10.0                  # 10 Vpp output amplitude

    # For Siglent DDS mode: 16384 points
    # At 50 Hz rep rate, period = 20 ms
    # Effective sample rate = 16384 points / 20 ms = 819.2 kHz
    n_points = 16384
    period_s = 1.0 / rep_rate_hz
    sample_rate_hz = n_points / period_s  # 819.2 kHz

    print(f"=== Gaussian Pulse Generator for Siglent SDG1000X ===")
    print(f"Carrier frequency: {center_freq_hz/1000:.0f} kHz")
    print(f"Envelope width: {sigma_cycles} cycles (σ)")
    print(f"Repetition rate: {rep_rate_hz} Hz ({period_s*1000:.0f} ms period)")
    print(f"Output amplitude: {vpp} Vpp")
    print(f"Waveform: {n_points} points at {sample_rate_hz/1000:.1f} kHz effective sample rate")

    # Generate the pulse - centered in the waveform
    pulse = generate_gaussian_pulse(
        n_points=n_points,
        center_freq_hz=center_freq_hz,
        sample_rate_hz=sample_rate_hz,
        sigma_cycles=sigma_cycles
    )

    # Calculate pulse duration
    sigma_t = sigma_cycles / center_freq_hz
    pulse_duration_ms = 6 * sigma_t * 1000  # 6-sigma contains 99.7% of energy
    print(f"Pulse duration (6σ): {pulse_duration_ms:.2f} ms")

    # Check how many carrier cycles fit
    n_cycles_in_pulse = pulse_duration_ms / 1000 * center_freq_hz
    print(f"Carrier cycles in pulse: ~{n_cycles_in_pulse:.0f}")

    # Connect and upload
    print(f"\nConnecting to Siglent...")

    try:
        with SiglentSDG1000X(channel=1) as awg:
            print(f"\nUploading waveform...")
            awg.output_arbitrary(
                data=pulse,
                freq_hz=rep_rate_hz,
                vpp=vpp,
                name="GAUSS40K",
                enable=True
            )
            print(f"\nOutput enabled!")
            print(f"Status: {awg.status()}")

            input("\nPress Enter to disable output and exit...")
            awg.set_output(False)
            print("Output disabled.")

    except Exception as e:
        print(f"\nError: {e}")
        print("\nTo find your Siglent, run:")
        print("  import pyvisa")
        print("  rm = pyvisa.ResourceManager()")
        print("  print(rm.list_resources())")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
