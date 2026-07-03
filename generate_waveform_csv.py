#!/usr/bin/env python3
"""
Generate CSV waveform files for WaveStation USB import

The WaveStation can read .csv files from a USB stick:
1. Run this script to generate the CSV file
2. Copy the .csv file to a USB stick
3. Plug USB into front of WaveStation
4. Save/Recall -> Recall -> USB Device -> select file
"""

import numpy as np
import os

def generate_gaussian_pulse_csv(filename, n_points=1024, carrier_cycles=40,
                                 sigma_frac=0.15, amp_volts=2.0):
    """
    Generate a Gaussian-envelope pulse and save as CSV for WaveStation.

    Args:
        filename: Output CSV filename
        n_points: Number of sample points (max 16384 for WaveStation 2000)
        carrier_cycles: Number of carrier oscillation cycles
        sigma_frac: Gaussian width as fraction of total length
        amp_volts: Peak amplitude in volts
    """
    t = np.linspace(-0.5, 0.5, n_points)

    # Gaussian envelope
    envelope = np.exp(-t**2 / (2 * sigma_frac**2))

    # Carrier wave
    carrier = np.sin(2 * np.pi * carrier_cycles * t)

    # Combined waveform, scaled to voltage
    waveform = envelope * carrier * amp_volts

    # Write CSV with Amplitude header
    with open(filename, 'w') as f:
        f.write("Amplitude\n")
        for v in waveform:
            f.write(f"{v:.6f}\n")

    print(f"Generated: {filename}")
    print(f"  Points: {n_points}")
    print(f"  Carrier cycles: {carrier_cycles}")
    print(f"  Peak amplitude: {amp_volts}V")
    return filename


def generate_sine_burst_csv(filename, n_points=1024, cycles=10, amp_volts=2.0):
    """
    Generate a simple sine burst (for testing).
    """
    t = np.linspace(0, 1, n_points)
    waveform = amp_volts * np.sin(2 * np.pi * cycles * t)

    with open(filename, 'w') as f:
        f.write("Amplitude\n")
        for v in waveform:
            f.write(f"{v:.6f}\n")

    print(f"Generated: {filename}")
    print(f"  Points: {n_points}")
    print(f"  Cycles: {cycles}")
    return filename


def generate_chirp_csv(filename, n_points=2048, f_start=5, f_end=50,
                       amp_volts=2.0, envelope="gaussian"):
    """
    Generate a chirp (frequency sweep) waveform.

    Args:
        f_start: Starting frequency (cycles per waveform)
        f_end: Ending frequency (cycles per waveform)
        envelope: "rectangular", "gaussian", or "hanning"
    """
    t = np.linspace(0, 1, n_points)

    # Chirp: instantaneous frequency varies linearly
    phase = 2 * np.pi * (f_start * t + (f_end - f_start) * t**2 / 2)
    waveform = np.sin(phase)

    # Apply envelope
    if envelope == "gaussian":
        env = np.exp(-((t - 0.5) ** 2) / (2 * 0.15**2))
        waveform *= env
    elif envelope == "hanning":
        env = 0.5 * (1 - np.cos(2 * np.pi * t))
        waveform *= env

    waveform *= amp_volts

    with open(filename, 'w') as f:
        f.write("Amplitude\n")
        for v in waveform:
            f.write(f"{v:.6f}\n")

    print(f"Generated: {filename}")
    print(f"  Points: {n_points}")
    print(f"  Freq sweep: {f_start} -> {f_end} cycles")
    print(f"  Envelope: {envelope}")
    return filename


if __name__ == "__main__":
    # Create output directory
    output_dir = "/Users/cyrusurheim_1/Downloads/Stubbs_Stuff/Claude_Dir/waveforms"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 50)
    print("WaveStation CSV Waveform Generator")
    print("=" * 50)

    # Generate test waveforms
    print("\n[1] Generating 40kHz Gaussian pulse for batping...")
    generate_gaussian_pulse_csv(
        f"{output_dir}/batping_40khz_pulse.csv",
        n_points=1024,
        carrier_cycles=40,
        sigma_frac=0.15,
        amp_volts=2.0
    )

    print("\n[2] Generating simple sine burst (test)...")
    generate_sine_burst_csv(
        f"{output_dir}/test_sine_10cycles.csv",
        n_points=512,
        cycles=10,
        amp_volts=2.0
    )

    print("\n[3] Generating chirp waveform...")
    generate_chirp_csv(
        f"{output_dir}/chirp_5_50.csv",
        n_points=2048,
        f_start=5,
        f_end=50,
        amp_volts=2.0,
        envelope="gaussian"
    )

    print("\n" + "=" * 50)
    print("CSV files created in:", output_dir)
    print("")
    print("To load onto WaveStation:")
    print("  1. Copy .csv files to USB stick")
    print("  2. Plug USB into front of WaveStation")
    print("  3. Press Save/Recall button")
    print("  4. Select 'Data' as FileType")
    print("  5. Navigate to USB Device (A:)")
    print("  6. Select file and press Recall")
    print("=" * 50)
