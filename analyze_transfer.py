#!/usr/bin/env python3
"""
Analyze ultrasonic recordings to extract transfer function.

Usage:
    python analyze_transfer.py <input_directory> [--output-dir OUTPUT_DIR]

Input: Directory containing WAV files named by frequency (e.g., 40000.wav)
Output: Transfer function plot and CSV with frequency/amplitude data
"""

import argparse
import sys
from pathlib import Path
import numpy as np

from utils import (
    load_wav,
    compute_fft,
    find_fundamental,
    parse_frequency_from_filename,
    plot_transfer_function
)


def analyze_single_file(filepath, expected_freq=None):
    """
    Analyze a single WAV file and extract the fundamental amplitude.

    Returns:
        measured_freq: detected fundamental frequency
        amplitude_linear: linear amplitude
        amplitude_db: amplitude in dB (relative, will be normalized later)
    """
    sample_rate, signal = load_wav(filepath)

    # Compute FFT
    freqs, magnitude, magnitude_db = compute_fft(signal, sample_rate)

    # Find fundamental
    fund_freq, fund_mag, _ = find_fundamental(freqs, magnitude, expected_freq)

    # Convert to dB (not normalized yet)
    fund_db = 20 * np.log10(fund_mag + 1e-12)

    return fund_freq, fund_mag, fund_db, sample_rate


def process_directory(input_dir, verbose=True):
    """
    Process all WAV files in a directory.

    Returns:
        results: list of (expected_freq, measured_freq, amplitude_db) tuples
    """
    input_path = Path(input_dir)

    # Find all WAV files
    wav_files = sorted(input_path.glob('*.wav'))

    if not wav_files:
        print(f"No WAV files found in {input_dir}")
        return []

    results = []
    reference_amplitude = None

    for filepath in wav_files:
        expected_freq = parse_frequency_from_filename(filepath)

        if expected_freq is None:
            print(f"  Skipping {filepath.name}: couldn't parse frequency from filename")
            continue

        if verbose:
            print(f"  Processing {filepath.name} (expected: {expected_freq/1000:.1f} kHz)...")

        try:
            measured_freq, amp_linear, amp_db, sr = analyze_single_file(filepath, expected_freq)

            # Use first file as reference for relative dB
            if reference_amplitude is None:
                reference_amplitude = amp_linear

            relative_db = 20 * np.log10(amp_linear / reference_amplitude + 1e-12)

            results.append({
                'filename': filepath.name,
                'expected_freq_hz': expected_freq,
                'measured_freq_hz': measured_freq,
                'amplitude_linear': amp_linear,
                'amplitude_db': relative_db,
                'sample_rate': sr
            })

            if verbose:
                freq_error = abs(measured_freq - expected_freq) / expected_freq * 100
                print(f"    Measured: {measured_freq/1000:.2f} kHz (error: {freq_error:.1f}%), "
                      f"Amplitude: {relative_db:.1f} dB")

        except Exception as e:
            print(f"  Error processing {filepath.name}: {e}")

    return results


def save_results_csv(results, output_path):
    """Save results to CSV file."""
    with open(output_path, 'w') as f:
        f.write('filename,expected_freq_hz,measured_freq_hz,amplitude_linear,amplitude_db\n')
        for r in results:
            f.write(f"{r['filename']},{r['expected_freq_hz']:.1f},{r['measured_freq_hz']:.2f},"
                    f"{r['amplitude_linear']:.6e},{r['amplitude_db']:.2f}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze ultrasonic recordings to extract transfer function'
    )
    parser.add_argument('input_dir', help='Directory containing WAV files named by frequency')
    parser.add_argument('--output-dir', '-o', help='Output directory (default: same as input)')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting (just generate CSV)')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing transfer function from: {input_dir}")
    print()

    results = process_directory(input_dir)

    if not results:
        print("No files processed successfully")
        sys.exit(1)

    # Sort by frequency
    results.sort(key=lambda x: x['expected_freq_hz'])

    # Save CSV
    csv_path = output_dir / 'transfer_function.csv'
    save_results_csv(results, csv_path)
    print(f"\nSaved data to: {csv_path}")

    # Plot
    if not args.no_plot:
        frequencies = [r['expected_freq_hz'] for r in results]
        amplitudes_db = [r['amplitude_db'] for r in results]

        plot_path = output_dir / 'transfer_function.png'
        plot_transfer_function(
            frequencies,
            amplitudes_db,
            title='End-to-End Transfer Function (Emitter + Mic)',
            save_path=plot_path
        )
        print(f"Saved plot to: {plot_path}")

    # Print summary
    print("\n--- Summary ---")
    print(f"{'Frequency (kHz)':<18} {'Amplitude (dB)':<15}")
    print("-" * 35)
    for r in results:
        print(f"{r['expected_freq_hz']/1000:<18.1f} {r['amplitude_db']:<15.1f}")


if __name__ == '__main__':
    main()
