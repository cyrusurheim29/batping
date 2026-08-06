#!/usr/bin/env python3
"""
Analyze harmonic content and THD in ultrasonic recordings.

Usage:
    python analyze_harmonics.py <input_directory> [--output-dir OUTPUT_DIR]

Input: Directory containing WAV files named by frequency (e.g., 40000.wav)
Output: Harmonic spectrum plots, THD summary table, and CSV data
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    load_wav,
    compute_fft,
    find_fundamental,
    find_harmonics,
    calculate_thd,
    parse_frequency_from_filename,
    plot_spectrum
)


def analyze_harmonics_single(filepath, expected_freq=None, num_harmonics=5):
    """
    Analyze harmonic content of a single WAV file.

    Returns:
        dict with fundamental, harmonics, THD, and spectrum data
    """
    sample_rate, signal = load_wav(filepath)

    # Compute FFT
    freqs, magnitude, magnitude_db = compute_fft(signal, sample_rate)

    # Find fundamental
    fund_freq, fund_mag, fund_idx = find_fundamental(freqs, magnitude, expected_freq)

    # Find harmonics
    harmonic_freqs, harmonic_mags = find_harmonics(
        freqs, magnitude, fund_freq, num_harmonics
    )

    # Calculate THD
    thd = calculate_thd(harmonic_mags)

    # Calculate individual harmonic levels relative to fundamental
    harmonic_levels_db = []
    for mag in harmonic_mags:
        if fund_mag > 1e-12:
            level = 20 * np.log10(mag / fund_mag + 1e-12)
        else:
            level = -100
        harmonic_levels_db.append(level)

    return {
        'sample_rate': sample_rate,
        'fundamental_freq': fund_freq,
        'fundamental_mag': fund_mag,
        'harmonic_freqs': harmonic_freqs,
        'harmonic_mags': harmonic_mags,
        'harmonic_levels_db': harmonic_levels_db,
        'thd_percent': thd,
        'freqs': freqs,
        'magnitude': magnitude,
        'magnitude_db': magnitude_db
    }


def plot_harmonic_bar_chart(results, output_path):
    """
    Create bar chart showing harmonic levels for each test frequency.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    n_files = len(results)
    n_harmonics = max(len(r['harmonic_levels_db']) for r in results)

    x = np.arange(n_files)
    width = 0.15

    colors = plt.cm.viridis(np.linspace(0, 0.8, n_harmonics))

    for h_idx in range(n_harmonics):
        levels = []
        for r in results:
            if h_idx < len(r['harmonic_levels_db']):
                levels.append(r['harmonic_levels_db'][h_idx])
            else:
                levels.append(-100)

        label = 'Fundamental' if h_idx == 0 else f'H{h_idx + 1}'
        ax.bar(x + h_idx * width, levels, width, label=label, color=colors[h_idx])

    ax.set_xlabel('Test Frequency')
    ax.set_ylabel('Level (dB relative to fundamental)')
    ax.set_title('Harmonic Content by Frequency')

    freq_labels = [f"{r['expected_freq_hz']/1000:.0f}k" for r in results]
    ax.set_xticks(x + width * (n_harmonics - 1) / 2)
    ax.set_xticklabels(freq_labels)

    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(-60, 5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_thd_vs_frequency(results, output_path):
    """
    Plot THD percentage vs test frequency.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    freqs = [r['expected_freq_hz'] / 1000 for r in results]
    thds = [r['thd_percent'] for r in results]

    ax.plot(freqs, thds, 'b-o', linewidth=1.5, markersize=8)

    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('THD (%)')
    ax.set_title('Total Harmonic Distortion vs Frequency')
    ax.grid(True, alpha=0.3)

    # Add horizontal reference lines
    for level in [1, 5, 10]:
        ax.axhline(level, color='gray', linestyle='--', alpha=0.5)
        ax.text(freqs[-1] + 2, level, f'{level}%', va='center', fontsize=9, color='gray')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def process_directory(input_dir, output_dir, verbose=True, plot_spectra=True):
    """
    Process all WAV files and analyze harmonics.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    wav_files = sorted(input_path.glob('*.wav'))

    if not wav_files:
        print(f"No WAV files found in {input_dir}")
        return []

    results = []

    for filepath in wav_files:
        expected_freq = parse_frequency_from_filename(filepath)

        if expected_freq is None:
            print(f"  Skipping {filepath.name}: couldn't parse frequency")
            continue

        if verbose:
            print(f"  Processing {filepath.name}...")

        try:
            analysis = analyze_harmonics_single(filepath, expected_freq)
            analysis['filename'] = filepath.name
            analysis['expected_freq_hz'] = expected_freq

            results.append(analysis)

            if verbose:
                print(f"    Fundamental: {analysis['fundamental_freq']/1000:.2f} kHz")
                print(f"    THD: {analysis['thd_percent']:.2f}%")
                if len(analysis['harmonic_levels_db']) > 1:
                    h2_db = analysis['harmonic_levels_db'][1]
                    print(f"    2nd harmonic: {h2_db:.1f} dB")

            # Save individual spectrum plot
            if plot_spectra:
                spectrum_path = output_path / f"spectrum_{filepath.stem}.png"
                max_freq = min(analysis['fundamental_freq'] * 6, analysis['sample_rate'] / 2)
                plot_spectrum(
                    analysis['freqs'],
                    analysis['magnitude_db'],
                    title=f"Spectrum: {filepath.stem} (THD: {analysis['thd_percent']:.1f}%)",
                    fundamental_freq=analysis['fundamental_freq'],
                    harmonic_freqs=analysis['harmonic_freqs'],
                    max_freq=max_freq,
                    save_path=spectrum_path
                )

        except Exception as e:
            print(f"  Error processing {filepath.name}: {e}")

    return results


def save_results_csv(results, output_path):
    """Save harmonic analysis results to CSV."""
    with open(output_path, 'w') as f:
        # Header
        f.write('filename,expected_freq_hz,measured_freq_hz,thd_percent,'
                'h1_db,h2_db,h3_db,h4_db,h5_db\n')

        for r in results:
            levels = r['harmonic_levels_db']
            # Pad to 5 harmonics
            while len(levels) < 5:
                levels.append(-100)

            f.write(f"{r['filename']},{r['expected_freq_hz']:.1f},"
                    f"{r['fundamental_freq']:.2f},{r['thd_percent']:.2f},"
                    f"{levels[0]:.2f},{levels[1]:.2f},{levels[2]:.2f},"
                    f"{levels[3]:.2f},{levels[4]:.2f}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze harmonic content and THD in ultrasonic recordings'
    )
    parser.add_argument('input_dir', help='Directory containing WAV files')
    parser.add_argument('--output-dir', '-o', help='Output directory (default: same as input)')
    parser.add_argument('--no-spectra', action='store_true',
                        help='Skip individual spectrum plots')
    parser.add_argument('--num-harmonics', '-n', type=int, default=5,
                        help='Number of harmonics to analyze (default: 5)')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing harmonic content from: {input_dir}")
    print()

    results = process_directory(
        input_dir, output_dir,
        plot_spectra=not args.no_spectra
    )

    if not results:
        print("No files processed successfully")
        sys.exit(1)

    # Sort by frequency
    results.sort(key=lambda x: x['expected_freq_hz'])

    # Save CSV
    csv_path = output_dir / 'harmonic_analysis.csv'
    save_results_csv(results, csv_path)
    print(f"\nSaved data to: {csv_path}")

    # Summary plots
    thd_plot_path = output_dir / 'thd_vs_frequency.png'
    plot_thd_vs_frequency(results, thd_plot_path)
    print(f"Saved THD plot to: {thd_plot_path}")

    bar_plot_path = output_dir / 'harmonic_levels.png'
    plot_harmonic_bar_chart(results, bar_plot_path)
    print(f"Saved harmonic bar chart to: {bar_plot_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("HARMONIC ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"{'Freq (kHz)':<12} {'THD %':<10} {'H2 (dB)':<10} {'H3 (dB)':<10} {'H4 (dB)':<10}")
    print("-" * 70)

    for r in results:
        levels = r['harmonic_levels_db']
        h2 = levels[1] if len(levels) > 1 else -100
        h3 = levels[2] if len(levels) > 2 else -100
        h4 = levels[3] if len(levels) > 3 else -100

        print(f"{r['expected_freq_hz']/1000:<12.1f} {r['thd_percent']:<10.2f} "
              f"{h2:<10.1f} {h3:<10.1f} {h4:<10.1f}")

    # Flag high THD frequencies
    high_thd = [r for r in results if r['thd_percent'] > 5]
    if high_thd:
        print("\n*** Frequencies with THD > 5%: ***")
        for r in high_thd:
            print(f"  {r['expected_freq_hz']/1000:.1f} kHz: {r['thd_percent']:.1f}%")


if __name__ == '__main__':
    main()
