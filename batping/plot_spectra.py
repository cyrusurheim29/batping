#!/usr/bin/env python3
"""Create Audacity-style filled spectrum plots from txt files."""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt


def load_spectrum(filename):
    """Load frequency/level data from Audacity export."""
    freq = []
    level = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('Frequency'):
                continue  # Skip header
            parts = line.strip().split('\t')
            if len(parts) == 2:
                freq.append(float(parts[0]))
                level.append(float(parts[1]))
    return np.array(freq), np.array(level)


def plot_spectrum_audacity_style(filename, title, output_path):
    """Create an Audacity-style filled spectrum plot."""
    freq, level = load_spectrum(filename)

    fig, ax = plt.subplots(figsize=(14, 6))

    # Filled area plot like Audacity
    ax.fill_between(freq / 1000, level, -100, color='#9370DB', alpha=0.9)
    ax.plot(freq / 1000, level, color='#6A0DAD', linewidth=0.5)

    # Styling
    ax.set_xlabel('Frequency (kHz)', fontsize=12)
    ax.set_ylabel('Level (dB)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Set axis limits
    ax.set_xlim([0, 192])
    ax.set_ylim([-100, 0])

    # Grid
    ax.grid(True, alpha=0.3, which='both')
    ax.set_axisbelow(True)

    # Mark key frequencies
    ax.axvline(x=40, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='40 kHz')
    ax.axvline(x=33, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, label='33 kHz')
    ax.axvline(x=80, color='green', linestyle='--', alpha=0.5, linewidth=1, label='80 kHz (2nd harmonic)')

    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='white')
    print(f"Saved: {output_path}")

    # Print peak info
    peak_idx = np.argmax(level)
    print(f"  Peak: {freq[peak_idx]/1000:.2f} kHz @ {level[peak_idx]:.1f} dB")

    # Find 40kHz and 33kHz levels
    idx_40k = np.argmin(np.abs(freq - 40000))
    idx_33k = np.argmin(np.abs(freq - 33000))
    print(f"  40 kHz: {level[idx_40k]:.1f} dB")
    print(f"  33 kHz: {level[idx_33k]:.1f} dB")


if __name__ == "__main__":
    print("Physics Building (motion sensor lights environment):")
    plot_spectrum_audacity_style(
        "spectrum_iu_lab_mar17.txt",
        "Power Spectrum - Physics Building (Mar 17)\nMotion sensor lights environment",
        "spectrum_iu_lab_mar17.png"
    )

    print("\nHallway Mirror (clean home environment):")
    plot_spectrum_audacity_style(
        "mar22_hallway_mirror.txt",
        "Power Spectrum - Hallway Mirror Reflection (Mar 22)\nClean home environment",
        "spectrum_hallway_mirror_mar22.png"
    )
