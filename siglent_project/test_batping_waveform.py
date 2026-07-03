#!/usr/bin/env python3
"""
Test script to visualize the BATPING waveform before uploading.
"""

import numpy as np
import matplotlib.pyplot as plt


def create_batping_pulse(rep_rate_hz=62.5, carrier_freq_hz=40000, sigma_cycles=5,
                         samples_per_carrier_cycle=10):
    """
    Create a proper BATPING pulse with 40kHz carrier.
    """
    period_sec = 1.0 / rep_rate_hz
    carrier_period_sec = 1.0 / carrier_freq_hz
    total_carrier_cycles = period_sec * carrier_freq_hz
    n_points = int(total_carrier_cycles * samples_per_carrier_cycle)

    max_points = 16384
    if n_points > max_points:
        print(f"Warning: {n_points} points needed, limiting to {max_points}")
        n_points = max_points
        samples_per_carrier_cycle = n_points / total_carrier_cycles

    print(f"  Period: {period_sec*1000:.1f} ms")
    print(f"  Carrier cycles per period: {total_carrier_cycles:.0f}")
    print(f"  Samples per carrier cycle: {samples_per_carrier_cycle:.1f}")
    print(f"  Total points: {n_points}")
    print(f"  Required sample rate: {n_points * rep_rate_hz / 1e6:.3f} MSa/s")

    t = np.linspace(0, period_sec, n_points, endpoint=False)
    t_center = period_sec / 2
    sigma_sec = sigma_cycles * carrier_period_sec

    envelope = np.exp(-0.5 * ((t - t_center) / sigma_sec)**2)
    carrier = np.sin(2 * np.pi * carrier_freq_hz * t)
    pulse = envelope * carrier
    pulse = pulse / np.max(np.abs(pulse))

    return pulse, n_points, t


def main():
    print("Creating BATPING waveform...")
    pulse, n_points, t = create_batping_pulse(
        rep_rate_hz=62.5,
        carrier_freq_hz=40000,
        sigma_cycles=5,
        samples_per_carrier_cycle=10
    )

    t_ms = t * 1000  # Convert to ms

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Full waveform
    axes[0].plot(t_ms, pulse, 'b-', linewidth=0.5)
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'Full BATPING Waveform ({n_points} points, 62.5 Hz rep rate)')
    axes[0].grid(True, alpha=0.3)

    # Zoomed to center (where the Gaussian envelope is)
    center_idx = n_points // 2
    zoom_width = n_points // 8  # Show ~2ms around center
    start = center_idx - zoom_width
    end = center_idx + zoom_width

    axes[1].plot(t_ms[start:end], pulse[start:end], 'b-', linewidth=0.8)
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Zoomed: Center of Gaussian envelope')
    axes[1].grid(True, alpha=0.3)

    # Envelope only
    envelope = np.exp(-0.5 * ((t - t[n_points//2]) / (5 * 1/40000))**2)
    axes[2].plot(t_ms, envelope, 'r-', linewidth=1, label='Gaussian envelope')
    axes[2].plot(t_ms, np.abs(pulse), 'b-', linewidth=0.3, alpha=0.5, label='|pulse|')
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_title('Gaussian Envelope')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/cyrusurheim_1/Downloads/Stubbs_Stuff/Claude_Dir/siglent_project/batping_waveform_test.png', dpi=150)
    print("\nSaved: batping_waveform_test.png")
    plt.show()

    # Print some stats
    print(f"\nWaveform stats:")
    print(f"  Min: {pulse.min():.4f}")
    print(f"  Max: {pulse.max():.4f}")
    print(f"  Non-zero samples (>0.01): {np.sum(np.abs(pulse) > 0.01)}")


if __name__ == "__main__":
    main()
