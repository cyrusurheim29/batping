#!/usr/bin/env python3
"""Plot waveform showing beats between 40kHz signal and 34kHz noise."""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import aifc


def read_aiff(filename):
    """Read an AIFF audio file."""
    with aifc.open(filename, 'r') as f:
        n_channels = f.getnchannels()
        sample_width = f.getsampwidth()
        sample_rate = f.getframerate()
        n_frames = f.getnframes()
        raw_data = f.readframes(n_frames)

        if sample_width == 2:
            data = np.frombuffer(raw_data, dtype='>i2')
        elif sample_width == 4:
            data = np.frombuffer(raw_data, dtype='>i4')
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        if n_channels == 2:
            data = data[::2]

        max_val = 2 ** (sample_width * 8 - 1)
        data = data.astype(np.float64) / max_val

        return sample_rate, data


# Load data
sample_rate, data = read_aiff("iu_lab_aligned_mar16.aiff")
print(f"Sample rate: {sample_rate} Hz")

# Time range: 1.750 to 1.751 seconds (1 ms window)
t_start = 1.750
t_end = 1.751

start_idx = int(t_start * sample_rate)
end_idx = int(t_end * sample_rate)

t = np.arange(start_idx, end_idx) / sample_rate
waveform = data[start_idx:end_idx]

print(f"Plotting {len(waveform)} samples from {t_start} to {t_end} s")

# Calculate beat frequency
f1 = 40000  # Hz (signal)
f2 = 33000  # Hz (interference)
beat_freq = abs(f1 - f2)
beat_period = 1 / beat_freq * 1000  # ms

# Plot
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(t * 1000, waveform, 'b-', linewidth=0.8)
ax.fill_between(t * 1000, waveform, 0, alpha=0.3, color='blue')

ax.set_xlabel('Time (ms)', fontsize=12)
ax.set_ylabel('Amplitude', fontsize=12)
ax.set_title(f'Waveform - Physics Building (Mar 17)\n'
             f'Beat frequency ≈ {beat_freq/1000:.0f} kHz (period ≈ {beat_period:.3f} ms)',
             fontsize=12, fontweight='bold')

ax.set_xlim([t_start * 1000, t_end * 1000])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('iu_lab_beats_waveform.png', dpi=150, facecolor='white')
print("Saved: iu_lab_beats_waveform.png")
