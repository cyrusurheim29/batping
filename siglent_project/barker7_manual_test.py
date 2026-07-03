#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Barker-7 Manual Test Script

Simple script to upload Barker-7 BPSK waveforms to SDG for manual inspection.
Run this, then use Audacity to record and examine the output.

Usage:
    python barker7_manual_test.py [cycles_per_chip]

    Default: 24 cycles/chip

The waveform will play continuously until you press Enter to stop.
"""

import numpy as np
import socket
import time
import sys
from scipy.io import wavfile

# Hardware configuration
SDG_IP = "10.11.13.230"
SDG_PORT = 5025
SDG_SAMPLE_RATE = 400000  # 400 kSa/s

# Waveform configuration
CARRIER_FREQ = 41350  # Hz (center from stepped-sine characterization)
AMPLITUDE = 0.9

# V3: Inter-pulse spacing for piezo settling
TARGET_REP_RATE = 40  # Hz (25 ms period allows ~20 ms settling)
SDG_MAX_SAMPLES = 16384

# Barker-7 code
BARKER_7 = [+1, +1, +1, -1, -1, +1, -1]

OUTPUT_DIR = '/Users/cyrusurheim_1/Downloads/Stubbs_Stuff/Claude_Dir/siglent_project/'


class SDG1062X_LAN:
    """Minimal SDG control for testing."""

    def __init__(self, ip=SDG_IP, port=SDG_PORT, timeout=10.0):
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self.sock = None

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect((self.ip, self.port))
        print(f"Connected to SDG @ {self.ip}:{self.port}")

    def disconnect(self):
        if self.sock:
            self.sock.close()
            self.sock = None
            print("Disconnected")

    def write(self, cmd):
        self.sock.sendall((cmd + '\n').encode('utf-8'))

    def write_raw(self, data):
        self.sock.sendall(data)

    def read(self, bufsize=4096):
        return self.sock.recv(bufsize).decode('utf-8').strip()

    def query(self, cmd):
        self.write(cmd)
        time.sleep(0.1)
        return self.read()

    def upload_waveform(self, name, wave_data, freq_hz, ampl_vpp=2.0, channel=1):
        """Upload waveform with little-endian 16-bit encoding."""
        wave_int16 = np.clip(wave_data, -1.0, 1.0)
        wave_int16 = (wave_int16 * 32767).astype(np.int16)
        bin_data = wave_int16.astype('<i2').tobytes()

        cmd_prefix = f"C{channel}:WVDT WVNM,{name},FREQ,{freq_hz},AMPL,{ampl_vpp},OFST,0,PHASE,0,WAVEDATA,"
        full_cmd = cmd_prefix.encode('utf-8') + bin_data + b'\n'

        print(f"  Uploading {len(wave_data)} samples ({len(bin_data)} bytes)...")
        self.write_raw(full_cmd)
        time.sleep(1.0)

        self.write(f"C{channel}:ARWV NAME,{name}")
        time.sleep(0.3)
        print(f"  Selected waveform: {name}")

    def set_arb_mode(self, channel=1):
        self.write(f"C{channel}:BSWV WVTP,ARB")
        time.sleep(0.2)

    def output_on(self, channel=1):
        self.write(f"C{channel}:OUTP ON")
        print(f"  Output ON (CH{channel})")

    def output_off(self, channel=1):
        self.write(f"C{channel}:OUTP OFF")
        print(f"  Output OFF (CH{channel})")


def generate_barker_bpsk_simple(code, carrier_freq, cycles_per_chip, sample_rate):
    """
    Generate BPSK-modulated Barker code at SDG sample rate.

    Returns the waveform normalized to [-1, 1].
    """
    chip_duration = cycles_per_chip / carrier_freq  # seconds
    samples_per_chip = int(chip_duration * sample_rate)
    total_samples = samples_per_chip * len(code)

    print(f"\nWaveform parameters:")
    print(f"  Carrier: {carrier_freq} Hz")
    print(f"  Cycles/chip: {cycles_per_chip}")
    print(f"  Chip duration: {chip_duration*1e6:.1f} μs")
    print(f"  Samples/chip: {samples_per_chip}")
    print(f"  Total samples: {total_samples}")
    print(f"  Total duration: {total_samples/sample_rate*1e3:.2f} ms")

    if total_samples > 16384:
        print(f"  WARNING: {total_samples} exceeds SDG limit of 16384!")

    waveform = np.zeros(total_samples)

    for i, chip in enumerate(code):
        start = i * samples_per_chip
        end = start + samples_per_chip
        t_chip = np.arange(samples_per_chip) / sample_rate
        # BPSK: multiply carrier by +1 or -1
        waveform[start:end] = chip * np.sin(2 * np.pi * carrier_freq * t_chip)

    # Normalize
    waveform = waveform / np.max(np.abs(waveform)) * AMPLITUDE

    return waveform


def save_waveform_wav(waveform, sample_rate, filename):
    """Save waveform as WAV file for inspection."""
    # Scale to int16
    wav_data = (waveform * 32767).astype(np.int16)
    wavfile.write(filename, sample_rate, wav_data)
    print(f"  Saved WAV: {filename}")


def main():
    # Parse command line
    if len(sys.argv) > 1:
        cycles_per_chip = int(sys.argv[1])
    else:
        cycles_per_chip = 24

    print("=" * 60)
    print("Barker-7 BPSK Manual Test")
    print(f"Carrier: {CARRIER_FREQ} Hz, Cycles/chip: {cycles_per_chip}")
    print("=" * 60)

    # Generate waveform at SDG sample rate
    waveform = generate_barker_bpsk_simple(
        BARKER_7, CARRIER_FREQ, cycles_per_chip, SDG_SAMPLE_RATE
    )

    # Save as WAV for inspection
    wav_path = OUTPUT_DIR + f'barker7_{cycles_per_chip}cyc_waveform_V5.wav'
    save_waveform_wav(waveform, SDG_SAMPLE_RATE, wav_path)

    # Also save a plot of the waveform
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        chip_duration_ms = cycles_per_chip / CARRIER_FREQ * 1000

        fig, axes = plt.subplots(2, 1, figsize=(12, 6))

        t_ms = np.arange(len(waveform)) / SDG_SAMPLE_RATE * 1000

        # Full waveform
        axes[0].plot(t_ms, waveform, 'b-', linewidth=0.5)
        axes[0].set_xlabel('Time (ms)')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title(f'Barker-7 BPSK Waveform ({cycles_per_chip} cycles/chip, {CARRIER_FREQ} Hz)')
        axes[0].grid(True, alpha=0.3)

        # Zoom to first 2 chips to see BPSK transitions
        zoom_end = 2.5 * chip_duration_ms
        zoom_mask = t_ms <= zoom_end
        axes[1].plot(t_ms[zoom_mask], waveform[zoom_mask], 'b-', linewidth=1)
        axes[1].set_xlabel('Time (ms)')
        axes[1].set_ylabel('Amplitude')
        axes[1].set_title('Zoom: First 2.5 chips (look for phase inversions)')
        axes[1].grid(True, alpha=0.3)

        # Mark chip boundaries
        for i in range(3):
            axes[1].axvline(i * chip_duration_ms, color='r', linestyle='--', alpha=0.5)

        # Annotate chip values
        for i, chip in enumerate(BARKER_7[:3]):
            x_pos = (i + 0.5) * chip_duration_ms
            axes[1].text(x_pos, 1.1, f'chip={chip:+d}', ha='center', fontsize=10)

        plt.tight_layout()
        plot_path = OUTPUT_DIR + f'barker7_{cycles_per_chip}cyc_waveform_V5.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Saved plot: {plot_path}")
    except Exception as e:
        print(f"  Warning: Could not save plot: {e}")

    # Connect to SDG and upload only
    print("\n" + "-" * 60)
    sdg = SDG1062X_LAN()
    try:
        sdg.connect()
        idn = sdg.query("*IDN?")
        print(f"Device: {idn}")

        # Set up ARB mode
        sdg.set_arb_mode(channel=1)

        # V3: Embed waveform in larger buffer for inter-pulse settling
        buffer_samples = int(SDG_SAMPLE_RATE / TARGET_REP_RATE)  # 10,000 samples for 40 Hz

        if buffer_samples > SDG_MAX_SAMPLES:
            buffer_samples = SDG_MAX_SAMPLES
            actual_rep_rate = SDG_SAMPLE_RATE / buffer_samples
            print(f"  Warning: Using max buffer, rep_rate = {actual_rep_rate:.1f} Hz")
        else:
            actual_rep_rate = TARGET_REP_RATE

        # Create buffer with waveform at start, zeros for settling
        waveform_buffer = np.zeros(buffer_samples)
        waveform_buffer[:len(waveform)] = waveform

        wf_duration_ms = len(waveform) / SDG_SAMPLE_RATE * 1000
        dead_time_ms = (buffer_samples - len(waveform)) / SDG_SAMPLE_RATE * 1000

        print(f"\nUpload info (V3 - with inter-pulse spacing):")
        print(f"  Waveform samples: {len(waveform)}")
        print(f"  Buffer samples: {buffer_samples}")
        print(f"  Waveform duration: {wf_duration_ms:.2f} ms")
        print(f"  Dead time (settling): {dead_time_ms:.1f} ms")
        print(f"  Rep rate: {actual_rep_rate:.0f} Hz")

        sdg.upload_waveform(f"bk7_{cycles_per_chip}_V5", waveform_buffer, actual_rep_rate, ampl_vpp=2.0)

        print("\n" + "=" * 60)
        print(f"Waveform 'bk7_{cycles_per_chip}_V5' uploaded.")
        print(f"Expected carrier: {CARRIER_FREQ} Hz")
        print(f"Waveform repeats at: {actual_rep_rate:.0f} Hz (with {dead_time_ms:.0f} ms settling)")
        print("Output is OFF. Enable manually on SDG front panel.")
        print("=" * 60)

    finally:
        sdg.disconnect()

    print("\nDone!")


if __name__ == '__main__':
    main()
