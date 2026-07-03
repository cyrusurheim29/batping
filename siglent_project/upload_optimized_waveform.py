#!/usr/bin/env python3
"""
Upload Bayesian-optimized waveform to SDG1062X.

The optimized waveform was created for Agilent @ 1 MSa/s (4000 points, 40kHz carrier).
This script adapts it for SDG1062X @ 62.5 Hz rep rate.
"""

import numpy as np
import socket
import time

# === Configuration ===
SDG_IP = "10.11.13.230"
SDG_PORT = 5025

# Target parameters for SDG
TARGET_REP_RATE = 62.5  # Hz
CARRIER_FREQ = 40000    # Hz

# Source waveform info (from Agilent setup)
SOURCE_SAMPLE_RATE = 1e6  # 1 MSa/s
SOURCE_POINTS = 4000

NPZ_PATH = '/Users/cyrusurheim_1/Downloads/Stubbs_Stuff/Claude_Dir/agilent_project/bayesian_optimization/bayesian_optimizer_state.npz'


class SDG1062X_LAN:
    def __init__(self, ip=SDG_IP, port=SDG_PORT, timeout=5.0):
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
        bin_data = wave_int16.astype('<i2').tobytes()  # Little-endian!

        print(f"  Binary data: {len(bin_data)} bytes ({len(wave_data)} points)")

        cmd_prefix = f"C{channel}:WVDT WVNM,{name},FREQ,{freq_hz},AMPL,{ampl_vpp},OFST,0,PHASE,0,WAVEDATA,"
        full_cmd = cmd_prefix.encode('utf-8') + bin_data + b'\n'

        self.write_raw(full_cmd)
        time.sleep(1.0)

        self.write(f"C{channel}:ARWV NAME,{name}")
        time.sleep(0.3)

    def set_arb_mode(self, channel=1):
        self.write(f"C{channel}:BSWV WVTP,ARB")
        time.sleep(0.3)

    def output_on(self, channel=1):
        self.write(f"C{channel}:OUTP ON")

    def output_off(self, channel=1):
        self.write(f"C{channel}:OUTP OFF")


def load_optimized_waveform():
    """Load and adapt the Bayesian-optimized waveform for SDG."""
    print(f"Loading: {NPZ_PATH}")
    state = np.load(NPZ_PATH, allow_pickle=True)

    orig_waveform = state['best_waveform']
    best_fwhm = float(state['best_fwhm'])
    baseline_fwhm = float(state['baseline_fwhm'])

    print(f"\nOriginal waveform:")
    print(f"  Points: {len(orig_waveform)}")
    print(f"  Best FWHM: {best_fwhm:.1f} μs (vs baseline {baseline_fwhm:.1f} μs)")
    print(f"  Improvement: {(1 - best_fwhm/baseline_fwhm)*100:.1f}%")

    # Original: 4000 points @ 1 MSa/s = 4ms period = 250 Hz
    # Target: 62.5 Hz = 16ms period
    # At 1 MSa/s, 16ms needs 16000 points (too many for SDG max 16384)

    # Strategy: Keep 40kHz carrier, embed pulse in longer buffer
    # Use same samples-per-carrier-cycle as Gaussian (10 samples/cycle)
    # 62.5 Hz @ 10 samples/40kHz cycle = 6400 points @ 400 kSa/s

    # Resample from 1 MSa/s (25 samples/cycle) to 400 kSa/s (10 samples/cycle)
    # Resample ratio: 400k / 1M = 0.4

    from scipy import signal

    # Calculate resample parameters
    orig_samples_per_cycle = SOURCE_SAMPLE_RATE / CARRIER_FREQ  # 25
    target_samples_per_cycle = 10
    resample_ratio = target_samples_per_cycle / orig_samples_per_cycle  # 0.4

    new_length = int(len(orig_waveform) * resample_ratio)
    resampled = signal.resample(orig_waveform, new_length)

    print(f"\nResampled waveform:")
    print(f"  Points: {len(resampled)} (from {len(orig_waveform)})")

    # Now embed in buffer for 62.5 Hz rep rate
    # 62.5 Hz @ 400 kSa/s = 6400 points per period
    target_points = 6400

    if len(resampled) > target_points:
        print(f"  Warning: Resampled pulse ({len(resampled)}) > target ({target_points})")
        # Trim to fit, centered
        excess = len(resampled) - target_points
        start = excess // 2
        final_waveform = resampled[start:start + target_points]
    else:
        # Pad with zeros, center the pulse
        final_waveform = np.zeros(target_points)
        start = (target_points - len(resampled)) // 2
        final_waveform[start:start + len(resampled)] = resampled

    # Normalize
    final_waveform = final_waveform / np.max(np.abs(final_waveform))

    print(f"\nFinal waveform for SDG:")
    print(f"  Points: {len(final_waveform)}")
    print(f"  Rep rate: {TARGET_REP_RATE} Hz")
    print(f"  Required sample rate: {len(final_waveform) * TARGET_REP_RATE / 1e6:.3f} MSa/s")
    print(f"  Min: {final_waveform.min():.4f}, Max: {final_waveform.max():.4f}")

    return final_waveform


def main():
    print("=" * 60)
    print("Upload Bayesian-Optimized Waveform to SDG1062X")
    print("=" * 60)

    # Load and adapt waveform
    waveform = load_optimized_waveform()

    # Connect and upload
    sdg = None
    try:
        print("\n--- Connecting to SDG1062X ---")
        sdg = SDG1062X_LAN()
        sdg.connect()

        idn = sdg.query("*IDN?")
        print(f"Device: {idn}")

        print("\n--- Uploading Optimized Waveform ---")
        sdg.set_arb_mode(channel=1)
        sdg.upload_waveform("optimized", waveform, freq_hz=TARGET_REP_RATE, ampl_vpp=2.0)

        print("\n>>> SUCCESS! Waveform uploaded <<<")
        print("\nIMPORTANT: Set SDG to TrueArb mode with sample rate = 400 kSa/s")

        enable = input("\nEnable CH1 output? (y/N): ").strip().lower()
        if enable == 'y':
            sdg.output_on(channel=1)
            print("CH1 ON")

        return 0

    except socket.timeout:
        print(f"\nConnection timed out. Check IP: {SDG_IP}")
        return 1
    except ConnectionRefusedError:
        print(f"\nConnection refused. Check SDG is on and LAN enabled.")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if sdg:
            sdg.disconnect()


if __name__ == "__main__":
    import sys
    sys.exit(main())
