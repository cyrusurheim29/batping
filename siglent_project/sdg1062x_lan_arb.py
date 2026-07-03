#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 10:49:30 2026

@author: cyrusurheimog
"""

import socket
import numpy as np
import time
import binascii


class SDG1062X_LAN:
    def __init__(self, ip_address, port=5025, timeout=5.0):
        self.ip = ip_address
        self.port = port
        self.timeout = timeout
        self.sock = None

    def connect(self):
        """Connect to SDG via LAN socket."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect((self.ip, self.port))
        print(f"Connected to {self.ip}:{self.port}")

    def disconnect(self):
        """Close connection."""
        if self.sock:
            self.sock.close()
            self.sock = None
            print("Disconnected")

    def write(self, cmd):
        """Send command (adds newline terminator)."""
        self.sock.sendall((cmd + '\n').encode('utf-8'))

    def write_raw(self, data):
        """Send raw bytes."""
        self.sock.sendall(data)

    def read(self, bufsize=4096):
        """Read response."""
        return self.sock.recv(bufsize).decode('utf-8').strip()

    def query(self, cmd):
        """Send command and read response."""
        self.write(cmd)
        time.sleep(0.1)
        return self.read()

    def idn(self):
        """Query device identity."""
        return self.query("*IDN?")

    def upload_arb_waveform(self, name, wave_data, freq_hz=1000.0, ampl_vpp=2.0,
                            offset_v=0.0, phase_deg=0.0, channel=1):

        # Convert float array to 16-bit signed integers
        # SDG uses range -32768 to 32767
        wave_int16 = np.clip(wave_data, -1.0, 1.0)
        wave_int16 = (wave_int16 * 32767).astype(np.int16)

        # Convert to binary - SDG expects little-endian 16-bit integers
        bin_data = wave_int16.astype('<i2').tobytes()

        print(f"Waveform: {name}")
        print(f"  Points: {len(wave_data)}")
        print(f"  Binary size: {len(bin_data)} bytes")
        print(f"  Freq: {freq_hz} Hz, Ampl: {ampl_vpp} Vpp")

        # Build WVDT command (SDG1000X/SDG2000X/SDG6000X format)
        # Format: C1:WVDT WVNM,<name>,FREQ,<f>,AMPL,<a>,OFST,<o>,PHASE,<p>,WAVEDATA,<data>
        cmd_prefix = f"C{channel}:WVDT WVNM,{name},FREQ,{freq_hz},AMPL,{ampl_vpp},OFST,{offset_v},PHASE,{phase_deg},WAVEDATA,"

        # Send command prefix + binary data
        full_cmd = cmd_prefix.encode('utf-8') + bin_data + b'\n'

        print(f"Sending {len(full_cmd)} bytes...")
        self.write_raw(full_cmd)
        time.sleep(1.0)  # Give device time to process

        # Select the waveform
        self.write(f"C{channel}:ARWV NAME,{name}")
        time.sleep(0.5)

        # Verify
        result = self.query(f"C{channel}:ARWV?")
        print(f"ARWV?: {result}")

        return name in result.upper() or name.lower() in result.lower()

    def set_arb_mode(self, channel=1):
        """Set channel to arbitrary waveform mode."""
        self.write(f"C{channel}:BSWV WVTP,ARB")
        time.sleep(0.3)

    def output_on(self, channel=1):
        """Enable output."""
        self.write(f"C{channel}:OUTP ON")

    def output_off(self, channel=1):
        """Disable output."""
        self.write(f"C{channel}:OUTP OFF")


def create_gaussian_pulse(n_points, sigma_cycles=5, carrier_freq_cycles=1):
    t = np.linspace(-n_points/2, n_points/2, n_points)
    sigma = n_points / (2 * sigma_cycles)

    # Gaussian envelope
    envelope = np.exp(-0.5 * (t / sigma)**2)

    # Carrier
    carrier = np.sin(2 * np.pi * carrier_freq_cycles * t / n_points)

    # Combine
    pulse = envelope * carrier

    # Normalize to [-1, 1]
    pulse = pulse / np.max(np.abs(pulse))

    return pulse


def create_simple_sine(n_points, n_cycles=1):
    """Create simple sine wave."""
    t = np.linspace(0, 2 * np.pi * n_cycles, n_points, endpoint=False)
    return np.sin(t)


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

    t = np.linspace(0, period_sec, n_points, endpoint=False)
    t_center = period_sec / 2
    sigma_sec = sigma_cycles * carrier_period_sec

    envelope = np.exp(-0.5 * ((t - t_center) / sigma_sec)**2)
    carrier = np.sin(2 * np.pi * carrier_freq_hz * t)
    pulse = envelope * carrier
    pulse = pulse / np.max(np.abs(pulse))

    return pulse, n_points


def main():
    print("=" * 60)
    print("SDG1062X Arbitrary Waveform Upload via LAN")
    print("=" * 60)

    # Get IP address
    ip = input("Enter SDG1062X IP address (check Utility > I/O on device): ").strip()
    if not ip:
        print("No IP provided. To find the IP:")
        print("  1. On SDG1062X, press Utility button")
        print("  2. Navigate to I/O Config or Interface")
        print("  3. Look for LAN settings and IP address")
        print("  4. Make sure LAN is enabled and connected")
        return 1

    try:
        sdg = SDG1062X_LAN(ip)
        sdg.connect()

        # Query identity
        idn = sdg.idn()
        print(f"\nDevice: {idn}")

        if 'SDG' not in idn.upper():
            print("Warning: Device doesn't appear to be an SDG")

        # Create test waveform
        print("\n--- Creating Test Waveform ---")

        print("\nOption 1: 8-point staircase (sanity check)")
        print("Option 2: 40kHz BATPING Gaussian pulse @ 62.5Hz rep rate")
        print("Option 3: 256-point sine wave")

        choice = input("\nWhich waveform to upload? (1/2/3, default=2): ").strip()

        if choice == '1':
            wave = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0])
            name = "stair8"
            freq = 1000.0
        elif choice == '3':
            wave = create_simple_sine(256, n_cycles=1)
            name = "sine256"
            freq = 1000.0
        else:
            print("\nCreating BATPING pulse:")
            wave, n_pts = create_batping_pulse(
                rep_rate_hz=62.5,
                carrier_freq_hz=40000,
                sigma_cycles=5,
                samples_per_carrier_cycle=10
            )
            name = "batping"
            freq = 62.5

        print(f"\nUploading '{name}' ({len(wave)} points, {freq} Hz)...")

        # Set to ARB mode first
        sdg.set_arb_mode(channel=1)

        # Upload waveform
        success = sdg.upload_arb_waveform(
            name=name,
            wave_data=wave,
            freq_hz=freq,
            ampl_vpp=2.0,
            offset_v=0.0,
            channel=1
        )

        if success:
            print("\n>>> SUCCESS! Waveform uploaded and selected! <<<")

            enable = input("\nEnable CH1 output? (y/N): ").strip().lower()
            if enable == 'y':
                sdg.output_on(channel=1)
                print("CH1 output enabled")
        else:
            print("\nWaveform may not have been selected correctly.")
            print("Check the device front panel.")

        sdg.disconnect()
        return 0

    except socket.timeout:
        print("Connection timed out. Check:")
        print(f"  - IP address is correct: {ip}")
        print("  - Device is on the same network")
        print("  - LAN is enabled on the device")
        return 1
    except ConnectionRefusedError:
        print("Connection refused. Check:")
        print(f"  - IP address is correct: {ip}")
        print("  - Port 5025 is not blocked")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
