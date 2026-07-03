#!/usr/bin/env python3
"""
SDG1062X Arbitrary Waveform Upload - Direct PyVISA
Based on Siglent Programming Guide Section 3.32 and 5.1.4

Usage: sudo python sdg1062x_upload.py
"""

import pyvisa
import numpy as np
import time
import binascii


def wave_to_binary(wave_data):
    """
    Convert float waveform [-1, 1] to binary format for SDG.
    Returns bytes ready to send.
    """
    # Clip and convert to 16-bit signed integers
    wave_clipped = np.clip(wave_data, -1.0, 1.0)
    wave_int16 = (wave_clipped * 32767).astype(np.int16)

    # Convert to hex string (4 hex chars per sample)
    hex_data = ''
    for val in wave_int16:
        # Handle negative values (2's complement)
        if val < 0:
            val = val + 65536
        hex_data += format(val, '04x')

    # Convert hex string to binary bytes
    return binascii.a2b_hex(hex_data)


def main():
    print("=" * 60)
    print("SDG1062X Arbitrary Waveform Upload (Direct PyVISA)")
    print("=" * 60)

    # Connect
    rm = pyvisa.ResourceManager()
    resources = rm.list_resources()
    print(f"Found: {resources}")

    # Find SDG
    sdg_resource = None
    for r in resources:
        if 'USB' in r and ('F4EC' in r or '62700' in r or 'SDG' in r.upper()):
            sdg_resource = r
            break

    if not sdg_resource:
        print("SDG1062X not found!")
        return 1

    print(f"\nConnecting to: {sdg_resource}")
    dev = rm.open_resource(sdg_resource)
    dev.timeout = 5000  # 5 seconds
    dev.chunk_size = 1024 * 1024  # 1MB chunks for large waveforms

    # Try different termination settings for Siglent
    dev.read_termination = None  # No auto termination
    dev.write_termination = '\n'

    # Query ID - fire and forget style, don't wait for response
    print("Sending *IDN?...")
    dev.write("*IDN?")
    time.sleep(0.5)
    try:
        idn = dev.read_bytes(256).decode('utf-8', errors='ignore').strip()
        print(f"Device: {idn}")
    except Exception as e:
        print(f"IDN read failed: {e}")
        print("(Writes may still work - Siglent USB can be finicky)")

    # Clear any errors
    dev.write("*CLS")
    time.sleep(0.2)

    # Create test waveform - simple 256-point sine
    print("\n--- Creating Test Waveform ---")
    n_points = 256
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    wave = np.sin(t)

    print(f"Waveform: {n_points} points, sine wave")
    print(f"Range: [{wave.min():.3f}, {wave.max():.3f}]")

    # Convert to binary
    bin_data = wave_to_binary(wave)
    print(f"Binary data: {len(bin_data)} bytes")

    # Build command
    name = "sine256"
    freq = 1000.0
    ampl = 2.0
    ofst = 0.0
    phase = 0.0

    # Command format from manual (Section 3.32):
    # C1:WVDT WVNM,<name>,FREQ,<f>,AMPL,<a>,OFST,<o>,PHASE,<p>,WAVEDATA,<binary>
    cmd_header = f"C1:WVDT WVNM,{name},FREQ,{freq},AMPL,{ampl},OFST,{ofst},PHASE,{phase},WAVEDATA,"

    print(f"\nCommand: {cmd_header}[{len(bin_data)} bytes of data]")

    response = input("\nUpload waveform? (y/N): ").strip().lower()
    if response != 'y':
        dev.close()
        return 0

    # Send command + binary data
    print("Sending...")
    full_cmd = cmd_header.encode('utf-8') + bin_data
    dev.write_raw(full_cmd)
    time.sleep(1.0)

    # Check for errors
    try:
        dev.write("*OPC?")
        time.sleep(0.3)
        opc = dev.read().strip()
        print(f"*OPC?: {opc}")
    except:
        print("(OPC query failed, continuing...)")

    # Set to ARB mode and select waveform
    print("\nSelecting waveform...")
    dev.write("C1:BSWV WVTP,ARB")
    time.sleep(0.3)

    dev.write(f"C1:ARWV NAME,{name}")
    time.sleep(0.5)

    # Verify
    try:
        dev.write("C1:ARWV?")
        time.sleep(0.3)
        result = dev.read().strip()
        print(f"C1:ARWV?: {result}")

        if name.lower() in result.lower():
            print("\n>>> SUCCESS! Waveform uploaded and selected! <<<")
        else:
            print("\nWaveform uploaded. Check device front panel under Arb menu.")
    except Exception as e:
        print(f"Query failed: {e}")
        print("Check device front panel to verify waveform was uploaded.")

    # Ask about output
    enable = input("\nEnable CH1 output? (y/N): ").strip().lower()
    if enable == 'y':
        dev.write("C1:OUTP ON")
        print("CH1 output enabled")

    dev.close()
    print("\nDone!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
