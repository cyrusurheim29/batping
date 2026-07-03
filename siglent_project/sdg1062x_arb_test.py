#!/usr/bin/env python3
"""
SDG1062X arbitrary waveform upload - cleaner approach.
"""

import pyvisa
import numpy as np
import time
import sys


def clear_buffers(device):
    """Clear any pending data in buffers."""
    device.write("*CLS")
    time.sleep(0.1)
    try:
        # Read any pending data
        device.timeout = 500
        while True:
            device.read_raw(1024)
    except:
        pass
    device.timeout = 10000


def query_clean(device, cmd):
    """Query with buffer clearing."""
    time.sleep(0.2)
    try:
        result = device.query(cmd).strip()
        return result
    except Exception as e:
        return f"(error: {e})"


def main():
    print("=" * 60)
    print("SDG1062X: Arbitrary Waveform Test (Clean)")
    print("=" * 60)

    rm = pyvisa.ResourceManager('@py')
    resources = rm.list_resources()

    sdg = None
    for r in resources:
        if 'USB' in r and ('SDG' in r or '62700' in r):
            sdg = r
            break

    if not sdg:
        print("SDG not found!")
        return 1

    print(f"Connecting to: {sdg}")
    device = rm.open_resource(sdg)
    device.timeout = 10000
    device.read_termination = '\n'
    device.write_termination = '\n'
    device.chunk_size = 1024 * 1024  # 1MB chunks

    idn = query_clean(device, "*IDN?")
    print(f"Device: {idn}")

    # Clear and reset
    print("\nClearing device state...")
    device.write("*RST")
    time.sleep(2)
    device.write("*CLS")
    time.sleep(0.5)

    # Check what arbitrary waveforms exist
    print("\n--- Stored Arbitrary Waveforms ---")
    arwv_info = query_clean(device, "C1:ARWV?")
    print(f"C1:ARWV?: {arwv_info}")

    # Method 1: Try STL (store list) command format
    # SDG1000X may use different command than SDG2000X
    print("\n--- Method 1: WVDT with terminator ---")

    # Simple 32-point sine wave (minimum complexity)
    n_points = 32
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    waveform = np.sin(t)
    waveform_int16 = (waveform * 32767).astype('<i2')

    wave_bytes = waveform_int16.tobytes()
    byte_count = len(wave_bytes)

    print(f"Waveform: {n_points} points, {byte_count} bytes")

    # Try different WVDT formats
    name = "MYSINE"

    # Format A: With IEEE 488.2 block header
    header = f"#{len(str(byte_count))}{byte_count}"
    cmd_a = f"C1:WVDT WVNM,{name},WAVEDATA,{header}".encode('ascii') + wave_bytes + b'\n'

    print(f"\nFormat A: C1:WVDT WVNM,{name},WAVEDATA,{header}<data>")
    response = input("Send? (y/N): ").strip().lower()

    if response == 'y':
        try:
            device.write_raw(cmd_a)
            time.sleep(2)

            # Wait and query
            device.write("*WAI")
            time.sleep(0.5)

            # Try to select it
            device.write(f"C1:ARWV NAME,{name}")
            time.sleep(0.5)

            result = query_clean(device, "C1:ARWV?")
            print(f"C1:ARWV?: {result}")

            if name.lower() in result.lower():
                print(">>> SUCCESS! <<<")
            else:
                print("Not selected.")

        except Exception as e:
            print(f"Error: {e}")

    # Method 2: Try writing to specific memory slot
    print("\n--- Method 2: Direct memory slot M0 ---")

    # Format: WVDT M0,WVNM,<name>,WAVEDATA,<data>
    cmd_b = f"WVDT M0,WVNM,TESTWV,WAVEDATA,{header}".encode('ascii') + wave_bytes + b'\n'

    print("Format B: WVDT M0,WVNM,TESTWV,WAVEDATA,<block><data>")
    response = input("Send? (y/N): ").strip().lower()

    if response == 'y':
        try:
            device.write_raw(cmd_b)
            time.sleep(2)

            # Select from M0
            device.write("C1:BSWV WVTP,ARB")
            time.sleep(0.3)
            device.write("C1:ARWV INDEX,0")
            time.sleep(0.5)

            result = query_clean(device, "C1:ARWV?")
            print(f"C1:ARWV?: {result}")

            bswv = query_clean(device, "C1:BSWV?")
            print(f"C1:BSWV?: {bswv}")

        except Exception as e:
            print(f"Error: {e}")

    # Method 3: Check if device supports WVDT at all
    print("\n--- Method 3: Query capabilities ---")

    # Some SDG models use different commands
    print("Trying to list all stored waveforms...")

    for i in range(10):
        device.write(f"C1:ARWV INDEX,{i}")
        time.sleep(0.2)
        result = query_clean(device, "C1:ARWV?")
        if result and "error" not in result.lower():
            print(f"  Index {i}: {result}")

    device.close()
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
