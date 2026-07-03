#!/usr/bin/env python3
"""
WaveStation: Test with minimum 32 points.

From WaveStation specs: Minimum arbitrary waveform = 32 points
Maybe 8 points is causing the crash because it's below minimum?
"""

import pyvisa
import time
import sys
import numpy as np

def main():
    print("=" * 60)
    print("WaveStation: 32-Point Minimum Test")
    print("=" * 60)

    rm = pyvisa.ResourceManager()
    print(f"VISA: {rm.visalib}")

    resources = rm.list_resources()
    ws_resource = [r for r in resources if '05FF' in r or 'LCRY' in r]

    if not ws_resource:
        print("WaveStation not found!")
        return 1

    device = rm.open_resource(ws_resource[0])
    device.timeout = 30000

    idn = device.query("*IDN?").strip()
    print(f"Connected: {idn}")

    # Clear status
    device.write("*CLS")
    time.sleep(0.3)

    # Generate 32-point sine wave
    print("\nGenerating 32-point sine wave...")
    t = np.linspace(0, 2*np.pi, 32, endpoint=False)
    sine32 = np.sin(t) * 0.5  # 0.5V amplitude, within safe range

    # Format as CSV data
    data_values = ",".join([f"{v:.4f}" for v in sine32])
    print(f"Data: {data_values[:50]}...")
    print(f"Number of values: {len(sine32)}")

    # Build command
    cmd = f"WVCSV M50,WAVENM,SINE32,CSVLENG,32,CSVDATA,Amplitude,{data_values}"
    print(f"Command length: {len(cmd)} bytes")

    print("\n" + "=" * 60)
    print("Sending WVCSV with 32 points (minimum allowed)")
    print("=" * 60)

    response = input("Send command? (y/N): ").strip().lower()
    if response != 'y':
        device.close()
        return 0

    print("\nSending (watch device screen)...")
    try:
        bytes_written = device.write(cmd)
        print(f"Bytes written: {bytes_written}")

        print("Waiting 3 seconds...")
        time.sleep(3)

        opc = device.query("*OPC?").strip()
        print(f"*OPC?: {opc}")
        print(">>> Device still responding! <<<")

        # Verify
        device.write("C1:BSWV WVTP,ARB")
        time.sleep(0.3)
        device.write("C1:ARWV NAME,SINE32")
        time.sleep(0.5)
        result = device.query("C1:ARWV?").strip()
        print(f"ARWV: {result}")

        if "SINE32" in result.upper():
            print("\n>>> SUCCESS! 32-point waveform uploaded! <<<")
        else:
            device.write("C1:ARWV INDEX,50")
            time.sleep(0.5)
            result = device.query("C1:ARWV?").strip()
            print(f"ARWV INDEX,50: {result}")

    except Exception as e:
        print(f">>> ERROR: {e} <<<")
        print("Device crashed even with 32 points")
        return 1

    device.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
