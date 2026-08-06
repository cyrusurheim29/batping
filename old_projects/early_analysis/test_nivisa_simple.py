#!/usr/bin/env python3
"""
Simple NI-VISA WVCSV test - skips OPC check, goes straight to verification
"""

import pyvisa
import numpy as np
import time

def main():
    print("="*50)
    print("NI-VISA WVCSV Simple Test")
    print("="*50)

    # Connect with NI-VISA
    rm = pyvisa.ResourceManager()
    print(f"Using: {rm.visalib}")

    resources = rm.list_resources()
    ws_resource = [r for r in resources if 'USB' in r and ('05FF' in r or 'LCRY' in r)]

    if not ws_resource:
        print("WaveStation not found!")
        return

    device = rm.open_resource(ws_resource[0])
    device.timeout = 120000  # 2 minute timeout
    print(f"Connected to: {device.query('*IDN?').strip()}")

    # Test with different point counts
    for n_points in [100, 512, 1024]:
        print(f"\n{'='*50}")
        print(f"Testing with {n_points} points")
        print('='*50)

        # Generate simple waveform
        t = np.linspace(0, 2*np.pi*5, n_points)
        waveform = np.sin(t) * 0.5  # 0.5V amplitude
        data_values = ",".join([f"{v:.4f}" for v in waveform])

        cmd = f"WVCSV M50,WAVENM,TEST{n_points},CSVLENG,{n_points},CSVDATA,Amplitude,{data_values}"
        print(f"Command: {len(cmd)} bytes")

        # Send and wait
        device.write(cmd)
        print("Sent. Waiting 10 seconds for processing...")
        time.sleep(10)

        # Clear any pending responses
        try:
            device.read_raw()
        except:
            pass

        # Check if it's stored by trying to select it
        print("Checking if waveform stored...")

        device.write("C1:BSWV WVTP,ARB")
        time.sleep(0.5)

        # Try NAME
        device.write(f"C1:ARWV NAME,TEST{n_points}")
        time.sleep(0.5)
        result = device.query("C1:ARWV?").strip()
        print(f"ARWV after NAME: {result}")

        if f"TEST{n_points}" in result.upper():
            print(f"\n*** SUCCESS with {n_points} points! ***")
            break

        # Try INDEX
        device.write("C1:ARWV INDEX,50")
        time.sleep(0.5)
        result = device.query("C1:ARWV?").strip()
        print(f"ARWV after INDEX,50: {result}")

        if "stairup" not in result.lower():
            print(f"\n*** DIFFERENT waveform at INDEX 50! Might have worked! ***")
            break

    print("\n" + "="*50)
    print("CHECK THE DEVICE SCREEN:")
    print("  Arb -> 1/2 down -> Load Wform -> Stored Wforms")
    print("="*50)

    device.close()


if __name__ == "__main__":
    main()
