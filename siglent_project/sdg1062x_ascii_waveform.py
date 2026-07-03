#!/usr/bin/env python3
"""
SDG1062X arbitrary waveform upload using ASCII format.
Based on Siglent's official example.
"""

import pyvisa
import numpy as np
import time
import sys


def main():
    print("=" * 60)
    print("SDG1062X: ASCII Waveform Upload")
    print("=" * 60)

    rm = pyvisa.ResourceManager('@py')
    resources = rm.list_resources()

    sdg = None
    for r in resources:
        if 'USB' in r:
            sdg = r
            break

    if not sdg:
        print("No USB instrument found!")
        return 1

    print(f"Connecting to: {sdg}")
    device = rm.open_resource(sdg)
    device.timeout = 30000  # 30 seconds for waveform upload
    device.read_termination = '\n'
    device.write_termination = '\n'

    idn = device.query("*IDN?").strip()
    print(f"Device: {idn}")

    # Create a simple test waveform: 2 points (simplest case from Siglent example)
    print("\n--- Test 1: Minimal 2-point waveform ---")

    # Siglent example uses LOW and HIGH values
    # Format appears to be ASCII integers representing DAC values
    # DAC range: -32768 to 32767 (16-bit signed)

    # Two point square wave: low, high
    data_2pt = "-32768,32767"

    cmd = f"C1:WVDT WVNM,wave1,FREQ,2000.0,AMPL,3.0,OFST,0.0,PHASE,0.0,WAVEDATA,{data_2pt}"
    print(f"Command: {cmd}")

    response = input("Send 2-point waveform? (y/N): ").strip().lower()
    if response == 'y':
        try:
            device.write(cmd)
            time.sleep(2)

            # Select it
            device.write("C1:BSWV WVTP,ARB")
            time.sleep(0.3)
            device.write("C1:ARWV NAME,wave1")
            time.sleep(0.5)

            result = device.query("C1:ARWV?").strip()
            print(f"C1:ARWV?: {result}")

            if "wave1" in result.lower():
                print(">>> SUCCESS! <<<")

                # Turn on output
                device.write("C1:OUTP ON")
                print("Output ON - check oscilloscope!")

        except Exception as e:
            print(f"Error: {e}")

    # Test 2: Sine wave with more points
    print("\n--- Test 2: 100-point sine wave ---")

    n_points = 100
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    waveform = np.sin(t)

    # Convert to DAC values (-32768 to 32767)
    dac_values = (waveform * 32767).astype(int)

    # Create comma-separated ASCII string
    data_str = ",".join(str(v) for v in dac_values)

    cmd2 = f"C1:WVDT WVNM,sine100,FREQ,1000.0,AMPL,2.0,OFST,0.0,PHASE,0.0,WAVEDATA,{data_str}"
    print(f"Command length: {len(cmd2)} chars")
    print(f"First 100 chars: {cmd2[:100]}...")

    response = input("Send 100-point sine? (y/N): ").strip().lower()
    if response == 'y':
        try:
            device.write(cmd2)
            time.sleep(2)

            device.write("C1:BSWV WVTP,ARB")
            time.sleep(0.3)
            device.write("C1:ARWV NAME,sine100")
            time.sleep(0.5)

            result = device.query("C1:ARWV?").strip()
            print(f"C1:ARWV?: {result}")

            if "sine100" in result.lower():
                print(">>> SUCCESS! <<<")
                device.write("C1:OUTP ON")
                print("Output ON!")

        except Exception as e:
            print(f"Error: {e}")

    # Test 3: Gaussian burst (what we actually want)
    print("\n--- Test 3: Gaussian burst (1000 points) ---")

    n_points = 1000
    t = np.linspace(0, 1, n_points, endpoint=False)

    # 5-cycle carrier with Gaussian envelope
    carrier = np.sin(2 * np.pi * 5 * t)
    envelope = np.exp(-0.5 * ((t - 0.5) / 0.15)**2)
    waveform = carrier * envelope
    waveform = waveform / np.max(np.abs(waveform))  # Normalize

    dac_values = (waveform * 32767).astype(int)
    data_str = ",".join(str(v) for v in dac_values)

    cmd3 = f"C1:WVDT WVNM,burst,FREQ,40000.0,AMPL,2.0,OFST,0.0,PHASE,0.0,WAVEDATA,{data_str}"
    print(f"Command length: {len(cmd3)} chars")

    response = input("Send Gaussian burst? (y/N): ").strip().lower()
    if response == 'y':
        try:
            device.write(cmd3)
            time.sleep(3)

            device.write("C1:BSWV WVTP,ARB")
            time.sleep(0.3)
            device.write("C1:ARWV NAME,burst")
            time.sleep(0.5)

            result = device.query("C1:ARWV?").strip()
            print(f"C1:ARWV?: {result}")

            bswv = device.query("C1:BSWV?").strip()
            print(f"C1:BSWV?: {bswv}")

            if "burst" in result.lower() or "ARB" in bswv.upper():
                print(">>> SUCCESS! <<<")
                device.write("C1:OUTP ON")
                print("Output ON!")
            else:
                print("Waveform may not have loaded correctly.")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    device.close()
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
