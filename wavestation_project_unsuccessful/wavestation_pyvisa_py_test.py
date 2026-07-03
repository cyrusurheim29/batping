#!/usr/bin/env python3
"""
WaveStation test using pyvisa-py (pure Python backend).
No NI-VISA required - should work better on Apple Silicon.
"""

import pyvisa
import numpy as np
import time
import sys


def main():
    print("=" * 60)
    print("WaveStation: PyVISA-py Pure Python Backend Test")
    print("=" * 60)

    # Force use of pyvisa-py backend (pure Python, no NI-VISA)
    try:
        rm = pyvisa.ResourceManager('@py')
        print(f"Backend: {rm.visalib}")
    except Exception as e:
        print(f"Failed to use pyvisa-py backend: {e}")
        print("Falling back to default...")
        rm = pyvisa.ResourceManager()
        print(f"Backend: {rm.visalib}")

    # List resources
    print("\nSearching for instruments...")
    try:
        resources = rm.list_resources()
        print(f"Found {len(resources)} resource(s):")
        for r in resources:
            print(f"  {r}")
    except Exception as e:
        print(f"Error listing resources: {e}")
        return 1

    # Find WaveStation (Siglent/LeCroy)
    ws_resource = None
    for r in resources:
        if 'USB' in r and ('05FF' in r or 'LCRY' in r or 'F4ED' in r or 'SIGLENT' in r.upper()):
            ws_resource = r
            break

    if not ws_resource:
        print("\nWaveStation not found!")
        print("Make sure it's connected via USB and powered on.")
        return 1

    print(f"\nConnecting to: {ws_resource}")

    try:
        device = rm.open_resource(ws_resource)
        device.timeout = 10000  # 10 seconds
        device.read_termination = '\n'
        device.write_termination = '\n'

        # Test basic communication
        idn = device.query("*IDN?").strip()
        print(f"Device ID: {idn}")

        # Check current state
        print("\n--- Current Device State ---")

        try:
            ch1_state = device.query("C1:OUTP?").strip()
            print(f"CH1 Output: {ch1_state}")
        except:
            print("CH1 Output: (query failed)")

        try:
            ch1_wave = device.query("C1:BSWV?").strip()
            print(f"CH1 Waveform: {ch1_wave}")
        except:
            print("CH1 Waveform: (query failed)")

        try:
            arwv = device.query("C1:ARWV?").strip()
            print(f"CH1 Arb Waveform: {arwv}")
        except:
            print("CH1 Arb Waveform: (query failed)")

        # Test arbitrary waveform upload with WVDT command
        # (different from WVCSV - WVDT uses binary data)
        print("\n--- Testing WVDT (binary waveform upload) ---")

        # Create a simple test waveform: 1 cycle sine, 1000 points
        n_points = 1000
        t = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        waveform = np.sin(t)  # -1 to +1

        # Convert to 16-bit signed integers (-32768 to 32767)
        # Siglent uses DAC range of -32768 to 32767
        waveform_int16 = (waveform * 32767).astype('<i2')  # little-endian int16

        print(f"Waveform: {n_points} points, dtype={waveform_int16.dtype}")
        print(f"First 10 values: {waveform_int16[:10]}")

        # WVDT command format: WVDT WVNM,<name>,WAVEDATA,<binary_block>
        # Binary block format: #<n><length><data>
        # where n = number of digits in length, length = byte count

        wave_bytes = waveform_int16.tobytes()
        byte_count = len(wave_bytes)
        byte_count_str = str(byte_count)
        n_digits = len(byte_count_str)

        header = f"C1:WVDT WVNM,test_sine,WAVEDATA,#{n_digits}{byte_count_str}"
        print(f"Header: {header}")
        print(f"Data: {byte_count} bytes")

        response = input("\nSend WVDT command? (y/N): ").strip().lower()
        if response == 'y':
            try:
                # Send header + binary data
                full_command = header.encode('ascii') + wave_bytes
                device.write_raw(full_command)
                time.sleep(1)

                # Check if it worked
                opc = device.query("*OPC?").strip()
                print(f"*OPC?: {opc}")

                # Try to select and use the waveform
                device.write("C1:BSWV WVTP,ARB")
                time.sleep(0.3)
                device.write("C1:ARWV NAME,test_sine")
                time.sleep(0.5)

                arwv = device.query("C1:ARWV?").strip()
                print(f"C1:ARWV?: {arwv}")

                if "test_sine" in arwv.lower():
                    print(">>> SUCCESS! Waveform uploaded and selected! <<<")
                else:
                    print("Waveform may not have been selected correctly.")

            except Exception as e:
                print(f"Error during WVDT: {e}")

        # Also try the awg-scpi library approach if available
        print("\n--- Checking for awg-scpi library ---")
        try:
            import awg_scpi
            print("awg-scpi is available!")
            print("Consider using: pip install awg-scpi")
            print("It handles Siglent's SCPI dialect automatically.")
        except ImportError:
            print("awg-scpi not installed. Optional: pip install awg-scpi")

        device.close()
        print("\nDone!")
        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
