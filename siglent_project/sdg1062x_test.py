#!/usr/bin/env python3
"""
Siglent SDG1062X test using pyvisa-py (pure Python backend).
"""

import pyvisa
import numpy as np
import time
import sys


def main():
    print("=" * 60)
    print("Siglent SDG1062X: PyVISA-py Test")
    print("=" * 60)

    # Use pyvisa-py backend (pure Python, no NI-VISA)
    try:
        rm = pyvisa.ResourceManager('@py')
        print(f"Backend: {rm.visalib}")
    except Exception as e:
        print(f"Failed to use pyvisa-py backend: {e}")
        return 1

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

    if not resources:
        print("\nNo instruments found!")
        print("Make sure SDG1062X is connected via USB and powered on.")
        return 1

    # Try to find SDG or let user pick
    sdg_resource = None
    for r in resources:
        # Siglent USB vendor ID is typically F4ED
        if 'USB' in r and ('F4ED' in r or 'SIGLENT' in r.upper() or 'SDG' in r.upper()):
            sdg_resource = r
            break

    if not sdg_resource:
        print("\nSDG not auto-detected. Using first USB resource...")
        usb_resources = [r for r in resources if 'USB' in r]
        if usb_resources:
            sdg_resource = usb_resources[0]
        else:
            print("No USB resources found!")
            return 1

    print(f"\nConnecting to: {sdg_resource}")

    try:
        device = rm.open_resource(sdg_resource)
        device.timeout = 10000
        device.read_termination = '\n'
        device.write_termination = '\n'

        # Test basic communication
        idn = device.query("*IDN?").strip()
        print(f"Device ID: {idn}")

        if 'SDG' not in idn.upper() and 'SIGLENT' not in idn.upper():
            print("Warning: This doesn't look like a Siglent SDG!")

        # Check current state
        print("\n--- Current Device State ---")

        try:
            ch1_out = device.query("C1:OUTP?").strip()
            print(f"CH1 Output: {ch1_out}")
        except Exception as e:
            print(f"CH1 Output query failed: {e}")

        try:
            ch1_wave = device.query("C1:BSWV?").strip()
            print(f"CH1 Basic Wave: {ch1_wave}")
        except Exception as e:
            print(f"CH1 Basic Wave query failed: {e}")

        # Simple test: set a 40kHz sine wave
        print("\n--- Test: Set 40kHz Sine Wave ---")
        response = input("Set CH1 to 40kHz sine, 1Vpp? (y/N): ").strip().lower()

        if response == 'y':
            device.write("C1:BSWV WVTP,SINE,FRQ,40000,AMP,1")
            time.sleep(0.5)
            result = device.query("C1:BSWV?").strip()
            print(f"Result: {result}")

            # Turn on output
            device.write("C1:OUTP ON")
            time.sleep(0.3)
            out_state = device.query("C1:OUTP?").strip()
            print(f"Output: {out_state}")

        # Test arbitrary waveform capability
        print("\n--- Test: Arbitrary Waveform Upload ---")
        response = input("Test arbitrary waveform upload? (y/N): ").strip().lower()

        if response == 'y':
            # Create test waveform: Gaussian-windowed sine burst
            n_points = 4096  # SDG supports up to 16384 points
            t = np.linspace(0, 1, n_points, endpoint=False)

            # 5-cycle 40kHz burst with Gaussian envelope
            carrier = np.sin(2 * np.pi * 5 * t)
            envelope = np.exp(-0.5 * ((t - 0.5) / 0.15)**2)
            waveform = carrier * envelope

            # Normalize to -1 to +1
            waveform = waveform / np.max(np.abs(waveform))

            # Convert to 16-bit signed integers (SDG format)
            waveform_int16 = (waveform * 32767).astype('<i2')

            print(f"Waveform: {n_points} points")
            print(f"Min/Max: {waveform_int16.min()}, {waveform_int16.max()}")

            # SDG uses C1:WVDT command for binary waveform data
            # Format: C1:WVDT WVNM,<name>,FREQ,<freq>,AMPL,<amp>,OFST,<offset>,PHASE,<phase>,WAVEDATA,<binary>

            wave_bytes = waveform_int16.tobytes()
            byte_count = len(wave_bytes)
            byte_count_str = str(byte_count)
            n_digits = len(byte_count_str)

            # Build command
            cmd_header = f"C1:WVDT WVNM,test_burst,FREQ,1000,AMPL,2,OFST,0,PHASE,0,WAVEDATA,#{n_digits}{byte_count_str}"

            print(f"Command header: {cmd_header}")
            print(f"Binary data: {byte_count} bytes")

            try:
                # Send as raw bytes
                full_cmd = cmd_header.encode('ascii') + wave_bytes
                device.write_raw(full_cmd)
                time.sleep(2)

                opc = device.query("*OPC?").strip()
                print(f"*OPC?: {opc}")

                # Select the uploaded waveform
                device.write("C1:BSWV WVTP,ARB")
                time.sleep(0.3)
                device.write("C1:ARWV NAME,test_burst")
                time.sleep(0.5)

                # Check result
                arwv = device.query("C1:ARWV?").strip()
                print(f"C1:ARWV?: {arwv}")

                bswv = device.query("C1:BSWV?").strip()
                print(f"C1:BSWV?: {bswv}")

                if "ARB" in bswv.upper():
                    print("\n>>> Waveform type set to ARB! <<<")

            except Exception as e:
                print(f"Error during waveform upload: {e}")
                import traceback
                traceback.print_exc()

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
