#!/usr/bin/env python3
"""
Test different WVCSV command variations to find what works.

Hypothesis: Some syntax variation might work where others crash.
"""

import pyvisa
import time
import sys

def test_variant(device, variant_name, cmd, wait_after=3):
    """Test a command variant and check if device survives."""
    print(f"\n--- {variant_name} ---")
    print(f"Command: {cmd[:80]}{'...' if len(cmd) > 80 else ''}")
    print(f"Length: {len(cmd)} bytes")

    input("Press Enter to send (watch screen)...")

    try:
        bytes_written = device.write(cmd)
        print(f"Bytes written: {bytes_written}")
        print(f"Waiting {wait_after} seconds...")
        time.sleep(wait_after)

        # Check if device responds
        device.read_termination = '\n'
        opc = device.query("*OPC?").strip()
        print(f"*OPC?: {opc}")
        return True

    except Exception as e:
        print(f">>> CRASHED: {e} <<<")
        return False


def main():
    print("=" * 60)
    print("WaveStation WVCSV Variant Test")
    print("=" * 60)
    print("Testing different command syntaxes to find what works")
    print("=" * 60)

    rm = pyvisa.ResourceManager()
    resources = rm.list_resources()
    ws_resource = [r for r in resources if '05FF' in r or 'LCRY' in r]

    if not ws_resource:
        print("WaveStation not found!")
        return 1

    device = rm.open_resource(ws_resource[0])
    device.timeout = 30000

    # Verify connection
    idn = device.query("*IDN?").strip()
    print(f"Connected: {idn}")

    # No write terminator per manual
    device.write_termination = ''
    device.read_termination = '\n'

    # Define variants to test
    # Values: 8 simple numbers
    data8 = "0,0.5,1,0.5,0,-0.5,-1,-0.5"

    variants = [
        # Variant 1: Short form WVCSV
        ("Short form (WVCSV)", f"WVCSV M50,WAVENM,TEST1,CSVLENG,8,CSVDATA,Amplitude,{data8}"),

        # Variant 2: Long form WAVECSV
        ("Long form (WAVECSV)", f"WAVECSV M50,WAVENM,TEST2,CSVLENG,8,CSVDATA,Amplitude,{data8}"),

        # Variant 3: Lowercase
        ("Lowercase", f"wvcsv M50,WAVENM,TEST3,CSVLENG,8,CSVDATA,Amplitude,{data8}"),

        # Variant 4: With trailing comma (as in manual example)
        ("With trailing comma", f"WVCSV M50,WAVENM,TEST4,CSVLENG,8,CSVDATA,Amplitude,{data8},"),

        # Variant 5: Integer values only (no decimals)
        ("Integer values", "WVCSV M50,WAVENM,TEST5,CSVLENG,8,CSVDATA,Amplitude,0,1,2,1,0,-1,-2,-1"),

        # Variant 6: Very simple - all zeros
        ("All zeros", "WVCSV M50,WAVENM,TEST6,CSVLENG,8,CSVDATA,Amplitude,0,0,0,0,0,0,0,0"),

        # Variant 7: Different memory slot (M51)
        ("Different slot M51", f"WVCSV M51,WAVENM,TEST7,CSVLENG,8,CSVDATA,Amplitude,{data8}"),

        # Variant 8: Minimum points = 32 (per manual)
        ("32 points", "WVCSV M50,WAVENM,TEST8,CSVLENG,32,CSVDATA,Amplitude," + ",".join(["0"]*32)),
    ]

    print(f"\n{len(variants)} variants to test.")
    print("Device will be power cycled between crashes.")

    for i, (name, cmd) in enumerate(variants):
        print(f"\n{'='*60}")
        print(f"VARIANT {i+1}/{len(variants)}: {name}")
        print("="*60)

        # Check if device is still connected
        try:
            device.query("*IDN?")
        except:
            print("Device not responding - needs power cycle")
            print("Power cycle device and press Enter to continue...")
            input()
            # Reconnect
            device.close()
            rm.close()
            rm = pyvisa.ResourceManager()
            ws_resource = [r for r in rm.list_resources() if '05FF' in r or 'LCRY' in r]
            if not ws_resource:
                print("Can't reconnect!")
                return 1
            device = rm.open_resource(ws_resource[0])
            device.timeout = 30000
            device.write_termination = ''
            device.read_termination = '\n'

        success = test_variant(device, name, cmd)

        if success:
            print(f"\n>>> VARIANT '{name}' WORKED! <<<")
            # Verify it was stored
            device.write("C1:BSWV WVTP,ARB")
            time.sleep(0.3)
            device.write("C1:ARWV INDEX,50")
            time.sleep(0.3)
            result = device.query("C1:ARWV?").strip()
            print(f"ARWV: {result}")
            break
        else:
            cont = input("\nContinue testing? (y/N): ").strip().lower()
            if cont != 'y':
                break

    device.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
