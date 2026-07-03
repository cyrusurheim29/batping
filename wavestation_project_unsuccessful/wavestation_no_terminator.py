#!/usr/bin/env python3
"""
WaveStation WVCSV test with NO terminator.

CRITICAL FINDING from SCPI manual page 7:
"Do not use a terminator of any kind, including NULL terminators."

Default pyvisa adds '\r\n' terminator which may be causing the crash!
"""

import pyvisa
import time
import sys

def main():
    print("=" * 60)
    print("WaveStation WVCSV Test - NO TERMINATOR")
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

    # CRITICAL: Check default terminators
    print(f"\nDefault write_termination: {repr(device.write_termination)}")
    print(f"Default read_termination: {repr(device.read_termination)}")

    # First test: normal communication with default terminator
    print("\n--- Test with DEFAULT terminator (\\r\\n) ---")
    idn = device.query("*IDN?").strip()
    print(f"*IDN?: {idn}")

    # Now test with NO terminator
    print("\n--- Setting NO write terminator ---")
    device.write_termination = ''  # Empty string = no terminator
    print(f"New write_termination: {repr(device.write_termination)}")

    # Test basic command still works
    print("\nTesting *OPC? with no terminator...")
    try:
        # For query, we still need read termination
        device.read_termination = '\n'
        result = device.query("*OPC?").strip()
        print(f"*OPC?: {result}")
    except Exception as e:
        print(f"Query failed: {e}")
        print("Restoring default terminators...")
        device.write_termination = '\r\n'

    # Now the WVCSV test
    print("\n" + "=" * 60)
    print("WVCSV TEST with no terminator")
    print("=" * 60)

    # Make sure we have no write terminator
    device.write_termination = ''

    # Minimal 8-point waveform
    data = "0,0.5,1,0.5,0,-0.5,-1,-0.5"
    cmd = f"WAVECSV M50,WAVENM,NOTERM,CSVLENG,8,CSVDATA,Amplitude,{data}"

    print(f"Command: {cmd}")
    print(f"Length: {len(cmd)} bytes")
    print(f"write_termination: {repr(device.write_termination)}")

    input("\nPress Enter to send WVCSV (watch device screen)...")

    try:
        bytes_written = device.write(cmd)
        print(f"Bytes written: {bytes_written}")

        print("Waiting 3 seconds...")
        time.sleep(3)

        # Check if device responds
        device.read_termination = '\n'
        opc = device.query("*OPC?").strip()
        print(f"*OPC?: {opc}")
        print(">>> Device still responding! <<<")

        # Verify waveform
        device.write("C1:BSWV WVTP,ARB")
        time.sleep(0.3)
        device.write("C1:ARWV NAME,NOTERM")
        time.sleep(0.5)
        result = device.query("C1:ARWV?").strip()
        print(f"ARWV: {result}")

        if "NOTERM" in result.upper():
            print("\n>>> SUCCESS! Waveform uploaded! <<<")

    except Exception as e:
        print(f">>> ERROR: {e} <<<")
        print("Device may have crashed")

    device.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
