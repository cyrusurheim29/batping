#!/usr/bin/env python3
"""
Debug what bytes are actually being sent over USB.
"""

import pyvisa
import time

def main():
    rm = pyvisa.ResourceManager()
    resources = rm.list_resources()
    ws_resource = [r for r in resources if '05FF' in r or 'LCRY' in r][0]

    device = rm.open_resource(ws_resource)
    device.timeout = 10000

    print("=" * 60)
    print("Wire-level debug")
    print("=" * 60)

    # Check what device is seeing
    print(f"write_termination: {repr(device.write_termination)}")
    print(f"read_termination: {repr(device.read_termination)}")
    print(f"encoding: {device.encoding}")

    # When we do device.write("WVCSV..."), it sends:
    # "WVCSV..." + write_termination

    # For USBTMC, the termination might be handled differently
    # Let's see if we can access raw write

    print("\nTesting raw write access...")

    # Simple test command
    cmd = "WVCSV M50,WAVENM,T,CSVLENG,8,CSVDATA,Amplitude,0,1,0,-1,0,1,0,-1"

    # What bytes will actually be sent?
    encoded = cmd.encode(device.encoding)
    term = (device.write_termination or '').encode(device.encoding)
    total = encoded + term

    print(f"\nCommand: {cmd}")
    print(f"Encoded bytes: {len(encoded)}")
    print(f"Terminator bytes: {repr(term)} ({len(term)} bytes)")
    print(f"Total bytes to send: {len(total)}")
    print(f"Last 10 bytes: {total[-10:]}")

    # For USBTMC, there's also an EOM (end-of-message) indicator
    # This is separate from the character termination

    print("\n" + "=" * 60)
    print("USBTMC uses a different mechanism than terminators")
    print("The \\r\\n may actually be fine, or may confuse the parser")
    print("=" * 60)

    # Let's try sending the command with write_raw if available
    print("\nAttempting test...")

    # First verify device is working
    idn = device.query("*IDN?").strip()
    print(f"Device: {idn}")

    # Try setting write_termination to None or empty
    print("\nSetting write_termination = ''")
    device.write_termination = ''

    # Now write a simple test
    print("Writing *CLS...")
    device.write("*CLS")
    time.sleep(0.3)

    print("Querying *OPC?...")
    device.read_termination = '\n'
    opc = device.query("*OPC?").strip()
    print(f"*OPC?: {opc}")

    if opc == "1":
        print("Communication works with empty terminator!")

    device.close()


if __name__ == "__main__":
    main()
