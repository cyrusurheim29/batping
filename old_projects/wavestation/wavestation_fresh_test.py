#!/usr/bin/env python3
"""
Fresh WaveStation WVCSV test - following manual exactly.

From SCPI manual:
- SYNTAX: WAVECSV <address>,WAVENM,<wavename>,CSVLENG,<csvlength>,CSVDATA,<csvdata>
- <address> for WS2000: M50 to M59
- <csvdata>: List of Amplitude values (Volts), using Amplitude as header
- Example: WAVECSV M36,WAVENM,WAVE1,CSVLENG,137166,CSVDATA,Amplitude,0,0.0012,0.0023,0.0035,

Key observations:
1. Manual example has trailing comma after last value
2. Uses long form WAVECSV (though WVCSV should work)
3. Amplitude is part of CSVDATA, not a separate parameter
"""

import pyvisa
import time
import sys

def main():
    print("=" * 60)
    print("WaveStation Fresh WVCSV Test")
    print("Following manual example exactly")
    print("=" * 60)

    # Try both backends
    backends = ['@py', '']  # pyvisa-py and NI-VISA (if available)

    rm = None
    for backend in backends:
        try:
            rm = pyvisa.ResourceManager(backend) if backend else pyvisa.ResourceManager()
            print(f"\nUsing backend: {rm.visalib}")
            break
        except Exception as e:
            print(f"Backend {backend or 'NI-VISA'} not available: {e}")
            continue

    if rm is None:
        print("No VISA backend available!")
        return 1

    # Find WaveStation
    resources = rm.list_resources()
    print(f"Available resources: {resources}")

    ws_resource = None
    for r in resources:
        if 'LCRY' in r or '05FF' in r or '1535' in r:
            ws_resource = r
            break

    if not ws_resource:
        print("WaveStation not found!")
        return 1

    print(f"Connecting to: {ws_resource}")
    device = rm.open_resource(ws_resource)
    device.timeout = 30000  # 30 second timeout

    # Clear any pending state
    device.write("*CLS")
    time.sleep(0.5)

    idn = device.query("*IDN?").strip()
    print(f"Connected: {idn}")

    # Check firmware version
    print(f"\nFirmware info from IDN: {idn}")

    # Test 1: Very minimal - just 8 values, following manual format exactly
    print("\n" + "=" * 60)
    print("TEST 1: Minimal 8-point waveform (manual format)")
    print("=" * 60)

    # Manual example format: CSVDATA,Amplitude,val1,val2,...
    # Using exactly 8 simple values within voltage limits
    values = "0,0.1,0.2,0.3,0.2,0.1,0,-0.1"

    # Try LONG form first (WAVECSV) as shown in manual example
    cmd = f"WAVECSV M50,WAVENM,TEST8,CSVLENG,8,CSVDATA,Amplitude,{values}"
    print(f"Command (long form): {cmd}")
    print(f"Length: {len(cmd)} bytes")

    print("\nSending command...")
    try:
        device.write(cmd)
        print("Write completed without error")
    except Exception as e:
        print(f"Write error: {e}")
        # Try short form
        print("\nTrying short form WVCSV...")
        cmd = f"WVCSV M50,WAVENM,TEST8,CSVLENG,8,CSVDATA,Amplitude,{values}"
        try:
            device.write(cmd)
            print("Short form write completed")
        except Exception as e2:
            print(f"Short form also failed: {e2}")

    print("Waiting 3 seconds...")
    time.sleep(3)

    # Check if device is still responding
    print("\nChecking device status...")
    try:
        opc = device.query("*OPC?").strip()
        print(f"*OPC?: {opc}")
    except Exception as e:
        print(f"Device not responding after WVCSV: {e}")
        print(">>> DEVICE MAY HAVE CRASHED <<<")
        return 1

    # Test 2: Try with trailing comma (as in manual example)
    print("\n" + "=" * 60)
    print("TEST 2: With trailing comma (as in manual example)")
    print("=" * 60)

    # Manual example ends with trailing comma
    cmd_trailing = f"WAVECSV M51,WAVENM,TEST8B,CSVLENG,8,CSVDATA,Amplitude,{values},"
    print(f"Command: {cmd_trailing}")

    try:
        device.write(cmd_trailing)
        print("Write completed")
        time.sleep(2)
        opc = device.query("*OPC?").strip()
        print(f"*OPC?: {opc}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 3: Check if waveform was stored
    print("\n" + "=" * 60)
    print("TEST 3: Verify waveform storage")
    print("=" * 60)

    # Set to ARB mode
    device.write("C1:BSWV WVTP,ARB")
    time.sleep(0.5)

    # Try selecting by NAME
    print("Trying C1:ARWV NAME,TEST8...")
    device.write("C1:ARWV NAME,TEST8")
    time.sleep(0.5)

    try:
        result = device.query("C1:ARWV?").strip()
        print(f"Result: {result}")
        if "TEST8" in result.upper():
            print(">>> SUCCESS! Waveform found! <<<")
        else:
            print("Waveform not found by name")
    except Exception as e:
        print(f"Query error: {e}")

    # Try INDEX 50
    print("\nTrying C1:ARWV INDEX,50...")
    device.write("C1:ARWV INDEX,50")
    time.sleep(0.5)

    try:
        result = device.query("C1:ARWV?").strip()
        print(f"Result: {result}")
        if "stairup" not in result.lower():
            print(">>> Different waveform at INDEX 50 <<<")
    except Exception as e:
        print(f"Query error: {e}")

    # Final summary
    print("\n" + "=" * 60)
    print("Test complete.")
    print("Please check device screen: Arb -> Load Wform -> Stored Wforms")
    print("=" * 60)

    device.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
