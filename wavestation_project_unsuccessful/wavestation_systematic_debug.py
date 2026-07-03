#!/usr/bin/env python3
"""
Systematic debugging of WaveStation WAVECSV command
Testing hypotheses one at a time
"""

import pyvisa
import time

def check_errors(device):
    """Check all error/status registers."""
    results = {}
    for cmd in ["*ESR?", "*STB?", "*OPC?"]:
        try:
            results[cmd] = device.query(cmd).strip()
        except Exception as e:
            results[cmd] = f"ERROR: {e}"
    return results

def test_command(device, cmd, description, wait_after=1.0):
    """Send command and check for errors."""
    print(f"\n--- {description} ---")
    print(f"Command: {cmd[:80]}{'...' if len(cmd) > 80 else ''}")
    print(f"Length: {len(cmd)} bytes")

    # Check errors before
    errors_before = check_errors(device)
    print(f"ESR before: {errors_before.get('*ESR?', 'N/A')}")

    # Send command
    try:
        device.write(cmd)
        print("Write: OK")
    except Exception as e:
        print(f"Write: FAILED - {e}")
        return False

    time.sleep(wait_after)

    # Check errors after
    try:
        errors_after = check_errors(device)
        print(f"ESR after: {errors_after.get('*ESR?', 'N/A')}")
        print(f"STB after: {errors_after.get('*STB?', 'N/A')}")
        print(f"OPC after: {errors_after.get('*OPC?', 'N/A')}")
    except Exception as e:
        print(f"Error check failed: {e}")
        return False

    return True


def main():
    print("=" * 60)
    print("WaveStation WAVECSV Systematic Debug")
    print("=" * 60)

    rm = pyvisa.ResourceManager()

    # List resources
    resources = rm.list_resources()
    print(f"\nAvailable resources: {resources}")

    ws_resource = None
    for r in resources:
        if 'LCRY' in r or '05FF' in r:
            ws_resource = r
            break

    if not ws_resource:
        print("WaveStation not found!")
        return

    print(f"Using: {ws_resource}")

    device = rm.open_resource(ws_resource)
    device.timeout = 10000

    # Clear status
    device.write("*CLS")
    time.sleep(0.5)

    idn = device.query("*IDN?").strip()
    print(f"Connected: {idn}")

    # TEST 1: Baseline - check error registers are clear
    print("\n" + "=" * 60)
    print("TEST 1: Baseline error status")
    print("=" * 60)
    errors = check_errors(device)
    for k, v in errors.items():
        print(f"  {k}: {v}")

    # TEST 2: Minimal WVCSV (short form) with just 8 points
    print("\n" + "=" * 60)
    print("TEST 2: Minimal WVCSV (8 points)")
    print("=" * 60)
    data_8 = "0,0.5,1,0.5,0,-0.5,-1,-0.5"
    cmd_8 = f"WVCSV M50,WAVENM,TEST8,CSVLENG,8,CSVDATA,Amplitude,{data_8}"
    if not test_command(device, cmd_8, "WVCSV 8 points", wait_after=2.0):
        print("TEST 2 FAILED - stopping")
        device.close()
        return

    # TEST 3: Check if waveform exists by trying to select it
    print("\n" + "=" * 60)
    print("TEST 3: Try to select uploaded waveform")
    print("=" * 60)

    # First set ARB mode
    test_command(device, "C1:BSWV WVTP,ARB", "Set ARB mode", wait_after=0.5)

    # Try selecting by name
    print("\nTrying ARWV NAME,TEST8...")
    try:
        device.write("C1:ARWV NAME,TEST8")
        time.sleep(1)
        result = device.query("C1:ARWV?")
        print(f"ARWV result: {result.strip()}")
        if "TEST8" in result.upper():
            print("*** SUCCESS - waveform found! ***")
        elif "stairup" in result.lower():
            print("Still showing stairup - waveform NOT uploaded")
        else:
            print(f"Different result: {result}")
    except Exception as e:
        print(f"ARWV NAME failed: {e}")

    # TEST 4: Try INDEX 50 directly
    print("\n" + "=" * 60)
    print("TEST 4: Try INDEX 50")
    print("=" * 60)
    try:
        device.write("C1:ARWV INDEX,50")
        time.sleep(1)
        result = device.query("C1:ARWV?")
        print(f"ARWV INDEX,50 result: {result.strip()}")
    except Exception as e:
        print(f"INDEX 50 failed: {e}")

    # TEST 5: Query STL? to see all stored waveforms
    print("\n" + "=" * 60)
    print("TEST 5: Query stored waveform list (STL?)")
    print("=" * 60)
    device.timeout = 30000  # Longer timeout for STL
    try:
        # Just write STL? and try to read
        device.write("STL?")
        time.sleep(2)
        response = device.read_raw()
        # Look for M50 in response
        response_str = response.decode('ascii', errors='ignore')
        print(f"STL response length: {len(response_str)} chars")

        # Find M50 section
        if "M50" in response_str:
            idx = response_str.find("M50")
            print(f"M50 section: {response_str[idx:idx+50]}")
        else:
            print("M50 not found in response")
            print(f"First 200 chars: {response_str[:200]}")
    except Exception as e:
        print(f"STL? failed: {e}")

    device.close()
    rm.close()

    print("\n" + "=" * 60)
    print("Debug complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
