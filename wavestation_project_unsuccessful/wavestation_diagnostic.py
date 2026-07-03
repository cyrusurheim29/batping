#!/usr/bin/env python3
"""
WaveStation full diagnostic - check device state before WVCSV test.

This script:
1. Queries all device information
2. Checks stored waveform list
3. Reports error status
4. Tests basic commands
5. Then optionally tests WVCSV

The goal is to identify if something is wrong with device state
before WVCSV crashes it.
"""

import pyvisa
import time
import sys

def query_safe(device, cmd, timeout_ms=5000):
    """Query with timeout handling."""
    old_timeout = device.timeout
    device.timeout = timeout_ms
    try:
        result = device.query(cmd).strip()
        return result
    except Exception as e:
        return f"ERROR: {e}"
    finally:
        device.timeout = old_timeout


def main():
    print("=" * 60)
    print("WaveStation Full Diagnostic")
    print("=" * 60)

    # Try NI-VISA first
    rm = pyvisa.ResourceManager()
    print(f"VISA Library: {rm.visalib}")

    resources = rm.list_resources()
    print(f"Resources: {resources}")

    ws_resource = None
    for r in resources:
        if '05FF' in r or 'LCRY' in r:
            ws_resource = r
            break

    if not ws_resource:
        print("WaveStation not found!")
        return 1

    print(f"\nConnecting to: {ws_resource}")
    device = rm.open_resource(ws_resource)
    device.timeout = 10000

    # === Section 1: Device Information ===
    print("\n" + "=" * 60)
    print("SECTION 1: Device Information")
    print("=" * 60)

    idn = query_safe(device, "*IDN?")
    print(f"*IDN?: {idn}")

    # Parse firmware version from IDN
    # Format: *IDN WST,WaveStation 2052,LCRY0000C11111,1.01.02.28,02-00-00-21-24
    if "WaveStation" in idn:
        parts = idn.split(",")
        if len(parts) >= 4:
            print(f"  Model: {parts[1]}")
            print(f"  Serial: {parts[2]}")
            print(f"  SW Version: {parts[3]}")
            if len(parts) >= 5:
                print(f"  FW Version: {parts[4]}")

    # === Section 2: Status Registers ===
    print("\n" + "=" * 60)
    print("SECTION 2: Status Registers")
    print("=" * 60)

    # Clear first
    device.write("*CLS")
    time.sleep(0.3)

    status_cmds = [
        ("*ESR?", "Event Status Register (0=no errors)"),
        ("*STB?", "Status Byte"),
        ("*OPC?", "Operation Complete"),
    ]

    for cmd, desc in status_cmds:
        result = query_safe(device, cmd)
        print(f"{cmd}: {result}  ({desc})")

    # === Section 3: Channel Status ===
    print("\n" + "=" * 60)
    print("SECTION 3: Channel Status")
    print("=" * 60)

    for ch in [1, 2]:
        print(f"\nChannel {ch}:")
        ch_cmds = [
            f"C{ch}:OUTP?",
            f"C{ch}:BSWV?",
            f"C{ch}:ARWV?",
        ]
        for cmd in ch_cmds:
            result = query_safe(device, cmd)
            print(f"  {cmd}: {result}")

    # === Section 4: Stored Waveforms ===
    print("\n" + "=" * 60)
    print("SECTION 4: Stored Waveform List (STL?)")
    print("=" * 60)
    print("(This may take a moment...)")

    # STL? can return a lot of data
    stl_result = query_safe(device, "STL?", timeout_ms=30000)
    if "ERROR" in stl_result:
        print(f"STL? failed: {stl_result}")
    else:
        print(f"STL? response length: {len(stl_result)} chars")
        # Look for user memory section M50-M59
        if "M50" in stl_result:
            # Find M50 section
            idx = stl_result.find("M50")
            section = stl_result[idx:idx+200]
            print(f"User memory section (M50+):\n{section}")
        else:
            print("M50 not found in STL response")
            print(f"First 300 chars: {stl_result[:300]}")

    # === Section 5: Built-in ARB Test ===
    print("\n" + "=" * 60)
    print("SECTION 5: Built-in ARB Waveform Test")
    print("=" * 60)

    # Set to ARB mode
    device.write("C1:BSWV WVTP,ARB")
    time.sleep(0.3)

    # Test selecting some built-in waveforms
    test_indices = [
        (2, "StairUp"),
        (19, "Gussian"),
        (28, "Chirp"),
        (50, "User slot M50"),
    ]

    for idx, name in test_indices:
        device.write(f"C1:ARWV INDEX,{idx}")
        time.sleep(0.3)
        result = query_safe(device, "C1:ARWV?")
        print(f"INDEX {idx:2d} ({name:15s}): {result}")

    # === Section 6: Summary ===
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print("If all above tests pass, device is functioning normally.")
    print("WVCSV crash may be:")
    print("  - Firmware bug in WVCSV handler")
    print("  - Need specific VICP protocol (not standard USBTMC)")
    print("  - Command format issue")
    print("\n")

    # === Optional WVCSV Test ===
    print("=" * 60)
    print("OPTIONAL: Test WVCSV command")
    print("=" * 60)
    print("WARNING: This may crash the device!")

    response = input("Run WVCSV test? (y/N): ").strip().lower()
    if response == 'y':
        print("\nRunning minimal WVCSV test...")

        # Absolute minimum: 8 points
        data = "0,0.5,1,0.5,0,-0.5,-1,-0.5"
        cmd = f"WAVECSV M50,WAVENM,DIAGTEST,CSVLENG,8,CSVDATA,Amplitude,{data}"

        print(f"Command: {cmd}")
        print(f"Length: {len(cmd)} bytes")

        print("Sending in 3 seconds... (watch device screen)")
        time.sleep(3)

        try:
            device.write(cmd)
            print("Write completed. Waiting...")
            time.sleep(3)

            opc = query_safe(device, "*OPC?")
            print(f"*OPC?: {opc}")

            if "ERROR" in opc:
                print(">>> DEVICE CRASHED OR NOT RESPONDING <<<")
            else:
                print("Device still responding!")

                # Check if waveform was stored
                device.write("C1:ARWV NAME,DIAGTEST")
                time.sleep(0.5)
                result = query_safe(device, "C1:ARWV?")
                print(f"ARWV after NAME select: {result}")

        except Exception as e:
            print(f">>> EXCEPTION: {e} <<<")
    else:
        print("Skipping WVCSV test.")

    device.close()
    print("\nDiagnostic complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
