#!/usr/bin/env python3
"""
WaveStation test using NI-VISA backend explicitly.

The SCPI manual says USB interface requires:
1. NI-VISA
2. VICP Passport (may already be included or not needed for basic USBTMC)

This test uses NI-VISA backend and starts with the absolute minimum
waveform to isolate whether the crash is from:
- Command syntax
- Data size
- VISA backend issue
- Device firmware bug
"""

import pyvisa
import time
import sys

def test_basic_comms(device):
    """Test basic SCPI communication."""
    print("\n--- Basic Communication Test ---")
    tests = [
        ("*IDN?", "Identity query"),
        ("*OPC?", "Operation complete"),
        ("C1:BSWV?", "Basic wave query"),
        ("C1:ARWV?", "Arbitrary wave query"),
    ]

    for cmd, desc in tests:
        try:
            result = device.query(cmd).strip()
            print(f"  {desc}: OK - {result[:60]}...")
        except Exception as e:
            print(f"  {desc}: FAILED - {e}")
            return False
    return True


def test_arb_builtin(device):
    """Test selecting built-in ARB waveforms (no upload needed)."""
    print("\n--- Built-in ARB Waveform Test ---")

    # Set to ARB mode
    device.write("C1:BSWV WVTP,ARB")
    time.sleep(0.3)

    # Try selecting built-in waveforms by index
    builtins = [(19, "Gussian"), (18, "Sinc"), (28, "Chirp")]

    for idx, name in builtins:
        try:
            device.write(f"C1:ARWV INDEX,{idx}")
            time.sleep(0.3)
            result = device.query("C1:ARWV?").strip()
            print(f"  INDEX {idx} ({name}): {result}")
        except Exception as e:
            print(f"  INDEX {idx}: FAILED - {e}")

    return True


def test_wvcsv_minimal(device, slot=50):
    """Test WVCSV with absolute minimum data."""
    print(f"\n--- WVCSV Minimal Test (M{slot}) ---")

    # Absolute minimum: 8 points, simple integers
    # Following manual example exactly:
    # WAVECSV M36,WAVENM,WAVE1,CSVLENG,137166,CSVDATA,Amplitude,0,0.0012,...

    # 8 simple voltage values
    data = "0,0.5,1,0.5,0,-0.5,-1,-0.5"

    # Try the command
    cmd = f"WAVECSV M{slot},WAVENM,MINTEST,CSVLENG,8,CSVDATA,Amplitude,{data}"
    print(f"  Command length: {len(cmd)} bytes")
    print(f"  Command: {cmd}")

    print("  Sending...")
    try:
        bytes_written = device.write(cmd)
        print(f"  Bytes written: {bytes_written}")
    except Exception as e:
        print(f"  Write failed: {e}")
        return False

    print("  Waiting 2 seconds...")
    time.sleep(2)

    # Check if device still responds
    print("  Checking device response...")
    try:
        opc = device.query("*OPC?").strip()
        print(f"  *OPC?: {opc}")
    except Exception as e:
        print(f"  >>> DEVICE NOT RESPONDING: {e} <<<")
        return False

    # Check ESR for errors
    try:
        esr = device.query("*ESR?").strip()
        print(f"  *ESR?: {esr} (0=no errors)")
    except:
        pass

    return True


def verify_waveform(device, slot=50, name="MINTEST"):
    """Try to verify if waveform was stored."""
    print(f"\n--- Verify Waveform Storage ---")

    # Set ARB mode
    device.write("C1:BSWV WVTP,ARB")
    time.sleep(0.3)

    # Try by NAME
    print(f"  Trying NAME,{name}...")
    device.write(f"C1:ARWV NAME,{name}")
    time.sleep(0.5)
    result = device.query("C1:ARWV?").strip()
    print(f"  Result: {result}")

    if name.upper() in result.upper():
        print("  >>> SUCCESS: Waveform found by name! <<<")
        return True

    # Try by INDEX
    print(f"  Trying INDEX,{slot}...")
    device.write(f"C1:ARWV INDEX,{slot}")
    time.sleep(0.5)
    result = device.query("C1:ARWV?").strip()
    print(f"  Result: {result}")

    if "stairup" not in result.lower() and "empty" not in result.lower():
        print("  >>> Something at this INDEX (might have worked) <<<")
        return True

    print("  Waveform not found")
    return False


def main():
    print("=" * 60)
    print("WaveStation NI-VISA WVCSV Test")
    print("=" * 60)

    # Explicitly use NI-VISA (not pyvisa-py)
    try:
        rm = pyvisa.ResourceManager()  # Uses NI-VISA if available
        backend_str = str(rm.visalib)
        if 'pyvisa-py' in backend_str.lower() or 'py' == backend_str.lower():
            print("WARNING: Using pyvisa-py, not NI-VISA")
            print("NI-VISA may be required for WVCSV to work correctly")
        else:
            print(f"Using: {rm.visalib}")
    except Exception as e:
        print(f"Failed to initialize VISA: {e}")
        return 1

    # List and find WaveStation
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
    device.timeout = 30000

    # Clear status
    device.write("*CLS")
    time.sleep(0.3)

    # Run tests
    if not test_basic_comms(device):
        print("\nBasic communication failed!")
        device.close()
        return 1

    if not test_arb_builtin(device):
        print("\nBuilt-in ARB test failed!")
        device.close()
        return 1

    # THE CRITICAL TEST
    print("\n" + "=" * 60)
    print("CRITICAL: Testing WVCSV command")
    print("If device crashes here, the issue is with WVCSV itself")
    print("=" * 60)

    input("Press Enter to send WVCSV command (watch device screen)...")

    if not test_wvcsv_minimal(device, slot=50):
        print("\n>>> WVCSV TEST FAILED <<<")
        print("Device may have crashed or command was rejected")
        return 1

    # Verify
    verify_waveform(device, slot=50, name="MINTEST")

    print("\n" + "=" * 60)
    print("Test complete")
    print("Check device: Arb -> Load Wform -> Stored Wforms")
    print("=" * 60)

    device.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
