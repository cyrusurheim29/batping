#!/usr/bin/env python3
"""
WaveStation: Reset device first, then try WVCSV.

Hypothesis: Device internal state may be corrupted from previous
failed WVCSV attempts. A *RST might clear this.

From SCPI manual:
- *RST: "initiates a device reset, sets all eight traces to GND line
        and recalls the default setup"
- *TST?: Performs internal self-test
"""

import pyvisa
import time
import sys

def main():
    print("=" * 60)
    print("WaveStation: Reset then WVCSV Test")
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

    # Get device info
    idn = device.query("*IDN?").strip()
    print(f"Connected: {idn}")

    # Parse firmware version
    parts = idn.split(",")
    if len(parts) >= 5:
        print(f"  SW: {parts[3]}, FW: {parts[4]}")

    # Step 1: Full device reset
    print("\n" + "=" * 60)
    print("STEP 1: Device Reset (*RST)")
    print("=" * 60)

    print("Sending *RST...")
    device.write("*RST")
    print("Waiting 5 seconds for reset to complete...")
    time.sleep(5)

    # Verify device is back
    try:
        opc = device.query("*OPC?").strip()
        print(f"*OPC?: {opc} (device responded)")
    except Exception as e:
        print(f"Device not responding after reset: {e}")
        return 1

    # Step 2: Clear status registers
    print("\n" + "=" * 60)
    print("STEP 2: Clear Status (*CLS)")
    print("=" * 60)

    device.write("*CLS")
    time.sleep(0.5)

    esr = device.query("*ESR?").strip()
    print(f"*ESR?: {esr} (should be 0)")

    # Step 3: Self-test
    print("\n" + "=" * 60)
    print("STEP 3: Self-Test (*TST?)")
    print("=" * 60)

    print("Running self-test (may take a moment)...")
    device.timeout = 60000
    try:
        tst = device.query("*TST?").strip()
        print(f"*TST?: {tst} (0=pass, 1=fail)")
        if "1" in tst:
            print(">>> SELF-TEST FAILED - device may have hardware issue <<<")
    except Exception as e:
        print(f"Self-test query failed: {e}")

    device.timeout = 30000

    # Step 4: Check initial state
    print("\n" + "=" * 60)
    print("STEP 4: Check Initial State")
    print("=" * 60)

    bswv = device.query("C1:BSWV?").strip()
    print(f"C1:BSWV?: {bswv}")

    arwv = device.query("C1:ARWV?").strip()
    print(f"C1:ARWV?: {arwv}")

    # Step 5: Try WVCSV
    print("\n" + "=" * 60)
    print("STEP 5: WVCSV Test")
    print("=" * 60)

    # Minimal 8-point waveform, small voltage values
    data = "0,0.5,1,0.5,0,-0.5,-1,-0.5"
    cmd = f"WVCSV M50,WAVENM,RSTTEST,CSVLENG,8,CSVDATA,Amplitude,{data}"

    print(f"Command: {cmd}")
    print(f"Length: {len(cmd)} bytes")

    response = input("\nSend WVCSV command? (y/N): ").strip().lower()
    if response != 'y':
        print("Skipping WVCSV test.")
        device.close()
        return 0

    print("\nSending WVCSV (watch device screen)...")
    try:
        bytes_written = device.write(cmd)
        print(f"Bytes written: {bytes_written}")

        print("Waiting 3 seconds...")
        time.sleep(3)

        # Check response
        opc = device.query("*OPC?").strip()
        print(f"*OPC?: {opc}")
        print(">>> Device still responding! <<<")

        # Check if waveform stored
        print("\nVerifying waveform...")
        device.write("C1:BSWV WVTP,ARB")
        time.sleep(0.3)
        device.write("C1:ARWV NAME,RSTTEST")
        time.sleep(0.5)
        result = device.query("C1:ARWV?").strip()
        print(f"ARWV: {result}")

        if "RSTTEST" in result.upper():
            print("\n>>> SUCCESS! Waveform uploaded! <<<")
        else:
            print("\nWaveform not found by name, trying INDEX 50...")
            device.write("C1:ARWV INDEX,50")
            time.sleep(0.5)
            result = device.query("C1:ARWV?").strip()
            print(f"ARWV INDEX,50: {result}")

    except Exception as e:
        print(f"\n>>> ERROR: {e} <<<")
        print("Device may have crashed")
        return 1

    device.close()
    print("\nTest complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
