#!/usr/bin/env python3
"""
WaveStation ARB Upload Debug Script
Tests WVCSV upload and various selection methods
"""

from wavestation import WaveStation
import time

def main():
    print("=" * 50)
    print("WaveStation ARB Upload Debug")
    print("=" * 50)

    ws = WaveStation()
    ws.connect()

    # Step 1: Confirm communication
    print("\n[1] Communication check:")
    print(f"    IDN: {ws.query('*IDN?')}")
    print(f"    Current BSWV: {ws.query('C1:BSWV?')}")
    print(f"    Current ARWV: {ws.query('C1:ARWV?')}")

    # Step 2: Upload waveform via WVCSV
    print("\n[2] Uploading test waveform to M50...")
    data_values = ",".join(["0.5"] * 32)
    cmd = f"WVCSV M50,WAVENM,TEST,CSVLENG,32,CSVDATA,Amplitude,{data_values}"
    print(f"    Command length: {len(cmd)} bytes")

    bytes_written = ws.device.write(cmd)
    print(f"    Bytes written: {bytes_written}")

    time.sleep(1)
    opc = ws.query("*OPC?")
    print(f"    OPC status: {opc}")

    # Step 3: Set to ARB mode
    print("\n[3] Setting channel to ARB mode...")
    ws.write("C1:BSWV WVTP,ARB")
    time.sleep(0.5)
    print(f"    BSWV after: {ws.query('C1:BSWV?')}")

    # Step 4: Try selecting by NAME
    print("\n[4] Trying ARWV NAME,TEST...")
    ws.write("C1:ARWV NAME,TEST")
    time.sleep(0.5)
    result = ws.query("C1:ARWV?")
    print(f"    Result: {result}")
    if "TEST" in result.upper():
        print("    SUCCESS!")

    # Step 5: Try selecting by various indices
    print("\n[5] Trying various ARWV INDEX values...")
    test_indices = [50, 51, 52, 59, 31, 32, 33, 40, 41, 49]
    for idx in test_indices:
        ws.write(f"C1:ARWV INDEX,{idx}")
        time.sleep(0.3)
        result = ws.query("C1:ARWV?")
        # Only print if it's not the default stairup
        if "stairup" not in result.lower():
            print(f"    INDEX {idx}: {result} <-- DIFFERENT!")
        else:
            print(f"    INDEX {idx}: {result}")

    # Step 6: Try STL? with longer timeout
    print("\n[6] Trying STL? query (may timeout)...")
    ws.device.timeout = 30000
    try:
        ws.device.write("STL?")
        time.sleep(2)
        response = ws.device.read()
        # Look for M50 in response
        if "M50" in response:
            m50_start = response.find("M50")
            m50_section = response[m50_start:m50_start+30]
            print(f"    M50 section: {m50_section}")
        else:
            print(f"    Response (first 200 chars): {response[:200]}")
    except Exception as e:
        print(f"    STL? failed: {e}")
    finally:
        ws.device.timeout = 10000

    # Step 7: Summary
    print("\n" + "=" * 50)
    print("Debug complete. Check device screen:")
    print("  Arb -> 1/2 down -> Load Wform -> Stored Wforms")
    print("  Look for 'TEST' in the list")
    print("=" * 50)

    ws.disconnect()

if __name__ == "__main__":
    main()
