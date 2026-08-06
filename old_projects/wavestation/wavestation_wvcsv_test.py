#!/usr/bin/env python3
"""
WaveStation WVCSV Test with Various Point Counts
Testing if minimum point count is the issue
"""

from wavestation import WaveStation
import numpy as np
import time

def test_wvcsv_upload(ws, n_points, slot=50):
    """Test WVCSV upload with specified number of points."""
    print(f"\n--- Testing {n_points} points to M{slot} ---")

    # Generate simple sine wave data (voltage values)
    t = np.linspace(0, 2*np.pi*10, n_points)  # 10 cycles
    waveform = np.sin(t)  # -1 to 1 range

    # Format as voltage values (staying within safe range)
    data_values = ",".join([f"{v:.4f}" for v in waveform])
    csv_data = f"Amplitude,{data_values}"

    # Build command
    cmd = f"WVCSV M{slot},WAVENM,TEST{n_points},CSVLENG,{n_points},CSVDATA,{csv_data}"
    print(f"  Command length: {len(cmd)} bytes")

    # Send
    bytes_written = ws.device.write(cmd)
    print(f"  Bytes written: {bytes_written}")

    time.sleep(2)  # Give more time for larger waveforms

    # Check OPC
    opc = ws.query("*OPC?")
    print(f"  OPC: {opc}")

    # Try to select it
    ws.write("C1:BSWV WVTP,ARB")
    time.sleep(0.3)

    # Try by name
    ws.write(f"C1:ARWV NAME,TEST{n_points}")
    time.sleep(0.3)
    result = ws.query("C1:ARWV?")
    print(f"  ARWV after NAME select: {result}")

    if f"TEST{n_points}" in result.upper():
        print(f"  SUCCESS with {n_points} points!")
        return True

    # Try by index
    ws.write(f"C1:ARWV INDEX,{slot}")
    time.sleep(0.3)
    result = ws.query("C1:ARWV?")
    print(f"  ARWV after INDEX select: {result}")

    if "stairup" not in result.lower():
        print(f"  POSSIBLE SUCCESS - different waveform selected!")
        return True

    return False


def main():
    print("=" * 60)
    print("WaveStation WVCSV Point Count Test")
    print("=" * 60)

    ws = WaveStation()
    ws.connect()
    ws.device.timeout = 60000  # 60 second timeout for large uploads

    print(f"\nDevice: {ws.query('*IDN?')}")
    print(f"Initial ARWV: {ws.query('C1:ARWV?')}")

    # Test various point counts
    test_counts = [32, 100, 512, 1024, 4096, 8192, 16384]

    results = {}
    for n in test_counts:
        try:
            success = test_wvcsv_upload(ws, n, slot=50)
            results[n] = success
            if success:
                print(f"\n*** FOUND WORKING POINT COUNT: {n} ***")
                break
        except Exception as e:
            print(f"  ERROR with {n} points: {e}")
            results[n] = False

    print("\n" + "=" * 60)
    print("Results Summary:")
    for n, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {n:>6} points: {status}")
    print("=" * 60)

    # Final check - look at device
    print("\nPlease check device screen:")
    print("  Arb -> 1/2 down -> Load Wform -> Stored Wforms")
    print("  Look for any TEST* waveforms")

    ws.disconnect()


if __name__ == "__main__":
    main()
