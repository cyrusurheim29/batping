#!/usr/bin/env python3
"""
Test NI-VISA backend vs pyvisa-py backend
Run after installing NI-VISA
"""

import pyvisa
import time

def test_backend(backend_name):
    """Test a specific VISA backend."""
    print(f"\n{'='*50}")
    print(f"Testing backend: {backend_name}")
    print('='*50)

    try:
        if backend_name == "ni":
            rm = pyvisa.ResourceManager()  # Uses NI-VISA by default if installed
        else:
            rm = pyvisa.ResourceManager('@py')  # Pure Python backend

        print(f"Backend info: {rm.visalib}")

        # List resources
        resources = rm.list_resources()
        print(f"Found resources: {resources}")

        # Find WaveStation
        ws_resources = [r for r in resources if '1535' in r or 'LCRY' in r or 'USB' in r]
        if not ws_resources:
            print("No WaveStation found")
            return None

        resource = ws_resources[0]
        print(f"Connecting to: {resource}")

        device = rm.open_resource(resource)
        device.timeout = 10000

        # Test basic communication
        idn = device.query("*IDN?")
        print(f"IDN: {idn.strip()}")

        return device, rm

    except Exception as e:
        print(f"Error: {e}")
        return None


def test_wvcsv_with_backend(device, n_points=1024):
    """Test WVCSV command with the connected device."""
    import numpy as np

    print(f"\nTesting WVCSV with {n_points} points...")

    # Increase timeout significantly for large transfers
    device.timeout = 60000  # 60 seconds
    print(f"Timeout set to: {device.timeout}ms")

    # Generate waveform
    t = np.linspace(0, 2*np.pi*10, n_points)
    waveform = np.sin(t)
    data_values = ",".join([f"{v:.4f}" for v in waveform])

    cmd = f"WVCSV M50,WAVENM,NITEST,CSVLENG,{n_points},CSVDATA,Amplitude,{data_values}"
    print(f"Command length: {len(cmd)} bytes")

    # Send command
    bytes_written = device.write(cmd)
    print(f"Bytes written: {bytes_written}")

    print("Waiting for device to process (this may take a while)...")
    time.sleep(5)  # Give device time to process

    # Check OPC with extended timeout
    try:
        opc = device.query("*OPC?")
        print(f"OPC: {opc.strip()}")
    except Exception as e:
        print(f"OPC query failed: {e}")
        print("Continuing anyway to check if waveform was stored...")

    # Try to select it
    device.write("C1:BSWV WVTP,ARB")
    time.sleep(0.3)

    device.write("C1:ARWV NAME,NITEST")
    time.sleep(0.3)

    result = device.query("C1:ARWV?")
    print(f"ARWV result: {result.strip()}")

    if "NITEST" in result.upper():
        print("\n*** SUCCESS! Waveform uploaded and selected! ***")
        return True
    elif "stairup" not in result.lower():
        print("\n*** POSSIBLY WORKED - different waveform selected ***")
        return True
    else:
        print("\nWVCSV still not working with this backend")
        return False


def main():
    print("NI-VISA vs pyvisa-py Backend Test")
    print("="*50)

    # Check what backends are available
    print("\nChecking available backends...")

    # Try NI-VISA first
    print("\n--- Attempting NI-VISA backend ---")
    try:
        rm_ni = pyvisa.ResourceManager()
        print(f"NI-VISA available: {rm_ni.visalib}")
        has_nivisa = "pyvisa-py" not in str(rm_ni.visalib).lower()
    except Exception as e:
        print(f"NI-VISA not available: {e}")
        has_nivisa = False

    # Try pyvisa-py
    print("\n--- Attempting pyvisa-py backend ---")
    try:
        rm_py = pyvisa.ResourceManager('@py')
        print(f"pyvisa-py available: {rm_py.visalib}")
        has_pyvisapy = True
    except Exception as e:
        print(f"pyvisa-py not available: {e}")
        has_pyvisapy = False

    # Test with preferred backend
    if has_nivisa:
        print("\n" + "="*50)
        print("TESTING WITH NI-VISA BACKEND")
        print("="*50)
        result = test_backend("ni")
        if result:
            device, rm = result
            test_wvcsv_with_backend(device, n_points=1024)
            device.close()
    else:
        print("\n*** NI-VISA not installed ***")
        print("Please install NI-VISA from:")
        print("https://www.ni.com/en/support/downloads/drivers/download.ni-visa.html")

        if has_pyvisapy:
            print("\nFalling back to pyvisa-py...")
            result = test_backend("py")
            if result:
                device, rm = result
                test_wvcsv_with_backend(device, n_points=1024)
                device.close()


if __name__ == "__main__":
    main()
