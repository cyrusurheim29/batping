#!/usr/bin/env python3
"""
WaveStation: Try raw byte-level write for WVCSV.

Hypothesis: pyvisa's write() method may add something that causes the crash.
Let's try write_raw() to send exactly the bytes we want.
"""

import pyvisa
import time
import sys

def main():
    print("=" * 60)
    print("WaveStation: Raw Write WVCSV Test")
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

    idn = device.query("*IDN?").strip()
    print(f"Connected: {idn}")

    # Check device properties
    print(f"\nDevice properties:")
    print(f"  write_termination: {repr(device.write_termination)}")
    print(f"  read_termination: {repr(device.read_termination)}")
    print(f"  encoding: {device.encoding}")
    print(f"  chunk_size: {device.chunk_size}")

    # Test 1: Normal write vs raw write comparison
    print("\n" + "=" * 60)
    print("Comparing write() vs write_raw()")
    print("=" * 60)

    test_cmd = "*CLS"

    # Normal write
    print(f"\nNormal write('{test_cmd}'):")
    bytes_written = device.write(test_cmd)
    print(f"  Bytes reported: {bytes_written}")

    time.sleep(0.3)
    opc = device.query("*OPC?").strip()
    print(f"  *OPC?: {opc}")

    # Raw write - what bytes exactly?
    print(f"\nRaw write_raw(b'{test_cmd}'):")
    raw_bytes = test_cmd.encode('ascii')
    print(f"  Sending bytes: {raw_bytes}")

    # First clear any pending data
    device.write("*CLS")
    time.sleep(0.3)

    # Now the WVCSV test with raw write
    print("\n" + "=" * 60)
    print("WVCSV with write_raw()")
    print("=" * 60)

    # Build command as raw bytes - no terminator
    data = "0,0.5,1,0.5,0,-0.5,-1,-0.5"
    cmd_str = f"WVCSV M50,WAVENM,RAWTEST,CSVLENG,8,CSVDATA,Amplitude,{data}"

    print(f"Command string: {cmd_str}")
    print(f"Length: {len(cmd_str)} chars")

    # Convert to bytes
    cmd_bytes = cmd_str.encode('ascii')
    print(f"Bytes: {len(cmd_bytes)} bytes")
    print(f"Last 20 bytes: {cmd_bytes[-20:]}")

    response = input("\nSend with write_raw()? (y/N): ").strip().lower()
    if response != 'y':
        print("Skipping.")
        device.close()
        return 0

    print("\nSending via write_raw()...")
    try:
        # Use write_raw to bypass any pyvisa processing
        bytes_written = device.write_raw(cmd_bytes)
        print(f"Bytes written: {bytes_written}")

        print("Waiting 3 seconds...")
        time.sleep(3)

        # Check device
        device.read_termination = '\n'
        opc = device.query("*OPC?").strip()
        print(f"*OPC?: {opc}")
        print(">>> Device still responding! <<<")

        # Verify
        device.write("C1:BSWV WVTP,ARB")
        time.sleep(0.3)
        device.write("C1:ARWV NAME,RAWTEST")
        time.sleep(0.5)
        result = device.query("C1:ARWV?").strip()
        print(f"ARWV: {result}")

    except Exception as e:
        print(f">>> ERROR: {e} <<<")

    # Also try with \n terminator via raw
    print("\n" + "=" * 60)
    print("WVCSV with write_raw() + newline")
    print("=" * 60)

    response = input("Try with \\n terminator? (y/N): ").strip().lower()
    if response == 'y':
        # Reconnect if crashed
        try:
            device.query("*IDN?")
        except:
            print("Reconnecting...")
            device.close()
            rm.close()
            rm = pyvisa.ResourceManager()
            ws_resource = [r for r in rm.list_resources() if '05FF' in r or 'LCRY' in r][0]
            device = rm.open_resource(ws_resource)
            device.timeout = 30000

        cmd_str2 = f"WVCSV M51,WAVENM,NLTEST,CSVLENG,8,CSVDATA,Amplitude,{data}"
        cmd_bytes2 = (cmd_str2 + "\n").encode('ascii')

        print(f"Sending {len(cmd_bytes2)} bytes ending with: {cmd_bytes2[-15:]}")

        try:
            device.write_raw(cmd_bytes2)
            time.sleep(3)
            opc = device.query("*OPC?").strip()
            print(f"*OPC?: {opc}")
        except Exception as e:
            print(f">>> ERROR: {e} <<<")

    device.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
