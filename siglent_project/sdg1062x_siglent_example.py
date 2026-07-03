#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 10:49:30 2026

@author: cyrusurheimog
"""


import pyvisa as visa  # They use "import visa" but pyvisa works the same
import time
import binascii

# USB resource of Device - will be auto-detected
device_resource = None

# Little endian, 16-bit 2's complement
# Their example uses 9 points
wave_points = [0x0080, 0x0070, 0x0060, 0x0040, 0x0050, 0x0060, 0x0070, 0xff7f, 0x0050]


def create_wave_file():
    """Create a binary waveform file - exactly as in Siglent example."""
    f = open("wave1.bin", "wb")
    for a in wave_points:
        b = hex(a)
        b = b[2:]
        len_b = len(b)
        if (0 == len_b):
            b = '0000'
        elif (1 == len_b):
            b = '000' + b
        elif (2 == len_b):
            b = '00' + b
        elif (3 == len_b):
            b = '0' + b
        c = binascii.a2b_hex(b)  # Hexadecimal integer to ASCII encoded string
        f.write(c)
    f.close()
    print(f"Created wave1.bin with {len(wave_points)} points")


def send_wave_data(dev):
    """Send wave1.bin to the device."""
    f = open("wave1.bin", "rb")
    data = f.read()
    f.close()
    print('Binary data bytes:', len(data))

    # Set waveform type to ARB first
    print("Setting CH1 to ARB mode...")
    dev.write("C1:BSWV WVTP,ARB")
    time.sleep(0.5)

    # Build command as bytes - ASCII header + binary data
    # The %s approach doesn't work well in Python 3 with binary data
    header = b"C1:WVDT WVNM,wave1,FREQ,2000.0,AMPL,4.0,OFST,0.0,PHASE,0.0,WAVEDATA,"
    full_cmd = header + data + b"\n"

    print(f"Uploading: {len(header)} byte header + {len(data)} byte data")
    print(f"Header: {header}")
    dev.write_raw(full_cmd)
    time.sleep(2.0)

    print("Selecting waveform...")
    dev.write("C1:ARWV NAME,wave1")
    time.sleep(0.5)

    print("Waveform sent and selected")


def main():
    print("=" * 60)
    print("SDG1062X - Siglent Example (from Programming Guide 5.1.4)")
    print("=" * 60)

    # Find device
    rm = visa.ResourceManager()
    resources = rm.list_resources()
    print(f"Resources: {resources}")

    # Find SDG
    global device_resource
    for r in resources:
        if 'USB' in r and ('F4EC' in r or 'SDG' in r.upper()):
            device_resource = r
            break

    if not device_resource:
        print("SDG not found!")
        return 1

    print(f"\nUsing: {device_resource}")

    # Open EXACTLY like Siglent example: timeout=50000, chunk_size=24*1024*1024
    device = rm.open_resource(device_resource, timeout=50000, chunk_size=24*1024*1024)

    print("\n--- Step 1: Create waveform file ---")
    create_wave_file()

    print("\n--- Step 2: Send waveform to device ---")
    response = input("Send waveform? (y/N): ").strip().lower()
    if response != 'y':
        device.close()
        return 0

    send_wave_data(device)

    print("\n--- Step 3: Enable output ---")
    device.write("C1:OUTP ON")
    print("CH1 output enabled")

    print("\n>>> Done! Check SDG front panel. <<<")

    device.close()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
