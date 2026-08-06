#!/usr/bin/env python3
"""
Test WaveStation using VKEY (Virtual Key) commands.

If WVCSV crashes the device, we might be able to use VKEY to:
1. Navigate menus
2. Load waveforms from USB
3. As a workaround for WVCSV

From SCPI manual (p.27-28):
VKEY VALUE,<value>,STATE,1

WaveStation 2000 specific keys:
- KB_ARB: 9
- KB_STORE_RECALL: 10
- KB_UP: 45
- KB_DOWN: 39
- KB_LEFT: 44
- KB_RIGHT: 40
"""

import pyvisa
import time
import sys

# WaveStation 2000 virtual key codes
KEYS = {
    'SINE': 34,
    'SQUARE': 29,
    'RAMP': 24,
    'PULSE': 19,
    'NOISE': 14,
    'ARB': 9,
    'STORE_RECALL': 10,
    'UP': 45,
    'DOWN': 39,
    'LEFT': 44,
    'RIGHT': 40,
    'CHANNEL': 33,
    'OUTPUT1': 153,
    'OUTPUT2': 152,
    'UTILITY': 11,
    'FUNC1': 28,
    'FUNC2': 23,
    'FUNC3': 18,
    'FUNC4': 13,
    'FUNC5': 8,
    'KNOB_RIGHT': 175,
    'KNOB_LEFT': 177,
}


def press_key(device, key_name, delay=0.3):
    """Simulate pressing a front panel key."""
    if key_name not in KEYS:
        print(f"Unknown key: {key_name}")
        return False

    key_code = KEYS[key_name]
    cmd = f"VKEY VALUE,{key_code},STATE,1"
    try:
        device.write(cmd)
        time.sleep(delay)
        return True
    except Exception as e:
        print(f"VKEY failed: {e}")
        return False


def main():
    print("=" * 60)
    print("WaveStation VKEY Test")
    print("=" * 60)

    rm = pyvisa.ResourceManager()
    print(f"Backend: {rm.visalib}")

    resources = rm.list_resources()
    ws_resource = None
    for r in resources:
        if '05FF' in r or 'LCRY' in r:
            ws_resource = r
            break

    if not ws_resource:
        print("WaveStation not found!")
        return 1

    device = rm.open_resource(ws_resource)
    device.timeout = 10000

    idn = device.query("*IDN?").strip()
    print(f"Connected: {idn}")

    print("\n--- Testing VKEY commands ---")
    print("Watch the device screen as keys are 'pressed'\n")

    # Test 1: Press ARB button
    print("Test 1: Pressing ARB button...")
    press_key(device, 'ARB')
    time.sleep(1)

    # Query current state
    result = device.query("C1:BSWV?").strip()
    print(f"  BSWV after ARB press: {result}")

    # Test 2: Press STORE_RECALL
    print("\nTest 2: Pressing STORE_RECALL button...")
    press_key(device, 'STORE_RECALL')
    time.sleep(1)

    # Test 3: Navigate with UP/DOWN
    print("\nTest 3: Pressing DOWN to navigate menu...")
    press_key(device, 'DOWN')
    time.sleep(0.5)
    press_key(device, 'DOWN')
    time.sleep(0.5)

    # Test 4: Go back to ARB
    print("\nTest 4: Back to ARB...")
    press_key(device, 'ARB')
    time.sleep(0.5)

    # Test selecting a built-in waveform via VKEY navigation
    print("\n--- Navigating ARB waveforms ---")
    print("Pressing DOWN to cycle through waveforms...")

    for i in range(5):
        press_key(device, 'KNOB_RIGHT', delay=0.2)
        result = device.query("C1:ARWV?").strip()
        print(f"  After knob turn {i+1}: {result}")

    print("\n--- VKEY test complete ---")
    print("\nIf VKEY works, we could potentially:")
    print("1. Save waveform to USB stick on computer")
    print("2. Use VKEY to navigate Load Wform menu")
    print("3. Load from USB via simulated keypresses")
    print("\nBut this requires USB stick workflow, not real-time...")

    device.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
