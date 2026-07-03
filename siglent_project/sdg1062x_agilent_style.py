#!/usr/bin/env python3
"""
SDG1062X - Try Agilent-style connection/commands

Testing if basic PyVISA query/write works the same way as Agilent.
"""

import pyvisa
import numpy as np
import time


def main():
    print("=" * 60)
    print("SDG1062X - Agilent-Style Test")
    print("=" * 60)

    rm = pyvisa.ResourceManager()
    resources = rm.list_resources()
    print(f"Resources: {resources}")

    # Find SDG
    sdg_resource = None
    for r in resources:
        if 'USB' in r and ('F4EC' in r or 'SDG' in r.upper()):
            sdg_resource = r
            break

    if not sdg_resource:
        print("SDG not found!")
        return 1

    print(f"\nConnecting to: {sdg_resource}")

    # Connect exactly like Agilent code does
    dev = rm.open_resource(sdg_resource)
    dev.timeout = 10000  # 10 seconds, same as Agilent

    # Try basic query - exactly like Agilent
    print("\n--- Testing *IDN? query (Agilent style) ---")
    try:
        idn = dev.query("*IDN?")
        print(f"SUCCESS! Device: {idn.strip()}")
    except Exception as e:
        print(f"FAILED: {e}")
        print("\nTrying with explicit termination characters...")

        # Try setting termination like some Siglent examples show
        dev.read_termination = '\n'
        dev.write_termination = '\n'

        try:
            idn = dev.query("*IDN?")
            print(f"SUCCESS with termination! Device: {idn.strip()}")
        except Exception as e2:
            print(f"Still failed: {e2}")

            # One more try - some devices need CR+LF
            print("\nTrying CR+LF termination...")
            dev.read_termination = '\r\n'
            dev.write_termination = '\r\n'

            try:
                idn = dev.query("*IDN?")
                print(f"SUCCESS with CRLF! Device: {idn.strip()}")
            except Exception as e3:
                print(f"All query methods failed: {e3}")
                print("\n*** Cannot establish reliable communication ***")
                dev.close()
                return 1

    # If we got here, queries work
    print("\n--- Testing other queries ---")

    queries = [
        "*OPC?",
        "C1:BSWV?",      # Basic wave query
        "C1:OUTP?",      # Output state
    ]

    for q in queries:
        try:
            result = dev.query(q).strip()
            print(f"  {q} -> {result}")
        except Exception as e:
            print(f"  {q} -> FAILED: {e}")

    # Try Agilent-style DATA command (probably won't work but let's see)
    print("\n--- Testing waveform upload methods ---")

    # Simple 8-point test data
    test_data = [0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5]
    data_str = ",".join(f"{v:.6f}" for v in test_data)

    print(f"\nTest data: {test_data}")

    # Method 1: Agilent style
    print("\n1. Agilent-style: DATA VOLATILE,<values>")
    try:
        dev.write(f"DATA VOLATILE,{data_str}")
        time.sleep(0.5)
        opc = dev.query("*OPC?").strip()
        print(f"   *OPC?: {opc} (1 = success)")
    except Exception as e:
        print(f"   Failed: {e}")

    # Method 2: Siglent C1:WVDT with ASCII data
    print("\n2. Siglent WVDT with ASCII: C1:WVDT WVNM,test,WAVEDATA,<values>")
    try:
        dev.write(f"C1:WVDT WVNM,test1,FREQ,1000,AMPL,2,OFST,0,PHASE,0,WAVEDATA,{data_str}")
        time.sleep(0.5)
        opc = dev.query("*OPC?").strip()
        print(f"   *OPC?: {opc}")
    except Exception as e:
        print(f"   Failed: {e}")

    # Method 3: Check what waveforms exist
    print("\n3. Query stored waveforms: STL? USER")
    try:
        result = dev.query("STL? USER").strip()
        print(f"   Stored waveforms: {result}")
    except Exception as e:
        print(f"   Failed: {e}")

    dev.close()
    print("\nDone!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
