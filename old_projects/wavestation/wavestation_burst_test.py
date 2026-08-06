#!/usr/bin/env python3
"""
WaveStation Burst Mode Test
Alternative to custom ARB - uses burst mode for gated 40kHz pulses
This may be sufficient for batping TOF measurements
"""

from wavestation import WaveStation
import time

def main():
    print("=" * 50)
    print("WaveStation Burst Mode Test (40kHz)")
    print("=" * 50)

    ws = WaveStation()
    ws.connect()

    # Step 1: Configure basic 40kHz sine wave
    print("\n[1] Setting up 40kHz sine wave...")
    ws.write("C1:BSWV WVTP,SINE,FRQ,40000HZ,AMP,4V,OFST,0V")
    time.sleep(0.3)
    print(f"    BSWV: {ws.query('C1:BSWV?')}")

    # Step 2: Configure burst mode
    # NCYC = N-cycle burst (specific number of cycles)
    # TIME = number of cycles per burst
    # PRD = period between bursts
    # TRSR = trigger source (INT = internal, EXT = external, MAN = manual)
    print("\n[2] Configuring burst mode...")
    print("    10 cycles of 40kHz, repeating every 100ms")

    ws.write("C1:BTWV STATE,ON,TRSR,INT,GATE_NCYC,NCYC,TIME,10,PRD,0.1S")
    time.sleep(0.3)
    print(f"    BTWV: {ws.query('C1:BTWV?')}")

    # Step 3: Enable output
    print("\n[3] Enabling output...")
    ws.write("C1:OUTP ON,LOAD,HZ")
    time.sleep(0.3)
    print(f"    OUTP: {ws.query('C1:OUTP?')}")

    print("\n" + "=" * 50)
    print("Burst mode active!")
    print("You should see 10 cycles of 40kHz every 100ms on CH1")
    print("")
    print("To adjust:")
    print("  - Change TIME,10 to change number of cycles per burst")
    print("  - Change PRD,0.1S to change repetition period")
    print("  - Change TRSR to EXT for external triggering")
    print("=" * 50)

    input("\nPress Enter to disable output and exit...")

    ws.write("C1:OUTP OFF")
    ws.write("C1:BTWV STATE,OFF")
    ws.disconnect()
    print("Done.")

if __name__ == "__main__":
    main()
