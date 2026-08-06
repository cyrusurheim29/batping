#!/usr/bin/env python3
"""
sensehat_temperature_log.py

Log SenseHat v2 temperature + RPi CPU temperature to a CSV file with
UTC timestamps. Designed to run concurrently with picoscope_batping_record.py
on the Mac so that the two time series can be compared post-hoc.

Usage (on Raspberry Pi 4B):
    python3 sensehat_temperature_log.py [duration_s] [output_csv]

    duration_s  : recording length in seconds (default 1800 = 30 min)
    output_csv  : output filename (default auto-generated from timestamp)

Example:
    python3 sensehat_temperature_log.py 1500 heattest_sensehat_20260722.csv

Temperature channels logged:
    t_humidity_C  : SenseHat BME280 humidity-channel temperature (raw)
    t_pressure_C  : SenseHat BME280 pressure-channel temperature (raw)
    cpu_temp_C    : RPi CPU temperature via vcgencmd
    t_corrected_C : CPU-heat-corrected estimate of ambient temperature
                    Formula: t_hum - (cpu_temp - 40) * 0.32
                    (factor 0.32 is empirical; calibrate against reference
                     thermometer if available, but relative ΔT is valid regardless)

Note on absolute accuracy:
    The SenseHat sits above the RPi PCB and reads 5-10 °C above true ambient
    when idle. Use t_corrected_C for comparisons, but even raw t_humidity_C
    tracks relative temperature changes reliably (ΔT, not absolute T).
    For cross-correlation with BATPING, only relative changes matter.
"""

import time
import csv
import subprocess
import sys
import os
from datetime import datetime, timezone

# ── Attempt to import SenseHat ────────────────────────────────────────────────
try:
    from sense_hat import SenseHat
except ImportError:
    print("ERROR: sense_hat not found. Install with:  pip install sense-hat")
    print("       Or run on a Raspberry Pi with SenseHat attached.")
    sys.exit(1)

# ── Settings ──────────────────────────────────────────────────────────────────
INTERVAL_S  = 1.0       # sample interval in seconds
DEFAULT_DUR = 1800      # default recording duration (30 min)


def cpu_temp_c():
    """Read RPi CPU temperature via vcgencmd (Raspberry Pi only)."""
    try:
        out = subprocess.check_output(
            ['vcgencmd', 'measure_temp'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return float(out.replace('temp=', '').replace("'C", ''))
    except Exception:
        return float('nan')


def main():
    duration_s = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DUR
    if len(sys.argv) > 2:
        output_csv = sys.argv[2]
    else:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  f'sensehat_temp_{stamp}.csv')

    print(f"SenseHat temperature logger")
    print(f"  Duration : {duration_s} s ({duration_s/60:.1f} min)")
    print(f"  Interval : {INTERVAL_S} s")
    print(f"  Output   : {output_csv}")
    print(f"  Start UTC: {datetime.now(timezone.utc).isoformat()}")
    print()

    sense = SenseHat()
    sense.clear()

    file_exists = os.path.isfile(output_csv) and os.path.getsize(output_csv) > 0
    with open(output_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                'utc_timestamp',
                't_humidity_C',
                't_pressure_C',
                'cpu_temp_C',
                't_corrected_C',
            ])

        t_start   = time.time()
        n_samples = 0

        while True:
            elapsed = time.time() - t_start
            if elapsed >= duration_s:
                break

            utc_now  = datetime.now(timezone.utc).isoformat()
            t_hum    = sense.get_temperature()
            t_pres   = sense.get_temperature_from_pressure()
            t_cpu    = cpu_temp_c()
            # Empirical CPU-heat correction; adjust factor if calibration data available
            t_corr   = t_hum - (t_cpu - 40.0) * 0.32

            writer.writerow([
                utc_now,
                round(t_hum,  3),
                round(t_pres, 3),
                round(t_cpu,  2),
                round(t_corr, 3),
            ])
            f.flush()
            n_samples += 1

            if n_samples % 60 == 1:
                print(f"  t={elapsed:6.0f}s  T_hum={t_hum:.2f}°C  "
                      f"T_corr={t_corr:.2f}°C  CPU={t_cpu:.1f}°C", flush=True)

            time.sleep(INTERVAL_S)

    print(f"\nDone. {n_samples} samples written to {output_csv}")
    print(f"End UTC: {datetime.now(timezone.utc).isoformat()}")


if __name__ == '__main__':
    main()
