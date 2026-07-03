#!/usr/bin/env python3
"""
Plot Rubin Observatory mirror temperature data.
- Time series plot (most recent week) with sunset markers
- Histogram of temperature distribution
- Report max/min temperature events
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import EarthLocation, AltAz, get_sun
from astropy.time import Time
import astropy.units as u
from datetime import timedelta

# Observatory location
LAT = -30.24
LON = -70.74
ELEVATION = 2663  # meters
site = EarthLocation(lat=LAT*u.deg, lon=LON*u.deg, height=ELEVATION*u.m)

def get_sunset_time(date):
    """
    Estimate sunset time for a given date at Cerro Pachón.
    Sunset defined as sun altitude = -0.833 deg (standard refraction).
    """
    # Search from noon to midnight UTC (covers evening in Chile)
    # Chile is UTC-3 or UTC-4, so local evening is ~21:00-02:00 UTC
    start = Time(f"{date} 20:00:00")  # Start search at 20:00 UTC (~5pm local)

    # Search in 1-minute increments
    times = start + np.arange(0, 360, 1) * u.min  # Search 6 hours

    altaz_frame = AltAz(obstime=times, location=site)
    sun_altaz = get_sun(times).transform_to(altaz_frame)
    sun_alt = sun_altaz.alt.deg

    # Find where sun crosses -0.833 deg (going down)
    sunset_alt = -0.833
    for i in range(1, len(sun_alt)):
        if sun_alt[i-1] > sunset_alt and sun_alt[i] <= sunset_alt:
            # Linear interpolation for more precise time
            frac = (sunset_alt - sun_alt[i-1]) / (sun_alt[i] - sun_alt[i-1])
            sunset_time = times[i-1] + frac * u.min
            return sunset_time.datetime

    return None

# Load data
df = pd.read_csv('rubin_mirror_temps.csv', parse_dates=['timestamp'])

# Filter to most recent week
end_date = df['timestamp'].max()
start_date = end_date - timedelta(days=7)
df_week = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)].copy()

print(f"Showing data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

# Calculate sunset times for each day in the week
unique_dates = df_week['timestamp'].dt.date.unique()
sunset_times = []
print("\nCalculating sunset times...")
for date in unique_dates:
    sunset = get_sunset_time(date)
    if sunset:
        sunset_times.append(sunset)
        print(f"  {date}: sunset at {sunset.strftime('%H:%M:%S')} UTC")

# Get temperature at each sunset time (nearest data point)
sunset_temps = []
for st in sunset_times:
    idx = (df_week['timestamp'] - pd.Timestamp(st)).abs().idxmin()
    sunset_temps.append(df_week.loc[idx, 'temperature'])

# --- Figure 1: Temperature vs Time (Most Recent Week) ---
fig1, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(df_week['timestamp'], df_week['temperature'], 'b-', linewidth=1)
ax1.scatter(sunset_times, sunset_temps, c='red', s=100, zorder=5, label='Sunset')
ax1.set_xlabel('Date/Time (UTC)')
ax1.set_ylabel('Mirror Temperature (°C)')
ax1.set_title(f'Rubin Observatory Mirror Temperature - Final Week ({start_date.strftime("%b %d")} - {end_date.strftime("%b %d, %Y")})')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig('mirror_temp_timeseries.png', dpi=150)
print("\nSaved: mirror_temp_timeseries.png")

# --- Figure 2: Histogram (week data) ---
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.hist(df_week['temperature'], bins=30, edgecolor='black', alpha=0.7)
ax2.set_xlabel('Mirror Temperature (°C)')
ax2.set_ylabel('Frequency (count)')
ax2.set_title(f'Distribution of Mirror Temperatures ({start_date.strftime("%b %d")} - {end_date.strftime("%b %d, %Y")})')
ax2.grid(True, alpha=0.3, axis='y')
fig2.tight_layout()
fig2.savefig('mirror_temp_histogram.png', dpi=150)
print("Saved: mirror_temp_histogram.png")

# --- Report Max/Min (for the week) ---
idx_max = df_week['temperature'].idxmax()
idx_min = df_week['temperature'].idxmin()

max_row = df_week.loc[idx_max]
min_row = df_week.loc[idx_min]

print("\n" + "="*50)
print("TEMPERATURE EXTREMES (FINAL WEEK)")
print("="*50)
print(f"\nMAXIMUM TEMPERATURE:")
print(f"  Date:  {max_row['timestamp'].strftime('%Y-%m-%d')}")
print(f"  Time:  {max_row['timestamp'].strftime('%H:%M:%S')}")
print(f"  Value: {max_row['temperature']:.2f} °C")

print(f"\nMINIMUM TEMPERATURE:")
print(f"  Date:  {min_row['timestamp'].strftime('%Y-%m-%d')}")
print(f"  Time:  {min_row['timestamp'].strftime('%H:%M:%S')}")
print(f"  Value: {min_row['temperature']:.2f} °C")

plt.show()
