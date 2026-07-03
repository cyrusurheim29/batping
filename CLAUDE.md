# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a telescope/optical device orientation system using a Raspberry Pi with Sense HAT IMU. The goal is to determine pointing direction in both local (Alt/Az) and celestial (RA/Dec) coordinates.

## Main Components

**accel_magneto_v2.py** - Core orientation calculation pipeline:
- Accelerometer → roll/pitch from gravity vector
- Magnetometer → tilt-compensated heading (yaw)
- Body frame → ENU frame transformation via rotation matrices
- Alt/Az → RA/Dec conversion via Astropy

## Dependencies

- `sense_hat` - Raspberry Pi Sense HAT library (accelerometer, magnetometer, gyroscope)
- `numpy` - Vector/matrix operations
- `astropy` - Celestial coordinate transformations

## Running

```bash
# On Raspberry Pi with Sense HAT attached
python accel_magneto_v2.py
```

## Coordinate Frames

- **Body frame**: Sense HAT coordinate system, boresight along +x axis
- **ENU**: East-North-Up local tangent plane
- **Alt/Az**: Altitude (elevation from horizon), Azimuth (angle from North, clockwise)
- **RA/Dec**: Right Ascension / Declination (ICRS celestial coordinates)

## Key Configuration (in accel_magneto_v2.py)

- `lat_deg`, `long_deg`: Observer site coordinates (currently placeholder 0.0)
- `magn_decl`: Local magnetic declination (currently placeholder 0.0)
- `boresight`: Unit vector defining telescope pointing direction in body frame
