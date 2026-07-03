#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Relative Alt/Az Tracker (Accelerometer-Only)

Tracks orientation changes relative to boot position using only accelerometer data.
Designed for lab/dome environments where absolute orientation is not needed.

IMPORTANT: Device must be tilted at least 45° from vertical at boot for reliable
azimuth tracking. Optimal tilt is ~60°.

Degeneracy: When boresight is parallel to gravity (pointing straight up or down),
azimuth becomes unobservable. The code warns when approaching this singularity.
"""

from sense_hat import SenseHat
import numpy as np
import math
import time

sense = SenseHat()

# Configuration
SAMPLE_RATE = 0.1  # seconds between samples
MIN_TILT_DEG = 30  # minimum safe tilt angle from vertical (warn below this)
WARN_TILT_DEG = 45  # tilt angle at which azimuth tracking degrades significantly


def normalize(v):
    """Normalize a vector to unit length."""
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def wrap_angle(deg):
    """Wrap angle to [-180, 180] range."""
    while deg > 180:
        deg -= 360
    while deg < -180:
        deg += 360
    return deg


class RelativeTracker:
    """
    Tracks relative Alt/Az changes from boot position using accelerometer only.

    The accelerometer measures the gravity vector in body frame. By comparing
    the current gravity vector to the boot reference, we can compute how the
    device has rotated.

    Degeneracy: When the boresight (device +x axis) points parallel to gravity,
    the azimuth component becomes unobservable. This occurs at tilt angles
    near 0° or 180° from vertical.
    """

    def __init__(self):
        self.a_boot = None  # Reference gravity vector at boot
        self.tilt_at_boot = None  # Tilt angle from vertical at boot
        self.calibrated = False

    def calibrate(self):
        """
        Store the current accelerometer reading as the boot reference.
        Call this once at startup with the device in its initial position.
        """
        a = sense.get_accelerometer_raw()
        self.a_boot = normalize(np.array([a["x"], a["y"], a["z"]]))

        # Compute tilt angle from vertical (angle between gravity and +z body axis)
        # Assuming +z is "up" in body frame when level
        self.tilt_at_boot = math.degrees(math.acos(abs(self.a_boot[2])))

        self.calibrated = True

        # Check if tilt is sufficient for reliable azimuth tracking
        if self.tilt_at_boot < MIN_TILT_DEG:
            print(f"WARNING: Device tilt ({self.tilt_at_boot:.1f}°) is below minimum "
                  f"({MIN_TILT_DEG}°). Azimuth tracking will be unreliable!")
            print("Please tilt device to at least 45° from vertical.")
        elif self.tilt_at_boot < WARN_TILT_DEG:
            print(f"CAUTION: Device tilt ({self.tilt_at_boot:.1f}°) is marginal. "
                  f"Consider tilting to ~60° for best azimuth sensitivity.")
        else:
            print(f"Calibrated at tilt = {self.tilt_at_boot:.1f}° from vertical (good)")

        print(f"Boot reference: ax={self.a_boot[0]:.4f}, ay={self.a_boot[1]:.4f}, "
              f"az={self.a_boot[2]:.4f}")

        return self.tilt_at_boot

    def get_relative_orientation(self):
        """
        Compute current Alt/Az offset from boot position.

        Returns:
            delta_alt: Altitude change in degrees (positive = tilted up)
            delta_az: Azimuth change in degrees (positive = rotated clockwise when viewed from above)
            tilt_from_vertical: Current tilt angle from vertical (for degeneracy warning)
        """
        if not self.calibrated:
            raise RuntimeError("Must call calibrate() before get_relative_orientation()")

        # Read current accelerometer
        a = sense.get_accelerometer_raw()
        a_now = normalize(np.array([a["x"], a["y"], a["z"]]))

        # Current tilt from vertical
        tilt_now = math.degrees(math.acos(abs(a_now[2])))

        # Compute azimuth change using atan2 of x,y components
        # This tracks rotation around the z-axis (vertical when level)
        az_boot = math.atan2(self.a_boot[1], self.a_boot[0])
        az_now = math.atan2(a_now[1], a_now[0])
        delta_az = math.degrees(az_now - az_boot)
        delta_az = wrap_angle(delta_az)

        # Compute altitude change using z component
        # asin gives the angle from the horizontal plane
        alt_boot = math.asin(self.a_boot[2])
        alt_now = math.asin(a_now[2])
        delta_alt = math.degrees(alt_now - alt_boot)

        return delta_alt, delta_az, tilt_now

    def check_degeneracy(self, tilt_from_vertical):
        """Check if we're approaching the singularity and return warning level."""
        if tilt_from_vertical < MIN_TILT_DEG:
            return "CRITICAL"
        elif tilt_from_vertical < WARN_TILT_DEG:
            return "WARNING"
        return "OK"


def main():
    print("=" * 60)
    print("Relative Alt/Az Tracker (Accelerometer-Only)")
    print("=" * 60)
    print()
    print("This tracker measures orientation changes RELATIVE to boot position.")
    print("For reliable azimuth tracking, tilt the device ~60° from vertical.")
    print()
    print("Press Ctrl+C to exit")
    print()

    tracker = RelativeTracker()

    print("Calibrating in 3 seconds... hold device steady in initial position")
    time.sleep(3)

    tracker.calibrate()
    print()
    print("-" * 60)
    print(f"{'Δ Alt':>10}  {'Δ Az':>10}  {'Tilt':>8}  {'Status':>10}")
    print("-" * 60)

    try:
        while True:
            delta_alt, delta_az, tilt = tracker.get_relative_orientation()
            status = tracker.check_degeneracy(tilt)

            # Format status with indicator
            if status == "CRITICAL":
                status_str = "!! DEGEN !!"
            elif status == "WARNING":
                status_str = "* warn *"
            else:
                status_str = "OK"

            print(f"{delta_alt:>+10.2f}° {delta_az:>+10.2f}° {tilt:>8.1f}° {status_str:>10}")

            time.sleep(SAMPLE_RATE)

    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
