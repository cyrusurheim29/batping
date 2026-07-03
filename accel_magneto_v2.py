#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 00:39:43 2026

@author: cyrusurheimog
"""

from sense_hat import SenseHat
import time, math
import numpy as np

from astropy.coordinates import SkyCoord, AltAz, EarthLocation
from astropy.time import Time
import astropy.units as u

sense = SenseHat()

# coordinates at site
lat_deg = 0.0
long_deg = 0.0


# Magnetic declination at site
magn_decl = 0.0


# telescope points along sensehat +x axis: [1,0,0]
boresight = np.array([1.0, 0.0, 0.0])

# Sampling / smoothing
dt = 0.1         # seconds


def wrap360(deg):
    return deg % 360.0

def normalize(v):
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

def Rx(phi):
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def Ry(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def Rz(psi):
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])




def accel_to_roll_pitch(ax, ay, az):

    # normalize (optional but good)
    g = normalize(np.array([ax, ay, az]))
    gx, gy, gz = g

    roll  = math.atan2(gy, gz)
    pitch = math.atan2(-gx, math.sqrt(gy*gy + gz*gz))
    return roll, pitch


def tilt_compensated_heading(mx, my, mz, roll, pitch):

    # normalize mag
    m = normalize(np.array([mx, my, mz]))
    mx, my, mz = m

    # tilt compensation: rotate mag from BODY into a "leveled" frame
    # using inverse of roll/pitch rotation:
    R_tilt = Ry(pitch) @ Rx(roll)
    mh = R_tilt @ np.array([mx, my, mz])

    # MIGHT need to flip mh[0], mh[1], so long as:
    # heading = atan2(East, North)
    heading = math.atan2(mh[0], mh[1])
    return heading



def boresight_to_altaz(boresight_body, roll, pitch, yaw_true):

    v_body = normalize(boresight_body)

    # body frame -> ENU (approx)
    R = Rz(yaw_true) @ Ry(pitch) @ Rx(roll)
    v_enu = normalize(R @ v_body)

    e, n, u = v_enu[0], v_enu[1], v_enu[2]

    alt = math.asin(u)
    az = math.atan2(e, n)  # 0=N, 90=E

    alt_deg = math.degrees(alt)
    az_deg = wrap360(math.degrees(az))
    return alt_deg, az_deg


def altaz_to_radec_api(alt_deg, az_deg, site, obstime):
    altaz = AltAz(
        alt=alt_deg * u.deg,
        az=az_deg * u.deg,
        obstime=obstime,
        location=site
    )
    sky = SkyCoord(altaz).icrs
    return sky.ra, sky.dec


def main():
    site = EarthLocation(lat=lat_deg*u.deg, lon=long_deg*u.deg)

    while True:
        a = sense.get_accelerometer_raw()
        m = sense.get_compass_raw()

        ax, ay, az = a["x"], a["y"], a["z"]
        mx, my, mz = m["x"], m["y"], m["z"]

        # roll/pitch from accel
        roll, pitch = accel_to_roll_pitch(ax, ay, az)


        # yaw (magnetic) from tilt-comp mag
        yaw_mag = tilt_compensated_heading(mx, my, mz, roll, pitch)

        # convert to true yaw by adding declination
        yaw_true = yaw_mag + math.radians(magn_decl)

        # boresight -> Alt/Az
        alt_deg, az_deg = boresight_to_altaz(boresight, roll, pitch, yaw_true)

        # Alt/Az -> RA/Dec
        t = Time.now()
        ra, dec = altaz_to_radec_api(alt_deg, az_deg, site, t)

        print(
            f"Alt={alt_deg:7.2f} deg  Az={az_deg:7.2f} deg  |  "
            f"RA={ra.to_string(unit=u.hour, sep=':', precision=1)}  "
            f"Dec={dec.to_string(unit=u.deg, sep=':', precision=1)}"
        )

        time.sleep(dt)


if __name__ == "__main__":
    main()
