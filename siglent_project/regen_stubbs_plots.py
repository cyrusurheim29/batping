#!/usr/bin/env python3
"""Regenerate all Stubbs PNGs with time (seconds) x-axis on the Δτ panel."""
import os, sys, glob
sys.path.insert(0, os.path.dirname(__file__))
import chirp_stubbs_analysis as st

BASE = os.path.dirname(__file__)
SDG_SR = 400000; SDG_MAX = 16384

def rep_rate(dur_ms, dead_ms=15.0):
    buf = min(int((dur_ms + dead_ms) / 1000 * SDG_SR), SDG_MAX)
    return SDG_SR / buf

# ── Jul 8 alpha sweep (10s recordings) ────────────────────────────────────────
ALPHA_DIR  = os.path.join(BASE, 'recordings/20260708_133617_alpha_sweep')
ALPHA_WAVS = sorted(glob.glob(os.path.join(ALPHA_DIR, 'raw/*_received.wav')))

# bw_6kHz_flat uses dead_ms=23; all others dead_ms=15
def alpha_rr(name):
    dur = 4.0 if '4ms' in name else 2.0
    return rep_rate(dur, dead_ms=15.0)

# ── Jul 10 chirp sweep (30s recordings) ───────────────────────────────────────
CHIRP_DIR  = os.path.join(BASE, 'recordings/20260710_030748_chirp_sweep')
CHIRP_WAVS = sorted(glob.glob(os.path.join(CHIRP_DIR, 'raw/*_received.wav')))

# Dead-time map for chirp sweep waveforms
DEAD_MAP = {'bw_6kHz': 23.0}   # bw_6kHz_flat uses 23ms dead; rest use 15ms

def chirp_rr(name):
    dur = 4.0 if '4ms' in name else 2.0
    dead = next((v for k, v in DEAD_MAP.items() if k in name), 15.0)
    return rep_rate(dur, dead_ms=dead)

# ── Jul 10 alpha sweep (30s recordings) ───────────────────────────────────────
ALPHA30_DIR  = os.path.join(BASE, 'recordings/20260710_194120_alpha_sweep') \
               if os.path.isdir(os.path.join(BASE, 'recordings/20260710_194120_alpha_sweep')) \
               else None
# Check for any other alpha sweep dirs on Jul10
import re
alpha30_dirs = [d for d in glob.glob(os.path.join(BASE, 'recordings/20260710_*'))
                if os.path.isdir(d) and 'alpha' in d.lower() and 'longterm' not in d]

all_jobs = []

# Jul 8 alpha sweep
for wav in ALPHA_WAVS:
    name = os.path.basename(wav).replace('_received.wav', '')
    rr = alpha_rr(name)
    out_dir = os.path.join(ALPHA_DIR, 'plots', name)
    os.makedirs(out_dir, exist_ok=True)
    all_jobs.append((wav, rr, out_dir, name))

# Jul 10 chirp sweep
for wav in CHIRP_WAVS:
    name = os.path.basename(wav).replace('_received.wav', '')
    rr = chirp_rr(name)
    out_dir = os.path.join(CHIRP_DIR, 'plots', name)
    os.makedirs(out_dir, exist_ok=True)
    all_jobs.append((wav, rr, out_dir, name))

# Jul 10 alpha30 dirs
for d in alpha30_dirs:
    for wav in sorted(glob.glob(os.path.join(d, 'raw/*_received.wav'))):
        name = os.path.basename(wav).replace('_received.wav', '')
        rr = alpha_rr(name)
        out_dir = os.path.join(d, 'plots', name)
        os.makedirs(out_dir, exist_ok=True)
        all_jobs.append((wav, rr, out_dir, name))

print(f"Total jobs: {len(all_jobs)}")
for i, (wav, rr, out_dir, name) in enumerate(all_jobs):
    print(f"\n[{i+1}/{len(all_jobs)}] {name}  rep_rate={rr:.2f} Hz")
    try:
        st.analyze(wav, rep_rate=rr, output_dir=out_dir)
    except Exception as e:
        print(f"  ERROR: {e}")
        continue

print("\nDone.")
