#!/usr/bin/env python3
"""Fast chain transfer function H(f) characterization for the T-connector setup.

Wraps ecm999_stepped_sine.py:
  - Narrows the sweep to 30–50 kHz (covers the 37–43 kHz chirp with margin)
  - Coarser 100 Hz step (~200 points) → ~2 min total
  - Ch B range set to ±10V (accepts raw SDG drive via T-connector)
  - Output goes to a timestamped NPZ named chain_tconnector_YYYYMMDD_HHMMSS.npz
    with the same keys ('freqs', 'H_mag') consumed by upload_variable_chirp.py

Signal chain (current):
    SDG CH1 → T-connector → (a) power amp ×10 → piezo array → air → corner cube → air
                           → ECM999 mic → preamp G=100 → PicoScope Ch A
                        (b) PicoScope Ch B   (trigger reference, ±10V FS)

Usage:
    DYLD_LIBRARY_PATH=/Library/Frameworks/PicoSDK.framework/Libraries/libps5000a \\
    python3 characterize_chain_tconnector.py
"""
import os
# Monkey-patch module constants BEFORE calling main().
# Python resolves module-level globals at call time, so overriding attributes
# here reaches the capture + sweep loop inside ecm999_stepped_sine.main().
import ecm999_stepped_sine as _css
from picosdk.ps5000a import ps5000a as _ps
from datetime import datetime

# ── Sweep range and step (narrow + coarse → ~2 min) ──────────────────────
_css.FREQ_START = 30_000
_css.FREQ_END   = 50_000
_css.FREQ_STEP  = 100

# ── PicoScope ranges for the current chain ──────────────────────────────
# Ch B now sees the raw 10 Vpp SDG drive via T-connector → use ±10V FS
_css.CH_B_RANGE = _ps.PS5000A_RANGE["PS5000A_10V"]
# Ch A: preamp G=100 with ×10 power amp gives a strong signal → ±5V FS
_css.CH_A_RANGE = _ps.PS5000A_RANGE["PS5000A_5V"]

# ── Drive amplitude: ×10 power amp in chain → stay conservative ──────────
# SDG 0.3 Vpp → power amp ×10 → 3 Vpp into piezo (same order as previous
# characterization that used SDG 3 Vpp with no power amp).  Adjust if SNR
# weak or Ch A overflows.
_css.BURST_AMPL_VPP = 0.3

# Rename the output NPZ so it's identifiable as the T-connector characterization
_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
_out_base = os.path.join(_css.OUTPUT_DIR, f'chain_tconnector_{_stamp}')
_css.datetime = type('_FD', (), {'now': staticmethod(lambda: type('_T', (), {'strftime': staticmethod(lambda _: _stamp)})())})
# Also patch the base filename constructor by replacing the global datetime
# used inside main():
#   stamp = datetime.now().strftime(...)
#   out_base = os.path.join(OUTPUT_DIR, f'ecm999_stepped_sine_burst8ms_{stamp}')
# We can't rename the prefix via patching easily; instead, run main() then
# rename the resulting files.
if __name__ == '__main__':
    import glob
    print(f"Fast chain characterization — target file: {_out_base}.npz")
    _css.main()
    # Rename the latest output files to the new base name
    for ext in ('.npz', '.png'):
        cands = sorted(glob.glob(os.path.join(_css.OUTPUT_DIR,
                                               f'ecm999_stepped_sine_burst8ms_*{ext}')),
                        key=os.path.getmtime)
        if cands:
            latest = cands[-1]
            dst = _out_base + ext
            os.rename(latest, dst)
            print(f"  renamed → {dst}")
