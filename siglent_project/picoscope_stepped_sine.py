#!/usr/bin/env python3
"""
Stepped-sine transfer function measurement via PicoScope 5444D.

Signal chain:
  SDG Ch1 BNC → piezo array
              └─ BNC T-split → PicoScope Ch B  (reference: actual SDG voltage)
  ECM999 → phantom box → SR560 (BNC out) → PicoScope Ch A  (mic response)

H(f) = V_mic(f) / V_SDG(f)  extracted via Hann-windowed FFT at each step freq.

@author: cyrusurheimog
"""

import numpy as np
import socket
import time
import ctypes
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# PicoScope SDK  (pip install picosdk)
from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import adc2mV, assert_pico_ok

# ═══════════════════════════════════════════════════════════════════════════════
# USER-CONFIGURABLE SETTINGS — review before each run
# ═══════════════════════════════════════════════════════════════════════════════

# ── SDG ───────────────────────────────────────────────────────────────────────
SDG_IP        = "10.11.13.230"
SDG_PORT      = 5025
SDG_AMPL_VPP  = 2.0          # Vpp drive to piezo. Keep ≤2V to protect SR560 at ×100.

# ── Frequency sweep ───────────────────────────────────────────────────────────
FREQ_START    = 20_000        # Hz
FREQ_END      = 65_000        # Hz
FREQ_STEP     = 50            # Hz  → 901 frequencies, ~5 min total
SETTLE_MS     = 8             # ms after SDG freq change before capture

# ── PicoScope: resolution and sample rate ─────────────────────────────────────
# 12-bit FlexRes: 4096 ADC levels. With 2 channels at 12-bit: max 250 MS/s.
# Timebase formula (2-channel 12-bit):  t_ns = (timebase - 2) × 4  for tb ≥ 3
# timebase=252 → (252-2)×4 = 1000 ns/sample = 1 MS/s  (1000 samples/ms)
RESOLUTION    = ps.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_12BIT"]
TIMEBASE      = 252           # → 1 μs/sample (1 MS/s)  — 25× oversampling at 40 kHz
N_SAMPLES     = 50_000        # samples per capture = 50 ms window (~2000 cycles at 40 kHz)
                               # Longer window = more coherent FFT integration.
                               # This is the correct noise reduction strategy, not averaging
                               # multiple short captures (which only helps if amplitude
                               # jitter dominates, which it doesn't here).
N_AVG         = 1             # single long capture per frequency — see N_SAMPLES above

# ── PicoScope: Channel A (mic response via SR560 at ×100) ────────────────────
# SR560 at ×100 clips at ±10 mV input → ±1V output.
# ±1V scope range matches the SR560's full output swing exactly — no scope
# clipping at resonance, full ADC range used.
# If signal is well below ±100mV across the sweep (check in PicoScope 6 first),
# switch to PS5000A_200MV for 5× better ADC resolution.
CH_A_RANGE    = ps.PS5000A_RANGE["PS5000A_1V"]       # ±1 V  (matches SR560 ×100 clip)
CH_A_COUPLING = ps.PS5000A_COUPLING["PS5000A_AC"]    # AC: block SR560 DC offset

# ── PicoScope: Channel B (SDG reference via BNC T-split) ──────────────────────
# SDG at 2.0 Vpp → ±1V peak into 50Ω. Into scope's 1MΩ input, actual voltage
# is ≈2× higher (open-circuit).  ±5V range gives ≥2× headroom.
# If you drive higher (e.g. 6Vpp), switch to PS5000A_10V.
CH_B_RANGE    = ps.PS5000A_RANGE["PS5000A_5V"]       # ±5 V
CH_B_COUPLING = ps.PS5000A_COUPLING["PS5000A_DC"]    # DC: preserve waveform shape

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR    = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════════════════════
# SDG control  (same TCP/IP interface as stepped_sine_characterization.py)
# ═══════════════════════════════════════════════════════════════════════════════

class SDG1062X:
    def __init__(self, ip=SDG_IP, port=SDG_PORT, timeout=5.0):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.sock.connect((ip, port))

    def write(self, cmd):
        self.sock.sendall((cmd + '\n').encode())

    def query(self, cmd):
        self.write(cmd); time.sleep(0.05)
        return self.sock.recv(4096).decode().strip()

    def set_sine(self, freq_hz, ampl_vpp, channel=1):
        self.write(f"C{channel}:BSWV WVTP,SINE,FRQ,{freq_hz},AMP,{ampl_vpp},OFST,0")
        time.sleep(0.02)

    def output_on(self, channel=1):
        self.write(f"C{channel}:OUTP ON")

    def output_off(self, channel=1):
        self.write(f"C{channel}:OUTP OFF")

    def close(self):
        self.sock.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PicoScope helpers
# ═══════════════════════════════════════════════════════════════════════════════

def open_picoscope():
    """Open device, set both channels, verify timebase. Returns (handle, dt_ns)."""
    status   = {}
    chandle  = ctypes.c_int16()

    status["open"] = ps.ps5000aOpenUnit(ctypes.byref(chandle), None, RESOLUTION)
    assert_pico_ok(status["open"])

    # Channel A — mic
    chA = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
    status["chA"] = ps.ps5000aSetChannel(chandle, chA, 1,
                                          CH_A_COUPLING, CH_A_RANGE, 0.0)
    assert_pico_ok(status["chA"])

    # Channel B — SDG reference
    chB = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_B"]
    status["chB"] = ps.ps5000aSetChannel(chandle, chB, 1,
                                          CH_B_COUPLING, CH_B_RANGE, 0.0)
    assert_pico_ok(status["chB"])

    # Disable unused channels C and D to avoid spurious captures
    for ch_name in ["PS5000A_CHANNEL_C", "PS5000A_CHANNEL_D"]:
        ch = ps.PS5000A_CHANNEL[ch_name]
        ps.ps5000aSetChannel(chandle, ch, 0,
                             ps.PS5000A_COUPLING["PS5000A_DC"],
                             ps.PS5000A_RANGE["PS5000A_1V"], 0.0)

    # Verify actual sample interval
    max_samples  = ctypes.c_int32(N_SAMPLES)
    dt_ns        = ctypes.c_float()
    status["tb"] = ps.ps5000aGetTimebase2(chandle, TIMEBASE, N_SAMPLES,
                                           ctypes.byref(dt_ns),
                                           ctypes.byref(max_samples), 0)
    assert_pico_ok(status["tb"])
    actual_dt = float(dt_ns.value)
    actual_fs = 1e9 / actual_dt
    print(f"  PicoScope: {actual_dt:.1f} ns/sample = {actual_fs/1e6:.3f} MS/s  "
          f"({N_SAMPLES} samples = {N_SAMPLES*actual_dt/1e6:.2f} ms window)")
    return chandle, actual_dt


def capture_one(chandle):
    """
    Run one triggered block capture and return (buf_A_mV, buf_B_mV) as float arrays.
    Uses auto-trigger (2 ms timeout) so no external trigger signal is needed.
    """
    chA = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
    chB = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_B"]

    # Auto-trigger after 2 ms — sufficient for steady-state sine
    ps.ps5000aSetSimpleTrigger(chandle, 0, chA, 0,
                               ps.PS5000A_THRESHOLD_DIRECTION["PS5000A_NONE"],
                               0, 2)   # 0 = trigger disabled, auto after 2 ms

    # Run block
    ps.ps5000aRunBlock(chandle, 0, N_SAMPLES, TIMEBASE, None, 0, None, None)

    # Wait until ready
    ready = ctypes.c_int16(0)
    while ready.value == 0:
        ps.ps5000aIsReady(chandle, ctypes.byref(ready))

    # Allocate buffers (int16 ADC counts)
    buf_A = (ctypes.c_int16 * N_SAMPLES)()
    buf_B = (ctypes.c_int16 * N_SAMPLES)()
    ps.ps5000aSetDataBuffer(chandle, chA, ctypes.byref(buf_A),
                            N_SAMPLES, 0,
                            ps.PS5000A_RATIO_MODE["PS5000A_RATIO_MODE_NONE"])
    ps.ps5000aSetDataBuffer(chandle, chB, ctypes.byref(buf_B),
                            N_SAMPLES, 0,
                            ps.PS5000A_RATIO_MODE["PS5000A_RATIO_MODE_NONE"])

    overflow = ctypes.c_int16()
    n_ret    = ctypes.c_uint32(N_SAMPLES)
    ps.ps5000aGetValues(chandle, 0, ctypes.byref(n_ret), 1, 0, 0,
                        ctypes.byref(overflow))

    # Convert ADC counts → mV
    maxADC = ctypes.c_int16()
    ps.ps5000aMaximumValue(chandle, ctypes.byref(maxADC))
    mV_A = np.array(adc2mV(buf_A, CH_A_RANGE, maxADC), dtype=np.float32)
    mV_B = np.array(adc2mV(buf_B, CH_B_RANGE, maxADC), dtype=np.float32)

    if overflow.value:
        print("  WARNING: overflow detected — reduce SR560 gain or scope range")

    return mV_A, mV_B


def fft_amplitude_mV(signal_mV, freq_hz, dt_ns):
    """
    Hann-windowed FFT → amplitude (mV peak) at freq_hz.
    Returns (amplitude_mV, snr_db).
    """
    N   = len(signal_mV)
    fs  = 1e9 / dt_ns                       # Hz
    win = np.hanning(N)
    S   = np.fft.rfft(signal_mV * win)
    # Correct for Hann window power loss (factor 2/N, ×2 for single-sided)
    amps = 2.0 * np.abs(S) / (N * 0.5)     # mV peak
    freqs = np.fft.rfftfreq(N, d=1.0/fs)

    # Nearest bin
    k    = int(round(freq_hz * N / fs))
    k    = max(1, min(k, len(amps)-2))
    amp  = float(amps[k])

    # SNR: peak vs RMS of noise floor (exclude ±5 bins around peak)
    mask = np.ones(len(amps), bool)
    mask[max(0,k-5):k+6] = False
    noise_rms = float(np.sqrt(np.mean(amps[mask]**2)))
    snr_db    = 20*np.log10(amp / (noise_rms + 1e-12))

    return amp, snr_db


# ═══════════════════════════════════════════════════════════════════════════════
# Main sweep
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    freqs = np.arange(FREQ_START, FREQ_END + FREQ_STEP, FREQ_STEP, dtype=float)
    N_f   = len(freqs)
    amp_A = np.zeros(N_f)     # mic amplitude mV (averaged over N_AVG)
    amp_B = np.zeros(N_f)     # SDG reference amplitude mV
    snr_A = np.zeros(N_f)

    print(f"Stepped-sine sweep: {FREQ_START/1000:.0f}–{FREQ_END/1000:.0f} kHz, "
          f"{FREQ_STEP} Hz steps  ({N_f} points)")
    print(f"PicoScope: 12-bit, Ch A ±{_range_label(CH_A_RANGE)} AC, "
          f"Ch B ±{_range_label(CH_B_RANGE)} DC")
    print(f"SDG: {SDG_AMPL_VPP} Vpp sine  |  {N_AVG} captures/freq\n")

    # Open instruments
    print("Connecting to SDG...")
    sdg = SDG1062X()
    print(f"  SDG: {sdg.query('*IDN?')[:50]}")

    print("Opening PicoScope...")
    chandle, dt_ns = open_picoscope()

    sdg.set_sine(FREQ_START, SDG_AMPL_VPP)
    sdg.output_on()
    time.sleep(0.2)   # let piezo settle on first frequency

    try:
        for i, f in enumerate(freqs):
            sdg.set_sine(f, SDG_AMPL_VPP)
            time.sleep(SETTLE_MS / 1000.0)

            a_runs, b_runs, snr_runs = [], [], []
            for _ in range(N_AVG):
                mV_A, mV_B = capture_one(chandle)
                a, snr = fft_amplitude_mV(mV_A, f, dt_ns)
                b, _   = fft_amplitude_mV(mV_B, f, dt_ns)
                a_runs.append(a); b_runs.append(b); snr_runs.append(snr)

            amp_A[i] = np.mean(a_runs)
            amp_B[i] = np.mean(b_runs)
            snr_A[i] = np.mean(snr_runs)

            if i % 50 == 0 or i == N_f-1:
                H_dB = 20*np.log10(amp_A[i]/(amp_B[i]+1e-12))
                print(f"  {f/1000:6.2f} kHz  V_mic={amp_A[i]:.3f}mV  "
                      f"V_SDG={amp_B[i]:.1f}mV  H={H_dB:.1f}dB  SNR={snr_A[i]:.1f}dB",
                      flush=True)

    finally:
        sdg.output_off(); sdg.close()
        ps.ps5000aStop(chandle); ps.ps5000aCloseUnit(chandle)

    # ── Compute H(f) ──────────────────────────────────────────────────────────
    H_mag    = amp_A / (amp_B + 1e-12)
    H_mag_dB = 20 * np.log10(H_mag + 1e-12)
    # Normalise so peak = 0 dB
    peak_dB  = np.max(H_mag_dB[freqs > 25_000])   # ignore sub-25kHz
    H_norm   = H_mag_dB - peak_dB

    # Savitzky-Golay smooth
    H_smooth = savgol_filter(H_norm, window_length=21, polyorder=3)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('ECM999 + SR560 Transfer Function (PicoScope 5444D stepped-sine)',
                 fontsize=12, fontweight='bold')

    # Left: full sweep
    ax = axes[0]
    ax.plot(freqs/1000, H_norm,   'b-', lw=0.5, alpha=0.35, label='Raw')
    ax.plot(freqs/1000, H_smooth, 'b-', lw=2.0, label='Smoothed')
    ax.axhline(-3,  color='C3',     lw=1.5, ls='--', label='-3 dB')
    ax.axhline(-6,  color='orange', lw=1.2, ls='--', label='-6 dB')
    ax.axvline(40,  color='C2',     lw=1.2, ls=':',  label='40 kHz (BATPING)')
    ax.set_xlabel('Frequency (kHz)'); ax.set_ylabel('H(f) (dB, norm. to peak)')
    ax.set_title('Full sweep'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlim(FREQ_START/1000, FREQ_END/1000)

    # Right: SNR
    ax = axes[1]
    ax.plot(freqs/1000, snr_A, 'g-', lw=0.8)
    ax.axhline(10, color='C3', lw=1.5, ls='--', label='10 dB floor')
    ax.axvline(40, color='C2', lw=1.2, ls=':')
    ax.set_xlabel('Frequency (kHz)'); ax.set_ylabel('SNR (dB)')
    ax.set_title('SNR on Ch A (mic)'); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(FREQ_START/1000, FREQ_END/1000)

    plt.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, 'ecm999_transfer_function.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot: {out_png}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_npz = os.path.join(OUTPUT_DIR, 'ecm999_transfer_function.npz')
    np.savez(out_npz, freqs=freqs, amp_mic_mV=amp_A, amp_sdg_mV=amp_B,
             H_mag=H_mag, H_mag_dB=H_mag_dB, H_norm=H_norm, H_smooth=H_smooth,
             snr_dB=snr_A, SDG_AMPL_VPP=SDG_AMPL_VPP, TIMEBASE=TIMEBASE,
             CH_A_RANGE=str(CH_A_RANGE), N_AVG=N_AVG)
    print(f"Data: {out_npz}")


def _range_label(r):
    """Return human-readable voltage range string."""
    labels = {0:'10mV', 1:'20mV', 2:'50mV', 3:'100mV', 4:'200mV',
              5:'500mV', 6:'1V', 7:'2V', 8:'5V', 9:'10V', 10:'20V'}
    return labels.get(r, str(r))


if __name__ == '__main__':
    main()
