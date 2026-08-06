#!/usr/bin/env python3
"""
Burst stepped-sine transfer function measurement for ECM999 + SR560 via PicoScope 5444D.

Mirrors stepped_sine_characterization.py exactly (same burst waveform, same analysis)
but replaces PyAudio recording with a PicoScope block capture so it works with the
ECM999 → phantom box → SR560 (×100) → PicoScope Ch A signal chain.

Signal chain:
  SDG Ch1 → piezo array → air → reflector → air
  → ECM999 → phantom box → SR560 (×100) → BNC → PicoScope Ch A

Usage:
  DYLD_LIBRARY_PATH=/Library/Frameworks/PicoSDK.framework/Libraries/libps5000a \
  python ecm999_stepped_sine.py
"""

import numpy as np
import socket, time, ctypes, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import signal as scipy_signal
from scipy.signal import hilbert, savgol_filter

from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import adc2mV, assert_pico_ok

# ═══════════════════════════════════════════════════════════════════════════════
# Settings — match stepped_sine_characterization.py
# ═══════════════════════════════════════════════════════════════════════════════

SDG_IP   = "10.11.13.230"
SDG_PORT = 5025

FREQ_START        = 20_000    # Hz
FREQ_END          = 65_000    # Hz
FREQ_STEP         = 50        # Hz  → 901 pts
SDG_SAMPLE_RATE   = 400_000   # Sa/s
SDG_MAX_SAMPLES   = 16_384
AMPLITUDE         = 0.9

BURST_DURATION_MS = 8.0       # ms — same as stepped_sine_characterization.py
BURST_AMPL_VPP    = 0.5       # Vpp starting point — SR560 ×100 clips at ±10mV input
                               # → ±1V output; 0.5 Vpp is conservative, increase if
                               # signal is weak (no PicoScope overflow warning = OK)
BURST_SETTLE_FRAC = 0.5       # ignore first half of burst (ring-up)

# PicoScope
RESOLUTION    = ps.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_12BIT"]
TIMEBASE      = 64             # → 1.025 MSa/s (verified by ps5000aGetTimebase2,
                               # PicoScope 5444D, 1-channel 12-bit).  Was 165 → 384 kHz.
CH_A_RANGE    = ps.PS5000A_RANGE["PS5000A_2V"]      # ±2V  (SR560 output ≤ ±1V)
CH_A_COUPLING = ps.PS5000A_COUPLING["PS5000A_AC"]   # block SR560 DC offset

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════════════════════
# SDG control
# ═══════════════════════════════════════════════════════════════════════════════

class SDG1062X_LAN:
    def __init__(self, ip=SDG_IP, port=SDG_PORT, timeout=30.0):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.sock.connect((ip, port))

    def write(self, cmd):
        self.sock.sendall((cmd + '\n').encode('utf-8'))

    def write_raw(self, data):
        # Temporarily block without timeout — WVDT binary upload can be slow
        # and the SDG's TCP receive window fills up at high upload rates.
        self.sock.settimeout(None)
        try:
            self.sock.sendall(data)
        finally:
            self.sock.settimeout(30.0)

    def read(self, bufsize=4096):
        return self.sock.recv(bufsize).decode('utf-8').strip()

    def query(self, cmd):
        self.write(cmd); time.sleep(0.1)
        return self.read()

    def upload_waveform(self, name, wave_data, freq_hz, ampl_vpp=2.0, channel=1):
        wave_int16 = (np.clip(wave_data, -1.0, 1.0) * 32767).astype(np.int16)
        bin_data   = wave_int16.astype('<i2').tobytes()
        prefix     = (f"C{channel}:WVDT WVNM,{name},FREQ,{freq_hz},"
                      f"AMPL,{ampl_vpp},OFST,0,PHASE,0,WAVEDATA,").encode('utf-8')
        self.write_raw(prefix + bin_data + b'\n')
        time.sleep(0.5)
        self.write(f"C{channel}:ARWV NAME,{name}")
        time.sleep(0.2)

    def set_sine(self, freq_hz, ampl_vpp, channel=1):
        """Built-in BSWV sine — no waveform upload, no Flash writes, no timeouts."""
        self.write(f"C{channel}:BSWV WVTP,SINE,FRQ,{freq_hz},AMP,{ampl_vpp},OFST,0")
        time.sleep(0.01)

    def set_burst(self, freq_hz, ampl_vpp, burst_ms=8.0, dead_ms=20.0, channel=1):
        """N-cycle burst using SDG built-in BTWV — no ARB upload, no Flash writes.

        Blocks until the SDG confirms the new frequency has been applied by
        querying C1:BSWV? after writing — this prevents the SDG from lagging
        behind the Python loop when sweeping 900+ frequencies.
        """
        n_cyc  = max(1, round(freq_hz * burst_ms / 1000.0))
        period = (burst_ms + dead_ms) / 1000.0
        self.write(f"C{channel}:BSWV WVTP,SINE,FRQ,{freq_hz},AMP,{ampl_vpp},OFST,0")
        self.write(f"C{channel}:BTWV STATE,ON,PRD,{period:.4f},STPS,0,"
                   f"TRSR,INT,DLAY,0,TIME,{n_cyc},GATE_NCYC,NCY")
        # Block until SDG has processed both commands — query forces in-order execution
        self.query(f"C{channel}:BSWV?")

    def burst_off(self, channel=1):
        """Return channel to continuous sine after burst sweep."""
        self.write(f"C{channel}:BTWV STATE,OFF")
        time.sleep(0.05)

    def output_on(self,  channel=1): self.write(f"C{channel}:OUTP ON")
    def output_off(self, channel=1): self.write(f"C{channel}:OUTP OFF")
    def disconnect(self):            self.sock.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PicoScope helpers
# ═══════════════════════════════════════════════════════════════════════════════

CH_B_RANGE    = ps.PS5000A_RANGE["PS5000A_1V"]   # SDG output ≤ 500 mVpp = 250 mV peak

def open_picoscope():
    """Open device, enable Ch A (mic) + Ch B (SDG reference), return (chandle, actual_fs_hz).

    Ch B sees the SDG burst via the T-splitter and is used as the hardware trigger.
    This gives exact burst-start synchronisation with zero acoustic delay.
    """
    status  = {}
    chandle = ctypes.c_int16()
    status["open"] = ps.ps5000aOpenUnit(ctypes.byref(chandle), None, RESOLUTION)
    assert_pico_ok(status["open"])

    chA = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
    status["chA"] = ps.ps5000aSetChannel(chandle, chA, 1,
                                          CH_A_COUPLING, CH_A_RANGE, 0.0)
    assert_pico_ok(status["chA"])

    # Ch B: SDG output via T-splitter — DC coupled, ±1V range
    chB = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_B"]
    status["chB"] = ps.ps5000aSetChannel(chandle, chB, 1,
                                          ps.PS5000A_COUPLING["PS5000A_DC"],
                                          CH_B_RANGE, 0.0)
    assert_pico_ok(status["chB"])

    for ch_name in ["PS5000A_CHANNEL_C", "PS5000A_CHANNEL_D"]:
        ps.ps5000aSetChannel(chandle, ps.PS5000A_CHANNEL[ch_name], 0,
                             ps.PS5000A_COUPLING["PS5000A_DC"],
                             ps.PS5000A_RANGE["PS5000A_1V"], 0.0)

    max_s  = ctypes.c_int32(1)
    dt_ns  = ctypes.c_float()
    status["tb"] = ps.ps5000aGetTimebase2(chandle, TIMEBASE, 1,
                                           ctypes.byref(dt_ns),
                                           ctypes.byref(max_s), 0)
    assert_pico_ok(status["tb"])
    actual_fs = 1e9 / float(dt_ns.value)
    print(f"  PicoScope: {float(dt_ns.value):.1f} ns/sample = {actual_fs/1000:.1f} kHz")
    return chandle, actual_fs


def capture_block(chandle, n_samples, trig_adc_chB):
    """
    Block capture of n_samples from Ch A, triggered on Ch B rising edge.

    Ch B carries the SDG burst (via T-splitter) so the trigger fires exactly
    at burst onset — no acoustic delay, no phase ambiguity.
    Returns Ch A as float64 normalised to ±1 FS (±2V → divide by 2000 mV).
    """
    chA = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
    chB = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_B"]
    ps.ps5000aSetSimpleTrigger(
        chandle, 1, chB,
        trig_adc_chB,
        ps.PS5000A_THRESHOLD_DIRECTION["PS5000A_RISING"],
        0, 500)   # auto-trigger fallback after 500 ms
    ps.ps5000aRunBlock(chandle, 0, n_samples, TIMEBASE, None, 0, None, None)

    ready = ctypes.c_int16(0)
    while ready.value == 0:
        time.sleep(0.005)
        ps.ps5000aIsReady(chandle, ctypes.byref(ready))

    buf = (ctypes.c_int16 * n_samples)()
    ps.ps5000aSetDataBuffer(chandle, chA, ctypes.byref(buf),
                            n_samples, 0,
                            ps.PS5000A_RATIO_MODE["PS5000A_RATIO_MODE_NONE"])
    overflow = ctypes.c_int16()
    n_ret    = ctypes.c_uint32(n_samples)
    ps.ps5000aGetValues(chandle, 0, ctypes.byref(n_ret), 1, 0, 0,
                        ctypes.byref(overflow))

    if overflow.value:
        print("  WARNING: Ch A overflow — reduce BURST_AMPL_VPP or SR560 gain")

    maxADC = ctypes.c_int16()
    ps.ps5000aMaximumValue(chandle, ctypes.byref(maxADC))
    mV = np.array(adc2mV(buf, CH_A_RANGE, maxADC), dtype=np.float32)
    return (mV / 2000.0).astype(np.float64)   # normalise to ±1 FS


# ═══════════════════════════════════════════════════════════════════════════════
# Main sweep
# ═══════════════════════════════════════════════════════════════════════════════

BURST_MS          = 8.0    # ms burst duration at each frequency step
DEAD_MS           = 20.0   # ms dead time after burst
SETTLE_MS         = 20     # ms to wait after frequency change before capturing
BURST_SETTLE_FRAC = 0.5    # ignore first half of burst (ring-up transient)
# Capture window = BURST_MS + DEAD_MS + BURST_MS = 36 ms.
# This guarantees one complete burst is always present regardless of the phase
# offset between the SDG burst clock and the PicoScope auto-trigger.
# Burst is located at runtime via Hilbert envelope peak detection.
CAPTURE_MS        = DEAD_MS + 2 * BURST_MS   # 36 ms


def fft_amplitude_mV(signal_mV, freq_hz, actual_fs):
    """Hann-windowed FFT → (amplitude_mV_peak, snr_db). Mirrors picoscope_stepped_sine.py."""
    N    = len(signal_mV)
    win  = np.hanning(N)
    S    = np.fft.rfft(signal_mV * win)
    amps = 2.0 * np.abs(S) / (N * 0.5)
    freqs_f = np.fft.rfftfreq(N, d=1.0/actual_fs)
    k    = int(round(freq_hz * N / actual_fs))
    k    = max(1, min(k, len(amps)-2))
    amp  = float(amps[k])
    mask = np.ones(len(amps), bool); mask[max(0,k-5):k+6] = False
    noise_rms = float(np.sqrt(np.mean(amps[mask]**2)))
    snr_db    = 20*np.log10(amp / (noise_rms + 1e-12))
    return amp, snr_db


def find_burst(signal, burst_samples, settle_frac=0.5):
    """Locate the burst in the capture via Hilbert envelope, return analysis slice.

    Returns (sig_slice, burst_found) where sig_slice is the second half of the
    burst (after ring-up).  Returns (None, False) if no burst is found.
    """
    env = np.abs(hilbert(signal))
    # Smooth envelope coarsely to avoid getting fooled by carrier oscillations
    win = max(1, burst_samples // 20)
    env_smooth = np.convolve(env, np.ones(win)/win, mode='same')
    threshold  = 0.25 * env_smooth.max()

    above = env_smooth > threshold
    if not above.any():
        return None, False

    burst_start = int(np.argmax(above))
    settle_samp = int(settle_frac * burst_samples)
    i_start     = burst_start + settle_samp
    i_end       = burst_start + burst_samples
    if i_end > len(signal) or (i_end - i_start) < burst_samples // 4:
        return None, False   # burst truncated at edge of capture

    return signal[i_start:i_end], True


def main():
    freq_list = np.arange(FREQ_START, FREQ_END + FREQ_STEP, FREQ_STEP)
    est_min   = len(freq_list) * (SETTLE_MS + CAPTURE_MS + 100 + 5) / 1000 / 60  # +100ms for blocking query

    print(f"ECM999 burst stepped-sine: {FREQ_START/1000:.0f}–{FREQ_END/1000:.0f} kHz, "
          f"{FREQ_STEP} Hz steps ({len(freq_list)} pts)")
    print(f"BTWV N-cycle burst  |  {BURST_AMPL_VPP} Vpp  |  {BURST_MS:.0f} ms burst  "
          f"|  {CAPTURE_MS:.0f} ms capture  |  envelope burst-find")
    print(f"Estimated time: ~{est_min:.0f} min\n")
    print("Ensure: phantom box ON, SR560 gain ×100, BNC → Ch A\n")

    print("Connecting to SDG...")
    sdg = SDG1062X_LAN()
    print(f"  SDG: {sdg.query('*IDN?')[:60]}")

    print("Opening PicoScope...")
    chandle, actual_fs = open_picoscope()
    n_capture     = int(round(actual_fs * CAPTURE_MS / 1000.0))
    burst_samples = int(round(actual_fs * BURST_MS   / 1000.0))

    # Acoustic TOF offset: trigger fires at burst electrical onset (t=0), but
    # sound takes ~5.88 ms to travel 1 m to reflector and back to microphone.
    # Analysis window is offset past this delay, then skips the first half of
    # the burst (ring-up) and analyses the steady-state second half.
    TOF_MS    = 5.88   # ms — 2 m round trip at 340 m/s; adjust if path differs
    tof_samp  = int(round(actual_fs * TOF_MS / 1000.0))
    i_start   = tof_samp + int(BURST_SETTLE_FRAC * burst_samples)
    i_end     = tof_samp + burst_samples
    print(f"  TOF offset: {TOF_MS:.2f} ms = {tof_samp} samples")
    print(f"  Analysis window: [{i_start/actual_fs*1000:.1f}, {i_end/actual_fs*1000:.1f}] ms from trigger")

    # Ch B trigger threshold: 100 mV — well above noise, well below 250 mV peak
    maxADC   = ctypes.c_int16()
    ps.ps5000aMaximumValue(chandle, ctypes.byref(maxADC))
    TRIG_MV  = 100.0   # mV on Ch B (SDG output, 500 mVpp → 250 mV peak)
    trig_adc = int(TRIG_MV / 1000.0 * maxADC.value)   # Ch B range is ±1V
    print(f"  Ch B trigger: rising edge at {TRIG_MV:.0f} mV → {trig_adc} ADC counts")

    amp_A    = np.zeros(len(freq_list))
    snr_A    = np.zeros(len(freq_list))
    n_missed = 0

    sdg.set_burst(FREQ_START, BURST_AMPL_VPP, BURST_MS, DEAD_MS)
    sdg.output_on()
    time.sleep(0.5)

    try:
        t0 = time.time()
        for i, freq in enumerate(freq_list):
            sdg.set_burst(freq, BURST_AMPL_VPP, BURST_MS, DEAD_MS)
            time.sleep(SETTLE_MS / 1000.0)

            # Ch B trigger fires at electrical burst onset; acoustic signal
            # arrives at Ch A after TOF_MS. Analysis window accounts for both.
            received = capture_block(chandle, n_capture, trig_adc)
            if i_end > len(received):
                amp_A[i], snr_A[i] = 0.0, -99.0
                n_missed += 1
                continue
            mV = received[i_start:i_end] * 2000.0
            amp_A[i], snr_A[i] = fft_amplitude_mV(mV, freq, actual_fs)

            if snr_A[i] < -20:
                n_missed += 1

            if i % 50 == 0 or i == len(freq_list)-1:
                elapsed = time.time() - t0
                eta = elapsed/(i+1)*(len(freq_list)-i-1)/60 if i > 0 else est_min
                flag = ' [weak?]' if snr_A[i] < -20 else ''
                print(f"[{i+1}/{len(freq_list)}] {freq/1000:.1f} kHz  "
                      f"amp={amp_A[i]:.2f} mV  SNR={snr_A[i]:.1f} dB  "
                      f"(ETA {eta:.1f} min){flag}", flush=True)

    finally:
        sdg.burst_off()
        sdg.output_off(); sdg.disconnect()
        ps.ps5000aStop(chandle); ps.ps5000aCloseUnit(chandle)

    if n_missed:
        print(f"\nWARNING: {n_missed}/{len(freq_list)} frequencies had no burst detected.")

    freqs_m  = np.array(freq_list)
    amp_dB   = 20 * np.log10(amp_A + 1e-12)   # mV peak → dB
    peak_raw = np.max(amp_dB[freqs_m > 25000])
    H_mag    = amp_dB - peak_raw               # normalise to 0 dB at peak (raw)

    # Smoothed curve (w=71 SG, normalised to its own peak)
    H_smooth = savgol_filter(H_mag, window_length=71, polyorder=3)

    # Bandwidth
    search   = (freqs_m >= 35000) & (freqs_m <= 50000)
    peak_v   = H_smooth[search].max()
    peak_f   = freqs_m[search][np.argmax(H_smooth[search])]
    H_norm   = H_smooth - peak_v

    def bw_at(thr):
        above = freqs_m[search & (H_norm >= thr)]
        return (above.min(), above.max(), int(above.max()-above.min())) if len(above) else (None,None,None)

    f3lo,f3hi,bw3 = bw_at(-3)
    f6lo,f6hi,bw6 = bw_at(-6)
    print(f"\n{'='*50}")
    print(f"Peak:  {peak_f/1000:.2f} kHz")
    if bw3: print(f"-3 dB: {f3lo/1000:.2f}–{f3hi/1000:.2f} kHz  BW={bw3} Hz")
    if bw6: print(f"-6 dB: {f6lo/1000:.2f}–{f6hi/1000:.2f} kHz  BW={bw6} Hz")
    print('='*50)

    # Save
    stamp    = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_base = os.path.join(OUTPUT_DIR, f'ecm999_stepped_sine_burst8ms_{stamp}')
    np.savez(out_base + '.npz',
             freqs=freqs_m, amp_mV=amp_A, H_mag=H_mag, H_smooth=H_smooth,
             snr_dB=snr_A, SDG_AMPL_VPP=BURST_AMPL_VPP, actual_fs=actual_fs,
             burst_ms=BURST_MS, dead_ms=DEAD_MS, burst_settle_frac=BURST_SETTLE_FRAC)

    # Plot
    bw3_str = f'-3 dB: {f3lo/1000:.1f}–{f3hi/1000:.1f} kHz (BW={bw3} Hz)' if bw3 else ''
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'ECM999 + SR560  —  BTWV burst stepped-sine  '
        f'({BURST_AMPL_VPP} Vpp, {BURST_MS:.0f} ms burst, SG w=71)\n'
        f'Peak: {peak_f/1000:.2f} kHz  |  {bw3_str}',
        fontsize=10, fontweight='bold')

    ax = axes[0]
    ax.plot(freqs_m/1000, H_mag,    'b-', lw=0.4, alpha=0.25, label='Raw')
    ax.plot(freqs_m/1000, H_norm,   'b-', lw=2.2, label='SG smooth (w=71)')
    ax.axhline(-3, color='C3',     lw=1.5, ls='--', label='−3 dB')
    ax.axhline(-6, color='orange', lw=1.2, ls='--', label='−6 dB')
    ax.axvline(peak_f/1000, color='C2', lw=1.2, ls=':')
    if bw3: ax.axvspan(f3lo/1000, f3hi/1000, alpha=0.12, color='C0',
                       label=f'-3dB BW={bw3} Hz')
    ax.set_xlabel('Frequency (kHz)'); ax.set_ylabel('H(f) (dB, norm. to peak)')
    ax.set_title('Full sweep (20–65 kHz)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlim(20, 65); ax.set_ylim(-40, 3)

    ax = axes[1]
    ax.plot(freqs_m/1000, snr_A, 'g-', lw=0.8)
    ax.axhline(10, color='C3', lw=1.5, ls='--', label='10 dB floor')
    ax.axvline(peak_f/1000, color='C2', lw=1.2, ls=':')
    ax.set_xlabel('Frequency (kHz)'); ax.set_ylabel('SNR (dB)')
    ax.set_title('SNR on Ch A'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlim(20, 65)

    plt.tight_layout()
    plt.savefig(out_base + '.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nData: {out_base}.npz")
    print(f"Plot: {out_base}.png")


if __name__ == '__main__':
    main()
