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
TIMEBASE      = 653            # → (653-2)×4 = 2604 ns/sample ≈ 384 kHz
CH_A_RANGE    = ps.PS5000A_RANGE["PS5000A_2V"]      # ±2V  (SR560 output ≤ ±1V)
CH_A_COUPLING = ps.PS5000A_COUPLING["PS5000A_AC"]   # block SR560 DC offset

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════════════════════
# SDG control
# ═══════════════════════════════════════════════════════════════════════════════

class SDG1062X_LAN:
    def __init__(self, ip=SDG_IP, port=SDG_PORT, timeout=10.0):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.sock.connect((ip, port))

    def write(self, cmd):
        self.sock.sendall((cmd + '\n').encode('utf-8'))

    def write_raw(self, data):
        self.sock.sendall(data)

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

    def set_arb_mode(self, channel=1):
        self.write(f"C{channel}:BSWV WVTP,ARB"); time.sleep(0.2)

    def output_on(self,  channel=1): self.write(f"C{channel}:OUTP ON")
    def output_off(self, channel=1): self.write(f"C{channel}:OUTP OFF")
    def disconnect(self):            self.sock.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PicoScope helpers
# ═══════════════════════════════════════════════════════════════════════════════

def open_picoscope():
    """Open device, enable Ch A only, return (chandle, actual_fs_hz)."""
    status  = {}
    chandle = ctypes.c_int16()
    status["open"] = ps.ps5000aOpenUnit(ctypes.byref(chandle), None, RESOLUTION)
    assert_pico_ok(status["open"])

    chA = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
    status["chA"] = ps.ps5000aSetChannel(chandle, chA, 1,
                                          CH_A_COUPLING, CH_A_RANGE, 0.0)
    assert_pico_ok(status["chA"])

    for ch_name in ["PS5000A_CHANNEL_B", "PS5000A_CHANNEL_C", "PS5000A_CHANNEL_D"]:
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


def capture_block(chandle, n_samples):
    """
    Single block capture of n_samples from Ch A.
    Returns float64 array normalised to ±1 FS (±2V range → divide by 2000 mV).
    """
    chA = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
    # Auto-trigger after 10 ms — SDG burst is already looping when we arm
    ps.ps5000aSetSimpleTrigger(chandle, 0, chA, 0,
                               ps.PS5000A_THRESHOLD_DIRECTION["PS5000A_NONE"],
                               0, 10)
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

def main():
    freq_list = np.arange(FREQ_START, FREQ_END + FREQ_STEP, FREQ_STEP)

    burst_s        = BURST_DURATION_MS / 1000.0
    dead_ms        = max(BURST_DURATION_MS * 2, 10.0)
    period_ms      = BURST_DURATION_MS + dead_ms
    buf_n          = min(int(period_ms / 1000 * SDG_SAMPLE_RATE), SDG_MAX_SAMPLES)
    rep_rate       = SDG_SAMPLE_RATE / buf_n
    settle_frac    = BURST_SETTLE_FRAC

    est_min = len(freq_list) * (period_ms * 2 / 1000 + 0.8) / 60
    print(f"ECM999 burst stepped-sine: {FREQ_START/1000:.0f}–{FREQ_END/1000:.0f} kHz, "
          f"{FREQ_STEP} Hz steps ({len(freq_list)} pts)")
    print(f"Burst: {BURST_DURATION_MS:.0f} ms + {dead_ms:.0f} ms dead  "
          f"| rep {rep_rate:.1f} Hz  |  {BURST_AMPL_VPP} Vpp")
    print(f"Estimated time: ~{est_min:.0f} min\n")
    print("Ensure: phantom box ON, SR560 gain set (×100 recommended), BNC → Ch A\n")

    print("Connecting to SDG...")
    sdg = SDG1062X_LAN()
    print(f"  SDG: {sdg.query('*IDN?')[:60]}")
    sdg.set_arb_mode()

    print("Opening PicoScope...")
    chandle, actual_fs = open_picoscope()

    # n_samples to capture = 2 burst periods (enough to reliably catch a burst)
    n_capture   = int(round(actual_fs * period_ms * 2 / 1000))
    burst_samps = int(round(actual_fs * burst_s))
    settle_samps = int(settle_frac * burst_samps)

    transfer_mags, transfer_phases, received_example = [], [], None

    try:
        for i, freq in enumerate(freq_list):
            if i % 50 == 0:
                elapsed = time.time() - t0 if i > 0 else 0
                eta = elapsed / i * (len(freq_list) - i) / 60 if i > 0 else est_min
                print(f"\n[{i}/{len(freq_list)}] {freq/1000:.1f} kHz  (ETA {eta:.1f} min)")

            # Build burst buffer
            n_cycles    = max(1, int(freq * burst_s))
            actual_bs   = n_cycles / freq
            n_burst_sdg = int(actual_bs * SDG_SAMPLE_RATE)
            t_b         = np.arange(n_burst_sdg) / SDG_SAMPLE_RATE
            burst_sdg   = (AMPLITUDE * np.sin(2 * np.pi * freq * t_b)
                           * scipy_signal.windows.tukey(n_burst_sdg, alpha=0.1))
            sdg_buf     = np.zeros(buf_n)
            sdg_buf[:min(n_burst_sdg, buf_n)] = burst_sdg[:min(n_burst_sdg, buf_n)]

            wf_name = f"bs{int(freq/1000*10):05d}"[:15]
            sdg.upload_waveform(wf_name, sdg_buf, rep_rate, ampl_vpp=BURST_AMPL_VPP)
            sdg.output_on()

            # Capture
            received = capture_block(chandle, n_capture)
            sdg.output_off()

            if i == 0: t0 = time.time()

            # Find burst peak via Hilbert envelope, extract settled window
            env      = np.abs(hilbert(received))
            peak_idx = int(np.argmax(env))
            win_start = max(0, peak_idx - burst_samps + settle_samps)
            win_end   = min(len(received), peak_idx + burst_samps // 4)
            seg       = received[win_start:win_end]

            tx_rms = np.sqrt(0.5) * AMPLITUDE
            rx_rms = float(np.sqrt(np.mean(seg**2))) if len(seg) > 0 else 1e-10
            mag_db = 20 * np.log10(rx_rms / (tx_rms + 1e-10) + 1e-10)

            # Phase via FFT of settled segment
            if len(seg) > 0:
                fft_seg  = np.fft.fft(seg)
                freqs_sg = np.fft.fftfreq(len(seg), 1.0 / actual_fs)
                k        = np.argmin(np.abs(freqs_sg - freq))
                phase_rad = float(np.angle(fft_seg[k]))
            else:
                phase_rad = 0.0

            transfer_mags.append(mag_db)
            transfer_phases.append(phase_rad)

            if 40000 <= freq <= 41000 and received_example is None:
                received_example = received.copy()

            print(f"  {freq/1000:.1f} kHz: {mag_db:.1f} dB", end="\r")

    finally:
        sdg.output_off(); sdg.disconnect()
        ps.ps5000aStop(chandle); ps.ps5000aCloseUnit(chandle)

    freqs_m  = np.array(freq_list)
    H_mag    = np.array(transfer_mags)
    H_phase  = np.array(transfer_phases)

    # Smoothed curve
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
    out_base = os.path.join(OUTPUT_DIR, f'ecm999_stepped_sine_burst{BURST_DURATION_MS:.0f}ms_{stamp}')
    np.savez(out_base + '.npz',
             freqs=freqs_m, H_mag=H_mag, H_smooth=H_smooth, H_phase=H_phase,
             BURST_DURATION_MS=BURST_DURATION_MS, BURST_AMPL_VPP=BURST_AMPL_VPP,
             actual_fs=actual_fs)

    # Plot
    H_raw_n = H_mag - peak_v
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'ECM999 + SR560 Transfer Function — burst stepped-sine '
        f'({BURST_DURATION_MS:.0f} ms burst, {BURST_AMPL_VPP} Vpp, SG w=71)\n'
        f'Peak: {peak_f/1000:.2f} kHz  |  '
        f'-3 dB: {f3lo/1000:.1f}–{f3hi/1000:.1f} kHz (BW={bw3} Hz)' if bw3 else
        f'ECM999 + SR560 — burst stepped-sine',
        fontsize=10, fontweight='bold')

    for ax, (xlo, xhi, title) in zip(axes, [(20,65,'Full sweep (20–65 kHz)'),
                                              (35,50,'Passband zoom (35–50 kHz)')]):
        zm = (freqs_m >= xlo*1000) & (freqs_m <= xhi*1000)
        ax.plot(freqs_m[zm]/1000, H_raw_n[zm], 'b-', lw=0.4, alpha=0.25, label='Raw')
        ax.plot(freqs_m[zm]/1000, H_norm[zm],  'b-', lw=2.2, label='SG smooth (w=71)')
        ax.axhline(-3, color='C3',     lw=1.5, ls='--', label='−3 dB')
        ax.axhline(-6, color='orange', lw=1.2, ls='--', label='−6 dB')
        ax.axvline(peak_f/1000, color='C2', lw=1.2, ls=':',
                   label=f'{peak_f/1000:.2f} kHz')
        if bw3: ax.axvspan(f3lo/1000, f3hi/1000, alpha=0.12, color='C0',
                           label=f'-3dB BW={bw3} Hz')
        ax.set_xlabel('Frequency (kHz)'); ax.set_ylabel('H(f) (dB, norm. to peak)')
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(-35 if xhi==65 else -20, 3)

    plt.tight_layout()
    plt.savefig(out_base + '_smoothed71.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nData: {out_base}.npz")
    print(f"Plot: {out_base}_smoothed71.png")


if __name__ == '__main__':
    main()
