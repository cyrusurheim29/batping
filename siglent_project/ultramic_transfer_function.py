#!/usr/bin/env python3
"""
Stepped-sine transfer function measurement via Dodotronic UltraMic 384k.

Two methods selectable via METHOD constant below:

  'hann'   — 50 ms capture, 2 Vpp, Hann-windowed FFT amplitude.
             Mirrors picoscope_stepped_sine.py; correctly identifies noise floor.
             Output: ultramic_transfer_function_hann_YYYYMMDD_HHMMSS.*

  'legacy' — 300 ms capture, 6 Vpp, FFT-ratio amplitude (H = FFT_rx / FFT_tx).
             Reproduces the original stepped_sine_characterization.py 300ms run.
             Note: above ~52 kHz this method measures noise/noise, not real signal —
             the response appears to extend to 65 kHz but is an artefact of the
             FFT-ratio not distinguishing signal from noise.
             Output: ultramic_transfer_function_legacy_YYYYMMDD_HHMMSS.*
"""

import numpy as np
import pyaudio
import socket
import time
import os
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ═══════════════════════════════════════════════════════════════════════════════
# USER-CONFIGURABLE SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

METHOD        = 'legacy'       # 'hann' or 'legacy'

SDG_IP        = "10.11.13.230"
SDG_PORT      = 5025

FREQ_START    = 20_000         # Hz
FREQ_END      = 65_000         # Hz
FREQ_STEP     = 50             # Hz  → 901 frequencies

FS            = 384_000        # UltraMic sample rate (Hz)
CHUNK         = 4096

OUTPUT_DIR    = os.path.dirname(os.path.abspath(__file__))

# ── Hann method settings ──────────────────────────────────────────────────────
HANN_AMPL_VPP = 2.0            # Vpp  — same as ECM999 run
HANN_SETTLE_MS = 20            # ms settle after frequency change
HANN_N_SAMPLES = 19_200        # 50 ms at 384 kHz

# ── Legacy method settings ────────────────────────────────────────────────────
LEGACY_AMPL_VPP  = 2.0         # Vpp  — matches original 300ms run (was 2.0 Vpp,
                               # NOT 6 Vpp; 6 Vpp was burst-mode only)
                               # WARNING: continuous-sine with equal incident/reflected
                               # amplitude creates standing-wave nulls. At ~40 kHz the
                               # path length (Δl ≈ 4 m → Δl/λ ≈ n+0.5) causes destructive
                               # cancellation that fully resolves in the 300ms / 3.3 Hz
                               # FFT but is smeared out in the 50ms Hann method.
                               # Use stepped_sine_characterization.py BURST_MODE=True for
                               # a reliable H(f) with this mic geometry.
LEGACY_SETTLE_MS = 200         # ms settle (original used 200ms wait after output_on)
LEGACY_DURATION_S = 0.3        # s per frequency  (300 ms)

# ═══════════════════════════════════════════════════════════════════════════════
# SDG control
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
        time.sleep(0.01)

    def output_on(self, channel=1):
        self.write(f"C{channel}:OUTP ON")

    def output_off(self, channel=1):
        self.write(f"C{channel}:OUTP OFF")

    def close(self):
        self.sock.close()


# ═══════════════════════════════════════════════════════════════════════════════
# UltraMic helpers
# ═══════════════════════════════════════════════════════════════════════════════

def open_ultramic():
    pa = pyaudio.PyAudio()
    idx = None
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0 and 'UltraMic' in info['name']:
            idx = i
            break
    if idx is None:
        pa.terminate()
        raise RuntimeError("UltraMic not found — check USB connection")
    print(f"  UltraMic found at device index {idx}")
    return pa, idx


def read_samples(stream, n):
    """Read exactly n samples from an open PyAudio stream. Returns float64 array."""
    frames = []
    remaining = n
    while remaining > 0:
        chunk = min(CHUNK, remaining)
        data = stream.read(chunk, exception_on_overflow=False)
        frames.append(data)
        remaining -= chunk
    return np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float64) / 32768.0


# ── Hann method ───────────────────────────────────────────────────────────────

def fft_amplitude_hann(signal, freq_hz):
    """Hann-windowed FFT → (amplitude, snr_db). Correct noise floor detection."""
    N    = len(signal)
    win  = np.hanning(N)
    S    = np.fft.rfft(signal * win)
    amps = 2.0 * np.abs(S) / (N * 0.5)
    k    = int(round(freq_hz * N / FS))
    k    = max(1, min(k, len(amps) - 2))
    amp  = float(amps[k])
    mask = np.ones(len(amps), bool)
    mask[max(0, k-5):k+6] = False
    noise_rms = float(np.sqrt(np.mean(amps[mask]**2)))
    snr_db    = 20 * np.log10(amp / (noise_rms + 1e-12))
    return amp, snr_db


# ── Legacy method ─────────────────────────────────────────────────────────────

def fft_ratio_legacy(received, tx_sine, freq_hz):
    """
    FFT ratio H = FFT_rx[bin] / FFT_tx[bin], matching original stepped_sine script.
    Also computes SNR from received FFT for diagnostics.
    Returns (mag_linear, snr_db).
    """
    N        = len(received)
    fft_rx   = np.fft.fft(received)
    # Reference: ideal sine at this frequency, same length
    tx_matched = np.zeros(N)
    tx_matched[:len(tx_sine)] = tx_sine[:N]
    fft_tx   = np.fft.fft(tx_matched)
    freqs_fft = np.fft.fftfreq(N, 1.0 / FS)
    k        = np.argmin(np.abs(freqs_fft - freq_hz))
    H        = fft_rx[k] / (fft_tx[k] + 1e-10)
    mag      = float(np.abs(H))

    # SNR from received FFT amplitude at target bin vs noise floor
    amps_rx  = np.abs(fft_rx) / N * 2
    amp_sig  = float(amps_rx[k])
    mask     = np.ones(N // 2, bool)
    mask[max(0, k-5):k+6] = False
    noise_rms = float(np.sqrt(np.mean(amps_rx[:N//2][mask]**2)))
    snr_db    = 20 * np.log10(amp_sig / (noise_rms + 1e-12))
    return mag, snr_db


def make_tx_sine(freq_hz, duration_s, amplitude=0.9):
    """Ideal sine reference for FFT-ratio normalisation."""
    n = int(duration_s * FS)
    t = np.arange(n) / FS
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


# ═══════════════════════════════════════════════════════════════════════════════
# Bandwidth helper (shared)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_bandwidth(freqs, H_smooth):
    search_mask = freqs > 25_000
    smooth_peak = H_smooth[search_mask].max()
    peak_f      = freqs[search_mask][np.argmax(H_smooth[search_mask])]

    def bw_at(threshold_dB):
        level = smooth_peak + threshold_dB
        above = freqs[search_mask & (H_smooth >= level)]
        if len(above) == 0:
            return None, None, None
        return above.min(), above.max(), above.max() - above.min()

    return peak_f, smooth_peak, bw_at(-3.0), bw_at(-6.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Main sweep
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    freqs = np.arange(FREQ_START, FREQ_END + FREQ_STEP, FREQ_STEP, dtype=float)
    N_f   = len(freqs)
    amp_A = np.zeros(N_f)
    snr_A = np.zeros(N_f)

    if METHOD == 'hann':
        ampl_vpp   = HANN_AMPL_VPP
        settle_ms  = HANN_SETTLE_MS
        n_capture  = HANN_N_SAMPLES
        window_ms  = n_capture / FS * 1000
        est_min    = N_f * (settle_ms + window_ms * 2 + 5) / 1000 / 60
        method_tag = 'hann'
        print(f"Method: HANN  ({window_ms:.0f} ms Hann FFT, {ampl_vpp} Vpp)")
    else:
        ampl_vpp   = LEGACY_AMPL_VPP
        settle_ms  = LEGACY_SETTLE_MS
        n_capture  = int(LEGACY_DURATION_S * FS)
        window_ms  = LEGACY_DURATION_S * 1000
        est_min    = N_f * (settle_ms + window_ms + 300) / 1000 / 60
        method_tag = 'legacy'
        print(f"Method: LEGACY  ({window_ms:.0f} ms FFT-ratio, {ampl_vpp} Vpp)")

    print(f"Sweep: {FREQ_START/1000:.0f}–{FREQ_END/1000:.0f} kHz, "
          f"{FREQ_STEP} Hz steps ({N_f} pts)  |  Est. {est_min:.1f} min\n")

    print("Connecting to SDG...")
    sdg = SDG1062X()
    print(f"  SDG: {sdg.query('*IDN?')[:60]}")

    print("Opening UltraMic...")
    pa, mic_idx = open_ultramic()

    if METHOD == 'hann':
        # Keep stream open throughout; flush ring buffer each step
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=FS,
                         input=True, input_device_index=mic_idx,
                         frames_per_buffer=CHUNK)
        sdg.set_sine(FREQ_START, ampl_vpp)
        sdg.output_on()
        time.sleep(0.3)

    try:
        t0 = time.time()
        for i, f in enumerate(freqs):

            if METHOD == 'hann':
                sdg.set_sine(f, ampl_vpp)
                time.sleep(settle_ms / 1000.0)
                read_samples(stream, n_capture)          # flush stale buffer
                audio = read_samples(stream, n_capture)
                amp_A[i], snr_A[i] = fft_amplitude_hann(audio, f)

            else:  # legacy
                # Open stream fresh each frequency (matches original behaviour)
                sdg.set_sine(f, ampl_vpp)
                sdg.output_on()
                stream = pa.open(format=pyaudio.paInt16, channels=1, rate=FS,
                                 input=True, input_device_index=mic_idx,
                                 frames_per_buffer=CHUNK)
                time.sleep(settle_ms / 1000.0)
                audio = read_samples(stream, n_capture)
                stream.stop_stream(); stream.close()
                sdg.output_off()
                tx = make_tx_sine(f, LEGACY_DURATION_S)
                amp_A[i], snr_A[i] = fft_ratio_legacy(audio, tx, f)

            if i % 50 == 0 or i == N_f - 1:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (N_f - i - 1)
                if METHOD == 'hann':
                    print(f"  {f/1000:6.2f} kHz  amp={amp_A[i]:.5f}  "
                          f"SNR={snr_A[i]:.1f} dB  (ETA {eta/60:.1f} min)", flush=True)
                else:
                    print(f"  {f/1000:6.2f} kHz  |H|={amp_A[i]:.4f}  "
                          f"SNR={snr_A[i]:.1f} dB  (ETA {eta/60:.1f} min)", flush=True)

    finally:
        sdg.output_off(); sdg.close()
        if METHOD == 'hann':
            stream.stop_stream(); stream.close()
        pa.terminate()

    # ── Compute H(f) ──────────────────────────────────────────────────────────
    amp_dB  = 20 * np.log10(amp_A + 1e-12)
    peak_dB = np.max(amp_dB[freqs > 25_000])
    H_norm  = amp_dB - peak_dB
    H_smooth = savgol_filter(H_norm, window_length=21, polyorder=3)

    peak_f, smooth_peak, (f_lo3, f_hi3, bw3), (f_lo6, f_hi6, bw6) = \
        compute_bandwidth(freqs, H_smooth)

    print(f"\n{'='*50}")
    print(f"Peak:  {peak_f/1000:.2f} kHz")
    if bw3:
        print(f"-3 dB: {f_lo3/1000:.1f}–{f_hi3/1000:.1f} kHz  BW={bw3/1000:.2f} kHz")
    if bw6:
        print(f"-6 dB: {f_lo6/1000:.1f}–{f_hi6/1000:.1f} kHz  BW={bw6/1000:.2f} kHz")
    snr_40 = snr_A[np.argmin(np.abs(freqs - 40_000))]
    print(f"SNR @ 40 kHz: {snr_40:.0f} dB")
    usable = freqs[snr_A >= 10]
    if len(usable):
        print(f"Usable (SNR>10 dB): {usable.min()/1000:.0f}–{usable.max()/1000:.0f} kHz")
    print('='*50)

    # ── Plot ──────────────────────────────────────────────────────────────────
    if METHOD == 'hann':
        title = (f'UltraMic 384k  —  HANN method  '
                 f'({HANN_N_SAMPLES/FS*1000:.0f} ms Hann FFT, {ampl_vpp} Vpp)')
        snr_label = 'SNR (Hann FFT noise floor)'
    else:
        title = (f'UltraMic 384k  —  LEGACY method  '
                 f'({LEGACY_DURATION_S*1000:.0f} ms FFT-ratio, {ampl_vpp} Vpp)\n'
                 f'Note: response above ~52 kHz is noise artefact (SNR < 0 dB)')
        snr_label = 'SNR on received signal'

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=11, fontweight='bold')

    ax = axes[0]
    ax.plot(freqs/1000, H_norm,   'b-', lw=0.5, alpha=0.35, label='Raw')
    ax.plot(freqs/1000, H_smooth, 'b-', lw=2.0, label='Smoothed')
    ax.axhline(smooth_peak - 3, color='C3',     lw=1.5, ls='--', label='-3 dB')
    ax.axhline(smooth_peak - 6, color='orange', lw=1.2, ls='--', label='-6 dB')
    ax.axvline(40, color='C2', lw=1.2, ls=':', label='40 kHz')
    if bw3:
        ax.axvspan(f_lo3/1000, f_hi3/1000, alpha=0.12, color='C0',
                   label=f'-3 dB BW = {bw3/1000:.2f} kHz')
    ax.set_xlabel('Frequency (kHz)'); ax.set_ylabel('H(f) (dB, norm. to peak)')
    ax.set_title('Full sweep'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlim(FREQ_START/1000, FREQ_END/1000)
    ax.set_ylim(-70, 5)

    ax = axes[1]
    ax.plot(freqs/1000, snr_A, 'g-', lw=0.8)
    ax.axhline(10,  color='C3',  lw=1.5, ls='--', label='10 dB floor')
    ax.axhline(0,   color='gray', lw=1.0, ls=':',  label='0 dB (noise)')
    ax.axvline(40,  color='C2',  lw=1.2, ls=':')
    ax.set_xlabel('Frequency (kHz)'); ax.set_ylabel('SNR (dB)')
    ax.set_title(snr_label); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlim(FREQ_START/1000, FREQ_END/1000)

    plt.tight_layout()
    stamp   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_png = os.path.join(OUTPUT_DIR, f'ultramic_transfer_function_{method_tag}_{stamp}.png')
    out_npz = os.path.join(OUTPUT_DIR, f'ultramic_transfer_function_{method_tag}_{stamp}.npz')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot: {out_png}")

    np.savez(out_npz,
             freqs=freqs, amp=amp_A, amp_dB=amp_dB, H_norm=H_norm, H_smooth=H_smooth,
             snr_dB=snr_A, method=method_tag,
             SDG_AMPL_VPP=ampl_vpp, FS=FS,
             N_SAMPLES=n_capture, SETTLE_MS=settle_ms)
    print(f"Data: {out_npz}")


if __name__ == '__main__':
    main()
