#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stepped-Sine Piezoelectric Transducer Bandwidth Characterization

Validates chirp-based bandwidth measurement by sweeping single-frequency
sines across the band and measuring response amplitude at each frequency.

For each frequency:
1. Generate sine wave at that frequency
2. Upload to Siglent SDG1062X via LAN
3. Record response with Dodotronic UltraMic @ 384 kHz
4. Extract transfer function magnitude at that frequency
5. Store point in overall transfer function

@author: cyrusurheimog
Created: June 15, 2026
"""

import numpy as np
import pyaudio
import socket
import time
from scipy import signal
from scipy.io import wavfile
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Waveform generator configuration
SDG_IP = "10.11.13.230"
SDG_PORT = 5025
SDG_SAMPLE_RATE = 400000  # 400 kSa/s for SDG

# Mic settings
DODOMIC_SAMPLE_RATE = 384000
RECORD_DURATION = 0.3
CHUNK_SIZE = 4096

# Piezo speaker specifications
RES_FREQ = 40000

# Frequency sweep configuration
FREQ_START = 20000   # Hz  (extended to capture full chirp band 26-57 kHz + margins)
FREQ_END = 65000     # Hz
FREQ_STEP = 50       # Hz  (~900 frequencies, ~13 min total)
AMPLITUDE = 0.9

# ── Burst mode (advisor recommendation) ──────────────────────────────────────
# Instead of 300ms continuous sine per frequency, send a short burst and measure
# the received peak amplitude. Much faster (~75×) and avoids heating.
# Piezo ring-up time ≈ Q/(π·f₀) ≈ 23/(π·40e3) ≈ 180 μs, so measure amplitude
# in the SECOND HALF of the burst once the piezo has settled.
BURST_MODE = True           # True = short burst; False = original 300ms continuous
BURST_DURATION_MS = 8.0     # ms per frequency (8ms → ~320 cycles at 40kHz; reflection arrives at ~11.6ms so settled window is clear)
BURST_AMPL_VPP = 1.2        # Vpp — 2.0 Vpp still clips at resonance (+2.5 dB peak);
                             # 1.2 Vpp backs off by factor 0.6 → puts peak at ~-2 dB
BURST_SETTLE_FRAC = 0.5     # ignore first SETTLE_FRAC of burst (ring-up), measure rest

OUTPUT_DIR = '/Users/cyrusurheim_1/Downloads/Stubbs_Stuff/Claude_Dir/siglent_project/'


class SDG1062X_LAN:
    """Siglent SDG1062X remote control via LAN (direct ethernet cable) connection."""

    def __init__(self, ip=SDG_IP, port=SDG_PORT, timeout=10.0):
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self.sock = None

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect((self.ip, self.port))
        print(f"Connected to SDG @ {self.ip}:{self.port}")

    def disconnect(self):
        if self.sock:
            self.sock.close()
            self.sock = None

    def write(self, cmd):
        self.sock.sendall((cmd + '\n').encode('utf-8'))

    def write_raw(self, data):
        self.sock.sendall(data)

    def read(self, bufsize=4096):
        return self.sock.recv(bufsize).decode('utf-8').strip()

    def query(self, cmd):
        self.write(cmd)
        time.sleep(0.1)
        return self.read()

    def upload_waveform(self, name, wave_data, freq_hz, ampl_vpp=2.0, channel=1):
        """Upload waveform with little-endian 16-bit encoding."""
        wave_int16 = np.clip(wave_data, -1.0, 1.0)
        wave_int16 = (wave_int16 * 32767).astype(np.int16)
        bin_data = wave_int16.astype('<i2').tobytes()

        cmd_prefix = f"C{channel}:WVDT WVNM,{name},FREQ,{freq_hz},AMPL,{ampl_vpp},OFST,0,PHASE,0,WAVEDATA,"
        full_cmd = cmd_prefix.encode('utf-8') + bin_data + b'\n'

        self.write_raw(full_cmd)
        time.sleep(0.5)

        self.write(f"C{channel}:ARWV NAME,{name}")
        time.sleep(0.2)

    def set_arb_mode(self, channel=1):
        self.write(f"C{channel}:BSWV WVTP,ARB")
        time.sleep(0.2)

    def output_on(self, channel=1):
        self.write(f"C{channel}:OUTP ON")

    def output_off(self, channel=1):
        self.write(f"C{channel}:OUTP OFF")


def generate_sine_waveform(freq, duration, sample_rate, amplitude=0.9):
    """Generate a windowed sine wave at the specified frequency."""
    n_samples = int(duration * sample_rate)
    t = np.arange(n_samples) / sample_rate
    waveform = amplitude * np.sin(2 * np.pi * freq * t)

    # Apply Tukey window to avoid spectral leakage from edge discontinuities
    window = signal.windows.tukey(n_samples, alpha=0.1)
    waveform = waveform * window

    return t, waveform


def create_sdg_waveform(waveform, sdg_sample_rate, dodomic_sample_rate, duration, freq):
    """Create SDG waveform buffer with sine wave embedded."""
    # Number of SDG samples that make up one period
    period_samples = int(sdg_sample_rate / freq)

    # Resample the waveform to SDG sample rate
    waveform_samples_needed = int(duration * sdg_sample_rate)
    if len(waveform) != waveform_samples_needed:
        waveform_resampled = signal.resample(waveform, waveform_samples_needed)
    else:
        waveform_resampled = waveform

    # Create buffer for one period (SDG repeats this)
    waveform_final = np.zeros(period_samples)

    if len(waveform_resampled) > period_samples:
        # For continuous sine, just take one period
        waveform_final = waveform_resampled[:period_samples]
    else:
        waveform_final[:len(waveform_resampled)] = waveform_resampled

    # Normalize to [-1, 1]
    max_val = np.max(np.abs(waveform_final))
    if max_val > 0:
        waveform_final = waveform_final / max_val * 0.9

    return waveform_final


def find_ultramic():
    """Find and return UltraMic device."""
    pa = pyaudio.PyAudio()
    mic_index = None

    print("\nAvailable audio devices:")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"  [{i}] {info['name']} (in: {info['maxInputChannels']}ch)")
            if "UltraMic" in info['name']:
                mic_index = i

    if mic_index is None:
        print("\nUltraMic not found!")
        pa.terminate()
        return None, None

    print(f"\nUsing UltraMic at index {mic_index}")
    return pa, mic_index


def record_audio(pa, mic_index, duration=RECORD_DURATION):
    """Record audio from UltraMic."""
    n_samples = int(duration * DODOMIC_SAMPLE_RATE)

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=DODOMIC_SAMPLE_RATE,
        input=True,
        input_device_index=mic_index,
        frames_per_buffer=CHUNK_SIZE
    )

    frames = []
    remaining = n_samples

    while remaining > 0:
        chunk = min(CHUNK_SIZE, remaining)
        data = stream.read(chunk, exception_on_overflow=False)
        frames.append(data)
        remaining -= chunk

    stream.stop_stream()
    stream.close()

    audio = np.frombuffer(b''.join(frames), dtype=np.int16)
    audio_float = audio.astype(np.float64) / 32768.0

    return audio_float


def find_bandwidth(freqs, H_mag_db, threshold_db=-3, search_range=(35000, 50000), smooth_window=11):
    """
    Find the bandwidth at a given threshold below peak.

    Returns:
        f_center, f_low, f_high, bandwidth, peak_freq, peak_mag, H_mag_smooth
    """
    # Smooth the magnitude response
    if smooth_window > 3 and len(H_mag_db) > smooth_window:
        H_mag_smooth = savgol_filter(H_mag_db, smooth_window, 3)
    else:
        H_mag_smooth = H_mag_db.copy()

    # Constrain search to expected piezo frequency range
    search_mask = (freqs >= search_range[0]) & (freqs <= search_range[1])
    search_indices = np.where(search_mask)[0]

    if len(search_indices) == 0:
        raise ValueError(f"No frequencies in search range {search_range}")

    # Find peak within search range
    H_mag_search = H_mag_smooth[search_mask]
    peak_idx_in_search = np.argmax(H_mag_search)
    peak_idx = search_indices[peak_idx_in_search]

    peak_freq = freqs[peak_idx]
    peak_mag = H_mag_smooth[peak_idx]

    threshold = peak_mag + threshold_db

    # Find where magnitude crosses threshold
    above_threshold = H_mag_smooth >= threshold

    # Find edges
    transitions = np.diff(above_threshold.astype(int))
    rise_idx = np.where(transitions == 1)[0]
    fall_idx = np.where(transitions == -1)[0]

    # Find the passband containing the peak
    if len(rise_idx) > 0:
        rises_before_peak = rise_idx[rise_idx < peak_idx]
        if len(rises_before_peak) > 0:
            low_idx = rises_before_peak[-1]
        else:
            low_idx = 0
    else:
        low_idx = 0

    if len(fall_idx) > 0:
        falls_after_peak = fall_idx[fall_idx > peak_idx]
        if len(falls_after_peak) > 0:
            high_idx = falls_after_peak[0]
        else:
            high_idx = len(freqs) - 1
    else:
        high_idx = len(freqs) - 1

    f_low = freqs[low_idx]
    f_high = freqs[high_idx]
    bandwidth = f_high - f_low
    f_center = (f_low + f_high) / 2

    return f_center, f_low, f_high, bandwidth, peak_freq, peak_mag, H_mag_smooth


def plot_stepped_sine_result(freqs, H_mag, H_mag_smooth, H_phase,
                             f_center_3db, f_low_3db, f_high_3db, bw_3db,
                             f_center_6db, f_low_6db, f_high_6db, bw_6db,
                             peak_mag, received_example, output_path,
                             chirp_data_path=None):
    """Stepped-sine characterization plot: full TF | zoomed passband | info panel."""

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[5, 1.8],
                          hspace=0.38, wspace=0.30,
                          left=0.07, right=0.97, top=0.93, bottom=0.03)
    ax1     = fig.add_subplot(gs[0, 0])
    ax2     = fig.add_subplot(gs[0, 1])
    ax_info = fig.add_subplot(gs[1, :])

    # ── Panel 1: Full magnitude response (unchanged) ───────────────────────────
    ax1.plot(freqs/1000, H_mag, 'b-', linewidth=0.5, alpha=0.5, label='Raw')
    ax1.plot(freqs/1000, H_mag_smooth, 'b-', linewidth=2, label='Smoothed')
    ax1.axhline(peak_mag - 3, color='r', linestyle='--', label='-3 dB')
    ax1.axhline(peak_mag - 6, color='orange', linestyle='--', label='-6 dB')
    ax1.axvline(f_low_3db/1000, color='g', linestyle=':', alpha=0.7)
    ax1.axvline(f_high_3db/1000, color='g', linestyle=':', alpha=0.7)
    ax1.axvline(f_center_3db/1000, color='g', linestyle='-', alpha=0.7,
                label=f'Center: {f_center_3db/1000:.1f} kHz')
    ax1.fill_betweenx([np.min(H_mag_smooth)-5, np.max(H_mag_smooth)+5],
                      f_low_3db/1000, f_high_3db/1000,
                      alpha=0.2, color='green', label=f'BW: {bw_3db/1000:.1f} kHz')
    ax1.set_xlabel('Frequency (kHz)')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_title('Piezo Transfer Function - Magnitude Response')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([FREQ_START/1000, FREQ_END/1000])

    # ── Panel 2: Zoomed passband detail (stepped-sine only, no chirp) ─────────
    zoom_lo, zoom_hi = 35000, 48000
    zoom_mask = (freqs >= zoom_lo) & (freqs <= zoom_hi)
    ax2.plot(freqs[zoom_mask]/1000, H_mag[zoom_mask], 'b-', linewidth=0.5, alpha=0.5, label='Raw')
    ax2.plot(freqs[zoom_mask]/1000, H_mag_smooth[zoom_mask], 'b-', linewidth=2, label='Smoothed')
    ax2.axhline(peak_mag - 3, color='r', linestyle='--', linewidth=1.5, label='-3 dB')
    ax2.axhline(peak_mag - 6, color='orange', linestyle='--', linewidth=1.2, label='-6 dB')
    ax2.axvline(f_low_3db/1000, color='g', linestyle=':', alpha=0.8)
    ax2.axvline(f_high_3db/1000, color='g', linestyle=':', alpha=0.8)
    ax2.axvline(f_center_3db/1000, color='g', linestyle='-', alpha=0.8,
                label=f'Center: {f_center_3db/1000:.1f} kHz')
    y_lo = peak_mag - 20
    y_hi = peak_mag + 2
    ax2.fill_betweenx([y_lo, y_hi], f_low_3db/1000, f_high_3db/1000,
                      alpha=0.2, color='green', label=f'BW: {bw_3db/1000:.1f} kHz')
    ax2.set_xlabel('Frequency (kHz)')
    ax2.set_ylabel('Magnitude (dB)')
    ax2.set_title('Passband Detail (Zoomed)')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([zoom_lo/1000, zoom_hi/1000])
    ax2.set_ylim([y_lo, y_hi])

    # ── Info panel (no axes, three columns, no overlap) ────────────────────────
    ax_info.axis('off')

    rec_dur_str = (f"{BURST_DURATION_MS:.0f} ms/freq (BURST)"
                   if BURST_MODE else f"{RECORD_DURATION*1000:.0f} ms/freq")
    min_chip_us = 1e6 / bw_3db if bw_3db > 0 else 0

    col1 = (
        f"MEASUREMENT PARAMETERS\n"
        f"  Freq range:  {FREQ_START/1000:.0f} – {FREQ_END/1000:.0f} kHz\n"
        f"  Step size:   {FREQ_STEP} Hz\n"
        f"  N points:    {len(freqs)}\n"
        f"  Rec. dur.:   {rec_dur_str}"
    )
    col2 = (
        f"RESULTS\n"
        f"  Center Freq:    {f_center_3db/1000:.2f} kHz\n"
        f"  −3 dB BW:       {bw_3db/1000:.2f} kHz\n"
        f"  −3 dB Passband: {f_low_3db/1000:.1f} – {f_high_3db/1000:.1f} kHz\n"
        f"  −6 dB BW:       {bw_6db/1000:.2f} kHz\n"
        f"  −6 dB Passband: {f_low_6db/1000:.1f} – {f_high_6db/1000:.1f} kHz\n"
        f"  Q Factor:       {f_center_3db/bw_3db:.1f}"
    )
    col3 = (
        f"DERIVED PARAMETERS\n"
        f"  Min chip dur.: {min_chip_us:.0f} μs\n"
        f"    (= 1/BW = 1/{bw_3db/1000:.2f} kHz)\n"
        f"  Rec. T_c:      {min_chip_us:.0f} – {2*min_chip_us:.0f} μs"
    )

    kw = dict(transform=ax_info.transAxes, fontsize=9.5, family='monospace',
              va='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax_info.text(0.01, 0.5, col1, ha='left', **kw)
    ax_info.text(0.36, 0.5, col2, ha='left', **kw)
    ax_info.text(0.70, 0.5, col3, ha='left', **kw)

    fig.suptitle('Piezoelectric Transducer Characterization (Stepped-Sine)',
                 fontsize=14, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved standalone plot: {output_path}")
    plt.close()


def plot_comparison_with_chirp(freqs_ss, H_mag_ss_smooth, f_center_ss, f_low_ss, f_high_ss, bw_ss, chirp_data_path, output_path):
    """Plot stepped-sine result overlaid with chirp-based result."""

    # Load chirp data
    try:
        chirp_data = np.load(chirp_data_path)
        freqs_chirp = chirp_data['freqs']
        H_mag_chirp = chirp_data['H_mag_smooth'] if 'H_mag_smooth' in chirp_data else chirp_data['H_mag']
        f_center_chirp = float(chirp_data['f_center'])
        bw_chirp = float(chirp_data['bandwidth'])
        print(f"Loaded chirp data: center={f_center_chirp/1000:.2f} kHz, BW={bw_chirp/1000:.2f} kHz")
    except Exception as e:
        print(f"Warning: Could not load chirp data: {e}")
        print("Generating comparison plot with stepped-sine only.")
        freqs_chirp = None
        H_mag_chirp = None
        f_center_chirp = None
        bw_chirp = None

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot stepped-sine
    ax.plot(freqs_ss/1000, H_mag_ss_smooth, 'b-', linewidth=2,
            label=f'Stepped-Sine (BW={bw_ss/1000:.2f} kHz)')

    # Plot chirp if available
    if freqs_chirp is not None:
        # Normalize chirp to match stepped-sine scale (align peaks)
        peak_ss = np.max(H_mag_ss_smooth)
        peak_chirp = np.max(H_mag_chirp)
        H_mag_chirp_normalized = H_mag_chirp - peak_chirp + peak_ss

        ax.plot(freqs_chirp/1000, H_mag_chirp_normalized, 'r--', linewidth=2, alpha=0.7,
                label=f'Chirp (BW={bw_chirp/1000:.2f} kHz)')

    # Add threshold lines
    peak_mag = np.max(H_mag_ss_smooth)
    ax.axhline(peak_mag - 3, color='gray', linestyle=':', alpha=0.5, label='-3 dB threshold')

    # Passband shading
    ax.axvline(f_low_ss/1000, color='b', linestyle=':', alpha=0.5)
    ax.axvline(f_high_ss/1000, color='b', linestyle=':', alpha=0.5)

    ax.set_xlabel('Frequency (kHz)', fontsize=12)
    ax.set_ylabel('Magnitude (dB, normalized)', fontsize=12)
    ax.set_title('Chirp vs Stepped-Sine Bandwidth Measurement Comparison', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([30, 50])

    # Add comparison text box
    if bw_chirp is not None:
        diff_bw = bw_ss - bw_chirp
        diff_pct = (diff_bw / bw_chirp) * 100
        comparison_text = (
            f"Stepped-Sine: {f_center_ss/1000:.2f} kHz center, {bw_ss/1000:.2f} kHz BW\n"
            f"Chirp:        {f_center_chirp/1000:.2f} kHz center, {bw_chirp/1000:.2f} kHz BW\n"
            f"Difference:   {diff_bw:.0f} Hz ({diff_pct:+.1f}%)"
        )
    else:
        comparison_text = f"Stepped-Sine: {f_center_ss/1000:.2f} kHz center, {bw_ss/1000:.2f} kHz BW"

    ax.text(0.02, 0.02, comparison_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved comparison plot: {output_path}")
    plt.close()


def main():
    print("=" * 60)
    print("Stepped-Sine Piezo Bandwidth Characterization")
    print("=" * 60)

    if BURST_MODE:
        burst_s = BURST_DURATION_MS / 1000.0
        # SDG buffer: burst + equal dead time for mic to settle between freqs
        # Use a repeating burst at a low rep rate so we capture one clean burst
        dead_ms = max(BURST_DURATION_MS * 2, 10.0)  # at least 10ms dead time
        period_ms = BURST_DURATION_MS + dead_ms
        buf_n = min(int(period_ms / 1000 * SDG_SAMPLE_RATE), 16384)
        rep_rate = SDG_SAMPLE_RATE / buf_n
        record_dur = period_ms / 1000 * 2  # record 2 periods to catch burst reliably
        settle_samples = int(BURST_SETTLE_FRAC * burst_s * DODOMIC_SAMPLE_RATE)
        burst_samples = int(burst_s * DODOMIC_SAMPLE_RATE)
        print(f"\nMode: BURST  ({BURST_DURATION_MS:.0f}ms burst + {dead_ms:.0f}ms dead)")
        print(f"  Rep rate: {rep_rate:.1f} Hz  |  Record: {record_dur*1000:.0f}ms/freq")
        print(f"  Amplitude: {BURST_AMPL_VPP:.1f} Vpp  |  Settle fraction: {BURST_SETTLE_FRAC:.0%}")
        est_time = len(np.arange(FREQ_START, FREQ_END + FREQ_STEP, FREQ_STEP)) * (record_dur + 0.3) / 60
    else:
        print(f"\nMode: CONTINUOUS ({RECORD_DURATION*1000:.0f}ms per frequency)")
        est_time = len(np.arange(FREQ_START, FREQ_END + FREQ_STEP, FREQ_STEP)) * (RECORD_DURATION + 1.0) / 60

    print(f"\nFrequency range: {FREQ_START/1000:.0f} - {FREQ_END/1000:.0f} kHz")
    print(f"Step size: {FREQ_STEP} Hz")
    freq_list = np.arange(FREQ_START, FREQ_END + FREQ_STEP, FREQ_STEP)
    print(f"Total measurements: {len(freq_list)}")
    print(f"Estimated time: {est_time:.1f} minutes")

    # Initialize storage
    transfer_magnitudes = []
    transfer_phases = []
    received_example = None  # Store one example recording for plot

    # Find mic once
    pa, mic_index = find_ultramic()
    if pa is None:
        return 1

    # Connect to SDG once
    sdg = None
    try:
        sdg = SDG1062X_LAN()
        sdg.connect()
        idn = sdg.query("*IDN?")
        print(f"\nDevice: {idn}")

        if "Siglent" not in idn:
            print("Warning: Device may not be Siglent SDG")

        sdg.set_arb_mode(channel=1)

        print("\n" + "-" * 60)
        print("Starting frequency sweep...")
        print("-" * 60)

        start_time = time.time()

        for i, freq in enumerate(freq_list):
            # Progress indicator
            if i % 10 == 0:
                elapsed = time.time() - start_time
                if i > 0:
                    eta = elapsed / i * (len(freq_list) - i)
                    print(f"\n[{i}/{len(freq_list)}] {freq/1000:.1f} kHz (ETA: {eta/60:.1f} min)")
                else:
                    print(f"\n[{i}/{len(freq_list)}] {freq/1000:.1f} kHz")

            if BURST_MODE:
                # ── Burst mode ────────────────────────────────────────────────
                # Build SDG buffer: sine burst + dead time, repeated at rep_rate.
                # Embed integer number of cycles so the burst is phase-coherent.
                n_cycles = max(1, int(freq * burst_s))  # cycles in burst
                actual_burst_s = n_cycles / freq         # exact integer-cycle duration
                n_burst_sdg = int(actual_burst_s * SDG_SAMPLE_RATE)
                t_burst = np.arange(n_burst_sdg) / SDG_SAMPLE_RATE
                burst_sdg = (AMPLITUDE * np.sin(2 * np.pi * freq * t_burst)
                             * signal.windows.tukey(n_burst_sdg, alpha=0.1))
                # Embed in SDG buffer with dead time
                sdg_buf = np.zeros(buf_n)
                sdg_buf[:min(n_burst_sdg, buf_n)] = burst_sdg[:min(n_burst_sdg, buf_n)]

                sdg.upload_waveform(f"bs{int(freq)}", sdg_buf, rep_rate,
                                    ampl_vpp=BURST_AMPL_VPP)
                sdg.output_on(channel=1)
                received = record_audio(pa, mic_index, record_dur)
                sdg.output_off(channel=1)

                # Find the burst in the recording (peak of Hilbert envelope)
                from scipy.signal import hilbert as _hilbert
                env = np.abs(_hilbert(received))
                peak_idx = np.argmax(env)
                # Extract the settled portion: skip first SETTLE_FRAC of burst
                win_start = max(0, peak_idx - burst_samples + settle_samples)
                win_end   = max(win_start + 1, peak_idx + burst_samples // 4)
                win_end   = min(win_end, len(received))
                seg = received[win_start:win_end]

                # Amplitude = RMS of settled burst window (peak-normalise to tx)
                tx_rms = np.sqrt(0.5) * AMPLITUDE  # RMS of sine
                rx_rms = np.sqrt(np.mean(seg**2)) if len(seg) > 0 else 1e-10
                mag_db = 20 * np.log10(rx_rms / (tx_rms + 1e-10) + 1e-10)

                # Phase via FFT of settled segment
                n_seg = len(seg)
                if n_seg > 0:
                    fft_seg = np.fft.fft(seg)
                    freqs_seg = np.fft.fftfreq(n_seg, 1/DODOMIC_SAMPLE_RATE)
                    idx_seg = np.argmin(np.abs(freqs_seg - freq))
                    phase_rad = np.angle(fft_seg[idx_seg])
                else:
                    phase_rad = 0.0

                if 40000 <= freq <= 41000 and received_example is None:
                    received_example = received.copy()

            else:
                # ── Original continuous-sine mode ─────────────────────────────
                t_wf, wf = generate_sine_waveform(freq, RECORD_DURATION,
                                                  DODOMIC_SAMPLE_RATE, AMPLITUDE)
                sdg_waveform = create_sdg_waveform(wf, SDG_SAMPLE_RATE,
                                                   DODOMIC_SAMPLE_RATE,
                                                   RECORD_DURATION, freq)
                sdg.upload_waveform(f"ss{int(freq)}", sdg_waveform, freq, ampl_vpp=2.0)
                sdg.output_on(channel=1)
                time.sleep(0.2)
                received = record_audio(pa, mic_index, RECORD_DURATION)
                sdg.output_off(channel=1)

                n_fft = len(received)
                fft_rx = np.fft.fft(received)
                wf_matched = np.zeros(n_fft)
                wf_len = min(len(wf), n_fft)
                wf_matched[:wf_len] = wf[:wf_len]
                fft_tx = np.fft.fft(wf_matched)
                freqs_fft = np.fft.fftfreq(n_fft, 1/DODOMIC_SAMPLE_RATE)
                idx = np.argmin(np.abs(freqs_fft - freq))
                H = fft_rx[idx] / (fft_tx[idx] + 1e-10)
                mag_db = 20 * np.log10(np.abs(H) + 1e-10)
                phase_rad = np.angle(H)

                if 40000 <= freq <= 41000 and received_example is None:
                    received_example = received.copy()

            transfer_magnitudes.append(mag_db)
            transfer_phases.append(phase_rad)

            # Compact progress
            print(f"  {freq/1000:.1f} kHz: {mag_db:.1f} dB", end="\r")

        print("\n" + "-" * 60)
        print("Sweep complete!")
        print("-" * 60)

    except Exception as e:
        print(f"\nError during measurement: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if sdg:
            sdg.output_off(channel=1)
            sdg.disconnect()
        if pa:
            pa.terminate()

    # Convert to arrays
    freqs_measured = np.array(freq_list)
    H_mag = np.array(transfer_magnitudes)
    H_phase = np.array(transfer_phases)

    # Find bandwidth at -3dB and -6dB
    print("\nAnalyzing results...")
    try:
        f_center_3db, f_low_3db, f_high_3db, bw_3db, peak_freq, peak_mag, H_mag_smooth = find_bandwidth(
            freqs_measured, H_mag, threshold_db=-3, search_range=(35000, 50000), smooth_window=11
        )
        f_center_6db, f_low_6db, f_high_6db, bw_6db, _, _, _ = find_bandwidth(
            freqs_measured, H_mag, threshold_db=-6, search_range=(35000, 50000), smooth_window=11
        )

        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"  Center frequency: {f_center_3db/1000:.2f} kHz")
        print(f"  Peak frequency:   {peak_freq/1000:.2f} kHz")
        print(f"  -3 dB bandwidth:  {bw_3db/1000:.2f} kHz")
        print(f"  -3 dB passband:   {f_low_3db/1000:.2f} - {f_high_3db/1000:.2f} kHz")
        print(f"  -6 dB bandwidth:  {bw_6db/1000:.2f} kHz")
        print(f"  -6 dB passband:   {f_low_6db/1000:.2f} - {f_high_6db/1000:.2f} kHz")
        print(f"  Q factor:         {f_center_3db/bw_3db:.1f}")
        print(f"  Min chip duration: {1e6/bw_3db:.0f} us")

    except Exception as e:
        print(f"Warning: Could not find bandwidth: {e}")
        H_mag_smooth = H_mag
        f_center_3db = f_low_3db = f_high_3db = bw_3db = 0
        f_center_6db = f_low_6db = f_high_6db = bw_6db = 0
        peak_mag = 0

    # Ensure we have an example recording
    if received_example is None:
        received_example = np.zeros(1000)

    from datetime import datetime as _dt
    stamp = _dt.now().strftime('%Y%m%d_%H%M%S')
    mode_tag = f"burst{BURST_DURATION_MS:.0f}ms" if BURST_MODE else "300ms"

    # Save raw data
    data_path = OUTPUT_DIR + f'stepped_sine_characterization_{mode_tag}_{stamp}.npz'
    np.savez(data_path,
             freqs=freqs_measured,
             H_mag=H_mag,
             H_mag_smooth=H_mag_smooth,
             H_phase=H_phase,
             f_center=f_center_3db,
             f_low=f_low_3db,
             f_high=f_high_3db,
             bandwidth=bw_3db,
             f_center_6db=f_center_6db,
             f_low_6db=f_low_6db,
             f_high_6db=f_high_6db,
             bandwidth_6db=bw_6db)
    print(f"\nSaved data: {data_path}")

    # Generate plots
    print("\nGenerating plots...")

    # Standalone plot
    plot_stepped_sine_result(
        freqs_measured, H_mag, H_mag_smooth, H_phase,
        f_center_3db, f_low_3db, f_high_3db, bw_3db,
        f_center_6db, f_low_6db, f_high_6db, bw_6db,
        peak_mag, received_example,
        OUTPUT_DIR + f'stepped_sine_characterization_{mode_tag}_{stamp}.png',
        chirp_data_path=OUTPUT_DIR + 'piezo_characterization.npz'
    )

    # Comparison plot
    plot_comparison_with_chirp(
        freqs_measured, H_mag_smooth, f_center_3db, f_low_3db, f_high_3db, bw_3db,
        chirp_data_path=OUTPUT_DIR + 'piezo_characterization.npz',
        output_path=OUTPUT_DIR + f'stepped_sine_vs_chirp_comparison_{mode_tag}_{stamp}.png'
    )

    print("\nDone!")
    return 0


if __name__ == '__main__':
    exit(main())
