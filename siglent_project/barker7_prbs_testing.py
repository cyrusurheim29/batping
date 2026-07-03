#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Barker-7 PRBS TOF Testing (V5)

Characterizes how cycles-per-chip (16-48) affects:
1. Autocorrelation peak sharpness (FWHM)
2. Sidelobe levels (relative to main peak)
3. TOF measurement precision

V5 fixes:
- Sidelobe level now relative to main peak (should be ~-16.9 dB for Barker-7)
- Transfer efficiency normalized by signal length
- Data archival with WAV files and manifest.json

Based on stepped-sine bandwidth characterization:
- Piezo bandwidth: 1.70 kHz at 41.35 kHz center
- Min chip duration: T_c = 1/BW = 588 μs (24 cycles)

@author: cyrusurheimog
Created: June 16, 2026
Updated: June 17, 2026 (V5)
"""

import numpy as np
import pyaudio
import socket
import time
import os
import json
from datetime import datetime
from scipy import signal
from scipy.signal import butter, filtfilt
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Hardware configuration
SDG_IP = "10.11.13.230"
SDG_PORT = 5025
SDG_SAMPLE_RATE = 400000  # 400 kSa/s for SDG
DODOMIC_SAMPLE_RATE = 384000  # UltraMic sample rate
CHUNK_SIZE = 4096

# Waveform configuration
CARRIER_FREQ = 41350  # Hz (center from stepped-sine characterization)
AMPLITUDE = 0.9
RECORD_DURATION = 0.5  # seconds
SDG_MAX_SAMPLES = 16384  # SDG1062X waveform point limit

# V3: Inter-pulse spacing for piezo settling
TARGET_REP_RATE = 40  # Hz (25 ms period allows ~20 ms settling after ~4 ms waveform)

# Barker codes (all known Barker sequences)
BARKER_CODES = {
    2:  [+1, -1],
    3:  [+1, +1, -1],
    4:  [+1, +1, -1, +1],  # or [+1, +1, +1, -1]
    5:  [+1, +1, +1, -1, +1],
    7:  [+1, +1, +1, -1, -1, +1, -1],
    11: [+1, +1, +1, -1, -1, -1, +1, -1, -1, +1, -1],
    13: [+1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, -1, +1],
}

# Theoretical sidelobe levels for Barker codes (1/N)
BARKER_SIDELOBE_DB = {
    7: -16.9,   # 20*log10(1/7)
    11: -20.8,  # 20*log10(1/11)
    13: -22.3,  # 20*log10(1/13)
}

# V5: Version string for data archival
VERSION = "V5"

OUTPUT_DIR = '/Users/cyrusurheim_1/Downloads/Stubbs_Stuff/Claude_Dir/siglent_project/'


class SDG1062X_LAN:
    """Siglent SDG1062X remote control via LAN connection."""

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


def generate_barker_bpsk(code, carrier_freq, cycles_per_chip, sample_rate, amplitude=0.9):
    """
    Generate BPSK-modulated Barker code.

    Args:
        code: List of +1/-1 chip values
        carrier_freq: Carrier frequency (Hz)
        cycles_per_chip: Number of carrier cycles per chip
        sample_rate: Sample rate (Hz)
        amplitude: Peak amplitude (0-1)

    Returns:
        t: Time array
        waveform: BPSK modulated signal
    """
    chip_duration = cycles_per_chip / carrier_freq  # seconds
    samples_per_chip = int(chip_duration * sample_rate)
    total_samples = samples_per_chip * len(code)

    t = np.arange(total_samples) / sample_rate
    waveform = np.zeros(total_samples)

    for i, chip in enumerate(code):
        start = i * samples_per_chip
        end = start + samples_per_chip
        t_chip = np.arange(samples_per_chip) / sample_rate
        # BPSK: phase is 0 for +1, π for -1
        waveform[start:end] = chip * amplitude * np.sin(2 * np.pi * carrier_freq * t_chip)

    # Apply Tukey window to smooth edges
    window = signal.windows.tukey(total_samples, alpha=0.1)
    waveform = waveform * window

    return t, waveform


def create_sdg_waveform(waveform, sdg_sample_rate, source_sample_rate):
    """Resample waveform for SDG and normalize."""
    n_samples_sdg = int(len(waveform) * sdg_sample_rate / source_sample_rate)
    waveform_resampled = signal.resample(waveform, n_samples_sdg)

    # Normalize to [-0.9, 0.9]
    max_val = np.max(np.abs(waveform_resampled))
    if max_val > 0:
        waveform_resampled = waveform_resampled / max_val * 0.9

    return waveform_resampled


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


def bandpass_filter(sig, fs, low=35000, high=45000, order=4):
    """Apply bandpass filter around piezo resonance."""
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, sig)


def analyze_theoretical_autocorrelation(waveform, fs, chip_duration_samples=None, n_chips=None):
    """
    Compute autocorrelation of the THEORETICAL transmitted waveform.

    This gives the IDEAL pulse compression properties assuming perfect
    transmission. The actual achieved resolution will be worse due to
    piezo bandwidth limitations.

    Returns raw autocorr for display and envelope for metrics.
    """
    from scipy.signal import hilbert

    # Bandpass filter
    wf_filt = bandpass_filter(waveform, fs, 35000, 45000)

    # Autocorrelation (raw, for display)
    autocorr_raw = np.correlate(wf_filt, wf_filt, mode='full')
    autocorr_raw = autocorr_raw / np.max(np.abs(autocorr_raw))  # Normalize

    center = len(autocorr_raw) // 2
    lags_samples = np.arange(len(autocorr_raw)) - center
    lags_us = lags_samples / fs * 1e6

    # Extract envelope via Hilbert transform
    autocorr_envelope = np.abs(hilbert(autocorr_raw))
    autocorr_envelope = autocorr_envelope / np.max(autocorr_envelope)

    # Theoretical FWHM and sidelobes from envelope
    # Pass chip info for targeted sidelobe search at integer chip-multiple lags
    fwhm_us, sidelobe_db = measure_envelope_metrics(
        autocorr_envelope, lags_samples, fs,
        chip_duration_samples=chip_duration_samples, n_chips=n_chips
    )

    return autocorr_raw, lags_us, fwhm_us, sidelobe_db, autocorr_envelope


def measure_envelope_metrics(envelope, lags_samples, fs, chip_duration_samples=None, n_chips=None):
    """
    Measure FWHM and peak sidelobe from a normalized envelope.

    V5 fix: Sidelobe is now RELATIVE to main peak (dB below main peak).
    Barker-7 ideal = 1/7 → -16.9 dB

    If chip_duration_samples and n_chips are provided, uses a targeted search
    at each integer chip-multiple lag (±25% window) rather than a blind threshold.
    This correctly handles short chips where the fixed 1ms exclusion zone was
    swallowing all sidelobes.
    """
    center = len(envelope) // 2
    main_peak = envelope[center]  # Main peak at lag=0

    # Find FWHM
    half_max = main_peak * 0.5
    above_half = envelope >= half_max

    left_region = above_half[:center+1][::-1]
    right_region = above_half[center:]

    left_crossing = np.argmax(left_region == False) if np.any(left_region == False) else len(left_region)
    right_crossing = np.argmax(right_region == False) if np.any(right_region == False) else len(right_region)

    fwhm_samples = left_crossing + right_crossing
    fwhm_us = fwhm_samples / fs * 1e6

    # Find peak sidelobe level RELATIVE to main peak
    if chip_duration_samples is not None and n_chips is not None:
        # Targeted search: look in ±25% window around each integer chip-multiple lag.
        # This is accurate regardless of chip duration — no fixed exclusion zone.
        search_half = max(chip_duration_samples // 4, 5)
        max_sl = 0.0
        for k in range(1, n_chips):
            target = k * chip_duration_samples
            sl_mask = (np.abs(lags_samples) > target - search_half) & \
                      (np.abs(lags_samples) < target + search_half)
            if np.any(sl_mask):
                max_sl = max(max_sl, np.max(envelope[sl_mask]))
        peak_sidelobe_db = 20 * np.log10(max_sl / main_peak + 1e-10) if max_sl > 0 else -np.inf
    else:
        # Fallback: blind search outside main lobe
        main_lobe_half_width = max(fwhm_samples * 2, int(0.001 * fs))
        sidelobe_mask = np.abs(lags_samples) > main_lobe_half_width
        if np.any(sidelobe_mask):
            peak_sidelobe = np.max(envelope[sidelobe_mask])
            peak_sidelobe_db = 20 * np.log10(peak_sidelobe / main_peak + 1e-10)
        else:
            peak_sidelobe_db = -np.inf

    return fwhm_us, peak_sidelobe_db


def analyze_received_autocorrelation(received_filt, fs):
    """
    V4: Autocorrelation of RECEIVED signal to measure actual FWHM and TOF.

    This is better than matched filtering because:
    1. No template mismatch issues (measures what we actually received)
    2. TOF emerges naturally from peak spacing (incident ⊗ reflected)
    3. FWHM from main peak reflects true achievable resolution

    Autocorrelation structure:
    - Main peak at lag=0 (always largest) → FWHM
    - Secondary peaks at lag=±TOF (incident ⊗ reflected cross-terms)
    - Peaks at lag=±rep_period (consecutive pulses)

    Returns:
        autocorr: raw autocorrelation
        autocorr_envelope: envelope via Hilbert transform
        lags_us: lag axis in microseconds
        fwhm_us: FWHM measured from main peak envelope
        sidelobe_db: peak sidelobe level
        tof_us: time-of-flight from secondary peak
    """
    from scipy.signal import hilbert, find_peaks

    # Autocorrelation
    autocorr = np.correlate(received_filt, received_filt, mode='full')
    center = len(autocorr) // 2
    lags_samples = np.arange(len(autocorr)) - center
    lags_us = lags_samples / fs * 1e6

    # Normalize
    autocorr = autocorr / np.max(np.abs(autocorr))

    # Extract envelope
    autocorr_envelope = np.abs(hilbert(autocorr))
    autocorr_envelope = autocorr_envelope / np.max(autocorr_envelope)

    # --- FWHM from main peak (at lag=0) ---
    peak_val = autocorr_envelope[center]
    half_max = 0.5

    # Search left from center
    left_idx = center
    while left_idx > 0 and autocorr_envelope[left_idx] > half_max:
        left_idx -= 1

    # Search right from center
    right_idx = center
    while right_idx < len(autocorr_envelope) - 1 and autocorr_envelope[right_idx] > half_max:
        right_idx += 1

    fwhm_samples = right_idx - left_idx
    fwhm_us = fwhm_samples / fs * 1e6

    # --- Sidelobe level RELATIVE to main peak (V5 fix) ---
    # Look outside the main lobe but within one pulse duration
    main_lobe_half_width = max(fwhm_samples * 3, int(0.002 * fs))  # 2 ms minimum
    pulse_region_width = int(0.010 * fs)  # 10 ms - look within one pulse
    main_peak = autocorr_envelope[center]

    sidelobe_mask = (np.abs(lags_samples) > main_lobe_half_width) & (np.abs(lags_samples) < pulse_region_width)

    if np.any(sidelobe_mask):
        sidelobe_region = autocorr_envelope[sidelobe_mask]
        peak_sidelobe = np.max(sidelobe_region)
        # V5: Relative to main peak (should be ~-16.9 dB for Barker-7)
        sidelobe_db = 20 * np.log10(peak_sidelobe / main_peak + 1e-10)
    else:
        sidelobe_db = -np.inf

    # --- TOF from secondary peaks ---
    # The autocorrelation has peaks at:
    # - lag = 0 (main peak)
    # - lag = ±TOF (incident ⊗ reflected)
    # - lag = ±rep_period (consecutive pulses, ~25ms for 40Hz)
    #
    # TOF for 2m round trip ≈ 5800 μs, so look in range [3000, 15000] μs

    # Search only positive lags for TOF
    positive_mask = (lags_us > 3000) & (lags_us < 15000)

    tof_fwhm_us = 0.0
    if np.any(positive_mask):
        positive_env = autocorr_envelope[positive_mask]
        positive_lags = lags_us[positive_mask]

        # Find the highest peak in this region
        tof_peak_idx = np.argmax(positive_env)
        tof_us = positive_lags[tof_peak_idx]
        tof_peak_val = positive_env[tof_peak_idx]

        # Measure FWHM of the TOF peak — this is the pulse compression metric
        # (how sharp the incident⊗reflected cross-correlation is)
        global_tof_idx = np.where(positive_mask)[0][tof_peak_idx]
        half_tof = tof_peak_val * 0.5
        li = global_tof_idx
        while li > 0 and autocorr_envelope[li] > half_tof:
            li -= 1
        ri = global_tof_idx
        while ri < len(autocorr_envelope) - 1 and autocorr_envelope[ri] > half_tof:
            ri += 1
        tof_fwhm_us = (ri - li) / fs * 1e6
    else:
        tof_us = 0
        tof_peak_val = 0

    return autocorr, autocorr_envelope, lags_us, fwhm_us, sidelobe_db, tof_us, tof_fwhm_us


def measure_barker_response(sdg, pa, mic_index, waveform_tx, name, record_duration=RECORD_DURATION):
    """Upload Barker waveform with inter-pulse spacing, record response, compute correlation."""
    # Upload to SDG - generate at SDG sample rate directly to preserve carrier freq
    sdg_wf = create_sdg_waveform(waveform_tx, SDG_SAMPLE_RATE, DODOMIC_SAMPLE_RATE)

    # V3: Embed in larger buffer for piezo settling time
    buffer_samples = int(SDG_SAMPLE_RATE / TARGET_REP_RATE)  # 10,000 samples for 40 Hz

    if buffer_samples > SDG_MAX_SAMPLES:
        buffer_samples = SDG_MAX_SAMPLES
        actual_rep_rate = SDG_SAMPLE_RATE / buffer_samples
        print(f"  Warning: Using max buffer, rep_rate = {actual_rep_rate:.1f} Hz")
    else:
        actual_rep_rate = TARGET_REP_RATE

    # Create buffer with waveform at start, zeros for settling
    waveform_buffer = np.zeros(buffer_samples)
    waveform_buffer[:len(sdg_wf)] = sdg_wf

    wf_duration_ms = len(sdg_wf) / SDG_SAMPLE_RATE * 1000
    dead_time_ms = (buffer_samples - len(sdg_wf)) / SDG_SAMPLE_RATE * 1000
    print(f"  Waveform: {wf_duration_ms:.1f} ms, Dead time: {dead_time_ms:.1f} ms, Rep: {actual_rep_rate:.0f} Hz")

    sdg.upload_waveform(name, waveform_buffer, actual_rep_rate, ampl_vpp=2.0)
    sdg.output_on(channel=1)
    time.sleep(0.3)  # Let piezo settle

    # Record
    received = record_audio(pa, mic_index, record_duration)
    sdg.output_off(channel=1)

    # Bandpass filter both signals
    received_filt = bandpass_filter(received, DODOMIC_SAMPLE_RATE, 35000, 45000)
    template_filt = bandpass_filter(waveform_tx, DODOMIC_SAMPLE_RATE, 35000, 45000)

    # Cross-correlation
    corr = np.correlate(received_filt, template_filt, mode='full')
    lags = np.arange(-len(template_filt) + 1, len(received_filt))
    lags_us = lags / DODOMIC_SAMPLE_RATE * 1e6

    return received, received_filt, corr, lags_us


def find_correlation_peak(corr, lags_us, search_range_us=(0, 10000)):
    """Find the correlation peak within a search range."""
    # Only look at positive lags (causal response)
    mask = (lags_us >= search_range_us[0]) & (lags_us <= search_range_us[1])
    corr_search = np.abs(corr[mask])
    lags_search = lags_us[mask]

    peak_idx = np.argmax(corr_search)
    tof_us = lags_search[peak_idx]
    peak_value = corr_search[peak_idx]

    return tof_us, peak_value


def compute_transfer_efficiency(waveform_tx, received_filt, corr_peak, fs):
    """
    Compute transfer efficiency: ratio of measured correlation peak to theoretical autocorrelation peak.

    This characterizes signal attenuation through the acoustic path.
    Higher values = more signal survives = chip duration within piezo bandwidth.

    Returns:
        efficiency: Ratio (0-1 typical, can exceed 1 due to noise/gain)
        efficiency_db: In decibels
        autocorr_peak: The theoretical autocorrelation peak value
    """
    # Compute theoretical autocorrelation peak (matched filter output for perfect transmission)
    template_filt = bandpass_filter(waveform_tx, fs, 35000, 45000)
    autocorr = np.correlate(template_filt, template_filt, mode='full')
    autocorr_peak = np.max(np.abs(autocorr))

    # Transfer efficiency = measured peak / theoretical peak
    efficiency = corr_peak / autocorr_peak if autocorr_peak > 0 else 0
    efficiency_db = 20 * np.log10(efficiency + 1e-10)

    return efficiency, efficiency_db, autocorr_peak


def plot_barker_result(result, output_path):
    """Create 4-panel plot for individual Barker test result."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    cycles = result['cycles_per_chip']
    wf = result['waveform']
    received = result['received']

    # V4: Use received autocorrelation instead of cross-correlation
    rx_autocorr = result.get('rx_autocorr', result.get('correlation', np.array([])))
    rx_env = result.get('rx_autocorr_envelope', np.abs(rx_autocorr))
    rx_lags_us = result.get('rx_autocorr_lags_us', result.get('lags_us', np.array([])))

    t_wf = np.arange(len(wf)) / DODOMIC_SAMPLE_RATE * 1000  # ms
    t_rx = np.arange(len(received)) / DODOMIC_SAMPLE_RATE * 1000  # ms

    # Panel 1: Transmitted waveform (theoretical)
    ax1 = axes[0, 0]
    ax1.plot(t_wf, wf, 'b-', linewidth=0.5)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'Theoretical Barker-7 BPSK ({cycles} cycles/chip)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, t_wf[-1]])

    # Panel 2: Received waveform (zoom to first 50 ms)
    ax2 = axes[0, 1]
    ax2.plot(t_rx, received, 'b-', linewidth=0.5)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Received Signal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, min(50, t_rx[-1])])

    # Panel 3: Received Autocorrelation (V4 - shows FWHM and TOF)
    ax3 = axes[1, 0]
    # Plot envelope and raw, zoom to ±15 ms to see main peak and TOF peak
    if len(rx_lags_us) > 0:
        mask = (rx_lags_us >= -5000) & (rx_lags_us <= 15000)
        ax3.plot(rx_lags_us[mask] / 1000, rx_autocorr[mask], 'b-', linewidth=0.5, alpha=0.5, label='Raw')
        ax3.plot(rx_lags_us[mask] / 1000, rx_env[mask], 'b-', linewidth=1.5, label='Envelope')

        # Mark TOF peak
        tof_us = result['tof_us']
        ax3.axvline(tof_us / 1000, color='r', linestyle='--', alpha=0.7, label=f'TOF: {tof_us:.0f} μs')
        ax3.axvline(0, color='gray', linestyle=':', alpha=0.5, label='lag=0')

    ax3.set_xlabel('Lag (ms)')
    ax3.set_ylabel('Autocorrelation')
    ax3.set_title('Received Signal Autocorrelation')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Info panel
    ax4 = axes[1, 1]
    ax4.axis('off')

    eff_db = result.get('transfer_efficiency_db', 0)
    eff_pct = result.get('transfer_efficiency', 0) * 100
    tof_us = result.get('tof_us', 0)
    theo_fwhm = result.get('theo_fwhm_us', result.get('autocorr_fwhm_us', 0))
    theo_sl = result.get('theo_sidelobe_db', result.get('sidelobe_db', 0))
    meas_fwhm = result.get('measured_fwhm_us', 0)
    meas_sl = result.get('measured_sidelobe_db', 0)

    info_text = (
        f"BARKER-7 BPSK TEST (V4)\n"
        f"{'='*35}\n\n"
        f"WAVEFORM PARAMETERS\n"
        f"  Carrier frequency: {CARRIER_FREQ/1000:.1f} kHz\n"
        f"  Cycles per chip: {cycles}\n"
        f"  Chip duration: {result['chip_duration_us']:.0f} μs\n"
        f"  Total duration: {result['total_duration_ms']:.2f} ms\n"
        f"\n"
        f"THEORETICAL (TX autocorr)\n"
        f"  FWHM: {theo_fwhm:.0f} μs\n"
        f"  Peak sidelobe: {theo_sl:.1f} dB\n"
        f"\n"
        f"MEASURED (RX autocorr)\n"
        f"  FWHM: {meas_fwhm:.0f} μs\n"
        f"  Peak sidelobe: {meas_sl:.1f} dB\n"
        f"  TOF: {tof_us:.0f} μs\n"
        f"  Transfer eff: {eff_db:.1f} dB"
    )

    ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(f'Barker-7 PRBS Test — {cycles} Cycles/Chip', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_cycles_sweep_summary(results, output_path):
    """Create summary plot of cycles/chip sweep."""
    cycles_list = [r['cycles_per_chip'] for r in results]
    theo_fwhm_list = [r.get('theo_fwhm_us', r.get('autocorr_fwhm_us', 0)) for r in results]
    measured_fwhm_list = [r.get('measured_fwhm_us', 0) for r in results]
    theo_sidelobe_list = [r.get('theo_sidelobe_db', r.get('sidelobe_db', 0)) for r in results]
    measured_sidelobe_list = [r.get('measured_sidelobe_db', 0) for r in results]
    tof_list = [r['tof_us'] for r in results]
    efficiency_db_list = [r['transfer_efficiency_db'] for r in results]

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

    # Panel 1: FWHM vs cycles/chip (theoretical vs measured)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(cycles_list, theo_fwhm_list, 'b^--', linewidth=1.5, markersize=6, label='Theoretical')
    ax1.plot(cycles_list, measured_fwhm_list, 'bo-', linewidth=2, markersize=8, label='Measured')
    ax1.set_xlabel('Cycles per Chip')
    ax1.set_ylabel('FWHM (μs)')
    ax1.set_title('Peak Sharpness (FWHM)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Transfer efficiency vs cycles/chip
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(cycles_list, efficiency_db_list, 'mo-', linewidth=2, markersize=8)
    ax2.set_xlabel('Cycles per Chip')
    ax2.set_ylabel('Transfer Efficiency (dB)')
    ax2.set_title('Signal Attenuation')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(24, color='orange', linestyle='--', alpha=0.5, label='BW limit (24 cyc)')
    ax2.legend(loc='lower right', fontsize=8)

    # Panel 3: Sidelobe level vs cycles/chip (theoretical vs measured)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(cycles_list, theo_sidelobe_list, 'r^--', linewidth=1.5, markersize=6, label='Theoretical')
    ax3.plot(cycles_list, measured_sidelobe_list, 'ro-', linewidth=2, markersize=8, label='Measured')
    ax3.axhline(-16.9, color='gray', linestyle=':', alpha=0.7, label='Barker-7 ideal')
    ax3.set_xlabel('Cycles per Chip')
    ax3.set_ylabel('Peak Sidelobe (dB)')
    ax3.set_title('Sidelobe Level')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel 4: TOF consistency
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.plot(cycles_list, tof_list, 'go-', linewidth=2, markersize=8)
    ax4.set_xlabel('Cycles per Chip')
    ax4.set_ylabel('Measured TOF (μs)')
    ax4.set_title('TOF Consistency')
    ax4.grid(True, alpha=0.3)

    # Panel 5: Overlay of RECEIVED autocorrelation envelopes (what measured FWHM comes from)
    ax5 = fig.add_subplot(gs[1, 0:2])
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    for i, r in enumerate(results):
        # Plot received autocorr envelope (V4 - actual measured resolution)
        rx_env = r.get('rx_autocorr_envelope', r.get('autocorr_envelope', np.array([])))
        rx_lags = r.get('rx_autocorr_lags_us', r.get('autocorr_lags_us', np.array([])))
        if len(rx_lags) > 0:
            mask = (rx_lags >= -2000) & (rx_lags <= 2000)  # Zoom to main peak
            ax5.plot(rx_lags[mask], rx_env[mask], color=colors[i], linewidth=1.5,
                     label=f"{r['cycles_per_chip']} cyc")
    ax5.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='FWHM level')
    ax5.set_xlabel('Lag (μs)')
    ax5.set_ylabel('Envelope (normalized)')
    ax5.set_title('Received Autocorrelation Envelopes (measured FWHM)')
    ax5.legend(loc='upper right', fontsize=8, ncol=2)
    ax5.grid(True, alpha=0.3)

    # Panel 6: Measured FWHM vs Transfer Efficiency trade-off
    ax6 = fig.add_subplot(gs[1, 2])
    scatter = ax6.scatter(measured_fwhm_list, efficiency_db_list, c=cycles_list, cmap='viridis', s=100)
    for i, cyc in enumerate(cycles_list):
        ax6.annotate(f'{cyc}', (measured_fwhm_list[i], efficiency_db_list[i]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax6.set_xlabel('Measured FWHM (μs)')
    ax6.set_ylabel('Transfer Efficiency (dB)')
    ax6.set_title('Trade-off: Sharpness vs Attenuation')
    ax6.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax6)
    cbar.set_label('Cycles/chip')

    # Panel 7: Info/recommendations
    ax7 = fig.add_subplot(gs[1, 3])
    ax7.axis('off')

    # Find optimal (balance FWHM and efficiency)
    # Simple heuristic: best measured FWHM among those with efficiency > -30 dB
    viable_mask = np.array(efficiency_db_list) > -30
    if np.any(viable_mask):
        viable_fwhm = [f for f, v in zip(measured_fwhm_list, viable_mask) if v]
        viable_cycles = [c for c, v in zip(cycles_list, viable_mask) if v]
        best_idx_viable = np.argmin(viable_fwhm)
        best_cycles = viable_cycles[best_idx_viable]
        best_fwhm = viable_fwhm[best_idx_viable]
        best_eff = efficiency_db_list[cycles_list.index(best_cycles)]
    else:
        best_idx = np.argmin(measured_fwhm_list)
        best_cycles = cycles_list[best_idx]
        best_fwhm = measured_fwhm_list[best_idx]
        best_eff = efficiency_db_list[best_idx]

    tof_mean = np.mean(tof_list)
    tof_std = np.std(tof_list)

    # Find where efficiency drops off
    eff_threshold = max(efficiency_db_list) - 6  # 6 dB below max
    eff_dropoff_cycles = None
    for i, (c, e) in enumerate(zip(cycles_list, efficiency_db_list)):
        if e < eff_threshold:
            eff_dropoff_cycles = c
            break

    info_text = (
        f"SWEEP SUMMARY\n"
        f"{'='*30}\n\n"
        f"Cycles tested: {min(cycles_list)}-{max(cycles_list)}\n"
        f"Number of tests: {len(results)}\n"
        f"\n"
        f"TRANSFER EFFICIENCY\n"
        f"  Max: {max(efficiency_db_list):.1f} dB\n"
        f"  -6dB dropoff: {eff_dropoff_cycles or 'N/A'} cyc\n"
        f"\n"
        f"OPTIMAL (sharpest viable)\n"
        f"  Cycles/chip: {best_cycles}\n"
        f"  FWHM: {best_fwhm:.0f} μs\n"
        f"  Efficiency: {best_eff:.1f} dB\n"
        f"\n"
        f"TOF CONSISTENCY\n"
        f"  Mean: {tof_mean:.1f} μs\n"
        f"  Std: {tof_std:.1f} μs\n"
    )

    ax7.text(0.05, 0.95, info_text, transform=ax7.transAxes, fontsize=9,
             verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    fig.suptitle('Barker-7 Cycles/Chip Sweep Summary', fontsize=14, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


class DataArchiver:
    """V5: Data archival system with WAV files and manifest."""

    def __init__(self, base_dir, description="barker7_sweep"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(base_dir, "recordings", f"{timestamp}_{description}")
        self.raw_dir = os.path.join(self.run_dir, "raw")
        self.plots_dir = os.path.join(self.run_dir, "plots")

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        self.manifest = {
            "timestamp": datetime.now().isoformat(),
            "version": VERSION,
            "description": description,
            "hardware": {
                "sdg": f"SDG1062X @ {SDG_IP}",
                "sdg_sample_rate_hz": SDG_SAMPLE_RATE,
                "mic": "Dodotronic UltraMic",
                "mic_sample_rate_hz": DODOMIC_SAMPLE_RATE,
                "piezo_center_freq_hz": CARRIER_FREQ,
                "piezo_bandwidth_hz": 1700,
            },
            "parameters": {},
            "files": {},
        }
        print(f"Data archiver initialized: {self.run_dir}")

    def set_parameters(self, **kwargs):
        self.manifest["parameters"].update(kwargs)

    def save_wav(self, name, audio_data, sample_rate):
        """Save audio as WAV file."""
        wav_path = os.path.join(self.raw_dir, f"{name}.wav")
        # Normalize to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wavfile.write(wav_path, sample_rate, audio_int16)
        return os.path.relpath(wav_path, self.run_dir)

    def save_plot(self, name, fig_path_source):
        """Copy plot to archive."""
        import shutil
        dest_path = os.path.join(self.plots_dir, f"{name}.png")
        shutil.copy(fig_path_source, dest_path)
        return os.path.relpath(dest_path, self.run_dir)

    def add_result(self, name, result_dict, wav_rel_path=None, plot_rel_path=None):
        """Add a result entry to the manifest."""
        entry = {k: v for k, v in result_dict.items()
                 if not isinstance(v, np.ndarray)}  # Exclude large arrays
        if wav_rel_path:
            entry["wav"] = wav_rel_path
        if plot_rel_path:
            entry["plot"] = plot_rel_path
        self.manifest["files"][name] = entry

    def save_manifest(self):
        """Save manifest.json."""
        manifest_path = os.path.join(self.run_dir, "manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2, default=str)
        print(f"Manifest saved: {manifest_path}")

    def save_results_npz(self, results):
        """Save numerical results as NPZ."""
        npz_path = os.path.join(self.run_dir, "results.npz")
        results_compact = [{k: v for k, v in r.items()
                           if not isinstance(v, np.ndarray)} for r in results]
        np.savez(npz_path, results=results_compact)
        print(f"Results saved: {npz_path}")
        return npz_path


def main():
    print("=" * 60)
    print(f"Barker-7 PRBS TOF Testing ({VERSION})")
    print("=" * 60)

    # Full sweep: steps of 2 from 2-24, steps of 1 in transition regime 8-12, steps of 4 from 24-48
    CYCLES_PER_CHIP_RANGE = [2, 4, 6, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40, 44, 48]
    BARKER_CODE = BARKER_CODES[7]
    BARKER_LEN = 7

    print(f"\nCarrier frequency: {CARRIER_FREQ/1000:.1f} kHz")
    print(f"Barker code length: {len(BARKER_CODE)} chips")
    print(f"Cycles/chip range: {list(CYCLES_PER_CHIP_RANGE)}")
    print(f"Number of tests: {len(list(CYCLES_PER_CHIP_RANGE))}")
    print(f"Theoretical sidelobe level: {BARKER_SIDELOBE_DB[BARKER_LEN]:.1f} dB")

    results = []

    # V5: Initialize data archiver
    archiver = DataArchiver(OUTPUT_DIR, description=f"barker{BARKER_LEN}_sweep_lowcyc")
    archiver.set_parameters(
        barker_length=BARKER_LEN,
        cycles_per_chip=list(CYCLES_PER_CHIP_RANGE),
        carrier_freq_hz=CARRIER_FREQ,
        rep_rate_hz=TARGET_REP_RATE,
        record_duration_s=RECORD_DURATION,
    )

    # Connect hardware
    pa, mic_index = find_ultramic()
    if pa is None:
        return 1

    sdg = None
    try:
        sdg = SDG1062X_LAN()
        sdg.connect()
        idn = sdg.query("*IDN?")
        print(f"\nDevice: {idn}")
        sdg.set_arb_mode(channel=1)

        for cycles in CYCLES_PER_CHIP_RANGE:
            print(f"\n{'='*50}")
            print(f"Testing {cycles} cycles/chip")
            print(f"{'='*50}")

            # Generate waveform
            t, wf = generate_barker_bpsk(BARKER_CODE, CARRIER_FREQ, cycles, DODOMIC_SAMPLE_RATE)
            chip_duration_us = cycles / CARRIER_FREQ * 1e6
            total_duration_ms = len(BARKER_CODE) * chip_duration_us / 1000

            print(f"  Chip duration: {chip_duration_us:.0f} μs")
            print(f"  Total duration: {total_duration_ms:.2f} ms")
            print(f"  Waveform samples: {len(wf)}")

            # Samples per chip for targeted sidelobe search
            chip_duration_samples = int(round(cycles / CARRIER_FREQ * DODOMIC_SAMPLE_RATE))

            # Analyze THEORETICAL autocorrelation (ideal pulse compression)
            theo_autocorr, theo_lags_us, theo_fwhm_us, theo_sidelobe_db, theo_env = analyze_theoretical_autocorrelation(
                wf, DODOMIC_SAMPLE_RATE,
                chip_duration_samples=chip_duration_samples, n_chips=len(BARKER_CODE)
            )
            print(f"  Theoretical FWHM: {theo_fwhm_us:.0f} μs")
            print(f"  Theoretical sidelobe: {theo_sidelobe_db:.1f} dB")

            # Measure actual response
            print("  Recording response...")
            received, received_filt, _, _ = measure_barker_response(
                sdg, pa, mic_index, wf, f"bk7_{cycles}_v6"
            )

            # V4: Autocorrelation of RECEIVED signal for actual FWHM and TOF
            # This measures what we actually achieved, not theoretical
            rx_autocorr, rx_env, rx_lags_us, measured_fwhm_us, measured_sidelobe_db, tof_us, tof_fwhm_us = analyze_received_autocorrelation(
                received_filt, DODOMIC_SAMPLE_RATE
            )
            print(f"  Measured lag-0 FWHM: {measured_fwhm_us:.0f} μs (bandwidth-limited, same for all waveforms)")
            print(f"  Measured TOF peak FWHM: {tof_fwhm_us:.0f} μs (pulse compression metric)")
            print(f"  Measured sidelobe: {measured_sidelobe_db:.1f} dB")
            print(f"  TOF: {tof_us:.1f} μs (from autocorr peak spacing)")

            # V5: Transfer efficiency - normalize by signal length
            # This accounts for different-length signals (theo ~4ms, rx 500ms)
            theo_peak_norm = np.max(np.abs(theo_autocorr)) / len(wf)
            rx_peak_norm = np.max(np.abs(rx_autocorr)) / len(received_filt)
            efficiency = rx_peak_norm / theo_peak_norm if theo_peak_norm > 0 else 0
            efficiency_db = 20 * np.log10(efficiency + 1e-10)
            print(f"  Transfer efficiency: {efficiency_db:.1f} dB")

            # Store results
            result = {
                'cycles_per_chip': cycles,
                'chip_duration_us': chip_duration_us,
                'total_duration_ms': total_duration_ms,
                # Theoretical (from transmitted waveform autocorrelation)
                'theo_fwhm_us': theo_fwhm_us,
                'theo_sidelobe_db': theo_sidelobe_db,
                # Measured (from received signal autocorrelation)
                # measured_fwhm_us = lag-0 FWHM = bandwidth-limited, ~constant for all waveforms
                # tof_fwhm_us = TOF peak FWHM = actual pulse compression metric, scales with chip
                'measured_fwhm_us': measured_fwhm_us,
                'tof_fwhm_us': tof_fwhm_us,
                'measured_sidelobe_db': measured_sidelobe_db,
                # Legacy keys for compatibility
                'autocorr_fwhm_us': theo_fwhm_us,
                'sidelobe_db': theo_sidelobe_db,
                # TOF (from received autocorr - incident/reflected peak spacing)
                'tof_us': tof_us,
                # Efficiency
                'transfer_efficiency': efficiency,
                'transfer_efficiency_db': efficiency_db,
                # Waveforms
                'waveform': wf,
                'received': received,
                'received_filt': received_filt,
                # Theoretical autocorrelation
                'theo_autocorr': theo_autocorr,
                'theo_autocorr_envelope': theo_env,
                'theo_autocorr_lags_us': theo_lags_us,
                # Received autocorrelation (V4 - for measured FWHM and TOF)
                'rx_autocorr': rx_autocorr,
                'rx_autocorr_envelope': rx_env,
                'rx_autocorr_lags_us': rx_lags_us,
            }
            results.append(result)

            # Plot individual result
            plot_path = OUTPUT_DIR + f'barker7_{cycles}cyc_v6.png'
            plot_barker_result(result, plot_path)
            print(f"  Saved: {plot_path}")

            # V6: Archive data
            wav_rel = archiver.save_wav(f"{cycles}cyc_received", received, DODOMIC_SAMPLE_RATE)
            wav_rel_filt = archiver.save_wav(f"{cycles}cyc_received_bp35-45kHz", received_filt, DODOMIC_SAMPLE_RATE)
            plot_rel = archiver.save_plot(f"{cycles}cyc", plot_path)
            archiver.add_result(f"{cycles}cyc", {
                'theo_fwhm_us': theo_fwhm_us,
                'theo_sidelobe_db': theo_sidelobe_db,
                'measured_fwhm_us': measured_fwhm_us,
                'tof_fwhm_us': tof_fwhm_us,
                'measured_sidelobe_db': measured_sidelobe_db,
                'tof_us': tof_us,
                'transfer_efficiency_db': efficiency_db,
            }, wav_rel_path=wav_rel, plot_rel_path=plot_rel)

            time.sleep(0.5)  # Brief pause between tests

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if sdg:
            sdg.output_off(channel=1)
            sdg.disconnect()
        if pa:
            pa.terminate()

    # Summary plot
    print(f"\n{'='*50}")
    print("Generating summary...")
    summary_path = OUTPUT_DIR + 'barker7_cycles_sweep_summary_v6.png'
    plot_cycles_sweep_summary(results, summary_path)
    print(f"Saved: {summary_path}")

    # V6: Archive summary and finalize
    archiver.save_plot("summary", summary_path)
    archiver.save_results_npz(results)
    archiver.save_manifest()

    # Save all results
    data_path = OUTPUT_DIR + 'barker7_cycles_sweep_v6.npz'
    results_compact = [{
        'cycles_per_chip': r['cycles_per_chip'],
        'chip_duration_us': r['chip_duration_us'],
        'total_duration_ms': r['total_duration_ms'],
        'theo_fwhm_us': r.get('theo_fwhm_us', 0),
        'theo_sidelobe_db': r.get('theo_sidelobe_db', 0),
        'measured_fwhm_us': r.get('measured_fwhm_us', 0),
        'tof_fwhm_us': r.get('tof_fwhm_us', 0),
        'measured_sidelobe_db': r.get('measured_sidelobe_db', 0),
        'tof_us': r['tof_us'],
        'transfer_efficiency': r['transfer_efficiency'],
        'transfer_efficiency_db': r['transfer_efficiency_db'],
    } for r in results]

    np.savez(data_path,
             results=results_compact,
             cycles_list=[r['cycles_per_chip'] for r in results],
             theo_fwhm_list=[r.get('theo_fwhm_us', 0) for r in results],
             measured_fwhm_list=[r.get('measured_fwhm_us', 0) for r in results],
             tof_fwhm_list=[r.get('tof_fwhm_us', 0) for r in results],
             theo_sidelobe_list=[r.get('theo_sidelobe_db', 0) for r in results],
             measured_sidelobe_list=[r.get('measured_sidelobe_db', 0) for r in results],
             tof_list=[r['tof_us'] for r in results],
             efficiency_db_list=[r['transfer_efficiency_db'] for r in results])
    print(f"Saved: {data_path}")

    # Print summary
    # Note: Meas lag-0 FWHM is bandwidth-limited (~constant for all waveforms, not a pulse
    # compression metric). TOF FWHM is the meaningful pulse compression metric.
    print(f"\n{'='*120}")
    print("RESULTS SUMMARY")
    print(f"{'='*120}")
    print(f"{'Cycles':<8} {'Chip(μs)':<10} {'Theo FWHM':<12} {'TOF FWHM':<12} {'Lag0 FWHM':<12} {'Theo SL(dB)':<13} {'Meas SL(dB)':<13} {'Eff(dB)':<10} {'TOF(μs)':<10}")
    print("-" * 120)
    for r in results:
        theo_fwhm = r.get('theo_fwhm_us', 0)
        tof_fwhm = r.get('tof_fwhm_us', 0)
        meas_fwhm = r.get('measured_fwhm_us', 0)
        theo_sl = r.get('theo_sidelobe_db', 0)
        meas_sl = r.get('measured_sidelobe_db', 0)
        print(f"{r['cycles_per_chip']:<8} {r['chip_duration_us']:<10.0f} {theo_fwhm:<12.0f} {tof_fwhm:<12.0f} {meas_fwhm:<12.0f} {theo_sl:<13.1f} {meas_sl:<13.1f} {r['transfer_efficiency_db']:<10.1f} {r['tof_us']:<10.1f}")

    # Best chip count: smallest TOF FWHM that has measurable sidelobe (>-30 dB = detectable)
    valid = [(i, r) for i, r in enumerate(results) if r.get('measured_sidelobe_db', -99) > -30]
    if valid:
        best_idx, best = min(valid, key=lambda x: x[1].get('tof_fwhm_us', float('inf')))
    else:
        best_idx = np.argmin([r.get('tof_fwhm_us', float('inf')) for r in results])
        best = results[best_idx]
    print(f"\nSharpest TOF peak: {best['cycles_per_chip']} cyc/chip (TOF FWHM = {best.get('tof_fwhm_us', 0):.0f} μs, SL = {best.get('measured_sidelobe_db', 0):.1f} dB, Eff = {best['transfer_efficiency_db']:.1f} dB)")

    print("\nDone!")
    return 0


if __name__ == '__main__':
    exit(main())
