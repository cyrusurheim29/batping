#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distance Characterization + PRBS Survey

Phase 1: Distance characterization (30-230 cm) with 3 waveforms:
  - Standard Gaussian (σ=5 cycles)
  - Optimized V3 (Bayesian-optimized)
  - Barker-7 @ 24 cyc/chip

Phase 2: PRBS survey at ~1m with Barker-7/11/13

Adaptive rep rate to avoid overlap at short distances.
User prompted between distance changes.

@author: cyrusurheimog
Created: June 19, 2026
"""

import numpy as np
import pyaudio
import socket
import time
import os
import json
import shutil
from datetime import datetime
from scipy import signal
from scipy.signal import butter, filtfilt, hilbert
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Hardware configuration
SDG_IP = "10.11.13.230"
SDG_PORT = 5025
SDG_SAMPLE_RATE = 400000
DODOMIC_SAMPLE_RATE = 384000
CHUNK_SIZE = 4096

# Waveform configuration
CARRIER_FREQ = 41350
AMPLITUDE = 0.9
RECORD_DURATION = 0.5
SDG_MAX_SAMPLES = 16384

SPEED_OF_SOUND = 343.0  # m/s at ~20°C

# All Barker codes
BARKER_CODES = {
    2:  [+1, -1],
    3:  [+1, +1, -1],
    4:  [+1, +1, -1, +1],
    5:  [+1, +1, +1, -1, +1],
    7:  [+1, +1, +1, -1, -1, +1, -1],
    11: [+1, +1, +1, -1, -1, -1, +1, -1, -1, +1, -1],
    13: [+1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, -1, +1],
}

BARKER_SIDELOBE_DB = {
    2: -6.0, 3: -9.5, 4: -12.0, 5: -14.0,
    7: -16.9, 11: -20.8, 13: -22.3,
}

# Golay complementary pair (length 8)
GOLAY_A = [+1, +1, +1, -1, +1, +1, -1, +1]
GOLAY_B = [+1, +1, +1, -1, -1, -1, +1, -1]

# MLS primitive polynomials (taps for LFSR - XOR positions, 0-indexed from LSB)
MLS_TAPS = {
    3: [2, 0],       # x^3 + x + 1, period 7
    4: [3, 0],       # x^4 + x + 1, period 15
    5: [4, 1],       # x^5 + x^2 + 1, period 31 (taps at positions 4 and 1)
    6: [5, 0],       # x^6 + x + 1, period 63
}

OUTPUT_DIR = '/Users/cyrusurheim_1/Downloads/Stubbs_Stuff/Claude_Dir/siglent_project/'
OPTIMIZED_V3_PATH = OUTPUT_DIR + 'optimized_waveform_v3_autocorr_20260611.npz'


class SDG1062X_LAN:
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


class DataArchiver:
    def __init__(self, base_dir, description):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(base_dir, "recordings", f"{timestamp}_{description}")
        self.raw_dir = os.path.join(self.run_dir, "raw")
        self.plots_dir = os.path.join(self.run_dir, "plots")

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        self.manifest = {
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "hardware": {
                "sdg": f"SDG1062X @ {SDG_IP}",
                "sdg_sample_rate_hz": SDG_SAMPLE_RATE,
                "mic": "Dodotronic UltraMic",
                "mic_sample_rate_hz": DODOMIC_SAMPLE_RATE,
                "carrier_freq_hz": CARRIER_FREQ,
            },
            "measurements": {},
        }
        print(f"Data archiver: {self.run_dir}")

    def save_wav(self, name, audio_data, sample_rate):
        wav_path = os.path.join(self.raw_dir, f"{name}.wav")
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wavfile.write(wav_path, sample_rate, audio_int16)
        return os.path.relpath(wav_path, self.run_dir)

    def save_plot(self, name, fig):
        plot_path = os.path.join(self.plots_dir, f"{name}.png")
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return os.path.relpath(plot_path, self.run_dir)

    def add_measurement(self, key, data):
        self.manifest["measurements"][key] = data

    def save_manifest(self):
        manifest_path = os.path.join(self.run_dir, "manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2, default=str)
        print(f"Manifest saved: {manifest_path}")


def get_rep_rate(distance_cm):
    """Calculate adaptive rep rate based on distance."""
    round_trip_m = 2 * distance_cm / 100
    tof_s = round_trip_m / SPEED_OF_SOUND
    rep_period = 2 * tof_s + 0.005  # 2× TOF + 5ms settling
    rep_rate = 1 / rep_period
    return max(20, min(100, rep_rate))


def generate_gaussian_pulse(sigma_cycles, carrier_freq, sample_rate, amplitude=0.9):
    """Generate Gaussian-enveloped pulse."""
    sigma_s = sigma_cycles / carrier_freq
    duration = 6 * sigma_s
    n_samples = int(duration * sample_rate)
    t = np.linspace(-3*sigma_s, 3*sigma_s, n_samples)
    envelope = np.exp(-t**2 / (2 * sigma_s**2))
    waveform = amplitude * envelope * np.sin(2 * np.pi * carrier_freq * t)
    return t + 3*sigma_s, waveform


def generate_linear_chirp(f_start, f_end, duration_s, sample_rate, amplitude=0.9):
    """Generate linear frequency modulation (LFM) chirp."""
    n_samples = int(duration_s * sample_rate)
    t = np.arange(n_samples) / sample_rate
    k = (f_end - f_start) / duration_s  # chirp rate
    phase = 2 * np.pi * (f_start * t + 0.5 * k * t**2)
    waveform = amplitude * np.sin(phase)
    window = signal.windows.tukey(n_samples, alpha=0.1)
    return t, waveform * window


def generate_mls(n_bits, sample_rate, cycles_per_chip, carrier_freq, amplitude=0.9):
    """Generate MLS (Maximum Length Sequence) via LFSR."""
    taps = MLS_TAPS.get(n_bits)
    if taps is None:
        raise ValueError(f"No taps defined for {n_bits}-bit MLS")

    period = 2**n_bits - 1
    state = [1] * n_bits  # Non-zero initial state

    sequence = []
    for _ in range(period):
        output = state[-1]
        sequence.append(1 if output else -1)  # Convert to +1/-1
        feedback = 0
        for tap in taps:
            feedback ^= state[tap]
        state = [feedback] + state[:-1]

    # Now BPSK modulate
    chip_duration = cycles_per_chip / carrier_freq
    samples_per_chip = int(chip_duration * sample_rate)
    total_samples = samples_per_chip * len(sequence)

    t = np.arange(total_samples) / sample_rate
    waveform = np.zeros(total_samples)

    for i, chip in enumerate(sequence):
        start = i * samples_per_chip
        end = start + samples_per_chip
        t_chip = np.arange(samples_per_chip) / sample_rate
        waveform[start:end] = chip * amplitude * np.sin(2 * np.pi * carrier_freq * t_chip)

    window = signal.windows.tukey(total_samples, alpha=0.1)
    return t, waveform * window, sequence


def generate_golay_bpsk(golay_seq, carrier_freq, cycles_per_chip, sample_rate, amplitude=0.9):
    """Generate BPSK-modulated Golay sequence."""
    chip_duration = cycles_per_chip / carrier_freq
    samples_per_chip = int(chip_duration * sample_rate)
    total_samples = samples_per_chip * len(golay_seq)

    t = np.arange(total_samples) / sample_rate
    waveform = np.zeros(total_samples)

    for i, chip in enumerate(golay_seq):
        start = i * samples_per_chip
        end = start + samples_per_chip
        t_chip = np.arange(samples_per_chip) / sample_rate
        waveform[start:end] = chip * amplitude * np.sin(2 * np.pi * carrier_freq * t_chip)

    window = signal.windows.tukey(total_samples, alpha=0.1)
    return t, waveform * window


def generate_barker_bpsk(code, carrier_freq, cycles_per_chip, sample_rate, amplitude=0.9):
    """Generate BPSK-modulated Barker code."""
    chip_duration = cycles_per_chip / carrier_freq
    samples_per_chip = int(chip_duration * sample_rate)
    total_samples = samples_per_chip * len(code)

    t = np.arange(total_samples) / sample_rate
    waveform = np.zeros(total_samples)

    for i, chip in enumerate(code):
        start = i * samples_per_chip
        end = start + samples_per_chip
        t_chip = np.arange(samples_per_chip) / sample_rate
        waveform[start:end] = chip * amplitude * np.sin(2 * np.pi * carrier_freq * t_chip)

    window = signal.windows.tukey(total_samples, alpha=0.1)
    waveform = waveform * window
    return t, waveform


def load_optimized_v3():
    """Load optimized waveform V3."""
    data = np.load(OPTIMIZED_V3_PATH)
    waveform = data['waveform']
    sample_rate = int(data['sample_rate'])
    return waveform, sample_rate


def create_sdg_buffer(waveform, rep_rate, source_sample_rate):
    """Create SDG waveform buffer with trailing zeros for rep rate."""
    n_samples_sdg = int(len(waveform) * SDG_SAMPLE_RATE / source_sample_rate)
    waveform_resampled = signal.resample(waveform, n_samples_sdg)

    max_val = np.max(np.abs(waveform_resampled))
    if max_val > 0:
        waveform_resampled = waveform_resampled / max_val * 0.9

    buffer_samples = int(SDG_SAMPLE_RATE / rep_rate)

    # Ensure buffer is at least as long as waveform (+ 10% margin for settling)
    min_buffer = int(len(waveform_resampled) * 1.1)
    if buffer_samples < min_buffer:
        buffer_samples = min_buffer

    if buffer_samples > SDG_MAX_SAMPLES:
        buffer_samples = SDG_MAX_SAMPLES
        # Truncate waveform if it still doesn't fit
        if len(waveform_resampled) > buffer_samples:
            print(f"    WARNING: Truncating waveform from {len(waveform_resampled)} to {buffer_samples} samples")
            waveform_resampled = waveform_resampled[:buffer_samples]

    waveform_buffer = np.zeros(buffer_samples)
    waveform_buffer[:len(waveform_resampled)] = waveform_resampled

    actual_rep_rate = SDG_SAMPLE_RATE / buffer_samples
    return waveform_buffer, actual_rep_rate


def find_ultramic():
    pa = pyaudio.PyAudio()
    mic_index = None

    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0 and "UltraMic" in info['name']:
            mic_index = i
            break

    if mic_index is None:
        print("UltraMic not found!")
        pa.terminate()
        return None, None

    print(f"Using UltraMic at index {mic_index}")
    return pa, mic_index


def record_audio(pa, mic_index, duration=RECORD_DURATION):
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
    return audio.astype(np.float64) / 32768.0


def bandpass_filter(sig, fs, low=35000, high=45000, order=4):
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, sig)


def measure_tof_autocorr(received_filt, fs, min_tof_us=500, max_tof_us=20000):
    """Measure TOF from autocorrelation peak spacing."""
    autocorr = np.correlate(received_filt, received_filt, mode='full')
    center = len(autocorr) // 2
    lags = np.arange(len(autocorr)) - center
    lags_us = lags / fs * 1e6

    envelope = np.abs(hilbert(autocorr))

    min_lag = int(min_tof_us * fs / 1e6)
    max_lag = int(max_tof_us * fs / 1e6)

    search_region = envelope[center + min_lag:center + max_lag]
    if len(search_region) == 0:
        return 0.0

    peak_idx = np.argmax(search_region)
    tof_us = (min_lag + peak_idx) / fs * 1e6

    return tof_us


def upload_and_record(sdg, pa, mic_index, wf_buffer, rep_rate, name):
    """Upload waveform, enable output, record, disable output."""
    sdg.upload_waveform(name, wf_buffer, rep_rate, ampl_vpp=2.0)
    time.sleep(0.3)
    sdg.output_on()
    time.sleep(0.1)

    received = record_audio(pa, mic_index)

    sdg.output_off()
    time.sleep(0.1)

    received_filt = bandpass_filter(received, DODOMIC_SAMPLE_RATE)
    return received, received_filt


def main():
    print("=" * 70)
    print("Distance Characterization + PRBS Survey")
    print("=" * 70)

    # Distance characterization settings (~10 distances from 30 to 230 cm)
    DISTANCES_CM = [30, 50, 75, 100, 125, 150, 175, 200, 215, 230]

    # Initialize archiver
    archiver = DataArchiver(OUTPUT_DIR, "distance_prbs")

    # Pre-generate waveforms
    print("\nPre-generating waveforms...")

    # 1. Standard Gaussian (σ=5 cycles)
    _, gaussian_wf = generate_gaussian_pulse(5, CARRIER_FREQ, DODOMIC_SAMPLE_RATE)
    print(f"  Gaussian: {len(gaussian_wf)} samples, {len(gaussian_wf)/DODOMIC_SAMPLE_RATE*1000:.1f} ms")

    # 2. Optimized V3
    optimized_wf, opt_sr = load_optimized_v3()
    if opt_sr != DODOMIC_SAMPLE_RATE:
        optimized_wf = signal.resample(optimized_wf, int(len(optimized_wf) * DODOMIC_SAMPLE_RATE / opt_sr))
    print(f"  Optimized V3: {len(optimized_wf)} samples, {len(optimized_wf)/DODOMIC_SAMPLE_RATE*1000:.1f} ms")

    # 3. Barker-7 @ 24 cyc/chip
    _, barker7_wf = generate_barker_bpsk(BARKER_CODES[7], CARRIER_FREQ, 24, DODOMIC_SAMPLE_RATE)
    print(f"  Barker-7 (24cyc): {len(barker7_wf)} samples, {len(barker7_wf)/DODOMIC_SAMPLE_RATE*1000:.1f} ms")

    waveforms = [
        ("gaussian", gaussian_wf),
        ("optimized_v3", optimized_wf),
        ("barker7_24cyc", barker7_wf),
    ]

    # Connect hardware
    pa, mic_index = find_ultramic()
    if pa is None:
        return 1

    sdg = None
    distance_results = {}

    try:
        sdg = SDG1062X_LAN()
        sdg.connect()
        print(f"Device: {sdg.query('*IDN?')}")
        sdg.set_arb_mode()

        # =====================
        # PHASE 1: Distance Characterization
        # =====================
        print("\n" + "=" * 70)
        print("PHASE 1: Distance Characterization")
        print("=" * 70)

        for target_dist in DISTANCES_CM:
            rep_rate = get_rep_rate(target_dist)

            print(f"\n>>> Target distance: ~{target_dist} cm")
            print(f"    Rep rate: {rep_rate:.1f} Hz")
            actual_input = input(f"    Move reflector to ~{target_dist} cm, then enter ACTUAL distance (cm): ")

            try:
                actual_dist = float(actual_input.strip())
            except ValueError:
                actual_dist = target_dist
                print(f"    (Using target {target_dist} cm)")

            expected_tof = 2 * actual_dist / 100 / SPEED_OF_SOUND * 1e6
            print(f"    Actual: {actual_dist} cm → expected TOF: {expected_tof:.0f} μs")

            distance_results[actual_dist] = {}

            for wf_name, wf in waveforms:
                wf_buffer, actual_rep = create_sdg_buffer(wf, rep_rate, DODOMIC_SAMPLE_RATE)

                received, received_filt = upload_and_record(
                    sdg, pa, mic_index, wf_buffer, actual_rep, f"d{int(actual_dist)}_{wf_name}"
                )

                tof_us = measure_tof_autocorr(received_filt, DODOMIC_SAMPLE_RATE)

                wav_path = archiver.save_wav(f"{actual_dist}cm_{wf_name}", received, DODOMIC_SAMPLE_RATE)
                wav_path_filt = archiver.save_wav(f"{actual_dist}cm_{wf_name}_bp35-45kHz", received_filt, DODOMIC_SAMPLE_RATE)

                distance_results[actual_dist][wf_name] = {
                    "tof_us": tof_us,
                    "expected_tof_us": expected_tof,
                    "rep_rate_hz": actual_rep,
                    "wav": wav_path,
                    "wav_filt": wav_path_filt,
                }

                print(f"    {wf_name}: TOF = {tof_us:.0f} μs (Δ = {tof_us - expected_tof:+.0f} μs)")

            archiver.add_measurement(f"distance_{actual_dist}cm", distance_results[actual_dist])

        # =====================
        # PHASE 2: Comprehensive Pulse Compression Survey
        # =====================
        print("\n" + "=" * 70)
        print("PHASE 2: Pulse Compression Survey (at ~1m)")
        print("=" * 70)
        input(">>> Return reflector to ~100 cm and press Enter...")

        rep_rate = 40  # Standard for 1m
        survey_results = {}

        # --- Baseline waveforms (Gaussian, Optimized V3) ---
        print("\n  [Baseline waveforms]")
        for wf_name, wf in [("gaussian", gaussian_wf), ("optimized_v3", optimized_wf)]:
            wf_buffer, actual_rep = create_sdg_buffer(wf, rep_rate, DODOMIC_SAMPLE_RATE)
            received, received_filt = upload_and_record(
                sdg, pa, mic_index, wf_buffer, actual_rep, f"survey_{wf_name}"
            )
            tof_us = measure_tof_autocorr(received_filt, DODOMIC_SAMPLE_RATE)
            wav_path = archiver.save_wav(f"survey_{wf_name}", received, DODOMIC_SAMPLE_RATE)
            wav_path_filt = archiver.save_wav(f"survey_{wf_name}_bp35-45kHz", received_filt, DODOMIC_SAMPLE_RATE)
            survey_results[wf_name] = {"type": "baseline", "tof_us": tof_us, "wav": wav_path, "wav_filt": wav_path_filt}
            print(f"    {wf_name}: TOF = {tof_us:.0f} μs")

        # --- All Barker codes @ 24 cyc/chip ---
        print("\n  [Barker codes @ 24 cyc/chip]")
        for code_len in [2, 3, 4, 5, 7, 11, 13]:
            code = BARKER_CODES[code_len]
            _, barker_wf = generate_barker_bpsk(code, CARRIER_FREQ, 24, DODOMIC_SAMPLE_RATE)
            wf_buffer, actual_rep = create_sdg_buffer(barker_wf, rep_rate, DODOMIC_SAMPLE_RATE)

            name = f"barker{code_len}_24cyc"
            received, received_filt = upload_and_record(
                sdg, pa, mic_index, wf_buffer, actual_rep, name
            )
            tof_us = measure_tof_autocorr(received_filt, DODOMIC_SAMPLE_RATE)
            wav_path = archiver.save_wav(name, received, DODOMIC_SAMPLE_RATE)
            wav_path_filt = archiver.save_wav(f"{name}_bp35-45kHz", received_filt, DODOMIC_SAMPLE_RATE)

            survey_results[name] = {
                "type": "barker", "length": code_len, "cycles_per_chip": 24,
                "tof_us": tof_us, "theo_sidelobe_db": BARKER_SIDELOBE_DB.get(code_len),
                "wav": wav_path, "wav_filt": wav_path_filt,
            }
            print(f"    {name}: TOF = {tof_us:.0f} μs")

        # --- Barker-7 at different cycles/chip ---
        print("\n  [Barker-7 cycles/chip sweep]")
        for cyc in [16, 48]:
            _, barker_wf = generate_barker_bpsk(BARKER_CODES[7], CARRIER_FREQ, cyc, DODOMIC_SAMPLE_RATE)
            wf_buffer, actual_rep = create_sdg_buffer(barker_wf, rep_rate, DODOMIC_SAMPLE_RATE)

            name = f"barker7_{cyc}cyc"
            received, received_filt = upload_and_record(
                sdg, pa, mic_index, wf_buffer, actual_rep, name
            )
            tof_us = measure_tof_autocorr(received_filt, DODOMIC_SAMPLE_RATE)
            wav_path = archiver.save_wav(name, received, DODOMIC_SAMPLE_RATE)
            wav_path_filt = archiver.save_wav(f"{name}_bp35-45kHz", received_filt, DODOMIC_SAMPLE_RATE)

            survey_results[name] = {
                "type": "barker", "length": 7, "cycles_per_chip": cyc,
                "tof_us": tof_us, "wav": wav_path, "wav_filt": wav_path_filt,
            }
            print(f"    {name}: TOF = {tof_us:.0f} μs")

        # --- Linear chirp ---
        print("\n  [Linear chirp]")
        _, chirp_wf = generate_linear_chirp(38000, 44000, 0.002, DODOMIC_SAMPLE_RATE)
        wf_buffer, actual_rep = create_sdg_buffer(chirp_wf, rep_rate, DODOMIC_SAMPLE_RATE)
        received, received_filt = upload_and_record(
            sdg, pa, mic_index, wf_buffer, actual_rep, "chirp_lfm"
        )
        tof_us = measure_tof_autocorr(received_filt, DODOMIC_SAMPLE_RATE)
        wav_path = archiver.save_wav("chirp_lfm", received, DODOMIC_SAMPLE_RATE)
        wav_path_filt = archiver.save_wav("chirp_lfm_bp35-45kHz", received_filt, DODOMIC_SAMPLE_RATE)
        survey_results["chirp_lfm"] = {
            "type": "chirp", "f_start": 38000, "f_end": 44000, "duration_ms": 2,
            "tof_us": tof_us, "wav": wav_path, "wav_filt": wav_path_filt,
        }
        print(f"    chirp_lfm: TOF = {tof_us:.0f} μs")

        # --- MLS sequences ---
        print("\n  [MLS sequences @ 24 cyc/chip]")
        for n_bits in [3, 4, 5]:
            period = 2**n_bits - 1
            _, mls_wf, _ = generate_mls(n_bits, DODOMIC_SAMPLE_RATE, 24, CARRIER_FREQ)
            wf_buffer, actual_rep = create_sdg_buffer(mls_wf, rep_rate, DODOMIC_SAMPLE_RATE)

            name = f"mls{period}"
            received, received_filt = upload_and_record(
                sdg, pa, mic_index, wf_buffer, actual_rep, name
            )
            tof_us = measure_tof_autocorr(received_filt, DODOMIC_SAMPLE_RATE)
            wav_path = archiver.save_wav(name, received, DODOMIC_SAMPLE_RATE)
            wav_path_filt = archiver.save_wav(f"{name}_bp35-45kHz", received_filt, DODOMIC_SAMPLE_RATE)

            survey_results[name] = {
                "type": "mls", "n_bits": n_bits, "period": period,
                "tof_us": tof_us, "wav": wav_path, "wav_filt": wav_path_filt,
            }
            print(f"    {name} (period {period}): TOF = {tof_us:.0f} μs")

        # --- Golay pair ---
        print("\n  [Golay complementary pair @ 24 cyc/chip]")
        for golay_name, golay_seq in [("golay_A", GOLAY_A), ("golay_B", GOLAY_B)]:
            _, golay_wf = generate_golay_bpsk(golay_seq, CARRIER_FREQ, 24, DODOMIC_SAMPLE_RATE)
            wf_buffer, actual_rep = create_sdg_buffer(golay_wf, rep_rate, DODOMIC_SAMPLE_RATE)

            received, received_filt = upload_and_record(
                sdg, pa, mic_index, wf_buffer, actual_rep, golay_name
            )
            tof_us = measure_tof_autocorr(received_filt, DODOMIC_SAMPLE_RATE)
            wav_path = archiver.save_wav(golay_name, received, DODOMIC_SAMPLE_RATE)
            wav_path_filt = archiver.save_wav(f"{golay_name}_bp35-45kHz", received_filt, DODOMIC_SAMPLE_RATE)

            survey_results[golay_name] = {
                "type": "golay", "length": 8,
                "tof_us": tof_us, "wav": wav_path, "wav_filt": wav_path_filt,
            }
            print(f"    {golay_name}: TOF = {tof_us:.0f} μs")

        archiver.add_measurement("pulse_compression_survey", survey_results)

        # =====================
        # Summary Plot
        # =====================
        print("\nGenerating summary plot...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Distance vs TOF
        distances = sorted(distance_results.keys())
        for wf_name in ["gaussian", "optimized_v3", "barker7_24cyc"]:
            tofs = [distance_results[d][wf_name]["tof_us"] for d in distances]
            axes[0].plot(distances, tofs, 'o-', label=wf_name, markersize=8)

        expected = [2 * d / 100 / SPEED_OF_SOUND * 1e6 for d in distances]
        axes[0].plot(distances, expected, 'k--', label='Expected (343 m/s)', linewidth=2)

        axes[0].set_xlabel('Distance (cm)')
        axes[0].set_ylabel('TOF (μs)')
        axes[0].set_title('Distance Characterization')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Pulse compression survey - all sequences
        survey_names = list(survey_results.keys())
        survey_tofs = [survey_results[n]["tof_us"] for n in survey_names]
        colors = []
        for n in survey_names:
            t = survey_results[n]["type"]
            if t == "baseline":
                colors.append("green")
            elif t == "barker":
                colors.append("steelblue")
            elif t == "chirp":
                colors.append("orange")
            elif t == "mls":
                colors.append("purple")
            elif t == "golay":
                colors.append("red")
            else:
                colors.append("gray")

        axes[1].bar(range(len(survey_names)), survey_tofs, color=colors)
        axes[1].set_xticks(range(len(survey_names)))
        axes[1].set_xticklabels(survey_names, rotation=45, ha='right', fontsize=8)
        axes[1].set_ylabel('TOF (μs)')
        axes[1].set_title('Pulse Compression Survey @ ~1m')
        axes[1].grid(True, alpha=0.3, axis='y')

        # Legend for colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Baseline'),
            Patch(facecolor='steelblue', label='Barker'),
            Patch(facecolor='orange', label='Chirp'),
            Patch(facecolor='purple', label='MLS'),
            Patch(facecolor='red', label='Golay'),
        ]
        axes[1].legend(handles=legend_elements, loc='upper right', fontsize=8)

        fig.tight_layout()
        archiver.save_plot("summary", fig)

        # Save manifest
        archiver.save_manifest()

        # Print summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print("\nDistance characterization:")
        print(f"{'Dist (cm)':<12} {'Gaussian':<12} {'Opt V3':<12} {'Barker-7':<12} {'Expected':<12}")
        print("-" * 60)
        for d in distances:
            g = distance_results[d]["gaussian"]["tof_us"]
            o = distance_results[d]["optimized_v3"]["tof_us"]
            b = distance_results[d]["barker7_24cyc"]["tof_us"]
            e = 2 * d / 100 / SPEED_OF_SOUND * 1e6
            print(f"{d:<12} {g:<12.0f} {o:<12.0f} {b:<12.0f} {e:<12.0f}")

        print("\nPulse compression survey (@ ~1m):")
        print(f"{'Sequence':<20} {'Type':<10} {'TOF (μs)':<12}")
        print("-" * 42)
        for name in survey_names:
            r = survey_results[name]
            print(f"{name:<20} {r['type']:<10} {r['tof_us']:<12.0f}")

        print(f"\nTotal WAV files: {len(survey_names) + len(distances) * 3}")
        print(f"Data saved to: {archiver.run_dir}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if sdg:
            sdg.output_off()
            sdg.disconnect()
        if pa:
            pa.terminate()

    print("\nDone!")
    return 0


if __name__ == '__main__':
    main()
