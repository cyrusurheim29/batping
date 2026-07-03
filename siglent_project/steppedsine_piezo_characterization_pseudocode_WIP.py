#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 09:46:55 2026

@author: cyrusurheimog

Multi-Measurement Piezoelectric Transducer Bandwidth Characterization

Write a variable (but single) frequency sine wave, remotely control the 
SDG via LAN connection, upload sine wave with FREQ initialized to 
Siglent SDG1062x waveform generator, command (via SCPI) SDG to output
signal with DURATION initialized, record for DURATION with Ultramic on
Mac, store/read aiff/wav file, generate plot of transfer function,
identify -3dB and -6dB passbands.
"""

import numpy as np
import pyaudio
import socket
import time
from scipy import signal
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# waveform generator configuration
SDG_IP = "10.11.13.230"
SDG_PORT = 5025
SDG_SAMPLE_RATE = 400000 # 400 kSa/s for SDG

# mic settings
DODOMIC_SAMPLE_RATE = 384000
RECORD_DURATION = 0.3
CHUNK_SIZE = 4096

# piezo speaker specifications
RES_FREQ = 40000


OUTPUT_DIR = '/Users/cyrusurheim_1/Downloads/Stubbs_Stuff/Claude_Dir/siglent_project/'

class SDG1062X_LAN:
    """Siglent SDG1062 remote control via LAN (direct ethernet cable) connection."""
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
        time.sleep(1.0)

        self.write(f"C{channel}:ARWV NAME,{name}")
        time.sleep(0.3)

    def set_arb_mode(self, channel=1):
        self.write(f"C{channel}:BSWV WVTP,ARB")
        time.sleep(0.3)

    def output_on(self, channel=1):
        self.write(f"C{channel}:OUTP ON")

    def output_off(self, channel=1):
        self.write(f"C{channel}:OUTP OFF")


def generate_sine_waveform(freq, duration, dodomic_sample_rate, amplitude =):
    n_samples = int(duration * dodomic_sample_rate)
    t = np.arange(n_samples) / dodomic_sample_rate
    waveform = amplitude * np.cos(2 * np.pi * freq * t)
    
    # should we still apply turkey window to avoid clicks at edges? 
    # if so, what is a turkey window?
    #window = signal.windows.turkey(n_samples, alpha=0.1)
    #waveform = waveform * window
    
    return t, waveform

def creat_sdg_waveform(waveform, sdg_sample_rate, dodmic_sample_rate, duration, freq):
    """Create SDG waveform buffer with sine wave embedded."""
    # number of sdg samples that makeup one period
    period_samples = int(sdg_sample_rate / freq)
    
    # resample the waveform to fill the sdg sample rate req
    waveform_samples_needed = int(duration * sdg_sample_rate)
    if len(waveform) != waveform_samples_needed:
        waveform_resampled = signal.resample(waveform, waveform_samples_needed)
    else:
        waveform_resampled = waveform
    
    # add a buffer if resampling was indeed needed, to fill the dodomic recording
    waveform_final = np.zeros(period_samples)
    
    if len(waveform_resampled) > period_samples:
        print(f"Warning: Chirp ({len(chirp_resampled)}) longer than period ({period_samples})")
        waveform_final = waveform_resampled[:period_samples]
    else:
        waveform_final[:len(waveform_resampled)] = waveform_resampled

    # Is normalization necessary for simple sine rather than chrip???? If so:
    #if np.max(np.abs(waveform)) > 0:
    #    waveform = waveform / np.max(np.abs(waveform)) * some_amplitude-var

    return waveform


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
    n_samples = int(duration * SAMPLE_RATE)

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=mic_index,
        frames_per_buffer=CHUNK_SIZE
    )

    print(f"Recording {duration}s @ {SAMPLE_RATE/1000:.0f}kHz...")
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

    print(f"Recorded {len(audio_float)} samples")
    return audio_float


def compute_transfer_function(transmitted, received, fs, f_low=15000, f_high=65000):
    """
    Compute the transfer function H(f) = Received(f) / Transmitted(f).

    Parameters:
        transmitted: Transmitted sine wave
        received: Recorded response
        fs: Sample rate
        f_low, f_high: Frequency range of interest

    Returns:
        freqs: Frequency array
        H_mag: Magnitude of transfer function (dB)
        H_phase: Phase of transfer function (radians)
    """
    # Zero-pad to same length
    n = max(len(transmitted), len(received))
    n_fft = 2 ** int(np.ceil(np.log2(n)))  # Next power of 2

    # Compute FFTs
    S_tx = np.fft.fft(transmitted, n_fft)
    S_rx = np.fft.fft(received, n_fft)
    freqs = np.fft.fftfreq(n_fft, 1/fs)

    # Only look at positive frequencies in range of interest
    pos_mask = (freqs >= f_low) & (freqs <= f_high)
    freqs_pos = freqs[pos_mask]
    S_tx_pos = S_tx[pos_mask]
    S_rx_pos = S_rx[pos_mask]

    # Compute transfer function (with regularization to avoid division by zero)
    epsilon = np.max(np.abs(S_tx_pos)) * 1e-6
    H = S_rx_pos / (S_tx_pos + epsilon)

    H_mag = 20 * np.log10(np.abs(H) + 1e-10)
    H_phase = np.angle(H)

    return freqs_pos, H_mag, H_phase


def find_bandwidth(freqs, H_mag_db, threshold_db, search_range=(30000, 50000), smooth_window=51):
    """
    Find the bandwidth at a given threshold below peak.

    Parameters:
        freqs: Frequency array
        H_mag_db: Magnitude in dB
        threshold_db: Threshold below peak (negative value, e.g., -3)
        search_range: (f_min, f_max) to search for peak (Hz)
        smooth_window: Window size for Savitzky-Golay smoothing (odd number)

    Returns:
        f_center: Center frequency
        f_low: Lower -3dB frequency
        f_high: Upper -3dB frequency
        bandwidth: f_high - f_low
    """
    # Smooth the magnitude response to avoid noise-induced false crossings
    from scipy.signal import savgol_filter
    if smooth_window > 3 and len(H_mag_db) > smooth_window:
        H_mag_smooth = savgol_filter(H_mag_db, smooth_window, 3)
    else:
        H_mag_smooth = H_mag_db

    # Constrain search to expected piezo frequency range
    search_mask = (freqs >= search_range[0]) & (freqs <= search_range[1])
    search_indices = np.where(search_mask)[0]

    if len(search_indices) == 0:
        raise ValueError(f"No frequencies in search range {search_range}")

    # Find peak within search range (using smoothed data)
    H_mag_search = H_mag_smooth[search_mask]
    peak_idx_in_search = np.argmax(H_mag_search)
    peak_idx = search_indices[peak_idx_in_search]

    peak_freq = freqs[peak_idx]
    peak_mag = H_mag_smooth[peak_idx]

    threshold = peak_mag + threshold_db  # threshold_db is negative

    # Find where smoothed magnitude crosses threshold
    above_threshold = H_mag_smooth >= threshold

    # Find edges
    transitions = np.diff(above_threshold.astype(int))
    rise_idx = np.where(transitions == 1)[0]
    fall_idx = np.where(transitions == -1)[0]

    # Find the passband containing the peak
    if len(rise_idx) > 0:
        # Find the rise before the peak
        rises_before_peak = rise_idx[rise_idx < peak_idx]
        if len(rises_before_peak) > 0:
            low_idx = rises_before_peak[-1]
        else:
            low_idx = 0
    else:
        low_idx = 0

    if len(fall_idx) > 0:
        # Find the fall after the peak
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


def plot_characterization(freqs, H_mag, H_mag_smooth, H_phase, f_center, f_low, f_high, bandwidth,
                         peak_mag, received, fs, output_path):
    """Create comprehensive characterization plot."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Magnitude response (full range)
    ax1 = axes[0, 0]
    ax1.plot(freqs/1000, H_mag, 'b-', linewidth=0.5, alpha=0.5, label='Raw')
    ax1.plot(freqs/1000, H_mag_smooth, 'b-', linewidth=2, label='Smoothed')
    # Use the actual peak magnitude from the piezo resonance, not global max
    ax1.axhline(peak_mag - 3, color='r', linestyle='--', label='-3 dB')
    ax1.axhline(peak_mag - 6, color='orange', linestyle='--', label='-6 dB')
    ax1.axvline(f_low/1000, color='g', linestyle=':', alpha=0.7)
    ax1.axvline(f_high/1000, color='g', linestyle=':', alpha=0.7)
    ax1.axvline(f_center/1000, color='g', linestyle='-', alpha=0.7, label=f'Center: {f_center/1000:.1f} kHz')
    ax1.fill_betweenx([np.min(H_mag), np.max(H_mag)], f_low/1000, f_high/1000,
                      alpha=0.2, color='green', label=f'BW: {bandwidth/1000:.1f} kHz')
    ax1.set_xlabel('Frequency (kHz)')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_title('Piezo Transfer Function - Magnitude Response')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([15, 65])

    # Panel 2: Magnitude response (zoomed on passband)
    ax2 = axes[0, 1]
    zoom_mask = (freqs >= 30000) & (freqs <= 50000)
    ax2.plot(freqs[zoom_mask]/1000, H_mag[zoom_mask], 'b-', linewidth=0.5, alpha=0.5)
    ax2.plot(freqs[zoom_mask]/1000, H_mag_smooth[zoom_mask], 'b-', linewidth=2)
    ax2.axhline(peak_mag - 3, color='r', linestyle='--', linewidth=1.5, label='-3 dB')
    ax2.axvline(f_low/1000, color='g', linestyle='-', linewidth=2, label=f'{f_low/1000:.1f} kHz')
    ax2.axvline(f_high/1000, color='g', linestyle='-', linewidth=2, label=f'{f_high/1000:.1f} kHz')
    ax2.axvline(f_center/1000, color='purple', linestyle='-', linewidth=2, label=f'Center: {f_center/1000:.1f} kHz')
    ax2.set_xlabel('Frequency (kHz)')
    ax2.set_ylabel('Magnitude (dB)')
    ax2.set_title(f'Passband Detail — Bandwidth: {bandwidth/1000:.2f} kHz')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Phase response
    ax3 = axes[1, 0]
    ax3.plot(freqs/1000, np.unwrap(H_phase), 'b-', linewidth=1)
    ax3.axvline(f_low/1000, color='g', linestyle=':', alpha=0.7)
    ax3.axvline(f_high/1000, color='g', linestyle=':', alpha=0.7)
    ax3.set_xlabel('Frequency (kHz)')
    ax3.set_ylabel('Phase (radians)')
    ax3.set_title('Piezo Transfer Function - Phase Response')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([15, 65])

    # Panel 4: Time-domain received signal
    ax4 = axes[1, 1]
    t_received = np.arange(len(received)) / fs * 1000  # ms
    ax4.plot(t_received, received, 'b-', linewidth=0.5)
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Amplitude')
    ax4.set_title('Received Signal (Time Domain)')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, min(50, t_received[-1])])  # Show first 50 ms

    # Add summary text
    fig.suptitle('Piezoelectric Transducer Characterization', fontsize=14, fontweight='bold')

    summary_text = (
        f"Center Frequency: {f_center/1000:.2f} kHz\n"
        f"-3 dB Bandwidth: {bandwidth/1000:.2f} kHz\n"
        f"Passband: {f_low/1000:.1f} - {f_high/1000:.1f} kHz\n"
        f"Recommended T_c: {1000/bandwidth:.0f} - {2000/bandwidth:.0f} μs\n"
        f"Cycles/chip at 40kHz: {40000/bandwidth:.1f} - {80000/bandwidth:.1f}"
    )
    fig.text(0.02, 0.02, summary_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved plot: {output_path}")
    plt.close()
    

def main():
    freq_list = # range of freqs from LB to UB--make it expansive and granular
    for freq in freq_list
        t_wf, wf = generate_sine_waveform(freq, RECORD_DURATION, DODMIC_SAMPLE_RATE, AMPLITUDE) # amplitude?
        print("\n--- Generated Sine Wave ---")
        print(f"  Frequency: {freq} Hz")
        print(f"  Duration: {CHIRP_DURATION*1000:.0f} ms")
        
        sdg_waveform = create_sdg_waveform(wf, SDG_SAMPLE_RATE, DODOMIC_SAMPLE_RATE, RECORD_DURATION, freq)
        print("\n--- Fit Sine Wave into array for SDG, Dodotronic Mic Sample Rates ---")
        
        # Find mic
        pa, mic_index = find_ultramic()
        if pa is None:
            return 1
        
        # Connect to SDG
        sdg = None
        try:
            sdg = SDG1062X_LAN()
            sdg.connect()
            idn = sdg.query("*IDN?")
            if "Siglent Technologies" in idn:
                print"\n--- Connected to SDG1062X ---"
            else:
                return 1
            print(f"Device: {idn}")
            
            # Upload sine wave
            sdg.set_arb_mode(channel=1)
            sdg.upload_waveform("sine_bw", sdg_waveform, freq, ampl_vpp=2.0)

            # Enable output
            print("\n--- Enabling Output ---")
            sdg.output_on(channel=1)
            print("  CH1 ON")
            time.sleep(0.5)  # Let transducer settle
    
            # Record response
            print("\n--- Recording Response ---")
            received = record_audio(pa, mic_index, RECORD_DURATION)
    
            # Turn off output
            print("\n--- Disabling Output ---")
            sdg.output_off(channel=1)
            print("  CH1 OFF")
    
            # Save raw recording
            wav_path = OUTPUT_DIR + 'piezo_chirp_response.wav'
            wavfile.write(wav_path, SAMPLE_RATE, (received * 32767).astype(np.int16))
            print(f"\nSaved recording: {wav_path}")
    
            # Compute transfer function
            print("\n--- Computing Transfer Function ---")
    
            # Use the original sine wave (at recording sample rate) for comparison
            freqs, H_mag, H_phase = compute_transfer_function(
                chirp, received[:len(chirp)*3], SAMPLE_RATE  # Use ~3x chirp length of response            
            )
            
            #IF my theory is right and this is constant--extract the constant value
            c = |H_mag|
            overall_trasnfer_function[freq] = c
        
        generate_overall_transfer_function(#use all of the points )
        print and plot char results as in piezo_characterization.py
            
    
    


