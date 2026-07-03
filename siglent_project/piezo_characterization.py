#!/usr/bin/env python3
"""
Piezo Transducer Bandwidth Characterization

Measures the frequency response of the piezoelectric transducer system
using a chirp (frequency sweep) method. This determines the usable bandwidth
for PRBS waveform design.

Usage:
    python piezo_characterization.py
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

# === Configuration ===
SDG_IP = "10.11.13.230"
SDG_PORT = 5025

# Audio settings
SAMPLE_RATE = 384000  # UltraMic sample rate
RECORD_DURATION = 0.3  # seconds (enough for chirp + reflections)
CHUNK_SIZE = 4096

# Chirp parameters
CHIRP_F1 = 20000      # Start frequency (Hz)
CHIRP_F2 = 60000      # End frequency (Hz)
CHIRP_DURATION = 0.010  # 10 ms chirp duration
CHIRP_AMPLITUDE = 0.9   # Normalized amplitude

# AWG parameters
AWG_SAMPLE_RATE = 400000  # 400 kSa/s for SDG
REP_RATE_HZ = 50  # Repetition rate for the chirp

OUTPUT_DIR = '/Users/cyrusurheim_1/Downloads/Stubbs_Stuff/Claude_Dir/siglent_project/'


class SDG1062X_LAN:
    """Siglent SDG1062X control via LAN."""

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


def generate_chirp_waveform(f1, f2, duration, sample_rate, amplitude=0.9):
    """
    Generate a linear chirp waveform.

    Parameters:
        f1: Start frequency (Hz)
        f2: End frequency (Hz)
        duration: Chirp duration (seconds)
        sample_rate: Sample rate (Hz)
        amplitude: Peak amplitude (0-1)

    Returns:
        t: Time array
        chirp: Chirp waveform
    """
    n_samples = int(duration * sample_rate)
    t = np.arange(n_samples) / sample_rate

    # Linear chirp: f(t) = f1 + (f2-f1) * t / duration
    # Phase: phi(t) = 2*pi * integral(f(t)) = 2*pi * (f1*t + (f2-f1)*t^2 / (2*duration))
    chirp_rate = (f2 - f1) / duration
    phase = 2 * np.pi * (f1 * t + 0.5 * chirp_rate * t**2)
    chirp = amplitude * np.cos(phase)

    # Apply Tukey window to avoid clicks at edges
    window = signal.windows.tukey(n_samples, alpha=0.1)
    chirp = chirp * window

    return t, chirp


def create_awg_waveform(chirp, awg_sample_rate, rep_rate):
    """
    Create AWG waveform buffer with chirp embedded.

    The AWG plays this buffer repeatedly at rep_rate Hz.
    """
    # Total samples for one period at rep_rate
    period_samples = int(awg_sample_rate / rep_rate)

    # Resample chirp if necessary
    chirp_samples_needed = int(len(chirp) * awg_sample_rate / SAMPLE_RATE)
    if len(chirp) != chirp_samples_needed:
        chirp_resampled = signal.resample(chirp, chirp_samples_needed)
    else:
        chirp_resampled = chirp

    # Create buffer with chirp at the start, zeros after
    waveform = np.zeros(period_samples)

    if len(chirp_resampled) > period_samples:
        print(f"Warning: Chirp ({len(chirp_resampled)}) longer than period ({period_samples})")
        waveform = chirp_resampled[:period_samples]
    else:
        waveform[:len(chirp_resampled)] = chirp_resampled

    # Normalize
    if np.max(np.abs(waveform)) > 0:
        waveform = waveform / np.max(np.abs(waveform)) * CHIRP_AMPLITUDE

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
        transmitted: Transmitted chirp waveform
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


def find_bandwidth(freqs, H_mag_db, threshold_db=-3, search_range=(30000, 50000), smooth_window=51):
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
    print("=" * 60)
    print("Piezoelectric Transducer Bandwidth Characterization")
    print("=" * 60)

    # Generate chirp
    print(f"\n--- Generating Chirp ---")
    print(f"  Frequency sweep: {CHIRP_F1/1000:.0f} - {CHIRP_F2/1000:.0f} kHz")
    print(f"  Duration: {CHIRP_DURATION*1000:.0f} ms")

    t_chirp, chirp = generate_chirp_waveform(
        CHIRP_F1, CHIRP_F2, CHIRP_DURATION, SAMPLE_RATE, CHIRP_AMPLITUDE
    )
    print(f"  Generated {len(chirp)} samples @ {SAMPLE_RATE/1000:.0f} kHz")

    # Create AWG waveform
    awg_waveform = create_awg_waveform(chirp, AWG_SAMPLE_RATE, REP_RATE_HZ)
    print(f"  AWG waveform: {len(awg_waveform)} samples @ {AWG_SAMPLE_RATE/1000:.0f} kHz")
    print(f"  Rep rate: {REP_RATE_HZ} Hz")

    # Find microphone
    pa, mic_index = find_ultramic()
    if pa is None:
        return 1

    sdg = None
    try:
        # Connect to SDG
        print("\n--- Connecting to SDG1062X ---")
        sdg = SDG1062X_LAN()
        sdg.connect()

        idn = sdg.query("*IDN?")
        print(f"Device: {idn}")

        # Upload chirp waveform
        print("\n--- Uploading Chirp Waveform ---")
        sdg.set_arb_mode(channel=1)
        sdg.upload_waveform("chirp", awg_waveform, freq_hz=REP_RATE_HZ, ampl_vpp=2.0)
        print("  Waveform uploaded")

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

        # Use the original chirp (at recording sample rate) for comparison
        freqs, H_mag, H_phase = compute_transfer_function(
            chirp, received[:len(chirp)*3], SAMPLE_RATE  # Use ~3x chirp length of response
        )

        # Find bandwidth (with smoothing to handle noisy spectrum)
        f_center, f_low, f_high, bandwidth, peak_freq, peak_mag, H_mag_smooth = find_bandwidth(
            freqs, H_mag, threshold_db=-3, search_range=(30000, 50000), smooth_window=101
        )

        # Also find -6 dB bandwidth
        _, f_low_6, f_high_6, bandwidth_6, _, _, _ = find_bandwidth(
            freqs, H_mag, threshold_db=-6, search_range=(30000, 50000), smooth_window=101
        )

        # Print results
        print(f"\n{'='*60}")
        print("CHARACTERIZATION RESULTS")
        print(f"{'='*60}")
        print(f"  Peak frequency:     {peak_freq/1000:.2f} kHz")
        print(f"  Center frequency:   {f_center/1000:.2f} kHz")
        print(f"  -3 dB bandwidth:    {bandwidth/1000:.2f} kHz")
        print(f"  -3 dB passband:     {f_low/1000:.1f} - {f_high/1000:.1f} kHz")
        print(f"  -6 dB bandwidth:    {bandwidth_6/1000:.2f} kHz")
        print(f"  -6 dB passband:     {f_low_6/1000:.1f} - {f_high_6/1000:.1f} kHz")

        print(f"\n--- Recommended PRBS Parameters ---")
        Tc_min = 1.0 / bandwidth  # Aggressive (full bandwidth)
        Tc_max = 2.0 / bandwidth  # Conservative (half bandwidth)
        print(f"  Chip duration T_c:  {Tc_min*1e6:.0f} - {Tc_max*1e6:.0f} μs")
        print(f"  Chip rate:          {1/Tc_max/1000:.1f} - {1/Tc_min/1000:.1f} kHz")
        print(f"  Cycles/chip (40kHz): {40000*Tc_min:.1f} - {40000*Tc_max:.1f}")

        # Suggest specific values
        # Round to nice numbers of carrier cycles
        cycles_per_chip = int(np.round(40000 * (Tc_min + Tc_max) / 2))
        Tc_recommended = cycles_per_chip / 40000
        print(f"\n  >>> Recommended: T_c = {Tc_recommended*1e6:.0f} μs ({cycles_per_chip} cycles/chip)")

        # Maximum code length before multipath
        multipath_window = 5e-3  # 5 ms assumed
        max_chips = int(multipath_window / Tc_recommended)
        print(f"  >>> Max code length: ~{max_chips} chips (assuming 5 ms multipath window)")

        # Plot results
        plot_path = OUTPUT_DIR + 'piezo_characterization.png'
        plot_characterization(freqs, H_mag, H_mag_smooth, H_phase, f_center, f_low, f_high, bandwidth,
                             peak_mag, received, SAMPLE_RATE, plot_path)

        # Save characterization data
        npz_path = OUTPUT_DIR + 'piezo_characterization.npz'
        np.savez(npz_path,
                 freqs=freqs,
                 H_mag=H_mag,
                 H_phase=H_phase,
                 f_center=f_center,
                 f_low=f_low,
                 f_high=f_high,
                 bandwidth=bandwidth,
                 bandwidth_6dB=bandwidth_6,
                 peak_freq=peak_freq,
                 Tc_recommended=Tc_recommended,
                 chirp=chirp,
                 received=received)
        print(f"Saved data: {npz_path}")

        print(f"\n{'='*60}")
        print("Characterization complete!")
        print(f"{'='*60}")

        return 0

    except socket.timeout:
        print(f"\nConnection timed out. Check IP: {SDG_IP}")
        return 1
    except ConnectionRefusedError:
        print(f"\nConnection refused. Check SDG is on and LAN enabled.")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if sdg and sdg.sock:
            try:
                sdg.output_off(channel=1)
            except:
                pass
            sdg.disconnect()
        if pa:
            pa.terminate()


if __name__ == "__main__":
    import sys
    sys.exit(main())
