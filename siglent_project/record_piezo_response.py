#!/usr/bin/env python3
"""
Record piezo response to BATPING waveform from SDG1062X.

Fully automated: uploads waveform, enables output, records, analyzes.
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

SAMPLE_RATE = 384000
RECORD_DURATION = 0.5  # seconds - captures ~31 pulses at 62.5Hz
CHUNK_SIZE = 4096

# BATPING waveform parameters
REP_RATE_HZ = 62.5
CARRIER_FREQ = 40000
SIGMA_CYCLES = 5
SAMPLES_PER_CARRIER = 10

# Bandpass filter for 40kHz
FILTER_LOW = 35000
FILTER_HIGH = 45000


# === SDG1062X LAN Interface ===
class SDG1062X_LAN:
    def __init__(self, ip=SDG_IP, port=SDG_PORT, timeout=5.0):
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


def create_batping_pulse(rep_rate_hz=REP_RATE_HZ, carrier_freq_hz=CARRIER_FREQ,
                         sigma_cycles=SIGMA_CYCLES, samples_per_cycle=SAMPLES_PER_CARRIER):
    """Create BATPING pulse: 40kHz carrier with Gaussian envelope."""
    period_sec = 1.0 / rep_rate_hz
    carrier_period_sec = 1.0 / carrier_freq_hz
    total_carrier_cycles = period_sec * carrier_freq_hz
    n_points = int(total_carrier_cycles * samples_per_cycle)

    if n_points > 16384:
        n_points = 16384

    t = np.linspace(0, period_sec, n_points, endpoint=False)
    t_center = period_sec / 2
    sigma_sec = sigma_cycles * carrier_period_sec

    envelope = np.exp(-0.5 * ((t - t_center) / sigma_sec)**2)
    carrier = np.sin(2 * np.pi * carrier_freq_hz * t)
    pulse = envelope * carrier
    pulse = pulse / np.max(np.abs(pulse))

    return pulse, n_points


def find_ultramic():
    """Find UltraMic device index."""
    pa = pyaudio.PyAudio()
    mic_index = None

    print("Available audio devices:")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"  [{i}] {info['name']} (in: {info['maxInputChannels']}ch)")
            if "UltraMic" in info['name']:
                mic_index = i

    if mic_index is None:
        print("\nUltraMic not found! Available input devices listed above.")
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


def bandpass_filter(audio, fs=SAMPLE_RATE, low=FILTER_LOW, high=FILTER_HIGH):
    """Apply bandpass filter around 40kHz."""
    nyq = fs / 2
    b, a = signal.butter(4, [low/nyq, high/nyq], btype='band')
    return signal.filtfilt(b, a, audio)


def extract_envelope(audio, fs=SAMPLE_RATE):
    """Extract envelope via bandpass + Hilbert transform."""
    filtered = bandpass_filter(audio, fs)
    analytic = signal.hilbert(filtered)
    return np.abs(analytic), filtered


def find_pulses(envelope, fs=SAMPLE_RATE, rep_rate=62.5, threshold_frac=0.3):
    """Find individual pulse peaks in the envelope."""
    period_samples = int(fs / rep_rate)
    threshold = threshold_frac * np.max(envelope)

    peaks, props = signal.find_peaks(envelope, height=threshold, distance=period_samples//2)
    return peaks


def measure_fwhm(envelope, peak_idx, fs=SAMPLE_RATE):
    """Measure FWHM of a single pulse centered at peak_idx."""
    peak_amp = envelope[peak_idx]
    half_max = peak_amp / 2

    # Find left crossing
    left_idx = peak_idx
    for i in range(peak_idx, max(0, peak_idx-5000), -1):
        if envelope[i] < half_max:
            if i < peak_idx - 1:
                frac = (half_max - envelope[i]) / (envelope[i+1] - envelope[i] + 1e-10)
                left_idx = i + frac
            else:
                left_idx = i
            break

    # Find right crossing
    right_idx = peak_idx
    for i in range(peak_idx, min(len(envelope), peak_idx+5000)):
        if envelope[i] < half_max:
            if i > peak_idx + 1:
                frac = (half_max - envelope[i-1]) / (envelope[i] - envelope[i-1] + 1e-10)
                right_idx = i - 1 + frac
            else:
                right_idx = i
            break

    fwhm_samples = right_idx - left_idx
    fwhm_us = fwhm_samples / fs * 1e6

    return fwhm_us, peak_amp


def plot_results(audio, envelope, filtered, peaks, fwhms, output_path):
    """Create visualization of the recording."""
    fs = SAMPLE_RATE
    t_ms = np.arange(len(audio)) / fs * 1000

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # 1. Full raw recording
    axes[0].plot(t_ms, audio, 'b-', linewidth=0.3, alpha=0.7)
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'Raw Recording ({len(audio)} samples, {len(peaks)} pulses detected)')
    axes[0].grid(True, alpha=0.3)

    # 2. Full envelope with detected peaks
    axes[1].plot(t_ms, envelope, 'b-', linewidth=0.5)
    if len(peaks) > 0:
        axes[1].plot(t_ms[peaks], envelope[peaks], 'ro', markersize=4, label=f'{len(peaks)} pulses')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Envelope')
    axes[1].set_title('Envelope (35-45kHz bandpass + Hilbert)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Zoomed single pulse - raw filtered signal
    if len(peaks) > 2:
        center_peak = peaks[len(peaks)//2]
        window = int(0.002 * fs)  # 2ms window
        start = max(0, center_peak - window)
        end = min(len(audio), center_peak + window)

        t_zoom = (np.arange(start, end) - center_peak) / fs * 1e6  # microseconds from peak

        axes[2].plot(t_zoom, filtered[start:end], 'b-', linewidth=0.8)
        axes[2].axhline(0, color='k', linewidth=0.5)
        axes[2].set_xlabel('Time from peak (μs)')
        axes[2].set_ylabel('Amplitude')
        axes[2].set_title('Single Pulse - Filtered 40kHz Signal')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlim(-1500, 1500)

    # 4. Zoomed single pulse - envelope with FWHM
    if len(peaks) > 2:
        axes[3].plot(t_zoom, envelope[start:end], 'b-', linewidth=1.5, label='Envelope')

        peak_amp = envelope[center_peak]
        half_max = peak_amp / 2
        axes[3].axhline(half_max, color='r', linestyle='--', linewidth=1, label=f'Half-max')

        if len(fwhms) > 0:
            mean_fwhm = np.mean(fwhms)
            std_fwhm = np.std(fwhms)
            axes[3].axvline(-mean_fwhm/2, color='g', linestyle=':', linewidth=1.5)
            axes[3].axvline(mean_fwhm/2, color='g', linestyle=':', linewidth=1.5,
                           label=f'FWHM = {mean_fwhm:.1f} ± {std_fwhm:.1f} μs')

        axes[3].set_xlabel('Time from peak (μs)')
        axes[3].set_ylabel('Envelope')
        axes[3].set_title('Single Pulse Envelope with FWHM')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        axes[3].set_xlim(-1500, 1500)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot: {output_path}")
    plt.close()


def main():
    print("=" * 60)
    print("BATPING Piezo Response - Automated Recording")
    print("=" * 60)

    # Find microphone first
    pa, mic_index = find_ultramic()
    if pa is None:
        return 1

    sdg = None
    try:
        # Connect to SDG1062X
        print("\n--- Connecting to SDG1062X ---")
        sdg = SDG1062X_LAN()
        sdg.connect()

        # Query identity
        idn = sdg.query("*IDN?")
        print(f"Device: {idn}")

        # Create and upload waveform
        print("\n--- Uploading BATPING Waveform ---")
        pulse, n_points = create_batping_pulse()
        print(f"  {n_points} points, {REP_RATE_HZ} Hz rep rate, {CARRIER_FREQ/1000:.0f}kHz carrier")

        sdg.set_arb_mode(channel=1)
        sdg.upload_waveform("batping", pulse, freq_hz=REP_RATE_HZ, ampl_vpp=2.0)
        print("  Waveform uploaded")

        # Enable output
        print("\n--- Enabling Output ---")
        sdg.output_on(channel=1)
        print("  CH1 ON")

        # Wait for output to stabilize
        time.sleep(0.5)

        # Record
        print("\n--- Recording ---")
        audio = record_audio(pa, mic_index)

        # Turn off output
        print("\n--- Disabling Output ---")
        sdg.output_off(channel=1)
        print("  CH1 OFF")

        # Process
        print("\n--- Processing ---")
        envelope, filtered = extract_envelope(audio)

        # Find pulses
        peaks = find_pulses(envelope)
        print(f"Found {len(peaks)} pulses")

        # Measure FWHM for each pulse
        fwhms = []
        amplitudes = []
        for peak in peaks:
            fwhm, amp = measure_fwhm(envelope, peak)
            if 50 < fwhm < 2000:  # Sanity check
                fwhms.append(fwhm)
                amplitudes.append(amp)

        if len(fwhms) > 0:
            print(f"\n=== RESULTS ===")
            print(f"FWHM: {np.mean(fwhms):.1f} ± {np.std(fwhms):.1f} μs (n={len(fwhms)})")
            print(f"Amplitude: {np.mean(amplitudes):.4f} ± {np.std(amplitudes):.4f}")
        else:
            print("\nNo valid pulses found! Check connections and signal level.")

        # Save WAV
        wav_path = '/Users/cyrusurheim_1/Downloads/Stubbs_Stuff/Claude_Dir/siglent_project/gaussian_response.wav'
        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(wav_path, SAMPLE_RATE, audio_int16)
        print(f"\nSaved: {wav_path}")

        # Plot
        plot_path = '/Users/cyrusurheim_1/Downloads/Stubbs_Stuff/Claude_Dir/siglent_project/gaussian_response.png'
        plot_results(audio, envelope, filtered, peaks, fwhms, plot_path)

        return 0

    except socket.timeout:
        print(f"\nConnection to SDG timed out. Check IP: {SDG_IP}")
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
        if sdg:
            sdg.output_off(channel=1)
            sdg.disconnect()
        pa.terminate()


if __name__ == "__main__":
    import sys
    sys.exit(main())
