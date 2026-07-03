#!/usr/bin/env python3
"""Quick script to run just MLS-31 test."""

import numpy as np
import pyaudio
import socket
import time
from scipy import signal
from scipy.signal import butter, filtfilt
from scipy.io import wavfile

SDG_IP = "10.11.13.230"
SDG_PORT = 5025
SDG_SAMPLE_RATE = 400000
DODOMIC_SAMPLE_RATE = 384000
CHUNK_SIZE = 4096
CARRIER_FREQ = 41350
SDG_MAX_SAMPLES = 16384
RECORD_DURATION = 0.5

OUTPUT_DIR = '/Users/cyrusurheim_1/Downloads/Stubbs_Stuff/Claude_Dir/siglent_project/'

MLS_TAPS_5 = [4, 1]  # Fixed taps for period-31


class SDG1062X_LAN:
    def __init__(self):
        self.sock = None

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(10.0)
        self.sock.connect((SDG_IP, SDG_PORT))
        print(f"Connected to SDG")

    def disconnect(self):
        if self.sock:
            self.sock.close()

    def write(self, cmd):
        self.sock.sendall((cmd + '\n').encode('utf-8'))

    def write_raw(self, data):
        self.sock.sendall(data)

    def query(self, cmd):
        self.write(cmd)
        time.sleep(0.1)
        return self.sock.recv(4096).decode('utf-8').strip()

    def upload_waveform(self, name, wave_data, freq_hz):
        wave_int16 = np.clip(wave_data, -1.0, 1.0)
        wave_int16 = (wave_int16 * 32767).astype(np.int16)
        bin_data = wave_int16.astype('<i2').tobytes()
        cmd_prefix = f"C1:WVDT WVNM,{name},FREQ,{freq_hz},AMPL,2.0,OFST,0,PHASE,0,WAVEDATA,"
        self.write_raw(cmd_prefix.encode('utf-8') + bin_data + b'\n')
        time.sleep(0.5)
        self.write(f"C1:ARWV NAME,{name}")
        time.sleep(0.2)

    def output_on(self):
        self.write("C1:OUTP ON")

    def output_off(self):
        self.write("C1:OUTP OFF")

    def set_arb_mode(self):
        self.write("C1:BSWV WVTP,ARB")
        time.sleep(0.2)


def generate_mls31():
    """Generate MLS-31 with corrected taps."""
    n_bits = 5
    taps = MLS_TAPS_5
    period = 31
    state = [1] * n_bits

    sequence = []
    for _ in range(period):
        output = state[-1]
        sequence.append(1 if output else -1)
        feedback = 0
        for tap in taps:
            feedback ^= state[tap]
        state = [feedback] + state[:-1]

    print(f"MLS-31 sequence: {sequence}")

    # BPSK modulate at 24 cyc/chip
    cycles_per_chip = 24
    chip_duration = cycles_per_chip / CARRIER_FREQ
    samples_per_chip = int(chip_duration * DODOMIC_SAMPLE_RATE)
    total_samples = samples_per_chip * len(sequence)

    waveform = np.zeros(total_samples)
    for i, chip in enumerate(sequence):
        start = i * samples_per_chip
        end = start + samples_per_chip
        t_chip = np.arange(samples_per_chip) / DODOMIC_SAMPLE_RATE
        waveform[start:end] = chip * 0.9 * np.sin(2 * np.pi * CARRIER_FREQ * t_chip)

    window = signal.windows.tukey(total_samples, alpha=0.1)
    return waveform * window


def create_sdg_buffer(waveform, rep_rate):
    n_samples_sdg = int(len(waveform) * SDG_SAMPLE_RATE / DODOMIC_SAMPLE_RATE)
    waveform_resampled = signal.resample(waveform, n_samples_sdg)
    waveform_resampled = waveform_resampled / np.max(np.abs(waveform_resampled)) * 0.9

    buffer_samples = int(SDG_SAMPLE_RATE / rep_rate)
    min_buffer = int(len(waveform_resampled) * 1.1)
    if buffer_samples < min_buffer:
        buffer_samples = min_buffer
    if buffer_samples > SDG_MAX_SAMPLES:
        buffer_samples = SDG_MAX_SAMPLES

    waveform_buffer = np.zeros(buffer_samples)
    waveform_buffer[:len(waveform_resampled)] = waveform_resampled
    return waveform_buffer, SDG_SAMPLE_RATE / buffer_samples


def find_ultramic():
    pa = pyaudio.PyAudio()
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0 and "UltraMic" in info['name']:
            print(f"Using UltraMic at index {i}")
            return pa, i
    pa.terminate()
    return None, None


def record_audio(pa, mic_index):
    n_samples = int(RECORD_DURATION * DODOMIC_SAMPLE_RATE)
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=DODOMIC_SAMPLE_RATE,
                     input=True, input_device_index=mic_index, frames_per_buffer=CHUNK_SIZE)
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


def bandpass_filter(sig):
    nyq = DODOMIC_SAMPLE_RATE / 2
    b, a = butter(4, [35000 / nyq, 45000 / nyq], btype='band')
    return filtfilt(b, a, sig)


def main():
    print("=" * 50)
    print("MLS-31 Only (Fixed)")
    print("=" * 50)

    pa, mic_index = find_ultramic()
    if pa is None:
        return

    sdg = SDG1062X_LAN()
    try:
        sdg.connect()
        sdg.set_arb_mode()

        mls_wf = generate_mls31()
        print(f"Waveform: {len(mls_wf)} samples, {len(mls_wf)/DODOMIC_SAMPLE_RATE*1000:.1f} ms")

        wf_buffer, actual_rep = create_sdg_buffer(mls_wf, 40)
        print(f"Buffer: {len(wf_buffer)} samples, rep rate: {actual_rep:.1f} Hz")

        sdg.upload_waveform("mls31_fix", wf_buffer, actual_rep)
        time.sleep(0.3)
        sdg.output_on()
        time.sleep(0.1)

        print("Recording...")
        received = record_audio(pa, mic_index)
        sdg.output_off()

        received_filt = bandpass_filter(received)

        wav_path = OUTPUT_DIR + "mls31_fixed.wav"
        audio_int16 = (received_filt * 32767).astype(np.int16)
        wavfile.write(wav_path, DODOMIC_SAMPLE_RATE, audio_int16)
        print(f"Saved: {wav_path}")

    finally:
        sdg.output_off()
        sdg.disconnect()
        pa.terminate()

    print("Done!")


if __name__ == '__main__':
    main()
