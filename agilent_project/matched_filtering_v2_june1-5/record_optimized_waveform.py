#!/usr/bin/env python3
"""
Record matched filter data using the Bayesian-optimized waveform.

Sends the optimized waveform at 50Hz and records the response.
"""

import numpy as np
import pyaudio
import time
from datetime import datetime
from scipy.io import wavfile
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agilent_33220a import Agilent33220A


def record_audio(pa, mic_index, duration_s, sample_rate=384000):
    """Record audio from the UltraMic."""
    n_samples = int(duration_s * sample_rate)
    chunk_size = 4096

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        input_device_index=mic_index,
        frames_per_buffer=chunk_size
    )

    frames = []
    remaining = n_samples
    while remaining > 0:
        chunk = min(chunk_size, remaining)
        data = stream.read(chunk, exception_on_overflow=False)
        frames.append(data)
        remaining -= chunk

    stream.stop_stream()
    stream.close()

    audio = np.frombuffer(b''.join(frames), dtype=np.int16)
    return audio


def find_ultramic(pa):
    """Find the UltraMic device index."""
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if "UltraMic" in info['name'] and info['maxInputChannels'] > 0:
            return i
    return None


def main():
    # Parameters
    REP_FREQ = 50  # Hz
    VPP = 10.0  # Volts peak-to-peak
    RECORD_DURATION = 0.5  # seconds (captures ~25 pulses at 50Hz)
    SAMPLE_RATE = 384000  # Hz
    SETTLING_TIME = 0.5  # seconds to let AWG stabilize

    print("=" * 60)
    print("Recording with Bayesian-Optimized Waveform")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Rep rate: {REP_FREQ} Hz")
    print(f"Amplitude: {VPP} Vpp")
    print(f"Record duration: {RECORD_DURATION} s")

    # Load optimized waveform
    bayesian_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'bayesian_optimization',
        'bayesian_optimizer_state.npz'
    )

    print(f"\nLoading optimized waveform from:\n  {bayesian_path}")
    data = np.load(bayesian_path)
    waveform = data['best_waveform']
    print(f"Waveform: {len(waveform)} samples")
    print(f"Original FWHM: {float(data['best_fwhm']):.1f} μs (vs {float(data['baseline_fwhm']):.1f} μs baseline)")

    # Connect to AWG
    print("\nConnecting to Agilent 33220A...")
    awg = Agilent33220A()
    awg.connect()

    # Initialize audio
    print("Initializing audio...")
    pa = pyaudio.PyAudio()
    mic_index = find_ultramic(pa)
    if mic_index is None:
        raise RuntimeError("UltraMic not found")
    print(f"Found UltraMic at index {mic_index}")

    try:
        # Upload and output waveform
        print(f"\nUploading waveform and setting output...")
        awg.output_arbitrary(waveform, freq_hz=REP_FREQ, vpp=VPP)
        print(f"AWG status: {awg.status()}")

        # Wait for settling
        print(f"Waiting {SETTLING_TIME}s for settling...")
        time.sleep(SETTLING_TIME)

        # Record
        print(f"\nRecording {RECORD_DURATION}s of audio...")
        audio = record_audio(pa, mic_index, RECORD_DURATION, SAMPLE_RATE)
        print(f"Recorded {len(audio)} samples ({len(audio)/SAMPLE_RATE*1000:.1f} ms)")

        # Save as WAV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"optimized_2m_reflection_{REP_FREQ}Hz.wav"
        wavfile.write(filename, SAMPLE_RATE, audio)
        print(f"\nSaved: {filename}")

        # Also save the waveform used for reference
        np.savez(
            f"optimized_waveform_used.npz",
            waveform=waveform,
            rep_freq=REP_FREQ,
            vpp=VPP,
            original_fwhm=float(data['best_fwhm']),
            baseline_fwhm=float(data['baseline_fwhm'])
        )
        print(f"Saved: optimized_waveform_used.npz")

    finally:
        print("\nDisabling output and disconnecting...")
        awg.set_output(False)
        awg.disconnect()
        pa.terminate()

    print("\nDone!")


if __name__ == '__main__':
    main()
