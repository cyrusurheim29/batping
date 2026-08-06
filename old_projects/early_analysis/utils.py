"""
Ultrasonic signal analysis utilities.
Shared functions for WAV loading, FFT analysis, and plotting.
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from pathlib import Path


def load_wav(filepath):
    """
    Load a WAV file and return sample rate and normalized signal.

    Returns:
        sample_rate: int, samples per second
        signal: numpy array, normalized to [-1, 1]
    """
    sample_rate, data = wavfile.read(filepath)

    # Handle stereo by taking first channel
    if len(data.shape) > 1:
        data = data[:, 0]

    # Normalize based on dtype
    if data.dtype == np.int16:
        signal = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        signal = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.float32 or data.dtype == np.float64:
        signal = data.astype(np.float64)
    else:
        signal = data.astype(np.float64) / np.max(np.abs(data))

    return sample_rate, signal


def compute_fft(signal, sample_rate, window='hann', zero_pad_factor=4):
    """
    Compute FFT with windowing and zero-padding for improved frequency resolution.

    Args:
        signal: input time-domain signal
        sample_rate: samples per second
        window: window type ('hann', 'hamming', 'blackman', or None)
        zero_pad_factor: multiply FFT length by this factor

    Returns:
        freqs: frequency axis in Hz
        magnitude: magnitude spectrum (linear scale)
        magnitude_db: magnitude spectrum in dB (relative to max)
    """
    n = len(signal)

    # Apply window
    if window == 'hann':
        win = np.hanning(n)
    elif window == 'hamming':
        win = np.hamming(n)
    elif window == 'blackman':
        win = np.blackman(n)
    else:
        win = np.ones(n)

    windowed = signal * win

    # Zero-pad for better frequency resolution
    nfft = n * zero_pad_factor

    # Compute FFT
    fft_result = np.fft.rfft(windowed, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0/sample_rate)

    # Magnitude (compensate for window energy loss)
    window_compensation = 2.0 / np.sum(win)
    magnitude = np.abs(fft_result) * window_compensation

    # Convert to dB
    magnitude_db = 20 * np.log10(magnitude + 1e-12)
    magnitude_db -= np.max(magnitude_db)  # Normalize to 0 dB peak

    return freqs, magnitude, magnitude_db


def find_fundamental(freqs, magnitude, expected_freq=None, search_width_hz=2000):
    """
    Find the fundamental frequency peak in a spectrum.

    Args:
        freqs: frequency axis
        magnitude: magnitude spectrum
        expected_freq: if provided, search near this frequency
        search_width_hz: search window width around expected frequency

    Returns:
        fund_freq: fundamental frequency in Hz
        fund_mag: magnitude at fundamental
        fund_idx: index in arrays
    """
    if expected_freq is not None:
        # Search within window around expected frequency
        mask = (freqs >= expected_freq - search_width_hz) & (freqs <= expected_freq + search_width_hz)
        search_mag = magnitude.copy()
        search_mag[~mask] = 0
    else:
        search_mag = magnitude

    # Find peaks
    peaks, properties = find_peaks(search_mag, height=np.max(search_mag) * 0.1)

    if len(peaks) == 0:
        # Fallback to max
        fund_idx = np.argmax(search_mag)
    else:
        # Take highest peak
        fund_idx = peaks[np.argmax(search_mag[peaks])]

    return freqs[fund_idx], magnitude[fund_idx], fund_idx


def find_harmonics(freqs, magnitude, fundamental_freq, num_harmonics=5, search_width_hz=1000):
    """
    Find harmonic peaks given a fundamental frequency.

    Args:
        freqs: frequency axis
        magnitude: magnitude spectrum
        fundamental_freq: fundamental frequency in Hz
        num_harmonics: number of harmonics to find (including fundamental)
        search_width_hz: search window around each harmonic

    Returns:
        harmonic_freqs: list of found harmonic frequencies
        harmonic_mags: list of magnitudes at harmonics
    """
    harmonic_freqs = []
    harmonic_mags = []

    max_freq = freqs[-1]

    for n in range(1, num_harmonics + 1):
        target_freq = fundamental_freq * n

        if target_freq > max_freq - search_width_hz:
            break

        freq, mag, _ = find_fundamental(freqs, magnitude, target_freq, search_width_hz)
        harmonic_freqs.append(freq)
        harmonic_mags.append(mag)

    return harmonic_freqs, harmonic_mags


def calculate_thd(harmonic_magnitudes):
    """
    Calculate Total Harmonic Distortion.

    THD = sqrt(sum of squared harmonic magnitudes) / fundamental magnitude * 100%

    Args:
        harmonic_magnitudes: list where [0] is fundamental, [1:] are harmonics

    Returns:
        thd_percent: THD as a percentage
    """
    if len(harmonic_magnitudes) < 2:
        return 0.0

    fundamental = harmonic_magnitudes[0]
    if fundamental < 1e-12:
        return 0.0

    harmonics = harmonic_magnitudes[1:]
    harmonic_power = np.sum(np.array(harmonics) ** 2)

    thd = np.sqrt(harmonic_power) / fundamental * 100.0
    return thd


def parse_frequency_from_filename(filepath):
    """
    Extract frequency in Hz from filename.
    Expects format like '40000.wav', '40kHz.wav', '40000Hz.wav', or 'prefix_40khz_suffix.wav'

    Returns:
        frequency in Hz, or None if unable to parse
    """
    filepath = Path(filepath)
    stem = filepath.stem.lower()

    import re

    # First try: match number followed by khz or hz (e.g., "40khz", "40000hz")
    match = re.search(r'(\d+(?:\.\d+)?)\s*(khz|hz)', stem)
    if match:
        value = float(match.group(1))
        unit = match.group(2)
        if unit == 'khz':
            return value * 1000
        else:
            return value

    # Second try: match standalone number that looks like a frequency
    # Look for number preceded by underscore or start, not part of a date
    match = re.search(r'(?:^|_)(\d{4,6})(?:_|$)', stem)
    if match:
        value = float(match.group(1))
        return value  # Assume Hz

    return None


def plot_spectrum(freqs, magnitude_db, title='Spectrum', fundamental_freq=None,
                  harmonic_freqs=None, max_freq=None, save_path=None):
    """
    Plot a frequency spectrum with optional harmonic markers.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if max_freq is not None:
        mask = freqs <= max_freq
        freqs = freqs[mask]
        magnitude_db = magnitude_db[mask]

    ax.plot(freqs / 1000, magnitude_db, 'b-', linewidth=0.8)

    if fundamental_freq is not None:
        ax.axvline(fundamental_freq / 1000, color='r', linestyle='--',
                   alpha=0.7, label=f'Fundamental: {fundamental_freq/1000:.1f} kHz')

    if harmonic_freqs is not None:
        for i, hf in enumerate(harmonic_freqs[1:], 2):
            ax.axvline(hf / 1000, color='orange', linestyle=':', alpha=0.5,
                      label=f'H{i}: {hf/1000:.1f} kHz' if i <= 4 else None)

    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(-80, 5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

    return fig, ax


def plot_transfer_function(frequencies_hz, magnitudes_db, title='Transfer Function',
                           save_path=None):
    """
    Plot transfer function (response vs frequency).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    freqs_khz = np.array(frequencies_hz) / 1000

    ax.plot(freqs_khz, magnitudes_db, 'b-o', linewidth=1.5, markersize=6)

    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Relative Response (dB)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Set reasonable y-axis limits
    y_min = min(magnitudes_db) - 5
    y_max = max(magnitudes_db) + 5
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

    return fig, ax
