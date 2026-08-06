#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LFM Chirp Sweep (V1)

Sweeps chirp bandwidth (f_start, f_stop) and duration. Uses matched filter
(cross-correlation of received signal with transmitted template) for analysis,
giving FWHM ≈ 0.886/B_chirp for a flat-spectrum LFM.

Also tests amplitude-compensated chirps where the envelope is pre-distorted
by 1/|H(f)| so the transmitted spectrum is flattened across a wider band.

Piezo passband (from stepped-sine characterization):
  -3 dB:  40500–42200 Hz  (1700 Hz)
  -6 dB:  39100–42850 Hz  (3750 Hz)
  center: 41350 Hz

@author: cyrusurheim
Created: July 2026
"""

import numpy as np
import pyaudio
import socket
import time
import os
import json
from datetime import datetime
from scipy import signal as scipy_signal
from scipy.signal import butter, filtfilt, hilbert
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Hardware ─────────────────────────────────────────────────────────────────
SDG_IP              = "10.11.13.230"
SDG_PORT            = 5025
SDG_SAMPLE_RATE     = 400000        # 400 kSa/s
DODOMIC_SAMPLE_RATE = 384000        # UltraMic
CHUNK_SIZE          = 4096
SDG_MAX_SAMPLES     = 16384
AMPLITUDE           = 0.9
RECORD_DURATION     = 30.0          # s — ~1500-1800 periods at 60Hz

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR       = '/Users/cyrusurheim_1/Downloads/Stubbs_Stuff/Claude_Dir/siglent_project/'
# Burst stepped-sine characterization, 8ms/1.2Vpp, Jul 15 2026 — no standing-wave bias
PIEZO_CHAR_FILE  = OUTPUT_DIR + 'stepped_sine_characterization_burst8ms_20260715_152456.npz'
VERSION          = "V2"  # V2: new burst H(f) characterization

# Piezo passband from burst stepped-sine (Jul 15 2026, 8ms/1.2Vpp):
#   -3 dB:  40400–42850 Hz  (2450 Hz)
#   -6 dB:  39250–43600 Hz  (4350 Hz)
#  -10 dB:  37200–45000 Hz  (7800 Hz)
#   center: 41450 Hz
# (Previous 300ms continuous: center 41350 Hz, -3dB 1700 Hz — narrowed by standing waves)

# ── Chirp configurations ──────────────────────────────────────────────────────
# (f_start Hz, f_stop Hz, duration_ms, label, amplitude_shape, window, ampl_vpp)
# ampl_vpp: starting values from Jul 8 UltraMic calibration — NEED RE-CALIBRATION
#   after first run with new H(f) since compensation envelopes have changed.
CHIRP_CONFIGS = [
    # 30s recordings → ~1800 periods at ~60Hz: 100 for Stubbs template + 1700 eval.

    # ── Best performers — Stubbs analysis configs ─────────────────────────────
    # BSWV H(f), direct SDG, burst-targeted calibration, max 20V — 2026-07-28
    (31475, 51475, 4.0, 'bw_20kHz_wide_4ms_comp',      'piezo_compensated',       'tukey', 20.00),
    (28975, 53975, 2.0, 'bw_25kHz_wide_2ms_comp',       'piezo_compensated',       'tukey', 20.00),
    (37200, 45000, 2.0, 'bw_10dB_asym_2ms_comp_rect',   'piezo_compensated',       'rect',  20.00),
    (37200, 45000, 2.0, 'bw_10dB_asym_2ms_comp',        'piezo_compensated',       'tukey', 20.00),
    (38450, 44450, 2.0, 'bw_6kHz_sym_2ms_flat',         'flat',                    'tukey', 20.00, 23.0),

    # ── Reduced-compensation variants ─────────────────────────────────────────
    (31475, 51475, 4.0, 'bw_20kHz_wide_4ms_comp_light', 'piezo_compensated_light', 'tukey', 20.00),
    (37200, 45000, 2.0, 'bw_10dB_asym_2ms_comp_light',  'piezo_compensated_light', 'tukey', 20.00),

    # ── Diffraction-compensated variants ──────────────────────────────────────
    (31475, 51475, 4.0, 'bw_20kHz_wide_4ms_diff', 'piezo_compensated_diffraction', 'tukey', 20.00),
    (37200, 45000, 2.0, 'bw_10dB_asym_2ms_diff',  'piezo_compensated_diffraction', 'tukey', 20.00),

    # ── Flat -3dB BW chirp (no compensation, full resonance drive) ────────────
    (40300, 42600, 4.0, 'flat3db_40p3_42p6kHz',         'flat',                    'tukey', 20.00),
]

# TOF search window for 2 m round trip (~5800 μs)
# Tight window: ±700 μs around expected 5800 μs avoids room echoes and boundary artifacts
TOF_SEARCH_MIN_US = 5000
TOF_SEARCH_MAX_US = 6500


# ── SDG1062X LAN class ────────────────────────────────────────────────────────

class SDG1062X_LAN:
    def __init__(self, ip=SDG_IP, port=SDG_PORT, timeout=10.0):
        self.ip, self.port, self.timeout = ip, port, timeout
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
        bin_data   = wave_int16.astype('<i2').tobytes()
        cmd_prefix = (f"C{channel}:WVDT WVNM,{name},FREQ,{freq_hz},"
                      f"AMPL,{ampl_vpp},OFST,0,PHASE,0,WAVEDATA,")
        self.write_raw(cmd_prefix.encode('utf-8') + bin_data + b'\n')
        time.sleep(0.5)
        self.write(f"C{channel}:ARWV NAME,{name}")
        time.sleep(0.2)

    def delete_waveform(self, name, channel=1):
        """Delete a user-defined waveform from SDG storage (WVDW command)."""
        self.write(f"C{channel}:WVDW WVNM,{name}")
        time.sleep(0.2)

    def set_arb_mode(self, channel=1):
        self.write(f"C{channel}:BSWV WVTP,ARB")
        time.sleep(0.2)

    def output_on(self, channel=1):
        self.write(f"C{channel}:OUTP ON")

    def output_off(self, channel=1):
        self.write(f"C{channel}:OUTP OFF")


# ── Piezo response loader ─────────────────────────────────────────────────────

def load_piezo_response():
    """
    Load stepped-sine frequency response for amplitude compensation.
    Applies w=71 SG smooth before converting to linear so the compensation
    envelope is clean — the raw burst data is too noisy at w=11.
    """
    try:
        from scipy.signal import savgol_filter as _sg
        d = np.load(PIEZO_CHAR_FILE)
        freqs = d['freqs'].astype(float)
        mag_db = _sg(d['H_mag'], window_length=71, polyorder=3)
        mag_lin = 10.0 ** (mag_db / 20.0)        # dB → linear amplitude
        mag_lin = mag_lin / np.max(mag_lin)       # normalize peak = 1
        print(f"Loaded piezo response: {len(freqs)} pts, "
              f"{freqs[0]/1000:.0f}–{freqs[-1]/1000:.0f} kHz  (SG w=71)")
        return {'freq': freqs, 'magnitude': mag_lin}
    except Exception as e:
        print(f"Could not load piezo response: {e}")
        return None


# ── Chirp generation ──────────────────────────────────────────────────────────

def generate_lfm_chirp(f_start, f_stop, duration_ms, sample_rate,
                       amplitude_shape='flat', piezo_response=None,
                       window='tukey', alpha=1.0):
    """
    Generate LFM (up-)chirp from f_start to f_stop over duration_ms.

    amplitude_shape='flat':
        Constant amplitude envelope — standard LFM, flat power spectrum over B.

    amplitude_shape='piezo_compensated':
        Amplitude ∝ 1/|H(f_inst)| so the spectrum seen at the reflector is
        flatter, extending useful bandwidth beyond the -3 dB piezo rolloff.
        Clipped at 10% of peak response to avoid blowing up in the noise floor.

    Returns: (t, chirp_signal, f_instantaneous)
    """
    dur_s  = duration_ms / 1000.0
    n      = int(dur_s * sample_rate)
    t      = np.arange(n) / sample_rate
    f_inst = f_start + (f_stop - f_start) / dur_s * t
    phase  = 2 * np.pi * (f_start * t + (f_stop - f_start) / (2 * dur_s) * t ** 2)

    if amplitude_shape in ('piezo_compensated', 'piezo_compensated_light') and piezo_response is not None:
        mag = np.interp(f_inst, piezo_response['freq'], piezo_response['magnitude'])
        floor = 0.10 if amplitude_shape == 'piezo_compensated' else 0.20  # 10%=20dB, 20%=14dB
        mag = np.clip(mag, floor, None)
        amp_env = AMPLITUDE / mag
        amp_env = amp_env / np.max(amp_env) * AMPLITUDE
    elif amplitude_shape == 'alpha_comp' and piezo_response is not None:
        # amp(f) ∝ H(f)^{-alpha}: alpha=0 → flat, alpha=1 → full 1/H
        # Smooth power-law, no hard floor — avoids ring-down overexcitation at high alpha
        mag = np.interp(f_inst, piezo_response['freq'], piezo_response['magnitude'])
        mag = np.clip(mag, 0.05, None)   # soft floor at -26 dB to avoid numeric blowup
        amp_env = mag ** (-alpha)
        amp_env = amp_env / np.max(amp_env) * AMPLITUDE
    elif amplitude_shape == 'piezo_compensated_diffraction' and piezo_response is not None:
        # 1/H(f) compensation + 1/λ = f/c diffraction pre-emphasis (two-way amplitude loss ∝ c/f).
        # Normalise diffraction factor to 1 at band centre so the floor clipping is meaningful.
        f_center = (f_start + f_stop) / 2.0
        mag = np.interp(f_inst, piezo_response['freq'], piezo_response['magnitude'])
        diffraction = f_inst / f_center      # >1 for high f (needs less boost), <1 for low f
        effective_mag = np.clip(mag / diffraction, 0.20, None)   # 20% floor = 14 dB max
        amp_env = AMPLITUDE / effective_mag
        amp_env = amp_env / np.max(amp_env) * AMPLITUDE
    else:
        amp_env = AMPLITUDE

    chirp = amp_env * np.sin(phase)
    if window == 'tukey':
        chirp *= scipy_signal.windows.tukey(n, alpha=0.05)
    elif window == 'hann':
        chirp *= scipy_signal.windows.hann(n)
    # window='rect' (or anything else): no taper applied
    return t, chirp, f_inst


def create_sdg_buffer(chirp_at_dodo_rate, duration_ms, dead_ms=15.0):
    """
    Resample chirp to SDG rate and zero-pad to rep period.

    dead_ms: silence after chirp end. Default 15ms. Increase for waveforms
    whose burst+reflection pair lands near the period boundary (e.g. flat 6kHz
    at new mic geometry: use dead_ms=23 to get 25ms period at 40Hz).
    """
    TOF_MARGIN_MS  = dead_ms
    period_ms      = duration_ms + TOF_MARGIN_MS
    buf_samples    = min(int(period_ms / 1000 * SDG_SAMPLE_RATE), SDG_MAX_SAMPLES)
    actual_rep_rate = SDG_SAMPLE_RATE / buf_samples

    n_sdg      = int(duration_ms / 1000 * SDG_SAMPLE_RATE)
    chirp_sdg  = scipy_signal.resample(chirp_at_dodo_rate, n_sdg)
    peak       = np.max(np.abs(chirp_sdg))
    if peak > 0:
        chirp_sdg = chirp_sdg / peak * 0.9

    buf = np.zeros(buf_samples)
    buf[:min(n_sdg, buf_samples)] = chirp_sdg[:min(n_sdg, buf_samples)]
    return buf, actual_rep_rate


# ── Recording ─────────────────────────────────────────────────────────────────

def find_ultramic():
    pa = pyaudio.PyAudio()
    mic_index = None
    print("\nAvailable audio devices:")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"  [{i}] {info['name']}")
            if 'UltraMic' in info['name']:
                mic_index = i
    if mic_index is None:
        print("UltraMic not found!")
        pa.terminate()
        return None, None
    print(f"Using UltraMic at index {mic_index}")
    return pa, mic_index


def record_audio(pa, mic_index, duration=RECORD_DURATION):
    n = int(duration * DODOMIC_SAMPLE_RATE)
    stream = pa.open(format=pyaudio.paInt16, channels=1,
                     rate=DODOMIC_SAMPLE_RATE, input=True,
                     input_device_index=mic_index, frames_per_buffer=CHUNK_SIZE)
    frames, remaining = [], n
    while remaining > 0:
        chunk = min(CHUNK_SIZE, remaining)
        frames.append(stream.read(chunk, exception_on_overflow=False))
        remaining -= chunk
    stream.stop_stream()
    stream.close()
    return np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float64) / 32768.0


def record_period_by_period(pa, mic_index, rep_rate, n_periods, mode='A', settle_s=2.0):
    """
    Record n_periods independent periods, one at a time.

    mode='A': stream stays open for the whole run; 2 s settle once at the start.
              Avoids per-period stream latency overhead.  SDG runs continuously.
    mode='B': stream is opened and closed for every period; settle_s wait each time.
              Cleanest isolation — no inter-period memory in the USB driver.

    Returns array of shape (n_periods, period_samples).
    Each row is one period ready to be saved as its own WAV.
    """
    period_samples = int(DODOMIC_SAMPLE_RATE / rep_rate)
    periods = []

    def _open_stream():
        return pa.open(format=pyaudio.paInt16, channels=1,
                       rate=DODOMIC_SAMPLE_RATE, input=True,
                       input_device_index=mic_index,
                       frames_per_buffer=CHUNK_SIZE)

    def _read_period(stream):
        frames, remaining = [], period_samples
        while remaining > 0:
            chunk = min(CHUNK_SIZE, remaining)
            frames.append(stream.read(chunk, exception_on_overflow=False))
            remaining -= chunk
        raw = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float64)
        return raw / 32768.0

    if mode == 'A':
        stream = _open_stream()
        time.sleep(settle_s)
        for i in range(n_periods):
            periods.append(_read_period(stream))
            if (i + 1) % 10 == 0:
                print(f'    ppp-A: {i+1}/{n_periods} periods', flush=True)
        stream.stop_stream(); stream.close()
    else:  # mode B
        for i in range(n_periods):
            stream = _open_stream()
            time.sleep(settle_s)
            periods.append(_read_period(stream))
            stream.stop_stream(); stream.close()
            if (i + 1) % 5 == 0:
                print(f'    ppp-B: {i+1}/{n_periods} periods', flush=True)

    return np.array(periods)   # (n_periods, period_samples)


# ── Analysis ──────────────────────────────────────────────────────────────────

def bandpass_filter(sig, fs, f_low, f_high, order=8):
    """Bandpass filter using SOS (numerically stable at order 8)."""
    from scipy.signal import sosfiltfilt
    nyq = fs / 2
    sos = butter(order, [f_low / nyq, f_high / nyq], btype='band', output='sos')
    return sosfiltfilt(sos, sig)


def find_peak_parabolic(envelope, lags_us, search_min_us, search_max_us):
    """Argmax + parabolic sub-sample interpolation within search range."""
    mask = (lags_us >= search_min_us) & (lags_us <= search_max_us)
    if not np.any(mask):
        return None, None
    region    = envelope[mask]
    lags_reg  = lags_us[mask]
    idx = np.argmax(region)
    if idx < 1 or idx >= len(region) - 1:
        return float(lags_reg[idx]), float(region[idx])
    y_m, y_0, y_p = region[idx-1], region[idx], region[idx+1]
    denom = 2 * (y_m - 2*y_0 + y_p)
    delta = np.clip((y_m - y_p) / denom, -0.5, 0.5) if abs(denom) > 1e-10 else 0.0
    step    = float(lags_reg[1] - lags_reg[0])
    peak_us = float(lags_reg[idx]) + delta * step
    peak_v  = float(y_0 - (y_m - y_p) * delta / 4)
    return peak_us, peak_v


def find_peak_parabolic_11pt(envelope, lags_us, search_min_us, search_max_us):
    """
    Argmax then 11-point least-squares parabolic fit (±5 samples around peak).
    Returns (peak_us, peak_val, sigma_peak_us):
      peak_us       — refined TOF estimate (μs)
      peak_val      — fitted peak amplitude
      sigma_peak_us — 1-sigma uncertainty on peak position from fit residuals (μs)
    Returns (None, None, None) if region is empty or fit is degenerate.
    """
    mask = (lags_us >= search_min_us) & (lags_us <= search_max_us)
    if not np.any(mask):
        return None, None, None
    region, lags_r = envelope[mask], lags_us[mask]
    idx = int(np.argmax(region))
    step = float(lags_r[1] - lags_r[0])    # μs per sample
    hw = 5
    lo, hi = max(0, idx - hw), min(len(region), idx + hw + 1)
    x = np.arange(lo, hi, dtype=float) - idx   # sample offsets centred at argmax
    y = region[lo:hi]
    if len(y) < 5 or np.ptp(y) < 1e-12:
        return float(lags_r[idx]), float(region[idx]), step
    coeffs, residuals, _, _, _ = np.polyfit(x, y, 2, full=True)
    a, b, _ = coeffs
    if abs(a) < 1e-10:
        return float(lags_r[idx]), float(region[idx]), step
    delta = float(np.clip(-b / (2 * a), -hw, hw))   # sub-sample offset from idx
    peak_us  = float(lags_r[idx]) + delta * step
    peak_val = float(np.polyval(coeffs, delta))
    # σ_peak = rms_residual / (|a| × sqrt(Σx²)) in samples, converted to μs
    rms = float(np.sqrt(residuals[0] / len(y))) if len(residuals) > 0 else step * 0.1
    sigma_peak_us = rms / (abs(a) * float(np.sqrt(np.sum(x ** 2)))) * step
    return peak_us, peak_val, sigma_peak_us


def measure_fwhm_at(envelope, lags_us, peak_us):
    """Half-max crossing FWHM centred on peak_us."""
    idx = int(np.argmin(np.abs(lags_us - peak_us)))
    half = envelope[idx] / 2.0
    l = idx
    while l > 0 and envelope[l] > half:
        l -= 1
    r = idx
    while r < len(envelope) - 1 and envelope[r] > half:
        r += 1
    return (r - l) / DODOMIC_SAMPLE_RATE * 1e6


def per_period_sigma_tof(received, rep_rate, tof_min_us=5000, tof_max_us=6500,
                         analysis_duration_s=3.0, fs=None):
    """
    Per-period σ_TOF — the real temperature precision metric.
    Slices received signal into periods, measures TOF from each via autocorr +
    parabolic interpolation, returns (mean_tof_us, std_tof_us, n_periods).

    analysis_duration_s: clip to this many seconds for the live analysis
    (full WAV is saved for offline Stubbs analysis — no need to process all 25s here).
    fs: sample rate in Hz (defaults to DODOMIC_SAMPLE_RATE if not provided).
    """
    if fs is None:
        fs = DODOMIC_SAMPLE_RATE
    # Clip to first analysis_duration_s for fast live analysis
    clip = int(analysis_duration_s * fs)
    sig = bandpass_filter(received[:clip], fs, 28000, 65000)
    period_idx = int(fs / rep_rate)
    env_n = np.abs(hilbert(sig)); env_n /= np.max(env_n)
    # Find trigger with numpy (not a Python loop over millions of samples)
    start_idx = int(0.008 * fs)
    above = env_n[start_idx:] > 0.25
    first = np.argmax(above)
    T0 = (start_idx + first) - int(0.008 * fs) if above[first] else 0
    if T0 < 0: T0 = 0
    tofs = []; fit_sigmas = []
    for i in range((len(sig) - T0) // period_idx - 1):
        seg = sig[T0 + i*period_idx : T0 + (i+1)*period_idx]
        ac = np.correlate(seg, seg, mode='full')
        ctr = len(ac) // 2
        lu = (np.arange(len(ac)) - ctr) / fs * 1e6
        ac /= np.max(np.abs(ac))
        ev = np.abs(hilbert(ac)); ev /= np.max(ev)
        peak_us, _, sigma_us = find_peak_parabolic_11pt(ev, lu, tof_min_us, tof_max_us)
        if peak_us is None:
            continue
        tofs.append(peak_us)
        if sigma_us is not None:
            fit_sigmas.append(sigma_us)
    tofs = np.array(tofs)
    if len(tofs) < 3:
        return None, None, len(tofs), None
    mean_fit_sigma = float(np.mean(fit_sigmas)) if fit_sigmas else None
    return float(np.mean(tofs)), float(np.std(tofs)), len(tofs), mean_fit_sigma


def hp_filter_30k(sig, fs, order=8):
    """Order-8 HP at 30 kHz via SOS (rejects 25 kHz lab noise, passes piezo band)."""
    from scipy.signal import sosfiltfilt
    sos = butter(order, 30000 / (fs / 2), btype='high', output='sos')
    return sosfiltfilt(sos, sig)


def matched_filter_analysis(received, template, f_start, f_stop, fs=None,
                            clip_s=1.0):
    """
    Full-signal autocorrelation approach.

    The matched-filter (cross-correlation with transmitted template) fails when
    the chirp duration is comparable to TOF (~5.7 ms) because incident and
    reflected chirps overlap in the recording.  The autocorrelation of the
    received signal is robust regardless of chirp duration — all 20-40
    repetitions contribute to the TOF peak with no phase alignment needed.

    Returns a dict with the same keys as before so the rest of the pipeline
    is unchanged.  xcorr / xcorr_env / lags_us now hold the autocorrelation
    rather than the cross-correlation; tof_us and fwhm_us are the TOF peak
    position and FWHM; incident_us and sidelobe_db are set to None.
    fs: sample rate in Hz (defaults to DODOMIC_SAMPLE_RATE if not provided).
    clip_s: seconds of signal to use (default 1.0 — fast at high sample rates).
    """
    if fs is None:
        fs = DODOMIC_SAMPLE_RATE

    # Clip for live analysis — enough periods for a clear autocorr peak.
    clip = int(clip_s * fs)
    rx  = bandpass_filter(received[:clip], fs, 28000, 65000)

    # Use scipy.signal.correlate (FFT-based, O(N log N)) not np.correlate (O(N²)).
    # np.correlate on 1M+ samples would take hours; scipy handles it in <1s.
    from scipy.signal import correlate as sp_correlate
    xcorr   = sp_correlate(rx, rx, mode='full')
    center  = len(xcorr) // 2
    lags_us = (np.arange(len(xcorr)) - center) / fs * 1e6
    xcorr  /= np.max(np.abs(xcorr))

    xcorr_env = np.abs(hilbert(xcorr))
    xcorr_env /= np.max(xcorr_env) + 1e-10

    # TOF peak
    tof_us, tof_val = find_peak_parabolic(
        xcorr_env, lags_us, TOF_SEARCH_MIN_US, TOF_SEARCH_MAX_US)

    # FWHM at TOF peak
    fwhm_us = measure_fwhm_at(xcorr_env, lags_us, tof_us) if tof_us is not None else None

    # Lag-0 FWHM (bandwidth reference — set on returned dict as extra key)
    half0 = 0.5
    li = center
    while li > 0 and xcorr_env[li] > half0: li -= 1
    ri = center
    while ri < len(xcorr_env) - 1 and xcorr_env[ri] > half0: ri += 1
    lag0_fwhm_us = (ri - li) / fs * 1e6

    # TOF/lag-0 ratio
    tof_idx  = int(np.argmin(np.abs(lags_us - tof_us))) if tof_us is not None else None
    ratio_db = (20 * np.log10(xcorr_env[tof_idx] / xcorr_env[center] + 1e-12)
                if tof_idx is not None else None)

    # ── SNR at the TOF peak ───────────────────────────────────────────────────
    # Noise floor = RMS of the Hilbert envelope in a signal-free region
    # (15–20 ms — beyond any acoustic arrival for a 2m bench path).
    # SNR_amplitude = A_TOF / sigma_noise.
    # Theoretical σ_TOF from Cramér-Rao: FWHM / (3.33 × SNR_amplitude).
    noise_mask = (lags_us > 15000) & (lags_us < 20000)
    if np.any(noise_mask) and tof_idx is not None:
        sigma_noise   = float(np.std(xcorr_env[noise_mask]))
        snr_amplitude = float(xcorr_env[tof_idx]) / (sigma_noise + 1e-12)
        snr_db        = float(20 * np.log10(snr_amplitude + 1e-12))
        theo_sigma_tof_us = (fwhm_us / (3.33 * snr_amplitude)) if fwhm_us else None
    else:
        sigma_noise = snr_amplitude = snr_db = theo_sigma_tof_us = None

    # Only keep a ±15ms window around the TOF peak for plotting — not the full
    # 6M-sample autocorr array. Storing many × 6M arrays = GB in memory.
    plot_mask = (lags_us >= -15000) & (lags_us <= 15000)
    return dict(xcorr=xcorr[plot_mask], xcorr_env=xcorr_env[plot_mask],
                lags_us=lags_us[plot_mask],
                incident_us=None, tof_us=tof_us,
                fwhm_us=fwhm_us, lag0_fwhm_us=lag0_fwhm_us,
                ratio_db=ratio_db, sidelobe_db=None,
                snr_amplitude=snr_amplitude, snr_db=snr_db,
                theo_sigma_tof_us=theo_sigma_tof_us)


def theoretical_fwhm_us(f_start, f_stop):
    """Ideal matched-filter FWHM for flat-spectrum LFM: 0.886/B."""
    return 0.886 / abs(f_stop - f_start) * 1e6


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_chirp_result(result, config, output_path):
    f_start, f_stop, dur_ms, label, amp_shape = config[:5]
    mf = result['mf']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Transmitted chirp
    ax = axes[0, 0]
    t_ms = np.arange(len(result['chirp'])) / DODOMIC_SAMPLE_RATE * 1000
    ax.plot(t_ms, result['chirp'], 'b-', lw=0.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Transmitted Chirp: {label}')
    ax.grid(True, alpha=0.3)

    # Received signal (first 50 ms) — notch filtered at 25 kHz to remove background interference
    ax = axes[0, 1]
    n50 = int(0.05 * DODOMIC_SAMPLE_RATE)
    fs_plot = DODOMIC_SAMPLE_RATE
    nyq = fs_plot / 2
    sos_notch = butter(4, [23500/nyq, 26500/nyq], btype='bandstop', output='sos')
    from scipy.signal import sosfiltfilt as _sff
    rx_plot = _sff(sos_notch, result['received'][:n50])
    t_rx = np.arange(n50) / fs_plot * 1000
    ax.plot(t_rx, rx_plot, 'b-', lw=0.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Received Signal (first 50 ms, notch 25 kHz)')
    ax.grid(True, alpha=0.3)

    # Autocorrelation of received signal — TOF peak at lag = round-trip travel time
    ax = axes[1, 0]
    mask = (mf['lags_us'] >= 0) & (mf['lags_us'] <= 15000)
    ax.plot(mf['lags_us'][mask] / 1000, mf['xcorr'][mask],
            'b-', lw=0.5, alpha=0.4, label='Autocorr')
    ax.plot(mf['lags_us'][mask] / 1000, mf['xcorr_env'][mask],
            'r-', lw=1.5, label='Envelope')
    if mf['tof_us']:
        ax.axvline(mf['tof_us'] / 1000, color='g', lw=1.5, ls='--',
                   label=f"TOF = {mf['tof_us']:.0f} μs")
    ax.set_xlabel('Lag (ms)')
    ax.set_ylabel('Normalised autocorrelation')
    ax.set_title('Received Signal Autocorrelation (HP 30 kHz)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Info panel
    ax = axes[1, 1]
    ax.axis('off')
    theo = theoretical_fwhm_us(f_start, f_stop)
    fwhm_s    = f"{mf['fwhm_us']:.0f} μs"    if mf['fwhm_us']    else 'N/A'
    tof_s     = f"{mf['tof_us']:.0f} μs"     if mf['tof_us']     else 'N/A'
    sl_s      = f"{mf['sidelobe_db']:.1f} dB" if mf['sidelobe_db'] else 'N/A'
    win_label = config[5] if len(config) > 5 else 'tukey'
    tbp       = abs(f_stop - f_start) * dur_ms / 1000
    lag0_s = f"{mf['lag0_fwhm_us']:.0f} μs" if mf.get('lag0_fwhm_us') else 'N/A'
    rat_s  = f"{mf['ratio_db']:.1f} dB"    if mf.get('ratio_db')    else 'N/A'
    info = (
        f"CHIRP PARAMETERS\n{'='*32}\n"
        f"  f_start:    {f_start/1000:.2f} kHz\n"
        f"  f_stop:     {f_stop/1000:.2f} kHz\n"
        f"  Bandwidth:  {abs(f_stop-f_start)/1000:.2f} kHz\n"
        f"  Duration:   {dur_ms} ms\n"
        f"  TBP:        {tbp:.0f}\n"
        f"  Amplitude:  {amp_shape}\n"
        f"  Window:     {win_label}\n\n"
        f"THEORETICAL (1/BW)\n"
        f"  FWHM:       {theo:.0f} μs\n\n"
        f"MEASURED (autocorr)\n"
        f"  TOF FWHM:   {fwhm_s}\n"
        f"  Lag-0 FWHM: {lag0_s}\n"
        f"  TOF:        {tof_s}\n"
        f"  Ratio:      {rat_s}"
    )
    ax.text(0.05, 0.95, info, transform=ax.transAxes, fontsize=10,
            va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(f'LFM Chirp Autocorrelation — {label}', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_sweep_summary(results, output_path):
    flat_10 = [r for r in results if r['amp_shape'] == 'flat'
               and r['duration_ms'] == 10]
    flat_10.sort(key=lambda r: r['bw_hz'])

    dur_sweep = [r for r in results if r['amp_shape'] == 'flat'
                 and abs(r['bw_hz'] - 3750) < 300]
    dur_sweep.sort(key=lambda r: r['duration_ms'])

    comp_10 = [r for r in results if r['amp_shape'] == 'piezo_compensated'
               and r['duration_ms'] == 10]
    comp_10.sort(key=lambda r: r['bw_hz'])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Panel 1: FWHM vs bandwidth (10 ms)
    ax = axes[0, 0]
    bws   = [r['bw_hz'] / 1000 for r in flat_10]
    theo  = [r['theo_fwhm_us'] for r in flat_10]
    meas  = [r['mf']['fwhm_us'] or 0 for r in flat_10]
    ax.plot(bws, theo, 'b^--', lw=1.5, ms=6, label='Theoretical 0.886/B')
    ax.plot(bws, meas, 'bo-',  lw=2,   ms=8, label='Measured (flat)')
    if comp_10:
        comp_bws  = [r['bw_hz'] / 1000 for r in comp_10]
        comp_meas = [r['mf']['fwhm_us'] or 0 for r in comp_10]
        ax.plot(comp_bws, comp_meas, 'rs--', lw=1.5, ms=8, label='Measured (compensated)')
    ax.axvline(1.7, color='orange', ls=':', alpha=0.7, label='−3 dB BW')
    ax.axvline(3.75, color='red',   ls=':', alpha=0.7, label='−6 dB BW')
    ax.set_xlabel('Chirp Bandwidth (kHz)')
    ax.set_ylabel('FWHM (μs)')
    ax.set_title('FWHM vs. Bandwidth (10 ms chirps)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: FWHM vs duration (3.75 kHz)
    ax = axes[0, 1]
    if len(dur_sweep) > 1:
        durs      = [r['duration_ms'] for r in dur_sweep]
        meas_d    = [r['mf']['fwhm_us'] or 0 for r in dur_sweep]
        theo_val  = dur_sweep[0]['theo_fwhm_us']
        ax.axhline(theo_val, color='b', ls='--', lw=1.5,
                   label=f'Theoretical ({theo_val:.0f} μs)')
        ax.plot(durs, meas_d, 'bo-', lw=2, ms=8, label='Measured')
        ax.set_xlabel('Duration (ms)')
        ax.set_ylabel('FWHM (μs)')
        ax.set_title('FWHM vs. Duration (3.75 kHz chirp)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Panel 3: TOF consistency
    ax = axes[1, 0]
    tof_data = [(r['label'], r['mf']['tof_us'])
                for r in results if r['mf']['tof_us']]
    if tof_data:
        lbls, tofs = zip(*tof_data)
        ax.bar(range(len(tofs)), tofs, color='steelblue', alpha=0.7)
        ax.set_xticks(range(len(lbls)))
        ax.set_xticklabels(lbls, rotation=45, ha='right', fontsize=7)
        ax.axhline(np.mean(tofs), color='r', ls='--',
                   label=f'Mean {np.mean(tofs):.0f} μs')
        ax.set_ylabel('TOF (μs)')
        ax.set_title('TOF Consistency Across Configurations')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    # Panel 4: MF envelope overlay near TOF peak (10 ms flat)
    ax = axes[1, 1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(flat_10)))
    for i, r in enumerate(flat_10):
        mf = r['mf']
        if mf['tof_us'] is None:
            continue
        centre = mf['tof_us']
        mask   = (mf['lags_us'] >= centre - 2000) & (mf['lags_us'] <= centre + 2000)
        if np.any(mask):
            ax.plot(mf['lags_us'][mask] - centre, mf['xcorr_env'][mask],
                    color=colors[i], lw=1.5,
                    label=f"{r['bw_hz']/1000:.1f} kHz")
    ax.axhline(0.5, color='gray', ls='--', alpha=0.5, label='FWHM level')
    ax.set_xlabel('Lag relative to TOF peak (μs)')
    ax.set_ylabel('Normalised envelope')
    ax.set_title('MF Envelope at TOF Peak (10 ms flat)')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle('LFM Chirp Sweep Summary', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='LFM Chirp Sweep')
    parser.add_argument('--ppp-mode', choices=['none', 'A', 'B'], default='none',
                        help='Period-by-period recording: none=30s continuous (default), '
                             'A=stream-open loop, B=stream-restart per period')
    parser.add_argument('--ppp-n', type=int, default=100,
                        help='Number of periods to record in ppp mode (default 100)')
    parser.add_argument('--ppp-settle', type=float, default=2.0,
                        help='Settle time (s) after opening stream in ppp mode (default 2.0)')
    args = parser.parse_args()

    print('=' * 60)
    print(f'LFM Chirp Sweep ({VERSION})')
    if args.ppp_mode != 'none':
        print(f'  Period-by-period mode {args.ppp_mode}: {args.ppp_n} periods, settle={args.ppp_settle}s')
    print('=' * 60)

    piezo_response = load_piezo_response()
    if piezo_response is None:
        print("Amplitude compensation unavailable — skipping 'piezo_compensated' configs.")

    # Archiver setup
    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(OUTPUT_DIR, 'recordings', f'{ts}_chirp_sweep_{VERSION}_newHf')
    raw_dir = os.path.join(run_dir, 'raw')
    plt_dir = os.path.join(run_dir, 'plots')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(plt_dir, exist_ok=True)

    manifest = {
        'timestamp':   datetime.now().isoformat(),
        'version':     VERSION,
        'description': 'lfm_chirp_sweep',
        'hardware': {
            'sdg':                f'SDG1062X @ {SDG_IP}',
            'sdg_sample_rate_hz': SDG_SAMPLE_RATE,
            'mic':                'Dodotronic UltraMic',
            'mic_sample_rate_hz': DODOMIC_SAMPLE_RATE,
        },
        'measurements': {},
    }

    pa, mic_index = find_ultramic()
    if pa is None:
        return 1

    results = []
    sdg = None
    try:
        sdg = SDG1062X_LAN()
        sdg.connect()
        print(f'Device: {sdg.query("*IDN?")}')
        sdg.set_arb_mode(channel=1)

        for cfg in CHIRP_CONFIGS:
            f_start, f_stop, dur_ms, label, amp_shape = cfg[:5]
            win      = cfg[5] if len(cfg) > 5 else 'tukey'
            ampl_vpp = cfg[6] if len(cfg) > 6 else 2.0
            dead_ms  = cfg[7] if len(cfg) > 7 else 15.0

            if amp_shape == 'piezo_compensated' and piezo_response is None:
                print(f'\nSkipping {label} (no piezo response data)')
                continue

            bw_hz = abs(f_stop - f_start)
            print(f"\n{'='*55}")
            print(f"  {label}")
            print(f"  {f_start/1000:.2f}–{f_stop/1000:.2f} kHz  |  "
                  f"BW={bw_hz/1000:.2f} kHz  |  {dur_ms} ms  |  {amp_shape}  |  {ampl_vpp:.2f} Vpp")
            print(f"  Theoretical FWHM: {theoretical_fwhm_us(f_start, f_stop):.0f} μs  |  "
                  f"TBP: {bw_hz * dur_ms / 1000:.0f}")

            # Generate chirp at mic sample rate (used as matched-filter template)
            _, chirp, _ = generate_lfm_chirp(
                f_start, f_stop, dur_ms, DODOMIC_SAMPLE_RATE,
                amp_shape, piezo_response, window=win)

            # Build SDG buffer (resampled, zero-padded)
            sdg_buf, rep_rate = create_sdg_buffer(chirp, dur_ms, dead_ms=dead_ms)
            dead_ms = len(sdg_buf) / SDG_SAMPLE_RATE * 1000 - dur_ms
            print(f"  SDG: {len(sdg_buf)} samples @ {rep_rate:.1f} Hz  "
                  f"({dur_ms} ms chirp + {dead_ms:.1f} ms dead time)")

            # Upload and record
            wf_name = 'ch_' + label.replace('.', 'p').replace('-', 'm')[:15]
            sdg.upload_waveform(wf_name, sdg_buf, rep_rate, ampl_vpp=ampl_vpp)
            sdg.output_on()
            time.sleep(0.3)
            if args.ppp_mode == 'none':
                received = record_audio(pa, mic_index, RECORD_DURATION)
            else:
                periods = record_period_by_period(
                    pa, mic_index, rep_rate, args.ppp_n,
                    mode=args.ppp_mode, settle_s=args.ppp_settle)
                # Save individual period WAVs
                ppp_dir = os.path.join(raw_dir, 'periods', safe)
                os.makedirs(ppp_dir, exist_ok=True)
                for pi, p in enumerate(periods):
                    wavfile.write(os.path.join(ppp_dir, f'{safe}_period_{pi:04d}.wav'),
                                  DODOMIC_SAMPLE_RATE, (p * 32767).astype(np.int16))
                # Concatenate for downstream analysis (same interface as continuous)
                received = periods.flatten()
            sdg.output_off()
            sdg.delete_waveform(wf_name)   # free SDG Flash storage slot

            # Autocorrelation analysis
            mf = matched_filter_analysis(received, chirp, f_start, f_stop)
            fwhm_s = f"{mf['fwhm_us']:.0f} μs" if mf['fwhm_us'] else 'not found'
            tof_s  = f"{mf['tof_us']:.0f} μs"  if mf['tof_us']  else 'not found'
            snr_s   = f"{mf['snr_db']:.1f} dB"          if mf.get('snr_db')            else 'N/A'
            theo_s  = f"{mf['theo_sigma_tof_us']:.2f} μs" if mf.get('theo_sigma_tof_us') else 'N/A'
            print(f"  TOF FWHM: {fwhm_s}    TOF: {tof_s}    SNR: {snr_s}    Theo σ_TOF: {theo_s}")

            # Per-period σ_TOF — the real temperature precision metric
            tof_mean, tof_std, n_pp, fit_sig = per_period_sigma_tof(received, rep_rate)
            if tof_std is not None:
                fit_s = f"  fit σ={fit_sig*1000:.0f} ns" if fit_sig else ""
                print(f"  σ_TOF:    {tof_std:.2f} μs / ping  →  {tof_std/10.3:.3f}°C/ping  (N={n_pp}){fit_s}")
            else:
                print(f"  σ_TOF:    not enough periods (N={n_pp})")

            # Save WAV first (full 25s), then discard the large array from memory.
            safe = label.replace('.', 'p')
            wav_path = os.path.join(raw_dir, f'{safe}_received.wav')
            wavfile.write(wav_path, DODOMIC_SAMPLE_RATE,
                          (received * 32767).astype(np.int16))

            # Store only 50ms of received for the plot — full WAV is on disk.
            # Keeping 25s × 22 configs = 1.6 GB in memory causes thrashing.
            n50ms = int(0.05 * DODOMIC_SAMPLE_RATE)
            result = dict(
                f_start=f_start, f_stop=f_stop, bw_hz=bw_hz,
                duration_ms=dur_ms, label=label, amp_shape=amp_shape,
                theo_fwhm_us=theoretical_fwhm_us(f_start, f_stop),
                chirp=chirp, received=received[:n50ms].copy(), mf=mf,
                sigma_tof_us=tof_std, n_periods=n_pp,
            )
            results.append(result)
            del received   # free the 25s array immediately

            # Individual plot
            plot_path = os.path.join(plt_dir, f'{safe}.png')
            plot_chirp_result(result, cfg, plot_path)

            manifest['measurements'][label] = {
                'f_start_hz':    f_start,
                'f_stop_hz':     f_stop,
                'bandwidth_hz':  bw_hz,
                'duration_ms':   dur_ms,
                'amp_shape':     amp_shape,
                'theo_fwhm_us':  theoretical_fwhm_us(f_start, f_stop),
                'meas_fwhm_us':       mf['fwhm_us'],
                'tof_us':             mf['tof_us'],
                'snr_db':             mf.get('snr_db'),
                'theo_sigma_tof_us':  mf.get('theo_sigma_tof_us'),
                'sigma_tof_us':       tof_std,
                'n_periods':          n_pp,
                'sidelobe_db':        mf['sidelobe_db'],
                'wav':           f'raw/{safe}_received.wav',
                'plot':          f'plots/{safe}.png',
            }
            time.sleep(0.5)

    except Exception as e:
        import traceback
        print(f'\nError: {e}')
        traceback.print_exc()
        return 1
    finally:
        if sdg:
            sdg.output_off()
            sdg.disconnect()
        if pa:
            pa.terminate()

    # Summary plot and manifest
    plot_sweep_summary(results, os.path.join(plt_dir, 'summary.png'))
    with open(os.path.join(run_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f'\nResults in: {run_dir}')

    # Console table
    print(f"\n{'='*155}")
    print(f"{'Label':<28} {'BW kHz':<8} {'ms':<5} {'Shape':<14} "
          f"{'Theo μs':<9} {'TOF μs':<9} {'SNR dB':<8} {'Theo σ μs':<11} {'σ_TOF μs':<10} {'°C/ping':<9} {'TOF FWHM':<10} {'Ratio dB'}")
    print('-' * 155)
    for r in results:
        mf      = r['mf']
        tof_s   = f"{mf['tof_us']:.1f}"              if mf['tof_us']                else 'N/A'
        fwhm_s  = f"{mf['fwhm_us']:.0f}"             if mf['fwhm_us']               else 'N/A'
        rat_s   = f"{mf['ratio_db']:.1f}"            if mf.get('ratio_db')          else 'N/A'
        snr_s   = f"{mf['snr_db']:.1f}"              if mf.get('snr_db')            else 'N/A'
        theo_s  = f"{mf['theo_sigma_tof_us']:.2f}"   if mf.get('theo_sigma_tof_us') else 'N/A'
        sig     = r.get('sigma_tof_us')
        sig_s   = f"{sig:.2f}"                       if sig is not None              else 'N/A'
        deg_s   = f"{sig/10.3:.3f}"                  if sig is not None              else 'N/A'
        print(f"{r['label']:<28} {r['bw_hz']/1000:<8.2f} {r['duration_ms']:<5} "
              f"{r['amp_shape']:<14} {r['theo_fwhm_us']:<9.0f} "
              f"{tof_s:<9} {snr_s:<8} {theo_s:<11} {sig_s:<10} {deg_s:<9} {fwhm_s:<10} {rat_s}")
    print('\nDone!')
    return 0


if __name__ == '__main__':
    exit(main())
