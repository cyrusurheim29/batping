#!/usr/bin/env python3
"""
Bayesian Optimizer Best-Waveform Analysis
  1. Load best params from LFM+Wiener and NLFM CSVs
  2. Plot both waveforms (time domain, spectrum, instantaneous freq)
  3. Transmit each, record 10s, save WAV
  4. Run Stubbs dual-template analysis on both recordings
"""

import sys, os, time, socket
import numpy as np
import pyaudio
from datetime import datetime
from scipy.signal import butter, sosfiltfilt, hilbert, resample
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import chirp_stubbs_analysis as stubbs

# ── Config ────────────────────────────────────────────────────────────────────
SDG_IP   = "10.11.13.230"
SDG_PORT = 5025
SDG_SR   = 400000
MIC_SR   = 384000
SDG_MAX  = 16384
AMPL     = 0.9
RECORD_DUR = 10.0
CHAR_FILE  = "stepped_sine_characterization.npz"

LFM_CSV  = "bayesian_opt_lfm_wiener_20260707_042302.csv"
NLFM_CSV = "bayesian_opt_nlfm_20260707_074557.csv"


# ── SDG ───────────────────────────────────────────────────────────────────────
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
        if self.sock: self.sock.close(); self.sock = None

    def write(self, cmd): self.sock.sendall((cmd + '\n').encode())
    def write_raw(self, data): self.sock.sendall(data)
    def read(self, n=4096): return self.sock.recv(n).decode().strip()
    def query(self, cmd): self.write(cmd); time.sleep(0.1); return self.read()

    def upload_waveform(self, name, data, freq_hz, ampl_vpp=2.0, ch=1):
        d = (np.clip(data, -1, 1) * 32767).astype(np.int16).astype('<i2').tobytes()
        cmd = f"C{ch}:WVDT WVNM,{name},FREQ,{freq_hz},AMPL,{ampl_vpp},OFST,0,PHASE,0,WAVEDATA,".encode()
        self.write_raw(cmd + d + b'\n'); time.sleep(0.5)
        self.write(f"C{ch}:ARWV NAME,{name}"); time.sleep(0.2)

    def set_arb(self, ch=1): self.write(f"C{ch}:BSWV WVTP,ARB"); time.sleep(0.2)
    def on(self, ch=1):  self.write(f"C{ch}:OUTP ON")
    def off(self, ch=1): self.write(f"C{ch}:OUTP OFF")


# ── Audio ─────────────────────────────────────────────────────────────────────
def find_ultramic(pa):
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if 'UltraMic' in info['name'] and info['maxInputChannels'] > 0:
            return i
    return None

def record_audio(pa, mic_idx, duration):
    n = int(duration * MIC_SR)
    chunk = 4096
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=MIC_SR,
                     input=True, input_device_index=mic_idx, frames_per_buffer=chunk)
    frames, rem = [], n
    while rem > 0:
        ch = min(chunk, rem)
        frames.append(stream.read(ch, exception_on_overflow=False))
        rem -= ch
    stream.stop_stream(); stream.close()
    return np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float64) / 32768.0


# ── Piezo ─────────────────────────────────────────────────────────────────────
def load_piezo():
    d = np.load(CHAR_FILE)
    freqs = d['freqs'].astype(float)
    mag_db = d['H_mag_smooth']
    mag_lin = 10.0 ** (mag_db / 20.0)
    mag_lin /= np.max(mag_lin)
    return freqs, mag_lin


# ── Waveform generators ───────────────────────────────────────────────────────
def gen_lfm_wiener(f_start, f_stop, dur_ms, lam, win_alpha, piezo_freqs, piezo_mag, sr=MIC_SR):
    dur_s = dur_ms / 1000.0
    n = int(dur_s * sr)
    t = np.arange(n) / sr
    f_inst = f_start + (f_stop - f_start) / dur_s * t
    phase  = 2 * np.pi * (f_start * t + (f_stop - f_start) / (2 * dur_s) * t**2)
    h = np.clip(np.interp(f_inst, piezo_freqs, piezo_mag), 1e-6, None)
    amp = h / (h**2 + lam)
    amp = amp / np.max(amp) * AMPL
    chirp = amp * np.sin(phase)
    if win_alpha > 0:
        from scipy.signal import windows
        chirp *= windows.tukey(n, alpha=win_alpha)
    return chirp, f_inst, t

def gen_nlfm(f_start, f_stop, dur_ms, lam, win_alpha, piezo_freqs, piezo_mag, sr=MIC_SR):
    dur_s = dur_ms / 1000.0
    n = int(dur_s * sr)
    f_grid = np.linspace(f_start, f_stop, 10000)
    h_grid = np.clip(np.interp(f_grid, piezo_freqs, piezo_mag), 1e-6, None)
    rate_weight = h_grid**2 / (h_grid**2 + lam)
    df = f_grid[1] - f_grid[0]
    dt_df = 1.0 / (rate_weight + 1e-10)
    t_of_f = np.cumsum(dt_df) * df
    t_of_f = t_of_f / t_of_f[-1] * dur_s
    t_samples = np.linspace(0, dur_s, n)
    f_of_t = np.interp(t_samples, t_of_f, f_grid)
    phase = 2 * np.pi * np.cumsum(f_of_t) / sr
    chirp = AMPL * np.sin(phase)
    if win_alpha > 0:
        from scipy.signal import windows
        chirp *= windows.tukey(n, alpha=win_alpha)
    return chirp, f_of_t, t_samples

def make_sdg_buffer(chirp, dur_ms):
    dead_ms = 15.0
    buf_samples = min(int((dur_ms + dead_ms) / 1000 * SDG_SR), SDG_MAX)
    n_sdg = int(dur_ms / 1000 * SDG_SR)
    chirp_sdg = resample(chirp, n_sdg)
    chirp_sdg = chirp_sdg / (np.max(np.abs(chirp_sdg)) + 1e-10) * AMPL
    buf = np.zeros(buf_samples)
    buf[:min(n_sdg, buf_samples)] = chirp_sdg[:min(n_sdg, buf_samples)]
    return buf, SDG_SR / buf_samples


# ── Load best params from CSVs ────────────────────────────────────────────────
def load_best(csv_path):
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    # Filter out penalty values (>100)
    valid = data[data[:, 6] < 100]
    best_idx = np.argmin(valid[:, 6])
    row = valid[best_idx]
    return {
        'eval': int(row[0]),
        'f_start': row[1], 'f_stop': row[2],
        'duration_ms': row[3], 'log10_lambda': row[4],
        'window_alpha': row[5], 'sigma_tof_us': row[6],
        'lambda': 10**row[4],
    }


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_waveform(chirp, f_inst, t_chirp, label, params, piezo_freqs, piezo_mag, ts):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f'{label} — Best Waveform\n'
                 f'f={params["f_start"]/1000:.1f}–{params["f_stop"]/1000:.1f}kHz  '
                 f'T={params["duration_ms"]:.2f}ms  λ={params["lambda"]:.2e}  '
                 f'α={params["window_alpha"]:.2f}  '
                 f'σ_TOF={params["sigma_tof_us"]:.3f}μs → {params["sigma_tof_us"]/10.3:.4f}°C/ping',
                 fontsize=10, fontweight='bold')

    ax = axes[0, 0]
    ax.plot(t_chirp * 1000, chirp, lw=0.8)
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Amplitude')
    ax.set_title('Waveform (time domain)'); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t_chirp * 1000, f_inst / 1000, lw=1.2, color='C1')
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Frequency (kHz)')
    ax.set_title('Instantaneous frequency'); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    N = len(chirp)
    freqs_fft = np.fft.rfftfreq(N, 1/MIC_SR)
    spec = np.abs(np.fft.rfft(chirp))
    spec_db = 20 * np.log10(spec / np.max(spec) + 1e-10)
    ax.plot(freqs_fft / 1000, spec_db, lw=0.8, label='Transmitted')
    # overlay piezo response
    ax2 = ax.twinx()
    ax2.plot(piezo_freqs / 1000, 20 * np.log10(piezo_mag + 1e-10),
             color='C2', lw=1, alpha=0.6, linestyle='--', label='H(f)')
    ax2.set_ylabel('H(f) (dB)', color='C2'); ax2.tick_params(axis='y', colors='C2')
    ax.set_xlabel('Frequency (kHz)'); ax.set_ylabel('Amplitude (dB)')
    ax.set_title('Spectrum vs piezo H(f)')
    ax.set_xlim(25, 65); ax.set_ylim(-60, 5); ax.grid(True, alpha=0.3)
    ax.axvline(params['f_start']/1000, color='r', lw=0.8, linestyle=':')
    ax.axvline(params['f_stop']/1000, color='r', lw=0.8, linestyle=':')

    ax = axes[1, 1]
    envelope = np.abs(hilbert(chirp))
    ax.plot(t_chirp * 1000, envelope, lw=1, color='C3', label='Envelope')
    ax.plot(t_chirp * 1000, chirp, lw=0.4, alpha=0.4, color='C0')
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Amplitude')
    ax.set_title('Envelope'); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f'best_waveform_{label.lower().replace("+","_").replace(" ","_")}_{ts}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")
    return fname


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    piezo_freqs, piezo_mag = load_piezo()
    print(f"Loaded piezo: {len(piezo_freqs)} pts")

    # Load best params
    lfm_p = load_best(LFM_CSV)
    nlfm_p = load_best(NLFM_CSV)

    print(f"\nLFM+Wiener best (eval {lfm_p['eval']}):")
    print(f"  f={lfm_p['f_start']/1000:.1f}–{lfm_p['f_stop']/1000:.1f}kHz  "
          f"T={lfm_p['duration_ms']:.2f}ms  λ={lfm_p['lambda']:.2e}  "
          f"α={lfm_p['window_alpha']:.2f}  σ={lfm_p['sigma_tof_us']:.3f}μs")
    print(f"\nNLFM best (eval {nlfm_p['eval']}):")
    print(f"  f={nlfm_p['f_start']/1000:.1f}–{nlfm_p['f_stop']/1000:.1f}kHz  "
          f"T={nlfm_p['duration_ms']:.2f}ms  λ={nlfm_p['lambda']:.2e}  "
          f"α={nlfm_p['window_alpha']:.2f}  σ={nlfm_p['sigma_tof_us']:.3f}μs")

    # Generate waveforms
    lfm_chirp, lfm_finst, lfm_t = gen_lfm_wiener(
        lfm_p['f_start'], lfm_p['f_stop'], lfm_p['duration_ms'],
        lfm_p['lambda'], lfm_p['window_alpha'], piezo_freqs, piezo_mag)

    nlfm_chirp, nlfm_finst, nlfm_t = gen_nlfm(
        nlfm_p['f_start'], nlfm_p['f_stop'], nlfm_p['duration_ms'],
        nlfm_p['lambda'], nlfm_p['window_alpha'], piezo_freqs, piezo_mag)

    # Plot waveforms
    print("\nGenerating waveform plots...")
    plot_waveform(lfm_chirp, lfm_finst, lfm_t, 'LFM+Wiener', lfm_p, piezo_freqs, piezo_mag, ts)
    plot_waveform(nlfm_chirp, nlfm_finst, nlfm_t, 'NLFM', nlfm_p, piezo_freqs, piezo_mag, ts)

    # Make SDG buffers
    lfm_buf, lfm_rr = make_sdg_buffer(lfm_chirp, lfm_p['duration_ms'])
    nlfm_buf, nlfm_rr = make_sdg_buffer(nlfm_chirp, nlfm_p['duration_ms'])
    print(f"\nLFM+Wiener buffer: {len(lfm_buf)} samples @ {lfm_rr:.2f} Hz rep")
    print(f"NLFM buffer:       {len(nlfm_buf)} samples @ {nlfm_rr:.2f} Hz rep")

    # Connect hardware
    sdg = SDG1062X_LAN()
    sdg.connect()
    try:
        print(f"  SDG: {sdg.query('*IDN?')}")
    except Exception:
        print("  SDG: connected (IDN skipped)")
    sdg.set_arb()

    pa = pyaudio.PyAudio()
    mic_idx = find_ultramic(pa)
    if mic_idx is None:
        print("UltraMic not found!"); pa.terminate(); return 1
    print(f"  UltraMic at index {mic_idx}")

    wav_paths = {}

    for label, chirp, buf, rr, params in [
        ('LFM+Wiener', lfm_chirp, lfm_buf, lfm_rr, lfm_p),
        ('NLFM',       nlfm_chirp, nlfm_buf, nlfm_rr, nlfm_p),
    ]:
        print(f"\n{'='*60}")
        print(f"  Recording {label} — {RECORD_DUR}s @ {rr:.1f} Hz")
        print(f"{'='*60}")

        sdg.upload_waveform('BEST', buf, rr, ampl_vpp=2.0)
        sdg.on()
        time.sleep(0.5)
        audio = record_audio(pa, mic_idx, RECORD_DUR)
        sdg.off()
        time.sleep(0.3)

        tag = label.lower().replace('+', '_').replace(' ', '_')
        wav_path = f'best_{tag}_{ts}.wav'
        wavfile.write(wav_path, MIC_SR, (audio * 32767).astype(np.int16))
        print(f"  Saved WAV: {wav_path}")
        wav_paths[label] = (wav_path, rr, params)

    sdg.disconnect()
    pa.terminate()

    # Stubbs analysis on both recordings
    print(f"\n{'='*60}")
    print("  Running Stubbs dual-template analysis...")
    print(f"{'='*60}")

    results = {}
    for label, (wav_path, rr, params) in wav_paths.items():
        print(f"\n--- {label} ---")
        try:
            r = stubbs.analyze(wav_path, rep_rate=rr)
            results[label] = r
        except Exception as e:
            print(f"  Stubbs analysis failed: {e}")
            results[label] = None

    # Summary plot — autocorrelations + Stubbs TOF scatter side by side
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f'Best Waveform Comparison — Bayesian Optimizer Results\n{ts}',
                 fontsize=11, fontweight='bold')

    for row, (label, (wav_path, rr, params)) in enumerate(wav_paths.items()):
        # Load recording for autocorr plot
        _, raw = wavfile.read(wav_path)
        audio = raw.astype(np.float64) / 32768.0
        from scipy.signal import butter, sosfiltfilt
        sos = butter(8, 30000 / (MIC_SR / 2), btype='high', output='sos')
        sig = sosfiltfilt(sos, audio[:int(3.0 * MIC_SR)])
        period_n = int(MIC_SR / rr)
        seg = sig[:period_n]
        from scipy.signal import correlate
        ac = correlate(seg, seg, mode='full')
        ac /= np.max(np.abs(ac))
        lags_us = (np.arange(len(ac)) - len(ac)//2) / MIC_SR * 1e6
        env = np.abs(hilbert(ac))

        ax = axes[row, 0]
        mask = (lags_us >= 4000) & (lags_us <= 8000)
        ax.plot(lags_us[mask], env[mask], lw=1.2)
        ax.set_xlabel('Lag (μs)'); ax.set_ylabel('Autocorr envelope')
        ax.set_title(f'{label} — Single-period autocorr')
        ax.grid(True, alpha=0.3)

        ax = axes[row, 1]
        # Instantaneous frequency
        if label == 'LFM+Wiener':
            f_inst = lfm_finst; t_c = lfm_t
        else:
            f_inst = nlfm_finst; t_c = nlfm_t
        ax.plot(t_c * 1000, f_inst / 1000, lw=1.5, color=f'C{row}')
        ax.set_xlabel('Time (ms)'); ax.set_ylabel('Freq (kHz)')
        ax.set_title(f'{label} — Instantaneous freq\n'
                     f'BW={abs(params["f_stop"]-params["f_start"])/1000:.1f}kHz  '
                     f'T={params["duration_ms"]:.2f}ms  λ={params["lambda"]:.2e}')
        ax.grid(True, alpha=0.3)

        ax = axes[row, 2]
        r = results.get(label)
        if r and r.get('delta_tau') is not None:
            dt = np.array(r['delta_tau'])
            ax.plot(dt, '.', ms=2, alpha=0.5)
            ax.axhline(np.mean(dt), color='r', lw=1, label=f'mean={np.mean(dt):.1f}μs')
            ax.set_xlabel('Period index'); ax.set_ylabel('Δτ (μs)')
            sigma = r.get('std_delta_tau_us', np.std(dt))
            ax.set_title(f'{label} — Stubbs Δτ\nσ={sigma:.3f}μs → {sigma/10.3:.4f}°C/ping')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Stubbs N/A', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f'{label} — Stubbs Δτ')

    plt.tight_layout()
    summary_fname = f'best_waveform_summary_{ts}.png'
    plt.savefig(summary_fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved summary: {summary_fname}")

    # Print final results
    print(f"\n{'='*60}")
    print("  FINAL RESULTS")
    print(f"{'='*60}")
    for label, r in results.items():
        params = wav_paths[label][2]
        print(f"\n{label}:")
        print(f"  Optimizer σ_TOF:  {params['sigma_tof_us']:.3f}μs → {params['sigma_tof_us']/10.3:.4f}°C/ping")
        if r and r.get('std_delta_tau_us') is not None:
            s = r['std_delta_tau_us']
            print(f"  Stubbs σ_Δτ:      {s:.3f}μs → {s/10.3:.4f}°C/ping")
        else:
            print(f"  Stubbs σ_Δτ:      N/A")

    return 0

if __name__ == '__main__':
    sys.exit(main())
