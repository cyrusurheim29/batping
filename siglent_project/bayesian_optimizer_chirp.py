#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bayesian Chirp Waveform Optimizer

Two sequential optimizations:
  1. LFM + Wiener amplitude  (--mode lfm_wiener)
  2. NLFM + constant amplitude (--mode nlfm)

Objective: theoretical σ_TOF = FWHM_tof / (3.33 × SNR_amplitude)
  - FWHM_tof: FWHM of TOF autocorr peak (μs)
  - SNR_amplitude: TOF peak / noise-floor RMS at 15–20ms lag
  - Single-recording estimate — stable, no 1/√N error, captures bandwidth + SNR jointly

Both modes share the same 5 free parameters:
  f_start [26500, 44000] Hz
  f_stop  [40000, 63000] Hz
  duration_ms [0.5, 4.5]
  log10_lambda [-4, 2]   (Wiener regularisation or NLFM nonlinearity)
  window_alpha [0, 0.5]  (Tukey taper)

Usage:
    python bayesian_optimizer_chirp.py               # run both sequentially
    python bayesian_optimizer_chirp.py --mode lfm_wiener
    python bayesian_optimizer_chirp.py --mode nlfm
    python bayesian_optimizer_chirp.py --dry-run      # offline test, no hardware

@author: cyrusurheimog
Created: July 2026
"""

import sys, argparse, time, socket
import numpy as np
import pyaudio
from datetime import datetime
from scipy.signal import butter, sosfiltfilt, hilbert, correlate, resample
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real


# ── Hardware ──────────────────────────────────────────────────────────────────
SDG_IP   = "10.11.13.230"
SDG_PORT = 5025
SDG_SR   = 400000
MIC_SR   = 384000
SDG_MAX  = 16384
CHUNK    = 4096
AMPL     = 0.9

# ── Optimization ──────────────────────────────────────────────────────────────
N_CALLS_RANDOM = 150   # random exploration before GP takes over
N_CALLS_TOTAL  = 900   # LFM+Wiener default; NLFM uses 800 (see main())
RECORD_DUR     = 5.0      # s per evaluation — ~300 periods

# ── Analysis ──────────────────────────────────────────────────────────────────
TOF_MIN_US   = 5000.0
TOF_MAX_US   = 6500.0
NOISE_LO_US  = 15000.0
NOISE_HI_US  = 20000.0
ANALYSIS_CLIP = 3.0       # s clip for live analysis (speed, memory)

# ── Characterisation file ─────────────────────────────────────────────────────
CHAR_FILE = "stepped_sine_characterization.npz"


# ── SDG class (from bayesian_optimizer_sdg.py) ────────────────────────────────

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

    def query(self, cmd):
        self.write(cmd); time.sleep(0.1); return self.read()

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
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=MIC_SR,
                     input=True, input_device_index=mic_idx, frames_per_buffer=CHUNK)
    frames, rem = [], n
    while rem > 0:
        ch = min(CHUNK, rem)
        frames.append(stream.read(ch, exception_on_overflow=False))
        rem -= ch
    stream.stop_stream(); stream.close()
    return np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float64) / 32768.0


# ── Piezo response loader ─────────────────────────────────────────────────────

def load_piezo(path=CHAR_FILE):
    d = np.load(path)
    freqs = d['freqs'].astype(float)
    mag_db = d['H_mag_smooth']
    mag_lin = 10.0 ** (mag_db / 20.0)
    mag_lin /= np.max(mag_lin)
    return freqs, mag_lin


# ── Waveform generation ───────────────────────────────────────────────────────

def gen_lfm_wiener(f_start, f_stop, dur_ms, lam, win_alpha,
                   piezo_freqs, piezo_mag, sr=MIC_SR):
    """LFM chirp with Wiener amplitude shaping.
    A(f) ∝ |H(f)| / (|H(f)|² + λ),  normalised to peak = AMPL."""
    dur_s = dur_ms / 1000.0
    n = int(dur_s * sr)
    t = np.arange(n) / sr
    f_inst = f_start + (f_stop - f_start) / dur_s * t
    phase  = 2 * np.pi * (f_start * t + (f_stop - f_start) / (2 * dur_s) * t**2)

    h = np.interp(f_inst, piezo_freqs, piezo_mag)
    h = np.clip(h, 1e-6, None)
    amp = h / (h**2 + lam)
    amp = amp / np.max(amp) * AMPL

    chirp = amp * np.sin(phase)
    if win_alpha > 0:
        from scipy.signal import windows
        chirp *= windows.tukey(n, alpha=win_alpha)
    return chirp


def gen_nlfm(f_start, f_stop, dur_ms, lam, win_alpha,
             piezo_freqs, piezo_mag, sr=MIC_SR):
    """NLFM with constant amplitude.
    Sweep rate df/dt ∝ |H(f)|² / (|H(f)|² + λ),
    giving flat(ish) received spectrum without amplitude modulation."""
    dur_s = dur_ms / 1000.0
    n = int(dur_s * sr)

    # Build frequency grid over [f_start, f_stop]
    f_grid = np.linspace(f_start, f_stop, 10000)
    h_grid = np.interp(f_grid, piezo_freqs, piezo_mag)
    h_grid = np.clip(h_grid, 1e-6, None)

    # Sweep rate weight: proportional to |H|² / (|H|² + λ)
    # High weight → fast sweep (less time, less energy)
    # Low weight → slow sweep (more time, more energy) — corrects attenuation
    rate_weight = h_grid**2 / (h_grid**2 + lam)

    # Integrate: dt/df = C / rate_weight  →  t(f) by cumsum
    df = f_grid[1] - f_grid[0]
    dt_df = 1.0 / (rate_weight + 1e-10)
    t_of_f = np.cumsum(dt_df) * df
    t_of_f = t_of_f / t_of_f[-1] * dur_s      # normalise to [0, dur_s]

    # Sample f(t) by inverting t(f)
    t_samples = np.linspace(0, dur_s, n)
    f_of_t = np.interp(t_samples, t_of_f, f_grid)

    # Integrate f(t) to get phase
    phase = 2 * np.pi * np.cumsum(f_of_t) / sr

    chirp = AMPL * np.sin(phase)
    if win_alpha > 0:
        from scipy.signal import windows
        chirp *= windows.tukey(n, alpha=win_alpha)
    return chirp


def make_sdg_buffer(chirp, dur_ms):
    """Resample to SDG rate, embed in buffer with 15ms dead time."""
    dead_ms = 15.0
    total_ms = dur_ms + dead_ms
    buf_samples = min(int(total_ms / 1000 * SDG_SR), SDG_MAX)
    n_sdg = int(dur_ms / 1000 * SDG_SR)
    chirp_sdg = resample(chirp, n_sdg)
    chirp_sdg = chirp_sdg / (np.max(np.abs(chirp_sdg)) + 1e-10) * AMPL
    buf = np.zeros(buf_samples)
    buf[:min(n_sdg, buf_samples)] = chirp_sdg[:min(n_sdg, buf_samples)]
    rep_rate = SDG_SR / buf_samples
    return buf, rep_rate


# ── Analysis ──────────────────────────────────────────────────────────────────

def hp_filter(sig, fs=MIC_SR):
    sos = butter(8, 30000 / (fs / 2), btype='high', output='sos')
    return sosfiltfilt(sos, sig)


def parabolic_peak(env, lags_us, lo, hi):
    mask = (lags_us >= lo) & (lags_us <= hi)
    if not np.any(mask): return None, None
    r, lr = env[mask], lags_us[mask]
    i = np.argmax(r)
    if i < 1 or i >= len(r) - 1: return float(lr[i]), float(r[i])
    ym, y0, yp = r[i-1], r[i], r[i+1]
    dn = 2*(ym - 2*y0 + yp)
    dl = np.clip((ym-yp)/dn, -0.5, 0.5) if abs(dn) > 1e-10 else 0.0
    return float(lr[i]) + dl*(lr[1]-lr[0]), float(y0 - (ym-yp)*dl/4)


def fwhm_at(env, lags_us, peak_us, peak_val):
    idx = int(np.argmin(np.abs(lags_us - peak_us)))
    half = peak_val * 0.5
    li, ri = idx, idx
    while li > 0 and env[li] > half: li -= 1
    while ri < len(env) - 1 and env[ri] > half: ri += 1
    return (ri - li) / MIC_SR * 1e6


def compute_sigma_tof(received, rep_rate, fs=MIC_SR):
    """
    Compute measured σ_TOF directly from per-period autocorrelation scatter.

    The theoretical formula FWHM/(3.33×SNR) is broken here because the
    noise window (15–20ms) is contaminated by the rep-rate autocorrelation peak
    (period = 17–19ms for our 52–59 Hz rep rates). This makes SNR unreliable
    and causes NEGATIVE correlation with the true σ_TOF.

    Instead: slice recording into periods, measure TOF from each via
    parabolic peak finding on the Hilbert envelope, return std over all periods.
    With ~300 periods in a 5s recording, uncertainty ≈ σ_TOF/√300 ≈ 6% — fine
    for a Bayesian optimizer that handles noise natively.

    Returns (sigma_tof_us, mean_tof_us, n_periods) or (None, None, 0).
    """
    clip = int(ANALYSIS_CLIP * fs)
    sig = hp_filter(received[:clip], fs)
    period_idx = int(fs / rep_rate)

    # Find T0 via envelope threshold
    env_n = np.abs(hilbert(sig)); env_n /= np.max(env_n)
    above = env_n[int(0.008*fs):] > 0.25
    first = np.argmax(above)
    T0 = max(0, int(0.008*fs) + first - int(0.008*fs)) if above[first] else 0

    tofs = []
    for i in range((len(sig) - T0) // period_idx - 1):
        seg = sig[T0 + i*period_idx : T0 + (i+1)*period_idx]
        from scipy.signal import correlate as sp_corr
        ac = sp_corr(seg, seg, mode='full')
        ctr = len(ac) // 2
        lu = (np.arange(len(ac)) - ctr) / fs * 1e6
        ac /= np.max(np.abs(ac))
        ev = np.abs(hilbert(ac)); ev /= np.max(ev)
        m = (lu >= TOF_MIN_US) & (lu <= TOF_MAX_US)
        if not np.any(m): continue
        ix = np.argmax(ev[m]); r = ev[m]; lg = lu[m]
        if ix < 1 or ix >= len(r)-1: tofs.append(lg[ix]); continue
        ym, y0, yp = r[ix-1], r[ix], r[ix+1]
        dn = 2*(ym-2*y0+yp)
        dl = np.clip((ym-yp)/dn, -0.5, 0.5) if abs(dn) > 1e-10 else 0.0
        tofs.append(float(lg[ix]) + dl*(lg[1]-lg[0]))

    if len(tofs) < 5:
        return None, None, 0
    tofs = np.array(tofs)
    return float(np.std(tofs)), float(np.mean(tofs)), len(tofs)


# ── Single evaluation ─────────────────────────────────────────────────────────

def evaluate_params(params, mode, rep_rate, sdg, pa, mic_idx, piezo_freqs, piezo_mag,
                    eval_count, dry_run=False):
    f_start, f_stop, dur_ms, log10_lam, win_alpha = params
    lam = 10.0 ** log10_lam

    # Enforce minimum BW
    if f_stop - f_start < 2000:
        print(f"  [{eval_count:4d}] BW too narrow ({(f_stop-f_start)/1000:.1f}kHz) → penalty")
        return 999.0

    # Generate waveform
    if mode == 'lfm_wiener':
        chirp = gen_lfm_wiener(f_start, f_stop, dur_ms, lam, win_alpha,
                               piezo_freqs, piezo_mag)
    else:
        chirp = gen_nlfm(f_start, f_stop, dur_ms, lam, win_alpha,
                         piezo_freqs, piezo_mag)

    buf, rep_rate = make_sdg_buffer(chirp, dur_ms)

    if dry_run:
        return float(np.random.uniform(0.1, 2.0))

    # Upload and record
    sdg.upload_waveform('CHIRPOPT', buf, rep_rate, ampl_vpp=2.0)
    sdg.on()
    time.sleep(0.3)
    received = record_audio(pa, mic_idx, RECORD_DUR)
    sdg.off()

    # Objective: measured σ_TOF from per-period autocorr scatter
    # (theo FWHM/SNR formula is broken — SNR window is contaminated by rep-rate peak)
    sigma_tof, mean_tof, n_p = compute_sigma_tof(received, rep_rate)

    if sigma_tof is None:
        print(f"  [{eval_count:4d}] No TOF peak found → penalty  "
              f"BW={abs(f_stop-f_start)/1000:.1f}kHz  T={dur_ms:.1f}ms  λ={lam:.2e}")
        return 999.0

    print(f"  [{eval_count:4d}] σ_TOF={sigma_tof:.3f}μs ({sigma_tof/10.3:.4f}°C)  "
          f"TOF={mean_tof:.0f}μs  N={n_p}  "
          f"BW={abs(f_stop-f_start)/1000:.1f}kHz  T={dur_ms:.1f}ms  "
          f"λ={lam:.2e}  α={win_alpha:.2f}")
    return float(sigma_tof)


# ── Optimisation run ──────────────────────────────────────────────────────────

def run_optimisation(mode, sdg, pa, mic_idx, piezo_freqs, piezo_mag,
                     n_calls=N_CALLS_TOTAL, n_random=N_CALLS_RANDOM,
                     dry_run=False):
    print(f"\n{'='*70}")
    print(f"  Mode: {mode.upper()}   n_calls={n_calls}  n_random={n_random}")
    print(f"{'='*70}")

    space = [
        Real(26500,  44000, name='f_start',       prior='uniform'),
        Real(40000,  63000, name='f_stop',         prior='uniform'),
        Real(0.5,    4.5,   name='duration_ms',    prior='uniform'),
        Real(-4.0,   2.0,   name='log10_lambda',   prior='uniform'),
        Real(0.0,    0.5,   name='window_alpha',   prior='uniform'),
    ]

    eval_count = [0]
    ts_start = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f'bayesian_opt_{mode}_{ts_start}.csv'
    with open(csv_path, 'w') as f:
        f.write('eval,f_start,f_stop,duration_ms,log10_lambda,window_alpha,sigma_tof_us\n')
    print(f"  Checkpointing every eval → {csv_path}")

    def objective(params):
        eval_count[0] += 1
        dur_ms = params[2]
        dead_ms = 15.0
        buf_samples = min(int((dur_ms + dead_ms) / 1000 * SDG_SR), SDG_MAX)
        rep_rate_eval = SDG_SR / buf_samples
        val = evaluate_params(params, mode, rep_rate_eval, sdg, pa, mic_idx,
                              piezo_freqs, piezo_mag, eval_count[0], dry_run)
        with open(csv_path, 'a') as f:
            f.write(f"{eval_count[0]},{params[0]:.2f},{params[1]:.2f},"
                    f"{params[2]:.4f},{params[3]:.6f},{params[4]:.4f},{val:.6f}\n")
        return val

    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        n_random_starts=n_random,
        acq_func='EI',
        xi=0.01,
        noise=1e-6,
        random_state=42,
        verbose=False,
    )

    print(f"\n  Best σ_TOF = {result.fun:.4f} μs  "
          f"({result.fun/10.3:.4f}°C/ping)")
    bp = result.x
    print(f"  Best params:")
    print(f"    f_start={bp[0]/1000:.2f}kHz  f_stop={bp[1]/1000:.2f}kHz  "
          f"BW={(bp[1]-bp[0])/1000:.2f}kHz")
    print(f"    duration={bp[2]:.2f}ms  λ={10**bp[3]:.3e}  window_α={bp[4]:.2f}")

    return result


# ── Plot helper ───────────────────────────────────────────────────────────────

def plot_results(result1, result2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for ax, res, label in [(ax1, result1, 'LFM + Wiener'), (ax2, result2, 'NLFM')]:
        if res is None: continue
        vals = [v for v in res.func_vals if v < 900]
        best = np.minimum.accumulate(vals)
        ax.plot(vals, 'b.', ms=3, alpha=0.4, label='evaluations')
        ax.plot(best, 'r-', lw=2, label=f'best: {res.fun:.3f}μs')
        ax.set_xlabel('Evaluation'); ax.set_ylabel('theo σ_TOF (μs)')
        ax.set_title(label); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle('Bayesian Chirp Optimisation', fontsize=11, fontweight='bold')
    plt.tight_layout()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'bayesian_opt_results_{ts}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved results plot.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Bayesian chirp waveform optimizer')
    parser.add_argument('--mode', choices=['lfm_wiener', 'nlfm', 'both'],
                        default='both', help='Which optimisation to run')
    parser.add_argument('--n-calls', type=int, default=N_CALLS_TOTAL)
    parser.add_argument('--n-random', type=int, default=N_CALLS_RANDOM)
    parser.add_argument('--dry-run', action='store_true',
                        help='Test offline without hardware')
    args = parser.parse_args()

    print(f"Loading piezo response from {CHAR_FILE}...")
    piezo_freqs, piezo_mag = load_piezo()
    print(f"  {len(piezo_freqs)} points, {piezo_freqs[0]/1000:.0f}–{piezo_freqs[-1]/1000:.0f} kHz")

    # Connect hardware
    sdg, pa, mic_idx = None, None, None
    if not args.dry_run:
        sdg = SDG1062X_LAN(); sdg.connect()
        try:
            print(f"  SDG: {sdg.query('*IDN?')}")
        except Exception:
            print("  SDG: connected (IDN query skipped)")
        sdg.set_arb()

        pa = pyaudio.PyAudio()
        mic_idx = find_ultramic(pa)
        if mic_idx is None:
            print("UltraMic not found!"); pa.terminate(); return 1
        print(f"  UltraMic at index {mic_idx}")
    else:
        print("  DRY RUN — no hardware")

    result_lfm, result_nlfm = None, None

    try:
        if args.mode in ('lfm_wiener', 'both'):
            result_lfm = run_optimisation(
                'lfm_wiener', sdg, pa, mic_idx, piezo_freqs, piezo_mag,
                n_calls=args.n_calls, n_random=args.n_random,
                dry_run=args.dry_run)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            np.savez(f'bayesian_opt_lfm_wiener_{ts}.npz',
                     best_params=result_lfm.x, best_theo_sigma=result_lfm.fun,
                     all_values=result_lfm.func_vals, all_params=result_lfm.x_iters)
            print(f"  Saved LFM+Wiener results.")

        if args.mode in ('nlfm', 'both'):
            result_nlfm = run_optimisation(
                'nlfm', sdg, pa, mic_idx, piezo_freqs, piezo_mag,
                n_calls=min(args.n_calls, 800), n_random=args.n_random,
                dry_run=args.dry_run)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            np.savez(f'bayesian_opt_nlfm_{ts}.npz',
                     best_params=result_nlfm.x, best_theo_sigma=result_nlfm.fun,
                     all_values=result_nlfm.func_vals, all_params=result_nlfm.x_iters)
            print(f"  Saved NLFM results.")

        if result_lfm and result_nlfm:
            print(f"\n{'='*70}")
            print(f"  LFM+Wiener best: {result_lfm.fun:.4f} μs → "
                  f"{result_lfm.fun/10.3:.4f}°C/ping")
            print(f"  NLFM       best: {result_nlfm.fun:.4f} μs → "
                  f"{result_nlfm.fun/10.3:.4f}°C/ping")
            print(f"{'='*70}")
            plot_results(result_lfm, result_nlfm)

    finally:
        if sdg:  sdg.off(); sdg.disconnect()
        if pa:   pa.terminate()

    return 0


if __name__ == '__main__':
    sys.exit(main())
