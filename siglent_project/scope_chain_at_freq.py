#!/usr/bin/env python3
"""Drive SDG at a single frequency EXACTLY as the characterize_chain pipeline
does, then capture + visualize Ch A + Ch B.  Overlays the characterization's
own analysis window so you can see what's being FFT'd.

Usage:
    DYLD_LIBRARY_PATH=/Library/Frameworks/PicoSDK.framework/Libraries/libps5000a \\
    python3 scope_chain_at_freq.py --freq 40350
    python3 scope_chain_at_freq.py --freq 40350 --tof-ms 23.3
"""
import os, sys, time, ctypes, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import hilbert

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ecm999_stepped_sine import SDG1062X_LAN, open_picoscope
from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import adc2mV

# ── Match the characterization pipeline exactly ──────────────────────────
BURST_MS           = 30.0
DEAD_MS            = 50.0
SETTLE_MS          = 100     # ms SDG settle after freq command
BURST_SETTLE_FRAC  = 0.5
CAPTURE_MS         = DEAD_MS + 2 * BURST_MS   # 110 ms
PIPELINE_TOF_MS    = 5.88    # the HARDCODED value in ecm999_stepped_sine.py (tuned for 2m!)
TIMEBASE           = 64      # Pico timebase → ~1.025 MHz

ap = argparse.ArgumentParser(description=__doc__.strip().split('\n')[0])
ap.add_argument('--freq',     type=float, required=True, help='Drive frequency (Hz)')
ap.add_argument('--ampl',     type=float, default=0.3,   help='SDG burst amplitude (Vpp)')
ap.add_argument('--tof-ms',   type=float, default=23.3,  help='Actual TOF of this geometry (for the "ideal window" overlay)')
ap.add_argument('--out-dir',  type=str,   default='.')
args = ap.parse_args()

print(f"Driving SDG at {args.freq/1e3:.3f} kHz, {args.ampl} Vpp, burst {BURST_MS:.0f} ms, dead {DEAD_MS:.0f} ms")

sdg = SDG1062X_LAN()
sdg.set_burst(args.freq, args.ampl, BURST_MS, DEAD_MS)
sdg.output_on()
time.sleep(SETTLE_MS / 1000.0)   # match characterization

chandle, actual_fs = open_picoscope()
print(f"  Pico fs = {actual_fs/1e6:.4f} MHz")

n_capture = int(round(actual_fs * CAPTURE_MS / 1000.0))
maxADC = ctypes.c_int16()
ps.ps5000aMaximumValue(chandle, ctypes.byref(maxADC))
trig_adc = int(0.1 * maxADC.value)   # 100 mV @ ±1V FS

chA = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
chB = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_B"]
ps.ps5000aSetSimpleTrigger(chandle, 1, chB, trig_adc,
                            ps.PS5000A_THRESHOLD_DIRECTION["PS5000A_RISING"], 0, 500)
ps.ps5000aRunBlock(chandle, 0, n_capture, TIMEBASE, None, 0, None, None)
ready = ctypes.c_int16(0)
while ready.value == 0:
    time.sleep(0.005)
    ps.ps5000aIsReady(chandle, ctypes.byref(ready))
bufA = (ctypes.c_int16 * n_capture)()
bufB = (ctypes.c_int16 * n_capture)()
ps.ps5000aSetDataBuffer(chandle, chA, ctypes.byref(bufA), n_capture, 0, 0)
ps.ps5000aSetDataBuffer(chandle, chB, ctypes.byref(bufB), n_capture, 0, 0)
overflow = ctypes.c_int16()
n_ret    = ctypes.c_uint32(n_capture)
ps.ps5000aGetValues(chandle, 0, ctypes.byref(n_ret), 1, 0, 0, ctypes.byref(overflow))
if overflow.value & 0x1: print("  WARN: Ch A overflow — reduce ampl")
if overflow.value & 0x2: print("  WARN: Ch B overflow — reduce ampl")

chA_mV = np.array(adc2mV(bufA, ps.PS5000A_RANGE["PS5000A_5V"], maxADC), dtype=np.float32)
chB_mV = np.array(adc2mV(bufB, ps.PS5000A_RANGE["PS5000A_1V"], maxADC), dtype=np.float32)
t_ms   = np.arange(n_capture) / actual_fs * 1000.0

sdg.burst_off(); sdg.output_off(); sdg.disconnect()
ps.ps5000aStop(chandle); ps.ps5000aCloseUnit(chandle)

# ── Analysis windows ─────────────────────────────────────────────────────
burst_samples = int(round(actual_fs * BURST_MS / 1000.0))
settle_samp   = int(BURST_SETTLE_FRAC * burst_samples)

# (a) What the characterization pipeline uses (TOF_MS = 5.88 hardcoded)
pipeline_tof_samp = int(round(actual_fs * PIPELINE_TOF_MS / 1000.0))
win_pipe_start = pipeline_tof_samp + settle_samp
win_pipe_end   = pipeline_tof_samp + burst_samples
win_pipe_t0_ms = win_pipe_start / actual_fs * 1000.0
win_pipe_t1_ms = win_pipe_end   / actual_fs * 1000.0

# (b) What it SHOULD use for your actual TOF
actual_tof_samp = int(round(actual_fs * args.tof_ms / 1000.0))
win_act_start = actual_tof_samp + settle_samp
win_act_end   = actual_tof_samp + burst_samples
win_act_t0_ms = win_act_start / actual_fs * 1000.0
win_act_t1_ms = win_act_end   / actual_fs * 1000.0

# ── Compute per-window FFT amplitude (match fft_amplitude_mV) ────────────
def fft_amp_mV(signal_mV, freq_hz, fs):
    N = len(signal_mV)
    if N < 8: return 0.0, -99.0
    win = np.hanning(N)
    S = np.fft.rfft(signal_mV * win)
    amps = 2.0 * np.abs(S) / (N * 0.5)
    k = max(1, min(int(round(freq_hz * N / fs)), len(amps)-2))
    amp = float(amps[k])
    mask = np.ones(len(amps), bool); mask[max(0,k-5):k+6] = False
    noise_rms = float(np.sqrt(np.mean(amps[mask]**2)))
    snr = 20*np.log10(amp / (noise_rms + 1e-12)) if amp > 0 else -99.0
    return amp, snr

def _amp_at_window(lo, hi):
    if hi > len(chA_mV) or lo < 0 or hi - lo < 10:
        return None, None
    return fft_amp_mV(chA_mV[lo:hi], args.freq, actual_fs)

amp_pipe, snr_pipe = _amp_at_window(win_pipe_start, win_pipe_end)
amp_act,  snr_act  = _amp_at_window(win_act_start,  win_act_end)

print(f"\nAnalysis windows (from Ch B trigger t=0):")
print(f"  (a) pipeline (TOF_MS={PIPELINE_TOF_MS} ms, hardcoded for 2m): [{win_pipe_t0_ms:.2f}, {win_pipe_t1_ms:.2f}] ms")
if amp_pipe is not None:
    print(f"      FFT amp @ drive freq = {amp_pipe:7.2f} mV   SNR = {snr_pipe:+.1f} dB")
print(f"  (b) correct for your geometry (TOF_MS={args.tof_ms} ms): [{win_act_t0_ms:.2f}, {win_act_t1_ms:.2f}] ms")
if amp_act is not None:
    print(f"      FFT amp @ drive freq = {amp_act:7.2f} mV   SNR = {snr_act:+.1f} dB")

# ── Plot ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(3, 1, height_ratios=[1, 2.2, 1.2])

# (1) Ch B drive
ax = fig.add_subplot(gs[0])
ax.plot(t_ms, chB_mV, color='#88aa44', lw=0.4)
ax.axvspan(win_pipe_t0_ms, win_pipe_t1_ms, color='orange', alpha=0.15,
           label=f'pipeline window [{win_pipe_t0_ms:.1f}, {win_pipe_t1_ms:.1f}] ms')
ax.axvspan(win_act_t0_ms, win_act_t1_ms, color='green', alpha=0.15,
           label=f'correct window [{win_act_t0_ms:.1f}, {win_act_t1_ms:.1f}] ms')
ax.set_ylabel('Ch B (mV)   SDG drive')
ax.set_title(f'Ch B electrical drive (±1V FS)')
ax.legend(fontsize=8, loc='upper right')
ax.grid(alpha=0.3)
ax.set_xlim(0, t_ms[-1])

# (2) Ch A with Hilbert envelope + both analysis windows
ax = fig.add_subplot(gs[1])
ch_a_env = np.abs(hilbert(chA_mV))
ax.plot(t_ms, chA_mV, color='#1f77b4', lw=0.4, alpha=0.7, label='Ch A waveform')
ax.plot(t_ms,  ch_a_env, color='k', lw=0.7, alpha=0.7, label='|envelope|')
ax.plot(t_ms, -ch_a_env, color='k', lw=0.7, alpha=0.7)
ax.axvspan(win_pipe_t0_ms, win_pipe_t1_ms, color='orange', alpha=0.15)
ax.axvspan(win_act_t0_ms,  win_act_t1_ms,  color='green',  alpha=0.15)
ax.axvline(args.tof_ms, color='g', ls=':', lw=0.8, alpha=0.5,
           label=f'actual TOF arrival {args.tof_ms:.1f} ms')
ax.set_ylabel('Ch A (mV)   acoustic return')
peak = float(np.max(np.abs(chA_mV)))
env_peak = float(np.max(ch_a_env))
pipe_str = f'{amp_pipe:.1f} mV (SNR {snr_pipe:+.1f} dB)' if amp_pipe is not None else 'N/A'
act_str  = f'{amp_act :.1f} mV (SNR {snr_act :+.1f} dB)' if amp_act  is not None else 'N/A'
ax.set_title(f'Ch A @ {args.freq/1e3:.3f} kHz   waveform peak={peak:.1f} mV, env-peak={env_peak:.1f} mV\n'
             f'FFT amp in PIPELINE window (orange): {pipe_str}   |   '
             f'FFT amp in CORRECT window (green):  {act_str}',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3)
ax.set_xlim(0, t_ms[-1])

# (3) Ch A FFT of the CORRECT window (zoom ±5 kHz around drive)
ax = fig.add_subplot(gs[2])
if win_act_end <= len(chA_mV):
    seg = chA_mV[win_act_start:win_act_end]
    N = len(seg); win = np.hanning(N)
    S = np.fft.rfft(seg * win)
    amps = 2.0 * np.abs(S) / (N * 0.5)
    freqs_fft = np.fft.rfftfreq(N, d=1.0/actual_fs)
    f_lo, f_hi = max(0, args.freq - 5000), args.freq + 5000
    mask = (freqs_fft >= f_lo) & (freqs_fft <= f_hi)
    ax.semilogy(freqs_fft[mask]/1e3, amps[mask], color='#4477cc', lw=0.8,
                label=f'Ch A FFT of CORRECT window (N={N})')
    ax.axvline(args.freq/1e3, color='#e6444d', ls='--', lw=0.8, alpha=0.6,
               label=f'drive freq {args.freq/1e3:.3f} kHz')
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Ch A spectrum (mV)')
    ax.set_title('Ch A FFT (zoom ±5 kHz around drive, from CORRECT window)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9)
else:
    ax.text(0.5, 0.5, 'correct window outside capture range', ha='center', va='center',
            transform=ax.transAxes)

fig.suptitle(f'Chain pickup @ {args.freq/1e3:.3f} kHz   —   '
             f'driven {args.ampl} Vpp, burst {BURST_MS:.0f} ms, dead {DEAD_MS:.0f} ms  '
             f'(mirrors characterization pipeline)',
             fontsize=12, fontweight='bold')
fig.tight_layout()
out = os.path.join(args.out_dir, f'scope_chain_{int(round(args.freq))}Hz.png')
fig.savefig(out, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f'\nwrote {out}')
