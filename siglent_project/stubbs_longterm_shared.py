#!/usr/bin/env python3
"""
stubbs_longterm_shared.py

Long-term Stubbs LS analysis using a SINGLE shared template built from
chunk 1, applied across all 30-second chunks — giving a continuous
absolute Δτ time series over the full recording.

For each period in every chunk (after chunk 1):
  1. Adaptive slicing positions the burst at ~768 samples (removes large
     positional clock drift automatically).
  2. xcorr of the period against chunk 1's ref0 → measure carrier phase
     offset φ at lag≈0 via the analytic signal.
  3. Pure phase rotation: corrected = period·cos(φ) + H{period}·sin(φ)
     This aligns the carrier to chunk 1's reference frame without any
     positional shift.
  4. xcorr corrected period with shared T_launch/T_return → τ_L, τ_R.
  5. Δτ = τ_R − τ_L is now absolute and continuous across all chunks.

Memory usage: one chunk (~46 MB float32) at a time.
"""
import sys, os, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, hilbert, correlate
from scipy.ndimage import uniform_filter1d
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chirp_stubbs_analysis as st

SR      = 384000
SDG_SR  = 400000
SDG_MAX = 16384

def rr(dur, dead=15.0):
    return SDG_SR / min(int((dur + dead) / 1000 * SDG_SR), SDG_MAX)

def hp(s):
    sos = butter(8, 30000 / (SR / 2), btype='high', output='sos')
    return sosfiltfilt(sos, s.astype(np.float64)).astype(np.float32)

def fft_shift_small(x, s):
    """FFT phase-ramp shift by |s| < 5 samples (no positional wraparound)."""
    N = len(x)
    X = np.fft.rfft(x.astype(np.float64))
    X *= np.exp(-2j * np.pi * np.fft.rfftfreq(N) * s)
    return np.fft.irfft(X, n=N).astype(np.float32)

def xcorr_lag_parabola(signal_seg, template, fs, search_radius_us=500.0):
    """
    Envelope-peak lag estimator (parabola on Hilbert envelope).
    Phase-insensitive — works correctly with shared templates regardless
    of accumulated clock-drift carrier phase offset.
    Returns (lag_us, sigma_us).
    """
    xc  = correlate(signal_seg.astype(np.float64),
                    template.astype(np.float64), mode='full')
    env = np.abs(hilbert(xc))
    env /= (np.max(env) + 1e-12)
    center  = len(template) - 1
    radius  = int(search_radius_us * fs / 1e6)
    lo, hi  = max(0, center - radius), min(len(env) - 1, center + radius)
    idx     = int(np.argmax(env[lo:hi + 1]))
    hw      = 5
    sl, sh  = max(0, idx - hw), min(len(env[lo:hi+1]), idx + hw + 1)
    x       = np.arange(sl, sh, dtype=float) - idx
    y       = env[lo:hi+1][sl:sh]
    step_us = 1e6 / fs
    if len(y) >= 5 and np.ptp(y) > 1e-12:
        coeffs, res, _, _, _ = np.polyfit(x, y, 2, full=True)
        a, b = coeffs[0], coeffs[1]
        if abs(a) > 1e-10:
            dl      = float(np.clip(-b / (2 * a), -hw, hw))
            peak_f  = float(lo + idx) + dl
            rms     = float(np.sqrt(res[0] / len(y))) if len(res) > 0 else step_us * 0.1
            sigma   = rms / (abs(a) * float(np.sqrt(np.sum(x**2)))) * step_us
            return (peak_f - center) / fs * 1e6, sigma
    return (lo + idx - center) / fs * 1e6, step_us


def main():
    parser = argparse.ArgumentParser(
        description='Long-term Stubbs with shared template across all chunks')
    parser.add_argument('wav')
    parser.add_argument('--rep-rate', type=float, default=None)
    parser.add_argument('--chunk-sec', type=float, default=30.0)
    parser.add_argument('--n-template', type=int, default=100)
    parser.add_argument('--out-dir', default=None)
    args = parser.parse_args()

    rep     = args.rep_rate or rr(4.0, 15.0)
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.wav))
    os.makedirs(out_dir, exist_ok=True)
    label   = os.path.splitext(os.path.basename(args.wav))[0]

    period_n      = round(SR / rep)
    chunk_periods = int(args.chunk_sec * rep)
    chunk_samples = chunk_periods * period_n
    n_tmpl        = args.n_template
    center        = period_n - 1   # lag=0 index in a full correlate

    print(f"Loading: {os.path.basename(args.wav)}")
    fs, raw = wavfile.read(args.wav)
    assert fs == SR
    total_samples = len(raw)
    total_periods = total_samples // period_n
    n_chunks      = (total_periods + chunk_periods - 1) // chunk_periods
    print(f"  {total_samples/SR/60:.1f} min  |  {total_periods:,} periods  "
          f"|  {n_chunks} chunks of {args.chunk_sec:.0f}s")

    # ── Chunk 1: build shared templates via full analyze() ──────────────────
    print(f"\n  Chunk 1/{n_chunks}  [0.00–{args.chunk_sec/60:.2f} min]  "
          f"(building shared templates)", flush=True)
    e1 = min(chunk_samples, total_samples)
    tmp_wav = os.path.join(out_dir, '_ltshared_tmp.wav')
    wavfile.write(tmp_wav, SR,
                  raw[:e1 - (e1 % period_n)].astype(np.int16))

    r1  = st.analyze(tmp_wav, rep_rate=rep, n_template=n_tmpl,
                     output_dir=out_dir, save_plot=False)

    T_launch = r1['T_launch'].astype(np.float64)
    T_return = r1['T_return'].astype(np.float64)
    iw       = r1['incident_win_ms']
    rw       = r1['reflection_win_ms']
    # Parabola (envelope) estimator used for subsequent chunks — phase-insensitive,
    # so shared template works correctly regardless of accumulated carrier phase drift.

    # Build chunk 1's ref0: first adaptive slice of chunk 1's signal
    chunk1_sig    = hp(raw[:e1].astype(np.float32) / 32768.0)
    chunk1_slices = st.adaptive_slice_periods(chunk1_sig, SR, rep)
    chunk1_ref0   = chunk1_sig[chunk1_slices[0][0]:
                               chunk1_slices[0][1]].astype(np.float64)
    print(f"    Shared template built.  "
          f"T_launch peak: {np.max(np.abs(T_launch)):.4f}  "
          f"T_return peak: {np.max(np.abs(T_return)):.4f}")
    print(f"    iw={iw}  rw={rw}")

    # Store chunk 1 eval results (already measured against its own template,
    # so these are already in chunk1's reference frame = shared reference)
    tau_l_all   = [r1['tau_launch_sf']]
    tau_r_all   = [r1['tau_return_sf']]
    sigma_l_all = [r1['sigma_launch_sf']]
    sigma_r_all = [r1['sigma_return_sf']]
    n1          = r1['n_eval']
    time_all    = [(np.arange(n1) + n_tmpl) / rep]

    med_sig = float(np.median(r1['sigma_dt_sf'])) * 1e3
    print(f"    N_eval={n1}  σ(Δτ)={r1['std_delta_tau_sf_us']:.4f}μs  "
          f"δ(Δτ) med={med_sig:.1f} ns")

    # ── Chunks 2…N: eval with shared templates ───────────────────────────────
    period_cursor = chunk_periods   # global period index of chunk start

    while period_cursor < total_periods:
        s_samp = period_cursor * period_n
        e_samp = min(s_samp + chunk_samples, total_samples)
        chunk_raw = raw[s_samp:e_samp]
        n_in_chunk = len(chunk_raw) // period_n
        if n_in_chunk < 5:
            period_cursor += n_in_chunk; continue

        chunk_idx = period_cursor // chunk_periods + 1
        t_lo = period_cursor / rep / 60.0
        t_hi = (period_cursor + n_in_chunk) / rep / 60.0
        print(f"\n  Chunk {chunk_idx}/{n_chunks}  "
              f"[{t_lo:.2f}–{t_hi:.2f} min]  ({n_in_chunk} periods)",
              flush=True)

        chunk_sig = hp(chunk_raw.astype(np.float32) / 32768.0)
        slices    = st.adaptive_slice_periods(chunk_sig, SR, rep)
        if not slices:
            period_cursor += n_in_chunk; continue

        tau_l_chunk = []; tau_r_chunk = []
        sig_l_chunk = []; sig_r_chunk = []
        abs_times   = []
        period_in_chunk = 0   # local period counter within this chunk

        for s_, e_ in slices:
            period = chunk_sig[s_:e_]
            if len(period) < period_n // 2:
                period_in_chunk += 1; continue

            # ── Carrier phase alignment to chunk 1's ref0 ──────────────────
            phi       = measure_carrier_phase(period, chunk1_ref0)
            corrected = phase_rotate(period, phi)

            # ── Mask windows and xcorr with shared templates ───────────────
            inc_m = st.mask_period(corrected, SR, iw)
            ref_m = st.mask_period(corrected, SR, rw)

            tl, σl, _, _ = st.xcorr_lag_shape_fit(
                inc_m.astype(np.float64), T_launch, SR, R_auto=R_auto_L)
            tr, σr, _, _ = st.xcorr_lag_shape_fit(
                ref_m.astype(np.float64), T_return, SR, R_auto=R_auto_R)

            if tl is not None and tr is not None:
                tau_l_chunk.append(tl);  tau_r_chunk.append(tr)
                sig_l_chunk.append(σl);  sig_r_chunk.append(σr)
                global_period = period_cursor + period_in_chunk
                abs_times.append(global_period / rep)

            period_in_chunk += 1

        if not tau_l_chunk:
            period_cursor += n_in_chunk; continue

        tl_arr = np.array(tau_l_chunk)
        tr_arr = np.array(tau_r_chunk)
        dt_arr = tr_arr - tl_arr
        sl_arr = np.array(sig_l_chunk)
        sr_arr = np.array(sig_r_chunk)
        sig_dt = np.sqrt(sl_arr**2 + sr_arr**2)

        tau_l_all.append(tl_arr);   tau_r_all.append(tr_arr)
        sigma_l_all.append(sl_arr); sigma_r_all.append(sr_arr)
        time_all.append(np.array(abs_times))

        print(f"    N={len(dt_arr)}  σ(Δτ)={dt_arr.std():.4f}μs → "
              f"{dt_arr.std()/10.3:.4f}°C  δ(Δτ) med={np.median(sig_dt)*1e3:.1f} ns")

        period_cursor += n_in_chunk

    if os.path.exists(tmp_wav):
        os.remove(tmp_wav)

    # ── Concatenate ───────────────────────────────────────────────────────────
    if not tau_l_all:
        print("No data."); return

    tau_l    = np.concatenate(tau_l_all)
    tau_r    = np.concatenate(tau_r_all)
    sigma_l  = np.concatenate(sigma_l_all)
    sigma_r  = np.concatenate(sigma_r_all)
    t_sec    = np.concatenate(time_all)
    t_min    = t_sec / 60.0
    dt       = tau_r - tau_l
    sigma_dt = np.sqrt(sigma_l**2 + sigma_r**2)

    print(f"\n{'='*55}")
    print(f"Total eval periods : {len(dt):,}")
    print(f"σ(Δτ)              = {dt.std():.4f} μs  →  {dt.std()/10.3:.5f} °C/ping")
    print(f"mean Δτ            = {dt.mean():.4f} μs")
    print(f"δ(Δτ) median       = {np.median(sigma_dt)*1e3:.1f} ns")
    print(f"Total drift range  = {(dt.max()-dt.min())/10.3:.3f} °C")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(15, 11))
    fig.suptitle(
        f'{label}  —  Long-term Δτ  (shared template, {args.chunk_sec:.0f}s chunks)\n'
        f'N={len(dt):,}   σ={dt.std():.3f}μs → {dt.std()/10.3:.4f}°C/ping   '
        f'δ(Δτ) med={np.median(sigma_dt)*1e3:.1f} ns',
        fontweight='bold', fontsize=11)

    every = max(1, len(dt) // 5000)
    smooth_n = max(1, int(rep * 60))

    ax = axes[0]
    ax.errorbar(t_min[::every], dt[::every], yerr=sigma_dt[::every],
                fmt='.', ms=1.5, lw=0, elinewidth=0.4,
                color='k', alpha=0.25, ecolor='gray')
    ax.plot(t_min, uniform_filter1d(dt, size=smooth_n), '-',
            lw=1.8, color='C0', label='1-min rolling mean')
    ax.axhline(dt.mean(), color='C3', lw=1, ls='--', alpha=0.7,
               label=f'mean = {dt.mean():.3f} μs')
    ax.set_xlabel('Time (min)'); ax.set_ylabel('Δτ (μs)')
    ax.set_title('Differential TOF — continuous absolute reference (shared template)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, ls='--')

    ax = axes[1]
    dt_c = dt[np.abs(dt - dt.mean()) < 5 * dt.std()]
    ax.hist(dt_c, bins=min(120, len(dt_c)//50), color='#1f77b4',
            edgecolor='white', lw=0.3, alpha=0.85)
    ax.axvline(dt.mean(), color='C3', lw=2, ls='-',
               label=f'mean = {dt.mean():.3f} μs')
    ax.axvline(dt.mean() - dt.std(), color='C3', lw=1.2, ls='--')
    ax.axvline(dt.mean() + dt.std(), color='C3', lw=1.2, ls='--',
               label=f'±1σ = {dt.std():.3f} μs  →  {dt.std()/10.3:.4f} °C/ping')
    ax.set_xlabel('Δτ (μs)'); ax.set_ylabel('Count')
    ax.set_title('Distribution'); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(t_min[::every], sigma_dt[::every] * 1e3, '.', ms=1.5,
            color='purple', alpha=0.3)
    ax.axhline(np.median(sigma_dt) * 1e3, color='purple', lw=1.5, ls='--',
               label=f'median δ(Δτ) = {np.median(sigma_dt)*1e3:.1f} ns')
    ax.set_xlabel('Time (min)'); ax.set_ylabel('δ(Δτ) (ns)')
    ax.set_title('Per-period LS fit uncertainty δ(Δτ) = √(δ_L² + δ_R²)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, ls='--')

    plt.tight_layout()
    out_png = os.path.join(out_dir, f'stubbs_longterm_shared_{label}.png')
    plt.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"\nPlot : {out_png}")

    out_npz = os.path.join(out_dir, f'stubbs_longterm_shared_{label}.npz')
    np.savez(out_npz, tau_launch=tau_l, tau_return=tau_r,
             sigma_launch=sigma_l, sigma_return=sigma_r,
             delta_tau=dt, sigma_dt=sigma_dt,
             time_sec=t_sec, rep_rate=rep)
    print(f"Data : {out_npz}")


if __name__ == '__main__':
    main()
