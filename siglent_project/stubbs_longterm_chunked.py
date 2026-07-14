#!/usr/bin/env python3
"""
stubbs_longterm_chunked.py

Processes a long recording (>10 min) in consecutive 30-second chunks,
running the full Stubbs LS shape-fit pipeline on each chunk independently.

Within each 30-second chunk (~1578 periods at 52.63 Hz), the accumulated
clock drift is ~2852 samples, well within the xcorr's valid range of
±7295 samples. Each chunk runs the same EMA drift correction that works
for the 30-second alpha/chirp sweep recordings.

Each chunk builds its own templates from its first N_TEMPLATE periods and
evaluates the remainder. Results are concatenated across all chunks.
"""
import sys, os, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.ndimage import uniform_filter1d
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chirp_stubbs_analysis as st

SR      = 384000
SDG_SR  = 400000
SDG_MAX = 16384

def rr(dur, dead=15.0):
    return SDG_SR / min(int((dur + dead) / 1000 * SDG_SR), SDG_MAX)


def main():
    parser = argparse.ArgumentParser(description='Long-term chunked Stubbs analysis')
    parser.add_argument('wav', help='Path to WAV file')
    parser.add_argument('--rep-rate', type=float, default=None,
                        help='Repetition rate Hz (default: 4ms+15ms=52.63 Hz)')
    parser.add_argument('--chunk-sec', type=float, default=30.0,
                        help='Chunk duration in seconds (default: 30.0)')
    parser.add_argument('--n-template', type=int, default=100,
                        help='Template periods per chunk (default: 100)')
    parser.add_argument('--out-dir', default=None,
                        help='Output directory (default: same as WAV)')
    args = parser.parse_args()

    rep     = args.rep_rate or rr(4.0, 15.0)
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.wav))
    os.makedirs(out_dir, exist_ok=True)
    label   = os.path.splitext(os.path.basename(args.wav))[0]

    period_n     = round(SR / rep)
    chunk_periods = int(args.chunk_sec * rep)      # periods per chunk
    chunk_samples = chunk_periods * period_n        # samples per chunk
    n_tmpl        = args.n_template

    # Sanity check: max drift within chunk vs xcorr range
    drift_ppm     = 187e-6                          # measured clock offset
    drift_per_p   = drift_ppm * period_n            # samples/period
    max_drift     = chunk_periods * drift_per_p
    xcorr_range   = period_n - 1
    print(f"Chunk = {args.chunk_sec:.0f}s = {chunk_periods} periods")
    print(f"Accumulated drift within chunk: {max_drift:.0f} samples "
          f"(xcorr range ±{xcorr_range})")
    if max_drift > xcorr_range * 0.8:
        print("WARNING: chunk too long — drift may exceed xcorr range. "
              "Reduce --chunk-sec.")

    print(f"\nLoading: {os.path.basename(args.wav)}")
    fs, raw = wavfile.read(args.wav)
    assert fs == SR, f"Expected {SR} Hz, got {fs} Hz"
    total_samples = len(raw)
    total_periods = total_samples // period_n
    print(f"  Duration: {total_samples/SR/60:.1f} min  "
          f"| {total_periods:,} periods total")

    # ── Process in chunks ─────────────────────────────────────────────────────
    tau_l_all   = []
    tau_r_all   = []
    sigma_l_all = []
    sigma_r_all = []
    time_all    = []   # absolute time in seconds for each eval period

    n_chunks      = (total_periods + chunk_periods - 1) // chunk_periods
    chunk_idx     = 0
    period_cursor = 0  # global period index of chunk start

    while period_cursor < total_periods:
        s_samp = period_cursor * period_n
        e_samp = min(s_samp + chunk_samples, total_samples)
        chunk_raw = raw[s_samp:e_samp]
        chunk_idx += 1

        n_periods_in_chunk = len(chunk_raw) // period_n
        if n_periods_in_chunk < n_tmpl + 5:
            print(f"\n  Chunk {chunk_idx}: only {n_periods_in_chunk} periods — skipping")
            period_cursor += n_periods_in_chunk
            continue

        print(f"\n  Chunk {chunk_idx}/{n_chunks}  "
              f"[{period_cursor/rep/60:.2f}–{(period_cursor+n_periods_in_chunk)/rep/60:.2f} min]  "
              f"({n_periods_in_chunk} periods)", flush=True)

        # Write chunk to a temp WAV so analyze() can read it
        tmp_wav = os.path.join(out_dir, '_chunk_tmp.wav')
        wavfile.write(tmp_wav, SR,
                      chunk_raw[:n_periods_in_chunk * period_n].astype(np.int16))

        # Full Stubbs LS pipeline on this chunk (same as 30s recordings)
        try:
            r = st.analyze(tmp_wav, rep_rate=rep, n_template=n_tmpl,
                           output_dir=out_dir, save_plot=False)
        except Exception as e:
            print(f"    analyze() failed: {e} — skipping chunk")
            period_cursor += n_periods_in_chunk
            continue

        n_eval = r['n_eval']
        if n_eval == 0:
            period_cursor += n_periods_in_chunk
            continue

        # Absolute time of each eval period (seconds from recording start)
        eval_period_offsets = np.arange(n_eval) + n_tmpl   # within chunk
        abs_times = (period_cursor + eval_period_offsets) / rep

        tau_l_all.append(r['tau_launch_sf'])
        tau_r_all.append(r['tau_return_sf'])
        sigma_l_all.append(r['sigma_launch_sf'])
        sigma_r_all.append(r['sigma_return_sf'])
        time_all.append(abs_times)

        spread   = float(np.std(r['delta_tau_sf']))
        med_sig  = float(np.median(r['sigma_dt_sf'])) * 1e3
        print(f"    N_eval={n_eval}  σ(Δτ)={spread:.4f}μs → "
              f"{spread/10.3:.4f}°C  δ(Δτ) med={med_sig:.1f} ns")

        period_cursor += n_periods_in_chunk

    # Clean up temp file
    if os.path.exists(tmp_wav):
        os.remove(tmp_wav)

    # ── Concatenate ───────────────────────────────────────────────────────────
    if not tau_l_all:
        print("No valid chunks processed."); return

    tau_l   = np.concatenate(tau_l_all)
    tau_r   = np.concatenate(tau_r_all)
    sigma_l = np.concatenate(sigma_l_all)
    sigma_r = np.concatenate(sigma_r_all)
    t_sec   = np.concatenate(time_all)
    t_min   = t_sec / 60.0
    dt      = tau_r - tau_l
    sigma_dt = np.sqrt(sigma_l**2 + sigma_r**2)

    print(f"\n{'='*55}")
    print(f"Total eval periods: {len(dt):,}")
    print(f"σ(Δτ) = {dt.std():.4f} μs  →  {dt.std()/10.3:.5f} °C/ping")
    print(f"mean Δτ = {dt.mean():.4f} μs")
    print(f"δ(Δτ) median = {np.median(sigma_dt)*1e3:.1f} ns")
    print(f"Total drift range: {(dt.max()-dt.min())/10.3:.3f} °C")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(15, 11))
    fig.suptitle(f'{label}  —  Long-term Δτ  (chunked, {args.chunk_sec:.0f}s/chunk)\n'
                 f'N={len(dt):,}   σ={dt.std():.3f}μs → {dt.std()/10.3:.4f}°C/ping   '
                 f'δ(Δτ) med={np.median(sigma_dt)*1e3:.1f} ns',
                 fontweight='bold', fontsize=11)

    # Panel 1: full Δτ time series with error bars (subsampled for speed)
    ax = axes[0]
    every = max(1, len(dt) // 5000)   # plot at most 5000 points
    ax.errorbar(t_min[::every], dt[::every], yerr=sigma_dt[::every],
                fmt='.', ms=1.5, lw=0, elinewidth=0.4, color='k', alpha=0.25,
                ecolor='gray')
    smooth_n = max(1, int(rep * 60))
    ax.plot(t_min, uniform_filter1d(dt, size=smooth_n), '-', lw=1.8,
            color='C0', label='1-min rolling mean')
    ax.axhline(dt.mean(), color='C3', lw=1, ls='--', alpha=0.7,
               label=f'mean = {dt.mean():.3f} μs')
    ax.set_xlabel('Time (min)'); ax.set_ylabel('Δτ (μs)')
    ax.set_title('Differential TOF time series (gray = LS error bars)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, ls='--')

    # Panel 2: Δτ histogram
    ax = axes[1]
    dt_c = dt[np.abs(dt - dt.mean()) < 5 * dt.std()]
    ax.hist(dt_c, bins=min(120, len(dt_c)//50), color='#1f77b4',
            edgecolor='white', lw=0.3, alpha=0.85)
    ax.axvline(dt.mean(),          color='C3', lw=2,   ls='-',
               label=f'mean = {dt.mean():.3f} μs')
    ax.axvline(dt.mean()-dt.std(), color='C3', lw=1.2, ls='--')
    ax.axvline(dt.mean()+dt.std(), color='C3', lw=1.2, ls='--',
               label=f'±1σ = {dt.std():.3f} μs  →  {dt.std()/10.3:.4f} °C/ping')
    ax.set_xlabel('Δτ (μs)'); ax.set_ylabel('Count')
    ax.set_title('Distribution'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 3: per-period δ(Δτ) over time
    ax = axes[2]
    ax.plot(t_min[::every], sigma_dt[::every] * 1e3, '.', ms=1.5, color='purple', alpha=0.3)
    ax.axhline(np.median(sigma_dt)*1e3, color='purple', lw=1.5, ls='--',
               label=f'median δ(Δτ) = {np.median(sigma_dt)*1e3:.1f} ns')
    ax.set_xlabel('Time (min)'); ax.set_ylabel('δ(Δτ) (ns)')
    ax.set_title('Per-period LS fit uncertainty δ(Δτ) = √(δ_L² + δ_R²)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, ls='--')

    plt.tight_layout()
    out_png = os.path.join(out_dir, f'stubbs_longterm_chunked_{label}.png')
    plt.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"\nPlot: {out_png}")

    # ── Save NPZ ─────────────────────────────────────────────────────────────
    out_npz = os.path.join(out_dir, f'stubbs_longterm_chunked_{label}.npz')
    np.savez(out_npz,
             tau_launch=tau_l, tau_return=tau_r,
             sigma_launch=sigma_l, sigma_return=sigma_r,
             delta_tau=dt, sigma_dt=sigma_dt,
             time_sec=t_sec, rep_rate=rep)
    print(f"Data: {out_npz}")


if __name__ == '__main__':
    main()
