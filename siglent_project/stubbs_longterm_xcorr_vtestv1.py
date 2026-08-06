#!/usr/bin/env python3
"""
stubbs_longterm_xcorr_vtest.py  —  vtest xcorr-stitched long-term Stubbs analysis.

Identical to stubbs_longterm_xcorr.py except:
  • _search_radius_us=700.0 passed to csa.analyze() (Category 2 fix: main CCF
    peak outside ±300μs search window no longer causes sidelobe-locking)
  • Output goes to analysis_xcorr_vtestv1/ and stubbs_pipeline_vtestv1/
  • PNG/NPZ names carry _vtest suffix for disambiguation

Each 30s chunk is processed using chirp_stubbs_analysis.analyze() with the
expanded search radius.  Local template per chunk, boundary stitching via
template cross-correlation or boundary-overlap median.

Usage:
    python3 stubbs_longterm_xcorr_vtest.py WAVFILE
        [--chunk-sec 30] [--n-template 100] [--mad-clip 5]
        [--inc-win L H] [--ref-win L H]
        [--out-dir PATH]
"""
import sys, os, argparse, shutil
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import correlate
import soundfile as _sf
import scipy.io.wavfile as _wv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chirp_stubbs_analysis as csa

SDG_SR = 400_000; SDG_MAX = 16_384
TOF_SENS = 10.3  # μs/°C


def rep_rate(dur_ms=4.0, dead_ms=15.0):
    return SDG_SR / min(int((dur_ms + dead_ms) / 1000 * SDG_SR), SDG_MAX)


def xcorr_offset(T_L_i, T_R_i, T_L1, T_R1, R_auto_L1, R_auto_R1, fs,
                 use_carrier_phase=False, f_carrier_hz=41000.0):
    """
    Compute absolute Δτ offset of chunk i relative to chunk 1 by
    cross-correlating their templates.  Returns (offset_us, sigma_us).

    use_carrier_phase=True: use xcorr_lag_carrier_phase for the template
    comparison — correct for narrow-BW waveforms (e.g. flat3db BW=2.3 kHz)
    where the LS shape-fit matrix is near-singular and parabola argmax slips.
    The carrier-phase weighted-complex-sum coarse estimator is immune to
    near-singular bandwidth and argmax-defaulting-to-wrong-position.
    """
    if use_carrier_phase:
        lag_L, sig_L, _, _ = csa.xcorr_lag_carrier_phase(
            T_L_i, T_L1, fs, f_carrier_hz=f_carrier_hz)
        lag_R, sig_R, _, _ = csa.xcorr_lag_carrier_phase(
            T_R_i, T_R1, fs, f_carrier_hz=f_carrier_hz)
    else:
        seed_L, _, _, _ = csa.xcorr_lag_us(T_L_i, T_L1, fs)
        seed_R, _, _, _ = csa.xcorr_lag_us(T_R_i, T_R1, fs)
        lag_L, sig_L, _, _ = csa.xcorr_lag_shape_fit(T_L_i, T_L1, fs, R_auto=R_auto_L1,
                                                       coarse_lag_us=seed_L)
        lag_R, sig_R, _, _ = csa.xcorr_lag_shape_fit(T_R_i, T_R1, fs, R_auto=R_auto_R1,
                                                       coarse_lag_us=seed_R)
    return lag_R - lag_L, float(np.sqrt(sig_L**2 + sig_R**2))


def save_hist_uncertainties(sl_us, sr_us, chunk_idx, pipeline_dir, label):
    """
    3-panel histogram of per-period fit uncertainties: δ_launch, δ_return,
    √(δ_L²+δ_R²)  in ns.  Matches the hist_uncertainties.png reference format.
    """
    dl  = sl_us * 1e3
    dr  = sr_us * 1e3
    ddt = np.sqrt(dl**2 + dr**2)

    N   = len(dl)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    specs = [
        (dl,  'steelblue',    'δ_launch (ns)',         'δ_launch  (xcorr fit uncertainty)'),
        (dr,  '#44aa77',      'δ_return (ns)',         'δ_return  (xcorr fit uncertainty)'),
        (ddt, 'mediumpurple', '√(δ_L²+δ_R²) (ns)',    '√(δ_L²+δ_R²)  (propagated Δτ fit uncertainty)'),
    ]
    for ax, (d, col, xl, ti) in zip(axes, specs):
        _p99 = float(np.percentile(d, 99))
        _med = float(np.median(d))
        xlim = max(min(_p99 * 1.2, _med * 10), 2.0)
        # Bin only within [0, xlim] so bins match the visible range.
        # Without range=, matplotlib bins over the full outlier-extended range
        # and the display shows the interior of one giant bin — a solid rectangle.
        ax.hist(d, bins=50, range=(0, xlim), color=col, alpha=0.75, edgecolor='none')
        m = float(np.mean(d));  s = float(np.std(d))
        ax.axvline(m,     color='red', lw=1.5,         label=f'mean = {m:.1f} ns')
        ax.axvline(m - s, color='red', lw=1.0, ls='--', label=f'±1σ = {s:.1f} ns')
        ax.axvline(m + s, color='red', lw=1.0, ls='--')
        ax.set_xlabel(xl, fontsize=10);  ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'{ti}  (N={N:,})', fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlim(0, xlim)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f'{label}  —  Per-period fit uncertainty  |  Chunk {chunk_idx:03d}\n'
        r'$\delta_{\Delta\tau} = \sqrt{\delta_L^2 + \delta_R^2}$  (LS shape-fit)',
        fontsize=11)
    fig.tight_layout()
    out_path = os.path.join(pipeline_dir, f'hist_uncertainties_chunk_{chunk_idx:03d}.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def save_partial_png(t_sec, dt, sig, out_path, label, rep):
    """
    Save accumulated diff-TOF in the exact lsfit panel-3 style:
    hollow black circles, error bars, blue thread, mean/σ dashed lines.
    Shows ALL points — no MAD-based hiding — so no chunks are silently dropped.
    Y-axis is set to ±5σ_robust so the scale stays sensible; points outside
    the range are still drawn (thread hits the boundary rather than gapping).
    """
    med   = float(np.median(dt))
    mad   = float(np.median(np.abs(dt - med))) * 1.4826
    _mean = float(np.mean(dt))
    _std  = float(np.std(dt))

    # y-axis: generous ±5× robust σ so stitching failures are visible at the
    # boundary rather than completely absent.  Use clip=False so the thread
    # draws to the boundary edge for outlying chunks.
    yr   = max(5 * mad, 0.5)
    ylim = (med - yr, med + yr)

    fig, ax = plt.subplots(figsize=(15, 4.5))

    # 1 — blue thread — ALL points, no NaN masking (no gaps)
    ax.plot(t_sec, dt, '-', color='#4477cc', lw=0.7, alpha=0.6, zorder=1)

    # 2 — hollow black circles — ALL points
    ax.plot(t_sec, dt, 'o', ms=1.5, mfc='none', mec='black', mew=0.5,
            alpha=0.6, zorder=2, label=f'Δτ per period  (N={len(dt):,})')

    # 3 — mean (navy dashed) and ±σ (gray dotted)  — use robust stats for lines
    ax.axhline(med,       color='navy', ls='--', lw=1.2, alpha=0.7,
               label=f'median = {med:.3f} μs')
    ax.axhline(med + mad, color='gray', ls=':', lw=0.9)
    ax.axhline(med - mad, color='gray', ls=':', lw=0.9,
               label=f'MAD σ = {mad:.3f} μs  →  {mad/TOF_SENS:.4f}°C/ping')

    ax.set_ylim(*ylim)

    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Δτ = τ_return − τ_launch  (μs)', fontsize=10)
    ax.set_title(
        f'Differential TOF Time Series  '
        f'(δ(Δτ) LS median = {np.median(sig)*1e3:.1f} ns)\n'
        f'{label}  —  vtest vtestv1 xcorr-stitch (±700μs + min-burst window)  '
        f'N={len(t_sec):,}  σ={_std:.3f} μs  '
        f'[partial: {t_sec[-1]/60:.1f} min accumulated]',
        fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('wav')
    ap.add_argument('--rep-rate',   type=float, default=None)
    ap.add_argument('--chunk-sec',  type=float, default=30.0)
    ap.add_argument('--n-template', type=int,   default=100)
    ap.add_argument('--out-dir',    default=None)
    ap.add_argument('--mad-clip',   type=float, default=None)
    ap.add_argument('--inc-win',    type=float, nargs=2, default=None, metavar=('L','H'))
    ap.add_argument('--ref-win',    type=float, nargs=2, default=None, metavar=('L','H'))
    ap.add_argument('--tof-us',       type=float, default=5700.0)
    ap.add_argument('--tol-ms',       type=float, default=2.0)
    ap.add_argument('--use-parabola',      action='store_true', default=False,
                    help='Use parabola for per-period Δτ (narrow-BW waveforms).')
    ap.add_argument('--use-carrier-phase', action='store_true', default=False,
                    help='Use carrier-phase estimator for template xcorr stitching '
                         'AND for per-period Δτ.  Best for narrow-BW waveforms '
                         '(flat3db BW=2.3kHz) where shape-fit and parabola both '
                         'suffer from flat Hilbert envelope argmax slipping.')
    ap.add_argument('--f-carrier', type=float, default=41000.0,
                    help='Carrier frequency for carrier-phase estimator (Hz). '
                         'Default 41000 (flat3db centre). Use 42000 for other chirps.')
    args = ap.parse_args()

    rep     = args.rep_rate or rep_rate()
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.wav)),
                                            'analysis_xcorr_vtestv1')
    os.makedirs(out_dir, exist_ok=True)
    chunks_dir = os.path.join(out_dir, 'chunks')
    os.makedirs(chunks_dir, exist_ok=True)

    label = os.path.splitext(os.path.basename(args.wav))[0]

    raw, SR = _sf.read(args.wav, dtype='int16', always_2d=False)
    period_n      = round(SR / rep)
    chunk_periods = int(args.chunk_sec * rep)
    chunk_samples = chunk_periods * period_n
    total_samples = len(raw)
    total_periods = total_samples // period_n
    n_tmpl        = args.n_template
    n_chunks      = (total_periods + chunk_periods - 1) // chunk_periods

    print(f"  SR={SR} Hz  rep={rep:.3f} Hz  period_n={period_n}")
    print(f"  Duration: {total_samples/SR/60:.1f} min  |  {total_periods:,} periods  |  {n_chunks} chunks")
    _use_cp = getattr(args, 'use_carrier_phase', False)
    _use_p  = getattr(args, 'use_parabola', False)
    print(f"  Mode: per-chunk analyze() + xcorr stitching  "
          f"[estimator: {'parabola' if (_use_cp or _use_p) else 'shape-fit'}]")

    # (boundary stitching: no chunk-1 template reference needed)

    # Windows: use explicit CLI args if given; otherwise learn from chunk 1 and
    # reuse for all subsequent chunks.  Consistent windows across all chunks are
    # critical: every chunk's templates are built in the same windows, so the
    # xcorr stitching compares apples to apples and offsets are not garbage.
    learned_inc_win = tuple(args.inc_win) if args.inc_win else None
    learned_ref_win = tuple(args.ref_win) if args.ref_win else None

    # Accumulating results
    dt_acc  = []   # absolute Δτ per period
    sig_acc = []   # per-period LS uncertainty
    t_acc   = []   # absolute time (s)

    period_cursor = 0

    while period_cursor < total_periods:
        s_samp    = period_cursor * period_n
        e_samp    = min(s_samp + chunk_samples, total_samples)
        chunk_raw = raw[s_samp:e_samp]
        chunk_idx = period_cursor // chunk_periods + 1
        n_in      = len(chunk_raw) // period_n

        if n_in < n_tmpl + 5:
            period_cursor += n_in
            continue

        t_lo = period_cursor / rep
        t_hi = (period_cursor + n_in) / rep
        print(f"\n  Chunk {chunk_idx}/{n_chunks}  "
              f"[{t_lo/60:.2f}–{t_hi/60:.2f} min]  ({n_in} periods)")

        # Write chunk WAV as chunks/chunk_001.wav (flat, no subdirectory).
        # analyze() saves PNG to <wav_dir>/../stubbs_pipeline_v3/stubbs_<stem>.png,
        # so with stem=chunk_001 each chunk gets its own unique PNG:
        #   analysis_xcorr/stubbs_pipeline_v3/stubbs_chunk_001.png
        chunk_wav = os.path.join(chunks_dir, f'chunk_{chunk_idx:03d}.wav')
        _wv.write(chunk_wav, SR, chunk_raw[:n_in * period_n].astype(np.int16))

        try:
            # vtest: _search_radius_us=700.0 (Category 2 fix — main peak findable
            # even when carrier-cycle slip pushes it to ±450 μs from lag=0)
            _pl_dir = os.path.normpath(os.path.join(
                os.path.dirname(chunk_wav), '..', 'stubbs_pipeline_vtestv1'))
            os.makedirs(_pl_dir, exist_ok=True)
            r = csa.analyze(
                chunk_wav,
                rep_rate=rep,
                n_template=n_tmpl,
                incident_win_ms=learned_inc_win,
                reflection_win_ms=learned_ref_win,
                save_plot=True,
                _plot_parabola=(_use_cp or _use_p),
                _search_radius_us=700.0,
                _pipeline_dir=_pl_dir,
            )
        except Exception as e:
            print(f"  ERROR in chunk {chunk_idx}: {e}")
            period_cursor += n_in
            continue

        # After chunk 1 (no explicit windows): lock in the auto-detected windows
        # so every subsequent chunk uses the same inc/ref extraction windows.
        if learned_inc_win is None:
            learned_inc_win = r['incident_win_ms']
            learned_ref_win = r['reflection_win_ms']
            print(f"  [windows] Learned from chunk 1: "
                  f"inc={learned_inc_win[0]:.2f}–{learned_inc_win[1]:.2f} ms  "
                  f"ref={learned_ref_win[0]:.2f}–{learned_ref_win[1]:.2f} ms")

        # ── Boundary stitching: offset chunk i so its START matches chunk i-1's END ──
        # Simpler and more robust than xcorr template comparison.
        # Uses the mean of the last N_BOUNDARY local Δτ values already accumulated
        # (absolute) and the mean of the first N_BOUNDARY local Δτ values of this
        # chunk, then sets offset = prev_end_mean − curr_start_mean.
        N_BOUNDARY  = 300   # window size for boundary median
        N_SKIP_CURR = 200   # skip first N periods of new chunk — these have worst template
                            # settling and most parabola failures (argmax still confused)
        # Use the SAME estimator for boundary stitching as for the actual analysis —
        # critical for narrow-BW waveforms where shape-fit is near-singular and
        # its values (σ > 200 μs) corrupt the boundary median.
        if _use_cp or _use_p:
            tl_loc = r['tau_launch']; tr_loc = r['tau_return']
        else:
            tl_loc = r['tau_launch_sf']; tr_loc = r['tau_return_sf']
        dt_loc_raw = tr_loc - tl_loc   # local Δτ, full eval array (before MAD clip)

        def _robust_boundary_median(arr, extreme_thresh_us=50.0):
            """Median after removing extreme parabola errors (argmax at search boundary)."""
            if len(arr) == 0:
                return 0.0
            m = float(np.median(arr))
            mad = float(np.median(np.abs(arr - m))) * 1.4826
            # Generous threshold: keep anything within max(5×MAD, 50 μs) of median
            thresh = max(5.0 * mad, extreme_thresh_us)
            good = np.abs(arr - m) <= thresh
            return float(np.median(arr[good])) if good.sum() > 5 else m

        if len(dt_acc) == 0:
            # Chunk 1: absolute reference, offset = 0
            offset_us = 0.0
            print(f"  [stitch] Chunk 1 = reference (offset = 0)")
        else:
            all_prev   = np.concatenate(dt_acc)
            n_b        = min(N_BOUNDARY, len(all_prev))
            n_b_curr   = min(N_BOUNDARY, len(dt_loc_raw))
            prev_end   = _robust_boundary_median(all_prev[-n_b:])
            # Skip the first N_SKIP_CURR periods of the new chunk — they have the
            # worst template-settling and most parabola failures (concentrated in
            # the first 2-3 seconds where the template CCF is still confused).
            skip       = min(N_SKIP_CURR, max(0, len(dt_loc_raw) - n_b_curr - 10))
            curr_win   = dt_loc_raw[skip : skip + n_b_curr]
            n_b_curr   = len(curr_win)
            curr_start = _robust_boundary_median(curr_win) if n_b_curr > 10 else 0.0
            offset_us  = prev_end - curr_start
            print(f"  [stitch] offset={offset_us:+.3f} μs  "
                  f"(prev_end={prev_end:+.3f}  curr_start={curr_start:+.3f}  "
                  f"skip={skip})")

        # ── Per-chunk uncertainty histogram ───────────────────────────────────
        pipeline_dir = os.path.normpath(os.path.join(
            os.path.dirname(chunk_wav), '..', 'stubbs_pipeline_vtestv1'))
        # Histogram uses whichever estimator is selected for Δτ
        if _use_cp or _use_p:
            _sig_L = r['sigma_launch']; _sig_R = r['sigma_return']
        else:
            _sig_L = r['sigma_launch_sf']; _sig_R = r['sigma_return_sf']
        save_hist_uncertainties(_sig_L, _sig_R, chunk_idx, pipeline_dir, label)

        # ── Absolute Δτ = local Δτ + boundary offset ─────────────────────────
        # Per-period estimator: parabola for narrow-BW, shape-fit otherwise.
        # (--use-carrier-phase affects template xcorr stitching only for now;
        #  per-period carrier-phase per eval period is a future improvement.)
        if _use_cp or _use_p:
            tl_use  = r['tau_launch'];  tr_use = r['tau_return']
            sl      = r['sigma_launch']; sr_ = r['sigma_return']
        else:
            tl_use  = r['tau_launch_sf']; tr_use = r['tau_return_sf']
            sl      = r['sigma_launch_sf']; sr_ = r['sigma_return_sf']
        dt_loc_raw = tr_use - tl_use
        dt_loc  = dt_loc_raw
        dt_abs  = dt_loc + offset_us
        sigma   = np.sqrt(sl**2 + sr_**2)

        n_eval  = r['n_eval']
        abs_t   = (period_cursor + np.arange(n_eval) + n_tmpl) / rep

        # MAD clip at args.mad_clip σ  (mad = 1.4826 × median|x−med| = robust σ)
        keep = np.ones(len(dt_abs), bool)
        if args.mad_clip and len(dt_abs) > 10:
            m   = np.median(dt_abs)
            mad = np.median(np.abs(dt_abs - m)) * 1.4826
            if mad > 1e-9:
                keep = np.abs(dt_abs - m) < args.mad_clip * mad
            nc = (~keep).sum()
            if nc:
                print(f"    MAD clip: {nc}/{len(dt_abs)} outliers")

        dt_acc.append(dt_abs[keep])
        sig_acc.append(sigma[keep])
        t_acc.append(abs_t[keep])

        _std_loc = float(np.std(dt_loc[keep]))
        _d_med   = float(np.median(sigma[keep]) * 1e3)
        print(f"    N_eval={keep.sum()}/{n_eval}  "
              f"σ(Δτ_local)={_std_loc:.4f} μs  "
              f"δ med={_d_med:.1f} ns")

        # ── Partial accumulated PNG ────────────────────────────────────────────
        partial_png = os.path.join(out_dir, f'partial_vtestv1_{label}.png')
        save_partial_png(
            np.concatenate(t_acc),
            np.concatenate(dt_acc),
            np.concatenate(sig_acc),
            partial_png, label, rep)
        print(f"  → partial PNG updated: {partial_png}")

        period_cursor += n_in


    # ── Final assembly ─────────────────────────────────────────────────────────
    t_final   = np.concatenate(t_acc)
    dt_final  = np.concatenate(dt_acc)
    sig_final = np.concatenate(sig_acc)

    print(f"\n[xcorr] Done.  Total eval periods: {len(t_final):,}")
    print(f"σ(Δτ) = {np.std(dt_final):.4f} μs  →  {np.std(dt_final)/TOF_SENS:.5f} °C/ping")
    print(f"δ(Δτ) median = {np.median(sig_final)*1e3:.1f} ns")

    # Final 3-panel lsfit-style plot
    every = max(1, len(t_final) // 5000)
    fig, axes = plt.subplots(3, 1, figsize=(15, 9))

    ax = axes[0]
    med_f = np.median(dt_final)
    mad_f = np.median(np.abs(dt_final - med_f)) * 1.4826
    inl_f = np.abs(dt_final - med_f) <= 5 * mad_f
    _mn   = float(np.mean(dt_final[inl_f]))
    _sd   = float(np.std(dt_final[inl_f]))
    dt_pf = dt_final.copy().astype(float); dt_pf[~inl_f] = np.nan
    ax.plot(t_final/60, dt_pf, '-', color='#4477cc', lw=0.7, alpha=0.6, zorder=1)
    ax.errorbar(t_final[::every]/60, dt_final[::every], yerr=sig_final[::every],
                fmt='o', ms=2.0, mfc='none', mec='black', mew=0.6,
                ecolor='black', elinewidth=0.4, capsize=2, zorder=2, alpha=0.6)
    from numpy.lib.stride_tricks import sliding_window_view as _svw
    dt_rep = float(np.median(np.diff(t_final))); rn = max(3, int(60/dt_rep))
    roll = np.nanmean(_svw(np.pad(dt_pf,(rn//2,rn//2),mode='edge'),rn),axis=1)[:len(dt_final)]
    ax.plot(t_final/60, roll, 'k-', lw=1.5, label='1-min rolling mean')
    ax.axhline(_mn, color='navy', ls='--', lw=1.2, alpha=0.7, label=f'mean={_mn:.3f}μs')
    ax.axhline(_mn+_sd, color='gray', ls=':', lw=0.9)
    ax.axhline(_mn-_sd, color='gray', ls=':', lw=0.9, label=f'σ={_sd:.3f}μs→{_sd/TOF_SENS:.4f}°C/ping')
    if inl_f.sum()>5:
        yr=max(dt_final[inl_f].max()-dt_final[inl_f].min(),2*mad_f,0.2)
        ax.set_ylim(dt_final[inl_f].min()-0.12*yr, dt_final[inl_f].max()+0.12*yr)
    ax.set_title(f'{label}  —  vtest xcorr-stitched long-term Δτ (search ±700μs)\n'
                 f'N={len(t_final):,}  σ={_sd:.3f}μs  δ med={np.median(sig_final)*1e3:.1f}ns', fontsize=9)
    ax.set_ylabel('Δτ (μs, re T_0)'); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

    axes[1].hist(dt_final[inl_f], bins=80, color='steelblue', alpha=0.7)
    axes[1].axvline(_mn, color='red', lw=1.5, label=f'mean={_mn:.3f}μs')
    axes[1].set_title('Distribution'); axes[1].set_xlabel('Δτ (μs)')
    axes[1].legend(fontsize=8); axes[1].grid(True,alpha=0.3)

    axes[2].plot(t_final[::every]/60, sig_final[::every]*1e3, '.', ms=1.2, color='purple', alpha=0.5)
    axes[2].axhline(np.median(sig_final)*1e3, color='purple', lw=1.5, ls='--',
                    label=f'median δ={np.median(sig_final)*1e3:.1f}ns')
    axes[2].set_title('Per-period LS uncertainty δ(Δτ)'); axes[2].set_ylabel('δ(Δτ) (ns)')
    axes[2].set_xlabel('Time (min)'); axes[2].legend(fontsize=8); axes[2].grid(True,alpha=0.3)

    fig.tight_layout()
    out_png = os.path.join(out_dir, f'stubbs_xcorr_vtestv1_{label}.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight'); plt.close(fig)

    out_npz = os.path.join(out_dir, f'stubbs_xcorr_vtestv1_{label}.npz')
    np.savez(out_npz, delta_tau=dt_final, sigma_dt=sig_final, time_sec=t_final, rep_rate=rep)
    print(f"Plot: {out_png}\nData: {out_npz}")


if __name__ == '__main__':
    main()
