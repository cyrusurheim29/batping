#!/usr/bin/env python3
"""
analyze_statistics_campaign.py

Matched-filter analysis of the 10× enclosure / 10× open statistics campaign.
Each 30s WAV is analyzed independently with the Stubbs dual-template pipeline.
No stitching — each recording is its own absolute reference.

Usage:
    python3 analyze_statistics_campaign.py
    python3 analyze_statistics_campaign.py --recordings-dir recordings/
    python3 analyze_statistics_campaign.py --plot
"""
import sys, os, glob, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chirp_stubbs_analysis as csa

TOF_SENS = 10.3   # μs/°C


def analyze_wav(wav_path, rep_rate=52.631578947368425, n_template=100):
    """Run Stubbs analysis on a single 30s WAV. Returns summary dict."""
    print(f"  {os.path.basename(os.path.dirname(wav_path))}  ", end='', flush=True)
    try:
        r = csa.analyze(wav_path, rep_rate=rep_rate, n_template=n_template,
                        save_plot=False)
    except Exception as e:
        print(f"ERROR: {e}")
        return None

    dt  = r['delta_tau_sf']
    sig = r['sigma_dt_sf']

    med  = float(np.median(dt))
    mad  = float(np.median(np.abs(dt - med))) * 1.4826
    inl  = np.abs(dt - med) <= 5 * mad
    n_out = int((~inl).sum())

    sigma_dt   = float(np.std(dt[inl]))
    delta_med  = float(np.median(sig[inl]))
    delta_mean = float(np.mean(sig[inl]))

    # Linear drift over the 30s window (μs/s)
    t_eval = (np.arange(len(dt)) + n_template) / rep_rate
    if inl.sum() > 20:
        slope = float(np.polyfit(t_eval[inl], dt[inl], 1)[0])
    else:
        slope = 0.0

    result = dict(
        wav=wav_path,
        n_eval=len(dt), n_outliers=n_out,
        sigma_dt_us=sigma_dt,
        sigma_dt_C=sigma_dt / TOF_SENS,
        delta_med_ns=delta_med * 1e3,
        delta_mean_ns=delta_mean * 1e3,
        drift_us_per_s=slope,
        mean_dt_us=float(np.mean(dt[inl])),
        delta_tau=dt, sigma_dt_sf=sig, t_eval=t_eval, inlier=inl,
    )
    print(f"σ={sigma_dt:.3f} μs  δ_med={delta_med*1e3:.1f} ns  "
          f"drift={slope*1e3:.2f} ns/s  outliers={n_out}/{len(dt)}")
    return result


def print_summary(label, results):
    valid = [r for r in results if r is not None]
    if not valid:
        print(f"  {label}: no valid results"); return

    sigmas = np.array([r['sigma_dt_us']   for r in valid])
    deltas = np.array([r['delta_med_ns']  for r in valid])
    drifts = np.array([r['drift_us_per_s'] for r in valid])

    print(f"\n  {label}  (N={len(valid)} recordings)")
    print(f"    σ(Δτ)   : {sigmas.mean():.3f} ± {sigmas.std():.3f} μs"
          f"  →  {sigmas.mean()/TOF_SENS:.4f} ± {sigmas.std()/TOF_SENS:.4f} °C/ping")
    print(f"    δ median: {deltas.mean():.1f} ± {deltas.std():.1f} ns")
    print(f"    drift   : {drifts.mean()*1e3:.2f} ± {drifts.std()*1e3:.2f} ns/s")
    print(f"    individual σ: " + "  ".join(f"{s:.3f}" for s in sigmas))


def make_plots(enc_results, open_results, out_dir):
    valid_e = [r for r in enc_results  if r is not None]
    valid_o = [r for r in open_results if r is not None]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Panel 1: σ(Δτ) per rep ---
    ax = axes[0]
    xe = np.arange(1, len(valid_e)+1)
    xo = np.arange(1, len(valid_o)+1)
    se = [r['sigma_dt_us'] for r in valid_e]
    so = [r['sigma_dt_us'] for r in valid_o]
    ax.plot(xe, se, 'o-', color='steelblue',  lw=1.4, ms=6, label=f'Enclosure  mean={np.mean(se):.3f} μs')
    ax.plot(xo, so, 's-', color='darkorange', lw=1.4, ms=6, label=f'Open       mean={np.mean(so):.3f} μs')
    ax.axhline(np.mean(se), color='steelblue',  ls='--', lw=1.0, alpha=0.6)
    ax.axhline(np.mean(so), color='darkorange', ls='--', lw=1.0, alpha=0.6)
    ax.set_xlabel('Rep #'); ax.set_ylabel('σ(Δτ) (μs)')
    ax.set_title('Per-rep noise floor σ(Δτ)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # --- Panel 2: δ median per rep ---
    ax = axes[1]
    de = [r['delta_med_ns'] for r in valid_e]
    do = [r['delta_med_ns'] for r in valid_o]
    ax.plot(xe, de, 'o-', color='steelblue',  lw=1.4, ms=6, label=f'Enclosure  mean={np.mean(de):.1f} ns')
    ax.plot(xo, do, 's-', color='darkorange', lw=1.4, ms=6, label=f'Open       mean={np.mean(do):.1f} ns')
    ax.set_xlabel('Rep #'); ax.set_ylabel('δ(Δτ) median (ns)')
    ax.set_title('Instrument precision δ (LS shape-fit)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # --- Panel 3: Δτ overlay (first rep of each) ---
    ax = axes[2]
    if valid_e:
        r = valid_e[0]
        dt_p = r['delta_tau'].copy().astype(float); dt_p[~r['inlier']] = np.nan
        ax.plot(r['t_eval'], dt_p, '-', color='steelblue', lw=0.6, alpha=0.7,
                label='Enclosure rep 1')
    if valid_o:
        r = valid_o[0]
        dt_p = r['delta_tau'].copy().astype(float); dt_p[~r['inlier']] = np.nan
        ax.plot(r['t_eval'], dt_p, '-', color='darkorange', lw=0.6, alpha=0.7,
                label='Open rep 1')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Δτ = τ_R − τ_L (μs)')
    ax.set_title('Example Δτ time series (rep 1 each)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle('a0.3_sm_4ms  —  Enclosure vs Open  (10 reps × 30 s each)', fontsize=11)
    fig.tight_layout()
    out_png = os.path.join(out_dir, 'statistics_campaign_summary.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSummary plot: {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--recordings-dir', default='recordings',
                    help='Root recordings directory (default: recordings/)')
    ap.add_argument('--plot', action='store_true', default=True,
                    help='Save summary PNG (default: on)')
    ap.add_argument('--rep-rate', type=float, default=52.631578947368425)
    ap.add_argument('--n-template', type=int, default=100)
    args = ap.parse_args()

    rec_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           args.recordings_dir)

    # Find enclosure and open recordings
    enc_wavs  = sorted(glob.glob(
        os.path.join(rec_dir, '*stats_enclosure*', 'a0.3_sm_4ms_received.wav')))
    open_wavs = sorted(glob.glob(
        os.path.join(rec_dir, '*stats_open*', 'a0.3_sm_4ms_received.wav')))

    if not enc_wavs and not open_wavs:
        print("No recordings found. Looking for:")
        print(f"  {rec_dir}/*stats_enclosure*/a0.3_sm_4ms_received.wav")
        print(f"  {rec_dir}/*stats_open*/a0.3_sm_4ms_received.wav")
        sys.exit(1)

    print(f"Found {len(enc_wavs)} enclosure + {len(open_wavs)} open recordings\n")

    print("=== ENCLOSURE ===")
    enc_results  = [analyze_wav(w, args.rep_rate, args.n_template) for w in enc_wavs]

    print("\n=== OPEN ===")
    open_results = [analyze_wav(w, args.rep_rate, args.n_template) for w in open_wavs]

    print("\n" + "="*60)
    print_summary('ENCLOSURE', enc_results)
    print_summary('OPEN',      open_results)

    # Ratio
    valid_e = [r for r in enc_results  if r is not None]
    valid_o = [r for r in open_results if r is not None]
    if valid_e and valid_o:
        ratio = np.mean([r['sigma_dt_us'] for r in valid_e]) / \
                np.mean([r['sigma_dt_us'] for r in valid_o])
        print(f"\n  σ ratio (enclosure / open) = {ratio:.2f}×"
              f"  ({'enclosure WORSE' if ratio>1 else 'enclosure BETTER'})")

    if args.plot:
        make_plots(enc_results, open_results,
                   os.path.dirname(os.path.abspath(__file__)))


if __name__ == '__main__':
    main()
