#!/usr/bin/env python3
"""Shape-fit-only summary plots for a delayed-chirp recording.

For each target NPZ (combined, optionally also per-segment), produces:

  diff_temp_timeseries_shape_only[_<tag>].png
    Single-panel diff-temp (ΔT in mK, Δτ in μs on twin axis), shape-fit ONLY.
    Blue thread, black open circles, black error bars.  NO rolling-mean.

  uncertainty_histograms_shape_only[_<tag>].png
    Three panels: σ_A (acoustic), σ_B (electrical), σ_diff.  Shape-fit ONLY.
    Each panel labels median, mean, and mean±std in ns AND mK.

  delta_tau_waterfall_shape_only[_<tag>].png
    2×2 PSD waterfall using shape-fit TOF.  X-axis clamped at Nyquist.

Usage:
    python3 oneway_delayedchirp_shape_only_plots.py <recording_dir> \\
        [--subdir oneway_delayedchirp_analysis_classic] [--per-segment]
"""
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import welch

if len(sys.argv) < 2:
    sys.exit("usage: oneway_delayedchirp_shape_only_plots.py <recording_dir> "
             "[--subdir <name>] [--per-segment]")
REC = sys.argv[1]
SUBDIR = 'oneway_delayedchirp_analysis_classic'
if '--subdir' in sys.argv:
    SUBDIR = sys.argv[sys.argv.index('--subdir') + 1]
PER_SEGMENT = '--per-segment' in sys.argv

# ── Collect targets ──────────────────────────────────────────────────────
targets = []  # list of (npz_path, label, out_dir, name_tag)
combined = os.path.join(REC, SUBDIR, 'per_ping_combined.npz')
if os.path.exists(combined):
    targets.append((combined, 'combined', os.path.join(REC, SUBDIR), ''))
else:
    # if no combined, treat the newest per-segment NPZ as the primary
    cands = sorted(glob.glob(os.path.join(REC, SUBDIR, '*', 'per_ping.npz')))
    if cands:
        targets.append((cands[0], 'single', os.path.join(REC, SUBDIR), ''))

if PER_SEGMENT:
    for p in sorted(glob.glob(os.path.join(REC, SUBDIR, '*_seg*', 'per_ping.npz'))):
        seg_dir = os.path.dirname(p)
        tag = os.path.basename(seg_dir)
        seg_num = tag.split('_seg')[-1][:2] if '_seg' in tag else '00'
        targets.append((p, f'seg{seg_num}', seg_dir, f'_seg{seg_num}'))

if not targets:
    sys.exit(f"no per_ping NPZ in {REC}/{SUBDIR}/")

print(f"Will process {len(targets)} target(s):")
for p, lbl, od, tag in targets:
    print(f"  {lbl:>10s}  {p}")


# ═══════════════════════════════════════════════════════════════════════════
def make_plots(NPZ, label, out_dir, name_tag):
    print(f"\n── {label} — {NPZ}")
    d = np.load(NPZ)
    t_s        = d['t_pulse_s']
    tof_shape  = d['tof_shape_us']
    sig_shape  = d['sigma_shape_us']
    sig_B      = d['sigma_B_shape_us']    if 'sigma_B_shape_us'    in d.files else np.zeros_like(sig_shape)
    sig_diff   = d['sigma_diff_shape_us'] if 'sigma_diff_shape_us' in d.files else np.zeros_like(sig_shape)
    inl        = d['inl_shape']
    tof_sens   = float(d['tof_sens_us_per_deg']) if 'tof_sens_us_per_deg' in d.files else 48.0
    rep        = float(d['rep_rate'])            if 'rep_rate'            in d.files else (1.0 / np.median(np.diff(t_s)))

    baseline   = float(np.mean(tof_shape[inl][:100]))
    dtau_us    = tof_shape - baseline
    dT_mK      = -dtau_us / tof_sens * 1000.0
    sig_mK     = sig_shape / tof_sens * 1000.0
    t_min      = t_s / 60.0

    sl, ic, r, _, _ = stats.linregress(t_s[inl], dT_mK[inl])
    drift_mK_per_min = sl * 60.0
    win_1s   = max(3, int(rep) | 1)
    rolling  = pd.Series(np.where(inl, dT_mK, np.nan)).rolling(win_1s, center=True, min_periods=win_1s//2).mean().values
    sigma_hf = float(np.nanstd(dT_mK[inl] - rolling[inl]))
    sigma_full = float(np.nanstd(dT_mK[inl]))

    # ── (A) Single-panel diff-temp (blue thread, black open-circles + err) ──
    fig, ax = plt.subplots(1, 1, figsize=(17, 7))
    ax.plot(t_min[inl], dT_mK[inl], '-', color='#1f77b4', lw=0.3, alpha=0.6,
            zorder=1, label='shape-fit thread')
    ax.errorbar(t_min[inl], dT_mK[inl], yerr=sig_mK[inl],
                fmt='o', mfc='none', mec='k', ecolor='k',
                ms=1.1, mew=0.2, elinewidth=0.15, alpha=0.6, zorder=2,
                label=f'per-ping ({inl.sum():,} pings, δ_fit error bars)')
    ax.axhline(0, color='k', lw=0.4, alpha=0.4)
    ax.set_xlabel('Time (min)', fontsize=12)
    ax.set_ylabel('ΔT (mK)   —   shape-fit', fontsize=12)
    net_mK = dT_mK[inl][-1] - dT_mK[inl][0]
    ax.set_title(f'Shape-fit ΔT from diff-TOF   [{label}]  '
                 f'σ_HF = {sigma_hf:.2f} mK/ping,  σ_full-run = {sigma_full:.1f} mK,  '
                 f'drift = {drift_mK_per_min:+.3f} mK/min (r²={r**2:.3f}),  '
                 f'net = {net_mK:+.1f} mK\n'
                 f'baseline TOF = {baseline:.1f} μs,  tof_sens = {tof_sens:.2f} μs/°C   —   '
                 f'{os.path.basename(REC)}',
                 fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    ax2 = ax.twinx()
    ylo, yhi = ax.get_ylim()
    ax2.set_ylim(ylo/1000 * -tof_sens, yhi/1000 * -tof_sens)
    ax2.set_ylabel('Δτ (μs)', fontsize=10, color='#666')
    ax2.tick_params(colors='#666')
    fig.tight_layout()
    out_a = os.path.join(out_dir, f'diff_temp_timeseries_shape_only{name_tag}.png')
    fig.savefig(out_a, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_a}")

    # ── (B) Three-panel σ histograms ──
    def _stat_block(arr_us, label_name, color, ax):
        v_ns = arr_us[inl] * 1e3
        if not np.any(v_ns > 0):
            ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                    transform=ax.transAxes); return
        med  = float(np.median(v_ns))
        mu   = float(np.mean(v_ns))
        sd   = float(np.std(v_ns))
        # Robust x-range: p5–p95 of the distribution, further capped by
        # median ± 8·MAD so a heavy right tail (common for degenerate shape fits
        # on low-SNR pings) doesn't blow up the plot.
        lo_pct, hi_pct = np.percentile(v_ns, [5, 95])
        mad = float(np.median(np.abs(v_ns - med))) * 1.4826
        x_lo = max(0.0, min(lo_pct, med - 8 * mad))
        x_hi = min(hi_pct * 1.1, med + 8 * mad)
        if x_hi <= x_lo:        # degenerate (all values equal)
            x_hi = x_lo + 1.0
        # how much data is off-screen (right tail)
        frac_above = float(np.mean(v_ns > x_hi))
        bins = np.linspace(x_lo, x_hi, 120)
        ax.hist(v_ns, bins=bins, color=color, alpha=0.75, edgecolor='k', lw=0.3)

        def _lbl(name, val_ns):
            return f'{name} = {val_ns:7.2f} ns   ({val_ns/tof_sens:+.3f} mK)'

        # Draw guides only when they fall inside the data range; otherwise,
        # list them in the legend text with an "off-plot" annotation.
        def _guide(x, color, ls, lw, label, alpha=1.0):
            if x_lo <= x <= x_hi:
                ax.axvline(x, color=color, ls=ls, lw=lw, alpha=alpha, label=label)
            else:
                ax.plot([], [], ls=ls, color=color, lw=lw, alpha=alpha,
                        label=f'{label}  [off-plot]')

        _guide(med, 'k',       '--', 1.2, _lbl('median', med))
        _guide(mu,  '#1f77b4', '-',  1.2, _lbl('mean',   mu))
        _guide(mu-sd, '#1f77b4', ':', 0.8,
               f'mean ± std   std = {sd:.2f} ns  ({sd/tof_sens:.3f} mK)', alpha=0.6)
        _guide(mu+sd, '#1f77b4', ':', 0.8, '_nolegend_', alpha=0.6)

        ax.set_xlim(x_lo, x_hi)
        ax.set_xlabel(f'{label_name} (ns)')
        ax.set_ylabel('count')
        tail_note = f'   ({frac_above*100:.1f}% of pings off-plot in right tail; p99 = {np.percentile(v_ns, 99):.0f} ns)' if frac_above > 0.005 else ''
        ax.set_title(f'{label_name}   N = {len(v_ns):,}{tail_note}',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(alpha=0.3)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    _stat_block(sig_shape, 'σ_A (acoustic return)',   '#e6444d', axes[0])
    _stat_block(sig_B,     'σ_B (electrical launch)', '#88aa44', axes[1])
    _stat_block(sig_diff,  'σ_diff = √(σ_A² + σ_B²)', '#4477cc', axes[2])
    fig.suptitle(f'Shape-fit σ histograms   [{label}]   —   '
                 f'tof_sens = {tof_sens:.2f} μs/°C   (1 ns ≈ {1/tof_sens:.4f} mK)   —   '
                 f'{os.path.basename(REC)}',
                 fontsize=11, fontweight='bold', y=1.00)
    fig.tight_layout()
    out_b = os.path.join(out_dir, f'uncertainty_histograms_shape_only{name_tag}.png')
    fig.savefig(out_b, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_b}")

    # ── (C) Δτ waterfall (shape-fit, Nyquist-clamped) ──
    WIN_S         = 5.0
    HF_BAND_HZ    = (0.5, 20.0)
    CONV_BAND_HZ  = (0.3, 5.0)
    MAD_CLIP_K    = 5.0

    med_dtau = float(np.median(dtau_us))
    mad      = float(np.median(np.abs(dtau_us - med_dtau))) * 1.4826
    inlier   = inl & (np.abs(dtau_us - med_dtau) <= MAD_CLIP_K * mad)
    if inlier.sum() < 20:
        print(f"  skip waterfall (only {inlier.sum()} inliers)")
        return

    t_uniform    = np.arange(t_s[0], t_s[-1], 1.0 / rep)
    dtau_uniform = np.interp(t_uniform, t_s[inlier], dtau_us[inlier])
    roll_1s = pd.Series(dtau_uniform).rolling(win_1s, center=True, min_periods=win_1s//2).mean().values
    dtau_hf = dtau_uniform - np.where(np.isfinite(roll_1s), roll_1s, 0.0)

    win_samps = int(WIN_S * rep)
    step_samps = win_samps
    n_wins = (len(dtau_hf) - win_samps) // step_samps + 1
    if n_wins < 2:
        print(f"  skip waterfall (only {n_wins} window)")
        return
    psds, win_t_min = [], []
    for i in range(n_wins):
        i0 = i * step_samps
        seg = dtau_hf[i0:i0+win_samps]
        freqs, p = welch(seg, fs=rep, nperseg=min(win_samps, len(seg)),
                          detrend='constant', scaling='density')
        psds.append(p)
        win_t_min.append((t_uniform[i0] + (win_samps / rep) / 2) / 60.0)
    psds   = np.array(psds)
    win_t_min = np.array(win_t_min)
    median_psd = np.median(psds, axis=0)

    nyquist_hz = rep / 2.0
    hf_lo = min(HF_BAND_HZ[0], nyquist_hz);   hf_hi = min(HF_BAND_HZ[1], nyquist_hz)
    cv_lo = min(CONV_BAND_HZ[0], nyquist_hz); cv_hi = min(CONV_BAND_HZ[1], nyquist_hz)
    hf_mask   = (freqs >= hf_lo) & (freqs <= hf_hi)
    conv_mask = (freqs >= cv_lo) & (freqs <= cv_hi)
    hf_pow_t   = np.trapz(psds[:, hf_mask],   freqs[hf_mask],   axis=1)
    conv_pow_t = np.trapz(psds[:, conv_mask], freqs[conv_mask], axis=1)
    psd_db = 10 * np.log10(psds + 1e-20)
    L_m = baseline * 1e-6 * 343.0

    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    ax = axes[0, 0]
    p5, p95 = np.percentile(psd_db, [5, 95])
    im = ax.imshow(psd_db, aspect='auto', origin='lower',
                   extent=[freqs[0], nyquist_hz, win_t_min[0], win_t_min[-1]],
                   cmap='inferno', vmin=p5, vmax=p95)
    for b_lo, b_hi, c in [(hf_lo, hf_hi, 'cyan'), (cv_lo, cv_hi, 'yellow')]:
        for x in (b_lo, b_hi):
            ax.axvline(x, color=c, ls='--', lw=0.6, alpha=0.6)
    ax.set_xlim(freqs[0], nyquist_hz)
    ax.set_xlabel(f'Frequency (Hz)   [Nyquist = {nyquist_hz:.2f} Hz at rep {rep:.2f} Hz]')
    ax.set_ylabel('Time (min)')
    ax.set_title(f'Δτ residual PSD waterfall  ({WIN_S:.0f}-s windows, 1-s detrended, shape-fit)')
    plt.colorbar(im, ax=ax, label='PSD (dB μs²/Hz)')

    ax = axes[0, 1]
    ax.loglog(freqs, median_psd, '-', color='#1f77b4', lw=1.4, label='shape-fit')
    ax.axvspan(hf_lo, hf_hi, alpha=0.08, color='#4477cc', label=f'HF turb {hf_lo:.1f}-{hf_hi:.1f} Hz')
    ax.axvspan(cv_lo, cv_hi, alpha=0.12, color='#e6444d', label=f'convection {cv_lo:.1f}-{cv_hi:.1f} Hz')
    ax.set_xlim(freqs[1], nyquist_hz)
    ax.set_xlabel(f'Frequency (Hz)   [Nyquist = {nyquist_hz:.2f} Hz]')
    ax.set_ylabel('median PSD (μs²/Hz)')
    ax.set_title('Median PSD across all windows')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9, loc='upper right')

    ax = axes[1, 0]
    ax.plot(win_t_min, hf_pow_t, '-', color='#4477cc', lw=1.2)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel(f'HF band power ({hf_lo:.1f}-{hf_hi:.1f} Hz) (μs²)')
    ax.set_title(f'Broadband acoustic turbulence vs time (shape-fit)  mean {np.mean(hf_pow_t):.4f} μs²')
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(win_t_min, conv_pow_t, '-', color='#e6444d', lw=1.2)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel(f'Convection band power ({cv_lo:.1f}-{cv_hi:.1f} Hz) (μs²)')
    ax.set_title(f'Thermal convection signature vs time (shape-fit)  mean {np.mean(conv_pow_t):.4f} μs²')
    ax.grid(alpha=0.3)

    fig.suptitle(f'Δτ waterfall (shape-fit)   [{label}]   —   {os.path.basename(REC)}   '
                 f'L ≈ {L_m*100:.0f} cm,  dτ/dT = {tof_sens:.2f} μs/°C,  '
                 f'rep = {rep:.2f} Hz,  N = {len(t_s):,} pings',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    out_c = os.path.join(out_dir, f'delta_tau_waterfall_shape_only{name_tag}.png')
    fig.savefig(out_c, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_c}")


for npz, lbl, od, tag in targets:
    make_plots(npz, lbl, od, tag)
