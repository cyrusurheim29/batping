#!/usr/bin/env python3
"""
Memory-efficient Stubbs LS shape-fit analysis for long recordings (>10 min).

Strategy:
  1. Use fixed-period slicing (k × period_n) throughout — same as analyze().
     The burst stays within the window for the full hour because the total
     accumulated clock drift (~9.5 ms for a0.3_sm_4ms) is less than one period.
  2. Maintain a single global EMA drift tracker across all chunks, exactly
     matching the post-bootstrap narrow-window alignment used in analyze().
     This eliminates the ±2ms wide-window jitter of the old per-chunk
     adaptive slicer approach.
  3. Templates built from the first N_TEMPLATE fixed-period windows.
     iw/rw window detection uses the adaptive slicer on just the first
     10 periods (no drift yet — reliable detection, no memory impact).
  4. Evaluate remaining periods in CHUNK_P batches for memory efficiency.
  5. LS shape-fit (xcorr_lag_shape_fit) with precomputed R_auto for speed.

Memory usage: O(CHUNK_P × period_samples) float32  ≈ a few hundred MB.
"""
import sys, os, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert, correlate
from scipy.io import wavfile
sys.path.insert(0, os.path.dirname(__file__))
import chirp_stubbs_analysis as st

SR = 384000
SDG_SR=400000; SDG_MAX=16384

def rr(dur, dead=15.0):
    return SDG_SR / min(int((dur+dead)/1000*SDG_SR), SDG_MAX)

def hp(s):
    sos = butter(8, 30000/(SR/2), btype='high', output='sos')
    return sosfiltfilt(sos, s.astype(np.float32)).astype(np.float32)

def fft_shift(x, s):
    N=len(x); X=np.fft.rfft(x)
    X *= np.exp(-2j*np.pi*np.fft.rfftfreq(N)*s)
    return np.fft.irfft(X, n=N).astype(np.float32)

def linear_shift(x, s):
    n=len(x); s=int(round(s)); out=np.zeros(n, dtype=np.float32)
    if 0<s<n:   out[:-s] = x[s:]
    elif -n<s<0: out[-s:] = x[:n+s]
    else:        out[:] = x
    return out

def apply_correction(period, shift_s):
    si=int(round(shift_s)); sf=shift_s-si
    out=linear_shift(period, -si)
    if abs(sf)>1e-6: out=fft_shift(out, -sf)
    return out

def main():
    parser = argparse.ArgumentParser(description='Long-term Stubbs analysis')
    parser.add_argument('wav', help='Path to WAV file')
    parser.add_argument('--rep-rate', type=float, default=None,
                        help='Rep rate Hz (default: auto from 4ms+15ms=52.63Hz)')
    parser.add_argument('--n-template', type=int, default=100)
    parser.add_argument('--chunk', type=int, default=500,
                        help='Periods per evaluation chunk (default 500)')
    parser.add_argument('--out-dir', default=None,
                        help='Output directory (default: same as WAV)')
    args = parser.parse_args()

    rep = args.rep_rate or rr(4.0, 15.0)
    out_dir = args.out_dir or os.path.dirname(args.wav)
    os.makedirs(out_dir, exist_ok=True)
    label = os.path.splitext(os.path.basename(args.wav))[0]

    print(f"Loading: {os.path.basename(args.wav)}")
    fs, raw = wavfile.read(args.wav)
    assert fs == SR, f"Expected {SR} Hz, got {fs}"
    total_samples = len(raw)
    period_n = round(SR / rep)
    total_periods = total_samples // period_n
    print(f"  Duration: {total_samples/SR/60:.1f} min  |  "
          f"Period: {period_n/SR*1000:.1f} ms  |  Total periods: {total_periods:,}")

    # ── Step 1: detect iw/rw from first 10 periods (adaptive slicer, no drift yet) ──
    print("\nDetecting incident/reflected windows from first 10 periods...")
    probe_samples = 15 * period_n
    probe_sig = hp(raw[:probe_samples].astype(np.float32) / 32768.0)
    slices_probe = st.adaptive_slice_periods(probe_sig, SR, rep)
    if not slices_probe:
        print("  ERROR: no periods found."); return
    i0 = min(5, len(slices_probe) - 1)
    iw, rw = st.detect_windows(probe_sig[slices_probe[i0][0]:slices_probe[i0][1]],
                                SR, tof_us=5700.0)
    print(f"  Incident:  [{iw[0]:.2f}, {iw[1]:.2f}] ms")
    print(f"  Reflected: [{rw[0]:.2f}, {rw[1]:.2f}] ms")

    # ── Step 2: templates from first N_TMPL fixed-period windows + global EMA ──
    # Fixed slicing: period k = raw[k*period_n : (k+1)*period_n].
    # ref0 = first fixed-window period (burst somewhere inside, not necessarily centred).
    # EMA tracks the burst's drift within the fixed window identically to analyze().
    N_TMPL      = args.n_template
    center_idx  = period_n - 1        # lag=0 index in a full-length xcorr
    radius_boot = int(0.002 * SR)     # ±2ms bootstrap search window
    narrow      = int(0.0003 * SR)    # ±0.3ms narrow window (post-bootstrap)
    N_BOOT      = min(50, N_TMPL)
    EMA_TAU     = 200.0; alpha_ema = 1.0 / EMA_TAU

    # Load template chunk (first N_TMPL periods) and HP-filter once
    tmpl_raw = raw[:N_TMPL * period_n].astype(np.float32) / 32768.0
    tmpl_sig = hp(tmpl_raw)
    ref0 = tmpl_sig[:period_n].copy()   # first fixed-window period as reference

    print(f"\nBuilding templates from {N_TMPL} fixed-period windows...")
    shifts_tmpl = np.zeros(N_TMPL); ema_rate = 0.0

    def _align_shift(period, k):
        """Return sub-sample alignment shift for period k using global EMA."""
        xc = correlate(period, ref0, mode='full')
        if k < N_BOOT:
            lo = max(0, center_idx - radius_boot)
            hi = min(len(xc), center_idx + radius_boot)
        else:
            pred_c = int(np.clip(center_idx + round(ema_rate * k),
                                 narrow, len(xc) - narrow - 1))
            lo = max(0, pred_c - narrow)
            hi = min(len(xc), pred_c + narrow)
        if hi <= lo:
            return 0.0
        env = np.abs(hilbert(xc[lo:hi]))
        lp = int(np.argmax(env)); hw = 5
        sl, sh = max(0, lp-hw), min(len(env), lp+hw+1)
        x = np.arange(sl, sh, dtype=float) - lp; y = env[sl:sh]
        shift = float(lp + lo - center_idx)
        if len(y) >= 5 and np.ptp(y) > 1e-12:
            cc = np.polyfit(x, y, 2); a, b = cc[0], cc[1]
            if abs(a) > 1e-10:
                shift = float(lp + np.clip(-b/(2*a), -hw, hw) + lo - center_idx)
        return shift

    for k in range(1, N_TMPL):
        period = tmpl_sig[k*period_n:(k+1)*period_n]
        shift = _align_shift(period, k)
        shifts_tmpl[k] = shift
        if k == N_BOOT:
            ema_rate = np.polyfit(np.arange(1, N_BOOT+1),
                                  shifts_tmpl[1:N_BOOT+1], 1)[0]
        elif k > N_BOOT:
            ema_rate = (1-alpha_ema)*ema_rate + alpha_ema*(shift/k)

    T_launch_acc = np.zeros(period_n, np.float64)
    T_return_acc = np.zeros(period_n, np.float64)
    for k in range(N_TMPL):
        period = tmpl_sig[k*period_n:(k+1)*period_n]
        corr = apply_correction(period, shifts_tmpl[k])
        T_launch_acc += st.mask_period(corr, SR, iw)
        T_return_acc += st.mask_period(corr, SR, rw)
    T_launch = (T_launch_acc / N_TMPL).astype(np.float32)
    T_return = (T_return_acc / N_TMPL).astype(np.float32)
    print(f"  T_launch peak: {np.max(np.abs(T_launch)):.4f}  "
          f"T_return peak: {np.max(np.abs(T_return)):.4f}")
    print(f"  EMA rate after template build: {ema_rate:.4f} samples/period "
          f"= {ema_rate/SR*1e6:.3f} μs/period")

    # Precompute template autocorrelations for LS shape-fit
    R_auto_L = correlate(T_launch.astype(float), T_launch.astype(float), mode='full')
    R_auto_R = correlate(T_return.astype(float), T_return.astype(float), mode='full')

    # ── Step 3: evaluate remaining periods — fixed slicing + global EMA ────────
    # EMA state (ema_rate) persists from template phase through all chunks.
    # Narrow ±0.3ms window used for all evaluation periods (EMA converged).
    print(f"\nEvaluating {total_periods - N_TMPL:,} periods in chunks of {args.chunk}...")
    tau_l_all=[]; tau_r_all=[]; eval_idx=[]; sigma_l_all=[]; sigma_r_all=[]
    CHUNK_P = args.chunk; eval_count = 0

    start_period = N_TMPL
    while start_period < total_periods:
        end_period = min(start_period + CHUNK_P, total_periods)
        chunk_sig = hp(raw[start_period*period_n : end_period*period_n]
                       .astype(np.float32) / 32768.0)

        for j in range(end_period - start_period):
            k = start_period + j          # global period index
            period = chunk_sig[j*period_n:(j+1)*period_n]

            shift = _align_shift(period, k)
            # EMA update (always in narrow phase for eval)
            if k > 0:
                ema_rate = (1-alpha_ema)*ema_rate + alpha_ema*(shift/k)

            corr  = apply_correction(period, shift)
            inc_m = st.mask_period(corr, SR, iw)
            ref_m = st.mask_period(corr, SR, rw)

            tl, σl, _, _ = st.xcorr_lag_shape_fit(inc_m, T_launch, SR, R_auto=R_auto_L)
            tr, σr, _, _ = st.xcorr_lag_shape_fit(ref_m, T_return, SR, R_auto=R_auto_R)
            if tl is not None and tr is not None:
                tau_l_all.append(tl); tau_r_all.append(tr)
                sigma_l_all.append(σl); sigma_r_all.append(σr)
                eval_idx.append(k); eval_count += 1

        start_period = end_period
        pct = start_period / total_periods * 100
        if eval_count % 2000 == 0 or start_period == total_periods:
            print(f"  {pct:.0f}%  eval={eval_count:,}  "
                  f"σ(Δτ)={np.std(np.array(tau_r_all)-np.array(tau_l_all)):.3f}μs",
                  end='\r')

    tau_l   = np.array(tau_l_all)
    tau_r   = np.array(tau_r_all)
    sigma_l = np.array(sigma_l_all)
    sigma_r = np.array(sigma_r_all)
    dt      = tau_r - tau_l
    print(f"\n\nResults:  N={len(dt):,}")
    print(f"  σ(Δτ) = {dt.std():.3f} μs  →  {dt.std()/10.3:.4f} °C/ping")
    print(f"  mean  = {dt.mean():.3f} μs")
    print(f"  range = [{dt.min():.2f}, {dt.max():.2f}] μs  "
          f"= {(dt.max()-dt.min())/10.3:.3f} °C total drift")

    # ── Step 4: plot ──────────────────────────────────────────────────────────
    # Write outputs to stubbs_pipeline_v1/ subdir (LS shape-fit results)
    pipeline_dir = os.path.join(out_dir, 'stubbs_pipeline_v1')
    os.makedirs(pipeline_dir, exist_ok=True)
    time_min = np.array(eval_idx) / rep / 60.0

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    fig.suptitle(f'{label}  —  Long-term Δτ  (N={len(dt):,}  '
                 f'σ={dt.std():.3f}μs → {dt.std()/10.3:.4f}°C/ping)',
                 fontweight='bold', fontsize=11)

    # Full time series
    ax = axes[0]
    ax.plot(time_min, dt, '.', ms=1.2, alpha=0.3, color='k', label='Δτ per period')
    # 1-min rolling mean
    from scipy.ndimage import uniform_filter1d
    smooth_n = int(rep * 60)  # 1-min window
    dt_smooth = uniform_filter1d(dt, size=smooth_n)
    ax.plot(time_min, dt_smooth, '-', lw=1.5, color='C0',
            label=f'1-min rolling mean')
    ax.axhline(dt.mean(), color='C3', lw=1, ls='--', alpha=0.7)
    ax.set_xlabel('Time (min)'); ax.set_ylabel('Δτ (μs)')
    ax.set_title('Full time series')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, ls='--')

    # Histogram
    ax = axes[1]
    dt_c = dt[np.abs(dt-dt.mean())<5*dt.std()]
    ax.hist(dt_c, bins=min(100, len(dt_c)//50), color='#1f77b4',
            edgecolor='white', lw=0.3, alpha=0.85)
    ax.axvline(dt.mean(), color='C3', lw=2, ls='-', label=f'mean={dt.mean():.3f}μs')
    ax.axvline(dt.mean()-dt.std(), color='C3', lw=1.2, ls='--')
    ax.axvline(dt.mean()+dt.std(), color='C3', lw=1.2, ls='--',
               label=f'±1σ={dt.std():.3f}μs')
    ax.set_xlabel('Δτ (μs)'); ax.set_ylabel('Count')
    ax.set_title('Distribution')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, ls='--')

    plt.tight_layout()
    out_png = os.path.join(pipeline_dir, f'stubbs_lsfit.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot: {out_png}")

    # ── hist_delta_tau.png ────────────────────────────────────────────────────
    good = np.abs(dt) <= 5.0
    dt_c = dt[good]; mean_c = dt_c.mean(); std_c = dt_c.std()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f'{label}  —  Δτ between-measurement spread  (N={good.sum():,})',
                 fontsize=12, fontweight='bold')
    ax.hist(dt_c, bins=120, color='#4878cf', edgecolor='white', lw=0.3, alpha=0.85)
    ax.axvline(mean_c, color='C3', lw=2, label=f'mean = {mean_c:.3f} μs')
    ax.axvline(mean_c - std_c, color='C3', lw=1.5, ls='--')
    ax.axvline(mean_c + std_c, color='C3', lw=1.5, ls='--',
               label=f'±1σ = {std_c:.3f} μs → {std_c/10.3:.4f} °C/ping')
    ax.set_xlabel('Δτ (μs)', fontsize=11); ax.set_ylabel('Count', fontsize=11)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, ls='--')
    plt.tight_layout()
    plt.savefig(os.path.join(pipeline_dir, 'hist_delta_tau.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── hist_uncertainties.png ────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    sl_ns = sigma_l[good] * 1000; sr_ns = sigma_r[good] * 1000
    sp_ns = np.sqrt(sl_ns**2 + sr_ns**2)
    # clip extreme lsfit outliers for display
    sl_d = sl_ns[sl_ns < np.percentile(sl_ns, 99)]
    sr_d = sr_ns[sr_ns < np.percentile(sr_ns, 99)]
    sp_d = sp_ns[sp_ns < np.percentile(sp_ns, 99)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'{label}  —  Per-period fit uncertainty propagation\n'
                 r'$\delta_{\Delta\tau}=\sqrt{\delta_L^2+\delta_R^2}$  (LS shape-fit, full 1-hr run)',
                 fontsize=10, fontweight='bold')
    def up(ax, data, label_str, color, xlabel):
        mu, sig = data.mean(), data.std()
        ax.hist(data, bins=80, color=color, edgecolor='white', lw=0.3, alpha=0.85)
        ax.axvline(mu, color='C3', lw=2)
        ax.axvline(mu-sig, color='C3', lw=1.3, ls='--')
        ax.axvline(mu+sig, color='C3', lw=1.3, ls='--')
        ax.set_title(f'{label_str}  (N={len(data):,})', fontsize=9)
        ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel('Count', fontsize=9)
        ax.text(0.97, 0.97, f'mean = {mu:.1f} ns\nσ = {sig:.1f} ns',
                transform=ax.transAxes, fontsize=8, va='top', ha='right',
                family='monospace', bbox=dict(boxstyle='round', fc='white', alpha=0.85, ec='#ccc'))
        ax.legend([Line2D([0],[0],color='C3',lw=2), Line2D([0],[0],color='C3',lw=1.3,ls='--')],
                  [f'mean = {mu:.1f} ns', f'±1σ = {sig:.1f} ns'], fontsize=8)
        ax.grid(True, alpha=0.3, ls='--')
    up(axes[0], sl_d, r'$\delta_{launch}$  (xcorr fit uncertainty)', '#4878cf', 'δ_launch (ns)')
    up(axes[1], sr_d, r'$\delta_{return}$  (xcorr fit uncertainty)', '#6acc65', 'δ_return (ns)')
    up(axes[2], sp_d, r'$\sqrt{\delta_L^2+\delta_R^2}$  (propagated Δτ fit uncertainty)', '#b47cc7',
       r'$\sqrt{\delta_L^2+\delta_R^2}$ (ns)')
    axes[2].axvline(std_c * 1000, color='C1', lw=2, ls=':')
    axes[2].legend([Line2D([0],[0],color='C3',lw=2),
                    Line2D([0],[0],color='C3',lw=1.3,ls='--'),
                    Line2D([0],[0],color='C1',lw=2,ls=':')],
                   [f'mean = {sp_d.mean():.1f} ns', f'±1σ = {sp_d.std():.1f} ns',
                    f'σ(Δτ) meas = {std_c*1000:.0f} ns'], fontsize=7.5)
    plt.tight_layout()
    plt.savefig(os.path.join(pipeline_dir, 'hist_uncertainties.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── per_period.xlsx — 1-second bins ──────────────────────────────────────
    import pandas as pd
    t_sec_arr = np.array(eval_idx)[good] / rep
    sp_all = np.sqrt(sigma_l[good]**2 + sigma_r[good]**2) * 1000
    rows = []
    for i in range(int(t_sec_arr.max()) + 1):
        m = (t_sec_arr >= i) & (t_sec_arr < i + 1)
        if not m.any(): continue
        rows.append({'time_s': i+0.5, 'time_min': round((i+0.5)/60, 4),
                     'n_periods': int(m.sum()),
                     'mean_delta_tau_us': round(float(dt_c[m].mean()), 4),
                     'std_delta_tau_us':  round(float(dt_c[m].std()),  4),
                     'median_delta_tau_us': round(float(np.median(dt_c[m])), 4),
                     'mean_sigma_prop_ns': round(float(sp_all[m].mean()), 2),
                     'median_sigma_prop_ns': round(float(np.median(sp_all[m])), 2)})
    pd.DataFrame(rows).to_excel(os.path.join(pipeline_dir, 'per_period.xlsx'), index=False)
    print(f"Spreadsheet: per_period.xlsx  ({len(rows)} rows, 1-second bins)")

    # Save arrays — include sigma arrays for future use
    out_npz = os.path.join(pipeline_dir, f'stubbs_lsfit.npz')
    np.savez(out_npz, tau_launch=tau_l, tau_return=tau_r, delta_tau=dt,
             sigma_launch=sigma_l, sigma_return=sigma_r,
             eval_idx=np.array(eval_idx), rep_rate=rep)
    print(f"Data: {out_npz}")

if __name__ == '__main__':
    main()
