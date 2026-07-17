#!/usr/bin/env python3
"""
Memory-efficient Stubbs analysis for long recordings (>10 min).

Strategy:
  1. Build templates from the first N_TEMPLATE periods (streaming, no full load)
  2. Evaluate remaining periods in CHUNK_PERIODS batches
  3. Store only scalar Δτ values — never hold all period arrays in RAM
  4. Plot the full Δτ time series at the end

Memory usage: O(N_TEMPLATE × period_samples) + O(CHUNK_PERIODS × period_samples)
             ≈ a few hundred MB instead of tens of GB
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
    # Keep as int16 until needed — defers 8GB allocation
    total_samples = len(raw)
    period_n = round(SR / rep)
    total_periods = total_samples // period_n
    print(f"  Duration: {total_samples/SR/60:.1f} min  |  "
          f"Period: {period_n/SR*1000:.1f} ms  |  Total periods: {total_periods:,}")

    # ── Step 1: find adaptive slices using first 2 min (memory-safe) ──────────
    print("\nDetecting period boundaries from first 2 min...")
    probe_s = min(120, total_samples/SR)
    probe_samples = int(probe_s * SR)
    probe_sig = hp(raw[:probe_samples].astype(np.float32) / 32768.0)
    slices_probe = st.adaptive_slice_periods(probe_sig, SR, rep)
    if not slices_probe:
        print("  ERROR: no periods found. Check rep_rate and signal."); return

    # Detect incident/reflected windows from a settled probe period
    i0 = min(10, len(slices_probe)-1)
    iw, rw = st.detect_windows(probe_sig[slices_probe[i0][0]:slices_probe[i0][1]],
                                SR, tof_us=5700.0)
    print(f"  Incident:  [{iw[0]:.1f}, {iw[1]:.1f}] ms")
    print(f"  Reflected: [{rw[0]:.1f}, {rw[1]:.1f}] ms  (sep={rw[0]-iw[0]:.1f} ms)")

    # ── Step 2: build templates from first N_TEMPLATE periods ─────────────────
    N_TMPL = min(args.n_template, len(slices_probe) - 5)
    print(f"\nBuilding templates from {N_TMPL} periods...")

    # Reference for alignment: full period of probe period 0
    ref0 = probe_sig[slices_probe[0][0]:slices_probe[0][1]].copy()
    center_idx = period_n - 1

    # FM3 fix: windowed drift reference (incident window only).
    # Full-period xcorr dominated by noise for low-SNR waveforms → false drift.
    iw_s_d = int(iw[0] / 1000 * SR)
    iw_e_d = min(int(iw[1] / 1000 * SR), period_n)
    ref0_drift = np.zeros(period_n, dtype=np.float32)
    ref0_drift[iw_s_d:iw_e_d] = ref0[iw_s_d:iw_e_d]
    radius_boot = int(0.002 * SR)
    narrow      = int(0.0003 * SR)
    N_BOOT      = min(50, N_TMPL)
    EMA_TAU     = 200.0; alpha_ema = 1.0/EMA_TAU

    shifts_tmpl = np.zeros(N_TMPL)
    ema_rate    = 0.0

    for i in range(1, N_TMPL):
        s_,e_ = slices_probe[i]
        seg = probe_sig[s_:e_]
        xc = correlate(seg, ref0_drift, mode='full')   # FM3: windowed ref

        if i < N_BOOT:
            lo = max(0, center_idx-radius_boot)
            hi = min(len(xc), center_idx+radius_boot)
        else:
            pred_c = center_idx + int(round(ema_rate*i))
            lo = max(0, pred_c-narrow); hi = min(len(xc), pred_c+narrow)

        env = np.abs(hilbert(xc[lo:hi]))
        lp  = int(np.argmax(env)); hw=5
        sl,sh = max(0,lp-hw), min(len(env),lp+hw+1)
        x=np.arange(sl,sh,dtype=float)-lp; y=env[sl:sh]
        shift = float(lp+lo-center_idx)
        if len(y)>=5 and np.ptp(y)>1e-12:
            coeffs=np.polyfit(x,y,2); a,b=coeffs[0],coeffs[1]
            if abs(a)>1e-10:
                dl=float(np.clip(-b/(2*a),-hw,hw))
                shift = float(lp+dl+lo-center_idx)
        shifts_tmpl[i] = shift
        if i>=N_BOOT:
            ema_rate=(1-alpha_ema)*ema_rate+alpha_ema*(shift/i)
        elif i==N_BOOT:
            ema_rate = np.polyfit(np.arange(1,N_BOOT+1),shifts_tmpl[1:N_BOOT+1],1)[0]

    # FM3 drift threshold guard: >0.5 μs/period → detection noise, zero it.
    drift_rate_us = ema_rate * 1e6 / SR
    if abs(drift_rate_us) > 0.5:
        shifts_tmpl[:] = 0.0
        print(f"  FM3 drift zeroed: {drift_rate_us:.2f} μs/period > 0.5 threshold")
    else:
        print(f"  FM3 drift kept: {drift_rate_us:.3f} μs/period")

    T_launch_acc = np.zeros(period_n, dtype=np.float64)
    T_return_acc = np.zeros(period_n, dtype=np.float64)
    for i in range(N_TMPL):
        s_,e_ = slices_probe[i]
        corrected = apply_correction(probe_sig[s_:e_], shifts_tmpl[i])
        T_launch_acc += st.mask_period(corrected, SR, iw)
        T_return_acc += st.mask_period(corrected, SR, rw)
    T_launch = (T_launch_acc / N_TMPL).astype(np.float32)
    T_return = (T_return_acc / N_TMPL).astype(np.float32)
    print(f"  T_launch peak: {np.max(np.abs(T_launch)):.4f}  "
          f"T_return peak: {np.max(np.abs(T_return)):.4f}")

    # Precompute template autocorrelations for LS shape-fit (avoids recomputing per period)
    R_auto_L = correlate(T_launch.astype(float), T_launch.astype(float), mode='full')
    R_auto_R = correlate(T_return.astype(float), T_return.astype(float), mode='full')
    print(f"  R_auto precomputed (len={len(R_auto_L)})")

    # ── Step 3: evaluate all remaining periods in chunks ──────────────────────
    # Uses xcorr_lag_shape_fit (LS shape-fit) — fits the measured CCF to the
    # template autocorrelation shape using carrier phase, not just envelope max.
    # Precomputed R_auto_L / R_auto_R avoid recomputing per period.
    print(f"\nEvaluating {total_periods - N_TMPL:,} periods in chunks of {args.chunk}...")
    tau_l_all = []; tau_r_all = []; eval_idx = []
    sigma_l_all = []; sigma_r_all = []

    CHUNK_P = args.chunk
    start_period = N_TMPL
    eval_count = 0
    global_period = N_TMPL   # monotonically increasing period counter

    while start_period < total_periods:
        end_period = min(start_period + CHUNK_P, total_periods)
        s_raw = start_period * period_n
        e_raw = end_period   * period_n
        chunk_raw = raw[s_raw:e_raw]
        chunk_sig = hp(chunk_raw.astype(np.float32) / 32768.0)

        # Adaptively locate each burst within this chunk — handles any
        # accumulated clock drift transparently.
        chunk_slices = st.adaptive_slice_periods(chunk_sig, SR, rep)
        if not chunk_slices:
            start_period = end_period
            global_period += end_period - start_period
            continue

        for slice_s, slice_e in chunk_slices:
            period = chunk_sig[slice_s:slice_e]
            if len(period) < period_n // 2:
                global_period += 1
                continue

            # Fine alignment: full-period xcorr against ref0 (both burst windows).
            # Using ref0_drift (windowed) here breaks return-window alignment.
            xc = correlate(period, ref0, mode='full')
            lo = max(0, center_idx - radius_boot)
            hi = min(len(xc), center_idx + radius_boot)
            if hi <= lo:
                global_period += 1
                continue
            env = np.abs(hilbert(xc[lo:hi]))
            lp = int(np.argmax(env)); hw = 5
            sl, sh = max(0, lp-hw), min(len(env), lp+hw+1)
            x = np.arange(sl, sh, dtype=float) - lp; y = env[sl:sh]
            shift = float(lp + lo - center_idx)
            if len(y) >= 5 and np.ptp(y) > 1e-12:
                coeffs = np.polyfit(x, y, 2); a, b = coeffs[0], coeffs[1]
                if abs(a) > 1e-10:
                    dl = float(np.clip(-b / (2*a), -hw, hw))
                    shift = float(lp + dl + lo - center_idx)

            corrected = apply_correction(period, shift)
            inc_m = st.mask_period(corrected, SR, iw)
            ref_m = st.mask_period(corrected, SR, rw)

            tl, σl, _, _ = st.xcorr_lag_shape_fit(inc_m, T_launch, SR, R_auto=R_auto_L)
            tr, σr, _, _ = st.xcorr_lag_shape_fit(ref_m, T_return, SR, R_auto=R_auto_R)
            if tl is not None and tr is not None:
                tau_l_all.append(tl); tau_r_all.append(tr)
                sigma_l_all.append(σl); sigma_r_all.append(σr)
                eval_idx.append(global_period)
                eval_count += 1
            global_period += 1

        start_period = end_period
        pct = start_period / total_periods * 100
        if eval_count % 1000 == 0 or start_period == total_periods:
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
    pipeline_dir = os.path.join(out_dir, 'stubbs_pipeline_v2')
    os.makedirs(pipeline_dir, exist_ok=True)
    time_min = np.array(eval_idx) / rep / 60.0

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    fig.suptitle(f'{label}  —  Long-term Δτ  (N={len(dt):,}  '
                 f'σ={dt.std():.3f}μs → {dt.std()/10.3:.4f}°C/ping)',
                 fontweight='bold', fontsize=11)

    # Full time series — clean periods only, standard pipeline format
    from scipy.ndimage import uniform_filter1d
    good_plot = np.abs(dt) <= 5.0
    n_out = int((~good_plot).sum())
    dt_clean = np.where(good_plot, dt, np.nan)
    sp_plot  = np.sqrt(sigma_l**2 + sigma_r**2)
    sp_clean = np.where(good_plot, sp_plot, np.nan)
    mean_c2  = float(np.nanmean(dt_clean)); std_c2 = float(np.nanstd(dt_clean))
    ax = axes[0]
    ax.errorbar(time_min, dt_clean, yerr=sp_clean,
                fmt='o-', ms=1.5, lw=0.6, elinewidth=0.4, capsize=0,
                mfc='none', mec='black', mew=0.5,
                ecolor='black', color='#1a6aff', alpha=0.6, zorder=3,
                label=f'Δτ per period'
                      + (f'  ({n_out} outliers hidden)' if n_out else ''))
    ax.axhline(mean_c2, color='navy', ls='--', lw=1.2, alpha=0.7,
               label=f'mean = {mean_c2:.3f} μs')
    ax.axhline(mean_c2 + std_c2, color='gray', ls=':', lw=1)
    ax.axhline(mean_c2 - std_c2, color='gray', ls=':', lw=1,
               label=f'σ = {std_c2:.3f} μs → {std_c2/10.3:.4f}°C/ping')
    ax.set_xlabel('Time (min)'); ax.set_ylabel('Δτ (μs)')
    ax.set_title('Full time series  (FM3 pipeline)')
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
