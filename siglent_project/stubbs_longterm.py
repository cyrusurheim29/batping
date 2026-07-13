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
    radius_boot = int(0.002 * SR)
    narrow      = int(0.0003 * SR)
    N_BOOT      = min(50, N_TMPL)
    EMA_TAU     = 200.0; alpha_ema = 1.0/EMA_TAU

    shifts_tmpl = np.zeros(N_TMPL)
    ema_rate    = 0.0

    for i in range(1, N_TMPL):
        s_,e_ = slices_probe[i]
        seg = probe_sig[s_:e_]
        xc = correlate(seg, ref0, mode='full')

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

    # ── Step 3: evaluate all remaining periods in chunks ──────────────────────
    # Use adaptive slicing per chunk so accumulated EMA drift never goes out of
    # bounds.  After adaptive extraction the burst sits near center_idx, so
    # alignment only needs a small ±narrow search — no cumulative prediction.
    print(f"\nEvaluating {total_periods - N_TMPL:,} periods in chunks of {args.chunk}...")
    tau_l_all = []; tau_r_all = []; eval_idx = []

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

            # Fine alignment: xcorr with ref0, small window around center_idx
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

            tl, _, _, _ = st.xcorr_lag_us(inc_m, T_launch, SR)
            tr, _, _, _ = st.xcorr_lag_us(ref_m, T_return, SR)
            if tl is not None and tr is not None:
                tau_l_all.append(tl); tau_r_all.append(tr)
                eval_idx.append(global_period)
                eval_count += 1
            global_period += 1

        start_period = end_period
        pct = start_period / total_periods * 100
        if eval_count % 1000 == 0 or start_period == total_periods:
            print(f"  {pct:.0f}%  eval={eval_count:,}  "
                  f"σ(Δτ)={np.std(np.array(tau_r_all)-np.array(tau_l_all)):.3f}μs",
                  end='\r')

    tau_l = np.array(tau_l_all)
    tau_r = np.array(tau_r_all)
    dt    = tau_r - tau_l
    print(f"\n\nResults:  N={len(dt):,}")
    print(f"  σ(Δτ) = {dt.std():.3f} μs  →  {dt.std()/10.3:.4f} °C/ping")
    print(f"  mean  = {dt.mean():.3f} μs")
    print(f"  range = [{dt.min():.2f}, {dt.max():.2f}] μs  "
          f"= {(dt.max()-dt.min())/10.3:.3f} °C total drift")

    # ── Step 4: plot ──────────────────────────────────────────────────────────
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
    out_png = os.path.join(out_dir, f'stubbs_longterm_{label}.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot: {out_png}")

    # Save arrays
    out_npz = os.path.join(out_dir, f'stubbs_longterm_{label}.npz')
    np.savez(out_npz, tau_launch=tau_l, tau_return=tau_r,
             delta_tau=dt, eval_idx=np.array(eval_idx), rep_rate=rep)
    print(f"Data: {out_npz}")

if __name__ == '__main__':
    main()
