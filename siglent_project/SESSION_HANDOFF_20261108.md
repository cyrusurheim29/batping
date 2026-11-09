# BATPING Session Handoff — 2026-10 to 2026-11

Session arc: moved from one-way delayed-chirp (two-SDG-channel) → round-trip bidirectional (Ch 2 delayed chirp) → single-channel T-connector setup. Built up the analysis pipeline, chased down several template / stitching / trigger bugs, and started investigating whether flat-amplitude chirps or α=0.3-compensated chirps work better in the new T-connector geometry. Where we stopped: trying to validate the H(f) characterization is physically sound before the next long recording.

## Current hardware state

- **SDG CH1 only**, into a T-connector →
  - (a) SPA1010 power amp ×10 → piezo array
  - (b) PicoScope Ch B (electrical reference, no delay)
- **Ch A**: ECM999 mic → SR560 preamp **G=100** → PicoScope Ch A
- **Ch C**: TMP36 temperature sensor
- Separation: 4 m to corner cube (8 m round trip; TOF ≈ 23.3 ms)
- PicoScope ranges typically: **Ch A ±5V, Ch B ±10V** for recording (but **Ch B ±1V** for characterization — see bug note below)

## Analysis pipeline — current state

### Main scripts (all delayed-chirp family)

| Script | Purpose | Key CLI flags added this session |
|---|---|---|
| `oneway_delayedchirp_analysis.py` | per-WAV matched-filter analysis | `--shape-method {dechirped,classic}`, `--per-chunk-plots` |
| `oneway_combine_segments.py` | stitch per-segment NPZs into one | `--subdir`, `--stitch-xcorr` (boundary-median TOF alignment) |
| `oneway_delta_tau_waterfall.py` | Δτ PSD waterfall + band power | `--subdir`, `--per-segment`; auto-clamps x-axis to Nyquist |
| `oneway_batping_vs_tmp36.py` | overlay batping ΔT vs TMP36 | `--subdir` |
| `oneway_delayedchirp_diff_temp.py` | 2-panel diff-temp traces (both methods) | `--subdir`, `--per-segment` |
| `oneway_delayedchirp_shape_only_plots.py` | **shape-only**: single-panel diff-temp (blue thread, open circles, black errbars), 3-panel σ histograms, waterfall | `--subdir`, `--per-segment` |
| `oneway_delayedchirp_chunk_plots.py` | regenerate 30-s stubbs-style 3-panel chunk plots from existing WAV + NPZ | `--every-min N` |

### Upload scripts

| Script | Description |
|---|---|
| `upload_variable_chirp.py` | single-channel CH1, variable burst-ms + dead-ms, uses H(f) compensation. `--h-file` arg points to NPZ |
| `upload_flat_chirp.py` | single-channel CH1, no amplitude compensation (used for the 2.4-hr tconnector recording) |
| `upload_chirps_delayed_ch2.py` | legacy — two-channel CH1+CH2 delayed |
| `upload_waveform_only.py` | legacy — hardcoded 4ms chirp, single channel |

### H(f) chain characterization

| Script | Description |
|---|---|
| `characterize_chain_tconnector_careful.py` | **primary**: 4-pass randomized sweep, 50 Hz step, 30 ms burst / 50 ms dead / 100 ms settle. CLI: `--n-passes`, `--freq-step`, `--f-lo/f-hi`, `--ampl`, `--settle-frac` |
| `characterize_chain_tconnector.py` | fast 2-min variant (older) |
| `plot_chain_hf.py` | 3-panel plot of averaged H(f) NPZ (freq-sorted H(f) + SNR + per-freq σ/μ) |
| `compare_chain_hf.py` | A/B/C overlay for multipath-vs-piezo test |
| `salvage_chain_tconnector.py` | recovers V_real from auto-trigger-failure NPZ via `V = mean + std²/mean` |
| `scope_chain_at_freq.py` | single-freq diagnostic: drives SDG + captures Ch A/B, overlays pipeline's analysis window vs geometry-correct window |

## Key recordings produced this session

| Recording | Duration | Notes |
|---|---|---|
| `20261019_200200_batping_batping_roundtrip_8m_delayedchirp` | 150 min | Two-SDG delayed-chirp @ 4m sep. σ_HF ≈ 17 mK/ping (good). Earlier σ-vs-reality investigation showed shape σ underestimated 10× and parabolic σ ~1800× underestimated |
| `20261026_165605_batping_batping_tconnector_4m_4ms_chirp` | 7 min | First T-connector test. Multi-arrival in Ch A search window; fixed by narrowing `--search-ms 24 29` |
| `20261107_195236_batping_batping_tconnector_4m_4ms_flat_3hr` | **143.76 min** (streaming stopped due to disk full) | Flat chirp (no α comp). σ_HF reported as 167 mK/ping — but **the acoustic Ch A σ_shape median is 6 ns (≈0.13 mK/ping)** — the big HF number is pulled up by a tail of degenerate shape fits |

## Bugs / issues found + fixed

1. **oneway_delayedchirp_analysis.py — phase-decoherent templates** (fixed Oct 20)
   `build_template` averaged windows aligned only to envelope-peak (integer sample). Carrier-phase decoherence killed amplitude ~4-7×. Fix: `build_template(..., n_iter=2)` adds two passes of sub-sample xcorr alignment + FFT phase-shift realign. Template now reaches ~97% of envelope ceiling.

2. **Between-segment stitching: template-xcorr returned ~0** (fixed Oct 20)
   Each segment's template is envelope-centered → xcorring them gave ~0 lag regardless of real baseline shift. Replaced with **boundary-median alignment**: `jump = median(first 200 inliers of seg N+1) − median(last 200 inliers of seg N)`, cumulative. Flag preserved as `--stitch-xcorr` for backward compatibility.

3. **`oneway_combine_segments.py` globbed `*_seg*`** which also matched the PNGs I later added (fixed Oct 20 — restrict to directories).

4. **`oneway_delta_tau_waterfall.py` x-axis extended to 20 Hz** regardless of rep rate → empty space beyond Nyquist (fixed Oct 22 — dynamically clamps x-axis + HF/conv band labels).

5. **`oneway_combine_segments.py` didn't carry `sigma_B` / `sigma_diff`** forward into the combined NPZ (fixed Nov 7 — shape-only plotter was crashing with KeyError).

6. **Characterization Ch B range override broke the trigger** (fixed Nov 7)
   I'd set Ch B to ±10V FS for the characterization but the inherited `TRIG_MV = 100` + `TRIG_ADC = 0.1 * maxADC` formula assumes Ch B is ±1V FS. At ±10V, threshold became 1 V while SDG drive peak is 150 mV → trigger never fired → fell back to 500 ms auto-trigger at random phase → `find_burst` missed ~half the time → `std/mean ≈ 100%` across ALL freqs (textbook signature of binary success/fail at 50/50). **Fixed by keeping Ch B at ±1V FS** for characterization — 0.3 Vpp SDG drive fits comfortably there.

7. **Uncertainty histogram axes blew up for shape σ_A** (fixed Nov 8)
   p99 of σ_A is 9000× the median because a few % of pings have degenerate shape fits with σ in the microseconds. x-axis bounded by mean±std guides → mostly empty. Fix: `xlim = min(p95×1.1, median + 8·MAD)`; guides that fall off-plot are listed in the legend with `[off-plot]`.

## Open issues / what's half-done

1. **σ_B underestimation on both methods**: in the 150-min delayed-chirp recording, median σ_shape = 73 ns vs actual σ_HF ≈ 750 ns → still **~10× underestimated**. Parabolic σ ~1800× underestimated (uses fit-residual RMS instead of SNR-weighted CRB). Not yet fixed — proposed replacement with actual CRB formula `σ_lag ≈ 1/(2π·B·ρ·√N_eff)`.

2. **Characterization reproducibility is 16% not <5%**. Hardware trigger is confirmed working (no 100%-σ/μ pattern). Remaining variability is physical. Current hypothesis: **multipath interference** between direct corner-cube path and nearby reflections. Evidence:
   - Ripple period in H(f) ≈ 500-1000 Hz → path differences of 30-70 cm
   - Smooth envelope is stable; raw dots have oscillating excursions

   **Test proposed, not yet run**: run characterization at (A) original geometry, (B) mic shifted laterally 5-10 mm, (C) mic restored. If A-B ripples disagree but A-B smoothed envelopes agree → multipath confirmed. See `compare_chain_hf.py`. User was skeptical of smoothing-as-physics-model, so we need this test before trusting any H(f) for α compensation.

3. **Characterization pipeline uses hardcoded `TOF_MS = 5.88` (set for 2m setup)** in its FFT analysis window. For 4m setup the analysis window is `[20.88, 35.88] ms` but acoustic arrival is at ~23.3 ms → catches 2 ms of pre-burst silence + ring-up + ~11 ms of steady state only. This dilutes the FFT amplitude estimate systematically AND likely contributes to the 16% reproducibility problem. **Not yet fixed in `ecm999_stepped_sine.py`** — would need to patch `TOF_MS` to match actual geometry, or auto-detect via `find_burst`. `scope_chain_at_freq.py` shows BOTH the pipeline window and the correct one side-by-side.

4. **Disk full**: 8.8 GB free at session end. The 2.4-hr flat recording used 33 GB. Need to free ≥50 GB before next long recording. Candidates listed in session — the 35 GB `20260908` heattest, 17/16/14/14 GB old heat/baseline recordings are prime candidates. Earlier offered to generate exact `rm` commands for the user's approval.

## Key physics / design findings

1. **Flat (uncompensated) chirp vs α=0.3 compensation**: without compensation, the received Ch A envelope is weighted by H(f)^1 so energy concentrates where H(f) is peaked (~1 ms window of a 4 ms chirp). Matched-filter precision scales as 1/√(T·B), so losing 3 ms of effective duration costs ~2× in σ. With α=0.3, received envelope is weighted by H(f)^0.7 — wider, more usable. **The 2.4-hr flat recording proves it works but σ is degraded.** Need to re-do with properly-characterized H(f) α=0.3 compensation.

2. **Piezo ringdown is small** — <1 ms. Dead time of 30-50 ms is >>50× the time constant. Not a source of issue.

3. **Multi-arrival in the acoustic return**: at 4m setup, the "return" is spread over ~8-10 ms (not just 4 ms). Causes:
   - α=0.3 pre-emphasis shifts envelope-peak toward chirp end (higher H(f) there)
   - Possible floor/ceiling multi-path arrivals ~30-70 cm different from direct corner-cube path
   - This necessitates narrow search windows (`--search-ms 24 29`) to lock onto the dominant return and avoid envelope-argmax jumping between arrivals

4. **ADC range strategy**: with G=100 + ×10 power amp and the 4m geometry, the acoustic return is ~4.4 Vpp on Ch A. Ch A ±5V FS gives 2× headroom and lets the launch (via mic pickup) hard-clip at the ADC — ADC clips recover in 1 sample, so this is clean. Preamp must not clip (SR560 rails ±5V).

## Immediate next steps when resuming

1. **Decide multipath-vs-real-piezo**: run the A/B/C characterization test (see `compare_chain_hf.py` and the plan in-doc)
2. **If multipath confirmed**: use the smoothed H(f) from `chain_tconnector_careful_20261108_164734.npz` for α=0.3 compensation; upload with `upload_variable_chirp.py --burst-ms 4 --dead-ms 30 --ampl 10 --h-file chain_tconnector_careful_20261108_164734.npz`; re-record
3. **If real piezo structure**: investigate narrower chirp band that avoids the ripple nulls, OR consider no-compensation + longer averaging
4. **Fix `TOF_MS` in the characterization pipeline** before trusting any new H(f) number
5. **Clear disk space** — ~50 GB — before next long recording
6. **Fix σ formulas** to use CRB-based uncertainty instead of fit-residual RMS, so error bars match actual scatter

## Important user preferences (observed)

- **Shape-fit only** for electrical launch pings (parabolic doesn't apply to non-parabolic sine-like peaks)
- Plots: single-panel diff-temp with **blue thread + black open circles + black error bars**, **no 1-s rolling overlay**
- Uncertainty histograms: label median + mean + mean±std in **both ns AND mK** units (using the recording-specific tof_sens)
- Dislikes overly complex multi-panel plots when a focused one tells the story
- Values physics-based reasoning over phenomenological fits ("I don't trust the smoothing, there's nothing physics-related about it")
