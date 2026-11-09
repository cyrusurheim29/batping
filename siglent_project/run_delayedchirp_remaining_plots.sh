#!/bin/bash
# Fills in the two missing analysis pieces for an already-processed
# delayed-chirp recording:
#   1. between-segment xcorr template stitching
#   2. per-30-s chunk plots (templates / example CCF / 30-s time series)
#   3. per-segment diff-temp traces
#   4. re-run downstream plots on the stitched combined NPZ
#
# Assumes you have ALREADY run `./run_delayedchirp_full_analysis.sh <rec>`
# so per-segment NPZs exist.
#
# Usage:  ./run_delayedchirp_remaining_plots.sh <recording_dir>
set -e

if [ -z "$1" ]; then
    echo "usage: $0 <recording_dir>"
    exit 1
fi
REC="$1"

echo "═══════════════════════════════════════════════════════════════════════"
echo " Remaining-plots pass for: $REC"
echo "═══════════════════════════════════════════════════════════════════════"
echo

# ── 1. Re-combine WITH between-segment xcorr template stitching ──────────
echo "── Phase 1: xcorr-stitched combined NPZ ──"
python3 oneway_combine_segments.py "$REC" \
    --subdir oneway_delayedchirp_analysis \
    --stitch-xcorr \
    --rep-rate 29.41 --search-ms 18 28 --ch-a-fs-v 2 --ch-b-fs-v 5

# ── 2. Per-segment + combined diff-temp traces ───────────────────────────
echo
echo "── Phase 2: per-segment + combined diff-temp traces ──"
python3 oneway_delayedchirp_diff_temp.py "$REC" --per-segment

# ── 3. Waterfall + TMP36 overlay on the stitched combined NPZ ────────────
echo
echo "── Phase 3: waterfall (stitched) ──"
python3 oneway_delta_tau_waterfall.py "$REC" --subdir oneway_delayedchirp_analysis

echo
echo "── Phase 4: batping-vs-TMP36 overlay (stitched) ──"
python3 oneway_batping_vs_tmp36.py "$REC" --subdir oneway_delayedchirp_analysis

# ── 5. Per-30-s chunk LS-fit 3-panel plots ──────────────────────────────
echo
echo "── Phase 5: 30-s chunk plots (templates / example CCF / Δτ) ──"
python3 oneway_delayedchirp_chunk_plots.py "$REC" \
    --every-min 2.0 \
    --rep-rate 29.41 --search-ms 18 28 --ch-a-fs-v 2 --ch-b-fs-v 5

echo
echo "═══════════════════════════════════════════════════════════════════════"
echo " DONE."
echo "   stitched NPZ         : $REC/oneway_delayedchirp_analysis/per_ping_combined.npz"
echo "   combined diff-temp   : $REC/oneway_delayedchirp_analysis/diff_temp_timeseries.png"
echo "   per-seg diff-temp    : $REC/oneway_delayedchirp_analysis/diff_temp_timeseries_seg*.png"
echo "   30-s chunk plots     : $REC/oneway_delayedchirp_analysis/*_seg*/chunk_plots/*.png"
echo "═══════════════════════════════════════════════════════════════════════"
