#!/bin/bash
# Full analysis pipeline for a round-trip delayed-chirp recording.
#   - per-segment matched-filter analysis (uses patched iterative-xcorr template)
#   - combine segments into one 150-min NPZ
#   - diff-temp / diff-TOF time series
#   - Δτ waterfall / PSD
#   - batping-vs-TMP36 overlay
#
# Usage:  ./run_delayedchirp_full_analysis.sh <recording_dir>
#
set -e

if [ -z "$1" ]; then
    echo "usage: $0 <recording_dir>"
    exit 1
fi
REC="$1"

if [ ! -d "$REC" ]; then
    echo "no such directory: $REC"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════════════════"
echo " Full delayed-chirp analysis for: $REC"
echo "═══════════════════════════════════════════════════════════════════════"
echo

# ── 1. Per-segment pipelines ──────────────────────────────────────────────
echo "── Phase 1: per-segment delayed-chirp analysis ──"
# Only full-size WAVs: skip stubs < 1 MB (incomplete rotation artifacts)
WAVS=$(find "$REC" -maxdepth 1 -name "*received_seg??.wav" -size +1M | sort)
if [ -z "$WAVS" ]; then
    # Fall back to non-segmented single-WAV case
    WAVS=$(find "$REC" -maxdepth 1 -name "*received.wav" | sort)
fi
NSEG=$(echo "$WAVS" | wc -l | tr -d ' ')
echo "  found $NSEG segment WAV(s)"
I=0
for W in $WAVS; do
    I=$((I+1))
    echo
    echo "  ─── segment $I/$NSEG: $(basename "$W") ───"
    python3 oneway_delayedchirp_analysis.py "$W" \
        --ch-a-fs-v 2 --ch-b-fs-v 5 \
        --rep-rate 29.41 \
        --search-ms 18 28 \
        --chb-delay-ms 2
done

# ── 2. Combine segments ───────────────────────────────────────────────────
echo
echo "── Phase 2: combining segments ──"
python3 oneway_combine_segments.py "$REC" --subdir oneway_delayedchirp_analysis

# ── 3. Diff-temp / diff-TOF time series ───────────────────────────────────
echo
echo "── Phase 3: diff-temp time series ──"
python3 oneway_delayedchirp_diff_temp.py "$REC" --per-segment

# ── 4. Δτ waterfall ───────────────────────────────────────────────────────
echo
echo "── Phase 4: Δτ waterfall / PSD ──"
python3 oneway_delta_tau_waterfall.py "$REC" --subdir oneway_delayedchirp_analysis

# ── 5. BATPING vs TMP36 overlay ───────────────────────────────────────────
echo
echo "── Phase 5: batping-vs-TMP36 overlay ──"
python3 oneway_batping_vs_tmp36.py "$REC" --subdir oneway_delayedchirp_analysis

echo
echo "═══════════════════════════════════════════════════════════════════════"
echo " DONE.  Outputs in:  $REC/oneway_delayedchirp_analysis/"
echo "═══════════════════════════════════════════════════════════════════════"
ls -la "$REC/oneway_delayedchirp_analysis/"*.png 2>/dev/null | awk '{print "  " $NF}'
