#!/usr/bin/env bash
# run_oscillation_heatpad.sh
#
# 59-min WALL CLOCK oscillation protocol for CVS heating pad + BATPING + TMP36
#
# --duration 43 = 43 min of audio, which takes ~59 min wall clock (ratio 1.385×).
# Transition times are in WALL CLOCK minutes — watch a real clock.
# The TMP36 CSV saves both elapsed_s (wall clock) and audio_s for alignment.
#
# Wall clock schedule:
#   0:00  → OFF      (baseline)
#   3:00  → HIGH     ← turn pad to HIGH
#  16:00  → OFF      ← turn pad OFF
#  31:00  → HIGH     ← turn pad to HIGH
#  44:00  → OFF      ← turn pad OFF
#  ~59:00 → DONE     (recording ends automatically)

export DYLD_LIBRARY_PATH=/Library/Frameworks/PicoSDK.framework/Libraries/libps5000a

echo ""
echo "============================================================"
echo "  BATPING  —  Heatpad oscillation protocol"
echo "  CVS 12x24 heating pad  |  TMP36 on Ch B"
echo "  Wall clock: ~59 min  |  Audio: ~43 min  |  2× OFF/HIGH cycles"
echo "============================================================"
echo ""
echo "  WALL CLOCK schedule (watch a real clock):"
echo "    t=  0:00  OFF      baseline — pad already OFF"
echo "    t=  3:00  HIGH     turn pad to HIGH"
echo "    t= 16:00  OFF      turn pad OFF"
echo "    t= 31:00  HIGH     turn pad to HIGH"
echo "    t= 44:00  OFF      turn pad OFF"
echo "    t= ~59:00 DONE     recording ends automatically"
echo ""
echo "  RECORDING COMMAND (run this first in your main terminal):"
echo ""
echo "    python3 picoscope_batping_record.py \\"
echo "        --waveform a0.3_sm_4ms \\"
echo "        --duration 43 \\"
echo "        --adc-range 5V \\"
echo "        --ampl 5.9 \\"
echo "        --label heatpad_CVS_enc_osc_take1"
echo ""
echo "  Press Enter here once the recording has started..."
read -r

START=$(date +%s)

alert() {
    local TARGET_MIN=$1
    local MSG=$2
    local TARGET_S=$(( START + TARGET_MIN * 60 ))
    local NOW=$(date +%s)
    local WAIT=$(( TARGET_S - NOW ))
    if [ $WAIT -gt 0 ]; then
        sleep $WAIT
    fi
    local ELAPSED=$(( ($(date +%s) - START) / 60 ))
    echo ""
    echo "  *** t=${TARGET_MIN} min (wall clock)  →  ${MSG} ***"
    echo "      (actual elapsed: ~${ELAPSED} min)"
    osascript -e "display notification \"${MSG}\" with title \"BATPING Heat Protocol\" sound name \"Glass\"" 2>/dev/null || true
}

echo ""
echo "  Timer running. Alerts fire at wall clock transitions."
echo "  (Keep this terminal visible)"
echo ""

alert  3  "TURN PAD TO HIGH  ← now"
alert 16  "TURN PAD OFF       ← now"
alert 31  "TURN PAD TO HIGH  ← now"
alert 44  "TURN PAD OFF       ← now"
alert 59  "RECORDING SHOULD BE DONE — pad already off"

echo ""
echo "  Protocol complete."
