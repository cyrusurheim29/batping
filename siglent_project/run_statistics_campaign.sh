#!/usr/bin/env bash
# Statistics campaign: 10× 30s recordings of a0.3_sm_4ms
# Run this script twice:
#   bash run_statistics_campaign.sh enclosure
#   bash run_statistics_campaign.sh open
#
# Each rep is saved to a timestamped subdirectory automatically.

set -e
CONDITION=${1:-enclosure}   # 'enclosure' or 'open'
WAVEFORM="a0.3_sm_4ms"
REPS=10
DURATION=0.5                # minutes = 30 seconds

export DYLD_LIBRARY_PATH=/Library/Frameworks/PicoSDK.framework/Libraries/libps5000a

echo ""
echo "============================================================"
echo "  BATPING statistics campaign"
echo "  Waveform : $WAVEFORM"
echo "  Condition: $CONDITION"
echo "  Reps     : $REPS × ${DURATION} min (30 s each)"
echo "============================================================"
echo ""

if [ "$CONDITION" = "enclosure" ]; then
    echo ">>> Enclosure should be IN PLACE. Press Enter when ready."
else
    echo ">>> Enclosure should be REMOVED. Press Enter when ready."
fi
read -r

for i in $(seq 1 $REPS); do
    echo ""
    echo "--- Rep $i / $REPS ---"
    python3 picoscope_batping_record.py \
        --waveform "$WAVEFORM" \
        --duration "$DURATION" \
        --adc-range 5V \
        --ampl 5.9 \
        --label "stats_${CONDITION}"
    echo "Rep $i done."
done

echo ""
echo "============================================================"
echo "  All $REPS reps complete for condition: $CONDITION"
echo "============================================================"
