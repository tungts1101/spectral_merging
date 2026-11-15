#!/bin/bash
# E12 ISM extension queue, one machine (user plan 2026-11-12): waits until this
# machine's e12p driver has logged 'all done', then runs run_e12_ism_ext.py items
# in order.  Each item = (dataset, rank, stage); stage 'both' trains the rank chain
# with the full-protocol ISM cell and then runs the 10-cell proxy grid, 'grid' only
# runs the proxy cells (r 64: the 6 new alpha cells on the existing chain).
#
#   bash scripts/e12r_machine.sh bio         # cars r16, cars r32, cars r64 grid
#   bash scripts/e12r_machine.sh greenhorn   # cub r16, r32, r128, cub r64 grid, then cars r128
#
# Progress: logs_main/e12r_<machine>.out; per item logs_main/e12r_<machine>_<ds>_r<r>.out
set -u
export PYTHONUTF8=1
MACHINE=${1:?bio|greenhorn}
cd "$(dirname "$0")/.." || exit 1
case "$MACHINE" in
  bio)       PY=/home/bio/anaconda3/envs/cl/bin/python; export DATA_ROOT=/home/bio/data
             ITEMS="cars:16:both cars:32:both cars:64:grid" ;;
  greenhorn) PY=/home/greenhorn/miniconda3/envs/cl/bin/python; export DATA_ROOT=/home/greenhorn/data
             ITEMS="cub:16:both cub:32:both cub:128:both cub:64:grid cars:128:both" ;;
  *) echo "unknown machine $MACHINE"; exit 1 ;;
esac
SEEDS="1993 1994 1995"
OUT=logs_main/e12r_${MACHINE}.out
log() { echo "$(date '+%F %T') $*" >> "$OUT"; }

log "queued: $ITEMS; waiting for e12p_${MACHINE}.out to log 'all done'"
until grep -q "all done" "logs_main/e12p_${MACHINE}.out" 2>/dev/null; do sleep 120; done
log "e12p driver finished; starting the ISM extension"
for item in $ITEMS; do
  IFS=: read -r ds r stage <<< "$item"
  log "$ds r$r $stage: start"
  "$PY" run_e12_ism_ext.py --ds "$ds" --rank "$r" --stage "$stage" --seeds $SEEDS \
    >> "logs_main/e12r_${MACHINE}_${ds}_r${r}.out" 2>&1
  log "$ds r$r $stage: exit $?"
done
log "all done"
