#!/bin/bash
# greenhorn, queued 2026-11-15 00:10 JST (user decision 2026-11-14 23:55: Table 6 = full
# protocol, four settings per rule, ISM included): ISM's GRID4 cells
# (alpha {0.1, 0.3} x p {0.7, 0.9}, run_e12_merge_grid.py --budget 4, no --proto) on the
# cached e12_cars chain, 3 seeds, Stanford Cars, after the greenhorn appendix queue
# (scripts/queue_greenhorn_after_e12.sh) logs 'all done'.  ImageNet-R, CUB and
# CIFAR-100 run on bio (scripts/queue_ism_grid_bio.sh).  One experiment per machine.
set -u
export PYTHONUTF8=1 DATA_ROOT=/home/greenhorn/data
R=/home/greenhorn/Desktop/GitHub/spectral_merging
cd "$R" || exit 1
PY=/home/greenhorn/miniconda3/envs/cl/bin/python
OUT=logs_main/e12f_ism_greenhorn.out
log() { echo "$(date '+%F %T') $*" >> "$OUT"; }
log "queued: ISM GRID4 (full protocol) on cars after queue_e_greenhorn 'all done'"
until grep -q "all done$" logs_main/queue_e_greenhorn.out 2>/dev/null; do sleep 120; done
log "greenhorn appendix queue done; starting"
for ds in cars; do
  for attempt in 1 2 3; do
    log "ism $ds: start (attempt $attempt)"
    "$PY" run_e12_merge_grid.py --budget 4 --append --rules ism --ds "$ds" --seeds 1993 1994 1995 \
      >> "logs_main/e12f_greenhorn_ism_${ds}.out" 2>&1
    rc=$?; log "ism $ds: exit $rc"
    [ "$rc" -eq 0 ] && break
    sleep 30
  done
done
log "all done"
