#!/bin/bash
# E12 FULL-PROTOCOL grid (user decision 2026-11-12 19:10 JST): the same 4 cells per
# method and dataset as the proxy pass (run_e12_merge_grid.py GRID4), but evaluated
# with the paper's protocol (spectral/other merge + drift map + LCA alignment), 3 seeds,
# on the cached e12_<ds> chains.  ISM is not re-run (its full-protocol numbers exist).
#
#   bash scripts/e12f_machine.sh bio         # nufilt ties knots modelstock avg
#   bash scripts/e12f_machine.sh greenhorn   # dare della breadcrumbs max_abs
#
# Each (method, dataset) is one `run_e12_merge_grid.py --budget 4` call (no --proto);
# resume-safe.  Progress: logs_main/e12f_<machine>.out; cells log to logs_main/<ds>/e12_*.log
set -u
export PYTHONUTF8=1
MACHINE=${1:?bio|greenhorn}
cd "$(dirname "$0")/.." || exit 1
case "$MACHINE" in
  bio)       METHODS="nufilt ties knots modelstock"; PY=/home/bio/anaconda3/envs/cl/bin/python
             export DATA_ROOT=/home/bio/data ;;
  greenhorn) METHODS="dare della breadcrumbs max_abs avg"; PY=/home/greenhorn/miniconda3/envs/cl/bin/python
             export DATA_ROOT=/home/greenhorn/data ;;
  *) echo "unknown machine $MACHINE"; exit 1 ;;
esac
DATASETS="imagenetr cub cifar224 cars"
SEEDS="1993 1994 1995"
OUT=logs_main/e12f_${MACHINE}.out
log() { echo "$(date '+%F %T') $*" >> "$OUT"; }

log "start: full protocol, methods [$METHODS] datasets [$DATASETS] seeds [$SEEDS]"
for m in $METHODS; do
  for ds in $DATASETS; do
    for attempt in 1 2 3; do
      log "$m $ds: start (attempt $attempt)"
      "$PY" run_e12_merge_grid.py --budget 4 --append --rules "$m" --ds "$ds" --seeds $SEEDS \
        >> "logs_main/e12f_${MACHINE}_${m}_${ds}.out" 2>&1
      rc=$?; log "$m $ds: exit $rc"
      [ "$rc" -eq 0 ] && break
      sleep 30
    done
  done
done
log "all done"
