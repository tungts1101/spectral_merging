#!/bin/bash
# E12 proxy grid, one machine: runs its methods one after another, each method on
# all four datasets (ImageNet-R, CUB, CIFAR-100, Cars) before the next method, so a
# method's four numbers come from one machine, one environment, one code state.
#
#   bash scripts/e12p_machine.sh bio         # ism nufilt ties knots modelstock
#   bash scripts/e12p_machine.sh greenhorn   # dare della breadcrumbs avg max_abs
#
# Each (method, dataset) is one `run_e12_merge_grid.py --proto --budget 4` call
# (4 cells x 3 seeds; resume-safe, finished cells are skipped).  A dataset whose
# chain (checkpoints_main/rebuttal/e12_<ds>) is not complete yet is retried later:
# the loop over datasets repeats until every (method, dataset) has run.  On bio the
# Cars chain is pulled from greenhorn once greenhorn's watcher reports it complete
# (/tmp/kill_after_cars.out contains 'killed').  Progress: logs_main/e12p_<machine>.out
set -u
export PYTHONUTF8=1
MACHINE=${1:?bio|greenhorn}
cd "$(dirname "$0")/.." || exit 1
case "$MACHINE" in
  bio)       METHODS="ism nufilt ties knots modelstock"; PY=/home/bio/anaconda3/envs/cl/bin/python
             export DATA_ROOT=/home/bio/data ;;
  greenhorn) METHODS="dare della breadcrumbs avg max_abs"; PY=/home/greenhorn/miniconda3/envs/cl/bin/python
             export DATA_ROOT=/home/greenhorn/data ;;
  *) echo "unknown machine $MACHINE"; exit 1 ;;
esac
DATASETS="imagenetr cub cifar224 cars"
SEEDS="1993 1994 1995"
OUT=logs_main/e12p_${MACHINE}.out
GH=greenhorn@greenhorn.local
GH_REPO=/home/greenhorn/Desktop/GitHub/spectral_merging
log() { echo "$(date '+%F %T') $*" >> "$OUT"; }

chain_complete() {   # 3 seeds x (11 backbone + 10 classifier) files
  local d=checkpoints_main/rebuttal/e12_$1
  [ -d "$d" ] || return 1
  [ "$(ls "$d" 2>/dev/null | grep -c '_backbone_')" -ge 33 ] && [ "$(ls "$d" | grep -c '_classifier_')" -ge 30 ]
}

cars_ready() {
  if [ "$MACHINE" = greenhorn ]; then
    grep -q killed /tmp/kill_after_cars.out 2>/dev/null && chain_complete cars
  else
    if ! chain_complete cars; then
      if ssh -o BatchMode=yes -o ConnectTimeout=10 "$GH" "grep -q killed /tmp/kill_after_cars.out" 2>/dev/null; then
        log "pulling the Cars chain from greenhorn"
        rsync -a "$GH:$GH_REPO/checkpoints_main/rebuttal/e12_cars" checkpoints_main/rebuttal/ >> "$OUT" 2>&1
      fi
    fi
    chain_complete cars
  fi
}

ds_ready() {
  if [ "$1" = cars ]; then cars_ready; else chain_complete "$1"; fi
}

log "start: methods [$METHODS] datasets [$DATASETS] seeds [$SEEDS]"
for m in $METHODS; do
  declare -A fin=()
  while [ "${#fin[@]}" -lt 4 ]; do
    for ds in $DATASETS; do
      [ -n "${fin[$ds]:-}" ] && continue
      if ! ds_ready "$ds"; then
        log "$m $ds: chain not complete yet, waiting"
        continue
      fi
      log "$m $ds: start"
      "$PY" run_e12_merge_grid.py --proto --budget 4 --rules "$m" --ds "$ds" --seeds $SEEDS \
        >> "logs_main/e12p_${MACHINE}_${m}_${ds}.out" 2>&1
      rc=$?
      log "$m $ds: exit $rc"
      fin[$ds]=1
    done
    [ "${#fin[@]}" -lt 4 ] && sleep 120
  done
  unset fin
done
log "all done"
