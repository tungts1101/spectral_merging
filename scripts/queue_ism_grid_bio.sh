#!/bin/bash
# bio, queued 2026-11-15 00:10 JST (user decision 2026-11-14 23:55: Table 6 = full
# protocol, four settings per rule, ISM included): ISM's GRID4 cells
# (alpha {0.1, 0.3} x p {0.7, 0.9}, run_e12_merge_grid.py --budget 4, no --proto) on the
# cached e12_<ds> chains, 3 seeds, for ImageNet-R, CUB and CIFAR-100, after the bio
# appendix queue (scripts/queue_bio_after_e12.sh) logs 'all done'.  Cars runs on
# greenhorn (scripts/queue_ism_grid_greenhorn.sh).  One experiment per machine.
set -u
export PYTHONUTF8=1 DATA_ROOT=/home/bio/data
R=/home/bio/Desktop/GitHub/spectral_merging
cd "$R" || exit 1
PY=/home/bio/anaconda3/envs/cl/bin/python
OUT=logs_main/e12f_ism_bio.out
log() { echo "$(date '+%F %T') $*" >> "$OUT"; }
log "queued: ISM GRID4 (full protocol) on imagenetr cub cifar224 after queue_e_bio 'all done'"
until grep -q "all done$" logs_main/queue_e_bio.out 2>/dev/null; do sleep 120; done
log "bio appendix queue done; starting"
# the CIFAR-100 headline cell (paper setting) ran on greenhorn: copy its log so the
# driver's resume rule skips it here instead of running a fifth cell
if [ ! -f logs_main/cifar224/e12_ism.log ]; then
  scp -q greenhorn@greenhorn.local:/home/greenhorn/Desktop/GitHub/spectral_merging/logs_main/cifar224/e12_ism.log \
    logs_main/cifar224/e12_ism.log && log "copied cifar224 e12_ism.log from greenhorn"
fi
for ds in imagenetr cub cifar224; do
  for attempt in 1 2 3; do
    log "ism $ds: start (attempt $attempt)"
    "$PY" run_e12_merge_grid.py --budget 4 --append --rules ism --ds "$ds" --seeds 1993 1994 1995 \
      >> "logs_main/e12f_bio_ism_${ds}.out" 2>&1
    rc=$?; log "ism $ds: exit $rc"
    [ "$rc" -eq 0 ] && break
    sleep 30
  done
done
log "all done"
