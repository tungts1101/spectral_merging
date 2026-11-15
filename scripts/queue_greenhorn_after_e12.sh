#!/bin/bash
# greenhorn, queued 2026-11-14 02:30 JST (user: one experiment per machine): after
# the full-protocol E12 grid and the Task Arithmetic cells log 'all done (incl. avg)',
# run on ImageNet-R (3 seeds, cached headline chain e12_imagenetr)
#   E5  class statistics: full covariance / mean + diagonal / mean alone (prototype head)
#   E9  drift map fitted on a replay buffer, N = 5 / 10 / 20 per class (+ reference fit)
#   E6  discarded vs kept directions of the update (replay, no training), p = 0.9 and 0.1
#   E7  adaptive damping, two rules with lambda chosen on a held-out 20 % split of
#       the task (RanPAC-style): spectral-norm bound tau = 0.1 and feature-shift
#       bound kappa = 0.15 (alpha_t = min(1, bound / s_t)); reference = the main table
# E3 and E13 run on bio (scripts/queue_bio_after_e12.sh).
set -u
export PYTHONUTF8=1 DATA_ROOT=/home/greenhorn/data
R=/home/greenhorn/Desktop/GitHub/spectral_merging
cd "$R" || exit 1
PY=/home/greenhorn/miniconda3/envs/cl/bin/python
OUT=logs_main/queue_e_greenhorn.out
CHAIN=checkpoints_main/rebuttal/e12_imagenetr
log() { echo "$(date '+%F %T') $*" >> "$OUT"; }
log "queued (v2): E5 E9 E6 E7 after the E12 grid (incl. avg)"
until grep -q "all done (incl. avg)$" logs_main/e12f_greenhorn.out 2>/dev/null; do sleep 120; done
log "E12 grid done; starting"

log "E5 start"
"$PY" run_covmode.py --ds imagenetr --modes cov_class cov_diag cov_mean --timing none \
  --ckpt-src "$CHAIN" --seeds 1993 1994 1995 >> logs_main/queue_e5.out 2>&1
log "E5 exit $?"
log "E5 done"

log "E9 start"
"$PY" run_e9_buffer.py --ds imagenetr --n 5 10 20 --with-reference \
  --ckpt-src "$CHAIN" --seeds 1993 1994 1995 >> logs_main/queue_e9.out 2>&1
log "E9 exit $?"
log "E9 done"

log "E6 start"
"$PY" analyze_subspace.py --ds imagenetr --ckpt-dir "$CHAIN" --seeds 1993 1994 1995 \
  --out-dir figures_data >> logs_main/queue_e6.out 2>&1
log "E6 p0.9 exit $?"
"$PY" analyze_subspace.py --ds imagenetr --ckpt-dir "$CHAIN" --seeds 1993 1994 1995 \
  --p 0.1 --tag _p0.1 --out-dir figures_data >> logs_main/queue_e6.out 2>&1
log "E6 p0.1 exit $?"
log "E6 done"

log "E7 start"
"$PY" run_e7_adaptive_alpha.py --run --ds imagenetr --rules specnorm featshift \
  --tau 0.1 --kappa 0.15 --lambda-rule val --ckpt-src "$CHAIN" --seeds 1993 1994 1995 \
  >> logs_main/queue_e7.out 2>&1
log "E7 exit $?"
log "E7 done"
log "all done"
