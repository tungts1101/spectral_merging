#!/bin/bash
# bio, queued 2026-11-14 02:30 JST (user: one experiment per machine): after the
# full-protocol E12 grid logs 'all done', run on ImageNet-R
#   E3   drift-solve cost in the pipeline itself: the stage-timed headline cell
#        (dense solve) against the same cell with the Woodbury solve (same map)
#   E13  few-shot per task, K = 5 / 10 (user 2026-11-14 03:10 JST: K = 20 dropped): our rows (ISM, E2-LoRA, LCA) in
#        our pipeline, ISM timed with both solvers (n < d: Woodbury vs dense),
#        then SLCA and EASE through PILOT (env 'pilot')
# E5, E9, E6, E7 run on greenhorn (scripts/queue_greenhorn_after_e12.sh).
set -u
export PYTHONUTF8=1 DATA_ROOT=/home/bio/data
R=/home/bio/Desktop/GitHub/spectral_merging
cd "$R" || exit 1
PY=/home/bio/anaconda3/envs/cl/bin/python
PPY=/home/bio/anaconda3/envs/pilot/bin/python
OUT=logs_main/queue_e_bio.out
CHAIN=checkpoints_main/rebuttal/e12_imagenetr
log() { echo "$(date '+%F %T') $*" >> "$OUT"; }
log "queued (v4, K = 5 and 10): E3 (dense vs Woodbury, timed) then E13 (ours incl. ISM dense, then PILOT) after the E12 grid"
until grep -q "all done$" logs_main/e12f_bio.out 2>/dev/null; do sleep 120; done
log "E12 grid done; starting"

log "E3 start"
"$PY" run_covmode.py --ds imagenetr --modes cov_class --timing reference --ckpt-src "$CHAIN" \
  --seeds 1993 1994 1995 >> logs_main/queue_e3.out 2>&1
log "E3 timed reference exit $?"
"$PY" run_covmode.py --ds imagenetr --modes cov_class --timing all --solver woodbury --ckpt-src "$CHAIN" \
  --seeds 1993 1994 1995 >> logs_main/queue_e3.out 2>&1
log "E3 woodbury exit $?"
log "E3 done"

log "E13 ours start"
"$PY" run_e13_fewshot.py --ds imagenetr --k 5 10 --rows ism e2lora lca --timing \
  --seeds 1993 1994 1995 >> logs_main/queue_e13.out 2>&1
log "E13 ours (woodbury) exit $?"
"$PY" run_e13_fewshot.py --ds imagenetr --k 5 10 --rows ism --solver dense --timing \
  --seeds 1993 1994 1995 >> logs_main/queue_e13.out 2>&1
log "E13 ism dense exit $?"

log "E13 PILOT start"
cd /home/bio/Desktop/GitHub/pilot_e13 || exit 1
for name in slca ease; do
  for k in 5 10; do
    "$PPY" main.py --config "exps/e13/${name}_inr_fs${k}.json" >> "$R/logs_main/queue_e13_pilot_${name}_fs${k}.out" 2>&1
    log "E13 PILOT $name fs$k exit $?"
  done
done
log "E13 done"
log "all done"
