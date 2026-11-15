#!/bin/bash
# Progress of the E12 proxy grid (and the ISM rank / alpha extension) on this machine:
# per (method, dataset) the number of finished cell-seeds out of 4 cells x 3 seeds
# (Model Stock 1 x 3; ISM extension 10 x 3 per rank), the finished cells' 3-seed
# summary, the e12r full-protocol rank chains, and the drivers' current step.
cd "$(dirname "$0")/.." || exit 1
M=$(hostname)
OUT=$(ls logs_main/e12p_*.out 2>/dev/null | grep -vE "_(ism|nufilt|ties|knots|modelstock|dare|della|breadcrumbs|avg|max_abs)_" | head -1)
ROUT=$(ls logs_main/e12r_*.out 2>/dev/null | grep -vE "_(cars|cub)_r[0-9]+\.out$" | head -1)
echo "## $M  $(date '+%H:%M')"
[ -n "$OUT" ] && { echo "driver: $(grep -E 'start$|exit|waiting|all done' "$OUT" | tail -1)"; }
[ -n "$ROUT" ] && { echo "ext driver: $(tail -1 "$ROUT")"; }
FOUT=logs_main/e12f_${M}.out
[ -f "$FOUT" ] && { echo "full driver: $(tail -1 "$FOUT")"; echo "full exits: $(grep -cE 'exit [1-9]' "$FOUT") nonzero"; }
echo "method dataset cellseeds_done cells_done"
for m in ism nufilt ties knots modelstock dare della breadcrumbs avg max_abs ism_r16 ism_r32 ism_r128; do
  for ds in imagenetr cub cifar224 cars; do
    case "$m" in
      ism) logs=$(ls logs_main/$ds/e12p_ism_p*.log 2>/dev/null) ;;
      *)   logs=$(ls logs_main/$ds/e12p_${m}_*.log 2>/dev/null) ;;
    esac
    [ -z "$logs" ] && continue
    n=0; c=0
    for f in $logs; do
      k=$(grep -ac "=> Time:" "$f"); n=$((n+k))
      grep -aq "=>   ASA:" "$f" && c=$((c+1))
    done
    echo "$m $ds $n $c"
  done
done
echo "--- full-protocol grid cells (e12_<rule>): cellseeds_done cells_done"
for m in nufilt ties knots modelstock dare della breadcrumbs avg max_abs; do
  for ds in imagenetr cub cifar224 cars; do
    logs=$(ls logs_main/$ds/e12_${m}_*.log 2>/dev/null | grep -v partial_killed); [ -z "$logs" ] && continue
    n=0; c=0
    for f in $logs; do k=$(grep -ac "=> Time:" "$f"); n=$((n+k)); grep -aq "=>   ASA:" "$f" && c=$((c+1)); done
    echo "F $m $ds $n $c"
  done
done
echo "--- rank chains (full protocol e12r): seeds done, last step"
for ds in imagenetr cub cifar224 cars; do
  for f in $(ls logs_main/$ds/e12r_ism_r*.log 2>/dev/null); do
    n=$(grep -ac "=> Time:" "$f")
    last=$(grep -aE "Experiment:|Training\] Task [0-9]+|=>   ASA:" "$f" | tail -1 | sed 's/.*=> //' | cut -c1-70)
    echo "$ds $(basename "$f" .log) seeds_done=$n last: $last"
  done
done
echo "--- finished cells (3-seed FAA / ASA)"
for ds in imagenetr cub cifar224 cars; do
  for f in $(ls logs_main/$ds/e12p_*.log logs_main/$ds/e12r_*.log logs_main/$ds/e12_*.log 2>/dev/null | grep -v partial_killed); do
    # mean ± std (population) over the per-seed final evaluations, so an appended
    # (resumed) log is summarised over all its seeds, not only the seeds it added
    grep -aq "=>   ASA:" "$f" || continue
    s=$(grep -aE "Evaluation\] FAA|=> Time:" "$f" | awk '
      /Evaluation\] FAA/ { match($0, /FAA: [0-9.]+/); fa=substr($0, RSTART+5, RLENGTH-5);
                            match($0, /ASA: [0-9.]+/); aa=substr($0, RSTART+5, RLENGTH-5) }
      /=> Time:/ { n++; sf+=fa; sa+=aa; qf+=fa*fa; qa+=aa*aa }
      END { if (n==0) exit; mf=sf/n; ma=sa/n; vf=qf/n-mf*mf; va=qa/n-ma*ma; if (vf<0) vf=0; if (va<0) va=0;
            printf "ASA: %.2f ± %.2f | FAA: %.2f ± %.2f | n=%d", ma, sqrt(va), mf, sqrt(vf), n }')
    [ -n "$s" ] && echo "$ds $(basename "$f" .log) $s"
  done
done
echo "--- current"
f=$(ls -t logs_main/*/e12p_*.log logs_main/*/e12r_*.log logs_main/*/e12_*.log 2>/dev/null | head -1)
[ -n "$f" ] && { echo "$f"; grep -aE "Experiment:|Evaluation\] FAA|Time:|Training\] Task" "$f" | tail -2 | cut -c1-120; }
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
