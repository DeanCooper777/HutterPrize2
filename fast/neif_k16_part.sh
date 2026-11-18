#!/bin/bash
# neif_k16_part.sh SPLITS [START_UTC]: one machine's part of a K=16 split evaluation on several
# machines (e.g. SPLITS=0-7 on the first, 8-15 on the second), packed build, 4 MB warm-up, both
# passes:
#   1 compress, NEIF_PIPE=3, all listed splits at once, each pinned to 4 logical CPUs = 2 whole
#     physical cores (neif_cpusets.py N 4 on a 16-core / 32-thread cpuset)
#   2 verification decode (NEIF_PIPE=3 -u), same layout
# This is the driver of our two-instance run (results: fast/results/k16_two_machines/).
# START_UTC (optional, "HH:MM" or "YYYY-MM-DD HH:MM", UTC): wait until then, so that all
# machines start at the same moment.
# Environment: NEIF_ROOT (default fast/work) with build/tbK and stream.bin (neif_setup.sh);
#   OUTNAME (default k16part: a FRESH directory, nothing resumed from another run), SLOTS
#   (default: number of splits), PINW (default 4), PASSES (default "compress verify"),
#   SYNC_DIR (default $NEIF_ROOT: after each pass the small result files are packed into
#   $SYNC_DIR/<OUTNAME>_sync.tgz).
# Start detached:  setsid nohup bash fast/neif_k16_part.sh 0-7 "12:30" > /dev/null 2>&1 < /dev/null &
# Log: $NEIF_ROOT/out/<OUTNAME>.log (start / end / wall seconds of each pass); monitor:
# out/<OUTNAME>/monitor.tsv (10 s).
# Merge afterwards: copy all machines' out/<OUTNAME>/c* into one directory, then
#   bash fast/neif_split_eval.sh --summary DIR 16
SPL=${1:?splits, e.g. 0-7}; AT=${2:-}
HERE=$(cd "$(dirname "$0")" && pwd)
export NEIF_ROOT=${NEIF_ROOT:-$HERE/work}
cd "$NEIF_ROOT" || exit 1
N=${OUTNAME:-k16part}; O=out/$N; mkdir -p $O; L=out/$N.log
SYNC=${SYNC_DIR:-$NEIF_ROOT}
n=0; for p in ${SPL//,/ }; do if [[ $p == *-* ]]; then n=$((n + ${p#*-} - ${p%-*} + 1)); else n=$((n + 1)); fi; done
export SLOTS=${SLOTS:-$n} SPLITS=$SPL PIN=1 PIN_W=${PINW:-4}
say() { echo "$(date -u +%FT%TZ) $*" >> $L; }
sync_res() { tar czf "$SYNC/${N}_sync.tmp" --exclude='out.cmix*' --exclude=run --exclude=drun $O $L out/${N}_*.log 2> /dev/null; mv "$SYNC/${N}_sync.tmp" "$SYNC/${N}_sync.tgz"; }
ev() { bash "$HERE/neif_split_eval.sh" build/tbK stream.bin 16 4000000 $O 3; }
pass() {   # NAME LOGFILE (switches in the environment); one retry of what failed (the script resumes)
  echo "$1" > $O/.pass; say "$1 START splits $SPL, $SLOTS at once, $PIN_W CPUs each"; local t0=$(date +%s)
  ev > "$2" 2>&1; local rc=$?
  if [ $rc -ne 0 ]; then say "$1 first attempt exit $rc: $(grep -h 'failed splits\|cmix exit' "$2" | tr '\n' ' '); retrying"; ev > "$2.retry" 2>&1; rc=$?; fi
  say "$1 END exit $rc wall_s $(( $(date +%s) - t0 ))"
  echo idle > $O/.pass; sync_res
}
if [ -n "$AT" ]; then
  w=$(( $(date -u -d "$AT" +%s) - $(date +%s) )); say "waiting $w s for the start at $AT UTC"; [ $w -gt 0 ] && sleep $w
fi
say "driver start; cpu sets: $(python3 "$HERE/neif_cpusets.py" $SLOTS $PIN_W 2>&1 | tr '\n' ' ')"
say "host: loadavg $(cut -d' ' -f1-3 /proc/loadavg), cmix processes already running: $(pgrep -c -x cmix)"
python3 "$HERE/neif_monitor.py" $O 10 & MON=$!
case " ${PASSES:-compress verify} " in *" compress "*) STEPS=compress pass PASS1_compress out/${N}_compress.log ;; esac
case " ${PASSES:-compress verify} " in *" verify "*) STEPS=decode DECODE=verify pass PASS2_verify out/${N}_verify.log ;; esac
bash "$HERE/neif_split_eval.sh" --summary $O 16 > out/${N}_summary.log 2>&1
say "summary (this machine's splits only): $(tr '\n' ' ' < out/${N}_summary.log)"
kill $MON; sync_res
say "ALL DONE"
