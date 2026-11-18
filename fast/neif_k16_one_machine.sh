#!/bin/bash
# neif_k16_one_machine.sh: the full K=16 split evaluation on ONE machine (32 logical CPUs, about
# 170 GiB of memory), packed build, 4 MB warm-up, passes chained:
#   1 compress, NEIF_PIPE=3, 16 splits at once, unpinned (48 processes on 32 logical CPUs)
#   2 verification decode (NEIF_PIPE=3 -u), 16 at once, unpinned
#   3 (only with PLAIN_DECODE="all" or a split list such as "0,8,15") the real decoder, plain -u,
#     one process per split, one physical core each. Never run by us on a whole split.
# This is the driver of our one-instance run (results: fast/results/k16_one_machine/).
# Environment: NEIF_ROOT (default fast/work) with build/tbK and stream.bin (neif_setup.sh);
#   OUTNAME (default k16), SLOTS (default 16), SYNC_DIR (default $NEIF_ROOT: after every pass the
#   small result files are packed into $SYNC_DIR/<OUTNAME>_sync.tgz).
# Log: $NEIF_ROOT/out/<OUTNAME>.log (start and end of every pass); monitor:
# out/<OUTNAME>/monitor.tsv (10 s).
# Start detached:  setsid nohup bash fast/neif_k16_one_machine.sh > /dev/null 2>&1 < /dev/null &
HERE=$(cd "$(dirname "$0")" && pwd)
export NEIF_ROOT=${NEIF_ROOT:-$HERE/work}
cd "$NEIF_ROOT" || exit 1
N=${OUTNAME:-k16}; O=out/$N; mkdir -p $O; L=out/$N.log
SYNC=${SYNC_DIR:-$NEIF_ROOT}
say() { echo "$(date -u +%FT%TZ) $*" >> $L; }
sync_res() { tar czf "$SYNC/${N}_sync.tmp" --exclude='out.cmix*' --exclude=run --exclude=drun $O $L out/${N}_*.log 2> /dev/null && mv "$SYNC/${N}_sync.tmp" "$SYNC/${N}_sync.tgz"; }
ev() { bash "$HERE/neif_split_eval.sh" build/tbK stream.bin 16 4000000 $O 3; }
pass() {   # NAME LOGFILE: runs ev with the environment given; one retry of what failed (the script resumes)
  echo "$1" > $O/.pass; say "$1 START"; local t0=$(date +%s)
  ev > "$2" 2>&1; local rc=$?
  if [ $rc -ne 0 ]; then say "$1 first attempt exit $rc: $(grep -h 'failed splits\|cmix exit' "$2" | tr '\n' ' '); retrying those"; ev > "$2.retry" 2>&1; rc=$?; fi
  say "$1 END exit $rc wall_s $(( $(date +%s) - t0 ))"
  echo idle > $O/.pass; sync_res
}
say "driver start; host load before launch (2 min, every 10 s; busy = share of all host CPUs)"
python3 - >> $L <<'PY'
import time
def h():
    v = [int(x) for x in open("/proc/stat").readline().split()[1:]]; return sum(v), v[3] + v[4]
a = h()
for i in range(12):
    time.sleep(10); b = h()
    print("  %s loadavg %s host busy %.3f" % (time.strftime("%H:%M:%S", time.gmtime()), open("/proc/loadavg").read().split()[0], 1 - (b[1] - a[1]) / (b[0] - a[0]))); a = b
PY
python3 "$HERE/neif_monitor.py" $O 10 & MON=$!
export SLOTS=${SLOTS:-16}
PIN=0 STEPS=compress pass PASS1_compress out/${N}_compress.log
PIN=0 STEPS=decode DECODE=verify pass PASS2_verify out/${N}_verify.log
if [ "${PLAIN_DECODE:-}" = all ]; then
  PIN=1 PIN_W=1 STEPS=decode DECODE=plain pass PASS3_decode out/${N}_decode.log
elif [ -n "${PLAIN_DECODE:-}" ]; then
  PIN=1 PIN_W=1 SLOTS=3 DEC_SPLITS=$PLAIN_DECODE STEPS=decode DECODE=plain pass PASS3_decode out/${N}_decode.log
fi
bash "$HERE/neif_split_eval.sh" --summary $O 16 > out/${N}_summary.log 2>&1; say "summary: $(tr '\n' ' ' < out/${N}_summary.log)"
kill $MON; sync_res
say "ALL DONE"
