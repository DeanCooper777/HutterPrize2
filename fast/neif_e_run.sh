#!/bin/bash
# neif_e_run.sh NAME STOP PIPE ENWIK9 [BUILD_DIR]: one run of the NORMAL compress command
# (`cmix -e enwik9 out`) of a NEIF_TEST build made by neif_e_build.sh, in $NEIF_ROOT/e/NAME.
#   STOP    NEIF_E_STOP: the coder stops behind this many stream bytes and the process ends
#           (0 = the whole stream: about 50 h in one process on a 2024 laptop; with a pipeline
#           the end of the stream has NEVER been run, see fast/README.md)
#   PIPE    NEIF_PIPE: 0 = one process (the reference); 2 or 3 = the opt-in multi-process
#           compression (NOT the official single-core run)
#   ENWIK9  the 1 GB input file
#   BUILD_DIR default $NEIF_ROOT/build/tbKe
# While the coder runs, the stream it codes (out.cmix.temp) is compared with
# $NEIF_ROOT/stream.bin if that file exists (KEEP_STREAM=1: a copy is kept as
# $NEIF_ROOT/e/NAME/stream.bin). One line per run on stdout and in $NEIF_ROOT/e/results.txt:
# exit code, size and sha256 of the coder output, the figures of the binary's `neif_e:` line,
# wall time of the whole command.
# KILL_AFTER=s: kills the newest cmix process (a front stage of the pipeline) s seconds after the
# coder has started, to show that a broken pipeline leaves no output.
#
# Identity check of the multi-process path against one process (what we ran; 17 M stream bytes
# take about 50 + 35 min on a laptop):
#   bash neif_e_run.sh e_p0 17000000 0 /data/enwik9 && bash neif_e_run.sh e_p3 17000000 3 /data/enwik9
#   -> the two lines must show the same size and sha256.
# Refuses to start while another cmix/comp9 runs (memory: 10 GiB).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
R=${NEIF_ROOT:-$HERE/work}
NAME=${1:?usage: neif_e_run.sh NAME STOP PIPE ENWIK9 [BUILD_DIR]}; STOP=${2:?stop}; PIPE=${3:?pipe}
E=$(readlink -f "${4:?enwik9}"); B=$(cd "${5:-$R/build/tbKe}" && pwd)
KA=${KILL_AFTER:-0}
[ -f "$E" ] || { echo "$E: no such file"; exit 1; }
[ -x "$B/e/cmix" ] || { echo "$B/e/cmix is missing: neif_e_build.sh"; exit 1; }
if pgrep -x cmix > /dev/null || pgrep -x comp9 > /dev/null; then echo "another cmix/comp9 is running"; exit 1; fi
mkdir -p "$R/e"; R=$(cd "$R" && pwd)
REF=$R/stream.bin
D=$R/e/$NAME
rm -rf "$D"; mkdir -p "$D"; cd "$D" || exit 1
cp "$B/e/cmix" ./cmix
export PATH=.:$PATH
export NEIF_E_STOP=$STOP
[ "$PIPE" -gt 1 ] && export NEIF_PIPE=$PIPE
( # the stream the coder is given, checked while it runs; then the optional kill
  while [ ! -e out ] && [ -e cmix ]; do sleep 2; done
  sleep 3
  n=$(stat -c %s out.cmix.temp 2> /dev/null || echo 0)
  if [ ! -f "$REF" ]; then echo "stream NOT_COMPARED $n" > stream_check
  elif [ "$n" = "$(stat -L -c %s "$REF")" ] && cmp -s out.cmix.temp "$REF"; then echo "stream IDENTICAL $n" > stream_check
  else echo "stream DIFFERENT $n" > stream_check; fi
  [ "${KEEP_STREAM:-0}" = 1 ] && cp out.cmix.temp stream.bin
  if [ "$KA" -gt 0 ]; then
    sleep "$KA"
    echo "killing $(pgrep -n -x cmix) of $(pgrep -x cmix | tr '\n' ' ')" > kill_note
    kill -9 "$(pgrep -n -x cmix)"
  fi
) &
TIMEV=""; [ -x /usr/bin/time ] && TIMEV="/usr/bin/time -v -o time.txt"
$TIMEV ./cmix -e "$E" out > log 2>&1
rc=$?
echo $rc > rc
wait
rm -f ppm.page .main* .ready4cmix out.cmix.temp .intro .coda .decomp_bin weights-*.c5 cmix
if [ -f out ]; then sz=$(stat -c %s out); sha=$(sha256sum out | cut -d' ' -f1); else sz=NONE; sha=NO_OUTPUT_FILE; fi
ar=no; [ -e archive9 ] && ar=YES
line=$(tr '\r' '\n' < log | grep -a "^neif_e: [0-9]" | tail -n 1)
wall=$(grep "Elapsed (wall" time.txt 2> /dev/null | sed 's/.*): //'); rss=$(grep "Maximum resident" time.txt 2> /dev/null | sed 's/.*: //')
echo "$NAME e stop $STOP pipe $PIPE rc $rc out $sz B sha256 $sha archive9 $ar $(cat stream_check 2> /dev/null) | $line | command wall $wall, max rss of the first process $rss kB" | tee -a "$R/e/results.txt"
tr '\r' '\n' < log | grep -a "neif_pipe\|neif_e: STOP" | cut -c1-220 | tail -n 12
[ -f kill_note ] && cat kill_note
exit $rc
