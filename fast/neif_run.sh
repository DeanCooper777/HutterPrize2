#!/bin/bash
# neif_run.sh NAME MODE START END WARM BUILD_DIR [CODED]: ONE chunk run of a cmix-neif 2.0 test
# build, on any Linux machine (see fast/README.md).
#   MODE t   `cmix -t STREAM OUT`: code stream bytes [warm_from, END) (warm_from = first page start
#            at or behind START - WARM, byte 0 if WARM >= START); the coder output is kept in
#            $OUT/NAME/out.cmix
#   MODE u   `cmix -u STREAM CODED`: decode CODED (the out.cmix of a `t` run with the same START END
#            WARM; give its path or the NAME of that run) and compare every byte with the stream
# Environment:
#   STREAM   the coder stream (default $NEIF_ROOT/stream.bin)
#   OUT      directory for the run directories and results.tsv (default $NEIF_ROOT/out; LOCAL disk:
#            every run makes a sparse 14.7 GB paging file in its run directory)
#   CPUS     comma list of logical CPUs: the process (and with NEIF_PIPE its stages) is bound to
#            the set with taskset; with NEIF_MT the threads are bound one per CPU in this order:
#            main thread, transformer thread (NEIF_MT_SPEC), workers. Empty: no binding.
#   switches of the binary, passed on as they are: NEIF_PIPE (2|3: pipeline of processes; with
#            MODE u the VERIFICATION DECODE), NEIF_PIPE_SPLIT, NEIF_MT, NEIF_MT_SPEC, NEIF_MT_FAULT,
#            NEIF_PROF, NEIF_ACT16, NEIF_ACT16_DNBITS, NEIF_ACT16_DNMIN
# One line per run is appended to $OUT/results.tsv (header in its first line) and printed.
# Exit code: the one of cmix (0 = done / every byte decoded and equal).
set -u
NAME=${1:?name}; MODE=${2:?t|u}; S=${3:?start}; E=${4:?end}; WM=${5:?warm}; B=$(cd "${6:?build dir}" && pwd); CODED=${7:-}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${NEIF_ROOT:-$HERE/work}
STREAM=$(readlink -f "${STREAM:-$ROOT/stream.bin}"); OUT=${OUT:-$ROOT/out}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
D=$OUT/$NAME; WF=weights-poc8m_w2048_tf2t3_pruned_v5k_np2.c5
RES=$OUT/results.tsv
[ -f "$RES" ] || echo -e "utc\thost\tname\tmode\tflavour\tbinary_sha256\tpipe\tmt\tspec\tfault_permille\tact16\tstart\tend\twarm_from\twall_s\tmodel_s\tcpu_s\tpeak_pss_kb\tbits_or_decoded_bytes\tout_sha16\tstatus\tcpus" > "$RES"
rm -rf "$D"; mkdir -p "$D/run"; cd "$D/run" || exit 1
ln -sf "$B/$WF" .
PIPE=${NEIF_PIPE:-0}; MT=${NEIF_MT:-0}; CPUS=${CPUS:-}
pre=""
if [ -n "$CPUS" ]; then
  if [ "$MT" -gt 0 ] && [ "$PIPE" -lt 2 ]; then pre="taskset -c ${CPUS%%,*}"; export NEIF_MT_CPUS=$CPUS
  else pre="taskset -c $CPUS"; fi
elif [ "$MT" -gt 0 ]; then export NEIF_MT_PIN=0
fi
if [ "$MODE" = u ]; then
  [ -f "$CODED" ] || CODED=$OUT/$CODED/out.cmix
  [ -s "$CODED" ] || { echo "$NAME: coded input $CODED is missing"; exit 2; }
fi
TIMEV=""; [ -x /usr/bin/time ] && TIMEV="/usr/bin/time -v -o $D/time.txt"
export NEIF_T_START=$S NEIF_T_END=$E NEIF_T_WARM=$WM NEIF_T_JSON=$D/score.json
t0=$(date +%s.%N)
if [ "$MODE" = t ]; then $pre $TIMEV "$B/cmix" -t "$STREAM" "$D/out.cmix" > "$D/log" 2>&1
else $pre $TIMEV "$B/cmix" -u "$STREAM" "$CODED" > "$D/log" 2>&1; fi
rc=$?
rm -rf "$D/run"
python3 - "$D" "$rc" "$t0" "$MODE" <<EOF | tee -a "$RES"
import hashlib, json, os, re, sys, time, socket
d, rc, t0, mode = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
wall = time.time() - t0
try:
    j = json.load(open(d + "/score.json"))
except Exception:
    j = {}
cpu = rss = 0.0
try:
    for ln in open(d + "/time.txt"):
        if "User time" in ln or "System time" in ln: cpu += float(ln.split()[-1])
        if "Maximum resident" in ln: rss = float(ln.split()[-1])
except Exception:
    pass
cpu = j.get("cpu_seconds") or cpu
pss = max(j.get("pipe_peak_pss_kb", 0), j.get("peak_rss_kb", 0), rss)
sha = "-"
if mode == "t" and os.path.exists(d + "/out.cmix"):
    sha = hashlib.sha256(open(d + "/out.cmix", "rb").read()).hexdigest()[:16]
if mode == "t":
    val = "%.3f" % j.get("total_bits", 0); ok = rc == 0 and bool(j)
else:
    val = str(j.get("decoded_bytes", 0)); ok = rc == 0 and j.get("wrong_bytes", 1) == 0
flav = open("$B/FLAVOUR").read().strip() if os.path.exists("$B/FLAVOUR") else os.environ.get("FLAVOUR", "unknown")
bsha = hashlib.sha256(open("$B/cmix", "rb").read()).hexdigest()
e = os.environ
a16 = e.get("NEIF_ACT16", "0")
if a16 not in ("", "0"): a16 += "/" + e.get("NEIF_ACT16_DNBITS", "16") + "/" + e.get("NEIF_ACT16_DNMIN", "1")
print("\t".join(str(x) for x in [time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), socket.gethostname(), "$NAME", mode, flav, bsha,
      "$PIPE", "$MT", e.get("NEIF_MT_SPEC", "0"), e.get("NEIF_MT_FAULT", "0"), a16 or "0", j.get("start", "$S"), j.get("end", "$E"),
      j.get("warm_from", "-"), "%.1f" % wall, "%.1f" % j.get("model_seconds", 0), "%.1f" % cpu, int(pss), val, sha,
      ("OK" if mode == "t" else "PASS") if ok else "FAILED rc %d" % rc, "${CPUS:--}"]))
EOF
exit $rc
