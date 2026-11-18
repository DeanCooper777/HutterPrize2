#!/bin/bash
# neif_split_eval.sh BUILD_DIR STREAM K WARM OUT_DIR [PIPE]   run the split evaluation
# neif_split_eval.sh --plan STREAM K WARM                     only print the K split ranges
# neif_split_eval.sh --summary OUT_DIR K                      only the summary of result dirs
#                                                             (e.g. merged from several machines)
#
# Split ("chunk") evaluation of cmix-neif 2.0: an estimate of how a change alters the compressed
# size of enwik9 without the 49-hour full run, and a decode check of every split. The coder stream
# (what `-e enwik9` hands to the arithmetic coder, 501,062,281 B) is cut into K splits; each split
# is coded by its own run of a NEIF_TEST build (mode -t, src/neif_test.h in the source tree),
# several at a time, and the ideal code lengths (sum of -log2 p of every coded bit) of the splits'
# own bytes are added up. Portable: no path of any particular machine in here; needs bash,
# python3, coreutils, util-linux (taskset, flock) and, for the times, /usr/bin/time.
#
# One split [s, e):
#   * bytes [0, warm_from) get a structure-only pass (no model, nothing coded; about 2 MB/s);
#     warm_from = the first page start at or behind s - WARM.
#   * bytes [warm_from, e) are coded by the complete predictor, starting from cold models; only
#     the bits of [s, e) are counted (kept_bits). The coder output of the split stays in
#     OUT_DIR/c<k>/out.cmix (about 1/5.6 of the coded stream bytes) for the decode step.
#   * a split whose warm-up would reach into the head of the stream (s - WARM < HEAD) is coded
#     from byte 0 and is a true prefix of a real run.
# The sum is biased upward (each split starts cold): compare variants by the DIFFERENCE of their
# totals at the same K and WARM.
#
# Steps (environment STEPS, default "compress summary"; any of: compress decode summary):
#   compress  `cmix -t` per split -> c<k>/score.json, c<k>/out.cmix, c<k>/log, c<k>/time.txt
#   decode    per split of DEC_SPLITS, the coded output is decoded and every byte compared with
#             the stream -> c<k>/dec.json, c<k>/dec.log. DECODE selects how:
#               verify  (default) the VERIFICATION DECODE: `NEIF_PIPE=3 cmix -u`, the front stages
#                       walk the expected stream, the coder stage decodes and compares
#               plain   the real decoder, one process, one thread: `cmix -u`
#               spec    the threaded decoder test: `cmix -u` with NEIF_MT=DEC_MT and
#                       NEIF_MT_SPEC=2 (needs a NEIF_MT build in DEC_BUILD; the guess of the
#                       speculative step is taken from the stream: it tests the machinery and
#                       its speed, it is not yet a decoder that works without the stream)
#             a decode passes when cmix exits 0 (all bytes decoded and equal).
#   summary   total.json and one line: total kept bytes, estimated S (with BASE_TOTAL), wall
#             times, peak memory per split, all decodes PASS or not. Splits without a result
#             are listed (results of several machines are merged by copying their c<k>
#             directories into one OUT_DIR and running `--summary OUT_DIR K`).
#
# BUILD_DIR  directory with the test binary `cmix` and the weights container (a file FLAVOUR in
#            it, if present, names the build in the result lines; else environment FLAVOUR).
# STREAM     the coder stream (501,062,281 B, sha256 c35d3073eeba70f2... for the published entry)
# PIPE       6th argument, default 0: every split's compression runs as a pipeline of PIPE
#            processes (NEIF_PIPE of the test binary; 2 or 3). Same bits as PIPE=0.
#
# Environment:
#   SPLITS      the splits to run, e.g. "0-7" or "0,8,15" (default all K); DEC_SPLITS likewise for
#               the decode step (default SPLITS)
#   SLOTS       splits at a time (default: JOBS / processes per split if JOBS is set, else the
#               number of splits). Plan 10 GiB of RAM per split (plus 0.5 GiB with a pipeline).
#   PIN=1       pin every split to its own cpu set with taskset (default 0: no pinning). The sets
#               come from neif_cpusets.py (next to this script): SLOTS sets of PIN_W CPUs
#               (default: the processes or threads of a split), distinct physical cores when
#               there are enough, else it says that SMT siblings are shared. The stages of a
#               pipeline are processes forked by cmix: they inherit the set (the kernel spreads
#               them over it); with DECODE=spec the threads are bound one per CPU of the set
#               (NEIF_MT_CPUS).
#   CHUNK_BYTES code only the first CHUNK_BYTES of every split (a sampling / smoke-test mode)
#   DECODE, DEC_BUILD (default BUILD_DIR), DEC_MT (default 2), DEC_PIPE (default 3)
#   KEEP_OUT=0  delete the coded output after compression (then no decode step is possible)
#   EMIT_DIR    copy every finished split's small result files there (and results.tsv), e.g. a
#               network volume, as they finish
#   BASE_TOTAL  total kept bytes of the reference run with the same K and WARM: the summary then
#               prints estimated S = 93,388,956 + (total - BASE_TOTAL)
#   HEAD        default 16000000 (a warm-up target below it means "run from byte 0")
#   any switch of the binary (NEIF_ACT16..., NEIF_PIPE_SPLIT) is passed on unchanged.
# Resumable: a split with c<k>/score.json (and its out.cmix, if a decode is asked for) is not
# coded again; a split with a passing c<k>/dec.json of the same DECODE mode is not decoded again.
# Every finished run appends one line to OUT_DIR/results.tsv (columns in its first line); once
# per start a file OUT_DIR/session_<host>_<time>.txt records the machine (lscpu, cgroup limits,
# throttling and OOM counters before and after, environment, file system of OUT_DIR).
# Put OUT_DIR on a LOCAL disk: every run creates a sparse 14.7 GB paging file there.
set -uo pipefail
HEAD=${HEAD:-16000000}
HERE=$(cd "$(dirname "$0")" && pwd)
plan() {   # plan STREAM K WARM -> lines "k start end warm_arg"
  local n; n=$(stat -L -c %s "$1") || exit 1
  local K=$2 WARM=$3 k s e w
  for ((k = 0; k < K; k++)); do
    s=$((n * k / K)); e=$((n * (k + 1) / K))
    [ -n "${CHUNK_BYTES:-}" ] && [ $((s + CHUNK_BYTES)) -lt $e ] && e=$((s + CHUNK_BYTES))
    # warm_arg = s makes the binary start the model at byte 0
    if [ $((s - WARM)) -lt "$HEAD" ]; then w=$s; else w=$WARM; fi
    echo "$k $s $e $w"
  done
}
expand() {   # "0-3,8" -> 0 1 2 3 8
  local p a b
  for p in ${1//,/ }; do
    if [[ $p == *-* ]]; then a=${p%-*}; b=${p#*-}; seq "$a" "$b"; else echo "$p"; fi
  done
}
summary() {   # OUT K
  python3 - "$1" "$2" "${BASE_TOTAL:-}" <<'EOF'
import json, os, sys
out, K, base = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rs, ds, missing = {}, {}, []
for k in range(K):
    try:
        rs[k] = json.load(open(f"{out}/c{k}/score.json"))
    except Exception:
        missing.append(k)
    for f in os.listdir(f"{out}/c{k}") if os.path.isdir(f"{out}/c{k}") else []:
        if f.startswith("dec_") and f.endswith(".json"):
            try:
                ds.setdefault(k, {})[f[4:-5]] = json.load(open(f"{out}/c{k}/{f}"))
            except Exception:
                pass
bad = 0
for k, r in rs.items():
    if r["kept_from"] != r["start"]:
        print(f"split {k}: the model started behind the split start; raise WARM"); bad = 1
    if r["forced_byte_mismatch"] != 0:
        print(f"split {k}: structure pass out of step"); bad = 1
tot = sum(r["kept_bytes"] for r in rs.values())
n = sum(r["end"] - r["start"] for r in rs.values())
pss = lambda r: max(r.get("peak_rss_kb", 0), r.get("pipe_peak_pss_kb", 0)) / 2**20
dec = []
for k in sorted(ds):
    for mode, d in ds[k].items():
        ok = d.get("rc", 1) == 0 and d.get("wrong_bytes", 1) == 0
        dec.append(dict(k=k, mode=mode, passed=ok, decoded_bytes=d.get("decoded_bytes"), seconds=d.get("wall_seconds"),
                        peak_gib=max(d.get("peak_rss_kb", 0), d.get("pipe_peak_pss_kb", 0)) / 2**20))
res = dict(K=K, splits_done=sorted(rs), splits_missing=missing, total_kept_bytes=tot, stream_bytes=n,
           compress_max_split_hours=max([r["seconds"] for r in rs.values()] or [0]) / 3600,
           compress_cpu_hours=sum(r.get("cpu_seconds", r["seconds"]) for r in rs.values()) / 3600,
           pipe=max([r.get("pipe", 0) for r in rs.values()] or [0]),
           max_peak_gib=max([pss(r) for r in rs.values()] or [0]),
           decodes=dec, decodes_all_pass=bool(dec) and all(d["passed"] for d in dec),
           decode_max_split_hours=max([d["seconds"] or 0 for d in dec] or [0]) / 3600,
           splits=[dict(k=k, start=r["start"], end=r["end"], warm_from=r["warm_from"], kept_bytes=r["kept_bytes"],
                        bits_per_byte=r["kept_bits_per_byte"], seconds=r["seconds"], peak_gib=pss(r)) for k, r in sorted(rs.items())])
if base and not missing:
    res["estimated_S"] = 93388956 + (tot - float(base))
json.dump(res, open(f"{out}/total.json", "w"), indent=1)
print(f"{out}: {len(rs)} of {K} splits; kept {tot:,.1f} bytes over {n:,} stream bytes"
      + (f"; estimated S {res['estimated_S']:,.0f}" if "estimated_S" in res else "")
      + f"; compress: longest split {res['compress_max_split_hours']:.3f} h, {res['compress_cpu_hours']:.2f} core-hours,"
      f" peak memory per split up to {res['max_peak_gib']:.2f} GiB")
if missing:
    print(f"  splits without a result: {missing}")
if dec:
    np_ = sum(d["passed"] for d in dec)
    print(f"  decodes: {np_} of {len(dec)} PASS ({', '.join(sorted(set(d['mode'] for d in dec)))}),"
          f" longest {res['decode_max_split_hours']:.3f} h" + ("" if np_ == len(dec) else "  ** NOT ALL PASS: " + str([(d['k'], d['mode']) for d in dec if not d['passed']])))
else:
    print("  decodes: none run")
sys.exit(bad)
EOF
}
if [ "${1:-}" = "--plan" ]; then
  echo "# split start end warm_arg (model starts at byte 0 where warm_arg == start, else at the first page start >= start - warm_arg)"
  plan "${2:?stream}" "${3:?K}" "${4:?WARM}"
  exit 0
fi
if [ "${1:-}" = "--summary" ]; then summary "${2:?out dir}" "${3:?K}"; exit $?; fi

B=$(cd "${1:?usage: neif_split_eval.sh BUILD_DIR STREAM K WARM OUT_DIR [PIPE]}" && pwd)
S=$(readlink -f "${2:?stream}")
K=${3:?K}; WARM=${4:?WARM}; PIPE=${6:-0}
mkdir -p "${5:?out dir}"; O=$(cd "$5" && pwd)
STEPS=${STEPS:-compress summary}
DECODE=${DECODE:-verify}; DEC_MT=${DEC_MT:-2}; DEC_PIPE=${DEC_PIPE:-3}
DB=$(cd "${DEC_BUILD:-$B}" && pwd)
SPLITS=${SPLITS:-0-$((K - 1))}; DEC_SPLITS=${DEC_SPLITS:-$SPLITS}
STAGES=1; [ "$PIPE" -ge 2 ] && STAGES=2; [ "$PIPE" -ge 3 ] && STAGES=3
NSPL=$(expand "$SPLITS" | wc -l)
if [ -n "${SLOTS:-}" ]; then :; elif [ -n "${JOBS:-}" ]; then SLOTS=$((JOBS / STAGES)); else SLOTS=$NSPL; fi
[ "$SLOTS" -ge 1 ] || SLOTS=1
PIN=${PIN:-0}
WF=weights-poc8m_w2048_tf2t3_pruned_v5k_np2.c5
[ -x "$B/cmix" ] || { echo "$B/cmix is missing"; exit 1; }
[ -f "$B/$WF" ] || { echo "the weights container is missing in $B"; exit 1; }
flav() { if [ -f "$1/FLAVOUR" ]; then cat "$1/FLAVOUR"; else echo "${FLAVOUR:-unknown}"; fi; }
BSHA=$(sha256sum "$B/cmix" | cut -c1-64); DSHA=$(sha256sum "$DB/cmix" | cut -c1-64)
TIMEV=""; [ -x /usr/bin/time ] && TIMEV=/usr/bin/time
RES=$O/results.tsv
[ -f "$RES" ] || echo -e "utc\thost\tsplit\tstep\tmode\tpipe\tthreads\tflavour\tbinary_sha256\tstart\tend\twarm_from\twall_s\tcpu_s\tpeak_pss_kb\tkept_bits_or_decoded_bytes\tstatus\tcpus" > "$RES"

session_info() {   # TAG: the machine, once per start (before) and at the end (after)
  local f=$O/session_$(hostname)_$SESSION.txt
  {
    echo "===== $1 $(date -u +%FT%TZ) host $(hostname) nproc $(nproc)"
    for c in /sys/fs/cgroup/cpu.max /sys/fs/cgroup/memory.max /sys/fs/cgroup/cpu.stat /sys/fs/cgroup/memory.events \
             /sys/fs/cgroup/cpu/cpu.cfs_quota_us /sys/fs/cgroup/cpu/cpu.stat /sys/fs/cgroup/memory/memory.limit_in_bytes; do
      [ -r $c ] && { echo "--- $c"; cat $c; }
    done
    free -m | sed -n 1,2p; cat /proc/loadavg
    if [ "$1" = before ]; then
      echo "--- args: build $B ($(flav "$B"), sha256 $BSHA) decode build $DB ($(flav "$DB"), sha256 $DSHA) stream $S K=$K WARM=$WARM PIPE=$PIPE"
      echo "--- STEPS=$STEPS SPLITS=$SPLITS DEC_SPLITS=$DEC_SPLITS DECODE=$DECODE SLOTS=$SLOTS PIN=$PIN"
      echo "--- lscpu"; lscpu 2>&1; echo "--- lscpu -e"; lscpu -e 2>&1
      echo "--- cpus usable: $(python3 "$HERE/neif_cpusets.py" --info 2>&1); taskset: $(taskset -cp $$ 2>&1)"
      echo "--- file system of $O"; df -hT "$O" 2>&1; findmnt -T "$O" 2>&1
      echo "--- environment"; env | sort
    fi
  } >> "$f" 2>&1
}

slot_cpus() {   # N W -> array CPUSET[slot]
  CPUSET=()
  if [ "$PIN" = 1 ]; then
    mapfile -t CPUSET < <(python3 "$HERE/neif_cpusets.py" "$1" "$2" 2> "$O/.cpunote")
    echo "pinning: $(cat "$O/.cpunote")"; cat "$O/.cpunote" >> "$O/session_$(hostname)_$SESSION.txt"
  fi
}

emit() {   # k: copy the small result files of a split
  [ -n "${EMIT_DIR:-}" ] || return 0
  mkdir -p "$EMIT_DIR/c$1" && cp -p "$O/c$1"/*.json "$O/c$1"/*.txt "$EMIT_DIR/c$1/" 2> /dev/null
  cp -p "$RES" "$EMIT_DIR/" 2> /dev/null
  return 0
}
line() {   # one result line, appended under a lock
  ( flock 8; echo -e "$(date -u +%FT%TZ)\t$(hostname)\t$*" >> "$RES" ) 8>> "$O/.lock"
}
cpu_of() {   # time.txt -> user + system seconds
  [ -f "$1" ] || { echo 0; return; }
  awk '/User time|System time/ {s += $NF} END {printf "%.1f", s}' "$1"
}
rss_of() { [ -f "$1" ] && awk '/Maximum resident/ {print $NF}' "$1" || echo 0; }

run_compress() {   # k start end warm_arg slot
  local k=$1 d=$O/c$1 cpus=${CPUSET[$5]:-} pre=""
  rm -rf "$d/run"; mkdir -p "$d/run"; cd "$d/run" || return 1
  # the binary reads the weights from its working directory and writes ./.dict and ./ppm.page there
  ln -sf "$B/$WF" .
  [ -n "$cpus" ] && pre="taskset -c $cpus"
  local t0; t0=$(date +%s.%N)
  NEIF_T_START=$2 NEIF_T_END=$3 NEIF_T_WARM=$4 NEIF_T_JSON="$d/score.json.part" NEIF_PIPE=$PIPE \
    $pre ${TIMEV:+$TIMEV -v -o $d/time.txt} "$B/cmix" -t "$S" "$d/out.cmix.part" > "$d/log" 2>&1
  local rc=$? wall; wall=$(python3 -c "import time; print('%.1f' % (time.time() - $t0))")
  rm -rf "$d/run"
  if [ $rc -ne 0 ] || [ ! -s "$d/score.json.part" ]; then
    line "$k\tcompress\t-t\t$PIPE\t0\t$(flav "$B")\t$BSHA\t$2\t$3\t-\t$wall\t$(cpu_of "$d/time.txt")\t0\t0\tFAILED rc $rc\t${cpus:--}"
    echo "split $k: cmix exit $rc (see $d/log)"; return 1
  fi
  if [ "${KEEP_OUT:-1}" = 1 ]; then mv "$d/out.cmix.part" "$d/out.cmix"; else rm -f "$d/out.cmix.part"; fi
  mv "$d/score.json.part" "$d/score.json"
  line "$k\tcompress\t-t\t$PIPE\t0\t$(flav "$B")\t$BSHA\t$(python3 - "$d/score.json" "$wall" "$(cpu_of "$d/time.txt")" "$(rss_of "$d/time.txt")" <<'EOF'
import json, sys
j = json.load(open(sys.argv[1]))
cpu = j.get("cpu_seconds") or float(sys.argv[3])
pss = max(j.get("pipe_peak_pss_kb", 0), j.get("peak_rss_kb", 0), int(sys.argv[4] or 0))
print("%d\t%d\t%d\t%s\t%.1f\t%d\t%.3f\tOK" % (j["start"], j["end"], j["warm_from"], sys.argv[2], cpu, pss, j["kept_bits"]))
EOF
)\t${cpus:--}"
  emit "$k"
  echo "$(date -u +%T) split $k coded: $(tr -d '\n' < "$d/score.json" | cut -c1-160)"
}

run_decode() {   # k start end warm_arg slot
  local k=$1 d=$O/c$1 cpus=${CPUSET[$5]:-} pre="" mode pipe=0 thr=1 name
  [ -s "$d/out.cmix" ] || { echo "split $k: no coded output ($d/out.cmix); run the compress step first"; return 1; }
  rm -rf "$d/drun"; mkdir -p "$d/drun"; cd "$d/drun" || return 1
  ln -sf "$DB/$WF" .
  [ -n "$cpus" ] && pre="taskset -c $cpus"
  local -a ev=(NEIF_T_START=$2 NEIF_T_END=$3 NEIF_T_WARM=$4 NEIF_T_JSON="$d/dec.part")
  case $DECODE in
    verify) mode="verification decode (NEIF_PIPE=$DEC_PIPE -u)"; pipe=$DEC_PIPE; ev+=(NEIF_PIPE=$DEC_PIPE) ;;
    plain) mode="decode (-u, one thread)"; ev+=(NEIF_PIPE=0) ;;
    spec) mode="threaded decode test (-u, NEIF_MT=$DEC_MT NEIF_MT_SPEC=2)"; thr=$((DEC_MT + 1))
          ev+=(NEIF_PIPE=0 NEIF_MT=$DEC_MT NEIF_MT_SPEC=2)
          if [ -n "$cpus" ]; then ev+=(NEIF_MT_CPUS=$cpus); pre="taskset -c ${cpus%%,*}"; else ev+=(NEIF_MT_PIN=0); fi ;;
    *) echo "DECODE must be verify, plain or spec"; return 1 ;;
  esac
  name=dec_$DECODE
  local t0; t0=$(date +%s.%N)
  env "${ev[@]}" $pre ${TIMEV:+$TIMEV -v -o $d/${name}_time.txt} "$DB/cmix" -u "$S" "$d/out.cmix" > "$d/$name.log" 2>&1
  local rc=$? wall; wall=$(python3 -c "import time; print('%.1f' % (time.time() - $t0))")
  rm -rf "$d/drun"
  # the result file of the binary plus exit code, wall time and memory
  python3 - "$d/dec.part" "$d/$name.json" $rc "$wall" "$(cpu_of "$d/${name}_time.txt")" "$(rss_of "$d/${name}_time.txt")" "$DECODE" <<'EOF'
import json, os, sys
try:
    j = json.load(open(sys.argv[1]))
except Exception:
    j = {}
j.update(rc=int(sys.argv[3]), wall_seconds=float(sys.argv[4]), time_cpu_seconds=float(sys.argv[5]), mode=sys.argv[7])
j["peak_rss_kb"] = max(int(sys.argv[6] or 0), j.get("peak_rss_kb", 0))
json.dump(j, open(sys.argv[2], "w"), indent=1)
if os.path.exists(sys.argv[1]): os.remove(sys.argv[1])
EOF
  local st=PASS; [ $rc -eq 0 ] || st="FAILED rc $rc"
  line "$k\tdecode\t$mode\t$pipe\t$thr\t$(flav "$DB")\t$DSHA\t$(python3 - "$d/$name.json" <<'EOF'
import json, sys
j = json.load(open(sys.argv[1]))
cpu = j.get("cpu_seconds") or j["time_cpu_seconds"]
print("%s\t%s\t%s\t%.1f\t%.1f\t%d\t%s" % (j.get("start", "-"), j.get("end", "-"), j.get("warm_from", "-"), j["wall_seconds"], cpu,
      max(j.get("pipe_peak_pss_kb", 0), j["peak_rss_kb"]), j.get("decoded_bytes", 0)))
EOF
)\t$st\t${cpus:--}"
  emit "$k"
  echo "$(date -u +%T) split $k $mode: $st, $(grep -a -o 'neif_sv: [A-Z ]*[^\r]*' "$d/$name.log" | tail -n 1 | cut -c1-120) wall $wall s"
  [ $rc -eq 0 ]
}

pool() {   # FUNC SPLIT_LIST: run FUNC for the listed splits, SLOTS at a time, each in a free slot
  local fn=$1 list=$2 k s e w slot fifo=$O/.slots.$$
  rm -f "$fifo"; mkfifo "$fifo"; exec 9<> "$fifo"; rm -f "$fifo"
  for ((slot = 0; slot < SLOTS; slot++)); do echo $slot >&9; done
  while read -r k s e w; do
    echo " $(expand "$list" | tr '\n' ' ') " | grep -q " $k " || continue
    if [ $fn = run_compress ] && [ -f "$O/c$k/score.json" ] && { [ -s "$O/c$k/out.cmix" ] || [[ " $STEPS " != *" decode "* ]]; }; then
      echo "split $k: already coded"; continue
    fi
    if [ $fn = run_decode ] && [ -f "$O/c$k/dec_$DECODE.json" ] && grep -q '"rc": 0' "$O/c$k/dec_$DECODE.json"; then
      echo "split $k: already decoded ($DECODE), PASS"; continue
    fi
    mkdir -p "$O/c$k"
    read -r -u 9 slot
    ( $fn "$k" "$s" "$e" "$w" "$slot" < /dev/null || echo "$k" >> "$O/.failed.$$"; echo "$slot" >&9 ) &
  done < "$O/plan.tsv"
  wait
  exec 9>&-
}

SESSION=$(date -u +%Y%m%dT%H%M%SZ)
echo "$(date -u) split eval: build $B ($(flav "$B"), ${BSHA:0:16}) stream $S K=$K WARM=$WARM PIPE=$PIPE steps: $STEPS; splits $SPLITS, $SLOTS at a time, nproc=$(nproc)"
plan "$S" "$K" "$WARM" > "$O/plan.tsv"
cat "$O/plan.tsv"
session_info before
rm -f "$O/.failed.$$"
fail=0
for step in $STEPS; do
  case $step in
    compress) slot_cpus "$SLOTS" "${PIN_W:-$STAGES}"; pool run_compress "$SPLITS" ;;
    decode)
      w=1; [ "$DECODE" = verify ] && w=$DEC_PIPE; [ "$DECODE" = spec ] && w=$((DEC_MT + 1))
      slot_cpus "$SLOTS" "${PIN_W:-$w}"; pool run_decode "$DEC_SPLITS" ;;
    summary) summary "$O" "$K" || fail=1 ;;
    *) echo "unknown step $step"; fail=1 ;;
  esac
done
[ -s "$O/.failed.$$" ] && { echo "failed splits: $(sort -n "$O/.failed.$$" | tr '\n' ' ')"; fail=1; rm -f "$O/.failed.$$"; }
session_info after
[ -n "${EMIT_DIR:-}" ] && cp -p "$O"/session_*.txt "$O/results.tsv" "$O/total.json" "$EMIT_DIR/" 2> /dev/null
echo "$(date -u) done, exit $fail"
exit $fail
