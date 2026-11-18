#!/bin/bash
# neif_scale.sh [OUT_DIR]: scaling test of the fast flavour of cmix-neif 2.0 on a many-core
# machine (see fast/README.md). Two parts (environment STEPS, default "scale loaded"):
#
# scale   one run at a time on the window START-END (default 60.0-60.3 M, 100 k warm-up), each
#         bound to CPUs of its own physical cores (neif_cpusets.py):
#           s00_orig_t     original test build, one process, -t
#           s01_pack_t     packed build, one process, -t
#           s02_pack_u     packed build, one process, -u (the real decoder, one thread)
#           s03_pipe2_t    packed, compression pipeline of 2 processes
#           s04_pipe3_t    packed, compression pipeline of 3 processes
#           s05_verify3_u  packed, VERIFICATION DECODE (NEIF_PIPE=3 -u)
#           s06..s09       packed + threads build, -u with the speculative step, guess from the
#                          stream (NEIF_MT_SPEC=2), NEIF_MT = 1, 2, 3, 4 (transformer thread + 0..3
#                          workers beside the main thread)
#           s10..s12       the same with 23 % of the guesses forced wrong, NEIF_MT = 1, 2, 3
#           s13_orig_t2    the first run again (drift of the machine)
#         About 35 min on a Core Ultra 288V laptop.
# loaded  throughput with every slot busy at once, on the short window LSTART-LEND (default
#         60.0-60.1 M, 50 k warm-up), packed build, configurations LOADED (default
#         "16:0 16:2 8:3 16:3 5:3", N:PIPE = N runs at once, each a single process (0) or a pipeline
#         of 2 or 3 processes); every run is bound to its own cpu set (N sets of 1, 2 or 3 CPUs:
#         distinct physical cores if there are enough, else SMT siblings shared or oversubscribed,
#         which the output states). Reported: the sum of the runs' stream bytes per second.
#         N is cut to what the free memory allows at 10 GiB per run.
#
# Environment: NEIF_ROOT (default fast/work) with build/tbN, build/tbK, build/tbKM (override: BN,
# BK, BKM) and stream.bin (override: STREAM). OUT_DIR default $NEIF_ROOT/out/scale; put it on a
# LOCAL disk. Resumable: a run whose line in OUT_DIR/results.tsv ends OK/PASS is not repeated.
# Results: OUT_DIR/results.tsv (one line per run: wall, model seconds, cpu seconds, peak memory,
# binary sha256, pipe, threads, cpus), OUT_DIR/loaded.tsv, OUT_DIR/session_*.txt (machine, cgroup
# limits, throttling and OOM counters before and after), OUT_DIR/summary.txt.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
export NEIF_ROOT=${NEIF_ROOT:-$HERE/work}
BN=${BN:-$NEIF_ROOT/build/tbN}; BK=${BK:-$NEIF_ROOT/build/tbK}; BKM=${BKM:-$NEIF_ROOT/build/tbKM}
export STREAM=${STREAM:-$NEIF_ROOT/stream.bin}
export OUT=${1:-$NEIF_ROOT/out/scale}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
S=${START:-60000000}; E=${END:-60300000}; WM=${WARM:-100000}
LS=${LSTART:-60000000}; LE=${LEND:-60100000}; LW=${LWARM:-50000}
STEPS=${STEPS:-scale loaded}; LOADED=${LOADED:-16:0 16:2 8:3 16:3 5:3}
SESSION=$OUT/session_$(hostname)_$(date -u +%Y%m%dT%H%M%SZ).txt
unset NEIF_PIPE NEIF_MT NEIF_MT_SPEC NEIF_MT_FAULT NEIF_ACT16 CPUS

info() {
  {
    echo "===== $1 $(date -u +%FT%TZ) host $(hostname) nproc $(nproc)"
    for c in /sys/fs/cgroup/cpu.max /sys/fs/cgroup/memory.max /sys/fs/cgroup/cpu.stat /sys/fs/cgroup/memory.events \
             /sys/fs/cgroup/memory.peak /sys/fs/cgroup/cpuset.cpus.effective \
             /sys/fs/cgroup/cpu/cpu.cfs_quota_us /sys/fs/cgroup/cpu/cpu.stat /sys/fs/cgroup/memory/memory.limit_in_bytes \
             /sys/fs/cgroup/memory/memory.usage_in_bytes /sys/fs/cgroup/memory/memory.max_usage_in_bytes \
             /sys/fs/cgroup/memory/memory.failcnt /sys/fs/cgroup/memory/memory.oom_control \
             /sys/fs/cgroup/cpuset/cpuset.effective_cpus; do
      [ -r $c ] && { echo "--- $c"; cat $c; }
    done
    free -m | sed -n 1,2p; cat /proc/loadavg; df -k "$OUT" | tail -n 1
    if [ "$1" = before ]; then
      for b in "$BN" "$BK" "$BKM"; do echo "--- build $b: $(cat "$b/FLAVOUR" 2> /dev/null) sha256 $(sha256sum "$b/cmix" 2>&1 | cut -c1-64)"; done
      echo "--- stream $STREAM $(stat -L -c %s "$STREAM" 2>&1) bytes; windows $S-$E warm $WM, loaded $LS-$LE warm $LW; LOADED=$LOADED"
      echo "--- lscpu"; lscpu 2>&1; echo "--- lscpu -e"; lscpu -e 2>&1
      echo "--- cpus usable: $(python3 "$HERE/neif_cpusets.py" --info 2>&1); taskset: $(taskset -cp $$ 2>&1)"
      echo "--- file system of $OUT"; df -hT "$OUT" 2>&1; findmnt -T "$OUT" 2>&1
      echo "--- environment"; env | sort
    fi
  } >> "$SESSION" 2>&1
}
cg() {   # first readable of the given cgroup files (v2 name, v1 name)
  local f; for f in "$@"; do [ -r "/sys/fs/cgroup/$f" ] && { cat "/sys/fs/cgroup/$f"; return; }; done; echo 0
}
mem_avail_gib() {   # the smaller of the machine's MemAvailable and the container's limit - usage (page cache not counted)
  python3 - "$(cg memory.max memory/memory.limit_in_bytes)" "$(cg memory.current memory/memory.usage_in_bytes)" \
    "$(cat /sys/fs/cgroup/memory.stat /sys/fs/cgroup/memory/memory.stat 2> /dev/null | awk '$1 == "file" || $1 == "cache" {print $2; exit}')" <<'EOF'
import sys
a = [l.split() for l in open("/proc/meminfo")]
avail = int([x[1] for x in a if x[0] == "MemAvailable:"][0]) * 1024
lim, use, cache = sys.argv[1], int(sys.argv[2] or 0), int(sys.argv[3] or 0)
if lim.isdigit() and int(lim) < 2**60: avail = min(avail, int(lim) - (use - cache))
print(avail // 2**30)
EOF
}
monitor() {   # CFG: every 10 s one line in monitor.tsv while a loaded configuration runs
  [ -f "$OUT/monitor.tsv" ] || echo -e "utc\tconfig\tcgroup_mem_bytes\tcgroup_cache_bytes\tout_dir_du_kb\tfs_used_kb\tfs_avail_kb\toom_kills\tnr_throttled\tprocs_cmix\tloadavg1" > "$OUT/monitor.tsv"
  while :; do
    echo -e "$(date -u +%T)\t$1\t$(cg memory.current memory/memory.usage_in_bytes)\t$(cat /sys/fs/cgroup/memory.stat /sys/fs/cgroup/memory/memory.stat 2> /dev/null | awk '$1 == "file" || $1 == "cache" {print $2; exit}')\t$(du -sk "$OUT" 2> /dev/null | cut -f1)\t$(df -k "$OUT" | awk 'END {print $3 "\t" $4}')\t$(cat /sys/fs/cgroup/memory.events /sys/fs/cgroup/memory/memory.oom_control 2> /dev/null | awk '$1 == "oom_kill" {print $2; exit}')\t$(cat /sys/fs/cgroup/cpu.stat /sys/fs/cgroup/cpu/cpu.stat 2> /dev/null | awk '$1 == "nr_throttled" {print $2; exit}')\t$(pgrep -c -x cmix)\t$(cut -d' ' -f1 /proc/loadavg)" >> "$OUT/monitor.tsv"
    sleep 10
  done
}
done_ok() { [ -f "$OUT/results.tsv" ] && awk -F'\t' -v n="$1" '$3 == n && ($(NF-1) == "OK" || $(NF-1) == "PASS") {f = 1} END {exit !f}' "$OUT/results.tsv"; }
cpus() { python3 "$HERE/neif_cpusets.py" 1 "$1" 2> /dev/null | head -n 1; }
run() {   # NAME W MODE BUILD [CODED] (switches in the environment)
  local name=$1 w=$2
  if done_ok "$name"; then echo "$name: already done"; return 0; fi
  CPUS=$(cpus "$w") bash "$HERE/neif_run.sh" "$name" "$3" $S $E $WM "$4" "${5:-}" | cut -f3,4,5,7-10,15-22 | tr '\t' ' '
}

info before
for step in $STEPS; do
  case $step in
    scale)
      run s00_orig_t 1 t "$BN"
      run s01_pack_t 1 t "$BK"
      run s02_pack_u 1 u "$BK" s01_pack_t
      NEIF_PIPE=2 run s03_pipe2_t 2 t "$BK"
      NEIF_PIPE=3 run s04_pipe3_t 3 t "$BK"
      NEIF_PIPE=3 run s05_verify3_u 3 u "$BK" s01_pack_t
      i=6
      for n in 1 2 3 4; do
        NEIF_MT=$n NEIF_MT_SPEC=2 run s0${i}_spec${n}_u $((n + 1)) u "$BKM" s01_pack_t; i=$((i + 1))
      done
      for n in 1 2 3; do
        NEIF_MT=$n NEIF_MT_SPEC=2 NEIF_MT_FAULT=230 run s${i}_spec${n}f230_u $((n + 1)) u "$BKM" s01_pack_t; i=$((i + 1))
      done
      run s13_orig_t2 1 t "$BN" ;;
    loaded)
      [ -f "$OUT/loaded.tsv" ] || echo -e "utc\tconfig\truns\tprocesses_per_run\tcpu_sets\tstream_bytes_per_s_total\tmodel_s_min\tmodel_s_max\tpeak_pss_kb_sum\tall_ok" > "$OUT/loaded.tsv"
      for cfg in $LOADED; do
        # N:PIPE or N:PIPE:CPUS (CPUS = logical CPUs per run, default the number of processes)
        IFS=: read -r n pipe cw <<< "$cfg"
        w=1; [ "$pipe" -ge 2 ] && w=$pipe
        tag=l${n}x${w}; [ -n "${cw:-}" ] && { tag=${tag}c$cw; w=$cw; }
        if grep -q "	$cfg	" "$OUT/loaded.tsv"; then echo "loaded $cfg: already done"; continue; fi
        availg=$(mem_avail_gib)
        if [ $((n * 10)) -gt "$availg" ]; then echo "loaded $cfg: $availg GiB free, cut from $n to $((availg / 10)) runs"; n=$((availg / 10)); fi
        [ "$n" -ge 1 ] || continue
        mapfile -t SETS < <(python3 "$HERE/neif_cpusets.py" "$n" "$w" 2> "$OUT/.note")
        note=$(sed 's/.*: //' "$OUT/.note")
        echo "$(date -u +%T) loaded $cfg: $n runs x $w processes, $note"
        monitor "$cfg" & mon=$!
        pids=()
        for ((j = 0; j < n; j++)); do
          ( CPUS=${SETS[$j]} NEIF_PIPE=$pipe bash "$HERE/neif_run.sh" "${tag}_$(printf %02d $j)" t $LS $LE $LW "$BK" > /dev/null 2>&1 ) &
          pids+=($!)
        done
        wait "${pids[@]}"
        kill $mon 2> /dev/null; wait $mon 2> /dev/null
        python3 - "$OUT" "$tag" "$cfg" "$n" "$w" "$note" <<'EOF' | tee -a "$OUT/loaded.tsv"
import json, sys, time
out, tag, cfg, n, w, note = sys.argv[1:7]
bps = 0.0; ms = []; pss = 0; ok = True
for j in range(int(n)):
    try:
        r = json.load(open("%s/%s_%02d/score.json" % (out, tag, j)))
        bps += r["model_bytes_per_second"]; ms.append(r["model_seconds"])
        pss += max(r.get("pipe_peak_pss_kb", 0), r.get("peak_rss_kb", 0))
    except Exception:
        ok = False
print("\t".join(str(x) for x in [time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), cfg, n, w, note, "%.1f" % bps,
                                 "%.1f" % min(ms or [0]), "%.1f" % max(ms or [0]), pss, "yes" if ok else "NO"]))
EOF
        rm -f "$OUT"/${tag}_*/out.cmix
      done ;;
  esac
done
info after
python3 - "$OUT" <<'EOF' | tee "$OUT/summary.txt"
import sys
out = sys.argv[1]
rows = {}
try:
    lines = open(out + "/results.tsv").read().splitlines()
    hdr = lines[0].split("\t")
    for ln in lines[1:]:
        f = dict(zip(hdr, ln.split("\t")))
        if f["name"].startswith("s"): rows[f["name"]] = f
except Exception as ex:
    print("no results:", ex)
base = float(rows["s00_orig_t"]["model_s"]) if "s00_orig_t" in rows else 0
pack = float(rows["s01_pack_t"]["model_s"]) if "s01_pack_t" in rows else 0
print("%-20s %-10s %8s %8s %9s %9s %10s  %s" % ("run", "status", "model s", "cpu s", "x orig", "x packed", "peak GiB", "cpus"))
for name in sorted(rows):
    f = rows[name]; m = float(f["model_s"])
    print("%-20s %-10s %8.1f %8.1f %9s %9s %10.2f  %s" % (name, f["status"], m, float(f["cpu_s"]),
          "%.2f" % (base / m) if base and m else "-", "%.2f" % (pack / m) if pack and m else "-", int(f["peak_pss_kb"]) / 2**20, f["cpus"]))
try:
    print(open(out + "/loaded.tsv").read())
except Exception:
    pass
EOF
echo "scale test done: $OUT"
