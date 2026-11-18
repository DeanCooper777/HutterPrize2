#!/bin/bash
# neif_setup.sh: prepare a Linux machine for the test modes of this source tree (fast/README.md):
# compiler, coder stream, three test builds, self-check. Run as root on a machine (or in a
# container / chroot) that you may change: the step `deps` runs upstream's install.sh, which
# repoints apt to a dated Ubuntu snapshot.
#     bash fast/neif_setup.sh
# Steps (each is skipped when its result is there; STEPS="deps stream build check"):
#   deps    the compiler. The entry is built with clang-18 / lld-18 (NOT gcc; gcc only supplies
#           libstdc++ 13 headers), profile-guided with LTO. install.sh pins every package to the
#           Ubuntu 24.04 archive as of 2026-09-27 (clang 18.1.3-1ubuntu1) through the Ubuntu
#           snapshot service, so the machine must run UBUNTU 24.04 (apt 2.7 or later).
#           Whether another clang 18 point release gives the same model output is NOT known: the
#           float code of most files is compiled with -ffp-contract=off and explicit intrinsics,
#           but a plain (non-PGO) build of this source already differs from the entry. Hence the
#           self-check below; if it fails, do not trust any number from that machine.
#           If clang++-18 and ld.lld-18 are already there, install.sh is NOT run.
#   stream  the coder stream $NEIF_ROOT/stream.bin (what `-e enwik9` hands to the arithmetic
#           coder: 501,062,281 B): unpacked from $NEIF_ROOT/stream.bin.zst if that is there
#           (zstd --long=27), and its sha256 checked. If neither file exists the step stops and
#           says how to make the stream from enwik9 (neif_mkstream.sh).
#   build   three PGO test builds in $NEIF_ROOT/build (neif_build_test.sh), each as the entry is
#           compiled (make prof_use: the entry's shipped profile and LTO) with extra macros:
#             tbN   -DNEIF_TEST=1                              the original test build
#             tbK   -DNEIF_TEST=1 -DNEIF_PACKED=1              packed dense kernels
#             tbKM  -DNEIF_TEST=1 -DNEIF_MT=1 -DNEIF_PACKED=1  packed + threads (speculative step)
#           about 15 s each on 8 cores. BUILDS="tbK" limits the list.
#   check   self-check: the chunk 60.00-60.03 M must code to sha256 9e925d7846f34983... with
#           every build; then with the packed build and the 3-process pipeline:
#           60.0-60.5 M (100 k warm-up) -> e3c7340dce4cef65, the reference chunk 250.0-250.3 M
#           (1.2 M warm-up) -> 727e97767d69fb56, the verification decode of the first chunk and
#           the threaded decode test with 20 % forced wrong guesses.
#           CHECK=quick: only the 36 k chunk (about 1.5 min per run). CHECK=full is the default
#           (about 25 min on a 2024 laptop). Prints SELF-CHECK PASSED or FAILED.
# Environment:
#   NEIF_ROOT  working directory (default fast/work): stream.bin, build/, out/. It must be on a
#              LOCAL disk with 40 GB free (every running cmix makes a sparse 14.7 GB paging file
#              in its run directory). Memory: 10 GiB per run.
#   NEIF_SRC   the source tree (default: the directory above this script)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${NEIF_SRC:-$(cd "$HERE/.." && pwd)}
export NEIF_SRC=$SRC
export NEIF_ROOT=${NEIF_ROOT:-$HERE/work}
mkdir -p "$NEIF_ROOT"; NEIF_ROOT=$(cd "$NEIF_ROOT" && pwd); R=$NEIF_ROOT
STEPS=${STEPS:-deps stream build check}
BUILDS=${BUILDS:-tbN tbK tbKM}
STREAM_SHA=c35d3073eeba70f22a2e08a50202b53eb95466e3e9f1f2702f6f9b3f96fd8718
say() { echo "$(date -u +%T) $*"; }
has() { case " $BUILDS " in *" $1 "*) return 0 ;; esac; return 1; }
for step in $STEPS; do
  case $step in
    deps)
      . /etc/os-release
      [ "${VERSION_ID:-}" = 24.04 ] || say "WARNING: this is ${PRETTY_NAME:-unknown}, not Ubuntu 24.04: install.sh expects 24.04 (apt snapshot); the self-check decides"
      if command -v clang++-18 > /dev/null && command -v ld.lld-18 > /dev/null; then say "clang-18 is there: $(clang++-18 --version | head -n 1)"
      else
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -q > "$R/apt.log" 2>&1
        apt-get install -y -q --no-install-recommends ca-certificates time file zstd patch procps util-linux python3 >> "$R/apt.log" 2>&1
        (cd "$SRC" && bash install.sh > "$R/install.log" 2>&1) || { say "install.sh failed, see $R/install.log"; tail -n 5 "$R/install.log"; exit 1; }
      fi
      for t in python3 taskset flock make; do command -v $t > /dev/null || say "WARNING: $t is missing"; done
      [ -x /usr/bin/time ] || say "note: /usr/bin/time is missing (package 'time'): cpu seconds and peak memory of one-process runs will be 0"
      say "compiler: $(clang++-18 --version | head -n 1); $(dpkg-query -W -f='${Package}=${Version} ' clang-18 lld-18 libstdc++-13-dev libc6 2> /dev/null)" ;;
    stream)
      if [ ! -f "$R/stream.bin" ]; then
        if [ -f "$R/stream.bin.zst" ]; then zstd -d -q --long=27 "$R/stream.bin.zst" -o "$R/stream.bin" || exit 1
        else
          say "no $R/stream.bin (and no stream.bin.zst). Make it from enwik9:"
          say "    bash $HERE/neif_mkstream.sh /path/to/enwik9"
          exit 1
        fi
      fi
      n=$(stat -L -c %s "$R/stream.bin"); h=$(sha256sum "$R/stream.bin" | cut -c1-64)
      if [ "$n" = 501062281 ] && [ "$h" = "$STREAM_SHA" ]; then say "stream.bin: $n bytes, sha256 ok"
      else say "stream.bin: $n bytes sha256 $h, EXPECTED 501062281 bytes $STREAM_SHA"; exit 1; fi ;;
    build)
      mkdir -p "$R/build"
      for f in "tbN:orig:-DNEIF_TEST=1" "tbK:packed:-DNEIF_TEST=1 -DNEIF_PACKED=1" "tbKM:packed+mt:-DNEIF_TEST=1 -DNEIF_MT=1 -DNEIF_PACKED=1"; do
        d=${f%%:*}; rest=${f#*:}; flav=${rest%%:*}; x=${rest#*:}
        has "$d" || continue
        B=$R/build/$d
        if [ -x "$B/cmix" ] && [ -f "$B/FLAVOUR" ]; then say "$d: already built"; continue; fi
        bash "$HERE/neif_build_test.sh" "$B" "$x" "$flav" > "$R/build/$d.out" 2>&1 || { say "$d: build failed"; tail -n 20 "$R/build/$d.out"; exit 1; }
        say "$d ($flav): $(sha256sum "$B/cmix" | cut -c1-64), PGO profile entries discarded: $(grep -c 'hash mismatch' "$B/build.log")"
      done ;;
    check)
      export STREAM=$R/stream.bin OUT=$R/out/selfcheck
      ok=1
      chk() {   # NAME EXPECTED_SHA16
        local got; got=$(awk -F'\t' -v n="$1" '$3 == n {s = $20} END {print s}' "$OUT/results.tsv")
        if [ "$got" = "$2" ]; then say "  $1: sha $got ok"; else say "  $1: sha $got, EXPECTED $2  ** MISMATCH"; ok=0; fi
      }
      for d in tbN tbK tbKM; do
        has "$d" || continue
        bash "$HERE/neif_run.sh" q_$d t 60000000 60030000 10000 "$R/build/$d" > /dev/null; chk q_$d 9e925d7846f34983
      done
      if has tbK; then
        NEIF_PIPE=3 bash "$HERE/neif_run.sh" q_tbK_p3 t 60000000 60030000 10000 "$R/build/tbK" > /dev/null; chk q_tbK_p3 9e925d7846f34983
        NEIF_PIPE=3 bash "$HERE/neif_run.sh" q_tbK_v3 u 60000000 60030000 10000 "$R/build/tbK" q_tbK > /dev/null \
          && say "  q_tbK_v3: verification decode PASS" || { say "  q_tbK_v3: verification decode FAILED"; ok=0; }
      fi
      if has tbK && has tbKM; then
        NEIF_MT=2 NEIF_MT_SPEC=2 NEIF_MT_FAULT=200 bash "$HERE/neif_run.sh" q_tbKM_s2f u 60000000 60030000 10000 "$R/build/tbKM" q_tbK > /dev/null \
          && say "  q_tbKM_s2f: threaded decode test with 20 % forced wrong guesses PASS" || { say "  q_tbKM_s2f: FAILED"; ok=0; }
      fi
      if [ "${CHECK:-full}" = full ] && has tbK; then
        NEIF_PIPE=3 bash "$HERE/neif_run.sh" c60_tbK_p3 t 60000000 60500000 100000 "$R/build/tbK" > /dev/null; chk c60_tbK_p3 e3c7340dce4cef65
        NEIF_PIPE=3 bash "$HERE/neif_run.sh" ref_tbK_p3 t 250000000 250300000 1200000 "$R/build/tbK" > /dev/null; chk ref_tbK_p3 727e97767d69fb56
      fi
      if [ $ok = 1 ]; then say "SELF-CHECK PASSED"; else say "SELF-CHECK FAILED (results in $OUT/results.tsv)"; exit 1; fi ;;
    *) say "unknown step $step"; exit 1 ;;
  esac
done
