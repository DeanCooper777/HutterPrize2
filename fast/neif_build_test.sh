#!/bin/bash
# neif_build_test.sh BUILD_DIR [DEFS] [FLAVOUR]: one test build of this source tree.
#
# The entry's own recipe `make prof_use` (profile-guided with the shipped profile pgo_data/, LTO,
# clang++-18 / lld-18, as comp9 is compiled) in BUILD_DIR, which gets copies of src/, makefile,
# encoder_split.lds, runtime_exports.map, pgo_data/ and the weights container, with the macros
# DEFS added on every compile line through CC. No packer, no embedded blocks: the result
# BUILD_DIR/cmix serves the chunk modes -t / -u (it reads the weights container from its
# WORKING directory).
#
#   DEFS     default "-DNEIF_TEST=1 -DNEIF_PACKED=1" (the packed test build, "tbK")
#            "-DNEIF_TEST=1"                               original test build ("tbN")
#            "-DNEIF_TEST=1 -DNEIF_MT=1 -DNEIF_PACKED=1"   packed + threads ("tbKM")
#   FLAVOUR  a name written to BUILD_DIR/FLAVOUR; the run scripts put it into their result lines
#            (default: derived from DEFS)
# Environment: NEIF_SRC (default: the directory above this script = the source tree),
#              JOBS (default: nproc), MAKE_TARGET (default prof_use; `cmix` = plain build without
#              profile and LTO, whose coder output DIFFERS: debugging only).
# Prints the size and sha256 of the binary and the number of functions whose PGO profile entry
# was discarded (18 for tbN and tbK, 19 for tbKM on our machines; Predict, Perceive, Encode and
# Decode must not be among them: the list is printed).
# About 15 s on 8 cores. Needs the toolchain of install.sh (Ubuntu 24.04, clang 18.1.3).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${NEIF_SRC:-$(cd "$HERE/.." && pwd)}
B=${1:?usage: neif_build_test.sh BUILD_DIR [DEFS] [FLAVOUR]}
X=${2:--DNEIF_TEST=1 -DNEIF_PACKED=1}
WF=weights-poc8m_w2048_tf2t3_pruned_v5k_np2.c5
if [ -n "${3:-}" ]; then FL=$3
else
  FL=orig
  case "$X" in *NEIF_PACKED*) FL=packed ;; esac
  case "$X" in *NEIF_MT*) FL=$FL+mt ;; esac
  case "$X" in *NEIF_TEST*) ;; *) FL=$FL-notest ;; esac
fi
command -v clang++-18 > /dev/null || { echo "clang++-18 is missing: run install.sh of the source tree first (Ubuntu 24.04)"; exit 1; }
rm -rf "$B"; mkdir -p "$B"; B=$(cd "$B" && pwd)
(cd "$SRC" && cp -a src makefile encoder_split.lds runtime_exports.map pgo_data $WF "$B/")
cd "$B"
if ! make -j"${JOBS:-$(nproc)}" "${MAKE_TARGET:-prof_use}" CC="clang++-18 $X" > build.log 2>&1; then
  grep -n "error" build.log | head -n 40; tail -n 5 build.log; exit 1
fi
echo "$FL" > FLAVOUR
echo "warnings: $(grep -c 'warning:' build.log || true), PGO profile entries discarded: $(grep -c 'hash mismatch' build.log || true)"
grep 'hash mismatch' build.log | sed 's/.*(hash mismatch) //; s/ Hash.*//' | sort | tr '\n' ' '; echo
echo "$B/cmix ($FL; $X): $(stat -c %s cmix) B, sha256 $(sha256sum cmix | cut -c1-64)"
