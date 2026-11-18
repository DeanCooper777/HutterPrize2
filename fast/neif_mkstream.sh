#!/bin/bash
# neif_mkstream.sh ENWIK9 [STREAM_OUT]: produce the coder stream of cmix-neif 2.0 once: the file
# the arithmetic coder codes in `-e enwik9` (after all preprocessing; 501,062,281 B, sha256
# c35d3073eeba70f22a2e08a50202b53eb95466e3e9f1f2702f6f9b3f96fd8718). Every test mode of fast/
# works on this file.
#
# Uses upstream's own tool cmix_stream (training/tf/1_dump/build_dump_tools.sh with
# trainmodes.patch, TRAIN_MODE 1), which runs the whole `-e` preprocessing and stops before the
# coder. trainmodes.patch applies to the UNMODIFIED upstream source only, so the tool is built
# from a pristine copy:
#   NEIF_PRISTINE   a directory with cmix-neif 2.0 at commit 35d1957, unmodified; default: taken
#                   with `git archive` from the commit of this repository whose tree is upstream's
#                   (tree 0084a8876b81e7eac41903337dbb8b4762f0e6c7), which needs git and a
#                   non-shallow clone
# STREAM_OUT default $NEIF_ROOT/stream.bin (NEIF_ROOT default fast/work). Work files go to
# $NEIF_ROOT/mkstream (removed at the end, the tool stays in $NEIF_ROOT/tools).
# Needs the toolchain of install.sh and the package `patch`; several GiB of memory; about 2 min.
# (Another way to get the same file: a test build that can run `-e` (neif_e_build.sh) leaves the
# stream as OUT.cmix.temp while its coder runs; neif_e_run.sh keeps a copy with KEEP_STREAM=1.)
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${NEIF_SRC:-$(cd "$HERE/.." && pwd)}
R=${NEIF_ROOT:-$HERE/work}
E=$(readlink -f "${1:?usage: neif_mkstream.sh ENWIK9 [STREAM_OUT]}")
mkdir -p "$R"; R=$(cd "$R" && pwd)
OUT=${2:-$R/stream.bin}
TREE=0084a8876b81e7eac41903337dbb8b4762f0e6c7
WANT=c35d3073eeba70f22a2e08a50202b53eb95466e3e9f1f2702f6f9b3f96fd8718
[ -f "$E" ] || { echo "$E: no such file"; exit 1; }
command -v patch > /dev/null || { echo "the package 'patch' is missing"; exit 1; }
W=$R/mkstream; rm -rf "$W"; mkdir -p "$W/run" "$R/tools"
if [ ! -x "$R/tools/cmix_stream" ]; then
  P=${NEIF_PRISTINE:-}
  if [ -z "$P" ]; then
    c=$(git -C "$SRC" log --format='%H %T' | awk -v t=$TREE '$2 == t {print $1; exit}')
    [ -n "$c" ] || { echo "no commit with upstream's tree in the git history of $SRC: set NEIF_PRISTINE"; exit 1; }
    P=$W/pristine; mkdir -p "$P"; git -C "$SRC" archive "$c" | tar xf - -C "$P"
  fi
  bash "$P/training/tf/1_dump/build_dump_tools.sh" "$P" "$R/tools" stream
fi
cd "$W/run"
cp "$SRC/dictionary/english.dic" .dict
cp "$SRC/src/readalike_prepr/data/new_article_order" .new_article_order
"$R/tools/cmix_stream" -e "$E" out > run.log 2> run.err || { tail -n 30 run.err; exit 1; }
mv out.cmix.temp "$OUT"
cd "$R"; rm -rf "$W"
n=$(stat -c %s "$OUT"); h=$(sha256sum "$OUT" | cut -c1-64)
echo "$OUT: $n bytes, sha256 $h"
if [ "$n" = 501062281 ] && [ "$h" = "$WANT" ]; then echo "stream ok"; else echo "EXPECTED 501062281 bytes, sha256 $WANT"; exit 1; fi
