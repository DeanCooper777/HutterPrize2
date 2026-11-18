#!/bin/bash
# neif_build_official.sh [OUT_DIR]: the OFFICIAL build of this source tree (upstream's build.sh:
# PGO + LTO + packer, no test macro) and the comparison of its comp9 with the published one.
#
# It shows that the additions behind NEIF_TEST / NEIF_PACKED / NEIF_MT leave the contest entry
# unchanged: comp9 must have sha256
#   976bf2d309707ab555e0c92720da5508ff0e5e6b846fc7082234262b06767f82   (4,042,564 B)
# which is the value in upstream's environment.txt and README.cmix-neif.md.
# OUT_DIR (default $NEIF_ROOT/official, NEIF_ROOT default fast/work) receives comp9 and build.log.
# Environment: NEIF_SRC (default: the directory above this script).
# Needs the toolchain of install.sh (Ubuntu 24.04, clang 18.1.3). About 7 min on 8 cores; the
# profile pass needs several GiB of memory, so do not run it beside a large cmix run.
# Exit code 0 only if the sha256 matches.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${NEIF_SRC:-$(cd "$HERE/.." && pwd)}
O=${1:-${NEIF_ROOT:-$HERE/work}/official}
WANT=976bf2d309707ab555e0c92720da5508ff0e5e6b846fc7082234262b06767f82
mkdir -p "$O"; O=$(cd "$O" && pwd)
# build.sh copies its whole directory into the build; hand it a clean copy of the source without
# the git data and without the working directory of these scripts
T=$O/.src; rm -rf "$T"; mkdir -p "$T"
(cd "$SRC" && tar cf - --exclude=./.git --exclude=./fast/work --exclude=./.build --exclude=./comp9 .) | tar xf - -C "$T"
cd "$O"
s=$(date +%s)
rc=0; bash "$T/build.sh" > "$O/build.log" 2>&1 || rc=$?
rm -rf "$T"
echo "build.sh exit $rc in $(( $(date +%s) - s )) s"; tail -n 2 "$O/build.log"
[ -s "$O/comp9" ] || { echo "no comp9 was built, see $O/build.log"; exit 1; }
h=$(sha256sum "$O/comp9" | cut -d' ' -f1)
echo "comp9 $(stat -c %s "$O/comp9") B sha256 $h"
if [ "$h" = "$WANT" ]; then echo "OFFICIAL BUILD: IDENTICAL to the published comp9"
else echo "OFFICIAL BUILD: DIFFERENT from the published comp9 ($WANT)"; exit 1; fi
