#!/bin/bash
# neif_e_build.sh OFFICIAL_COMP9 [BUILD_DIR] [DEFS]: a NEIF_TEST build that can run the NORMAL
# compress command (`cmix -e enwik9 OUT`), for the opt-in multi-process compression (NEIF_PIPE=2|3
# in the environment; fast/docs/PROGRESS.md, "Multi-process compression").
#
# `make prof_use` (neif_build_test.sh; DEFS default "-DNEIF_TEST=1 -DNEIF_PACKED=1") into BUILD_DIR
# (default $NEIF_ROOT/build/tbKe), then BUILD_DIR/e/cmix = that binary with the embedded blocks of
# the official comp9 appended (hint package, article order, weights container, 12-byte size
# header: the last bytes of OFFICIAL_COMP9), which `-e` unpacks from its own file.
# OFFICIAL_COMP9: the comp9 of neif_build_official.sh ($NEIF_ROOT/official/comp9) or the
# published one (sha256 976bf2d3...).
# The binary is not packed; `-e` finds itself as ./cmix, so a run needs BUILD_DIR/e/cmix copied
# into its working directory and "." in PATH (neif_e_run.sh does both).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
R=${NEIF_ROOT:-$HERE/work}
OFF=$(readlink -f "${1:?usage: neif_e_build.sh OFFICIAL_COMP9 [BUILD_DIR] [DEFS]}")
B=${2:-$R/build/tbKe}
X=${3:--DNEIF_TEST=1 -DNEIF_PACKED=1}
[ -s "$OFF" ] || { echo "$OFF: no such file"; exit 1; }
bash "$HERE/neif_build_test.sh" "$B" "$X"
B=$(cd "$B" && pwd)
mkdir -p "$B/e"
python3 - "$OFF" "$B/cmix" "$B/e/cmix" <<'PY'
import struct, sys
off = open(sys.argv[1], 'rb').read()
d, o, w = struct.unpack('<3i', off[-12:])
assert 0 < d < 1 << 22 and 0 < o < 1 << 22 and 0 < w < 1 << 24, (d, o, w)
tail = off[len(off) - 12 - d - o - w:]
open(sys.argv[3], 'wb').write(open(sys.argv[2], 'rb').read() + tail)
print('embedded blocks of the official comp9: hint package %d B, article order %d B, weights %d B' % (d, o, w))
PY
chmod 755 "$B/e/cmix"
echo "$B/e/cmix: $(stat -c %s "$B/e/cmix") B, sha256 $(sha256sum "$B/e/cmix" | cut -c1-64)"
