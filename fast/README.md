# Test-time speed modes of cmix-neif 2.0: builds, switches, how to run every mode

The source tree in this repository is cmix-neif 2.0 with test tooling added behind compile-time
macros. A build without those macros is still the published compressor `comp9`, byte for byte
(sha256 `976bf2d309707ab555e0c92720da5508ff0e5e6b846fc7082234262b06767f82`,
`neif_build_official.sh`). This directory holds the scripts that build and run the test modes,
the design notes (`docs/`) and the raw results (`results/`). Summary and numbers: the top-level
`README.md`.

All scripts take their paths from arguments and the environment:

| variable | meaning | default |
|---|---|---|
| `NEIF_SRC` | the source tree | the directory above `fast/` |
| `NEIF_ROOT` | working directory: `stream.bin`, `build/`, `out/` | `fast/work` (ignored by git) |

`NEIF_ROOT` must be on a LOCAL disk: every running `cmix` writes `./.dict` and a sparse 14.7 GB
paging file `./ppm.page` into its run directory. Memory: 7 to 10 GiB per run plus about 0.5 GiB
with a pipeline (10 GiB is the program's own ceiling for one process).

## 1. What is needed

| what | size | sha256 |
|---|---|---|
| this source tree (with the weights container and the PGO profile) | 16 MB | |
| weights container `weights-poc8m_w2048_tf2t3_pruned_v5k_np2.c5` (in the tree; the binary reads it from its WORKING directory; the scripts link it into every run directory) | 3,744,629 B | `729694cba4969b7d833686ed3f597ff04d3b226da849d5f8004ec020a0ed9563` |
| coder stream `stream.bin` = the bytes `comp9 -e enwik9` hands to its arithmetic coder, after all preprocessing | 501,062,281 B | `c35d3073eeba70f22a2e08a50202b53eb95466e3e9f1f2702f6f9b3f96fd8718` |

The stream is not in the repository (it is derived from enwik9). Make it once:
`bash fast/neif_mkstream.sh /path/to/enwik9` builds upstream's own tool `cmix_stream`
(`training/tf/1_dump/trainmodes.patch`, from the unmodified upstream commit of this repository)
and runs the preprocessing; the script checks the sha256. Nothing else is needed: the dictionary
and the article order are inside the stream.

## 2. Compiler and builds

The entry is compiled with **clang 18.1.3 and lld 18 (Ubuntu 24.04 packages `1:18.1.3-1ubuntu1`),
not gcc**; gcc 13 only supplies the libstdc++ headers. Upstream's `install.sh` pins every package
to the Ubuntu archive of 2026-09-27 through snapshot.ubuntu.com, which needs Ubuntu 24.04
(apt 2.7+). Run it in a container, a chroot or a throw-away machine: it repoints apt.
The build is profile-guided (the profile is shipped in `pgo_data/`) with LTO.
**Whether the model output depends on the compiler version is not known.** A plain `make cmix`
(no profile) of the same source already gives another coder output, so treat any other compiler
as a different program until the self-check passes (reference values below).

A test build = the entry's own recipe `make prof_use` in a directory that holds copies of `src/`,
`makefile`, `encoder_split.lds`, `runtime_exports.map`, `pgo_data/` and the weights container,
with the macros added through `CC`. `neif_build_test.sh BUILD_DIR [DEFS]` does exactly that:

```
bash fast/neif_build_test.sh fast/work/build/tbN "-DNEIF_TEST=1"                              # original test build
bash fast/neif_build_test.sh fast/work/build/tbK "-DNEIF_TEST=1 -DNEIF_PACKED=1"              # packed
bash fast/neif_build_test.sh fast/work/build/tbKM "-DNEIF_TEST=1 -DNEIF_MT=1 -DNEIF_PACKED=1" # packed + threads
```

(one directory per build; result `BUILD_DIR/cmix`, about 15 s on 8 cores.) `neif_setup.sh` does
all of it: compiler, stream check, the three builds, self-check.

| macro | what it adds | default build |
|---|---|---|
| `NEIF_TEST` | chunk modes `-t` / `-u`, the pipeline (`NEIF_PIPE`), the verification decode, the opt-in pipeline in `-e`, bit counting, the 16-bit activation experiment (`NEIF_ACT16`), the measuring hooks `NEIF_PAR` | not defined |
| `NEIF_PACKED` | dense transformer matmuls on nibble-packed weights (`src/tf1/opt/qmat_dense.cpp`): half the weight bytes per token, plain AVX2, same integer sums and floats, so the same coder output. Usable with or without `NEIF_TEST` | not defined |
| `NEIF_MT` (inside `NEIF_TEST`) | threads: the speculative transformer step and threads inside the step (`src/neif_mt.h`) | not defined |
| `NEIF_PFD=n` | prefetch distance of the dense kernels (entry: 6144); measured: no gain | not defined |
| `FX2_PROF` | per-site time table of the transformer at the end of a one-process `-t` run | not defined |

Reference values (identical for tbN, tbK, tbKM, with any `NEIF_PIPE` / `NEIF_MT` setting; the
16-bit switch changes them):

| chunk | `NEIF_T_START` | `NEIF_T_END` | `NEIF_T_WARM` | sha256 of the coder output (first 16 hex) |
|---|---|---|---|---|
| quick | 60000000 | 60030000 | 10000 | `9e925d7846f34983` |
| 60 M window | 60000000 | 60500000 | 100000 | `e3c7340dce4cef65` |
| reference chunk | 250000000 | 250300000 | 1200000 | `727e97767d69fb56` (total_bits 2508178.191) |

Builds are reproducible: the same source and the same pinned compiler gave the same binaries on
the laptop and on the rented machines (`results/epyc_scale/setup_selfcheck.log`). **Those
binaries, which produced the results in `results/` except `laptop/e_*` (and the last lines of
`laptop/mt_results.txt`), were built from the
tree one change before the published one**: the opt-in pipeline in `-e` (`NeifEStart` /
`NeifEEnd` in `src/runner.cpp`, the failure message in `src/neif_test.h`) was added afterwards.
The published tree therefore gives other binaries (packed test build: sha256 `c768181b4017a177...`
against `02579d85ebda3008...` in the result files) that code the same self-check values. A
different binary is not an error; the self-check decides.

**The official build:** `bash fast/neif_build_official.sh` runs upstream's `build.sh` on this tree
and compares `comp9` with the published sha256 (about 7 min).

**AVX512-VNNI:** there is no AVX-512 or VNNI kernel in the program. The 512-bit forms exist only
in the micro-benchmark `neif_vnni_bench.cpp` (kernels C5 and D5, compiled with `-DBENCH_AVX512`):
`clang++-18 -O3 -march=x86-64-v3 -mtune=znver2 -mavxvnni -mavx512f -mavx512bw -mavx512dq -mavx512vnni -DBENCH_AVX512 neif_vnni_bench.cpp -o vb && taskset -c 2 ./vb 0 2 4 8`
(without AVX-VNNI on the CPU drop `-mavxvnni`; kernels C and D then are left out). It prints
microseconds per transformer row for every kernel; the arguments are MB of other memory touched
between rows. On the laptop and on the EPYC 9654 the packed AVX2 kernel (E) was at least as fast
as the VNNI forms once other memory is touched between rows (`results/laptop/vnni_bench_out.txt`,
`results/epyc_scale/vnni_bench/`).

## 3. Running one chunk (`neif_run.sh`, or by hand)

Every mode is a chunk run on the coder stream; positions are stream bytes. By hand, in an empty
working directory that has the weights container (a symlink will do):

```
export NEIF_T_START=60000000 NEIF_T_END=60500000 NEIF_T_WARM=100000 NEIF_T_JSON=score.json
B/cmix -t stream.bin out.cmix        # code: structure-only pass up to the warm-up start, then the model
B/cmix -u stream.bin out.cmix        # decode out.cmix and compare every byte with the stream (exit 0 = all equal)
```

The model starts at the first page start at or behind START - WARM (at byte 0 if WARM >= START;
there is no page start before about 15.28 M, so a window there must start at 0). Only the bits of
[START, END) are counted as `kept_bits`.

| mode | command line | environment |
|---|---|---|
| compress, one process | `cmix -t` | - |
| compress, pipeline of processes (same output) | `cmix -t` | `NEIF_PIPE=2` or `3`; `NEIF_PIPE_SPLIT=l` = first transformer layer of the second stage (default 8, measured best with the packed build) |
| **real compress command, pipeline of processes** (opt-in convenience, NOT the official single-core run; checked on a prefix only, see below) | `cmix -e enwik9 OUT` | `NEIF_PIPE=3` (or 2); `NEIF_E_STOP=n` stops behind n stream bytes (test) |
| decode, one thread (the real decoder loop) | `cmix -u` | - |
| **verification decode**: the front stages walk the EXPECTED stream, the coder stage decodes and compares | `cmix -u` | `NEIF_PIPE=3` (or 2); `NEIF_SV_LOG=s` progress line every s seconds (default 600) |
| threaded decode test (NEIF_MT build): speculative transformer step, guess taken from the stream | `cmix -u` | `NEIF_MT=n NEIF_MT_SPEC=2` (n = transformer thread + n-1 workers, beside the main thread); `NEIF_MT_FAULT=p` forces p per mille wrong guesses; `NEIF_MT_CPUS=a,b,c` binds main, transformer thread, workers; `NEIF_MT_PIN=0` no binding |
| compress with the speculative step (NEIF_MT build) | `cmix -t` | `NEIF_MT=n NEIF_MT_SPEC=1` |
| threads inside the step only | `-t` / `-u` | `NEIF_MT=n` (n >= 2), no `NEIF_MT_SPEC` |
| 16-bit activations (the setting we measured) | any | `NEIF_ACT16=162 NEIF_ACT16_DNBITS=10 NEIF_ACT16_DNMIN=2` (changes the coded bits: compressor and decoder need the same setting; not combined with the speculative step: the step then runs unspeculated) |
| time table | any | `NEIF_PROF=1` |

The threaded decode test is NOT a decoder that works without the stream (a real one must guess
the byte; that stage, "S5" in `docs/`, is not built). The verification decode needs the expected
stream by design. `NEIF_PIPE` and `NEIF_MT` exclude each other.

Exit codes of the verification decode: 0 `neif_sv: PASS`, 4 `FAILED at stream byte N`, 3 / 5
`NO VERDICT` (the run broke; says nothing about the data), 1 an internal check tripped by a wrong
byte. Only exit 0 is a pass. Details and the argument why a pass implies the plain decoder's
result: `docs/PROGRESS.md`, "Stage SV".

`neif_run.sh NAME t|u START END WARM BUILD_DIR [CODED]` wraps one run: own run directory under
`$OUT`, optional binding (`CPUS=0,1,2`), one line per run in `$OUT/results.tsv` (wall, model
seconds, cpu seconds, peak memory, pipe, threads, binary sha256, sha of the coder output,
OK/PASS). Example, the quick self-check by hand:

```
export NEIF_ROOT=$PWD/fast/work
bash fast/neif_run.sh q t 60000000 60030000 10000 fast/work/build/tbK          # out_sha16 9e925d7846f34983
NEIF_PIPE=3 bash fast/neif_run.sh q3 t 60000000 60030000 10000 fast/work/build/tbK   # the same sha
NEIF_PIPE=3 bash fast/neif_run.sh qv u 60000000 60030000 10000 fast/work/build/tbK q # PASS
```

## 4. Split evaluation (`neif_split_eval.sh`)

K splits of the stream, several at a time; per split a `-t` run with warm-up (coded output and
JSON kept) and, as a second step, a decode of that output checked against the stream. Read the
head of the script for all switches. Typical use on a 32-thread machine (16 splits, 3-process
pipeline, packed build, verification decode of every split, 8 splits at a time, each bound to
its own CPUs):

```
cd fast/work
PIN=1 PIN_W=4 SLOTS=8 SPLITS=0-7 STEPS="compress decode" DECODE=verify \
  bash ../neif_split_eval.sh build/tbK stream.bin 16 4000000 out/k16 3
# second machine: SPLITS=8-15. Afterwards copy both machines' out/k16/c* into one directory and
bash ../neif_split_eval.sh --summary DIR 16
```

* `SPLITS` / `DEC_SPLITS` select splits (`0-7`, `0,8,15`); `DECODE=verify|plain|spec`
  (verification decode / real one-thread decoder loop / threaded test with
  `DEC_BUILD=build/tbKM`, `DEC_MT=2`); per-split mode one process = last argument 0.
* Resumable; every finished run appends a line to `OUT/results.tsv` and, with `EMIT_DIR`, copies
  its small result files there at once. `OUT/session_*.txt` records lscpu, cgroup cpu.max /
  memory.max, cpu.stat and memory.events before and after, the file system and **the whole
  environment of the shell**: do not publish that file unread.
* The summary gives total kept bytes; with `BASE_TOTAL=<total of the reference run>` also the
  estimated S = 93,388,956 + (total - BASE_TOTAL). The K=16 / 4 MB warm-up total of the
  unmodified weights is **88,132,116.4** (`results/k16_one_machine/total.json`); the packed build
  codes the same bits as the original, so this is the baseline for any variant.
* WARM must reach a page start in front of every split: the summary says "raise WARM" otherwise.
* Smoke test (about 6 min on the laptop): `STEPS="compress decode summary" CHUNK_BYTES=30000 SLOTS=1 bash
  ../neif_split_eval.sh build/tbK stream.bin 2 200000 out/smoke 3`.
* `neif_k16_one_machine.sh` and `neif_k16_part.sh` are the drivers of our full runs (passes
  chained, log, 10-second monitor `neif_monitor.py`).

## 5. Scaling test (`neif_scale.sh`)

`bash fast/neif_scale.sh [OUT_DIR]`: one run at a time on 60.0-60.3 M (original single, packed
single `-t` and `-u`, pipeline 2 and 3, verification decode, threaded decode test with `NEIF_MT`
1/2/3/4 and with 23 % forced wrong guesses at 1/2/3, the first run again), each bound to its own
physical cores; then the loaded-throughput test on 60.0-60.1 M (`LOADED="16:0 16:2 8:3 16:3 5:3"`
= runs at once : processes per run), all slots busy at once. Results: `results.tsv`,
`loaded.tsv`, `summary.txt`, `session_*.txt` in OUT_DIR. About 35 min + 15 min at the laptop's
speed. Resumable. Needs the three builds of `neif_setup.sh`.

## 6. The opt-in multi-process compression in the real path

Default off. With `NEIF_PIPE=2|3` the normal `cmix -e enwik9 OUT` of a NEIF_TEST build runs its
coder as the forked pipeline from stream byte 0; it prints `neif_pipe: MULTI-PROCESS COMPRESSION
... NOT the official single-core run` at the start and at the end, and a `neif_e:` line with time
and peak memory. If the pipeline breaks, OUT is removed, no archive9 is written, exit code 3.
A test build must carry the embedded blocks of comp9 to run `-e`:

```
bash fast/neif_build_official.sh                               # -> fast/work/official/comp9
bash fast/neif_e_build.sh fast/work/official/comp9             # -> fast/work/build/tbKe/e/cmix
bash fast/neif_e_run.sh e_p0 17000000 0 /path/to/enwik9        # one process, stopped behind 17 M stream bytes
bash fast/neif_e_run.sh e_p3 17000000 3 /path/to/enwik9        # 3 processes: same size and sha256
```

Verified: `results/laptop/e_results.txt`, `e_batch.log` (identical output for stream bytes
[0, 17,000,000) with 1 and 3 processes, for [0, 3,000,000) with 2; a killed front stage leaves no
output). **Never run to the end of the stream**: the end-of-stream checks and the assembly of
archive9 after a pipelined run are untested.

## 7. Measured on the laptop (Core Ultra 288V, WSL2; model seconds, 17 Nov 2026)

| window | original, one process | packed, one process | packed + pipeline 3 (compress) | packed + verification decode | packed + threaded decode test |
|---|---|---|---|---|---|
| 60.0-60.5 M (598,569 model bytes) | 258.8, 259.1 | -t 197.3, 207.5; -u 197.6, 202.6 | 97.3, 87.6 | 91.3 | NEIF_MT=1 (2 threads) 159.1, 157.6; NEIF_MT=2 (3 threads) 154.0, 180.2; with 23 % wrong guesses 174.0, 174.9 (2) and 181.4, 181.6 (3) |
| 250.0-250.3 M (1,495,402 model bytes) | 681.0 | -t 525.8; -u 528.7 | 226.9 | 224.6 | 3 threads 419.3 (419.7 with 10 % wrong guesses) |

All runs of a window gave the reference sha / decoded every byte. Raw lines:
`results/laptop/fast_results.txt`. The rented EPYC machines: `results/epyc_scale/`.

## 8. Files in this directory

| file | purpose |
|---|---|
| `neif_setup.sh` | compiler, stream check, three test builds, self-check |
| `neif_build_test.sh` | one PGO test build with given macros |
| `neif_build_official.sh` | upstream's official build of this tree, compared with the published comp9 |
| `neif_mkstream.sh` | the coder stream from enwik9 |
| `neif_run.sh` | one chunk run (`-t` or `-u`), one result line |
| `neif_split_eval.sh`, `neif_cpusets.py` | split evaluation; cpu sets for pinning |
| `neif_k16_one_machine.sh`, `neif_k16_part.sh`, `neif_monitor.py` | drivers and monitor of the full K=16 runs |
| `neif_scale.sh` | scaling and loaded-throughput test |
| `neif_e_build.sh`, `neif_e_run.sh` | the real compress command with a test build (opt-in pipeline) |
| `neif_vnni_bench.cpp` | micro-benchmark of dense-kernel variants |
| `docs/DESIGN.md`, `docs/PROGRESS.md` | design, stage log, soundness argument of the verification decode (working notes) |
| `results/` | raw result tables (`results/README.md`) |
