# cmix-neif 2.0 with test-time speed modes

> This repository is a **modified version of cmix-neif 2.0** (Marcus Wölk, GPL-3,
> <https://github.com/marwoelk/cmix-neif-2.0>, commit `35d1957`). Upstream's own README is kept
> unmodified as [`README.cmix-neif.md`](README.cmix-neif.md); it describes the compressor.
> This file describes only what was added. What was changed, by whom and when:
> [`MODIFICATIONS.md`](MODIFICATIONS.md).

## 1. What this is

cmix-neif 2.0 is a Hutter Prize entry: cmix with a quantized transformer byte model, one core,
about 49 hours to compress enwik9 and the same to decompress. This repository adds **test-time
speed modes** to its source, for people who develop on it: ways to find out in hours instead of
days what a change does to the compressed size, and whether the result still decodes.

* It is **not a contest entry** and not a faster compressor for the contest. The contest run is
  single-core, and nothing here changes it.
* Every addition is behind compile-time macros (`NEIF_TEST`, `NEIF_PACKED`, `NEIF_MT`) that the
  official build does not define. **The official build of this tree is byte-identical to
  upstream's published compressor**: `comp9`, 4,042,564 B, sha256
  `976bf2d309707ab555e0c92720da5508ff0e5e6b846fc7082234262b06767f82`, the value in upstream's
  `environment.txt`. To check: `bash fast/neif_build_official.sh` (runs upstream's `build.sh` and
  compares; about 7 minutes; needs upstream's toolchain, section 5).

Every number below is **measured** unless it is marked *estimate*; the file it comes from is
named next to it (`fast/results/`, described in [`fast/results/README.md`](fast/results/README.md)).

## 2. Why a fast split evaluation matters

A full compression of enwik9 with cmix-neif 2.0 takes 49.18 h on one core and the decompression
49.3 h (upstream's measurement on its test machine, a Zen 3 cloud instance: result table of
`README.cmix-neif.md`). On our laptop (Intel Core Ultra 288V) the one-process decoder runs at
2,150-2,450 stream bytes per second in page text, so a full run would take 55-63 h there
(*estimate* from windows, `fast/docs/PROGRESS.md`). Trying one variant of the model, the weights
or a mixer therefore costs about four days of machine time before one knows the size and that
the archive decodes.

The **split evaluation** (`fast/neif_split_eval.sh`) replaces that experiment by one of a few
hours:

* It works on the *coder stream*: the 501,062,281 bytes that `comp9 -e enwik9` hands to its
  arithmetic coder after all preprocessing.
* The stream is cut into K splits. Each split is coded by its own process group, **in
  parallel**, starting from cold models at a page start at least WARM bytes in front of the
  split (the bytes in front of that get a fast structure-only pass); only the bytes of the split
  itself are counted.
* Inside each split the work runs as a **pipeline of three processes** (PPMd and the lower
  transformer layers / the upper layers / everything else and the coder).
* Then each split's coded output is **decoded and compared with the stream**, again as a
  three-process pipeline (the verification decode, section 3.3).

**What the K=16 total means.** It is the sum, over the 16 splits, of the ideal code length
(sum of -log2 p over every coded bit, in bytes) of each split's own bytes. For the unmodified
weights, K=16 with 4 MB warm-up gives **88,132,116.4 bytes over 501,062,281 stream bytes**
(`fast/results/k16_one_machine/total.json`; each split's kept bits are the same to the last digit
in every run we made of it, `fast/results/k16_per_split.tsv`).

It is a **baseline for comparing variants, not an archive size**:

* every split starts from cold models after a 4 MB warm-up, which a real run does not; how far
  that moves the total from the code length of the real run was not measured;
* it is an ideal code length, not the bytes of an archive, and it leaves out the decompressor
  and everything else that is in `archive9` besides the coder output;
* so only the **difference** of two totals at the same K and WARM is meaningful. For a variant
  the script prints the *estimate* S = 93,388,956 + (total - 88,132,116.4), where 93,388,956 B is
  the published size S of cmix-neif 2.0. A small difference between variants should be confirmed
  by a full run before it is believed.

## 3. The modifications

All switches, exit codes and commands: [`fast/README.md`](fast/README.md). Design and the log
of what was built and checked: `fast/docs/`.

### 3.1 Packed 4-bit dense kernels (`-DNEIF_PACKED=1`)

*What:* the dense transformer matmuls read weights packed as two 4-bit values per byte, in groups
of 8 rows (plain AVX2), so half as many weight bytes pass through the cache per token. The
integer sums and every float are the same, so the coder output is **identical**
(`src/tf1/opt/qmat_dense.cpp`). *Enable:* compile with `-DNEIF_PACKED=1`, with or without
`NEIF_TEST`.

| machine | measured | file |
|---|---|---|
| laptop, Core Ultra 288V (Lunar Lake) | transformer section of the time table 174-180 s -> 128-138 s (-21 % to -29 %, mean -25 %; an earlier session under background load: 190-210 s -> 147 s) on the 60.0-60.5 M window; model time of the same window 258.8 / 259.1 s -> 197.3 / 207.5 s (-20 % to -24 %), in the interleaved timing rounds -13 % to -23 % per pair (-18 % and -20 % on the means of the two sessions); reference chunk 681.0 s -> 525.8 s (-23 %) | `fast/results/laptop/packed_timing_k1.txt`, `packed_timing_k2.txt`, `fast_results.txt` |
| rented instance, EPYC 9654 (Zen 4) | 139.4 / 140.7 s -> 128.0 / 137.9 s on 60.0-60.3 M: **only 1 % to 9 %** | `fast/results/epyc_scale/summary.txt` (rows `s13`, `s15` against `s01`, `s14`) |

*Limits:* the gain depends on the cache situation of the CPU and is small on the EPYC. AVX-512
VNNI and AVX-VNNI forms of the kernel exist only in the micro-benchmark
(`fast/neif_vnni_bench.cpp`); they were **not faster** than the packed AVX2 form when other
memory is touched between rows, as it is in cmix
(`fast/results/laptop/vnni_bench_out.txt`, `fast/results/epyc_scale/vnni_bench/`), so the
program has no VNNI path.

### 3.2 Pipeline of three processes for compression (`NEIF_PIPE=3`)

*What:* at the first model byte the process forks. Stage 0 walks the stream with PPMd and the
lower transformer layers, stage 1 runs the upper layers (from layer `NEIF_PIPE_SPLIT`, default
8) and the output heads, and the calling process does everything else and codes. The stages pass
per-byte records through shared-memory queues. The output is **identical** to one process
(same sha256 in every run we made). *Enable:* `NEIF_PIPE=3` (or 2) in the environment of
`cmix -t` (test mode), or of the normal `cmix -e` of a `NEIF_TEST` build (opt-in, below).

| machine | window | one process, original build | packed + `NEIF_PIPE=3` | file |
|---|---|---|---|---|
| laptop | 60.0-60.5 M | 258.8, 259.1 s | 97.3, 87.6 s: **2.7x to 3.0x** (2.0x to 2.4x against the packed build in one process, 197.3 / 207.5 s) | `fast/results/laptop/fast_results.txt` |
| laptop | 250.0-250.3 M | 681.0 s | 226.9 s: 3.0x | same |
| EPYC 9654 | 60.0-60.3 M | 139.4, 140.7 s | 46.3 s: **3.0x** (2.8x against packed in one process, 128.0 s) | `fast/results/epyc_scale/summary.txt` |

**In the real compress path** (`NEIF_PIPE=3 cmix -e enwik9 OUT`, default off, prints
`NOT the official single-core run`): the output for the **first 17,000,000 stream bytes** is
identical with one and with three processes (3,062,257 B, the same sha256 `8288b14c...`;
coder time 2,846 s against 2,002 s), and for the first 3,000,000 with two processes; a front
stage killed on purpose leaves no output file and exit code 3
(`fast/results/laptop/e_results.txt`, `e_batch.log`). *Limits:* those 17 M bytes are mostly the
metadata region, where the transformer is largely off, so the 1.4x there says little about page
text. **The real path has never been run to the end of the stream**: the end-of-stream checks
and the assembly of `archive9` after a pipelined run are untested. It is a development
convenience, not a way to produce a contest archive.

### 3.3 Verification decode (`NEIF_PIPE=3 cmix -u STREAM CODED`)

*What:* the front stages walk the **expected** stream exactly as in compression; the last stage
does not code but **decodes** the coded input with the arithmetic decoder and the byte loop of
the decompressor, and compares every decoded byte with the expected one before it goes on. It
needs the expected stream, so it is **a correctness check of the decoder path, not a
decompressor**: it cannot produce an unknown file.

*Speed:* 2.2x to 2.7x the one-process decode of the same build: laptop 266-270 s -> 116-118 s
and 702-713 s -> 291-292 s (2.29x to 2.44x, `fast/docs/PROGRESS.md`, "Stage SV"); packed build
197.6 / 202.6 s -> 91.3 s and 528.7 s -> 224.6 s (`fast/results/laptop/fast_results.txt`); EPYC
128.1 s -> 48.1 s (2.66x, `fast/results/epyc_scale/summary.txt`). Against the original build
in one process that is 2.8x to 3.0x.

*What a PASS certifies* (argument in `fast/docs/PROGRESS.md`, "Why a pass implies the plain
decoder's result"): decoding this coded input with this build's model on this machine gives
exactly the expected stream. Each record a front stage sends is a function of a prefix of the
expected stream and is used only after every earlier decoded byte was found equal, so by
induction the decoding stage is in the state of the plain one-process decoder; a wrong byte is
compared with the stage's own read of the stream and cannot be masked. The premise is that the
pipeline's records equal what one process computes; if that failed somewhere, the result would
be a false FAIL, not a false PASS. Negative tests (a changed bit in the coded input, a changed
bit in the expected stream, a truncated input, a killed stage) were all reported at the right
byte or as "no verdict".

*What a PASS does not certify:* that the expected stream is the right one; the step from the
stream back to enwik9; the official flavour of the build and the official packed decompressor as
a program (the verification decode of the **official** compressor's output has passed for the
first 16,500,000 stream bytes only, `fast/results/laptop/e_batch.log`); other machines. The
table in `fast/docs/PROGRESS.md` lists each gap and the step that would close it.

### 3.4 Threaded speculative decoder (`-DNEIF_MT=1`): an upper-bound experiment only

*What:* a real multi-core decoder cannot run the transformer ahead, because the next byte is
not known. `NEIF_MT` adds the machinery for speculation: the transformer step of the next byte
starts on a second thread for a *guessed* byte while the main thread decodes the bits; a right
guess is committed, a wrong one is undone (shadow engine, PPMd undo journal). **The guessing
itself is not built.** In the test (`NEIF_MT=n NEIF_MT_SPEC=2 cmix -u`) the guess is read from
the known stream, optionally with a forced rate of wrong guesses, so the figures are an upper
bound for a decoder that does not exist yet. Output and decoded bytes are identical in every
run, also with forced wrong guesses.

| machine | 60 M window, against packed in one process | file |
|---|---|---|
| laptop | 2 threads (main + transformer thread): 159.1, 157.6 s against 197.6 / 202.6 s = **1.26x** (1.63x against the original build); with 23 % forced wrong guesses 174.0, 174.9 s = **1.15x**; a third thread: 154.0, 180.2 s, no gain | `fast/results/laptop/fast_results.txt` |
| EPYC 9654 (one run each, shared host) | 2 threads 100.5 s = 1.27x; 3 threads 82.5 s = 1.55x; 4 threads 83.8 s; 5 threads 102.8 s. With 23 % forced wrong guesses: 132.8 s (0.96x), 119.6 s (1.07x), 118.9 s | `fast/results/epyc_scale/summary.txt` (rows `s06`-`s12` against `s02`) |

*Limits:* 23 % is the measured miss rate of the transformer's top guess on that window; a real
decoder would land near the forced-miss figures, about 1.1x to 1.2x for two threads on the
laptop (*estimate*, `fast/docs/DESIGN.md` 0b). More threads do not help on the laptop.

### 3.5 Smaller pieces

* **Chunk scoring mode `cmix -t STREAM OUT`** (`NEIF_T_START`, `NEIF_T_END`, `NEIF_T_WARM`):
  codes a window of the coder stream and reports the ideal code length of the window; the basis
  of everything above.
* **Chunk decoder `cmix -u STREAM CODED`**: decodes the output of `-t` in one process with the
  decompressor's byte loop and compares every byte with the stream. Same speed as `-t`.
* **16-bit activation paths** (`NEIF_ACT16=162 NEIF_ACT16_DNBITS=10 NEIF_ACT16_DNMIN=2`,
  experimental): some transformer matmuls read 16-bit instead of 8-bit activations. This
  **changes the coded bits** (it is a model variant; compressor and decoder need the same
  setting). On three 1 MB windows the kept bits fell by 0.058 %, 0.082 % and 0.078 % (0.074 %
  over the three) (`fast/results/laptop/act16_results.txt`: lines `g10_2_*_162` against
  `a16_*_0`), at a time cost of 3 % to 8 % on the laptop (`f_k16on*` against `f_k16off*` in
  `fast_results.txt`). There is **no full-stream number**, and it cannot be combined with the
  speculative step.

## 4. Measured full split evaluations

Machines: rented 32-vCPU cloud CPU instances (RunPod), each a container with 16 cores / 32
threads of an AMD EPYC 9654 whose host is **shared with other tenants**. K=16, 4 MB warm-up,
packed build, three processes per split, coder stream of the unmodified entry. Per-split table:
[`fast/results/k16_per_split.tsv`](fast/results/k16_per_split.tsv).

| run | layout | compress | verification decode | file |
|---|---|---|---|---|
| **one instance**, all 16 splits at once | 48 processes on 32 logical CPUs, not pinned | **13,161 s (3 h 39 min)**, 99.3 core-hours | **15,226 s (4 h 14 min)**, 113.1 core-hours, **16 of 16 PASS** | `fast/results/k16_one_machine/` |
| two instances, splits 8-15 on the quieter host | 8 splits at once, each pinned to 4 logical CPUs = 2 cores | **7,885 s** | **7,825 s, 8 of 8 PASS** | `fast/results/k16_two_machines/quiet_host_splits_8-15/` |
| the same host afterwards, splits 0-7 | same | **6,992 s** | not run | `.../quiet_host_splits_0-7_compress_only/` |
| two instances, splits 0-7 on the busier host (at the same time as the second row) | same | **12,245 s** | **13,393 s, 8 of 8 PASS** | `.../busy_host_splits_0-7/` |

The two-instance run as it was actually run (rows 2 and 4, started together) is **16 of 16
PASS**; its wall time was set by the busier host: 12,245 s compress + 13,393 s verification
decode, against 7,885 s + 7,825 s on the quieter one.

* Peak memory of the one-instance run: 164.8 GiB for the container (cgroup figure, page cache
  included; `k16_one_machine/monitor.tsv`); 9.7 to 10.6 GiB per split.
* **The result does not depend on the machine or the layout**: every split has the same kept
  bits in all runs, and the coded output of each of the 16 splits has the same sha256 in the
  one-instance run and in the runs on both hosts of the two-instance layout (`k16_per_split.tsv`).
* **Shared-host contention changes the wall time by up to about 1.75x.** The same eight splits
  0-7, same binary, same pinning, identical kept bits, took 6,992 s on one host and 12,245 s on
  the other, whose load average was about 100 at the start against 20-50
  (`driver.log` of each run). On the busier host individual splits took between 7,403 s and
  12,244 s to compress and between 7,942 s and 13,393 s to verify. The first run of the scaling test on that host was 4.8x slower than its two repeats
  (678.5 s against 139.4 / 140.7 s). Timings on such instances are not reproducible to better
  than that; the bits are.
* So the two-day experiment "compress, then decompress and compare" (2 x 49 h on one core)
  becomes 7.9 h on one instance (3.66 h + 4.23 h), or about 4.4 h on two instances when both
  hosts are quiet (2.19 h + 2.17 h on the quieter one).
* Cost (*estimate*, not in the result files): at the 1.3 to 1.8 US dollars per instance-hour we
  paid, about 10 to 14 dollars for the one-instance run (7.9 h), 11 to 16 dollars for two instances
  on quiet hosts (2 x 4.4 h), and 18 to 26 dollars for the two-instance run as it went (2 x 7.1 h).
* What these runs did **not** do: the plain one-process decoder was never run over a whole
  split; the decode of every split is the verification decode of section 3.3.

Other loads on one instance (throughput with all slots busy; `fast/results/epyc_scale/loaded.tsv`,
`long_window/loaded.tsv`): on the 60.0-60.5 M window 16 pipelines of 3 processes on 32 logical
CPUs coded 64,027 stream bytes per second in total when unpinned and 56,153 when each was
pinned to 2 logical CPUs; 8 pipelines pinned to 4 logical CPUs each coded 42,755.

## 5. How to build and run

Requirements: x86-64 with AVX2; **Ubuntu 24.04** (upstream's `install.sh` pins clang 18.1.3 and
lld 18 from a dated Ubuntu snapshot and repoints apt, so use a container, a chroot or a
throw-away machine); about 10 GiB of RAM and a sparse 14.7 GB file on a local disk **per
running split**; enwik9 once, to make the coder stream.

```
bash fast/neif_mkstream.sh /path/to/enwik9     # -> fast/work/stream.bin (501,062,281 B; sha256 checked)
bash fast/neif_setup.sh                        # compiler, three test builds, self-check
```

`neif_setup.sh` must end with `SELF-CHECK PASSED`. The values it checks (sha256 of the coder
output, first 16 hex digits; the same for every build and every `NEIF_PIPE` / `NEIF_MT`
setting):

| chunk | sha |
|---|---|
| quick: stream bytes 60,000,000-60,030,000, warm-up 10,000 | `9e925d7846f34983` |
| 60,000,000-60,500,000, warm-up 100,000 | `e3c7340dce4cef65` |
| reference chunk 250,000,000-250,300,000, warm-up 1,200,000 | `727e97767d69fb56` |

If they differ on your machine or compiler, the binary is a different program: do not compare
its numbers with ours. `CHECK=quick BUILDS=tbK bash fast/neif_setup.sh` builds only the packed
build and checks only the first value (a few minutes).

Split evaluation, here on one 32-thread machine with 170 GiB of memory:

```
cd fast/work
SLOTS=16 STEPS="compress decode summary" DECODE=verify \
  bash ../neif_split_eval.sh build/tbK stream.bin 16 4000000 out/k16 3
```

(`SLOTS` = splits at a time; with less memory lower it, the run is resumable. Several machines:
`SPLITS=0-7` / `SPLITS=8-15`, then `--summary` over the merged directories. To compare a variant:
`BASE_TOTAL=88132116.4`.) Scaling test: `bash fast/neif_scale.sh`. Official build check:
`bash fast/neif_build_official.sh`. Everything else: [`fast/README.md`](fast/README.md).

## 6. Limits, and what was not verified

* **Other CPUs:** everything was measured on a Core Ultra 288V (Lunar Lake) and on EPYC 9654
  (Zen 4). **Zen 3** (the class of upstream's test machine) and **Tiger Lake** were not tested:
  neither the speed of the packed kernels nor the self-check values there.
* **Compiler:** whether another compiler version gives the same model output is not known; a
  build without the profile already does not. The self-check decides.
* **The opt-in multi-process compression** (`NEIF_PIPE` with `-e`) is verified for the first
  17,000,000 stream bytes only and has never run to the end of the stream.
* **There is no real multi-core decompressor.** The verification decode needs the expected
  stream; the speculative decoder takes its guesses from the stream.
* **The verification decode runs the test flavour**, not the official packed decompressor. All
  16 splits of the test flavour's own output pass; of the official compressor's output only the
  first 16.5 M stream bytes were decoded this way. The plain one-process decoder was run on
  windows (up to 10 M stream bytes), never on a whole split.
* **The K=16 total is not an archive size** (section 2); its offset from a real run is not
  measured.
* **The binaries behind the numbers are one source change older than this tree.** The full
  split evaluations, the scaling test and the laptop mode comparison ran before the opt-in
  pipeline in `-e` was added (`src/runner.cpp`, `src/neif_test.h`). The test builds of the
  published tree are therefore different files; they reproduce the self-check values of
  section 5, but the K=16 runs were not repeated with them.
* **Timing on shared hosts is noisy** (up to 1.75x between two hosts, 4.8x for one short run);
  the laptop drifts by up to 15 % over a day. Ratios come from runs next to each other where
  that was possible; the EPYC single-run figures of sections 3.1 to 3.4 are one run each.
* **The 16-bit activation option** has no full-stream number and no decode check beyond its
  windows.

## 7. Licence and credits

GNU General Public License, version 3 (`LICENSE`), as upstream. cmix-neif 2.0 is the work of
Marcus Wölk, built on cmix (Byron Knoll), fx2-cmix (Kaido Orav, Byron Knoll), cmix-lex (Ibrahim
Marcouch) and the further components and authors listed in [`NOTICE.md`](NOTICE.md), which is
unchanged. The modifications in this repository are described, with dates, in
[`MODIFICATIONS.md`](MODIFICATIONS.md); they are released under the same licence, without
warranty.
