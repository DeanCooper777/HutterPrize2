> **Working notes, published as written (17 Nov 2026), with machine-specific paths removed.**
> This is the running log of how the test modes were built and what was checked at each stage.
> It names scripts of our internal tooling (`neif_mt_one.sh`, `neif_sv_batch.sh`, `neif_e_batch.sh`,
> `neif_par_*.sh` and others) that are not in this repository; `fast/` holds the portable subset
> (see `fast/README.md`). Paths starting with `/work/` are directories of the build machine.
> "This PC" / "the laptop" is an Intel Core Ultra 288V (Lunar Lake) under WSL2.
> Later results are NOT in this log: the 17 M identity run of the opt-in multi-process compression
> and the verification decode of an official prefix into page text (`fast/results/laptop/e_batch.log`),
> and the full split evaluations on rented machines (`fast/results/`). Where this log and the
> top-level `README.md` disagree, the README is the later statement.

# Multi-threaded decoder for cmix-neif 2.0: running log

Design and the S0 measurements: `DESIGN.md` (section 0a). This file is the log of the build stages,
so that someone else can pick the work up. "Where things stand" first; details per stage below.

## Where things stand (17 Nov)

* **S3 done:** the chunk decoder `cmix -u` decodes what `cmix -t` writes and reproduces the stream.
* **S1 done:** thread pool, fused MLP region, split window attention; identical output at 2 and 3
  threads. Measured gain is far below the simulation (see S1).
* **S4 done:** the speculative transformer step with a shadow engine, a PPMd undo journal, a match
  peek and a fault injector; identical output with and without forced wrong guesses, in the coder
  and in the decoder. Measured numbers in S4.
* **SV done:** the reference-assisted validation decode (`NEIF_PIPE=2|3 cmix -u`): 2.3-2.4x the
  plain decode with three processes, passes on the official archive's first 10 M stream bytes,
  catches a changed coded byte and a changed expected byte at the right position. See "Stage SV".
* **Not started: S5** (guessing the byte in the decoder, re-guessing on a decoded bit; on hold).
  The undo paths S5 needs are exercised by the fault injector; an abort and a new launch inside
  one byte are new. Measured bounds for it are in "Measurements".
* **Multi-process compression in the real path (`NEIF_PIPE=2|3 cmix -e`): built on 17 Nov (evening),
  only smoke-tested.** Identical output on the first 200,000 stream bytes, kill test passed; the
  17 M identity run, the `NEIF_PIPE=2` run and the official-prefix decode in page text were started
  and **stopped by the host (low memory) before they produced anything**. See "Multi-process
  compression" below; the official build was NOT rebuilt after this change.
* The official build was re-verified at the end of the SV work (see the last section).

## How to work here

* Source: the source tree (the development tree; no commits). Everything new is behind
  `NEIF_MT` inside `NEIF_TEST` (the chunk decoder `-u` is plain `NEIF_TEST`); the measurement of S0
  is behind `NEIF_PAR`. The official build must stay sha256 976bf2d3...6767f82
  (`neif_build_official_dev.sh`, 6 min).
* Build: `bash neif_mt_build.sh /work/tbmt` in WSL (PGO test build with
  `-DNEIF_TEST=1 -DNEIF_MT=1`, 2 min). It prints the functions whose PGO profile entry was
  discarded: the 18 of the reference test builds plus `ppmd_page::kopiere` (an integer block copy
  that now has the journal branch inlined). `Predict`, `Perceive`, `Encode`, `Decode` keep theirs.
* One run: `bash neif_mt_one.sh NAME t|u START END WARM BUILD MT [CPUS] [CHUNK]`
  (`t` codes a chunk and keeps its output, `u` decodes the output of the `t` run CHUNK and compares
  every byte with the stream). One line per run goes to `mt_results.txt`
  (copy: `fast/results/laptop/mt_results.txt`).
* Quick identity checks on a 36 k byte chunk (7 min each): `neif_mt_quick.sh` (S1),
  `neif_mt_s4quick.sh` (S4). Timing and reference-chunk batches: `neif_mt_batch.sh`.
* Validation decode: `neif_sv_batch.sh quick|neg|speed60|speed250|official|mixed|head`
  (`NEIF_PIPE=2|3` in front of a `u` run of `neif_mt_one.sh`; its processes are not bound to a
  CPU). `neif_mt_look.sh NAME...` shows the end of a run's log, `neif_mt_stop.sh` stops a batch.
  Never edit a run script while a batch that uses it is running (bash reads it as it goes).
* Only one large cmix process at a time on this PC; the scripts refuse to start a second one. Do
  not build while a timing runs.
* Shell pitfalls that cost time: `$` and `\n` inside `wsl.exe -- bash -c "..."` and inside Python
  heredocs passed through the Bash tool get mangled; write script files, and use the Edit tool for
  C strings with `\n`.

## Run-time switches (all default off; without `NEIF_MT` in the environment no thread exists)

| variable | meaning |
|---|---|
| `NEIF_MT=n` | n threads for the transformer step. Without `NEIF_MT_SPEC`: the main thread and n-1 spinning workers (n >= 2). With it: the transformer thread and n-1 workers (n >= 1). |
| `NEIF_MT_SPEC=1` | speculative step on the transformer thread where the byte is known before its bits: every byte in `-t`, forced bytes in `-u` |
| `NEIF_MT_SPEC=2` | `-u` only, a test of the machinery, not a decoder: the guess is the stream byte |
| `NEIF_MT_FAULT=p` | p per mille of the guesses are made wrong on purpose (another byte of the transformer's set) |
| `NEIF_MT_CPUS=a,b,c` | logical CPUs: main thread, then the transformer thread (if any), then the workers (default 0,1,2,...) |
| `NEIF_MT_PIN=0` | do not pin |
| `NEIF_MT_TEST=1` | the workers run but the caller does every share (what a spinning thread costs) |

## Stage S3: chunk decoder `-u` (done)

`cmix -u STREAM CHUNK` (`src/runner.cpp`, `NeifDecompressChunk`, `NeifUntestMain`): CHUNK is the
output of `cmix -t STREAM CHUNK` with the same `NEIF_T_START/END/WARM`. `-t` already emits a
decodable chunk (its 5-byte header, then the coder output from the first model byte, flushed at the
end), so nothing had to be added on the coding side. The decoder makes its `Decoder` where `-t`
makes its `Encoder`, runs the same structure-only pass over the stream bytes in front of the
warm-up start, then the byte loop of `Decompress()`, and compares every decoded byte with the
stream; it stops at the first wrong byte (exit code 4) and writes a JSON result (`NEIF_T_JSON`).

## Stage S1: threads inside the transformer step (done)

Files: `src/neif_mt.h` (the pool: per-worker sequence words, spin with `pause`, pinned);
`src/tf1/opt/qmat_dense.cpp|h` (`qgemv_relu2q_part`, `relu2q_idx_part`),
`qmat_sparse.cpp|h` (`qsparse4_accum_part`, `qsparse4_epi_add`), `attn.cpp|h` (`attn_step_mt`),
`model_opt.cpp` (`mlp_block_mt`, the hook in `van_row`).

* MLP region (one fork): share s computes a range of the up projection's row groups, their relu^2
  codes, its own list of non-zero codes and the int32 sums of the down columns of those codes; the
  caller adds the sums, does the one float step and joins the lists.
* Window attention (two forks, from 512 valid slots on): the K scan by ranges of 32-position
  blocks with per-block maxima, folded by the caller in block order; then exp, weight
  quantisation and the V sweep by range, int32 partial sums added by the caller.
* Rows of a block run (unread rows, n > 1) stay on the single-thread kernels.

## Stage S4: the speculative step (done)

Flow for one stream byte (`g` = the guess of its model byte):

1. main thread, start of the byte (behind `SemByteStart`; `runner.cpp` -> `Predictor::NeifSpecLaunch`):
   checks that a plain step can follow (`PPMD::SpecSafe`, `Tf1Engine::SpecPossible`), computes the
   match channel behind g (`Match::Peek`, pure), posts the job to the transformer thread.
2. transformer thread (`Predictor::NeifSpecJob`): saves the PPMd model object and starts the undo
   journal (`PPMD::SpecBegin`, `ppmd_page::jn_*`), runs `ByteUpdate()` of a **shadow wrapper** (a
   second `PPMD` object on the same model, so the row comes from the same code), then
   `Tf1Engine::SpecRun`: prior and channels as `Step()` makes them, and
   `TransformerOpt::spec_run`: one read row on the **shadow engine** (own row and scratch, shared
   weights and K/V rings; the fast weights are not touched, the relu^2 codes of the last four MLPs
   are kept). The KDA states are not copied: `kda_layer_step_io` (`src/tf1/opt/kda.cpp`) reads a
   layer's state from the buffer that is current (`kS(ki)`) and writes the decayed and updated
   state into the other one of two buffers, without advancing the position.
3. main thread, `Perceive()` of the 8th bit: `Predictor::NeifPpmdByteUpdate` waits for the PPMd part;
   byte == g: keeps the journal's result and copies the shadow wrapper's arrays (`SpecAdopt`);
   else: tells the engine to stop, undoes heap and model object, runs the real `ByteUpdate()`.
4. main thread, `Tf1Engine::Step` unchanged; in `TransformerOpt::step` (4 arguments) the engine
   waits for the transformer thread and takes the shadow's row if token, prior (208 float16),
   channels, position and epoch are the ones it ran on and the row is a read row with an empty
   row queue (`spec_commit`: copies x, xn, key, logits, row; runs `fw_apply` x 4 and the second
   head; makes the buffers the shadow wrote the current KDA states, `kS_cur[ki]`, and advances
   their positions; t++). Otherwise it runs the row as always, in place in the current buffers.
5. Every other entry into the engine (`begin_article`, an unread row) first stops and joins the
   speculation (`spec_join`).

Known simplifications to revisit:
* (done on 17 Nov: the first S4 binary copied the KDA states, 12 x 58 KB per speculative step; the
  final one uses the two-buffer kernel above, DESIGN 3.4)
* the pool workers and the transformer thread spin; nothing parks them while the main thread works;
* four busy threads are slower than three under WSL2 on this PC (one run each);
* a heap-full exit of PPMd inside a speculation is not caught (14 GB heap; not seen);
* the main thread does not help the transformer thread while it waits for the step.

## Measurements (17 Nov; raw lines in `fast/results/laptop/mt_results.txt`)

All on the 60.0-60.5 M window with 100 k of warm-up (598,569 model bytes), WSL2, threads on virtual
CPUs 1-3. The single-thread time of this window drifted between 236 s and 280 s over the day (the
laptop slows when it has been loaded for hours), so every figure is a ratio to the single-thread run
next to it; ranges are the spread over the rounds.

Identity (the gate), all passed:

| run | result |
|---|---|
| reference chunk 250.0-250.3 M (warm 1.2 M), `-t`, 3 threads in the step (S1) | sha 727e97767d69fb56, total_bits 2508178.191 |
| the same, `-t`, speculative step + 1 worker, 10 % of the guesses forced wrong (149,765 PPMd undos), copy-KDA binary and final binary | sha 727e97767d69fb56 both times |
| the same chunk decoded with `-u`, 3 threads (S1) | 1,495,402 bytes, none wrong |
| decoded with the speculative step (guess from the stream), 10 % forced wrong, both binaries | 1,495,402 bytes, none wrong |
| 60.0-60.5 M, every timing run below | sha e3c7340dce4cef65 / none wrong |

S3: single-thread decode against single-thread `-t`: 263.5 / 257.1 s against 263.3 / 259.0 s in two
interleaved rounds: **the same time** (ratio 1.00).

S1 (threads inside the step, main thread coordinates), two interleaved rounds:

| | 2 threads | 3 threads | simulated (DESIGN 0a, large regions, with clock) |
|---|---|---|---|
| `-t` | 1.06x, 1.05x | 1.07x, 0.94x | 1.15x / 1.12x |
| `-u` | 1.10x, 1.03x | 1.07x, 1.04x | |

With the profile on (2 threads): the transformer section falls from 165.6 s to 151.6 s (step 1.09x
faster; the model said 1.25x), and fxcm's half of Predict on the main thread rises from 43.2 s to
47.8 s. A worker that only spins already costs fxcm 10-25 % (`NEIF_MT_TEST=1`; the cause is not
pinned down, `neif_par_bench2.cpp` is inconclusive).

S4 (speculative step, the guess always right: `-t`, and `-u` with the guess from the stream), final
binary (out-of-place KDA states), interleaved:

| | main + transformer thread (2) | + 1 worker (3) | + 2 workers (4) | simulated, perfect guess, with clock (2 / 3 large regions / 4) |
|---|---|---|---|---|
| `-t` | 1.19x, 1.33x | 1.31x, 1.43x | 1.26x (earlier binary, one run) | 1.30x / 1.47x / 1.51x |
| `-u` | 1.27x | 1.37x | 1.20x (earlier binary) | |

(Earlier binary, which copied the KDA states: `-t` 1.22x / 1.37x, `-u` 1.20x / 1.23x, against
baselines taken before the machine had slowed.) 99.9 % of the speculative rows are taken (597,970
of 598,569); the rest are rows the driver feeds differently (prompt rows, document starts).

The upper end of each S4 range rests on one slow single-thread run (`x_t0b`, 280.3 s; the others
of the day are 236-264 s). Against the median single-thread time (257.8 s): `-t` 1.19-1.23x at two
threads and 1.32x at three.

Profile of the main thread with the speculative step (2 threads, 205.6 s): fxcm's half of Predict
is 50.0 s against 43.2 s in the single process (+16 %), PPMd is 0.3 s (adoption only), and 117.7 s
(197 us per byte) go to the section "transformer driver and network", which here holds the wait
for the transformer thread, the driver, the fast-weight update and the commit. So the cycle is
bound by the transformer thread. How much longer the step itself runs there than inside the
single process is not measured (no timer on that thread yet). The cause of the shortfall is **not
pinned down**: a second thread that only spins already costs fxcm 10-25 %, so the 16 % needs no
cache explanation; shared L3, power/frequency limits and virtual-CPU placement all fit, and the
four-thread result fits the last two better. Next steps that would separate them: a busy/idle
timer on the transformer thread and parking the waiting threads with `tpause`/`umwait` (available
in this WSL2 guest).

Runs added after the review of these results (binary rebuilt with the SV code, sha256 7a189017...):

* **Head of the stream** (bytes 0-1.5 M, model from byte 0; a window cannot start in the metadata,
  the warm-up needs a page start): single `-t` sha d0508669f70a50e5; `-t` with the speculation and
  10 % faults the same sha; `-u` with the speculation and faults and `-u` as a real decoder would
  run (`NEIF_MT_SPEC=1`) 1,500,000 bytes, none wrong. Here the speculation is **refused**
  for 1,499,999 of 1,500,000 bytes (`not possible`), a path the page-text windows never took. The
  region where bytes that can and cannot be speculated alternate (13-15.6 M) is the `mixed` step of
  `neif_sv_batch.sh` (results under SV).
* **Fault-rate bracket: a measured lower bound for S5.** 60.0-60.5 M, `-t`, the speculative step
  with the transformer's own miss rate injected as wrong guesses that are found only at the byte's
  last bit (S5 would find most of them earlier and guess again). Three interleaved single-thread
  runs: 271.1, 271.0, 271.6 s. All sha e3c7340dce4cef65.

  | wrong guesses | 2 threads | 3 threads |
  |---|---|---|
  | 23 % (the 60 M window's top-1 miss rate) | 228.2 s, **1.19x** | 212.8 s, **1.27x** |
  | 32 % (the 250 M window's) | 235.9 s, **1.15x** | 221.8 s, **1.22x** |
  | none (S4, earlier rounds, other baselines) | 1.19-1.33x | 1.31-1.43x |

Two formal data races found by the review, neither can change the output (the values are not used
before the join): `Tf1Engine::NewDocument()` writes `pk_prev1/2` one line before it joins the
speculation that may read them; `Predictor::NeifSpecLaunch` writes `neif_spec_.g/mbyte/mlen`
before `SpecMark()` joins the previous job. Fix when the code is next touched: join first.

## What the measurements say about S5

The simulation of section 0a puts the transformer-row guess at 0.91-0.93 of the perfect guess at
two threads and 0.89-0.92 at three (lowest on the 250 M window and the reference chunk). On the
measured S4 figures that gives **about 1.1-1.2x with 2 threads and 1.2-1.25x with 3** for a real
decoder on this PC, against 1.20x and 1.34-1.36x simulated for the same scope (DESIGN 0a, large
regions only). S5 itself is a modest step from here (guess = best byte of the row at
`NeifSpecLaunch`, a hook behind each decoded bit to re-guess; the undo paths are exercised by the
fault injector, an abort and a new launch inside one byte are new), but it cannot change the bound
set by S4. The fault-rate bracket above measures the other side: **S5 would land between 1.15-1.19x
and about 1.2-1.3x with 2 threads, and between 1.22-1.27x and about 1.3-1.4x with 3**.

## Stage SV: the reference-assisted validation decode (built 17 Nov)

`NEIF_PIPE=2|3 cmix -u STREAM CODED` (plain `NEIF_TEST`, default off; `src/runner.cpp`
`NeifDecompressChunk` / `NeifUntestMain`, `src/neif_test.h` `g_sv`). STREAM is the EXPECTED coder
stream, CODED a chunk of `-t` or, with `NEIF_T_START=0`, a whole archive (header + coder output,
as `-e` writes it). At the first model byte the process forks the front stages of the pipeline of
mode `-t` (`NeifPipeStart`, unchanged): they walk the expected stream with PPMd and the
transformer (`NEIF_PIPE=2`: one process; 3: the layers from `NEIF_PIPE_SPLIT`, default 8, on in a
second one). The calling process stays the coder stage, but instead of coding the expected byte
it **decodes** CODED with the `Decoder` and the byte loop of `Decompress()` and compares every
decoded byte with the expected one. About 60 lines; `NEIF_PIPE` and `NEIF_MT` exclude each other.

Results (any exit code but 0 is not a pass):

| exit | line on stderr | meaning |
|---|---|---|
| 0 | `neif_sv: PASS: stream bytes [a, b) decoded and equal to the expected stream` | every byte of the range decoded and equal |
| 4 | `neif_sv: FAILED at stream byte N: ...` | the decoded byte N differs from the expected one, or the wrong byte N put the stages out of step before the comparison (same N) |
| 3 | `neif_sv: NO VERDICT: the run broke at stream byte N (...)` | the pipeline broke (a front stage died, no memory, a PPMd row that differs between stages): says nothing about the data; the bytes in front of N were equal |
| 5 | `neif_sv: NO VERDICT: the expected stream could not be read ...` | read error on STREAM (`NEIF_T_END` is clamped to its size, and the archive header must carry its length: exit 2 otherwise) |
| 1 | a message of the model's own checks | a wrong byte can trip an internal check inside `Perceive()` of its last bit, before the comparison; no position line then (the progress line gives it to within a block) |

`neif_sv progress: stream bytes [a, p) equal so far, R bytes/s, peak M MiB in all stages` is written
every `NEIF_SV_LOG` seconds (default 600), so a run that breaks late still certifies a prefix.
The JSON result (`NEIF_T_JSON`) has `first_wrong_byte`, `pipe`, `pipe_peak_pss_kb`.
`neif_mt_one.sh` exits with the exit code of cmix.

### Why a pass implies the plain decoder's result

The stages are processes. What they share: the two queues, the control block, the file `./.dict`,
the transformer's link-context temporary file (`tf1_z3.cpp`, made before the fork; every stage
writes its own tokens there and reads only below its own position, and what two stages write to
the same place is identical while decoded and expected bytes are equal), and the PPMd page pool
with `ppm.page` (the coder stage never touches it after `NeifPpmdDisown`). What the coder stage
takes from the front stages:

1. per model byte the PPMd result (record 'P': the integer symbol masses and bit-tree arrays, and
   the float row; the coder stage normalises the integers itself and stops if its row differs);
2. per transformer driver step the row, the fast-weight row, the hidden vector and the key of the
   semantic cache (record 'F');
3. the dictionary file `./.dict`, written by the first stage at the end of slot 1 of the text start.

Everything else the coder stage computes itself from the bytes it has DECODED: all stream
automata (forced bytes, grammar, gate zones, metadata blocks), fxcm, the cmix models, mixers, SSE,
BITLSTM, floor and end stages, the semantic cache, and the arithmetic decoder.

* **Each record is a function of a prefix of the expected stream** (the front stages never see the
  coded input), and the coder stage uses the record of byte i only behind the bits of byte i: the
  'P' and 'F' records of byte i are read inside `Perceive()` of its last bit, and the comparison
  of byte i with the expected byte follows before anything else happens (before
  `AfterStreamByte`, before the next byte's first prediction). `.dict` is read in
  `AfterStreamByte` of the last byte of slot 1, behind the comparison of every byte of the slot.
* **Induction.** Suppose the first i bytes decoded equal the expected ones. Then every record the
  coder stage has used is the value for exactly the prefix the plain decoder would have decoded,
  and by the pipeline's exactness (below) it is the value the plain decoder computes in place. So
  the coder stage's state is the plain decoder's state, the probabilities of the bits of byte i
  are the same, and the same coded input gives the same byte i. Hence: SV passes all bytes <=>
  the plain decoder decodes all bytes equal to the expected stream; and if SV fails first at byte
  N, the plain decoder produces the same wrong byte at N.
* **A divergence cannot be masked.** A wrong byte N is compared with the expected byte from the
  coder stage's own read of STREAM, not with anything a front stage sent. Between its decoding
  and that comparison the coder stage only finishes `Perceive()` of the last bit, where it reads
  the records the front stages wrote for the EXPECTED byte N. If the wrong byte makes it ask for
  another kind or number of records there, `Get()` stops the run (`the stages are out of step`,
  or `the stage in front has ended` at the end of the stream; exit 3) and prints the same
  position N. Nothing the front stages send can turn a wrong decoded byte into the expected one.
* **The premise: exactness of the pipeline** (a record equals the value one process computes in
  place). It is the property the `-t` pipeline was accepted on (identical coder output with
  `NEIF_PIPE` 0, 2, 3; earlier work, `neif_pipe_accept.sh`), it is checked here again per run
  (`-t` with `NEIF_PIPE=2` gives the single process's sha; plain `-u` and SV agree on pass and on
  the position of a failure). In all, pipeline and single process have been compared on about
  18 M of the 501 M stream bytes. A violation elsewhere gives a **false fail, not a false pass**: wrong
  records give wrong probabilities, and the arithmetic decoder leaves the stream within a few
  bytes of the first probability that differs by one step of its scale.

### What a pass certifies, what it does not, and the cheap step that closes each gap

A pass on a whole archive says: **decoding this archive with this build's model on this machine
gives exactly the expected stream**; the archive is intact and the coder made no error.

| not covered by a pass | why | cheap step that closes it |
|---|---|---|
| the expected stream is the right one | SV only compares with what it is given | sha256 of STREAM against the stream of the official compressor (its `.cmix.temp`, the file `-e` codes), or produce STREAM by running the official preprocessing |
| stream -> enwik9 (the inverse transforms behind `Decompress()`: text start, dictionary slot, rule stages, IP pack, metadata form, census block, preprocessor) | SV stops at the stream | run that tail once on the verified stream and compare the sha256 of the result with enwik9: some minutes. Not built: it needs the automata state at the end of the stream, which the coder stage of a whole-archive SV run has; about 40 lines copied from `RunDecompression()` behind a new switch |
| the official flavour (no `NEIF_TEST`, other PGO use) | SV runs the test flavour | SV on the OFFICIAL archive: a pass means the test flavour's probabilities equal the official COMPRESSOR's on every coded bit (S0c and the `official` run below do this for the first 10 M stream bytes, which are metadata; **no page text of the official flavour has been checked yet**). It says nothing about the official decompressor binary, which is a separate build |
| the official decompressor as a program: the packed executable and its stub, self-extraction, `Decompress()` itself (SV runs a copy of its loop with the comparison added) and the end-of-stream checks behind its loop, the PPMd page pool of helper processes (`ppmd_page::hilfs`, said not to change the model output), memory and time limits of the contest machine | not executed by SV | one real run of the packed decompressor; SV cannot replace it, it only says beforehand that the archive and the model agree. A watcher that compares the real run's growing stream file (`Decompress()` writes it byte by byte) with the expected stream stops a bad real run at the same byte, at no cost |
| bytes of the archive behind the last one the decoder reads | the decoder never looks at them | compare the archive's length and sha256 with the compressor's output |
| exactness of the pipeline on the parts of the stream where it was never compared with one process | compared on about 18 M of 501 M bytes | none needed for a pass (a violation makes SV fail, not pass); a fail there needs the plain decoder on that window to tell which |
| other machines | floating point of the predictor group is build- and CPU-dependent | run SV there (same command) |

### Results (17 Nov; `neif_sv_batch.sh`, raw lines `q_*`, `v_*`, `z_*`, `o_*`, `g_*` in `fast/results/laptop/mt_results.txt`)

Positive and negative tests on the quick chunk 60.00-60.03 M (36,302 bytes):

| run | result |
|---|---|
| `-t` single, `-t` with `NEIF_PIPE=2` | same sha 9e925d7846f34983 (the pipeline's records are the single process's values) |
| `-u` plain; SV with 2 and 3 processes | all 36,302 bytes equal; `neif_sv: PASS`, exit 0 |
| one bit of the coded input changed (byte 3477 of 6954) | plain `-u`: first wrong byte **60,013,096**; SV 2 and SV 3: `neif_sv: FAILED at stream byte 60013096`, exit 4 |
| one bit of the expected stream changed (byte 60,015,000) | plain `-u`: first wrong byte **60,015,000**; SV 2 and SV 3: `FAILED at stream byte 60015000`, exit 4 |

| the coded input cut in half (3477 of 6954 bytes) | plain `-u` and SV 3: first wrong byte **60,013,093**, exit 4 |
| the last byte of a page-end anchor of the expected stream changed (byte 60,005,536; the front stages then see another structure) | plain `-u`, SV 2, SV 3: **60,005,536**, exit 4 (found by the comparison; the out-of-step exit was not reached by any test) |
| a front stage killed with `kill -9` in a run of the official prefix from byte 0 | `neif_pipe: stage 2: a front stage ended early`, `neif_sv: NO VERDICT: the run broke at stream byte 724820 (...)`, exit 3; the progress lines up to 626,683 bytes are in the log |

The first four rows were run with the binary of the timings (sha256 7a189017...) and again, with
the last three, with the final binary (sha256 cd291351..., the reporting changes of the review);
the quick identity check of the speculative step (`neif_mt_s4quick.sh`) passes on the final
binary too (sha 9e925d7846f34983 with and without 20 % faults, decode none wrong).

Speed and memory, interleaved with the plain single-thread decode `-u` of the same binary (model
seconds; memory: the largest sum of the stages' proportional set sizes, sampled every 5 s; the
single process's figure is its peak RSS):

| window | plain `-u` | `NEIF_PIPE=2` | `NEIF_PIPE=3` | memory single / 2 / 3 |
|---|---|---|---|---|
| 60.0-60.5 M, round a | 266.2 s | 210.8 s, 1.26x | 116.4 s, **2.29x** | 8.32 / 8.49 / 8.69 GiB |
| 60.0-60.5 M, round b | 270.3 s | 189.0 s, 1.43x | 117.6 s, **2.30x** | |
| 250.0-250.3 M, round 1 | 701.8 s | 482.1 s, 1.46x | 290.8 s, **2.41x** | 8.48 / 8.67 / 8.89 GiB |
| 250.0-250.3 M, round 2 | 712.7 s | 485.1 s, 1.47x | 291.7 s, **2.44x** | |

With three processes the decode runs at 5,090-5,145 stream bytes per second in page text on both
windows. CPU time of all stages is 1.08-1.24x the single process's. With three processes all
stages are nearly always busy (first stage 90 %, second 97-98 %, coder 86-89 % of the wall time),
so the speed holds only while none of them gets slower than in these windows. The single-thread times are 5-10 % above the S0 figures of 16 Nov (the laptop
had been loaded for a day); the ratios are from runs next to each other.

From stream byte 0 (a whole archive is this, with `NEIF_T_END` unset):

| run | result |
|---|---|
| the OFFICIAL coder output (`/work/off/out9.prefix`, 2 MiB written by the official comp9), stream bytes 0-10 M, SV 3 | **PASS**, 10,000,000 bytes, 1005 s; plain `-u` 1051 s (1.05x: the head of the stream is metadata, where the transformer is mostly off and the coder stage is nearly the whole work); 10.35 GiB |
| the test flavour's chunk of stream bytes 0-15.6 M (metadata, dictionary slot, the passage into page text near 15.28 M), SV 3 | **PASS**, 15,600,000 bytes, 1921 s; single-thread `-t` 2805 s (1.46x); 9.96 GiB |
| the same chunk, the S4 speculative decoder with 10 % wrong guesses (`g_us2f`) | 15,600,000 bytes, none wrong; speculation launched for 2,576,080 bytes, refused for 13,023,920, 257,869 PPMd undos: the mixed region the S4 review asked for |

**Whole archive on this PC, expected:** `NEIF_PIPE=3 cmix -u STREAM ARCHIVE` (no `NEIF_T_*`):
0.5 h for the first 15.6 M bytes (measured) plus 485.5 M bytes of page text at about 5,100 bytes/s
= **about 27 hours, or more**, against about 55-63 hours for the plain single-thread decode
(2,150-2,450 bytes/s in page text depending on how warm the machine is). What the figure rests on
and what it does not know:

* the windows start PPMd from an empty heap; PPMd's paging at a heap of several GB has never run
  in a forked stage, and the first stage may get a smaller page pool than a single process (it
  sizes the pool from its own memory plus the coder's). With every stage 86-98 % busy, a stage
  that is 10 % slower in the full run makes the whole run 10 % slower;
* windows late in the stream are not measured (420 M was 3 % slower than 60 M in S0);
* memory: 10.35 GiB peak from byte 0 against 15.6 GiB in this WSL; late in the stream not measured
  (the progress line of a real run records it).

**Before a whole-archive run** (found by the review of this stage):

1. **The whole official archive is not on this PC** (only its first 2 MiB, `/work/off/out9.prefix`).
   Writing it is a single-process run of the official compressor, and it cannot overlap an SV run
   (memory).
2. **First extend the official prefix into page text**: let the official comp9 run to about 17 M
   stream bytes (about 3.1 MB of output, about 1 h: `neif_official_prefix.sh 3145728`), then
   `OFF_END=16500000 neif_sv_batch.sh official` (about 40 min). The 2 MiB prefix ends near stream
   byte 10.44 M, in the metadata; a test flavour that is not a proxy of the official one in page
   text would otherwise show only as a fail at byte 15.3 M of the long run.
3. **Decide the order.** SV first costs 27 h on the pass path. The alternative is the real
   decompressor run with a watcher on its stream file, which aborts a bad run at the same byte
   for free and covers everything SV does not; SV then only buys earlier detection (2.3x).
4. Accept only: exit 0, the `PASS` line with the range `[0, 501062281)`, `decoded_bytes` equal to
   the stream length, and the sha256 of stream and archive recorded next to it.

## Multi-process compression in the real path (built 17 Nov evening; NOT the official run)

A convenience for development: the contest is one core. Default off, plain `NEIF_TEST`
(`src/runner.cpp` `NeifEStart` / `NeifEEnd`, called from `RunCompression()`; `src/neif_test.h`
`g_out_path` in `PipeFail`). About 110 lines.

* **Enable:** `NEIF_PIPE=3` (or 2) in the environment of the normal compress command
  `cmix -e enwik9 OUT`; `NEIF_PIPE_SPLIT` as in `-t`. Without it nothing changes. `NEIF_MT` set
  together with it is refused.
* **How:** `NeifEStart` sets the state of mode `-t` for a run from stream byte 0 without a
  structure-only pass (`g_on`, `g_skip = false`), so `Compress()` forks the stages at byte 0 exactly
  as `-t` with `NEIF_T_START=0` does; unlike `-t` the run goes to the end of the stream and takes
  the end-of-stream checks of `Compress()`, `dict_article::Pruefen()` and the assembly of archive9.
  The front stages reopen the temporary stream file (`OUT.cmix.temp`). The page pool accounting
  across forks is the one of `-t` (`ppmd_page::neif_fork_child`), unchanged.
* **What it prints (stderr):** at the start `neif_pipe: MULTI-PROCESS COMPRESSION (NEIF_PIPE=3, 3
  processes; test build): a development convenience, NOT the official single-core run.`; at the end
  the stage lines of `-t`, `neif_e: N stream bytes -> M B, coder S s wall, C s cpu in all processes,
  processes 3, peak memory P MiB (largest sum of the processes' proportional set sizes)` and
  `neif_pipe: MULTI-PROCESS COMPRESSION ended in order (3 processes). NOT the official single-core
  run; ...`.
* **A broken pipeline leaves no output:** `PipeFail()` in the coder stage (a front stage died or
  ended out of order, no shared memory, stages out of step) and a front stage that did not end in
  order behind the last byte remove OUT, print `neif_pipe: MULTI-PROCESS COMPRESSION FAILED at
  stream byte N (...): the output file OUT was removed and no archive is written. Exit code 3.`
  and `_exit(3)` before archive9 is put together. Not covered: the coder stage itself killed from
  outside (a truncated OUT stays, as in a one-process run; no archive9 either way).
* **`NEIF_E_STOP=n`** (test switch, with or without `NEIF_PIPE`): the coder stops behind n stream
  bytes, flushes, prints `neif_e: STOPPED ...` and the process ends; OUT = header + coder output of
  the first n stream bytes, no archive9. This is the affordable identity check.
* **Running `-e` with a test build:** `-e` unpacks the hint package, the article order and the
  weights from its own file, so the unpacked test binary gets the last bytes of the official comp9
  appended: `neif_e_build.sh [/work/tbKe]` (packed test build `-DNEIF_TEST=1 -DNEIF_PACKED=1`, the
  same 18 discarded profile entries; result `/work/tbKe/e/cmix`). `neif_e_run.sh NAME STOP PIPE`
  runs it in `/work/e/NAME` (copy as `./cmix`, `PATH=.:$PATH`), compares the stream it codes with
  `/work/bed/stream.bin`, one line per run in `e_results.txt`.
  `neif_e_batch.sh single pipe3 pipe2 official sv quick` is the full check (about 3.5 h);
  `neif_official_prefix2.sh` writes an official prefix into `/work/off3` with a settable time limit.

### Verified (17 Nov; binary `/work/tbKe/e/cmix`, sha256 2d0ba2b1...0bd7e2; `e_results.txt`)

| run | result |
|---|---|
| `-e` stopped at 200,000 stream bytes, one process (`s_p0`) | OUT 42,694 B, sha256 5b3750d2b3a982a7e5a1e37af238943efa45ccd24431f2a437baec54f64d477c; coder 16.9 s; 6,223 MiB peak resident |
| the same with `NEIF_PIPE=3` (`s_p3`) | OUT 42,694 B, **the same sha256**; coder 17.5 s wall, 20.6 s cpu; 6,581 MiB (pss sum) |
| the stream both runs coded (`OUT.cmix.temp`, 501,062,281 B) | identical to `/work/bed/stream.bin`: the test binary's `-e` preprocessing gives the official stream |
| first 42,000 B of `s_p0` against the official prefix `/work/off/out9.prefix` | identical |
| a front stage killed 8 s into a 3-process run (`k_p3`, stop 2 M) | `a front stage ended early`, the FAILED lines, exit code 3, no OUT, no archive9 |
| `runner.cpp` preprocessed without test macros, before and after the edit | identical text (4,535,254 B) |

The 200,000 bytes are the head of the metadata region: the fork at byte 0 in the real path, the
flush and the orderly end of the stages are exercised; the pipeline has nothing to gain there (the
transformer is mostly off), so there is no speed figure for the real path yet.

### NOT verified

* **Everything behind stream byte 200,000 in the real path:** the dictionary slot (`.dict` handed
  from the first stage, `PpmdWaermen`), the passage into page text near 15.28 M, page text. The
  planned run (`single`, `pipe3` to 17 M with sha comparison, `pipe2` to 3 M) was stopped by the
  host 4 minutes in, during the first run, because the PC ran low on memory; nothing was produced.
  The same byte loop is covered by `-t` (about 18 M bytes compared, see SV), which is why no
  difference is expected, but it is not shown.
* **Timing and peak memory of the real path from byte 0 into page text:** not measured (same run).
* **The end of the stream:** the end-of-stream reports, `Pruefen()` and archive9 after a pipelined
  run have never run; only a full run shows them.
* **The decode of official page text** (`official` + `sv` steps: official comp9 to 3 MiB into
  `/work/off3`, then `NEIF_PIPE=3 cmix -u` over it): not started. Still worth doing; the `official`
  step also compares the official output with the test build's one-process `-e` output directly.
* **The official build** was not rebuilt after this change (only the preprocessor comparison
  above). Run `neif_build_official_dev.sh` before relying on it.
* `quick` (the `-t` reference chunk on `/work/tbKe`) was not run.

To finish: `bash neif_e_batch.sh single pipe3 pipe2 official sv quick` in WSL as root
with nothing else large running, then `neif_build_official_dev.sh`.

`neif_test.new.patch` (264,827 B, sha256 9d4d2aca...f7c822) is `git diff --binary HEAD
-- src` of the development tree plus the two untracked headers (`src/neif_mt.h`,
`src/tf1/opt/act16_test.h`); applied to a pristine checkout of 35d1957a it reproduces `dev/src`
byte for byte (219 files). `neif/neif_test.patch` was left alone: how it is generated is not
documented, and it holds an older subset (12 files).

## Official build

Re-verified on 17 Nov three times: after S3, S1 and S4 (with the two-buffer KDA kernel) were in
the development tree, after the SV code, and after the last source change (the reporting changes
of the SV review; the test binary of that state is `/work/tbmt/cmix`, sha256 cd291351...). `bash neif_build_official_dev.sh` builds the official
flavour of the edited tree and of the untouched one and compares them:

    comp9 4042564 B
    976bf2d309707ab555e0c92720da5508ff0e5e6b846fc7082234262b06767f82  /work/b1/comp9  (edited tree)
    976bf2d309707ab555e0c92720da5508ff0e5e6b846fc7082234262b06767f82  /work/b0/comp9
    OFFICIAL BUILD OF THE EDITED TREE: IDENTICAL to the published comp9

Nothing is committed; nothing ran in the cloud.
