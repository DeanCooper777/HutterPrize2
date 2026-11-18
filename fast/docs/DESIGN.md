> **Working notes, published as written (16-17 Nov 2026), with machine-specific paths removed.**
> The design of the threaded decoder and of the reference-assisted verification decode, with the
> measurements that decided what was built. Source paths are relative to the root of this
> repository; line numbers refer to the tree as it stood on 16 Nov and have moved since. It names
> scripts and raw files of our internal tooling that are not in this repository (`neif_par_*`,
> `meas/par_*`, `bench.txt`); `fast/` holds the portable subset (see `fast/README.md`).
> "This PC" is an Intel Core Ultra 288V (Lunar Lake) laptop under WSL2. Status of each stage:
> `PROGRESS.md` next to this file; later results: the top-level `README.md`.

# A multi-threaded decoder for cmix-neif 2.0: design and measurement plan

Status (17 Nov): stages S0a, S0b and S0c are **done and measured**; their results are in
section 0a and replace the predictions, which stay visible beside them. Stages S3 (chunk decoder),
S1 (threads inside the step, large regions) and S4 (speculative step with a certain guess and a
fault injector) are **built, identical in output and measured**: section 0b and `PROGRESS.md`.
The reference-assisted validation decode (SV, section 3.6) is **built and measured** as well:
2.3-2.4x the plain single-thread decode with three processes (1.3-1.5x with two), a pass on the
first 10 M stream bytes of the official archive, both negative tests caught at the right byte;
the argument why a pass implies the plain decoder's result, what a pass does not certify, and the
figures are in `PROGRESS.md`, "Stage SV". S5 is not built (on hold).
Sections 1-12 are the design as written on 16 Nov from reading the code: numbers marked *profile*
there come from the single-core profile I was given (350 us per stream byte, transformer 59-67 %),
numbers marked *model* are arithmetic on those. The only multi-thread times that exist are the
kernel bench of S0b. The recommendation was reviewed by an independent reviewer;
section 12 says what changed.

Source paths are relative to the root of the source tree (the tree as it stood on 16 Nov; the six files
the instrumentation changes are pinned in `base.sha256`).

Caveat on the base numbers: the only raw single-core table I could open
(`neif_a16_timing.txt`, 16 Nov, window 60.0-60.5 M, three runs of the unchanged
path) shows 442-498 us per byte with the transformer at 67 % (296-333 us), and 12 % between
identical runs. The shares agree with the profile I was given; the absolute times and the ratio
T : R do not exactly. All bounds below are therefore ratios on assumed inputs; the tool recomputes
them per window from measured times.

## 0. Summary

* **Measured on 17 Nov (section 0a), in place of the predictions below:** the decoder without a
  reference reaches about 1.2x with 2 threads, 1.35-1.5x with 3 and 1.4-1.6x with 4 on this PC
  under WSL2 (predicted 1.34x / 1.77x / 2.06x before the clock loss, "about 1.8x with 4 threads"
  after it). The reference-assisted decode reuses a pipeline that ran about 2x faster than one
  process in compression (1.8-2.2 observed). Verdict: build the reference-assisted decode if it
  is acceptable; C and D only if a decoder that needs no reference is required; the rest no-go
  or undecided (table at the end of 0a). The verdict went through the reviewer again; its
  corrections (the scope of C behind the 1.5x, the pipeline range, the main thread joining the
  step) are in 0a.
* The decoder knows byte n only after its 8th bit; the transformer step of byte n then produces the
  row that every bit of byte n+1 is mixed with. Per byte the critical path is `T + R`:
  T = the transformer step (220 us, *profile*), R = everything else (130 us).
* **First a question for the user, because it changes the plan: must the fast decoder work
  without the expected output?** A *validation* decode has the expected stream on disk. Then no
  guessing is needed: the existing forked pipeline (`NEIF_PIPE`, 2.4x in compression) can run its
  front stages (PPMd, transformer) on the expected stream while the coder stage *decodes* and
  compares every byte (section 3.6). It passes exactly when the plain decoder would reproduce the
  stream, costs about 250 lines on top of what exists, and its bound is the compression
  pipeline's (about 2.4-2.5x on three fast cores). If that is acceptable it should be built
  first; the rest of this document is the decoder that needs no reference.
* For a decoder without the reference, three mechanisms carry the gain:
  * **C** threads inside the transformer step. All large kernels can be split with bit-identical
    results (integer sums or element-wise float steps; checked in the code by me and by the
    reviewer). *Model*: T = 220 -> 157 / 129 / 115 / 101 us with 2 / 3 / 4 / 8 threads, floor 73 us.
  * **D** speculation: start the step of byte n+1 on a guess of that byte while its bits are
    decoded, on other threads, into a shadow slot; commit on a hit. Turns `T + R` into about
    `max(R, T)` on a hit. Needs a journalled PPMd update and a shadow of the KDA states.
  * **F** fxcm on a thread of its own (it reads nothing of the transformer or PPMd, and is half of
    R). No gain on this PC (the step stays the critical path); for many-core machines only.
* A (byte-start work under the step) is worth about 1.05x and is subsumed by D; B (deferred
  learning) is not possible bit-exactly at a profit. Sections 3.1, 3.2.
* Predicted speed-up of the model bytes as of 16 Nov (kept for comparison; the measured values
  are in section 0a), *model*; assumptions: hand-off 0.4 us, first guess right for 75 % of bytes,
  a wrong guess replaced after 60 us on average, every core at its single-thread clock:

  | threads | C alone | D alone | D + C | D + C, perfect guess |
  |---|---|---|---|---|
  | 2 | 1.22x | 1.34x | - | 1.43x (D) |
  | 3 | 1.35x | - | 1.77x | 1.92x |
  | 4 | 1.43x | - | 2.06x | 2.27x |
  | 8 (4 fast + 4 slow at 0.5) | 1.52x | - | 2.36x | 2.65x |

  With all cores loaded the clock is lower; taking 10-15 % off: **about 1.8x with 4 threads**
  (a 49 h decode in about 27 h). **The 8-thread figure is not a prediction**: it rests on the slow
  cores (their cache situation and their real speed on these kernels are unknown) and on spinning
  on every virtual CPU of WSL2; the model's 2.0-2.15x could turn out no better than 4 threads.
  Pessimistic corner (hand-off 0.8 us, 60 % first guesses, 15 % clock loss): 1.5x with 4 threads,
  1.3x for D alone. **Honest range before any measurement: 1.3x to 2.2x.**
* Build order (section 5): measurement, a threaded kernel bench and an official-archive check
  first, all cheap; then, depending on the answer above, the reference-assisted decode or
  S1 (two kernels of C) -> D -> rest of C.
* Everything goes behind one new macro (`NEIF_MT`, with `NEIF_TEST`), off by default at run time;
  the official build is not touched.

## 0a. Measured results of S0 (17 Nov), and what they change

All runs single-process on one pinned virtual CPU of WSL2, page text, three windows of 1 MB after
1 MB of warm-up (60 M, 250 M, 420 M) plus the reference chunk 250.0-250.3 M. Raw files:
`meas/par_*.json`, `meas/par_*.log`, `meas/summary_h*.json|txt`, `meas/par_results.txt` (the
reference chunk and the overhead runs), `bench.txt` (`bench_run1.txt`: an earlier run whose clock
chain was broken); tools `neif_par_*.sh|py|cpp`.

**The instrumentation changes no value and costs little.** With NEIF_PAR compiled in and on, the
reference chunk gives sha 727e97767d69fb56 and total_bits 2508178.191, and the three windows give
the `kept_bits` of the earlier plain runs (1122113.652, 1677040.845, 1256584.842). On the short
timing window (60.0-60.5 M) the same binary takes 236.7 / 237.4 s with NEIF_PAR off and 238.3 s
with it on (+0.5 %); the measuring build with the per-site table of the transformer takes 245.1 s
(+3.4 %). The PGO build discards the same 18 profile entries as the reference test builds, none
new. `sync_errors` is 0 in every run. The official build of the development tree with all of
this applied (and the 16-bit experiment's changes) is byte-identical to the published `comp9`
(sha256 976bf2d3...6767f82, `neif_build_official_dev.sh`, 17 Nov).

**Time per stream byte** (us; kept bytes only):

| window | total | transformer step | byte start, independent of the row | byte start, needs the row | per-bit work (of it fxcm) |
|---|---|---|---|---|---|
| 60 M | 398 | 272 | 36 | 16 | 73 (43) |
| 250 M | 429 | 286 | 40 | 20 | 84 (45) |
| 420 M | 408 | 277 | 37 | 17 | 77 (44) |
| assumed on 16 Nov | 350 | 220 | (15-20 for A) | | |

The transformer is 67-68 % of the time, not 63 %. The bit loop up to the step (W) is 107 / 120 /
112 us; fxcm's first half on bit 0 alone is 32 us (byte parse and context hashing), the other
bits about 6 us each. Bit 0 is known 33-37 us after the byte start, bit 7 after 97-109 us.
99.6-99.9 % of the steps are one read stream row with nothing waiting in the row queue, so
speculation is possible on practically every byte; injected bytes and extra rows are under
0.5 us per byte.

**Hit rates of the guesses** (bytes that are not forced; all three windows are one region, no gate
zone):

| window | bits per byte | transformer row top-1 | top-2 | right after 1 / 2 / 3 decoded bits (top-1 policy) | top-2 policy after 0 / 1 / 2 / 3 bits | any of six sources | PPMd top-1 | match byte |
|---|---|---|---|---|---|---|---|---|
| 60 M | 1.12 | 77.1 % | 85.1 % | 78.1 / 80.0 / 81.8 % | 85.0 / 86.0 / 87.5 / 88.8 % | 87.4 % | 63.5 % | 48.9 % |
| 250 M | 1.68 | 67.6 % | 77.7 % | 69.3 / 72.5 / 75.3 % | 77.7 / 79.4 / 82.2 / 84.8 % | 80.6 % | 52.1 % | 30.7 % |
| 420 M | 1.26 | 75.0 % | 83.5 % | 76.1 / 78.6 / 80.8 % | 83.4 / 84.6 / 86.6 / 88.3 % | 85.8 % | 58.9 % | 42.6 % |
| assumed | | 75 % | | wrong guess replaced after 60 us | | | | |

The row times the fast-weight row is no better than the row (+0.1 point); my adaptive selector
over six sources is slightly worse (-0.4); a better combiner is capped by "any of six". **A wrong
first guess is replaced late**: three decoded bits (60 us into the loop) recover only 5-8 points;
half of the misses are still open after bit 5. That is the main reason D falls short of the
prediction. (Only about a third of the bytes have all 8 bits coded; the policies measured do not
use the bits the grammar forces, a small reserve.)

**S0b, the threaded kernel bench** (`bench.txt`; pool of spinning threads, WSL2):

* One fork/join with a realistic payload (192 bytes written by the coordinator, one line back per
  thread), mean / median: 2 threads 0.25-0.33 / 0.22-0.26 us; 3 threads 0.41-0.49 / 0.32-0.34;
  4 threads 0.55-0.60 / 0.38-0.56; 8 threads 1.0-6 / 0.8-0.9 with outliers of 0.2-1.6 ms.
  `pause` in the spin makes no consistent difference.
* The dense kernel (16 layers of 1024 x 192, one fork per layer, outputs identical to one thread
  in every run), weights in cache, one thread 35 us per token: 2 threads 19-22 us (1.6-1.85x),
  3 threads 16-18 us (2.0-2.2x), 4 threads 14.8-15.2 us (2.3-2.4x) in half of the runs and
  22-26 us in the other half, 8 threads 21-24 us. The effective cost per fork over the ideal
  split is 0.1 / 0.3 / 0.36 us at 2 / 3 / 4 threads.
* With the coordinator's cache flushed before every token (48 MB of other data): one thread
  102-110 us, 2 threads 63-74, 3 threads 51-63, 4 threads 44-48 or 63-100.
* The small region (three 192 x 192 matrices, one fork): 0.89 us on one thread, 0.63-0.83 on two,
  0.59-0.89 on three: no reliable gain.
* Clock (a latency-bound integer chain, 1.125 ns per step alone on every virtual CPU): 2 busy
  threads -1 %, 3 busy -6 to -11 %, 4 busy -10 to -13 %, 8 busy -21 to -29 %.
* **Virtual CPUs are not cores.** Every one of the eight virtual CPUs runs at the fast-core speed
  when it is the only busy one, and all eight run at the same reduced speed when all are busy:
  Windows places them. Pinning inside WSL2 therefore fixes neither the core type nor, strictly,
  the core; the fixed-partition L2 argument of 3.3 holds only as long as Windows leaves a busy
  virtual CPU where it is. The slow cores' own speed cannot be measured from inside WSL2.

**S0c, decoding official bytes.** The test flavour (`cmix -d`, PGO test build with NEIF_PAR
compiled in and off) decoded the 2 MiB prefix of the official comp9's coder output: its first
**10,447,599 stream bytes are identical** to the stream the official compressor coded; the first
differing byte is where the 2 MiB of input end (the compressor's trace has 2,029,856 coded bytes
at stream byte 10,021,400 and 2,109,762 at 10,522,470). So the test flavour decodes official
bytes, as far as they exist here. What remains: these 10.4 MB are metadata blocks; page text
starts at 13.0 MB and needs about 2.6 MiB of official output (about an hour of the official
comp9, then an hour of decoding).

**Bounds with measured inputs** (model bytes; the per-byte simulation in the binary, the step's
speed from the measured per-site table, the hand-off measured for the pool size in use: 0.3 us at
2 threads, 0.45 at 3, 0.6 at 4; "clock" = times 0.99 / 0.92 / 0.88 for 2 / 3 / 4 busy threads):

| threads | mechanism | predicted 16 Nov | measured inputs, 60 / 250 / 420 M | with the clock loss, all regions split | with the clock loss, large regions only | perfect guess, with the clock loss (all / large) |
|---|---|---|---|---|---|---|
| 2 | C alone | 1.22x | 1.25 / 1.25 / 1.25x | 1.24x | 1.15x | |
| 3 | C alone | 1.35x | 1.36 / 1.36 / 1.36x | 1.25x | 1.12x | |
| 4 | C alone | 1.43x | 1.40 / 1.40 / 1.40x | 1.23x | 1.10x | |
| 2 | D alone | 1.34x | 1.23 / 1.21 / 1.23x | 1.20-1.21x | the same | 1.30-1.32x |
| 3 | D + C | 1.77x | 1.63 / 1.60 / 1.63x | 1.47-1.50x | 1.34-1.36x | 1.64-1.67x / 1.47-1.50x |
| 4 | D + C | 2.06x | 1.80 / 1.77 / 1.80x | 1.55-1.59x | 1.37-1.39x | 1.75-1.78x / 1.51-1.54x |
| 2 | A alone | 1.05x | 1.086x | | | |
| 2 | F alone (fxcm thread) | 1.08-1.11x | 1.03-1.09x | | | |
| 4 | D + C + F | below D + C | 1.60-1.61x | about 1.41x | | |
| 4 | D with two guesses | below D + C | 1.26x (both groups at one-thread speed; below D + C in any split) | | | |

"All regions" splits everything section 3.3 lists; "large regions only" splits the MLP region and
the window attention and leaves the q/k/v, gate and KDA regions serial (stage S1 without S2).
The step itself on k threads: all regions 1.42-1.43x at 2, 1.63-1.65x at 3, 1.71-1.74x at 4
(predicted 1.40 / 1.71 / 1.91 at a hand-off of 0.4 us for every pool size); large regions only
1.25 / 1.37 / 1.42x. The clock factors are frequency only, from one bench run; the kernel under
load was noisier, so every clock-adjusted figure carries about +-0.1x.

**Revised range for the decoder without a reference, on this PC under WSL2: 1.2x with 2 threads;
with 3 threads 1.35x (large regions only) to 1.5x (all regions); with 4 threads 1.4x to 1.6x**
(and 4 busy threads were unstable in the bench), against 1.8x predicted for 4 threads. A 49 h
decode would take about 31-36 h. The reasons, in order of weight: wrong guesses are replaced
late; the transformer step is longer relative to the rest, so D + C stays bound by the step; the
hand-off grows with the pool; the clock loss eats what the third and fourth thread add.

Whether the small regions pay is **not decided** by S0b: the bench region (three 192 x 192
matrices in cache, 0.89 us) is smaller than the regions in the program (kimi q/k/v 2.5 us per
call, about 3.5 us per fork with the gates; the KDA step 2.8 us), where a 0.3 us hand-off is not
obviously a loss. A bench at that size with the cache flushed takes minutes and decides between
the two columns.

One design change the measurement points to, not yet simulated: in D the main thread is idle for
about half of every cycle (step 272-286 us against a bit loop of 107-120 us). If it joins the
step's pool once its bits are decoded, the reviewer's estimate on the mean cycle is about
1.36-1.45x with 2 threads (the stable configuration, 1 % clock loss) and 1.4-1.6x with 3. To be
put into the in-binary simulation before the reference-free route is chosen.

**For comparison, the reference-assisted decode (3.6):** the compression pipeline it reuses
(`NEIF_PIPE=3`) ran on the same three windows on another day with another build of the same
source: 353.3 / 378.2 / 368.0 s for the 2 MB of model bytes, against 805.1 / 859.4 / 816.4 s for
the single process here (`seconds_model` of `meas/par_*.json`; minus 3.4 % of instrumentation):
2.2x / 2.2x / 2.15x. Three pipeline runs of the reference chunk with other builds give 2.20x,
2.12x and 1.78x against today's single process. No pair of the same build and day exists, so the
fair statement is **about 2x (1.8-2.2 observed)**. Two things it does not yet show: the upper
transformer stage is the binding one and the coder stage is busy 83-89 % of the time, so the
figure carries over to decoding only if the decoding coder stage is at most 11-17 % slower than
the encoding one; and a pass certifies that the archive matches the stream under the test
flavour, not the official packed decompressor or the step from the stream to enwik9.

**Go / no-go:**

| mechanism | verdict | why |
|---|---|---|
| reference-assisted verifying decode (SV) | **go, first**, if a decode that needs the expected stream is acceptable | about 2x (1.8-2.2 observed) for the same pipeline in compression; about 250 lines; S0c says the flavour decodes official bytes (metadata part checked). S3 must time a single-process decode against `-t` on one window first (the 11-17 % headroom), and SV needs a test with one flipped archive byte so that a pass cannot be vacuous. |
| C, large regions (MLP, window attention) | **go at 2-3 threads, only together with D**; marginal at 4; no-go at 8 | hand-off 0.25-0.33 us at 2 threads and 0.41-0.49 at 3 against the 0.4 us threshold, 0.55-0.60 at 4; split kernel 1.6-1.85x / 2.0-2.2x; but C alone ends at 1.1-1.25x whatever the thread count |
| C, small regions (S2: q/k/v, gates, KDA heads) | **undecided** | the bench region was too small to tell; worth 0.13-0.2x on D + C if the in-situ regions split as modelled |
| D, speculation | **no-go for validation; go only if a decoder without the reference is required** | 1.2x alone, 1.35-1.6x with C, for about 1900 lines at medium risk; technically nothing stands in the way (99.6-99.9 % plain steps) |
| A, B, F, two guesses, 8 threads | **no-go** on this PC | 1.09x / not exact / 1.03-1.09x / below D + C / no stable hand-off |

## 0b. Stages S3, S1 and S4 built and measured (17 Nov)

Details, switches and how to resume: `PROGRESS.md`. Raw lines: `fast/results/laptop/mt_results.txt`. Everything
is behind `NEIF_MT` inside `NEIF_TEST` (the chunk decoder is plain `NEIF_TEST`); the official build
is unchanged (PROGRESS.md, last section).

**Identity, the gate of every stage, holds.** On the reference chunk (250.0-250.3 M, warm 1.2 M)
the coder gives sha 727e97767d69fb56 / total_bits 2508178.191 with 3 threads inside the step, and
with the speculative step while 10 % of the guesses are forced wrong (149,765 PPMd undos); the
chunk decoder `-u` reproduces all 1,495,402 stream bytes in the same two configurations.

**S3.** `cmix -u STREAM CHUNK` decodes what `-t` writes (nothing had to be added to `-t`).
Single-thread decoding takes the same time as single-thread coding (ratio 1.00 in two rounds).

**Measured against simulated** (60.0-60.5 M window, ratios to the single-thread run next to each;
"simulated" = section 0a for the 60 M window with the clock loss, C on the large regions only,
which is what S1 built):

| stage | threads | measured `-t` | measured `-u` | simulated |
|---|---|---|---|---|
| S1: threads inside the step | 2 | 1.05-1.06x | 1.03-1.10x | 1.15x |
| S1 | 3 | 0.94-1.07x | 1.04-1.07x | 1.12x |
| S4: speculative step, guess always right | 2 | 1.19-1.33x | 1.27x | 1.30x |
| S4 + 1 worker | 3 | 1.31-1.43x | 1.37x | 1.47x |
| S4 + 2 workers | 4 | 1.26x (one run) | 1.20x (one run) | 1.51x |

The upper end of each S4 range comes from one slow single-thread run (280.3 s; the others of the
day are 236-264 s). Against the median single-thread time (257.8 s) the S4 coder figures are
1.19-1.23x at two threads and 1.32x at three; the decoder round gives 1.27x and 1.37x.

The simulation was optimistic: by about 8 % for S1, 3-8 % for S4 at two threads, 7-10 % at three,
and by a fifth at four threads, where the fourth busy virtual CPU makes things worse rather than
better. What the profiles show, and what they do not:

* **Work moved to a second thread does not keep its single-process time.** With the speculative
  step the main thread's fxcm half of Predict takes 16 % longer (43.2 s -> 50.0 s), and the cycle
  is bound by the transformer thread: the main thread spends 117.7 s (197 us per byte) in the
  section that holds the wait for the step, the driver, the fast-weight update and the commit.
  The simulation moved the measured single-process times to other threads unchanged. (There is
  no timer on the transformer thread yet, so how much longer the step itself runs there is
  inferred, not measured.)
* **A second busy thread costs the main thread even when it does no work**: fxcm ran 10 % slower
  on the warmed window (25 % on a cold chunk) with a worker that only spins. The 16 % above is
  inside that band, so it needs no cache explanation.
* **The split regions scale less than the model assumed**: the step got 1.09x faster with a second
  thread (model 1.25x); the main thread's share of the weights is cold after every bit loop.
* **The cause is not pinned down.** Candidates that fit the data: the two halves sharing the 12 MB
  L3; power or frequency limits of the laptop with two and more busy cores; WSL2 placing a virtual
  CPU on an efficiency core (the four-thread result fits the last two, not the first). Parking the
  waiting threads (`tpause`/`umwait` are available in this guest) and a busy/idle timer on the
  transformer thread would separate them; neither is done.
* The single-thread time of this window itself drifted between 236 s and 280 s during the day, so
  each ratio carries several percent of noise.

**What this means for S5 (not built).** The guess from the transformer row reaches 0.91-0.93 of
the perfect guess in the simulation at two threads and 0.89-0.92 at three (the lower figures are
the 250 M window and the reference chunk); applied to the measured S4 figures a real decoder would
run at **about 1.1-1.2x with 2 threads and 1.2-1.25x with 3** on this PC. S5 is a modest step on
top of S4 (the guess, a hook behind each decoded bit to guess again; the undo paths are exercised
by the fault injector, an abort and a new launch inside one byte are not), but S4 is its bound.
A measured lower bound exists too: the speculative step with the transformer's miss rate injected
as wrong guesses that are only found at the byte's last bit runs at 1.19x (2 threads) and 1.27x
(3 threads) with 23 % misses, and at 1.15x and 1.22x with 32 % (three interleaved single-thread
runs within 0.3 %). So S5 would land between 1.15-1.19x and about 1.2-1.3x with 2 threads, and
between 1.22-1.27x and about 1.3-1.4x with 3. Not tried: the main thread helping the transformer
thread while it waits. S1 measures what a cache-cold main thread adds to the step (1.09x), and
with the speculation it is free for about half of the cycle, so the expectation is a few percent
for a dynamic share scheme that the pool does not have.

## 1. What one decoded byte does

`Decompress()` (`src/runner.cpp:899-966`), per stream byte:

| step | where | needs |
|---|---|---|
| `SemByteStart(prev byte)` | `runner.cpp:908`, `predictor.cpp:885-1261` | the row of this byte (`:948-957`), the engine's key vector (`:968`) |
| forced byte / forced bits / 8 x `Decode()` | `runner.cpp:909-937` | |
| `AfterStreamByte(byte)`: stream automata, injected bytes | `runner.cpp:940`, `:565-615` | the byte |

`Decoder::Decode()` (`src/coder/decoder.cpp:27-75`), per bit: `Predict()` (`:37`) ->
`bit_head::Adjust` (`:44`) -> floor stage and `SemCorrect` (`:48-52`) -> the bit (`:53-59`) ->
`floor_.Update`, `SemLearn`, `bit_head::Observe` (`:60-66`) -> `Perceive(bit)` (`:67`).

`Predictor::Predict()` (`src/predictor.cpp:469-678`): fxcm first half (`:473`), selectors from the
transformer row (`:474-476`), cache readout (`:477`), layer-0 row from all model outputs
(`:481-608`; the transformer slot at `:575`, the fast-weight row `:607`, the cache `:608`), the 20
layer-0 mixers and the final mixer (`:619-645`), SSE (`:654`).

`Predictor::Perceive(bit)` (`:1695-1842`): cmix models (`:1699-1719`), mixer and SSE update
(`:1724-1743`), adaptive depth and contexts (`:1747-1758`); **on the 8th bit**: byte updates
(`:1759-1779`), PPMd (`:1783`), the match channel (`:1798-1806`), **the transformer driver step**
(`:1807-1809`), slots (`:1813-1829`); then the slot outputs of the next bit (`:1831-1833`) and
`fxcm_model_.Perceive(bit)` (`:1837`).

The driver step `Tf1Engine::Step` (`src/tf1/tf1_engine.cpp:468-775`): PPMd row to float16
(`:526-532`), fast-weight update with this token (`:540-542`), stream automata (`:546-617`),
then rows through `Feed()` (`:320-356`): normally one read stream row (`:680-689`), sometimes a
pre block (`:643-671`), a post block (`:692-716`), a link-context prefix (`:731-743`) or a page
pre-block (`:749-769`); the output row to float16 and back (`:744-746`, `:770`). One network row
is `TransformerOptImpl::step` (`src/tf1/opt/model_opt.cpp:1485-1536`).

Time per stream byte (*profile*; 350 us in total):

| part | share | us |
|---|---|---|
| transformer driver and network (T) | 59-67 % | 220 |
| fxcm first half of Predict, selectors, cache readout | 16-17 % | 58 |
| mixers and SSE (predict) | 4.3 % | 15 |
| semantic cache, byte start | 3-3.5 % | 11.5 |
| fxcm Perceive | 2.3 % | 8 |
| PPMd byte update | 1.7 % | 6 |
| BITLSTM | 1.3 % | 4.5 |
| the rest (coder, automata, layer-0 row, cmix models, mixer update, contexts, slots, end stages) | 7-8 % | 27 |

So R = 130 us, of which fxcm is at most 66 (the 58 include the selectors and the cache readout,
which read the row; the measurement separates them). Inside T (*profile*, shares of T): MLP up +
relu^2 code 28.5 %, q/k/v projections 12.4 %, gates 4.6 %, out projections 4.3 %, window attention
18.8 %, KDA step 13.5 %, MLP down 6.8 %, the rest (norms, quantisers, depth mix, embedding, heads,
fast weights, driver) about 11 %.

## 2. The dependency graph of one byte

```
 byte n known ........................................................ decoder.cpp:53-59
   | Perceive(bit 7): models, mixers, contexts            predictor.cpp:1699-1758
   | byte updates of the cmix and match models                        :1759-1779
   | PPMd(n)  --------------------------> prior row                   :1783        6 us
   | match channel (longest of 5 models)                              :1798-1806
   | STEP(n): fast-weight update, automata, network                   :1807      220 us
   |      inputs : token n, prior(n), match(n), prev1/prev2, engine state(n-1)
   |      outputs: row(n+1), fast-weight row, hidden vector, cache key
   | slots, slot outputs of bit 0, path state                         :1813-1833
   | fxcm Perceive(bit 7)                                             :1837
   | AfterStreamByte(n)                                   runner.cpp:940
   | SemByteStart(n+1): entry of n; lookup with key and row           :908        11.5 us
   +- bit 0 of n+1:  fxcm first half (byte parse, ~140 context hashes) ---+
   |                 layer-0 row <- row(n+1), fw row, cache                |  about 14 us
   |                 mixers <- selectors from row(n+1)                     |  per bit,
   |                 BITLSTM <- hidden vector (read live)                  |  bit 0 more
   |                 end stages <- cache, slots; the bit                   |
   |                 learn: end stages, models, mixers, SSE, fxcm       ---+
   +- bits 1..7 likewise
 byte n+1 known
```

Edges that matter:

* `row(n+1)` needs byte n, and through the prior input the PPMd update with byte n
  (`tf1_engine.cpp:526-532`; the prior enters the embedding, `model_opt.cpp:1028-1045`, and the
  head, `:1531`), and through the token row the match channel after byte n
  (`tf1_engine.cpp:344-348`, `model_opt.cpp:1427-1476`).
* Every bit of byte n+1 needs `row(n+1)`: as a layer-0 input (`predictor.cpp:575`, `:607`), as
  mixer selectors (`:474`, `:476`, `:1833`), through the cache (`:948-968`), through BITLSTM, which
  reads the engine's hidden vector through a pointer set once (`predictor.cpp:192`,
  `model_opt.cpp:1588`), and in the two end stages (`predictor.cpp:1348-1349`, `:1595`, `:1599`).
* fxcm needs nothing of the transformer or PPMd: no reference to either in `models/fxcmv1.cpp`;
  its outside inputs are the bit, `pagectx::surprise.cxt3` (final at `predictor.cpp:1700` of the
  8th bit, `page_contexts.h:52-80`), the rule automata of `invariants.h` (written at
  `predictor.cpp:1757-1763`), `g_injected_byte`, and the dictionary reload between two bytes.
* So nothing of byte n+1's bit loop can run before `row(n+1)` except fxcm and the model
  look-ups, and the step itself cannot start before byte n. That is the whole problem.

## 3. The mechanisms

### 3.1 A: transformer-independent byte-start work under the step

| work | where | needs the row? |
|---|---|---|
| fxcm `update()` of bit 7 | `fxcmv1.cpp:5947-6010` | no |
| fxcm byte parse, ~140 context hashes, bit-0 bucket look-ups, sentence scans: all in the **first half of Predict of bit 0** | `fxcmv1.cpp:5202-5668`, `:4515-5183` | no |
| byte updates of the cmix and match models, PPMd | `predictor.cpp:1759-1783` | no, but they run *before* the step: PPMd's row is its input |
| layer-0 row of bit 0 except three inputs | `predictor.cpp:481-567` | no (`:575`, `:607`, `:608` do) |
| cache: entry of the last byte, history, followers | `predictor.cpp:894-950` | no |
| cache: key, hashes, lookup, readout | `predictor.cpp:954-1261` | yes: row (`:954`) and key (`:968`) |
| path state, selectors, top-1 stage | `:325-341`, `:371-422`, `:1536-1581` | yes |
| stream automata of `AfterStreamByte` | `runner.cpp:565-615` | no, but they may inject bytes that step the transformer |

Only fxcm's bit-0 half and part of the layer-0 row can move under the step: about 15-20 us of 350
(measured: `sim.A`), **a bound of about 1.05x** with two threads. It needs `Perceive()` cut in two
around the step and has an ordering trap: the byte parse reads `g_injected_byte` for the byte that
*follows* (`invariants.h:330` through `fxcmv1.cpp:4316`), known only behind `AfterStreamByte`.
**No stage of its own:** D gives the same overlap and far more without touching `Perceive()`.

### 3.2 B: deferred learning

| learner | written at bit k | read again | condition to defer |
|---|---|---|---|
| layer-0 mixers 2, 3, 4, 5, 7, 8, 10, 11, 13, 14, 17, 18 (context constant over the byte, `context-manager.cpp:116-180`, `:249-255`) | their one row | bit k+1 | complete before `Prime()` of the next bit (`predictor.cpp:619`) |
| layer-0 mixers 0, 1, 9, 12, 15, 16, 19 (context has the bit position) | row of bit k | another row at k+1 | deferrable in principle, but the update reads the input row by reference (563 floats, overwritten by the next Predict) and the column scaling `mxp_l0col_` is one ordered chain over all updates (`mixer_precond.cpp:15-40`) |
| final mixer rows, Bayes term `g_mxs_bay` | `predictor.cpp:457-468`, `final_mix.cpp:27-38` | bit k+1 (`final_mix.cpp:12-25`) | before the next Predict |
| adaptive depth, top-1 stage weights | one row each | bit k+1 | before the next Predict |
| SSE, floor stage, end-stage rows | cells keyed by the bit position | a later byte | deferrable, small |
| BITLSTM `Observe` | frozen weights: only counters (`bit_head.cpp:1084-1113`) | | nothing to defer |
| fxcm | its updates of bit k run lazily inside the predict of bit k+1 (`fxcmv1.cpp:1765-1769`, `:6054`) | | serial inside fxcm |
| fast weights | pending per column until the column is read (`model_opt.cpp:1186-1237`) | | already deferred by the entry |
| cache entry of byte n | `predictor.cpp:894-908` | lookups at least `kGap` = 1024 bytes later (`semantic_cache.h:73`) | deferrable, about 1 us |

The deferrable part is a few percent of R in pieces under a microsecond, against a hand-off that
costs as much as a piece. **B is dropped.**

### 3.3 C: threads inside the transformer step

One read row (`model_opt.cpp:1485-1536`), 16 layers, 12 of the KDA kind and 4 window layers:

```
 embed: prior matmul, token row with channels, 33 depth-mix dots          :1015-1051
 for l = 0..15:                                                           :1059-1099
   depth-mix read                                                         :1065-1067
   KDA layer (:807-870)                     | window layer (:875-971)
     norm + 5 quantised copies              |   norm + 3 quantised copies
     [ q | k | v | forget gate | out gate | beta ]                     [ q | k | v ]
     KDA step: conv, gates (serial)         |   head prep, K/V insert
       [ head 0 | head 1 | head 2 ] sweeps  |   [ K scan ] -> max -> [ exp | quantise | V sweep ]
     gated norm, quantise, out projection   |   quantise, out projection
   depth-mix take, read                                                   :1082-1083
   MLP: norm, [ up + relu^2 code + down columns ]                         :974-1011
   depth-mix take                                                         :1096
 final read, norm, [ unembed + head | fast-weight head ]                  :1505-1534
```

The layers are strictly serial, and inside a layer mixer -> MLP is serial. All parallelism is
inside the bracketed regions. Each can be split with **bit-identical** results:

| region | kernel | why a split is exact |
|---|---|---|
| dense matmuls (q/k/v, gates, MLP up, out projections, unembed) | `run_g8`, `qmat_dense.cpp:93-152` | every group of 8 output rows is computed alone: integer sums, then one multiply per row |
| relu^2 code | `qgemv_relu2q`, `:464-522` | phase 2a is element-wise; the index list (2b) is a serial scan: build per thread, concatenate in row order |
| MLP down | `sparse4_accum`, `qmat_sparse.cpp:261-308`; `epi4_add`, `:364-373` | int32 sums over the columns of the non-zero codes (16-bit partial sums are flushed before they can overflow, `:192-197`), one float multiply-add at the end. Any partition of the column list adds exactly, and the dense and the sparse form give the same sums. So **each thread adds the down columns of the codes of its own rows**; the coordinator adds the partial vectors and does the float step once. MLP up and down are one region with one fork. |
| window attention | `attn_impl_shkv`, `attn.cpp:650-675` | the K scan is exact int32 dots (`qk_scan3`, `:533-589`); the exp pass is element-wise and its float sum is discarded on this path (`:669`); the weights are quantised to 13 bits element-wise and summed as integers (`:415-436`); the V sweep is int32 sums, unchunked for every length (`:594-648`; `EQ_CHUNK` = 1024 = the pair count, `:409-411`, `:596`). The one ordered float step is the running max (`:572-578`): keep the per-block maxima and fold them in block order on the coordinator. Hence **two forks**: scan, fold, then exp + quantise + V sweep by position chunks. The score, exp and weight buffers are file statics indexed by position (`:501-503`): threads write disjoint ranges. |
| KDA sweeps | `kda_sweep_head`, `kda.cpp:268-280`, `:426-430` | the three heads have separate states `S[h]`; inside a head, pass 1 sums over rows per column vector and pass 2 handles two column halves (`:81-112`, `:151-191`) |
| q, k, v / the two gates / beta | `model_opt.cpp:823-847` | separate outputs from the same input codes. The two gates share the scratch code `q64b` (`:829-841`): the second branch needs a buffer of its own. |
| heads | `model_opt.cpp:1527-1533`, `:1241-1266` | the static head and the fast-weight head write separate buffers |

Not parallel: the depth mix (`ArRow`, `:265-368`), norms and quantisers, the KDA front end
(`kda.cpp:308-421`), embedding, out projections (0.6 us each: a fork costs what it saves), the
driver. Serial floor about 52 us of 220 (*profile shares*; the per-site table of an FX2_PROF build
gives it exactly), plus 52 forks.

Critical path of the step, *model* (`neif_par_bounds.py`, `tf_factor`): 52 forks per step, each
costing one hand-off h; parallel parts divided by the summed speed of the threads:

| threads | step, h = 0.2 us | h = 0.4 us | h = 0.8 us | h = 0.4, matmuls and scan 1.5x faster in L2 |
|---|---|---|---|---|
| 1 | 220 | 220 | 220 | 220 |
| 2 fast | 146 | 157 (1.40x) | 177 | 136 |
| 3 fast | 118 | 129 (1.71x) | 149 | 115 |
| 4 fast | 104 | 115 (1.91x) | 135 | 104 |
| 4 fast + 4 slow (0.5) | 91 | 101 (2.18x) | 122 | 94 |
| 16 equal | 77 | 87 (2.53x) | 108 | 84 |

Per layer at 4 threads and h = 0.4 (*model*): a KDA layer 11.4 -> about 6.2 us (projections and
gates 2.5 -> 1.0, KDA 2.5 -> 1.8, MLP up + down 4.8 -> 1.8, serial 1.6), a window layer 18.3 ->
about 8 us (projections 1.7 -> 0.8, attention 10.3 -> 3.8, MLP 1.8, serial 1.6).

Memory: one step touches about 6 MB of dense weights (MLP up alone 196 KB per layer), up to 2 MB
of MLP-down columns, 0.26 MB of K and V per window layer scanned completely (`attn.h:101-105`),
0.58 MB of KDA state (`kda.h:48-53`). On one fast core that exceeds its L2 (2.5 MB), so most of it
streams from L3 every step at the measured 45-55 GB/s; MLP up runs at about that rate today
(196 KB in 3.9 us). With a **fixed** partition (thread w always gets the same row groups, position
chunks and KDA heads) each fast thread's share fits its own L2 from 3-4 threads on, and L2 is
2-2.5x faster than L3: the last column, an assumption that may make the split better than linear.
On a 1.25 MB L2 (contest machine class) or 0.5 MB (Zen 3) the shares do not fit and the threads
stream from the shared L3 side by side. The four slow cores of this PC share one L2 among them
and, to the reviewer's knowledge, do not reach the L3 at all; neither is checked on the machine,
which is why the 8-thread rows are not predictions.

Gain alone: 1.22x / 1.35x / 1.43x / 1.52x overall with 2 / 3 / 4 / 8 threads
(350 -> 287 / 259 / 245 / 231). About 700 lines. Identity risk **low**: the units are built with
`-ffp-model=precise -ffp-contract=off` and explicit intrinsics (`makefile:59`), nothing changes
MXCSR, the splits are integer or element-wise, and the single-thread path of the same binary calls
the same range functions with the full range. C works unchanged in the compressor, so the existing
`-t` mode verifies it.

### 3.4 D: speculation on the next byte

**Idea.** At the start of the bit loop of byte n+1 the row for it exists, so its most probable
value g is known. A transformer thread computes the step for token g while the main thread decodes
the 8 bits. When the main thread reaches the step, the result is there (hit) or is thrown away.

**Form: a memo of one network step, not a rollback.** The speculation runs
`TransformerOptImpl::step` for inputs it predicts: token, the 208 float16 prior values, the
channels (`prev1`, `prev2`, match byte, match class), `gelesen` = true. The real
`Tf1Engine::Step` runs unchanged on the main thread with all its automata; in `Feed()`
(`tf1_engine.cpp:320-356`) the engine compares the real inputs of the row with the predicted ones
bit for bit, together with an epoch counter of the engine state (no other `step`, `begin`, or
queue run since the speculation started). Equal: the shadow is committed in place of running the
network. Different: the shadow is dropped and the network runs as today. The result is identical
by construction: a network step is a deterministic function of (state, inputs), and the speculation
never writes state that anything reads before the commit. A wrong guess of *any* input (the byte,
the match channel, a prompt row in between) only costs time. (A systematically wrong predicted
input would show as a hit rate of zero, not as an error: the memo counts its hits and reports
them.)

One rule keeps the two threads apart: **every real entry into the engine** (`step`, the queue run,
`begin_article`) and the PPMd `ByteUpdate()` of the main thread **first stops the speculation and
waits for its acknowledgement** (the transformer thread checks a flag between two regions).
Between the launch and that point the main thread reads only committed buffers of the engine
(`last_hidden`, `last_key`, the slot rows).

**What the speculative step must not write, and where it writes instead:**

| state | written by a step | speculative step |
|---|---|---|
| row buffers `x, xn, a, b, c, q5, q768, idx768`, `pre`, `qq/kk/vv`, `yb`, `logits`, `probs208`, `tok_paket`, `sc_l2_stage`, depth-mix state (`ZRow`, `model_opt.cpp:370-388`, `:442-453`) | all | a **second row object** with its own scratch. `xn` is the hidden vector BITLSTM reads live during the bits (`:1588`) and `sc_key` the cache key (`:1517`): both are copied at commit. |
| KDA state `S[3][64*64]` per layer | `pass1_15` rewrites every entry, `kda.cpp:106-110` | **ping-pong**: pass 1 reads the committed state and writes the decayed rows to the shadow state; pass 2 works in the shadow. Same arithmetic, same order. Commit = swap the index. +0.58 MB. |
| KDA conv ring, `pos` | slot `pos & 3` (`:311-336`), `pos++` (`:476`) | the slot written holds x[t-4], which no step reads again (`kda.h:28-29`): write in place; `pos++` at commit |
| K/V ring | slot `t % 2048` (`model_opt.cpp:937-953`, `attn.cpp:773-783`) | that slot holds position t-2048, outside the window from step t on, and every step writes its slot before it scans: write in place; a miss overwrites it |
| position `t` | `t++` (`:1535`) | at commit |
| fast weights: `fw_update` (`:1273-1380`), `fw_apply` in the last four MLPs (`:1202-1237`), `fw_head` (`:1241-1266`) | dW columns, pending steps, statistics | **not speculated.** `fw_update` needs the real token and already runs in the driver before `Feed` (`tf1_engine.cpp:540-542`). `fw_apply` reads only the relu^2 code of its layer (`:1170-1184`, `:1205`) and neither it nor `fw_head` writes `x`: the shadow keeps the four codes, and commit runs `fw_apply` x 4 and `fw_head` in the original order. About 3 us on the commit path. |
| queue of unread rows (`zr`, `zn`, `:1129-1166`) | `zb_run` at the next read row | no speculation while a row waits in the queue (running it early would change the block boundaries; that blocks of different length give the same bits is not assumed) |

**PPMd.** The prior of the step is the PPMd distribution *after* the guessed byte, so the
speculation needs `ppmd_UpdateByte(g)` and `ppmd_PrepareByte()` (`models/ppmd.cpp:1594-1639`,
`:1556-1592`), with an undo journal:

* all heap writes go through `hzg::w()`, `w(i)`, `xl_w` and `kopiere`
  (`models/ppmd_page_ptr.h:39-42`, `ppmd_page.h:147-151`, `:180`, `ppmd_page.cpp:682-693`): log
  (canonical address, old bytes) there; undo in reverse order through the same write path;
* the rest of the state is the model object, about 17 KiB of plain fields: copy it before,
  restore it on a miss;
* the per-bit calls of the main thread read only the wrapper's bit-tree arrays
  (`ppmd.cpp:1693-1706`), so the model itself is free during the 8 bits; the speculation writes its
  result (`sqp`, `trF`, `trT`) to private arrays and runs the wrapper's post-processing
  (`:1714-1739`) through the same compiled function, so that the float16 prior is bit-equal;
* on a hit the main thread's `ByteUpdate()` adopts state and arrays: PPMd leaves the main thread;
* **only one thread is ever inside the paging layer.** Even reads change its global state
  (translation caches, page-ins, evictions: `ppmd_page.h:43-49`, `ppmd_page.cpp:9`); that is
  transparent to values but not to two threads. During the bit loop the transformer thread owns
  PPMd; at the byte end it hands it back before `ByteUpdate()`;
* in the speculation `HeapVoll` (`ppmd.cpp:316-319`, an `exit(1)`) and the zone switch
  (`:1563-1567`) must end the speculation instead of acting, and `dyn_tick()` (`:1562`) is skipped
  and run at adoption;
* never across a pending restart latch (`ppmd.cpp:193-209`), the dictionary warm-up
  (`runner.cpp:498`), or `AfterStreamByte` (the census decoder reaches `dyn_tick`). Hence the
  speculation starts at the top of the byte loop.

**Match channel.** `(MatchByte, MatchLength)` after byte g is a pure function of five values per
model at the byte start (`models/match.cpp:27-64`): L' = min(L + 8, 255) if g is the predicted
byte, else the number of matching bits behind the last mismatch; the candidate comes from the map
at the next context hash (`contexts/context-hash.cpp:11`), with two overlays for writes that have
not happened yet (the map insert of the 8th bit, `match.cpp:46-49`, and the history byte g). The
**main thread** takes the snapshot at the byte start and evaluates `Peek(g)` from it, also for a
later guess (the map and the history are not written before the 8th bit; length and bit position
are, hence the snapshot). About 80 lines.

**Injected and forced bytes.** Bytes the decoder knows in advance (`Tf1ZwangByte`,
`runner.cpp:909-913`; `ObserveKnownBytes`, `:230-249`) are speculation with a certain guess. Forced
stream bytes fit the scheme as they are; injected bytes run inside `AfterStreamByte` and stay
serial at first.

**Guessing again.** When a decoded bit rules the guess out, the main thread posts the best byte
the decoded bits allow; the transformer thread stops between two regions, undoes PPMd, and starts
over. The policy only decides time, never bytes, so it may be anything, even adaptive.

**Cost model**, per byte with a plain step: W = time from the start of the bit loop to the step,
p = PPMd, L = time at which the right guess starts (0 on a first-guess hit; else the time its bit
is known plus 4.5 us for stopping and undoing), T' = the step on the transformer threads:

```
 today      W + T + post
 with D     max(W - p, L + p + h + T') + h + commit + post        commit = 3 us, h = 0.45 us
```

With T' = T = 220, W = 115, p = 6, post = 15: a hit costs 245 instead of 350 (1.43x); a miss
replaced at L costs L + 245. A miss costs at most about 15 us over today, so D cannot lose more
than about 4 % at any hit rate; the gain is roughly `(W - 10 - E[L]) / 350`. First-guess rate
60 / 75 / 90 % with a wrong guess replaced after 60 us: 1.29 / 1.34 / 1.39x on two threads.
E[L] is what the measurement delivers (`k_hist`, `bit_known_us`); two letters of the same case
differ only in their low 5 bits, so many wrong guesses are replaced late. (The launch is taken
behind the cache's byte start, where the measurement hook sits; a policy that does not use the
cache's top byte can launch in front of it and gains 11.5 us per hit.)

About 1200 lines (shadow row and commit 400, PPMd journal 300, memo in the driver 150, match peek
80, guess policy and re-guess 250). Identity risk **medium**, in two places: the PPMd journal (a
missed write corrupts the model silently; the decoder then leaves the stream within bytes, which
the chunk decoder shows, and a fault injector forces wrong guesses on purpose) and the commit list
(a forgotten buffer). The hot functions `Predict`, `Perceive`, `Decode` get plain calls of
noinline functions only, as the NEIF_TEST marks do, so their profile entries hold.

### 3.5 E: other

* **F: fxcm on a thread of its own.** fxcm's state is all namespace-global and touched by nothing
  else; per bit it needs the bit and returns about 560 raw predictions. `FXCM::Perceive(bit)`
  posts the bit, the fxcm thread runs `update()` and the next `predict_half()`
  (`fxcmv1.cpp:6230-6238`), `FXCM::PredictHalf()` joins. Per bit the shorter of {fxcm} and {the
  main thread's other work} leaves the path: *model* R = 130 -> about 100 us. Ordering rules (from
  reading): post the bit behind `surprise.AddBit` (`predictor.cpp:1700`); on the 8th bit behind
  `cvz.Perceive`, `isbn2.Byte`, `umr.Byte` (`:1757-1763`); `wrtcxt` (written by fxcm,
  `fxcmv1.cpp:5905`, read by `UpdateContexts`, `context-manager.cpp:278`) through a latch;
  `g_isbn_rule` is read after the join (`predictor.cpp:658`); the byte parse of bit 0 only once it
  is known whether the next model byte is injected; the dictionary reload (`runner.cpp:309-312`)
  with fxcm idle. Alone about 1.1x. **On this PC it gains nothing with D + C** (the step stays
  the critical path: 2.30x against 2.36x at 8 threads); with 16 equal fast cores 2.8x against
  2.5x. It is also the next lever behind the reference-assisted decode (3.6), whose critical path
  is the coder stage. About 300 lines, identity risk medium (five shared globals).
* **BITLSTM on a thread:** no. Frozen weights; `Adjust` is about 11k multiply-adds on the critical
  path of every bit, `Observe` is three counters (`bit_head.cpp:958-1113`).
* **Inputs for both values of the next bit:** no. fxcm applies the updates of bit k inside the
  prediction of bit k+1 and reselects table slots at bits 2 and 5 (`fxcmv1.cpp:1521`, `:1699`).
  The 20 layer-0 dot products are independent (`mixer_bank.h:20-42`) but cost under a microsecond.
* **Speculation two bytes deep.** Step n+2 needs, per layer, only the state step n+1 left in that
  layer, so a second transformer group can follow one layer behind on a guessed byte n+2. On a
  streak of hits the decoder is limited by the main thread alone. Needs versions of the KDA state,
  a saved K/V slot and a second journal level; only for many fast cores, only after D's hit rates
  are known.

### 3.6 The reference-assisted verifying decode (no guessing)

A validation decode checks that the archive reproduces a stream that is on disk (the coder stream
of the compression run, the file `-t` reads). Then the "guess" is certain and can run arbitrarily
far ahead, which is exactly the compression pipeline (`src/neif_test.h:136-163`,
`runner.cpp:664-711`): front stages walk the *expected* stream and compute PPMd, the transformer
rows, hidden vector, key and fast-weight row; the coder stage runs the `Decompress` byte loop on
the archive, takes those results from the queue instead of computing them, and **compares every
decoded byte with the expected one before it goes on**.

* Correctness: a front-stage result for byte n+1 is consumed only after bytes 0..n were decoded
  and found equal to the expected ones, so it is the value the plain decoder would have computed
  (the pipeline's stages are already shown to compute what one process computes: its compression
  output is byte-identical). By induction the run reaches the end exactly when the plain decoder
  reproduces the stream; at the first differing byte it stops and reports the position, which is
  the validation result. It is not a decompressor: it cannot produce an unknown file.
* Bound: the compression pipeline's, about 2.4-2.5x on three fast cores (the coder stage is about
  40 % of the single-process time), at the pipeline's memory (about 9.3 GB in three stages).
  Beyond that: the fxcm thread in the coder stage, and C inside the two transformer stages.
* Cost: the chunk decoder of stage S3 below plus the coder-stage hooks in the decoder path, about
  250 lines; no journal, no shadow state, no kernel changes, low identity risk.
* Open: it validates what the test flavour decodes; whether that flavour decodes an *official*
  archive is item 11 of section 9, with or without threads.

## 4. Upper bounds for the decoder without a reference

*Profile* inputs T = 220, W = 115 (of it PPMd 6), post = 15, R = 130; T' from 3.3 at h = 0.4 us;
commit 3, stop-and-undo 4.5. Upper bounds: cache lines moving between cores are counted only as
the hand-offs.

```
 C alone, n threads:      T'(n) + R     2: 287 (1.22x)  3: 259 (1.35x)  4: 245 (1.43x)  8: 231 (1.52x)
 A alone, 2 threads:      350 - (15..20)     about 1.05x
 F alone, 2 threads:      350 - (25..34)     about 1.08-1.11x  (less: part of the 58 us is not fxcm)
 D alone, 2 threads:      hit 245, miss 309 (wrong guess replaced at 60 us)
        first-guess rate 60 / 75 / 90 %:  271 / 261 / 251  (1.29 / 1.34 / 1.39x); perfect 1.43x
 D + C, n threads = main + (n-1) transformer threads:
        cycle = max(109, L + 6.5 + T'(n-1)) + 18.5
        n = 3 (2 fast, 157):          hit 182, miss 246 -> 75 %: 198 (1.77x)   perfect 1.92x
        n = 4 (3 fast, 129):          hit 154, miss 218 -> 75 %: 170 (2.06x)   perfect 2.27x
        n = 8 (3 fast + 4 slow, 107): hit 132, miss 196 -> 75 %: 148 (2.36x)   perfect 2.65x
                                                           60 %: 158 (2.22x)   90 %: 138 (2.53x)
 D + C + F, n = main + fxcm + (n-2), W -> 85, L -> 44:
        n = 8 (2 fast + 4 slow, 115): hit 140, miss 188 -> 152 (2.30x): below D + C
        16 equal fast cores (88):     hit 113, miss 161 -> 125 (2.80x); D + C there: 140 (2.50x)
 two guesses on two transformer groups: fewer misses, half the threads per group; computed per
        window by the tool, expected to lose on 8 cores
 pessimistic corner, 4 threads: h = 0.8 (T' = 149), 60 %: 200 (1.75x); with 15 % clock loss 1.5x
```

## 5. Build order

"Identical" means: the coder output of a `-t` chunk run has the same sha256 as the single-thread
run of the *same binary* (`NEIF_MT=0`) on the three windows; from S3 on, a chunk decode reproduces
the stream.

| stage | content | verification | decides / useful alone |
|---|---|---|---|
| **S0a** (done 17 Nov) | the NEIF_PAR measurement: one run per window | `kept_bits` and the reference chunk's sha equal to the plain build: yes | section 0a: D + C is 1.5-1.6x, not 1.8x |
| **S0b** (done) | a threaded kernel bench, `neif_par_bench.cpp`: spin pool, the dense kernel split 2/3/4/8 ways, the q/k/v region, the clock under load | same bytes out as the single-thread kernel: yes | **C: go at 2-3 threads for the large regions, marginal at 4, no-go at 8 and for the small regions** |
| **S0c** (done for the metadata part) | the test flavour decodes the 2 MiB prefix of the official coder output | 10,447,599 decoded bytes identical to the stream, up to the end of the input | the flavour decodes official bytes; page text (from 13.0 MB of stream) still to be covered |
| **S3** | chunk decoder `cmix -u STREAM OUT.cmix`: structure pass over the stream as in `-t`, then `Decompress` of a `-t` output, every byte compared with the stream | decodes the three windows written by `-t` | test bed for everything below; decode timing in minutes |
| **SV** | *if a reference-assisted decode is acceptable:* S3 as the coder stage of `NEIF_PIPE` (section 3.6) | the three windows; a deliberately wrong reference byte is reported at its position | **about 2.4x for validation, the cheapest route**; then F in the coder stage |
| **S1** | thread pool (section 6); range forms of `run_g8`/relu^2 and the fused MLP region; chunk form of the window attention | identical `-t` output at 2, 3, 4, 8 threads | *model* 1.27x at 4 threads (MLP and attention are 54 % of T: 220 -> 145 us) |
| **S4** | D for certain bytes only (in `-t` every byte is certain): shadow row, KDA ping-pong, commit, PPMd journal with adoption, match peek, the memo in `Feed` | identical `-t` output with speculation on every plain step; then with a fault injector (`NEIF_MT_FAULT=p`: a wrong byte with probability p) through every undo path | in the decoder only forced bytes gain |
| **S5** | D with guesses in the decoder: policy, re-guess, stop | `-u` reproduces the stream with and without faults; hit counts against S0a | the main gain: *model* 1.8x at 4 threads with S1 |
| **S2** | the rest of C (q/k/v + gates region, KDA heads, the two heads), only if S0b gave h <= 0.4 us | identical `-t` output | *model* 1.8x -> 2.06x at 4 threads |
| later | F (fxcm thread), two guesses, two bytes deep | | many fast cores only; not for this PC |

**After S0 (17 Nov):** the next stage is S3 (with a decode time against `-t` as its first
output), followed by SV if the reference-assisted decode is acceptable. S1, S4 and S5 are worth
building only for a decoder without the reference, and then for 1.35-1.5x at three threads; S2
waits for an in-situ bench; the "later" row is dropped for this PC. Nothing behind S0 has been
started.

Why this order (as planned on 16 Nov): S0a-c cost an hour together and can each stop or redirect
the rest. S1 before D
because it is the low-risk part, needs no decoder test bed and D's gain grows with every
transformer thread; D before the rest of C because by the model D adds more (1.27x -> 1.8x) than
S2 does (1.8x -> 2.06x), and S2's small regions are the ones a slow hand-off eats.

## 6. Threading substrate

* **pthreads, created once** at the first model byte: `pthread_create` with affinity per thread
  (`sched_setaffinity`). The entry is a normal dynamic glibc program (no `-static`, no
  `-nostdlib`; link line `makefile:45`), glibc >= 2.34 has pthreads inside libc (the entry's own
  floor, `src/glibc_compat.cpp:1-2`), so no new library is linked. No `std::thread`, no
  exceptions (`-fno-exceptions`).
* **Two-stage packed start:** not involved; the test flavour is the unpacked `cmix` of
  `make prof_use`. Were it packed: stage 2 ends in `execveat` of the image from a memfd
  (`tools_src/packer/stub.c:170-185`), after which the program starts through the normal loader
  with TLS, so threads would work.
* **Hand-off: spin, never sleep.** A step has 52 forks of 0.3-4 us; a futex wake costs 5-50 us.
  Workers spin on one cache-line-aligned atomic sequence word each (`std::atomic`, as the
  pipeline's `Ring`, `src/neif_test.h:171-178`); the coordinator takes the first share and spins on
  a done counter. A worker without work for about 1 ms sleeps on a futex, so the structure-only
  pass does not burn the cores. Whether `_mm_pause` in the spin is harmless under Hyper-V (it can
  be intercepted) is for S0b.
* **Fixed partition, fixed cores.** Share w of every region always goes to worker w, pinned, so
  its weights, K/V chunks and KDA heads stay in its L2. `NEIF_MT_CPUS=list` names the cores,
  fastest first; default the fast cores only (the list of `neif_a16_cores.sh`), leaving the slow
  ones and so the Windows host alone.
* **Shares by speed** when unequal cores are used: measured once at start with one region.
* **No shared mutable statics on worker paths.** The units are built with
  `-fno-threadsafe-statics` (`makefile:20-66`); workers run only the range kernels, which have no
  function-local statics. The attention buffers are shared arrays written in disjoint position
  ranges with chunk borders at multiples of 32 positions.
* **Threads:** main (coder, mixers, cmix models), T0 (transformer coordinator, PPMd speculation),
  T1..Tk (transformer workers). With C alone the main thread is T0.

## 7. Switching it on; the binary

* Compile time: macro `NEIF_MT` (requires `NEIF_TEST`), passed like the other test macros:
  `make prof_use CC="clang++-18 -DNEIF_TEST=1 -DNEIF_MT=1"` (`neif_build_test.sh`).
  A build without it has none of the code: the **official `comp9` stays byte-identical**, to be
  confirmed as for NEIF_TEST with the full official build of the edited tree
  (`neif_build_official_dev.sh`, sha256 976bf2d3...6767f82).
* Run time: `NEIF_MT=n` threads (default 0: no thread is created and the same binary takes the
  single-thread path through the same kernel code), `NEIF_MT_CPUS`, `NEIF_MT_SPEC`,
  `NEIF_MT_FAULT=p`.
* Size: nothing in the official decompressor. If it were ever shipped it would count twice (the
  code is in `comp9` and in the decompressor part of `archive9`), and the rules ask for one core
  (`README.md:35-36`; "`grep Threads` shows 1", `:286-287`), so it is a tool for our runs only.
* **Which archives it can decode.** Identity "same build, same machine" holds between the
  flavour's compressor and its decoder. Decoding an archive written by the official `comp9` needs
  more: the flavour's arithmetic must equal comp9's. The NEIF_TEST build reaches that for the
  compressor with the shipped profile and unchanged control flow in the profiled functions
  (`src/neif_test.h:57-62`); the predictor group is built with `-ffp-model=fast` (`makefile:37`),
  so a hot function that loses its profile entry may round differently. NEIF_MT keeps the rule,
  but it must be tested (S0c for the test flavour as it is, again for NEIF_MT at 0 threads). If it
  fails, the flavour validates the pipeline end to end on its own archive, not the official bytes.

## 8. Identity rules

1. One compiled body per arithmetic kernel, used by both modes; the single-thread mode passes the
   full range.
2. Splits only where the sum is integer or the step is element-wise; every ordered float
   reduction stays on one thread in its original order.
3. Speculation never writes state that is read before commit; a speculative result is used only
   when all its inputs match bit for bit and the engine epoch is unchanged.
4. Policies and schedules may depend on timing; values may not.
5. Tests: `-t` output sha against `NEIF_MT=0` on the three windows at every thread count; `-u`
   decode of those outputs; fault injection for every undo path; a long run (16 MB) across a
   dictionary reload, a PPMd restart and the metadata block before a full validation is trusted.

## 9. What reading could not determine

(Items 1-8 and the metadata part of 11 are answered in section 0a. Open: 9, 10, the page-text part
of 11, the slow cores' own speed, and how often Windows moves a busy virtual CPU.)

Answered by S0a (`neif_par_run.sh` -> `summary.json`, `summary.txt`;
the field in brackets):

1. **First-guess hit rate and the time to the right guess** (D's whole gain), by policy and region
   [`hit`, `hit_top2`, `right_after_k_bits_on_plain_steps`, `regions`, `bit_known_us`], and whether
   a cheap combination beats the transformer row alone [`tfxfw_top1`, `selector_top1`,
   `any_of_first_six`; `final_greedy_path` as the reference of the full model].
2. **How many steps are plain** (one read stream row, no row waiting), i.e. the share of bytes D
   can speculate [`cycles.speculation_possible`, `steps`], and the transformer time outside them
   [`transformer_outside_step_us_per_byte`, `injected_bytes_us_per_byte`].
3. **The split of R** per bit position, fxcm alone against the rest, the true W, post and PPMd
   [`slot_us`, `cycles.fxcm_us_by_bit`, `main_us_by_bit`, `bit_loop_us_mean_on_plain`].
4. **The serial floor of the step** from the per-site table of the FX2_PROF build
   [`sites_us_per_byte`, `C_transformer_factor`], instead of my shares.
5. The simulated bounds per mechanism and thread count with the measured times of every byte
   [`bounds_by_threads`], replacing section 4.

Answered by S0b (real threads on kernels):

6. **The hand-off cost h** between pinned spinning threads, on this CPU and under WSL2 (a virtual
   CPU can be descheduled by the host); the tables move by about 9 % per 0.2 us.
7. **The clock with all cores loaded**, and the slow cores: their real speed on these kernels
   (0.5 assumed, from cmix as a whole) and whether their shared L2 without L3 makes them useless
   for streaming weights.
8. **L2 residency of the partitioned weights** (`--l2-bonus`), and the cost of the KDA ping-pong
   (a second 0.58 MB of state).

Answered only by building:

9. PPMd journal size and undo time in practice (estimate: tens of writes; rescales rare).
10. Whether the scan chunks should follow the ring layout or the valid prefix early in an
    article: balance only.
11. Whether the test flavour decodes an official archive (S0c).

## 10. The instrumentation (deliverable 2)

`neif_par.diff` (against the tree of 16 Nov; changed copies in `par/`, originals in `par_base/`,
their sha256 in `base.sha256`). Six files, 761 added lines and one changed, everything behind
`NEIF_PAR` inside `NEIF_TEST`; without `-DNEIF_PAR` every change is preprocessed away.

* `src/neif_test.h`, namespace `par`. Every profile mark `T(k)` also books its ticks by bit
  position (slots 0-7; 8 = between two model bytes) and kind of model byte; `Row()` counts the
  rows a driver step feeds; `CostBit()` follows the coded bits of a byte; `ByteStart()` takes the
  guesses and closes the previous byte; `EndCycle()` simulates the byte's cycle for A, F, D and
  D + F at nine speed factors of the step and six guess policies; `Json()` writes the result.
  Two new profile sections: 16 (BITLSTM behind the coded bit) and 17 (selectors and cache readout,
  split off section 1, which is then fxcm alone).
* `src/predictor.cpp`, `.h`: `NeifParByteStart(c)` hands the three rows (transformer slot,
  fast-weight slot, PPMd), the longest match, the cache's top byte and a forced byte to
  `par::ByteStart`, reads only; mark 17 behind `PredictHalf()`; marks around the step of an alias
  row.
* `src/runner.cpp`: the hook behind `SemByteStart` in `Compress()`; environment `NEIF_PAR`,
  `NEIF_PAR_JSON`, `NEIF_PAR_H` (hand-off, default 1500 ticks), `NEIF_PAR_COMMIT` (10000),
  `NEIF_PAR_ABORT` (15000); the profile table scaled to the kept bytes.
* `src/coder/encoder.cpp`: mark 16 in place of the second mark 11. `src/tf1/tf1_engine.cpp`: one
  call in `Feed()`.

Guess policies: the transformer row's best byte; the best of row x fast-weight row; a selector
over six sources (row top-1 and top-2, fast-weight top-1, PPMd top-1, match byte, cache top-1) with
running hit rates per context; two-guess forms of the first and third; an oracle. For each the
measurement simulates the real procedure (keep the guess until a decoded bit rules it out, then
take the best byte the bits allow) and records K, the number of decoded bits at the right guess.

Limits of the measurement: it covers the kept bytes only (the profile restarts at
`NEIF_T_START`, so with `NEIF_PAR=1` the profile tables are tables of the kept bytes). Its hook
costs about 3-4 us per stream byte and is kept out of all sections; the about 140 marks per byte
(roughly 2 us) are inside them. The simulation of D uses constants for commit and stop-and-undo,
launches behind the cache's byte start, treats injected bytes as serial, and scales every
transformer step by one factor; `neif_par_bounds.py` takes that factor from the per-site table
and the assumed core speeds and hand-off. No value of the coder changes (to be confirmed by
`kept_bits`).

Run, when the slot is free:

```
 bash neif_par_build.sh                  # PGO test build + NEIF_PAR + FX2_PROF -> /work/tbpar
 bash neif_par_run.sh /work/tbpar 2      # windows 60, 250, 420 M: 1 MB each after 1 MB warm-up
```

Checked: the changed units pass `clang++-18 -fsyntax-only` with the entry's language flags
(one low-priority core, a few seconds); the helpers (`Best`, `Diff`, `SimK1`, `SimK2`, `HBin`)
pass a unit test against a brute-force reference; a fake call sequence of 3000 bytes runs through
`T`, `Row`, `CostBit`, `ByteStart`, `Json` with consistent sums and valid JSON;
`neif_par_bounds.py` reads it. Since 17 Nov the changes are applied in the development tree itself
(the source tree; `neif_par.diff` and `par_base/` remain as the record of what was added),
built (`/work/tbpar` with the per-site table, `/work/tbpar0` without) and run: identical coder
output, +0.5 % time, no new profile mismatch (section 0a).

## 11. Not verified

* No multi-thread decoder code exists. The speed-ups of section 0a are a per-byte simulation on
  measured single-thread times plus a kernel bench; the simulation uses constants for the commit
  (3 us) and for giving up a guess (4.5 us), and a model of which parts of the step split.
* The bench ran inside WSL2, where the host places the virtual CPUs; three bench runs differ by
  up to 30 % for the same configuration, and the 4-thread results are bimodal.
* The reference-assisted figure (2.15-2.2x) is the *compression* pipeline's, from runs of another
  day with another build of the same source; a decoding coder stage has not been built or timed.
* S0c covers the metadata blocks only (10.4 MB of stream), not page text.
* The line references for fxcm, PPMd, the match models, the mixers, BITLSTM and the build files
  come from reading passes by sub-agents; I checked a sample against the files
  (`ppmd.cpp:1693-1712`, `:1556-1566`, `fxcmv1.cpp:5184-5190`, `:6012-6017`, `:6226-6238`,
  `match.cpp:27-64`, `README.md:35-36`, `:286-287`, `makefile:20-66`) and read the driver, the
  engine, the dense, sparse, attention and KDA kernels, the predictor and the runner myself. The
  reviewer re-checked the kernel exactness claims and the arithmetic; it did not read the fxcm
  references or the context-hash peek, and sampled the PPMd write paths only.
* That blocks of unread rows give the same bits for any block length is not assumed and not
  tested.
* The PPMd journal is designed from the write primitives; that no path writes the heap around
  them rests on the pointer type (reads are `const`), not on a test.
* The cache layout of the slow cores and their speed on these kernels.

## 12. What the review changed

The recommendation-reviewer confirmed the exactness of the kernel splits, the KDA ping-pong, the
deferral of the fast weights and the arithmetic, and found these, all taken over:

* a validation decode already knows every byte: the reference-assisted decode (section 3.6) is
  the cheapest route and is now the first question and stage SV;
* the ranges were too tight at the top: restated as about 1.8x at 4 threads, 8 threads not
  predicted, range 1.3x-2.2x;
* the base profile is not confirmed by the raw table that exists (caveat at the top);
* the window attention needs two forks (52 per step); the two gates share a scratch buffer;
* PPMd speculation: one thread inside the paging layer, the heap-full exit and the zone switch
  must end the speculation; the match peek belongs on the main thread, from a byte-start snapshot;
* the fxcm thread gains nothing on this PC and left the order; the rest of C moved behind D and
  behind a hand-off measurement; a threaded kernel bench (S0b) and the official-archive check
  (S0c) come first;
* instrumentation: section 1 mixed fxcm with row-dependent work (new mark 17); the printed profile
  table was scaled to the wrong time; the simulation had no cost for giving up a guess.
