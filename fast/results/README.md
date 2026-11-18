# Result files

Raw result tables behind the numbers in the top-level `README.md`. Coded outputs (5-7 MB per
split) and build products are not in the repository. Host names of the rented machines are
replaced by `hostA` / `hostB`; nothing else was changed in the tables. Times are UTC.

Machines:

* **laptop**: Intel Core Ultra 288V (Lunar Lake, 4 P-cores + 4 E-cores, no SMT), Ubuntu 24.04
  userland under WSL2, runs pinned to one virtual CPU unless stated. One large run at a time.
* **hostA, hostB**: two rented cloud CPU instances, each a container with 32 logical CPUs
  (16 cores / 32 threads) of an AMD EPYC 9654 (Zen 4) host that is shared with other tenants
  (`epyc_scale/lscpu.txt`), Ubuntu 24.04, memory limit 256 GB. hostA's host was the busier one
  during the two-machine run (load average about 100 at the start against 21-50 on hostB:
  `k16_two_machines/*/driver.log`).

## Files

| file | what |
|---|---|
| `k16_per_split.tsv` | all full split evaluations in one table: per run and split the range, kept bits, compress wall / cpu seconds, peak memory, size and sha256 of the coded output (where we hold the file), verification decode wall / cpu seconds and verdict |
| `k16_one_machine/` | one instance (hostA), 16 splits at once x 3 processes, unpinned. `results.tsv` (one line per run of `neif_split_eval.sh`), `total.json` and `summary.txt` (its summary), `driver.log` (start / end of each pass), `plan.tsv` (the split ranges), `monitor.tsv` (every 10 s: container memory, load, progress) |
| `k16_two_machines/quiet_host_splits_8-15/` | hostB: splits 8-15, 8 at once, each pinned to 4 logical CPUs = 2 cores; compress and verification decode |
| `k16_two_machines/quiet_host_splits_0-7_compress_only/` | hostB afterwards: splits 0-7, same layout, compress pass only |
| `k16_two_machines/busy_host_splits_0-7/` | hostA at the same time as `quiet_host_splits_8-15`: splits 0-7, same layout; compress and verification decode |
| `epyc_scale/summary.txt`, `results.tsv`, `loaded.tsv` | `neif_scale.sh` on hostA: one run at a time on 60.0-60.3 M (every mode), then the loaded-throughput configurations on 60.0-60.1 M. The first run `s00_orig_t` (678.5 s) hit a slow phase of the shared host; its repeats `s13_orig_t2`, `s15_orig_t3` (140.7, 139.4 s) are the usable baseline, so ignore the column "x orig" of `summary.txt` and divide by 139-141 s |
| `epyc_scale/long_window/` | loaded throughput again on the longer window 60.0-60.5 M |
| `epyc_scale/setup_selfcheck.log`, `selfcheck_results.tsv` | `neif_setup.sh` on hostA: compiler versions, binary sha256, the self-check values |
| `epyc_scale/vnni_bench/` | `neif_vnni_bench.cpp` on the EPYC 9654 with (`vb_avx512*.txt`) and without (`vb_plain_quiet.txt`) the AVX-512 VNNI kernels |
| `epyc_scale/lscpu.txt` | `lscpu` inside the container (shows the whole host CPU; the container may use 32 of its logical CPUs) |
| `laptop/fast_results.txt`, `fast_batch_out.txt` | every chunk run of the mode comparison on the laptop (names: `f_n*` original test build, `f_k*` packed, `f_kp<l>` pipeline 3 with stage split at layer l, `f_ksv*` verification decode, `f_kus<n>*` threaded decode test with `NEIF_MT=n` (main thread + transformer thread + n-1 workers), with an `f` behind the number: 23 % forced wrong guesses, `f_k16on*` 16-bit activations, `f_r*` the reference chunk) |
| `laptop/packed_timing_k1.txt`, `packed_timing_k2.txt` | original (`tbN`) against packed (`tbK`; `tbF` = other prefetch distance) with the per-part time table (`NEIF_PROF=1`), interleaved rounds |
| `laptop/act16_results.txt` | the 16-bit activation experiment (`mask` = `NEIF_ACT16`; `mask 0` lines are the baseline) |
| `laptop/mt_results.txt` | raw lines of the thread / speculation / verification-decode stages (`fast/docs/PROGRESS.md`) |
| `laptop/e_results.txt`, `e_batch.log` | the opt-in multi-process compression in the real path `cmix -e`: identity against one process, kill test, comparison with the official compressor's output prefix, verification decode of that official prefix |
| `laptop/vnni_bench_out.txt` | the kernel micro-benchmark on the laptop (AVX-VNNI forms C and D, packed AVX2 form E) |

## Columns

`results.tsv` of `neif_split_eval.sh`: `utc host split step mode pipe threads flavour
binary_sha256 start end warm_from wall_s cpu_s peak_pss_kb kept_bits_or_decoded_bytes status cpus`.
`start`/`end` are stream bytes of the split, `warm_from` is where the model started, `wall_s` the
wall time of the whole run including the structure-only pass over the bytes in front of
`warm_from`, `cpu_s` the cpu seconds of all processes, `peak_pss_kb` the largest sum of the
processes' proportional set sizes, `kept_bits` the ideal code length of the bytes
`[start, end)`, `cpus` the logical CPUs the run was pinned to (`-` = not pinned).

`results.tsv` of `neif_run.sh` (`epyc_scale/`): `utc host name mode flavour binary_sha256 pipe mt
spec fault_permille act16 start end warm_from wall_s model_s cpu_s peak_pss_kb
bits_or_decoded_bytes out_sha16 status cpus`. `model_s` is the time from the first model byte
to the end (without the structure-only pass), `out_sha16` the first 16 hex digits of the sha256
of the coder output.

Lines of the laptop files: `NAME t|u mt N cpus LIST rc RC start S end E warm_from W
total_bits B model_s T bytes_per_s R sha SHA16` (mode `t`) or `... decoded N wrong N first_wrong P
model_s T ...` (mode `u`).

## What is measured and what is not

Everything in these files is a measured run. Timing on hostA / hostB depends on what the other
tenants of the host do: the same eight splits with identical output took 6,992 s on hostB and
12,245 s on hostA (`k16_per_split.tsv`). The laptop slows by up to 15 % when it has been loaded
for hours, which is why the laptop comparisons are interleaved rounds.
