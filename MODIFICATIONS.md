# Modifications to cmix-neif 2.0

**This repository is a modified version of cmix-neif 2.0.** This notice is the one the GNU
General Public License, version 3, asks for in section 5 (a): it says that the work was
modified, and when.

## The original work

* **cmix-neif 2.0**, a Hutter Prize entry by Marcus Wölk (Marcus Woelk), submitted on
  30 October 2026, licensed under the GNU General Public License, version 3.
* Source: <https://github.com/marwoelk/cmix-neif-2.0>, commit `35d1957` ("cmix-neif-2.0").
* cmix-neif 2.0 is itself built on `cmix-neif-1.0`, `cmix-lex`, `fx2-cmix`, `cmix-hp`, `cmix`
  and the other components that its `NOTICE.md` lists with their authors (among them Byron
  Knoll, Kaido Orav, Ibrahim Marcouch) and licences. `NOTICE.md`, `LICENSE`, `third_party/` and
  every copyright and licence notice inside the source files are unchanged here.
* The first commit after the initial one in this repository's history ("Import cmix-neif 2.0 at
  35d1957 (unmodified)") is upstream's tree exactly (git tree
  `0084a8876b81e7eac41903337dbb8b4762f0e6c7`), so `git diff <that commit> HEAD` shows every
  change listed below.

## Who changed it, when, and under which licence

The changes were made by Dean Cooper between 15 and 18 November 2026, with the help of an AI
coding assistant (Anthropic's Claude; see the `Co-Authored-By` lines of the commits). They are
released under the same licence as the original, the **GNU General Public License, version 3**
(`LICENSE`); the whole repository is distributed under that licence.

There is no warranty, for the original or for the changes (sections 15 and 16 of the licence).

## What was changed

### 1. Source (commit "Speed modifications (all behind NEIF_TEST)", 15-17 November 2026)

27 files under `src/` (3 new, 24 changed; 4,759 lines added, 3 lines replaced). The added code is
inside compile-time macros that upstream's build never defines: `NEIF_TEST`, or `NEIF_MT` /
`NEIF_PAR` inside `NEIF_TEST`, or `NEIF_PACKED` / `NEIF_PFD` (usable alone). One change is
outside any macro: in `src/tf1/opt/qmat_dense.cpp` the literal prefetch distance `6144` in three
`_mm_prefetch` lines became the constant `QPF_DIST`, which is `constexpr int QPF_DIST = 6144`
unless `NEIF_PFD` is defined, so the compiled code is the same. Built without these macros
(upstream's `build.sh`), the tree gives the published compressor byte for byte: `comp9`,
4,042,564 B, sha256 `976bf2d309707ab555e0c92720da5508ff0e5e6b846fc7082234262b06767f82`
(`fast/neif_build_official.sh`).

| file | change |
|---|---|
| `src/neif_test.h` (new) | state and helpers of the test modes: chunk window (`NEIF_T_START/END/WARM`), bit counting and JSON result, the forked pipeline of 2 or 3 processes (shared-memory queues, records per byte, failure handling), the state of the verification decode, the per-part time table (`NEIF_PROF`), the measuring hooks of `NEIF_PAR` |
| `src/neif_mt.h` (new) | `NEIF_MT`: a small pool of pinned spinning threads and the hand-off for the speculative transformer step |
| `src/tf1/opt/act16_test.h` (new) | `NEIF_ACT16`: experimental kernels that carry 16-bit activations into selected matmuls |
| `src/runner.cpp` | command-line modes `-t` (code a chunk of the coder stream and count ideal code lengths) and `-u` (decode a chunk and compare every byte with the stream); the pipeline start in `-t`; the verification decode (`NEIF_PIPE` with `-u`); the opt-in pipeline in the normal compress path `-e` (`NeifEStart` / `NeifEEnd`, `NEIF_E_STOP`); the launch of the speculative step |
| `src/predictor.cpp`, `src/predictor.h` | hooks of the pipeline stages (send / take the PPMd result and the transformer rows), time-table marks, the speculative step (`NeifSpecLaunch`, `NeifSpecJob`, `NeifPpmdByteUpdate`), measuring hooks |
| `src/coder/encoder.cpp` | bit counting for the chunk mode (sum of -log2 p of every coded bit, also in front of the end stages); measuring hooks |
| `src/bit_head.cpp` | `NeifSyncSlot`: selects the BITLSTM weight set for a chunk that starts behind the metadata block |
| `src/models/ppmd.cpp`, `src/models/ppmd.h` | chunk mode: the pending restarts (`NeifSync`); pipeline: a stage without the PPMd model takes the byte's result from the queue (`NeifRow`, `NeifDisown`); `NEIF_MT`: shadow wrapper, begin / adopt / undo of a speculative byte update |
| `src/models/ppmd_page.cpp`, `src/models/ppmd_page.h` | page-pool accounting across forked stages; `NEIF_MT`: undo journal of heap writes |
| `src/models/match.cpp`, `src/models/match.h` | `NEIF_MT`: `Peek`, the match channel behind a guessed byte without changing state |
| `src/tf1/tf1_engine.cpp`, `src/tf1/tf1_engine.h` | structure-only pass of the chunk mode (rows booked, network step left out); pipeline record of the driver's outputs (`NeifStateIo`); `NEIF_MT`: `SpecPossible`, `SpecRun`, join points |
| `src/tf1/opt/model_opt.cpp`, `src/tf1/opt/model_opt.h` | split of the layer stack between pipeline stages (`NEIF_PIPE_SPLIT`); `NEIF_MT`: threaded MLP region, shadow engine and commit of a speculative row; the 16-bit activation switch; time-table marks |
| `src/tf1/opt/qmat_dense.cpp`, `src/tf1/opt/qmat_dense.h` | `NEIF_PACKED`: dense matmuls on nibble-packed weights in groups of 8 rows (AVX2), same integer sums; `NEIF_PFD`: prefetch distance; `NEIF_MT`: range forms of the kernels |
| `src/tf1/opt/qmat_sparse.cpp`, `src/tf1/opt/qmat_sparse.h` | `NEIF_MT`: range form of the sparse down-projection sums |
| `src/tf1/opt/attn.cpp`, `src/tf1/opt/attn.h` | `NEIF_MT`: window attention split over threads |
| `src/tf1/opt/kda.cpp`, `src/tf1/opt/kda.h` | `NEIF_MT`: KDA layer step that reads the state from one buffer and writes it to another (for the speculative row) |
| `src/tf1/weights_io_compressed.cpp` | test switch `NEIF_T_WEIGHTS_ANY` for loading another weights container in the chunk mode |

Some comments in these files point to `runs/neif-par/DESIGN.md` and
`runs/neif-par/PROGRESS.md`; those documents are `fast/docs/DESIGN.md` and
`fast/docs/PROGRESS.md` in this repository.

### 2. Files added or moved (commit "Scripts, results and documentation", 18 November 2026)

| file | change |
|---|---|
| `README.md` | new: describes the modifications. Upstream's `README.md` was moved, unmodified, to `README.cmix-neif.md` |
| `MODIFICATIONS.md` | new: this notice |
| `.gitattributes` | new: `* -text`, so that git never converts line ends (the official build is byte-sensitive) |
| `fast/` | new: scripts for the test modes, design notes (`fast/docs/`), result tables (`fast/results/`) |

Nothing else was changed: `LICENSE`, `NOTICE.md`, `CHANGES.md`, `ERRORS.md`, `build.sh`,
`install.sh`, `makefile`, the weights, `training/`, `third_party/`, `tools_src/`, `dictionary/`,
`doc/`, `pgo_data/` and `prof_input/` are upstream's files.
