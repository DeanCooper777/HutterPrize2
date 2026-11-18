// NEIF_MT: threads for the transformer step (a build with -DNEIF_TEST=1
// -DNEIF_MT=1; NOT part of the entry: without the macro NEIF_MT nothing of
// this is compiled, and without NEIF_MT=n in the environment no thread is
// created and every kernel takes its single-thread path).
//
// The design and the measurements behind it: runs/neif-par/DESIGN.md and
// PROGRESS.md of our project.  The rule for every use of the pool: a region
// is split only where the result is a sum of integers or an element-wise
// step, so the bytes the coder computes do not depend on the number of
// threads (checked by comparing the coder output and the decoded stream with
// the single-thread run of the same binary).
//
// The pool: NEIF_MT = n threads in all; the calling thread is share 0 and
// n - 1 workers spin on a sequence word of their own (no sleep: a region is
// 1-10 microseconds long).  Every thread is pinned (NEIF_MT_CPUS, a list of
// logical CPU numbers, default 0,1,2,...; NEIF_MT_PIN=0 leaves the placement
// to the system).  The workers start at the first Run(), i.e. at the first
// transformer step, not in the structure-only pass.
#ifndef NEIF_MT_H_
#define NEIF_MT_H_
#if defined(NEIF_TEST) && defined(NEIF_MT)

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <atomic>

namespace neif_mt {

constexpr int kMax = 8;
inline int g_n = 0;                 // threads of a region, the caller included (0, 1: no pool)
inline bool g_up = false;           // the workers run
inline bool g_pin = true;
inline int g_test = 0;              // NEIF_MT_TEST: 1 = the workers run but every share is done by the caller (what does a spinning thread cost?)
inline int g_cpu[kMax] = {0, 1, 2, 3, 4, 5, 6, 7};
struct alignas(64) Word { std::atomic<uint32_t> v; char pad[60]; };
inline Word g_go[kMax];             // worker w starts a region when its word changes
inline Word g_done;                 // workers that have finished the region
inline void (*g_fn)(int share, void* arg) = nullptr;
inline void* g_arg = nullptr;
inline pthread_t g_th[kMax];
inline int g_wid[kMax];
inline unsigned long long g_forks = 0;   // regions run on the pool
// The transformer thread (NEIF_MT_SPEC != 0): one more thread that runs the
// speculative step of the next byte (and, as the caller of Run(), its
// regions) while the main thread decodes.  It is pinned to the second CPU of
// the list, the workers of the pool to the CPUs behind it.
//   NEIF_MT_SPEC=1  speculate where the byte is known before its bits (the
//                   compressor of mode -t: every byte; the decoder: forced bytes)
//   NEIF_MT_SPEC=2  mode -u only, for tests: guess the byte from the stream
//   NEIF_MT_FAULT=p in 1/1000: that share of the guesses is made wrong on purpose
inline int g_spec = 0;
inline int g_fault = 0;
inline Word g_sp_go, g_sp_done;     // jobs posted to / finished by the transformer thread
inline void (*g_sp_fn)(void* arg) = nullptr;
inline void* g_sp_arg = nullptr;
inline pthread_t g_sp_th;

inline bool On() { return g_n > 1; }

inline void Pin(int cpu) {
  if (!g_pin) return;
  cpu_set_t s;
  CPU_ZERO(&s);
  CPU_SET(cpu, &s);
  pthread_setaffinity_np(pthread_self(), sizeof s, &s);
}

inline void* Worker(void* a) {
  const int w = *static_cast<int*>(a);
  Pin(g_cpu[w + 1 + (g_spec ? 1 : 0)]);
  uint32_t last = 0;
  for (;;) {
    while (g_go[w].v.load(std::memory_order_acquire) == last) _mm_pause();
    ++last;
    g_fn(w + 1, g_arg);
    g_done.v.fetch_add(1, std::memory_order_release);
  }
  return nullptr;
}

inline void* SpecThread(void*) {
  Pin(g_cpu[1]);
  uint32_t last = 0;
  for (;;) {
    while (g_sp_go.v.load(std::memory_order_acquire) == last) _mm_pause();
    ++last;
    g_sp_fn(g_sp_arg);
    g_sp_done.v.store(last, std::memory_order_release);
  }
  return nullptr;
}

// NEIF_MT, NEIF_MT_CPUS, NEIF_MT_PIN from the environment; once, at the start of the run
inline void Init() {
  const char* v = getenv("NEIF_MT");
  g_n = (v && *v) ? atoi(v) : 0;
  if (g_n > kMax - 1) g_n = kMax - 1;
  if (g_n < 1) g_n = 0;
  v = getenv("NEIF_MT_SPEC");
  g_spec = (v && *v && g_n > 0) ? atoi(v) : 0;
  v = getenv("NEIF_MT_FAULT");
  g_fault = (v && *v) ? atoi(v) : 0;
  v = getenv("NEIF_MT_TEST");
  g_test = (v && *v) ? atoi(v) : 0;
  v = getenv("NEIF_MT_PIN");
  g_pin = !(v && *v == '0');
  v = getenv("NEIF_MT_CPUS");
  if (v && *v) {
    int k = 0;
    while (*v && k < kMax) {
      g_cpu[k++] = atoi(v);
      while (*v && *v != ',') ++v;
      if (*v == ',') ++v;
    }
  }
}

inline void Start() {
  g_up = true;
  // the caller may have come with a one-CPU mask (taskset); the workers would inherit it
  Pin(g_cpu[0]);
  for (int w = 0; w + 1 < g_n; ++w) {
    g_wid[w] = w;
    if (pthread_create(&g_th[w], nullptr, Worker, &g_wid[w]) != 0) {
      fprintf(stderr, "neif_mt: cannot start a thread\n");
      exit(3);
    }
  }
  if (g_spec && pthread_create(&g_sp_th, nullptr, SpecThread, nullptr) != 0) {
    fprintf(stderr, "neif_mt: cannot start a thread\n");
    exit(3);
  }
  fprintf(stderr, "neif_mt: %d threads for the step%s, cpus", g_n, g_spec ? ", on a thread of their own" : "");
  for (int k = 0; k < g_n + (g_spec ? 1 : 0); ++k) fprintf(stderr, " %d", g_cpu[k]);
  fprintf(stderr, "%s\n", g_pin ? "" : " (not pinned)");
}

// one region: fn(share, arg) for share = 0 .. g_n - 1, share 0 on the calling
// thread; returns when all are done.  Only called when On().
inline void Run(void (*fn)(int, void*), void* arg) {
  if (!g_up) Start();
  ++g_forks;
  if (g_test == 1) {
    for (int s = 0; s < g_n; ++s) fn(s, arg);
    return;
  }
  g_fn = fn;
  g_arg = arg;
  g_done.v.store(0, std::memory_order_relaxed);
  const int nw = g_n - 1;
  for (int w = 0; w < nw; ++w)
    g_go[w].v.store(g_go[w].v.load(std::memory_order_relaxed) + 1, std::memory_order_release);
  fn(0, arg);
  while (g_done.v.load(std::memory_order_acquire) != (uint32_t)nw) _mm_pause();
}

// the transformer thread: start fn(arg) on it (it must be idle); is it idle; wait until it is
inline void SpecPost(void (*fn)(void*), void* arg) {
  if (!g_up) Start();
  g_sp_fn = fn;
  g_sp_arg = arg;
  g_sp_go.v.store(g_sp_go.v.load(std::memory_order_relaxed) + 1, std::memory_order_release);
}
inline bool SpecIdle() {
  return g_sp_done.v.load(std::memory_order_acquire) == g_sp_go.v.load(std::memory_order_relaxed);
}
inline void SpecWait() {
  while (!SpecIdle()) _mm_pause();
}

}  // namespace neif_mt

#endif  // NEIF_TEST && NEIF_MT
#endif  // NEIF_MT_H_
