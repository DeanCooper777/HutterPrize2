// Changes in this file: van_layer_kv(), the `gelesen` ("read") flag with its
// early exit in step() and set_gelesen(), for rows whose output is not read;
// and the mlp.down density threshold, which can follow the layer's neuron
// count instead of DMLP; the side-channel token row (proj_add,
// paket_token_row) and last_hidden(); the transformer form of the beehive
// entry (AttnRes depth mix, one K/V head for the three query heads, value
// embeddings, prior logit head); in the AttnRes form the queue of
// unread rows that runs them as one block of up to 16 rows, with one layer
// code for read and unread rows.  All additive, except that the AttnRes
// members and steps moved into a type of their own (ArRow) and, in the block
// form, the row buffers into ZRow.

// Origin: fx2-cmix-transformer by Vladimir Ivanov, GPL-3.0, file cpp_infer/src/opt/model_opt.cpp.
// Ported and modified for this entry in 2026; the changes are listed in
// sections 3 and 4 of README.md and of the README of cmix-neif-1.0,
// among them a K/V-only last layer for rows whose output is not read.
// TransformerOpt implementation. Wiring order matches SPEC section 3.3 and
// the arena stream order of arena_build.cpp exactly.
//
// Profiling (-DFX2_PROF): raw rdtsc (no lfence/rdtscp - the ~67-cycle
// serialized stamp would distort short sections; raw back-to-back stamps
// measure ~10-20 core cycles) section accumulators, one per call site,
// aggregated into the four SPEC section-9 groups at print time and
// overhead-corrected by the measured empty-section delta.

#include "../../meld.h"
#include "model_opt.h"
#include "../../cold_size.h"
#include "../../semantic_cache.h"
#ifdef NEIF_TEST
#include "../../neif_test.h"   // chunk evaluation: pipeline stages
#endif

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <new>
#include <vector>

#include <sys/mman.h>

#include "arena_build.h"
#include "attnres_mix.h"
#include "attn.h"
#include "glue.h"
#include "kda.h"
#include "kda_math.h"
#include "qmat_dense.h"
#include "qmat_sparse.h"
#ifdef NEIF_TEST
#include "act16_test.h"   // experiment: 16-bit activations (NEIF_ACT16)
#ifdef NEIF_MT
#include "../../neif_mt.h"   // threads for the transformer step (NEIF_MT)
#endif
#endif

// Debug-only component isolation for the naive-vs-opt cross-check (never
// defined in the production build targets): swap the fused KDA step and/or
// the attention kernel for the naive path (bit-exact vs src/model.cpp) to
// attribute divergence per component.
#if defined(FX2_XCHECK_NAIVE_KDA) || defined(FX2_XCHECK_NAIVE_ATTN)
#include "../kernels.h"
#include "../tf1_config.h"
#endif

namespace fx2 {
namespace opt {

namespace {
constexpr int V = 208, D = 192, NL = 16, DH = 64, NH = 3,
              DMLP = 1024;
constexpr int NKIMI = 12, NVAN = 4,
              NSKIP = 8;

#if !defined(FX2_XCHECK_NAIVE_ATTN)
constexpr int KVP = 1;
#else
constexpr int KVP = NH;
#endif

constexpr int WIN = ATTN_WIN, ROPE_LEN = 131072;
static_assert(WIN <= ATTN_WIN,
              "the ring cannot be wider than the scan.");
static_assert(WIN % ATTN_BLK == 0 && WIN >= 256,
              "the ring follows the block width of the scan.");
// density thresholds for the sparse column kernels (qmat.h)
constexpr int PRIOR_DENSE_NNZ =
    static_cast<int>(QMAT_SPARSE_DENSITY_THRESHOLD * V);  // 194

// beta projection: 3x192 fp32 matvec on xn. Bit-identical to the naive
// fx2::dot_f32(xn, w, 192) (same 8-wide fmadd chain + hsum tree).  Out of
// line: inlined, each of its call sites carried the unrolled chain.
__attribute__((noinline)) float dot192(const float* a, const float* b) {
  __m256 acc = _mm256_setzero_ps();
  for (int i = 0; i < 192; i += 8)
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
  return hsum256v(acc);
}
}  // namespace

// ---------------------------------------------------------------------------
// profiling sites
// ---------------------------------------------------------------------------
namespace {

enum PSite : int {
  // group 0: int4 x int8 matmuls (ALL arenas)
  PS_MM_PRIOR = 0,
  PS_MM_KQKV,
  PS_MM_GUP,
  PS_MM_GDN,
  PS_MM_KOP,
  PS_MM_VQKV,
  PS_MM_VOP,
  PS_MM_MLP_UP,
  PS_MM_MLP_DOWN,
  PS_MM_UNEMB,
  // group 1: vanilla attention mechanism
  PS_AT_INSERT,
  PS_AT_STEP,
  // group 2: kimi linear attention mechanism
  PS_KD_BETA,
  PS_KD_STEP,
  // group 3: everything else
  PS_EL_PRIORQ,
  PS_EL_EMBED,
  PS_EL_RESID,
  PS_EL_NQ_KIMI,
  PS_EL_NQ_VAN,
  PS_EL_NQ_MLP,
  PS_EL_NQ_FIN,
  PS_EL_HEADPREP,
  PS_EL_OQ,
  PS_EL_HEAD,
#if 16 && defined(FX2_PROF)
  PS_ZB_RUN,
#endif
  PS_N
};

#ifdef FX2_PROF
struct PSiteInfo {
  const char* name;
  int group;
};
const PSiteInfo kPSites[PS_N] = {
    {"prior spmv (int4 cols)", 0},
    {"kimi q/k/v proj (27)", 0},
    {"gate up (fg+og, 18)", 0},
    {"gate down (fg+og, 18)", 0},
    {"kimi out proj (9)", 0},
    {"van q/k/v proj (9)", 0},
    {"van out proj (3)", 0},
    {"mlp up + relu2q (12)", 0},
    {"mlp down spmv (12)", 0},
    {"unembed (1)", 0},
    {"kv insert (3)", 1},
    {"attn step qk+sm+pv (3)", 1},
    {"beta matvec (9)", 2},
    {"kda fused step (9)", 2},
    {"prior quant+idx", 3},
    {"embed combine", 3},
    {"resid/skip/tok ops", 3},
    {"norm+quant kimi (9)", 3},
    {"norm+quant van (3)", 3},
    {"norm+quant mlp (12)", 3},
    {"final norm+quant", 3},
    {"head-prep norm/rope/q (3)", 3},
    {"attnout quant (12)", 3},
    {"softcap+softmax head", 3},
    {"zblock run (spans sites)", 3},
};
const char* kGroups[4] = {"int4xint8 matmuls", "vanilla attention mechanism",
                          "kimi (KDA) mechanism", "everything else"};

inline uint64_t prof_rdtsc() {
  unsigned lo, hi;
  asm volatile("rdtsc" : "=a"(lo), "=d"(hi)::"memory");
  return (static_cast<uint64_t>(hi) << 32) | lo;
}
uint64_t g_pacc[PS_N] = {};
uint64_t g_pcalls[PS_N] = {};
struct PScope {
  int s;
  uint64_t t0;
  explicit PScope(int s_) : s(s_), t0(prof_rdtsc()) {}
  ~PScope() {
    g_pacc[s] += prof_rdtsc() - t0;
    g_pcalls[s]++;
  }
};
#define PSCOPE(site) PScope pscope_(site)
#else
#define PSCOPE(site) \
  do {               \
  } while (0)
#endif

}  // namespace

void prof_reset() {
#ifdef FX2_PROF
  std::memset(g_pacc, 0, sizeof(g_pacc));
  std::memset(g_pcalls, 0, sizeof(g_pcalls));
#endif
}

bool prof_enabled() {
#ifdef FX2_PROF
  return true;
#else
  return false;
#endif
}

void prof_print(double tsc_to_core, long tokens) {
#ifndef FX2_PROF
  (void)tsc_to_core;
  (void)tokens;
  std::printf("prof: not compiled in (build with -DFX2_PROF)\n");
#else
  // empty-section overhead: median back-to-back raw-rdtsc delta
  const int NS = 4001;
  std::vector<uint64_t> st(NS);
  for (int i = 0; i < NS; i++) st[i] = prof_rdtsc();
  std::vector<double> dd(NS - 1);
  for (int i = 0; i + 1 < NS; i++) dd[i] = double(st[i + 1] - st[i]);
  std::nth_element(dd.begin(), dd.begin() + (NS - 1) / 2, dd.end());
  const double stamp_ticks = dd[(NS - 1) / 2];
  std::printf("\n== per-part profile (raw rdtsc, overhead-corrected by %.1f "
              "ticks/section = %.1f core cyc) ==\n",
              stamp_ticks, stamp_ticks * tsc_to_core);
  double gcyc[4] = {}, gcalls[4] = {};
  double site_cyc[PS_N];
  double total = 0;
  for (int s = 0; s < PS_N; s++) {
    double ticks = double(g_pacc[s]) - double(g_pcalls[s]) * stamp_ticks;
    if (ticks < 0) ticks = 0;
    site_cyc[s] = ticks * tsc_to_core / double(tokens > 0 ? tokens : 1);
    gcyc[kPSites[s].group] += site_cyc[s];
    gcalls[kPSites[s].group] += double(g_pcalls[s]);
    total += site_cyc[s];
  }
  std::printf("%-34s %12s %6s %9s\n", "group", "cyc/token", "%", "calls/tok");
  for (int g = 0; g < 4; g++)
    std::printf("%-34s %12.0f %5.1f%% %9.1f\n", kGroups[g], gcyc[g],
                100.0 * gcyc[g] / total,
                gcalls[g] / double(tokens > 0 ? tokens : 1));
  std::printf("%-34s %12.0f %5.1f%%\n", "TOTAL (instrumented sections)", total,
              100.0);
  std::printf("\n%-34s %12s %6s %9s\n", "  site", "cyc/token", "%",
              "calls/tok");
  for (int g = 0; g < 4; g++) {
    std::printf("  -- %s --\n", kGroups[g]);
    for (int s = 0; s < PS_N; s++) {
      if (kPSites[s].group != g) continue;
      std::printf("  %-32s %12.0f %5.1f%% %9.1f\n", kPSites[s].name,
                  site_cyc[s], 100.0 * site_cyc[s] / total,
                  double(g_pcalls[s]) / double(tokens > 0 ? tokens : 1));
    }
  }
#endif
}

struct ArRow {
  static constexpr int AR_NQ = 2 * NL + 1;
  alignas(64) float ar_x0[D];
  alignas(64) float ar_qA[AR_NQ][D];
  float ar_qM[AR_NQ], ar_qZ[AR_NQ];
  int ar_n;               // completed sources (x0 + frozen block sums)
  int ar_qnext;           // first read that has not happened yet
  alignas(64) float ar_part[D];
  float ar_part_rstd;  // 1/rms of ar_part, kept current by take_f
  float ar_part_dot;   // dot(w_q, ar_part) for q == ar_part_dot_q
  int ar_part_dot_q = -1;
  bool ar_has_part;
  alignas(64) float ar_f[D];
  static const float* qw(const OptModel& M, int q) {
    return q == 2 * NL ? M.ar_q_final : ((q & 1) ? M.ar_q_mlp[q >> 1] : M.ar_q_mix[q >> 1]);
  }
  void push_x0(const OptModel& M, const float* x);
  void push_part(const OptModel& M);
  void read(const OptModel& M, float* x);
  void take_f(const OptModel& M, int next_q);
};

// source 0 = x0: M = l, Z = 1; A_q stays x0 (lazy) until the first block sum
__attribute__((noinline)) void ArRow::push_x0(const OptModel& M, const float* x) {
  std::memcpy(ar_x0, x, sizeof(float) * D);
  const float rs = ar_rstd192(ar_x0);
  for (int q = ar_qnext; q < AR_NQ; q++) {
    ar_qM[q] = ar_dot192(qw(M, q), ar_x0) * rs;
    ar_qZ[q] = 1.0f;
  }
  ar_n = 1;
}
// freeze the partial as a completed source for all remaining reads: logits,
// then both exps 8 reads at a time, then A
__attribute__((noinline)) void ArRow::push_part(const OptModel& M) {
  alignas(32) float ea[AR_NQ + 7], eb[AR_NQ + 7];
  const int q0 = ar_qnext, nq = AR_NQ - q0;
  for (int k = 0; k < nq; k++) {
    const int q = q0 + k;
    const float l = ar_dot192(qw(M, q), ar_part) * ar_part_rstd;
    const float m2 = l > ar_qM[q] ? l : ar_qM[q];
    ea[k] = ar_qM[q] - m2;  // old-sum scale exponent (<= 0)
    eb[k] = l - m2;         // new-source exponent (<= 0)
    ar_qM[q] = m2;
  }
  for (int k = nq; k < ((nq + 7) & ~7); k++) ea[k] = eb[k] = 0.0f;
  for (int k = 0; k < nq; k += 8) {
    _mm256_store_ps(ea + k, exp256_ps(_mm256_load_ps(ea + k)));
    _mm256_store_ps(eb + k, exp256_ps(_mm256_load_ps(eb + k)));
  }
  for (int k = 0; k < nq; k++) {
    const int q = q0 + k;
    const float* Aold = ar_n == 1 ? ar_x0 : ar_qA[q];
    float* Aq = ar_qA[q];
    const __m256 va = _mm256_set1_ps(ea[k]), vb = _mm256_set1_ps(eb[k]);
    for (int d = 0; d < D; d += 8)
      _mm256_store_ps(Aq + d, _mm256_fmadd_ps(vb, _mm256_load_ps(ar_part + d),
                                              _mm256_mul_ps(va, _mm256_load_ps(Aold + d))));
    ar_qZ[q] = ea[k] * ar_qZ[q] + eb[k];
  }
  ar_n++;
  ar_has_part = false;
}
// x = depth_mix(x0, completed block sums, partial) for the next read
__attribute__((noinline)) void ArRow::read(const OptModel& M, float* x) {
  const int q = ar_qnext++;
  const float* Aq = ar_n == 1 ? ar_x0 : ar_qA[q];
  if (!ar_has_part) {  // completed sources only: out = A / Z
    if (ar_n == 1) {   // x0 alone: exactly x0
      for (int d = 0; d < D; d += 8) _mm256_store_ps(x + d, _mm256_load_ps(Aq + d));
    } else {
      const __m256 r = _mm256_set1_ps(1.0f / ar_qZ[q]);
      for (int d = 0; d < D; d += 8) _mm256_store_ps(x + d, _mm256_mul_ps(_mm256_load_ps(Aq + d), r));
    }
  } else {
    const float lp = (ar_part_dot_q == q ? ar_part_dot : ar_dot192(qw(M, q), ar_part)) * ar_part_rstd;
    const float m = lp > ar_qM[q] ? lp : ar_qM[q];
    alignas(32) float ex[8] = {ar_qM[q] - m, lp - m, 0, 0, 0, 0, 0, 0};
    _mm256_store_ps(ex, exp256_ps(_mm256_load_ps(ex)));
    const float den = ex[0] * ar_qZ[q] + ex[1];
    const __m256 ra = _mm256_set1_ps(ex[0] / den), rb = _mm256_set1_ps(ex[1] / den);
    for (int d = 0; d < D; d += 8)
      _mm256_store_ps(x + d, _mm256_fmadd_ps(rb, _mm256_load_ps(ar_part + d),
                                             _mm256_mul_ps(ra, _mm256_load_ps(Aq + d))));
  }
}
// partial += f (the block's running sum), f = the sub-layer output;
// next_q: the read that uses this partial next (its logit's dot is fused
// into the sum), -1 when a block boundary freezes the partial first
__attribute__((noinline)) void ArRow::take_f(const OptModel& M, int next_q) {
  if (ar_has_part && next_q >= 0) {
    ar_part_rstd = ar_add_rstd_dot192(ar_part, ar_f, qw(M, next_q), &ar_part_dot);
    ar_part_dot_q = next_q;
    return;
  }
  ar_part_dot_q = -1;
  if (ar_has_part) {
    ar_part_rstd = ar_add_rstd192(ar_part, ar_f);
  } else {
    std::memcpy(ar_part, ar_f, sizeof(float) * D);
    ar_part_rstd = ar_rstd192(ar_part);
    ar_has_part = true;
  }
}

struct ZRow : ArRow {
  alignas(64) float x[D];
  alignas(64) float xn[D];
  alignas(64) float a[D];
  alignas(64) float b[D];
  alignas(64) float c[D];
  alignas(64) float graw[D];
  alignas(64) float og192[D];
  alignas(64) float gn[D];
  alignas(64) uint8_t q5[5 * D];  // up to 5 biased-u8 copies of xn
  alignas(64) uint8_t q64b[DH];
  alignas(64) uint8_t q768[DMLP];
  alignas(64) uint16_t idx768[DMLP + 16];
  alignas(64) float prior_f32[208];
  alignas(32) float braw[8];
  int nnz;
  int tok;
  PaketChannels pk;
};
struct TransformerOptImpl : ZRow {
  OptModel M;
  AttnKind kind;

  // streaming state (hugepage pool: KDA states, then KV caches)
  HugeBuf spool;
  KdaState* kst;   // [9]
  AttnKVF32* kvf;  // [3] iff kind == KVF32
  AttnKV* kvi;     // [3] iff kind == KVI8
  int64_t t;
  int64_t rope_off;

#ifdef FX2_XCHECK_NAIVE_KDA
  struct NaiveKst {
    alignas(64) float S[NH][DH * DH];
    alignas(64) float hist[3][3][D];
  };
  std::vector<NaiveKst> nkst = std::vector<NaiveKst>(NKIMI);
#endif
#ifdef FX2_XCHECK_NAIVE_ATTN
  struct NaiveVst {
    alignas(64) int8_t kring[WIN][D];
    alignas(64) int8_t vring[WIN][D];
  };
  std::vector<NaiveVst> nvst = std::vector<NaiveVst>(3);
#endif

  // per-token skip stack (written by layers 0..5, read by 6..11 of the SAME
  // token) - transient within step
  alignas(64) float skip_store[NSKIP][D];

  // scratch
  alignas(64) float yb[D];
  // AttnRes replace mode (Kimi Team, "Attention Residuals", arXiv
  // 2603.15031; engine implementation adapted from cmix-neif-beehive,
  // Vladimir Malinovskii, 2026, 92c07c0b, model_opt.cpp, depth-mix form
  // TF_ARV2 = 5).  Sources of a read: x0, the completed block sums and the
  // running partial sum of the current block.  `acc` is where a sub-layer
  // adds its output: ar_f inside the layer loop.
  // Online softmax per future read q over the completed sources, kept at push
  // time: M_q = max logit, Z_q = sum exp(l - M_q), A_q = sum exp(l - M_q) v.
  // A read only adds the partial: m = max(M_q, l_p), a = exp(M_q - m),
  // b = exp(l_p - m), out = (a A_q + b part) / (a Z_q + b).  Read index q:
  // 2l = mixer read of layer l, 2l+1 = its MLP read, 2 NL = the final read.
  static constexpr int ARB = 2;
  const float* ar_qw(int q) const { return ArRow::qw(M, q); }
  void ar_push_x0() { push_x0(M, x); }
  void ar_push_part() { push_part(M); }
  void ar_read() { read(M, x); }
  void ar_take_f(int next_q) { take_f(M, next_q); }
  // key of the semantic cache: staged during the step (in the AttnRes form
  // the MLP read of the last layer, else the residual behind layer NL-2), and
  // the key of the last read row
  alignas(64) float sc_l2_stage[D];
  alignas(64) float sc_key[D];
  alignas(64) float pre[D];   // the row buffers are in ZRow
  alignas(64) int8_t qq[D], kk[D], vv[D];
  alignas(64) uint8_t q8p[224];  // V rounded to 32
  alignas(64) uint16_t idxp[224 + 16];
  alignas(64) float logits[208];
  alignas(64) float probs208[208];
  alignas(32) float sfall[32], cfall[32];  // rope fallback row
  // The assembled token row with side channels.  `tok_paket` is the normed
  // sum, `pk` the channels of the row that step() is about to consume.
  alignas(64) float tok_paket[D];
  // fast-weight state, taken over from ex-girlfriend 1.0 (Ryan
  // Landay, GPL-3.0-only; model_opt.cpp FW-DUAL + FW-ROW + FW-RULES).  One dW
  // per block for the last FWK MLP blocks (block b = layer NL-FWK+b), column
  // per code index: dW_b[(b*DMLP + i)*D + j]; zero-initialised, never reset.
  // All blocks feed the ONE second head on x + sum_b dW_b h_b.
  // dW lives in the state pool behind the KDA states and the
  // KV rings (64-byte aligned columns, huge pages, zeroed by the pool).  The
  // dW step of update u is not applied at once: it stays pending PER COLUMN
  // until that column is read again (fw_apply) or until the slot of gx_u in
  // the ring of the last FWR updates is reused.  A column read in a row has no
  // pending step when that row's update arrives (the read applied it), so every
  // column carries at most one and its steps keep their order; nothing but
  // fw_apply reads dW.  Every element gets exactly the arithmetic of the
  // immediate step (col - c*gx, then dx + col*h).
  static constexpr int FWK = 4;
  static_assert(FWK >= 1 && FWK <= NL, "more blocks than layers");
  static constexpr int FWR = 64;   // ring of the last gx vectors (power of two)
  // Only the FW_TOPK largest codes of a block take part in forward and
  // update, and the mean RMS msd of a block is renewed every FW_MSD-th
  // update (v itself every update).
  static constexpr int FW_TOPK = 32;
  static constexpr int FW_MSD = 16;
  float* fw_dW;
  int fw_nnz[FWK];
  uint16_t fw_idx[FWK][DMLP];
  alignas(64) float fw_h[FWK][DMLP];
  bool fw_gel;     // the current step() is a read row (captures h); true from the ctor
  float fw_denom;
  uint64_t fw_t;      // update counter of the step schedule
  alignas(64) float fw_v[FWK][DMLP];   // EMA of h_i^2 per block
  alignas(64) float fw_sq[FWK][DMLP];  // sqrt(fw_v) of the last msd renewal
  float fw_sh[FWK][DMLP];
  alignas(64) float fw_gxr[FWR][D];    // gx of the last FWR updates
  alignas(64) int32_t fw_pt[FWK][DMLP];     // update whose step is pending, or -1
  float fw_pc[FWK][DMLP];              // its coefficient step * sh
  int32_t fw_u;                         // updates so far
  float fw_msd[FWK];                   // msd of the last renewal
  uint16_t fw_sel[DMLP];                    // the FW_TOPK codes of a block
  alignas(64) float fw_dx[D], fw_x2[D], fw_xn[D];
  alignas(64) uint8_t fw_q[256];
  alignas(64) float fw_logits[208],
                    fw_probs208[208];

  // The object (0.9 MB, members aligned to 64) comes from
  // MapNull (arena_build.h): page aligned and zero-filled, so neither the
  // aligned new/delete of the library nor member initialisers are needed;
  // the non-zero starts are set in the constructor.
  static void* operator new(size_t n, std::align_val_t) { return MapNull(n); }
  static void operator delete(void* p, size_t n, std::align_val_t) { ::munmap(p, n); }

  COLD_SIZE_ATTR explicit TransformerOptImpl(const char* path, AttnKind k) : kind(k) {
    gelesen = true;
    fw_gel = true;
    M.load(path);
#ifdef NEIF_TEST
    act16::init();   // NEIF_ACT16: the site groups with 16-bit activations
    if (act16::g_mask) {
      std::fprintf(stderr, "neif_act16: mlp widths");
      for (int l = 0; l < NL; l++) std::fprintf(stderr, " %d", M.mlp[l].down.d_in);
      std::fprintf(stderr, "\n");
    }
#endif
    const size_t kda_sz = sizeof(KdaState) * NKIMI;
    const size_t kv_sz =
        (kind == AttnKind::KVF32 ? sizeof(AttnKVF32) : sizeof(AttnKV)) * NVAN;
    // dW behind the states (kda_sz and kv_sz are multiples of 64)
    const size_t fw_sz = sizeof(float) * size_t(FWK) * DMLP * D;
    spool.alloc(kda_sz + kv_sz + fw_sz);
    spool.trim(kda_sz + kv_sz + fw_sz);
    fw_dW = reinterpret_cast<float*>(spool.p + kda_sz + kv_sz);
    for (int b = 0; b < FWK; b++)
      for (int j = 0; j < DMLP; j++) fw_pt[b][j] = -1;
    kst = reinterpret_cast<KdaState*>(spool.p);
    if (kind == AttnKind::KVF32)
      kvf = reinterpret_cast<AttnKVF32*>(spool.p + kda_sz);
    else
      kvi = reinterpret_cast<AttnKV*>(spool.p + kda_sz);
    for (int i = 0; i < NKIMI; i++) kda_layer_reset(kst[i]);
  }

  void begin(int64_t rope_position_offset) {
    zb_run();
    for (int i = 0; i < NKIMI; i++) kda_layer_reset(kst[i]);
#if defined(NEIF_TEST) && defined(NEIF_MT)
    for (int i = 0; i < NKIMI; i++) kS_cur[i] = nullptr;   // NEIF_MT: the (zeroed) states in kst are the committed ones
#endif
#ifdef FX2_XCHECK_NAIVE_KDA
    std::memset(nkst.data(), 0, sizeof(NaiveKst) * NKIMI);
#endif
    t = 0;  // invalidates the KV rings (validity is [0, t] within an article)
    rope_off = rope_position_offset;
  }

  static constexpr ptrdiff_t ZS = static_cast<ptrdiff_t>(sizeof(ZRow));
  // a matmul over the rows: one row (a read row) on the single-row kernel,
  // several fused
  static void mm_f32(const QDense& m, const uint8_t* act, float* out, int n) {
    if (n == 1)
      qgemv_f32(m, act, out);
    else
      qgemm(m, act, ZS, out, ZS, n, false);
  }
  static void mm_add(const QDense& m, const uint8_t* act, float* out, int n) {
    if (n == 1)
      qgemv_add(m, act, out);
    else
      qgemm(m, act, ZS, out, ZS, n, true);
  }
  static void mm_quant_bias(const QDense& m, const uint8_t* act, float s, uint8_t* q, int n) {
    if (n == 1)
      qgemv_quant_bias(m, act, s, q);
    else
      qgemm_quant_bias(m, act, ZS, s, q, ZS, n);
  }

  // the rope rows of position pos; the libm fallback beyond the shipped table
  // out of line
  __attribute__((noinline)) void zb_rope_fall(int64_t pos) {
    const float fpos = static_cast<float>(pos);
#pragma clang loop unroll(disable) vectorize(disable)
    for (int i = 0; i < 32; i++) {
      const float ang = fpos * M.inv_freq[i];
      sfall[i] = std::sin(ang);
      cfall[i] = std::cos(ang);
    }
  }
  void zb_rope(int64_t pos, const float*& sp, const float*& cp) {
    if (pos < ROPE_LEN) {
      sp = M.rope_sin + size_t(pos) * 32;
      cp = M.rope_cos + size_t(pos) * 32;
    } else {  // beyond the shipped table: libm fallback (same as naive)
      zb_rope_fall(pos);
      sp = sfall;
      cp = cfall;
    }
  }

  // after kv_expand of cmix-neif-beehive: k and v of row r come
  // out of the projections with KVH heads; v gets c0 v + c1 VE[token row],
  // then the K/V heads are repeated for the NH query heads, so the kernels and
  // the KV ring stay as they are.
  void kv_expand(const VanArenas& L, ZRow& r) {
    float* kf = r.b;
    float* vf = r.c;
    {
      constexpr int W = 1 * DH;
      int row = r.tok;
      if (row >= V) row = 1 == 1 ? V - 1 : (1 == 2 ? row - V : -1);
      if (row >= 0) {
        const float* e = L.ve + size_t(row) * W;
        for (int i = 0; i < W; i++) vf[i] = L.ve_c0 * vf[i] + L.ve_c1 * e[i];
      } else {
        for (int i = 0; i < W; i++) vf[i] = L.ve_c0 * vf[i];
      }
    }
    for (int h = 1; h < KVP; h++) {
      std::memcpy(kf + h * DH, kf + (h % 1) * DH, sizeof(float) * DH);
      std::memcpy(vf + h * DH, vf + (h % 1) * DH, sizeof(float) * DH);
    }
  }

  // kimi (KDA) layer ki for the rows r[0, n); the output goes to each row's ar_f
#ifdef NEIF_TEST
  // NEIF_ACT16 (act16_test.h): the KDA layer with 16-bit codes at the chosen
  // sites, row by row on the slow exact kernels.  A site that is not chosen
  // gets the entry's 8-bit code and kernel.
  __attribute__((noinline)) void kimi_layer16(int ki, ZRow* r, int n) {
    using namespace act16;
    KimiArenas& L = M.kimi[ki];
    const int am = g_mask;
    alignas(32) int16_t a16[D];
    alignas(32) int16_t g16[DH];
    alignas(32) float y64[DH];
    alignas(32) float y[D];
    for (int j = 0; j < n; j++) {
      ZRow& w = r[j];
      rms_norm192(w.x, w.xn);
      const QSite* const s3[3] = {&L.qp, &L.kp, &L.vp};
      float* const o3[3] = {w.a, w.b, w.c};
      for (int k = 0; k < 3; k++) {
        if (am & KQKV) {
          quant(w.xn, s3[k]->s_act, a16, D);
          dense(s3[k]->m, a16, K256, o3[k]);
        } else {
          quant192_u8(w.xn, s3[k]->s_act, w.q5);
          qgemv_f32(s3[k]->m, w.q5, o3[k]);
        }
      }
      const QSite* const up[2] = {&L.fg_up, &L.og_up};
      const QSite* const dn[2] = {&L.fg_down, &L.og_down};
      float* const go[2] = {w.graw, w.og192};
      for (int k = 0; k < 2; k++) {
        if (!(am & (KGUP | KGDN))) {
          quant192_u8(w.xn, up[k]->s_act, w.q5);
          qgemv_quant_bias(up[k]->m, w.q5, dn[k]->s_act, w.q64b);
          qgemv_f32(dn[k]->m, w.q64b, go[k]);
          continue;
        }
        float mult = 1.0f;
        if (am & KGUP) {
          quant(w.xn, up[k]->s_act, a16, D);
          mult = K256;
        } else {
          quant192_u8(w.xn, up[k]->s_act, w.q5);
          from_u8(w.q5, a16, D);
        }
        dense(up[k]->m, a16, mult, y64);
        if (am & KGDN) {
          quant(y64, dn[k]->s_act, g16, DH);
          dense(dn[k]->m, g16, K256, go[k]);
        } else {
          quant_bias_u8(y64, dn[k]->s_act, w.q64b, DH);
          qgemv_f32(dn[k]->m, w.q64b, go[k]);
        }
      }
      for (int h = 0; h < NH; h++) w.braw[h] = dot192(w.xn, L.beta_w + size_t(h) * D);
      kda_layer_step(L.kw, kst[ki], w.a, w.b, w.c, w.graw, w.og192, w.braw, w.gn);
      if (am & KOP) {
        quant(w.gn, L.op.s_act, a16, D);
        dense(L.op.m, a16, K256, y);
        for (int i = 0; i < D; i++) w.ar_f[i] += y[i];
      } else {
        quant192_u8(w.gn, L.op.s_act, w.q5);
        qgemv_add(L.op.m, w.q5, w.ar_f);
      }
    }
  }
  // the window layer likewise; the per-head q/k/v codes behind RoPE stay int8
  __attribute__((noinline)) void van_layer16(int vi, ZRow* r, int n, bool kv_only, int64_t t0) {
    using namespace act16;
    VanArenas& L = M.van[vi];
    const int am = g_mask;
    alignas(32) int16_t a16[D];
    alignas(32) float y[D];
    for (int j = 0; j < n; j++) {
      ZRow& w = r[j];
      rms_norm192(w.x, w.xn);
      const QSite* const s3[3] = {&L.qp, &L.kp, &L.vp};
      float* const o3[3] = {w.a, w.b, w.c};
      for (int k = kv_only ? 1 : 0; k < 3; k++) {
        if (am & VQKV) {
          quant(w.xn, s3[k]->s_act, a16, D);
          dense(s3[k]->m, a16, K256, o3[k]);
        } else {
          quant192_u8(w.xn, s3[k]->s_act, w.q5 + k * D);
          qgemv_f32(s3[k]->m, w.q5 + k * D, o3[k]);
        }
      }
      van_row(vi, w, t0 + j, kv_only);   // leaves the attention output in `pre`, its 8-bit code in w.q5
      if (kv_only) continue;
      if (am & VOP) {
        quant(pre, L.op.s_act, a16, D);
        dense(L.op.m, a16, K256, y);
        for (int i = 0; i < D; i++) w.ar_f[i] += y[i];
      } else {
        qgemv_add(L.op.m, w.q5, w.ar_f);
      }
    }
  }
  // the MLP likewise.  The 8-bit relu^2 code (q768, idx768, nnz) is always
  // written: the fast-weight head reads it, whatever the down matmul reads.
  __attribute__((noinline)) void mlp_block16(int l, ZRow* r, int n) {
    using namespace act16;
    MlpArenas& mm = M.mlp[l];
    const int am = g_mask;
    const int rows = mm.up.m.rows_padded;
    alignas(32) int16_t a16[D];
    alignas(32) int16_t c16[DMLP];
    alignas(64) float y[DMLP];
    for (int j = 0; j < n; j++) {
      ZRow& w = r[j];
      float mult = 1.0f;
      if (am & MLPUP) {
        rms_norm192(w.x, w.xn);
        quant(w.xn, mm.up.s_act, a16, D);
        mult = K256;
      } else {
        rms_norm_quant192(w.x, w.xn, mm.up.s_act, w.q5);
        from_u8(w.q5, a16, D);
      }
      dense(mm.up.m, a16, mult, y);
      w.nnz = relu2_u8(y, mm.down_s_act, w.q768, w.idx768, rows);
      if (am & MLPDN) {
        relu2_16(y, mm.down_s_act, c16, rows);
        sparse(mm.down, c16, mm.down.d_in, g_dn_mult, w.ar_f, true);
        {   // for the time estimate: how many columns a sparse 16-bit kernel would visit
          static unsigned long long s_calls, s_n8, s_n16, s_nhi;
          int n16 = 0, nhi = 0;
          for (int i = 0; i < rows; i++) {
            n16 += c16[i] != 0;
            nhi += c16[i] >= 256;
          }
          s_calls++;
          s_n8 += static_cast<unsigned long long>(w.nnz);
          s_n16 += static_cast<unsigned long long>(n16);
          s_nhi += static_cast<unsigned long long>(nhi);
          if ((s_calls & ((1ull << 22) - 1)) == 0)
            std::fprintf(stderr, "neif_act16: mlp.down calls %llu, nonzero codes per call: 8-bit %.1f, 16-bit %.1f, 16-bit >= 256: %.1f\n",
                         s_calls, double(s_n8) / double(s_calls), double(s_n16) / double(s_calls),
                         double(s_nhi) / double(s_calls));
        }
      } else if (w.nnz > static_cast<int>(QMAT_SPARSE_DENSITY_THRESHOLD * mm.down.d_in)) {
        qsparse4_dense_add(mm.down, w.q768, w.ar_f);
      } else {
        qsparse4_add(mm.down, w.q768, w.idx768, w.nnz, w.ar_f);
      }
    }
    if (fw_gel && l >= NL - FWK) fw_apply(l - (NL - FWK), r->nnz, mm.down_s_act);
  }
  // NEIF_ACT16, fast path of mlpdn: the entry's mlp.up kernels up to the float
  // row, then the wide relu^2 code and the two-plane sparse kernel
  // (act16_test.h).  The 8-bit code is computed only where it is read: on a
  // read row in the last FWK layers, for the fast-weight head.
  __attribute__((noinline)) void mlp_block_fast(int l, ZRow* r, int n) {
    using namespace act16;
    MlpArenas& mm = M.mlp[l];
    const int rows = mm.up.m.rows_padded;
    const bool fw = fw_gel && l >= NL - FWK;
    alignas(64) float y[ZB][DMLP];
    alignas(32) uint16_t c[DMLP];
    uint16_t il[DMLP + 32], ih[DMLP + 32];
    {
      PSCOPE(PS_EL_NQ_MLP);
      for (int j = 0; j < n; j++) rms_norm_quant192(r[j].x, r[j].xn, mm.up.s_act, r[j].q5);
    }
    {
      PSCOPE(PS_MM_MLP_UP);
      if (n == 1)
        qgemv_f32(mm.up.m, r->q5, y[0]);
      else
        qgemm(mm.up.m, r->q5, ZS, y[0], static_cast<ptrdiff_t>(sizeof(y[0])), n, false);
    }
    static unsigned long long s_calls, s_nl, s_nh;
    for (int j = 0; j < n; j++) {
      ZRow& w = r[j];
      int nl, nh;
      int32_t sum;
      {
        PSCOPE(PS_MM_MLP_UP);   // the quantizers belong to the up site, as in the entry's fused kernel
        if (fw) w.nnz = relu2_u8_fast(y[j], mm.down_s_act, w.q768, w.idx768, rows);
        relu2_wide(y[j], mm.down_s_act, rows, c, il, &nl, ih, &nh, &sum);
      }
      {
        PSCOPE(PS_MM_MLP_DOWN);
        sparse_fast(mm.down, c, il, nl, ih, nh, sum, w.ar_f);
      }
      s_nl += static_cast<unsigned long long>(nl);
      s_nh += static_cast<unsigned long long>(nh);
      if ((++s_calls & ((1ull << 22) - 1)) == 0)
        std::fprintf(stderr, "neif_act16: mlp.down calls %llu, columns per call: low-byte plane %.1f, high-byte plane %.1f\n",
                     s_calls, double(s_nl) / double(s_calls), double(s_nh) / double(s_calls));
    }
    if (fw) fw_apply(l - (NL - FWK), r->nnz, mm.down_s_act);
  }
#endif

  void kimi_layer(int ki, ZRow* r, int n) {
    KimiArenas& L = M.kimi[ki];
#ifdef NEIF_TEST
    if (!act16::g_fast && (act16::g_mask & (act16::KQKV | act16::KGUP | act16::KGDN | act16::KOP))) {
      kimi_layer16(ki, r, n);
      return;
    }
#endif
    {
      PSCOPE(PS_EL_NQ_KIMI);
      const float s5[5] = {L.qp.s_act, L.kp.s_act, L.vp.s_act, L.fg_up.s_act,
                           L.og_up.s_act};
      for (int j = 0; j < n; j++) rms_norm_quant192_multi(r[j].x, r[j].xn, s5, 5, r[j].q5);
    }
    {
      PSCOPE(PS_MM_KQKV);
      mm_f32(L.qp.m, r->q5, r->a, n);
      mm_f32(L.kp.m, r->q5 + D, r->b, n);
      mm_f32(L.vp.m, r->q5 + 2 * D, r->c, n);
    }
    {
      PSCOPE(PS_MM_GUP);
      mm_quant_bias(L.fg_up.m, r->q5 + 3 * D, L.fg_down.s_act, r->q64b, n);
    }
    {
      PSCOPE(PS_MM_GDN);
      mm_f32(L.fg_down.m, r->q64b, r->graw, n);
    }
    {
      PSCOPE(PS_MM_GUP);
      mm_quant_bias(L.og_up.m, r->q5 + 4 * D, L.og_down.s_act, r->q64b, n);
    }
    {
      PSCOPE(PS_MM_GDN);
      mm_f32(L.og_down.m, r->q64b, r->og192, n);
    }
    {
      PSCOPE(PS_KD_BETA);
      for (int j = 0; j < n; j++)
        for (int h = 0; h < NH; h++)
          r[j].braw[h] = dot192(r[j].xn, L.beta_w + size_t(h) * D);
    }
    {
      PSCOPE(PS_KD_STEP);
#if defined(NEIF_TEST) && defined(NEIF_MT)
      // NEIF_MT: where the committed head states of this layer live once a
      // speculative step was taken (kS_cur), and the shadow engine's step from
      // the committed states into the other buffer, without moving the position
      if (spec_mode) {
        const float* in = sp_main->kS(ki);
        float* out = in == &kst[ki].S[0][0] ? sp_main->kS_alt + static_cast<size_t>(ki) * kSN : &kst[ki].S[0][0];
        sp_out[ki] = out;
        kda_layer_step_io(L.kw, kst[ki], in, out, false, r->a, r->b, r->c, r->graw, r->og192, r->braw, r->gn);
      } else if (kS_cur[ki]) {
        for (int j = 0; j < n; j++)
          kda_layer_step_io(L.kw, kst[ki], kS_cur[ki], kS_cur[ki], true, r[j].a, r[j].b, r[j].c, r[j].graw,
                            r[j].og192, r[j].braw, r[j].gn);
      } else
#endif
      for (int j = 0; j < n; j++)   // the recurrence: rows in stream order
        kda_layer_step(L.kw, kst[ki], r[j].a, r[j].b, r[j].c, r[j].graw, r[j].og192, r[j].braw,
                       r[j].gn);
    }
#ifdef NEIF_TEST
    if (act16::g_fast & act16::KOP) {   // NEIF_ACT16: 16-bit code, fast kernel
      PSCOPE(PS_MM_KOP);
      for (int j = 0; j < n; j++) act16::op_fast(L.op.m, L.op.s_act, r[j].gn, r[j].ar_f);
      return;
    }
#endif
    {
      PSCOPE(PS_EL_OQ);
      for (int j = 0; j < n; j++) quant192_u8(r[j].gn, L.op.s_act, r[j].q5);
    }
    {
      PSCOPE(PS_MM_KOP);
      mm_add(L.op.m, r->q5, r->ar_f, n);
    }
  }

  // window layer vi for the rows r[0, n) at positions t0 + j.  kv_only: an
  // unread row in the last layer, which needs only its K and V in the ring
  // (everything after the insert only feeds the output head).
  void van_layer(int vi, ZRow* r, int n, bool kv_only, int64_t t0) {
    VanArenas& L = M.van[vi];
#ifdef NEIF_TEST
    if (!act16::g_fast && (act16::g_mask & (act16::VQKV | act16::VOP))) {
      van_layer16(vi, r, n, kv_only, t0);
      return;
    }
#endif
    {
      PSCOPE(PS_EL_NQ_VAN);
      const float s3[3] = {L.qp.s_act, L.kp.s_act, L.vp.s_act};
      for (int j = 0; j < n; j++)
        rms_norm_quant192_x3(r[j].x, r[j].xn, s3, r[j].q5, r[j].q5 + D, r[j].q5 + 2 * D);
    }
    {
      PSCOPE(PS_MM_VQKV);
      if (!kv_only) mm_f32(L.qp.m, r->q5, r->a, n);
      mm_f32(L.kp.m, r->q5 + D, r->b, n);
      mm_f32(L.vp.m, r->q5 + 2 * D, r->c, n);
    }
    for (int j = 0; j < n; j++)   // KV ring and window scan: rows in stream order
      van_row(vi, r[j], t0 + j, kv_only);
#ifdef NEIF_TEST
    if (act16::g_fast & act16::VOP) return;   // NEIF_ACT16: van_row has added the output projection
#endif
    if (!kv_only) {
      PSCOPE(PS_MM_VOP);
      mm_add(L.op.m, r->q5, r->ar_f, n);
    }
  }

  // the part of a window layer that runs row by row: head prep, KV insert
  // and window scan of row w at position tj
  __attribute__((noinline)) void van_row(int vi, ZRow& w, int64_t tj, bool kv_only) {
    VanArenas& L = M.van[vi];
    float* qf = w.a;
    float* kf = w.b;
    float* vf = w.c;
    kv_expand(L, w);
    {
      PSCOPE(PS_EL_HEADPREP);
      if (!kv_only)
        for (int h = 0; h < NH; h++) rms_norm64(qf + h * DH, qf + h * DH);
      for (int h = 0; h < KVP; h++) rms_norm64(kf + h * DH, kf + h * DH);
      const float *sp, *cp;
      zb_rope(rope_off + tj, sp, cp);
      if (!kv_only) rope_apply_192(qf, sp, cp);
      if constexpr (KVP == 1) {
        rope_apply_64(kf, sp, cp);
        if (!kv_only)
          for (int h = 0; h < NH; h++) quant64_i8(qf + h * DH, L.sq[h], qq + h * DH);
        quant64_i8(kf, L.sk[0], kk);
        quant64_i8(vf, L.sv[0], vv);
      } else {
        rope_apply_192(kf, sp, cp);
        for (int h = 0; h < NH; h++) {
          if (!kv_only) quant64_i8(qf + h * DH, L.sq[h], qq + h * DH);
          quant64_i8(kf + h * DH, L.sk[h], kk + h * DH);
          quant64_i8(vf + h * DH, L.sv[h], vv + h * DH);
        }
      }
    }
    const int slot = static_cast<int>(tj % WIN);
    const int64_t nn = tj + 1;
    if (!((3 & 2) && 1) && kind == AttnKind::KVF32) {
      {
        PSCOPE(PS_AT_INSERT);
        attn_kv_insert(kvf[vi], slot, kk, vv);
      }
      if (kv_only) return;
      PSCOPE(PS_AT_STEP);
      if (nn < WIN)
        attn_step_var(kvf[vi], qq, L.coef, L.sv, static_cast<int>(nn), pre, 0.0f);
      else
        attn_step_fixed(kvf[vi], qq, L.coef, L.sv, pre, 0.0f);
    } else {
      {
        PSCOPE(PS_AT_INSERT);
        attn_kv_insert(kvi[vi], slot, kk, vv);
      }
      if (kv_only) return;
      PSCOPE(PS_AT_STEP);
#if defined(NEIF_TEST) && defined(NEIF_MT)
      // NEIF_MT: the window split over the threads of the pool (the same bytes in `pre`)
      if (neif_mt::On() && nn >= 512)
        attn_step_mt(kvi[vi], qq, L.coef, L.sv, static_cast<int>(nn < WIN ? nn : WIN), pre);
      else
#endif
      if (nn < WIN)
        attn_step_var(kvi[vi], qq, L.coef, L.sv, static_cast<int>(nn), pre, 0.0f);
      else
        attn_step_fixed(kvi[vi], qq, L.coef, L.sv, pre, 0.0f);
    }
#ifdef NEIF_TEST
    if (act16::g_fast & act16::VOP) {   // NEIF_ACT16: 16-bit code, fast kernel
      PSCOPE(PS_MM_VOP);
      act16::op_fast(L.op.m, L.op.s_act, pre, w.ar_f);
      return;
    }
#endif
    PSCOPE(PS_EL_OQ);
    quant192_u8(pre, L.op.s_act, w.q5);
  }

#if defined(NEIF_TEST) && defined(NEIF_MT)
  // NEIF_MT (src/neif_mt.h): the MLP of one row as one region of the pool.
  // Share s takes a range of the up projection's row groups: their dot
  // products and relu^2 codes (written into the row's q768), the list of its
  // non-zero codes, and the int32 sums of the down columns of those codes.
  // The caller then adds the sums of the shares (integers: exact for any
  // split), does the one float step of the down projection, and puts the
  // lists one behind the other (the list qgemv_relu2q writes).
  struct MtMlp {
    const MlpArenas* mm;
    ZRow* r;
    float* y;
    int nth;
    struct alignas(64) Share {
      int32_t acc[192];
      int32_t corr;
      int nnz;
      uint16_t idx[DMLP + 64];
    } sh[neif_mt::kMax];
  };
  static void mt_mlp_share(int s, void* a) {
    MtMlp& j = *static_cast<MtMlp*>(a);
    const QDense& up = j.mm->up.m;
    const int ng = up.ngroups;
    // borders at multiples of 8 groups (64 rows: whole cache lines of q768, whole 32-row blocks of the list)
    const int g0 = (ng * s / j.nth) & ~7;
    const int g1 = s + 1 == j.nth ? ng : (ng * (s + 1) / j.nth) & ~7;
    MtMlp::Share& sh = j.sh[s];
    sh.nnz = 0;
    if (g1 > g0) {
      qgemv_relu2q_part(up, j.r->q5, j.mm->down_s_act, j.r->q768, j.y, g0, g1);
      sh.nnz = relu2q_idx_part(j.r->q768, 8 * g0, s + 1 == j.nth ? up.rows_padded : 8 * g1, sh.idx);
    }
    const uint16_t fill = sh.nnz ? sh.idx[sh.nnz - 1] : 0;   // behind the list: in-range entries for the prefetch
    for (int k = sh.nnz; k < sh.nnz + 12; k++) sh.idx[k] = fill;
    sh.corr = qsparse4_accum_part(j.mm->down, j.r->q768, sh.idx, sh.nnz, sh.acc);
  }
  void mlp_block_mt(int l, ZRow* r) {
    MlpArenas& mm = M.mlp[l];
    {
      PSCOPE(PS_EL_NQ_MLP);
      rms_norm_quant192(r->x, r->xn, mm.up.s_act, r->q5);
    }
    static MtMlp j;   // one MLP at a time
    alignas(64) float y[DMLP];
    {
      PSCOPE(PS_MM_MLP_UP);
      j.mm = &mm;
      j.r = r;
      j.y = y;
      j.nth = neif_mt::g_n;
      neif_mt::Run(mt_mlp_share, &j);
    }
    {
      PSCOPE(PS_MM_MLP_DOWN);
      alignas(64) int32_t acc[192];
      int32_t corr = 0;
      int nnz = 0;
      std::memcpy(acc, j.sh[0].acc, sizeof acc);
      for (int s = 0; s < j.nth; s++) {
        if (s)
          for (int i = 0; i < 192; i++) acc[i] += j.sh[s].acc[i];
        corr += j.sh[s].corr;
        std::memcpy(r->idx768 + nnz, j.sh[s].idx, sizeof(uint16_t) * static_cast<size_t>(j.sh[s].nnz));
        nnz += j.sh[s].nnz;
      }
      r->nnz = nnz;
      qsparse4_epi_add(mm.down, acc, corr, r->ar_f);
    }
    if (spec_mode && l >= NL - FWK) spec_save_fw(l, r);
    if (fw_gel && l >= NL - FWK)
      fw_apply(l - (NL - FWK), r->nnz, mm.down_s_act);
  }
#endif

  // the MLP of layer l for the rows r[0, n)
  void mlp_block(int l, ZRow* r, int n) {
    MlpArenas& mm = M.mlp[l];
#ifdef NEIF_TEST
#ifdef NEIF_MT
    // NEIF_MT: one row, the entry's 8-bit path: the MLP on the pool
    if (neif_mt::On() && n == 1 && !act16::g_fast && !(act16::g_mask & (act16::MLPUP | act16::MLPDN))) {
      mlp_block_mt(l, r);
      return;
    }
#endif
    if (act16::g_fast & act16::MLPDN) {
      mlp_block_fast(l, r, n);
      return;
    }
    if (!act16::g_fast && (act16::g_mask & (act16::MLPUP | act16::MLPDN))) {
      mlp_block16(l, r, n);
      return;
    }
#endif
    {
      PSCOPE(PS_EL_NQ_MLP);
      for (int j = 0; j < n; j++) rms_norm_quant192(r[j].x, r[j].xn, mm.up.s_act, r[j].q5);
    }
    {
      PSCOPE(PS_MM_MLP_UP);
      if (n == 1)
        r->nnz = qgemv_relu2q(mm.up.m, r->q5, mm.down_s_act, r->q768, r->idx768);
      else
        qgemm_relu2q(mm.up.m, r->q5, ZS, mm.down_s_act, r->q768, r->idx768, &r->nnz, ZS, n);
    }
    {
      PSCOPE(PS_MM_MLP_DOWN);
      for (int j = 0; j < n; j++) {
        ZRow& w = r[j];
        if (w.nnz > static_cast<int>(QMAT_SPARSE_DENSITY_THRESHOLD * mm.down.d_in))
          qsparse4_dense_add(mm.down, w.q768, w.ar_f);
        else
          qsparse4_add(mm.down, w.q768, w.idx768, w.nnz, w.ar_f);
      }
    }
#if defined(NEIF_TEST) && defined(NEIF_MT)
    if (spec_mode && l >= NL - FWK) spec_save_fw(l, r);
#endif
    // The relu^2 code of the last FWK MLPs, on read rows only (fw_gel; then
    // r is this object, n = 1, and fw_apply reads its q768 and idx768).
    if (fw_gel && l >= NL - FWK)
      fw_apply(l - (NL - FWK), r->nnz, mm.down_s_act);
  }

  // the embedding of row r (token r.tok, prior row r.prior_f32, side channels
  // pk) and source 0 of its depth mix
  __attribute__((noinline)) void zb_embed(ZRow& r) {
    const float* tok_row = M.tok_table + size_t(r.tok) * D;
    if (M.paket.present) tok_row = paket_token_row(r.tok);
    int nnzp;
#ifdef NEIF_TEST
    if (act16::g_mask & act16::PRIOR) {   // NEIF_ACT16: 16-bit prior code
      alignas(32) int16_t p16[208];
      act16::quant(r.prior_f32, M.prior_s_act, p16, 208);
      act16::sparse(M.prior, p16, V, act16::K256, yb, false);
    } else {
#endif
    {
      PSCOPE(PS_EL_PRIORQ);
      prior_quant_raw(r.prior_f32, M.prior_s_act, q8p);  // raw u8; q8p[208..224)=0
      nnzp = qsparse_make_idx(q8p, V, idxp);
      const uint16_t fillv = nnzp ? idxp[nnzp - 1] : 0;
      for (int k = nnzp; k < nnzp + 8; k++) idxp[k] = fillv;
    }
    {
      PSCOPE(PS_MM_PRIOR);
      if (nnzp > PRIOR_DENSE_NNZ)
        qsparse4_dense_f32(M.prior, q8p, yb);
      else
        qsparse4_f32(M.prior, q8p, idxp, nnzp, yb);
    }
#ifdef NEIF_TEST
    }
#endif
    {
      PSCOPE(PS_EL_EMBED);
      embed_combine192(tok_row, yb, r.x);  // x0 = tok_row + rms_norm(prior_y)
    }
    r.ar_qnext = 0;
    r.ar_has_part = false;
    r.ar_part_dot_q = -1;
    r.push_x0(M, r.x);                     // source 0 = x0
  }

  // AttnRes replace mode (after model_opt.cpp of cmix-neif-beehive).
  // No U-Net skips, no residual/token coefficients: every sub-layer reads
  // h = depth_mix(x0, completed block sums, partial) and its output f goes
  // into the partial sum of the current block.  Layers [l0, l1) of the rows
  // r[0, n) at positions t0 + j; kv_last: unread rows, whose last layer only
  // writes K and V.
  __attribute__((noinline)) void zb_layers(ZRow* r, int n, bool kv_last, int l0, int l1,
                                           int64_t t0) {
    for (int l = l0; l < l1; l++) {
#if defined(NEIF_TEST) && defined(NEIF_MT)
      if (spec_mode && sp_main->sp_abort.load(std::memory_order_relaxed)) return;   // the guess is known to be wrong
#endif
      {
        PSCOPE(PS_EL_RESID);
        for (int j = 0; j < n; j++) {
          if (l % ARB == 0 && r[j].ar_has_part)   // block boundary: freeze its sum
            r[j].push_part(M);
          r[j].read(M, r[j].x);                  // x = h for the mixer (q 2l, M.ar_q_mix[l])
        }
      }
      if (kv_last && l == NL - 1) {              // unread rows: K/V of the last layer only
        van_layer(M.layer2van[l], r, n, true, t0);
        return;
      }
      for (int j = 0; j < n; j++) std::memset(r[j].ar_f, 0, sizeof(r[j].ar_f));
      if (OptModel::KIMI_L[l])
        kimi_layer(M.layer2kimi[l], r, n);
      else
        van_layer(M.layer2van[l], r, n, false, t0);
      {
        PSCOPE(PS_EL_RESID);
        for (int j = 0; j < n; j++) {
          r[j].take_f(M, 2 * l + 1);
          r[j].read(M, r[j].x);                  // x = h for the MLP (q 2l+1, M.ar_q_mlp[l])
        }
      }
      // semantic-cache key: the MLP read of the last layer (q 2 NL - 1);
      // only a read row gets here
      if (l == NL - 1) std::memcpy(sc_l2_stage, r->x, sizeof(float) * D);
      for (int j = 0; j < n; j++) std::memset(r[j].ar_f, 0, sizeof(r[j].ar_f));
      mlp_block(l, r, n);
      {
        PSCOPE(PS_EL_RESID);
        // next read of this partial: the final read after the last layer,
        // else layer l+1's mixer read unless a block boundary freezes it first
        const int nq = l == NL - 1 ? 2 * NL : ((l + 1) % ARB != 0 ? 2 * l + 2 : -1);
        for (int j = 0; j < n; j++) r[j].take_f(M, nq);
      }
    }
  }

#ifdef NEIF_TEST
  // All layers of the embedded rows r[0, n).  In a pipeline (src/neif_test.h)
  // the network may be cut between two layers: the stage with the lower
  // layers sends each row's depth-mix state (ArRow: all a later layer reads of
  // the layers in front), the stage with the upper layers starts from it.
  // Every layer does the same arithmetic on the same values as in one process.
  __attribute__((noinline)) void zb_stage(ZRow* r, int n, bool kv_last) {
    const bool lo = (neif_test::g_own & neif_test::kOwnTfLo) != 0;
    const bool hi = (neif_test::g_own & neif_test::kOwnTfHi) != 0;
    if (lo != hi) {
      const size_t ar_bytes = (size_t)((const char*)(r->ar_f + D) - (const char*)r->ar_x0);
      if (lo) {
        zb_layers(r, n, kv_last, 0, neif_test::g_tf_split, t);
        for (int j = 0; j < n; j++) neif_test::Put('A', r[j].ar_x0, ar_bytes);
      } else {
        for (int j = 0; j < n; j++) neif_test::Get('A', r[j].ar_x0, ar_bytes);
        zb_layers(r, n, kv_last, neif_test::g_tf_split, NL, t);
      }
      return;
    }
    zb_layers(r, n, kv_last, 0, NL, t);
  }
#endif

  // The queue of unread rows (ZB rows): step() with an unread row only queues
  // it; the queue runs when the next read row arrives, at begin_article() or
  // when it is full.  What the rows leave (KV rings, KDA states, t) is written
  // before anything reads it; xn holds the last row's, as after an unread step.
  static constexpr int ZB = 16;
  ZRow zr[ZB];
  int zn;

  // The two functions below only call other functions; compiled for size.
  __attribute__((minsize)) void zb_push(uint8_t token) {
    ZRow& r = zr[zn];
    r.tok = token;
    std::memcpy(r.prior_f32, prior_f32, sizeof(r.prior_f32));
    r.pk = pk;
    if (++zn == ZB) zb_run();
  }

  __attribute__((minsize, noinline)) void zb_run() {
    const int n = zn;
    if (n == 0) return;
    zn = 0;
#ifdef FX2_PROF
    PSCOPE(PS_ZB_RUN);
#endif
    fw_gel = false;   // unread rows capture no MLP code
    const PaketChannels pk_keep = pk;   // paket_token_row() reads pk
#ifdef NEIF_TEST
    if (neif_test::g_own & neif_test::kOwnTfLo)   // pipeline: the embedding belongs to the lower layers
#endif
    for (int j = 0; j < n; j++) {
      pk = zr[j].pk;
      zb_embed(zr[j]);
    }
    pk = pk_keep;
#ifdef NEIF_TEST
    zb_stage(zr, n, true);
#else
    zb_layers(zr, n, true, 0, NL, t);
#endif
    std::memcpy(xn, zr[n - 1].xn, sizeof(float) * D);
    t += n;
  }

  // Only the FW_TOPK largest codes of a block take part
  // (forward and update; ties by index), in ascending index order.
  const uint16_t* fw_topk(int& nnz) {
    constexpr int K = FW_TOPK;
    if (nnz <= K) return idx768;
    int hist[256] = {};
    for (int k = 0; k < nnz; k++) hist[q768[idx768[k]]]++;
    int cnt = 0, t = 255;
    while (t > 0 && cnt + hist[t] <= K) cnt += hist[t--];
    int room = K - cnt, m = 0;
    for (int k = 0; k < nnz; k++) {
      const int v = q768[idx768[k]];
      if (v > t || (v == t && room-- > 0)) fw_sel[m++] = idx768[k];
    }
    nnz = m;
    return fw_sel;
  }

  // One column's pending dW step, col -= c * gx.
  __attribute__((noinline)) void fw_col_step(float* col, float c, const float* gx) {
    const __m256 vc = _mm256_set1_ps(c);
    for (int j = 0; j < D; j += 8)
      _mm256_store_ps(col + j, _mm256_sub_ps(_mm256_load_ps(col + j),
                                             _mm256_mul_ps(vc, _mm256_load_ps(gx + j))));
  }

  // after ex-girlfriend 1.0 (Ryan Landay, GPL-3.0-only), fast-weight
  // forward of block b.  h_k = q768[i_k] * s over the nonzero codes of this
  // MLP; their ascending index list is the one the up
  // projection just wrote (idx768, nnz; rows past d_out are zero there),
  // cut to the FW_TOPK largest (fw_topk).
  // x stays untouched, the delta goes to fw_dx; block 0 runs first in every
  // read row and starts it from zero.  64 columns at a time with the
  // accumulators in registers; a column with a pending dW step gets it first.
  __attribute__((noinline)) void fw_apply(int b, int nnz, float s) {
    const uint16_t* const lst = fw_topk(nnz);
    float* const h = fw_h[b];
    for (int k = 0; k < nnz; k++) h[k] = float(q768[lst[k]]) * s;
    float* const dWb = fw_dW + size_t(b) * DMLP * D;
    int32_t* const pt = fw_pt[b];
    const float* const pc = fw_pc[b];
    for (int jc = 0; jc < D; jc += 64) {
      __m256 a[8];
      for (int u = 0; u < 8; u++)
        a[u] = b == 0 ? _mm256_setzero_ps() : _mm256_load_ps(fw_dx + jc + 8 * u);
      for (int k = 0; k < nnz; k++) {
        const int j = lst[k];
        float* col = dWb + size_t(j) * D + jc;
        const __m256 vh = _mm256_set1_ps(h[k]);
        const int32_t t = pt[j];
        if (t >= 0) {   // the pending step, then the read
          const float* g = fw_gxr[t & (FWR - 1)] + jc;
          const __m256 vc = _mm256_set1_ps(pc[j]);
          for (int u = 0; u < 8; u++) {
            const __m256 w = _mm256_sub_ps(_mm256_load_ps(col + 8 * u),
                                           _mm256_mul_ps(vc, _mm256_load_ps(g + 8 * u)));
            _mm256_store_ps(col + 8 * u, w);
            a[u] = _mm256_add_ps(a[u], _mm256_mul_ps(w, vh));
          }
        } else {
          for (int u = 0; u < 8; u++)
            a[u] = _mm256_add_ps(a[u], _mm256_mul_ps(_mm256_load_ps(col + 8 * u), vh));
        }
      }
      for (int u = 0; u < 8; u++) _mm256_store_ps(fw_dx + jc + 8 * u, a[u]);
    }
    for (int k = 0; k < nnz; k++) pt[lst[k]] = -1;
    std::memcpy(fw_idx[b], lst, sizeof(uint16_t) * size_t(nnz));
    fw_nnz[b] = nnz;
  }

  // The second head from x + dW h into its own buffers; the static head
  // (xn, q5, logits, probs208, last_hidden()) stays untouched.
  __attribute__((noinline)) void fw_head() {
    for (int i = 0; i < D; i++) fw_x2[i] = x[i] + fw_dx[i];
#ifdef NEIF_TEST
    const bool a16 = (act16::g_mask & act16::FWUNEMB) != 0;   // NEIF_ACT16
    if (a16) rms_norm192(fw_x2, fw_xn); else
#endif
    rms_norm_quant192(fw_x2, fw_xn, M.unembed.s_act, fw_q);
    float sx = 0.0f, sn = 0.0f;
    for (int i = 0; i < D; i++) {
      sx += fw_x2[i] * fw_x2[i];
      sn += fw_xn[i] * fw_xn[i];
    }
    fw_denom = sx > 0.0f ? std::sqrt(sn / sx) : 0.0f;
#ifdef NEIF_TEST
    if (a16) {
      alignas(32) int16_t q16[D];
      act16::quant(fw_xn, M.unembed.s_act, q16, D);
      act16::dense(M.unembed.m, q16, act16::K256, fw_logits);
    } else
#endif
    qgemv_f32(M.unembed.m, fw_q, fw_logits);
    alignas(64) float raw[208];
    std::memcpy(raw, fw_logits, sizeof(raw));
    head_softcap_softmax(fw_logits, fw_probs208);  // fw_logits -> 15 tanh(l/15)
    head_plm_softcap_softmax(raw, prior_f32, dot192(M.plm_w, fw_xn), fw_probs208);
  }

  // Backward and update on the last read row's second-head prediction with
  // target y: cross-entropy gradient on the soft-capped logits, back through
  // the frozen unembedding (float copy) and the RMS norm to x, then one NLMS
  // step on all dW_b with per-feature RMS preconditioning and the step
  // schedule kappa_t = 1.5e-3 + 4.5e-3 / (1 + t / 8000).
  __attribute__((noinline)) void fw_update(uint8_t y) {
    // The ring slot of this update; steps of update u - FWR
    // that are still pending need its gx, so they go now.
    const int32_t u = fw_u++;
    float* const gx = fw_gxr[u & (FWR - 1)];
    if (u >= FWR) {
      const __m256i vo = _mm256_set1_epi32(u - FWR);
      for (int b = 0; b < FWK; b++)
        for (int j0 = 0; j0 < DMLP; j0 += 8) {
          unsigned mk = unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(
              _mm256_load_si256(reinterpret_cast<const __m256i*>(&fw_pt[b][j0])), vo))));
          while (mk) {
            const int j = j0 + __builtin_ctz(mk);
            mk &= mk - 1;
            fw_col_step(fw_dW + (size_t(b) * DMLP + j) * D, fw_pc[b][j], gx);
            fw_pt[b][j] = -1;
          }
        }
    }
    alignas(64) float dl[V];
    for (int o = 0; o < V; o++) {
      const float d = fw_probs208[o] - (o == int(y) ? 1.0f : 0.0f);
      const float c = fw_logits[o] * (1.0f / 15.0f);
      dl[o] = d * (1.0f - c * c);
    }
    // per-feature RMS preconditioning with per-block statistics (msd = mean
    // RMS over the block's DMLP features); one NLMS step over the joined code
    // of all blocks, joint denominator 1 + sum_b h_b . sh_b.
    // msd is renewed every FW_MSD-th update only (in between
    // the last one); then the square roots once per feature (vectorised,
    // exact), and the FWK sums of msd are independent chains, each in its own
    // order i = 0..DMLP-1, interleaved with the gxn loop below, which hides
    // their latency.
    const float fb = 999000.0e-6f;
    const bool msd_neu = fw_t % FW_MSD == 0;
    for (int b = 0; b < FWK; b++) {
      float* const v = fw_v[b];
      const uint16_t* const idx = fw_idx[b];
      const float* const h = fw_h[b];
      const int nnz = fw_nnz[b];
      for (int i = 0; i < DMLP; i++) v[i] *= fb;
      for (int k = 0; k < nnz; k++) v[idx[k]] += (1.0f - fb) * h[k] * h[k];
      if (msd_neu)
        for (int i = 0; i < DMLP; i++) fw_sq[b][i] = std::sqrt(v[i]);
    }
    float msd[FWK];
    for (int b = 0; b < FWK; b++) msd[b] = 0.0f;
    int ii = msd_neu ? 0 : DMLP;   // no renewal: the chains stay off
    // gxn = U^T dl, 64 columns at a time with the accumulators in registers;
    // every gxn[i] sums over o in the original order.
    static_assert(DMLP % 2 == 0 && (D / 64) * V * 2 >= DMLP, "msd chains fit the gxn loop");
    alignas(64) float gxn[D];
    const float* const U = M.unembed_f32;
    for (int jc = 0; jc < D; jc += 64) {
      __m256 a[8];
      for (int k = 0; k < 8; k++) a[k] = _mm256_setzero_ps();
      for (int o = 0; o < V; o++) {
        const float* r = U + size_t(o) * D + jc;
        const __m256 vg = _mm256_set1_ps(dl[o]);
        for (int k = 0; k < 8; k++)
          a[k] = _mm256_add_ps(a[k], _mm256_mul_ps(_mm256_loadu_ps(r + 8 * k), vg));
        if (ii < DMLP) {   // two steps of every msd chain
          for (int b = 0; b < FWK; b++) msd[b] += fw_sq[b][ii];
          for (int b = 0; b < FWK; b++) msd[b] += fw_sq[b][ii + 1];
          ii += 2;
        }
      }
      for (int k = 0; k < 8; k++) _mm256_store_ps(gxn + jc + 8 * k, a[k]);
    }
    float m = 0.0f;
    for (int i = 0; i < D; i++) m += gxn[i] * fw_xn[i];
    m *= (1.0f / float(D));
    for (int i = 0; i < D; i++) gx[i] = fw_denom * (gxn[i] - fw_xn[i] * m);
    const double kap =
        (1500.0e-6 + (6000.0e-6 - 1500.0e-6) / (1.0 + double(fw_t) / 8000.0))
        ;
    fw_t++;
    float hsh = 0.0f;
    for (int b = 0; b < FWK; b++) {
      if (msd_neu) fw_msd[b] = msd[b] * (1.0f / float(DMLP));
      const float ms = fw_msd[b];
      const float* const vv = fw_v[b];
      const uint16_t* const idx = fw_idx[b];
      const float* const h = fw_h[b];
      float* const sh = fw_sh[b];
      const int nnz = fw_nnz[b];
      for (int k = 0; k < nnz; k++) {
        // the root of the CURRENT v at this feature (msd may be older)
        const float sk = ms / (std::sqrt(vv[idx[k]]) + 1e-3f * ms + 1e-12f);
        sh[k] = sk * h[k];
        hsh += h[k] * sh[k];
      }
    }
    const float step = float(kap) / (1.0f + hsh);
    // the dW step itself stays pending (see the members)
    for (int b = 0; b < FWK; b++) {
      const uint16_t* const idx = fw_idx[b];
      const float* const sh = fw_sh[b];
      for (int k = 0; k < fw_nnz[b]; k++) {
        const int j = idx[k];
        if (fw_pt[b][j] >= 0)   // an update without a read row in between
          fw_col_step(fw_dW + (size_t(b) * DMLP + j) * D, fw_pc[b][j],
                      fw_gxr[fw_pt[b][j] & (FWR - 1)]);
        fw_pt[b][j] = u;
        fw_pc[b][j] = step * sh[k];
      }
    }
  }

  static_assert(PaketArenas::DIN_LAST == 32 && PaketArenas::DIN_MATCH == 32 && D % 8 == 0,
                "is written for d_in 32 and D a multiple of 8");
  // out of line: paket_token_row() calls it for both side channels
  __attribute__((noinline)) static void proj_add(const float* f, int d_in, const int8_t* q,
                       const float* fold, float s_act, float* acc_out) {
    (void)d_in;
    alignas(32) int16_t qa[32];
#ifdef NEIF_TEST
    const bool a16 = (act16::g_mask & act16::PAKET) != 0;   // NEIF_ACT16: 16-bit code, fold / 256
    if (a16) act16::quant(f, s_act, qa, 32); else
#endif
    for (int i = 0; i < 32; i++) {
      float v = f[i] / s_act;   // as below: true division, clamp, round half to even
      if (v < -128.0f) v = -128.0f;
      if (v > 127.0f) v = 127.0f;
      qa[i] = static_cast<int16_t>(std::nearbyintf(v));
    }
    const __m256i a0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(qa));
    const __m256i a1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(qa + 16));
    for (int o = 0; o < D; o += 8) {
      __m256i r[8];
      for (int k = 0; k < 8; k++) {
        const int8_t* w = q + size_t(o + k) * 32;
        const __m256i w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w)));
        const __m256i w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + 16)));
        r[k] = _mm256_add_epi32(_mm256_madd_epi16(a0, w0), _mm256_madd_epi16(a1, w1));
      }
      // lane-wise pair sums: h0123 = [r0..r3 low halves | r0..r3 high halves]
      const __m256i h0123 = _mm256_hadd_epi32(_mm256_hadd_epi32(r[0], r[1]), _mm256_hadd_epi32(r[2], r[3]));
      const __m256i h4567 = _mm256_hadd_epi32(_mm256_hadd_epi32(r[4], r[5]), _mm256_hadd_epi32(r[6], r[7]));
      const __m256i s = _mm256_add_epi32(_mm256_permute2x128_si256(h0123, h4567, 0x20),
                                         _mm256_permute2x128_si256(h0123, h4567, 0x31));
#ifdef NEIF_TEST
      const __m256 t = _mm256_mul_ps(
          a16 ? _mm256_mul_ps(_mm256_loadu_ps(fold + o), _mm256_set1_ps(act16::K256)) : _mm256_loadu_ps(fold + o),
          _mm256_cvtepi32_ps(s));
#else
      const __m256 t = _mm256_mul_ps(_mm256_loadu_ps(fold + o), _mm256_cvtepi32_ps(s));
#endif
      _mm256_storeu_ps(acc_out + o, _mm256_add_ps(_mm256_loadu_ps(acc_out + o), t));
    }
  }

  // The token row with side channels: raw embedding row + the two projection
  // terms, then the embedding norm (the order of the training model).
  const float* paket_token_row(int token) {
    const PaketArenas& P = M.paket;
    if (token >= P.emb_rows ||
        pk.prev1 >= P.emb_rows + 1 ||
        pk.prev2 >= P.emb_rows + 1 ||
        pk.match_byte >= PaketArenas::NMATCH ||
        (pk.match_class & ~15) != 0) {
      // A row index out of range would read past a table and produce numbers
      // that look plausible.  The engine never generates one; a caller that
      // does gets a message instead of a silent wrong distribution.
      std::fprintf(stderr, "E194 %d %d %d %d %d %d\n",
          token, pk.prev1, pk.prev2, pk.match_byte, pk.match_class, P.emb_rows);
      std::exit(1);
    }
    alignas(64) float t[D];
    std::memcpy(t, P.emb_raw + size_t(token) * D, sizeof(float) * D);
    if (P.has_last) {
      alignas(64) float f[PaketArenas::DIN_LAST];
      static_assert(PaketArenas::X == 2,
          "the predecessor list below is written out for two "
          "slots.  With a different slot count the remaining slots would "
          "silently read row 0 -- widen PaketChannels and this initializer "
          "together, and re-export the last_table (its row count is X*rows).");
      const int prev[PaketArenas::X] = {pk.prev1, pk.prev2};
      for (int k = 0; k < PaketArenas::X; k++) {
        const size_t row = size_t(k) * (P.emb_rows + 1)
                         + size_t(prev[k]);
        std::memcpy(f + k * PaketArenas::DSLOT,
                    P.last_rows + row * PaketArenas::DSLOT,
                    sizeof(float) * PaketArenas::DSLOT);
      }
      proj_add(f, PaketArenas::DIN_LAST, P.last_q, P.last_fold, P.last_s_act,
               t);
    }
    if (P.has_match) {
      alignas(64) float f[PaketArenas::DIN_MATCH];
      // as zctx/zpmodel.py match_features: rows [match_byte] and [NMATCH + class]
      std::memcpy(f, P.match_rows + size_t(pk.match_byte) * PaketArenas::DSLOT,
                  sizeof(float) * PaketArenas::DSLOT);
      std::memcpy(f + PaketArenas::DSLOT,
                  P.match_rows +
                      size_t(PaketArenas::NMATCH + pk.match_class) *
                          PaketArenas::DSLOT,
                  sizeof(float) * PaketArenas::DSLOT);
      proj_add(f, PaketArenas::DIN_MATCH, P.match_q, P.match_fold,
               P.match_s_act, t);
    }
    rms_norm192(t, tok_paket);
    return tok_paket;
  }

#if defined(NEIF_TEST) && defined(NEIF_MT)
  // ---- NEIF_MT: the speculative step (src/neif_mt.h, runs/neif-par/DESIGN.md 3.4) ----
  // A second engine object, the shadow, runs one read row for a guessed token
  // on the transformer thread while the main thread decodes the bits of that
  // token.  It has its own row and scratch buffers, shares the weights and the
  // K/V rings and the conv rings of the KDA layers, and writes the stepped KDA
  // head states to a second buffer (kda_layer_step_io).  What it writes that
  // is not its own: the K/V slot of position t, which holds position t - WIN
  // and is rewritten by every step before it is read, and the conv ring slot
  // of this step, likewise.  The fast weights are not touched: the
  // shadow keeps the relu^2 codes of the last FWK MLPs, and the commit runs
  // fw_apply and the second head on them, in the order of a real step.
  // The real step() takes the shadow's result only if its own inputs (token,
  // prior, channels, a read row, nothing in the row queue, the same position)
  // are the ones the shadow ran on, bit for bit; otherwise it runs as always.
  TransformerOptImpl* sp;         // main object: its shadow (null until the first speculation)
  TransformerOptImpl* sp_main;    // shadow: the engine it speculates for
  bool spec_mode;                 // this object is a shadow
  bool sp_live;                   // main: a speculation was started and not yet looked at
  // KDA head states: a speculative step reads the committed states of a
  // layer and writes the stepped ones to the other of two buffers (the home
  // buffer kst[ki].S and kS_alt); a commit makes that buffer the committed
  // one.  The conv ring and the position stay in kst (shared).
  static constexpr size_t kSN = 3 * 64 * 64;
  float* kS_alt;                  // main: NKIMI x kSN floats
  float* kS_cur[NKIMI];           // main: committed head states of layer ki (null: kst[ki].S)
  float* sp_out[NKIMI];           // shadow: where its step put them
  float* kS(int ki) { return kS_cur[ki] ? kS_cur[ki] : &kst[ki].S[0][0]; }
  std::atomic<int> sp_abort;      // main: the running speculation is not wanted
  uint64_t epoch;                 // main: counts every entry that changes the engine
  struct SpRec {                  // main: the inputs of the finished speculative step
    bool valid;
    int token;
    PaketChannels pk;
    uint64_t epoch;
    int64_t t;
    uint16_t prior[208];
  } sp_rec;
  struct SpFw {                   // shadow: the relu^2 code of a fast-weight block
    int nnz;
    uint8_t q768[DMLP];
    uint16_t idx768[DMLP + 16];
  } sp_fw[FWK];
  unsigned long long sp_hits, sp_miss, sp_runs;   // main: commits, speculations not used, speculative steps

  void spec_save_fw(int l, const ZRow* r) {
    SpFw& s = sp_fw[l - (NL - FWK)];
    s.nnz = r->nnz;
    std::memcpy(s.q768, r->q768, sizeof s.q768);
    std::memcpy(s.idx768, r->idx768, sizeof(uint16_t) * static_cast<size_t>(r->nnz));
  }
  __attribute__((cold, noinline)) void spec_make() {
    TransformerOptImpl* s = static_cast<TransformerOptImpl*>(MapNull(sizeof(TransformerOptImpl)));
    // the weights: the same arenas (the model object holds pointers into them)
    std::memcpy(static_cast<void*>(&s->M), static_cast<const void*>(&M), sizeof(OptModel));
    s->kind = kind;
    s->kvf = kvf;
    s->kvi = kvi;
    kS_alt = static_cast<float*>(MapNull(sizeof(float) * kSN * NKIMI));
    s->kst = kst;   // the conv rings and positions are shared
    s->spec_mode = true;
    s->sp_main = this;
    s->gelesen = true;
    s->fw_gel = false;
    sp = s;
  }
  // shadow: one read row for `token` (prior_f32, pk, t and rope_off are set)
  void spec_step(uint8_t token) {
    fw_gel = false;
    tok = token;
    zb_embed(*this);
    zb_layers(this, 1, false, 0, NL, t);
    if (sp_main->sp_abort.load(std::memory_order_relaxed)) return;
    ar_read();
    rms_norm_quant192(x, xn, M.unembed.s_act, q5);
    std::memcpy(sc_key, sc_l2_stage, sizeof(float) * D);
    qgemv_f32(M.unembed.m, q5, logits);
    head_plm_softcap_softmax(logits, prior_f32, dot192(M.plm_w, xn), probs208);
  }
  // main: take the shadow's row as this step (prior_f32 and pk are set by the caller)
  void spec_commit(float* probs_out) {
    TransformerOptImpl& s = *sp;
    tok = s.tok;
    std::memcpy(x, s.x, sizeof x);
    std::memcpy(xn, s.xn, sizeof xn);
    std::memcpy(q5, s.q5, sizeof q5);
    std::memcpy(sc_key, s.sc_key, sizeof sc_key);
    std::memcpy(logits, s.logits, sizeof logits);
    std::memcpy(probs208, s.probs208, sizeof probs208);
    fw_gel = true;
    for (int b = 0; b < FWK; b++) {   // the fast-weight forward of the last FWK MLPs, as mlp_block does it
      const SpFw& f = s.sp_fw[b];
      std::memcpy(q768, f.q768, sizeof f.q768);
      std::memcpy(idx768, f.idx768, sizeof(uint16_t) * static_cast<size_t>(f.nnz));
      nnz = f.nnz;
      fw_apply(b, f.nnz, M.mlp[NL - FWK + b].down_s_act);
    }
    for (int ki = 0; ki < NKIMI; ki++) {   // the stepped head states become the committed ones
      kS_cur[ki] = s.sp_out[ki];
      kst[ki].pos++;
    }
    std::memcpy(probs_out, probs208, sizeof(float) * V);
    fw_head();
    t++;
    epoch++;
  }
#endif

  // gelesen ("read"): the output of the next step() is used.  Applies to
  // that step only.  true from the ctor.
  bool gelesen;
  static_assert((16 % 4) == 0,
      "the K/V-only last layer needs (NL-1) % 4 == 3 (a window layer)");

  // prior must be the cap-208 internal buffer with pads [205..208) zeroed
  void step(uint8_t token, float* probs_out) {
    if (!gelesen) {
      gelesen = true;
      zb_push(token);
      return;
    }
    zb_run();
    fw_gel = true;
    tok = token;
#ifdef NEIF_TEST
    if (neif_test::g_own & neif_test::kOwnTfLo)   // pipeline: the embedding belongs to the lower layers
#endif
    zb_embed(*this);
#ifdef NEIF_TEST
    zb_stage(this, 1, false);
    // pipeline: the output heads belong to the stage with the upper layers
    if (!(neif_test::g_own & neif_test::kOwnTfHi)) { t++; return; }
#else
    zb_layers(this, 1, false, 0, NL, t);
#endif
    {
      PSCOPE(PS_EL_RESID);
      ar_read();                                  // x = the final read (q 2 NL, M.ar_q_final)
    }

    {
      PSCOPE(PS_EL_NQ_FIN);
#ifdef NEIF_TEST
      if (act16::g_mask & act16::UNEMB) rms_norm192(x, xn); else   // NEIF_ACT16
#endif
      rms_norm_quant192(x, xn, M.unembed.s_act, q5);
    }
    std::memcpy(sc_key, sc_l2_stage, sizeof(float) * D);
    {
      PSCOPE(PS_MM_UNEMB);
#ifdef NEIF_TEST
      if (act16::g_mask & act16::UNEMB) {   // NEIF_ACT16
        alignas(32) int16_t q16[D];
        act16::quant(xn, M.unembed.s_act, q16, D);
        act16::dense(M.unembed.m, q16, act16::K256, logits);
      } else
#endif
      qgemv_f32(M.unembed.m, q5, logits);  // rows_padded 208
    }
    {
      PSCOPE(PS_EL_HEAD);
      head_plm_softcap_softmax(logits, prior_f32, dot192(M.plm_w, xn), probs208);
      std::memcpy(probs_out, probs208, sizeof(float) * V);
      fw_head();
    }
    t++;
  }

};

COLD_SIZE_ATTR TransformerOpt::TransformerOpt(const char* weights_path, AttnKind attn)
    : impl(new TransformerOptImpl(weights_path, attn)) {}

COLD_SIZE_ATTR TransformerOpt::~TransformerOpt() = default;

#if defined(NEIF_TEST) && defined(NEIF_MT)
// NEIF_MT: before anything enters the engine on the main thread, a running
// speculation is stopped and waited for; `keep` leaves its result for step()
static void spec_join(TransformerOptImpl* m, bool keep) {
  if (!m->sp_live) return;
  if (!keep) m->sp_abort.store(1, std::memory_order_relaxed);
  neif_mt::SpecWait();
  m->sp_abort.store(0, std::memory_order_relaxed);
  m->sp_live = false;
  if (!keep) {
    if (m->sp_rec.valid) ++m->sp_miss;
    m->sp_rec.valid = false;
  }
}
bool TransformerOpt::spec_ready() const {
  return impl->zn == 0 && impl->kind == AttnKind::KVI8 && !act16::g_mask && !act16::g_fast;
}
void TransformerOpt::spec_mark() {
  spec_join(impl.get(), false);   // an earlier speculation nobody looked at
  impl->sp_live = true;
}
void TransformerOpt::spec_abort() { impl->sp_abort.store(1, std::memory_order_relaxed); }
void TransformerOpt::spec_run(uint8_t token, const uint16_t* prior_f16, const PaketChannels& channels) {
  TransformerOptImpl& m = *impl;
  if (!m.sp) m.spec_make();
  TransformerOptImpl& s = *m.sp;
  m.sp_rec.valid = false;
  ++m.sp_runs;
  s.pk = channels;
  prior_f16_to_f32(prior_f16, s.prior_f32);
  s.t = m.t;
  s.rope_off = m.rope_off;
  s.spec_step(token);
  if (m.sp_abort.load(std::memory_order_relaxed)) return;
  m.sp_rec.token = token;
  m.sp_rec.pk = channels;
  m.sp_rec.epoch = m.epoch;
  m.sp_rec.t = m.t;
  std::memcpy(m.sp_rec.prior, prior_f16, sizeof m.sp_rec.prior);
  m.sp_rec.valid = true;
}
void TransformerOpt::spec_stats(unsigned long long* v) const {
  v[0] = impl->sp_runs;
  v[1] = impl->sp_hits;
  v[2] = impl->sp_miss;
}
#endif

void TransformerOpt::begin_article(int64_t rope_position_offset) {
#ifdef NEIF_TEST
  if (!neif_test::OwnTf()) return;   // pipeline: the network lives in other stages
#ifdef NEIF_MT
  spec_join(impl.get(), false);
  impl->epoch++;
#endif
#endif
  impl->begin(rope_position_offset);
}

void TransformerOpt::step(uint8_t token, const uint16_t* prior_f16,
                          float* probs_out) {
  prior_f16_to_f32(prior_f16, impl->prior_f32);  // exact; pads zeroed
  impl->step(token, probs_out);
}

void TransformerOpt::step(uint8_t token, const float* prior205,
                          float* probs_out) {
  std::memcpy(impl->prior_f32, prior205, sizeof(float) * V);
  for (int i = 208; i < 208; ++i) impl->prior_f32[i] = 0.0f;
  impl->step(token, probs_out);
}

void TransformerOpt::step(uint8_t token, const uint16_t* prior_f16,
                          float* probs_out, const PaketChannels& channels) {
#if defined(NEIF_TEST) && defined(NEIF_MT)
  // NEIF_MT: the row the shadow ran on?  Then its result is this step.
  if (impl->sp_live) {
    TransformerOptImpl& m = *impl;
    const TransformerOptImpl::SpRec& r = m.sp_rec;
    // a row that cannot match (unread, another token) stops the speculation at once
    const bool may = m.gelesen && m.zn == 0;
    spec_join(impl.get(), may);
    if (may && r.valid && r.token == static_cast<int>(token) && r.epoch == m.epoch && r.t == m.t &&
        r.pk.prev1 == channels.prev1 && r.pk.prev2 == channels.prev2 && r.pk.match_byte == channels.match_byte &&
        r.pk.match_class == channels.match_class && std::memcmp(r.prior, prior_f16, sizeof r.prior) == 0) {
      m.sp_rec.valid = false;
      ++m.sp_hits;
      m.pk = channels;
      prior_f16_to_f32(prior_f16, m.prior_f32);
      m.spec_commit(probs_out);
      return;
    }
    if (r.valid) ++m.sp_miss;
    m.sp_rec.valid = false;
  }
  impl->epoch++;
#endif
  impl->pk = channels;
  prior_f16_to_f32(prior_f16, impl->prior_f32);  // exact; pads zeroed
  impl->step(token, probs_out);
}

bool TransformerOpt::paket_head() const { return impl->M.paket.present; }

const float* TransformerOpt::paket_row(uint8_t token,
                                       const PaketChannels& channels) {
  impl->pk = channels;
  return impl->paket_token_row(token);
}

int TransformerOpt::embedding_rows() const { return impl->M.paket.emb_rows; }

void TransformerOpt::set_gelesen(bool gelesen) { impl->gelesen = gelesen; }

const float* TransformerOpt::last_logits() const { return impl->logits; }

// `xn` is x after the final rms_norm, i.e. exactly the vector fed to
// the unembedding matmul one line later in step().  Nothing is written here.
const float* TransformerOpt::last_hidden() const { return impl->xn; }
int TransformerOpt::hidden_size() { return D; }
const float* TransformerOpt::last_key() const { return impl->sc_key; }

AttnKind TransformerOpt::attn_kind() const { return impl->kind; }

#ifdef NEIF_TEST
void TransformerOpt::fw_update(uint8_t target) {
  // pipeline: the fast weights live in the stage with the upper layers
  if (!(neif_test::g_own & neif_test::kOwnTfHi)) return;
  impl->fw_update(target);
}
// Pipeline: what the coder side reads of the network between two driver steps
// (hidden vector, cache key, second-head row), out of the engine or into it.
void TransformerOpt::neif_state(float* buf, bool load) {
  float* const part[3] = {impl->xn, impl->sc_key, impl->fw_probs208};
  const int n[3] = {D, D, 208};
  for (int i = 0; i < 3; buf += n[i], i++) {
    if (load) std::memcpy(part[i], buf, sizeof(float) * n[i]);
    else std::memcpy(buf, part[i], sizeof(float) * n[i]);
  }
}
#else
void TransformerOpt::fw_update(uint8_t target) { impl->fw_update(target); }
#endif
const float* TransformerOpt::fw_probs() const { return impl->fw_probs208; }

}  // namespace opt
}  // namespace fx2
