// neif_vnni_bench.cpp: dense int4 x int8 matmul kernels of cmix-neif 2.0 on the shapes and in the
// order of one transformer row (not part of any build).
// Per row, as src/tf1/opt/model_opt.cpp calls them (16 layers; layer l is a window layer when
// l % 4 == 3, else a KDA layer):
//   KDA layer:    q, k, v 192x192; gate up 64x192, gate down 192x64 (forget, output); out 192x192
//   window layer: q 192x192, k 64x192, v 64x192, out 192x192
//   every layer:  mlp.up 1024x192
// All arenas of one format lie one behind the other in this order (as arena_build.cpp lays them).
// Kernels (all give the float row y = float(dot) * scale; the bench checks them bit for bit
// against A):
//   A   the entry's kernel: run_g8 of qmat_dense.cpp, unpacked int8 weights, vpmaddubsw (AVX2)
//   A0  A without software prefetch;  A12: A with the prefetch 12 KB ahead instead of 6 KB
//   B   the entry's unused run_packed: nibble-packed weights, 4-row groups, vpmaddubsw (AVX2)
//   C   AVX-VNNI vpdpbusd on the entry's unpacked arena (no layout change)
//   D   AVX-VNNI vpdpbusd on nibble-packed weights, 8-row groups (half the weight bytes)
//   E   AVX2 vpmaddubsw on the packed 8-row layout of D (what a Zen 3 build could use)
//   C5/D5 (only with -DBENCH_AVX512 -mavx512f -mavx512bw -mavx512vnni): the 512-bit forms of C
//       and D for Tiger Lake class CPUs; they cannot run on a CPU without AVX-512 (such as the
//       Lunar Lake laptop this was written on).
// Cache state: before every row JUNK megabytes of other memory are read and written (one byte
// per line), which pushes the weights out of the cache as the other models do in cmix.
// Usage: neif_vnni_bench [JUNK_MB ...]      (default 0 16 48)
// Build: clang++-18 -O3 -march=x86-64-v3 -mtune=znver2 -mavxvnni neif_vnni_bench.cpp -o vnni_bench
#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct Site { int dout, din, cls; size_t oA, oB, oD; };   // cls: 0 mlp.up, 1 q/k/v, 2 gates, 3 out
static std::vector<Site> sites;
static uint8_t *poolA, *poolB, *poolD;
static const char* kCls[4] = {"mlp.up (16)", "q/k/v (48)", "gates (48)", "out proj (16)"};

static size_t bytesA(int dout, int din) { return size_t((dout + 7) / 8) * (64 + size_t(din) * 8); }
static size_t bytesB(int dout, int din) { return size_t((dout + 3) / 4) * (32 + size_t(din) * 2); }
static size_t bytesD(int dout, int din) { return size_t((dout + 7) / 8) * (64 + size_t(din) * 4); }

static uint32_t rs = 12345;
static uint32_t rnd() { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }

// ---- reduction of 8 int32 accumulators (8 lanes each) to 8 row sums ----
static inline __m256i red8(__m256i s0, __m256i s1, __m256i s2, __m256i s3, __m256i s4, __m256i s5, __m256i s6, __m256i s7) {
  const __m256i q0 = _mm256_hadd_epi32(_mm256_hadd_epi32(s0, s1), _mm256_hadd_epi32(s2, s3));
  const __m256i q1 = _mm256_hadd_epi32(_mm256_hadd_epi32(s4, s5), _mm256_hadd_epi32(s6, s7));
  return _mm256_add_epi32(_mm256_permute2x128_si256(q0, q1, 0x20), _mm256_permute2x128_si256(q0, q1, 0x31));
}
static inline void epi(float* y, int g, __m256i sums, const float* scl) {
  _mm256_storeu_ps(y + 8 * g, _mm256_mul_ps(_mm256_cvtepi32_ps(sums), _mm256_load_ps(scl)));
}

// ---- A: the entry's kernel (PFD = prefetch distance in bytes, 0 = none) ----
template <int NC, int PFD>
static inline void runA(const uint8_t* arena, int ng, const uint8_t* act, float* y) {
  const __m256i ones = _mm256_set1_epi16(1);
  const uint8_t* gp = arena;
  constexpr ptrdiff_t GB = 64 + NC * 256;
  for (int g = 0; g < ng; g++, gp += GB) {
    const __m256i* w = reinterpret_cast<const __m256i*>(gp + 64);
    __m256i av = _mm256_load_si256(reinterpret_cast<const __m256i*>(act));
    __m256i s0 = _mm256_maddubs_epi16(av, w[0]), s1 = _mm256_maddubs_epi16(av, w[1]);
    __m256i s2 = _mm256_maddubs_epi16(av, w[2]), s3 = _mm256_maddubs_epi16(av, w[3]);
    __m256i s4 = _mm256_maddubs_epi16(av, w[4]), s5 = _mm256_maddubs_epi16(av, w[5]);
    __m256i s6 = _mm256_maddubs_epi16(av, w[6]), s7 = _mm256_maddubs_epi16(av, w[7]);
    if (PFD) for (int k = 0; k < 4; k++) _mm_prefetch(reinterpret_cast<const char*>(gp) + PFD + 64 * k, _MM_HINT_T0);
#pragma clang loop unroll(disable)
    for (int c = 1; c < NC; c++) {
      const __m256i* wc = w + c * 8;
      if (PFD) for (int k = 0; k < 4; k++) _mm_prefetch(reinterpret_cast<const char*>(gp) + c * 256 + PFD + 64 * k, _MM_HINT_T0);
      av = _mm256_load_si256(reinterpret_cast<const __m256i*>(act + 32 * c));
      s0 = _mm256_add_epi16(s0, _mm256_maddubs_epi16(av, wc[0]));
      s1 = _mm256_add_epi16(s1, _mm256_maddubs_epi16(av, wc[1]));
      s2 = _mm256_add_epi16(s2, _mm256_maddubs_epi16(av, wc[2]));
      s3 = _mm256_add_epi16(s3, _mm256_maddubs_epi16(av, wc[3]));
      s4 = _mm256_add_epi16(s4, _mm256_maddubs_epi16(av, wc[4]));
      s5 = _mm256_add_epi16(s5, _mm256_maddubs_epi16(av, wc[5]));
      s6 = _mm256_add_epi16(s6, _mm256_maddubs_epi16(av, wc[6]));
      s7 = _mm256_add_epi16(s7, _mm256_maddubs_epi16(av, wc[7]));
    }
    __m256i sums = red8(_mm256_madd_epi16(s0, ones), _mm256_madd_epi16(s1, ones), _mm256_madd_epi16(s2, ones),
                        _mm256_madd_epi16(s3, ones), _mm256_madd_epi16(s4, ones), _mm256_madd_epi16(s5, ones),
                        _mm256_madd_epi16(s6, ones), _mm256_madd_epi16(s7, ones));
    sums = _mm256_sub_epi32(sums, _mm256_load_si256(reinterpret_cast<const __m256i*>(gp)));
    epi(y, g, sums, reinterpret_cast<const float*>(gp + 32));
  }
}

// ---- B: the entry's run_packed (4-row groups [4 x fp32 scale, 16 B pad | nibbles w + 7]) ----
template <int NPAIR>
static inline void runB(const uint8_t* arena, int ng, const int8_t* act, int32_t corr7, float* y) {
  const __m256i ones = _mm256_set1_epi16(1);
  const __m256i m0F = _mm256_set1_epi8(0x0F);
  const __m128i corr = _mm_set1_epi32(corr7);
  const uint8_t* gp = arena;
  const ptrdiff_t GB = 32 + static_cast<ptrdiff_t>(NPAIR) * 128;
  constexpr int NB = (NPAIR + 3) / 4;
  for (int g = 0; g < ng; g++, gp += GB) {
    const __m256i* w = reinterpret_cast<const __m256i*>(gp + 32);
    __m256i d0 = _mm256_setzero_si256(), d1 = d0, d2 = d0, d3 = d0;
    for (int b = 0; b < NB; b++) {
      const int np = (b + 1) * 4 <= NPAIR ? 4 : NPAIR - b * 4;
      __m256i s0 = _mm256_setzero_si256(), s1 = s0, s2 = s0, s3 = s0;
      for (int p = 0; p < np; p++) {
        if (p == 0) _mm_prefetch(reinterpret_cast<const char*>(gp) + b * 512 + 6144, _MM_HINT_T0);
        const int pp = b * 4 + p;
        __m256i a0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(act + 64 * pp));
        __m256i a1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(act + 64 * pp + 32));
#define QM_PSTEP(r)                                                  \
  {                                                                  \
    __m256i wv = _mm256_load_si256(w + pp * 4 + (r));                \
    __m256i lo = _mm256_and_si256(wv, m0F);                          \
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(wv, 4), m0F);    \
    s##r = _mm256_add_epi16(s##r, _mm256_maddubs_epi16(lo, a0));     \
    s##r = _mm256_add_epi16(s##r, _mm256_maddubs_epi16(hi, a1));     \
  }
        QM_PSTEP(0) QM_PSTEP(1) QM_PSTEP(2) QM_PSTEP(3)
#undef QM_PSTEP
      }
      d0 = _mm256_add_epi32(d0, _mm256_madd_epi16(s0, ones));
      d1 = _mm256_add_epi32(d1, _mm256_madd_epi16(s1, ones));
      d2 = _mm256_add_epi32(d2, _mm256_madd_epi16(s2, ones));
      d3 = _mm256_add_epi32(d3, _mm256_madd_epi16(s3, ones));
    }
    __m256i h = _mm256_hadd_epi32(_mm256_hadd_epi32(d0, d1), _mm256_hadd_epi32(d2, d3));
    __m128i sums = _mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1));
    sums = _mm_sub_epi32(sums, corr);
    _mm_storeu_ps(y + 4 * g, _mm_mul_ps(_mm_cvtepi32_ps(sums), _mm_load_ps(reinterpret_cast<const float*>(gp))));
  }
}

// ---- C: AVX-VNNI on the entry's unpacked arena ----
#if defined(__AVXVNNI__)
#define DPBUSD(acc, a, b) _mm256_dpbusd_avx_epi32(acc, a, b)
template <int NC>
static inline void runC(const uint8_t* arena, int ng, const uint8_t* act, float* y) {
  const uint8_t* gp = arena;
  constexpr ptrdiff_t GB = 64 + NC * 256;
  for (int g = 0; g < ng; g++, gp += GB) {
    const __m256i* w = reinterpret_cast<const __m256i*>(gp + 64);
    __m256i s0 = _mm256_setzero_si256(), s1 = s0, s2 = s0, s3 = s0, s4 = s0, s5 = s0, s6 = s0, s7 = s0;
#pragma clang loop unroll(disable)
    for (int c = 0; c < NC; c++) {
      const __m256i* wc = w + c * 8;
      for (int k = 0; k < 4; k++) _mm_prefetch(reinterpret_cast<const char*>(gp) + c * 256 + 6144 + 64 * k, _MM_HINT_T0);
      const __m256i av = _mm256_load_si256(reinterpret_cast<const __m256i*>(act + 32 * c));
      const __m256i z = _mm256_setzero_si256();   // a fresh register per product: no latency chain through the sums
      s0 = _mm256_add_epi32(s0, DPBUSD(z, av, wc[0])); s1 = _mm256_add_epi32(s1, DPBUSD(z, av, wc[1]));
      s2 = _mm256_add_epi32(s2, DPBUSD(z, av, wc[2])); s3 = _mm256_add_epi32(s3, DPBUSD(z, av, wc[3]));
      s4 = _mm256_add_epi32(s4, DPBUSD(z, av, wc[4])); s5 = _mm256_add_epi32(s5, DPBUSD(z, av, wc[5]));
      s6 = _mm256_add_epi32(s6, DPBUSD(z, av, wc[6])); s7 = _mm256_add_epi32(s7, DPBUSD(z, av, wc[7]));
    }
    __m256i sums = red8(s0, s1, s2, s3, s4, s5, s6, s7);
    sums = _mm256_sub_epi32(sums, _mm256_load_si256(reinterpret_cast<const __m256i*>(gp)));
    epi(y, g, sums, reinterpret_cast<const float*>(gp + 32));
  }
}
#endif

// sum of the n activation bytes (n a multiple of 32)
static inline int32_t act_sum(const uint8_t* act, int n) {
  __m256i s = _mm256_setzero_si256();
  for (int i = 0; i < n; i += 32)
    s = _mm256_add_epi64(s, _mm256_sad_epu8(_mm256_load_si256(reinterpret_cast<const __m256i*>(act + i)), _mm256_setzero_si256()));
  const __m128i t = _mm_add_epi64(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
  return static_cast<int32_t>(_mm_cvtsi128_si64(t) + _mm_extract_epi64(t, 1));
}

// ---- D / E: packed 8-row groups [8 x int32 corr | 8 x fp32 scale | per 64 columns: 8 rows x 32 B],
// byte i of a row vector = nibble (w + 7) of column 64p + i (low) and of column 64p + 32 + i (high).
// The activation stays the entry's biased u8 (qa + 128), as the UNSIGNED operand; the nibbles
// 0..14 are the signed operand:  sum (qa + 128)(w + 7) = dot + 7 sum(act) + 128 sum(w) [+ pads 0],
// so dot = acc - 7 * sum(act bytes) - corr[row], corr[row] = 128 * sum_i (w[row][i] + 7) - ... see build.
#if defined(__AVXVNNI__)
template <int NP>
static inline void runD(const uint8_t* arena, int ng, const uint8_t* act, float* y) {
  const __m256i m0F = _mm256_set1_epi8(0x0F);
  const uint8_t* gp = arena;
  constexpr ptrdiff_t GB = 64 + NP * 256;
  for (int g = 0; g < ng; g++, gp += GB) {
    const __m256i* w = reinterpret_cast<const __m256i*>(gp + 64);
    __m256i s0 = _mm256_setzero_si256(), s1 = s0, s2 = s0, s3 = s0, s4 = s0, s5 = s0, s6 = s0, s7 = s0;
#pragma clang loop unroll(disable)
    for (int p = 0; p < NP; p++) {
      const __m256i* wc = w + p * 8;
      for (int k = 0; k < 4; k++) _mm_prefetch(reinterpret_cast<const char*>(gp) + p * 256 + 6144 + 64 * k, _MM_HINT_T0);
      const __m256i a0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(act + 64 * p));
      const __m256i a1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(act + 64 * p + 32));
#define D_STEP(r)                                                                 \
  {                                                                               \
    const __m256i wv = _mm256_load_si256(wc + (r));                               \
    __m256i t = DPBUSD(_mm256_setzero_si256(), a0, _mm256_and_si256(wv, m0F));    \
    t = DPBUSD(t, a1, _mm256_and_si256(_mm256_srli_epi16(wv, 4), m0F));           \
    s##r = _mm256_add_epi32(s##r, t);   /* no latency chain through the sums */   \
  }
      D_STEP(0) D_STEP(1) D_STEP(2) D_STEP(3) D_STEP(4) D_STEP(5) D_STEP(6) D_STEP(7)
#undef D_STEP
    }
    __m256i sums = red8(s0, s1, s2, s3, s4, s5, s6, s7);
    sums = _mm256_sub_epi32(sums, _mm256_load_si256(reinterpret_cast<const __m256i*>(gp)));
    epi(y, g, sums, reinterpret_cast<const float*>(gp + 32));
  }
}
#endif
// E: the same layout with vpmaddubsw; int16 bound per lane: NP * 2 * 2 * 255 * 14 = 42840 for NP = 3,
// so the int16 sums are widened after every 64 columns
template <int NP>
static inline void runE(const uint8_t* arena, int ng, const uint8_t* act, float* y) {
  const __m256i m0F = _mm256_set1_epi8(0x0F);
  const __m256i ones = _mm256_set1_epi16(1);
  const uint8_t* gp = arena;
  constexpr ptrdiff_t GB = 64 + NP * 256;
  for (int g = 0; g < ng; g++, gp += GB) {
    const __m256i* w = reinterpret_cast<const __m256i*>(gp + 64);
    __m256i s0 = _mm256_setzero_si256(), s1 = s0, s2 = s0, s3 = s0, s4 = s0, s5 = s0, s6 = s0, s7 = s0;
#pragma clang loop unroll(disable)
    for (int p = 0; p < NP; p++) {
      const __m256i* wc = w + p * 8;
      for (int k = 0; k < 4; k++) _mm_prefetch(reinterpret_cast<const char*>(gp) + p * 256 + 6144 + 64 * k, _MM_HINT_T0);
      const __m256i a0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(act + 64 * p));
      const __m256i a1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(act + 64 * p + 32));
#define E_STEP(r)                                                                               \
  {                                                                                             \
    const __m256i wv = _mm256_load_si256(wc + (r));                                             \
    const __m256i t = _mm256_add_epi16(_mm256_maddubs_epi16(a0, _mm256_and_si256(wv, m0F)),     \
                                       _mm256_maddubs_epi16(a1, _mm256_and_si256(_mm256_srli_epi16(wv, 4), m0F))); \
    s##r = _mm256_add_epi32(s##r, _mm256_madd_epi16(t, ones));                                  \
  }
      E_STEP(0) E_STEP(1) E_STEP(2) E_STEP(3) E_STEP(4) E_STEP(5) E_STEP(6) E_STEP(7)
#undef E_STEP
    }
    __m256i sums = red8(s0, s1, s2, s3, s4, s5, s6, s7);
    sums = _mm256_sub_epi32(sums, _mm256_load_si256(reinterpret_cast<const __m256i*>(gp)));
    epi(y, g, sums, reinterpret_cast<const float*>(gp + 32));
  }
}

// ---- AVX-512 VNNI forms (Tiger Lake class); compiled only with -DBENCH_AVX512 ----
#if defined(BENCH_AVX512) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
// C5: the entry's unpacked arena; one zmm holds the 32 columns of two consecutive rows
template <int NC>
static inline void runC5(const uint8_t* arena, int ng, const uint8_t* act, float* y) {
  const uint8_t* gp = arena;
  constexpr ptrdiff_t GB = 64 + NC * 256;
  for (int g = 0; g < ng; g++, gp += GB) {
    const __m512i* w = reinterpret_cast<const __m512i*>(gp + 64);
    __m512i s01 = _mm512_setzero_si512(), s23 = s01, s45 = s01, s67 = s01;
    for (int c = 0; c < NC; c++) {
      const __m512i* wc = w + c * 4;
      for (int k = 0; k < 4; k++) _mm_prefetch(reinterpret_cast<const char*>(gp) + c * 256 + 6144 + 64 * k, _MM_HINT_T0);
      const __m512i av = _mm512_broadcast_i64x4(_mm256_load_si256(reinterpret_cast<const __m256i*>(act + 32 * c)));
      s01 = _mm512_dpbusd_epi32(s01, av, _mm512_load_si512(wc + 0));
      s23 = _mm512_dpbusd_epi32(s23, av, _mm512_load_si512(wc + 1));
      s45 = _mm512_dpbusd_epi32(s45, av, _mm512_load_si512(wc + 2));
      s67 = _mm512_dpbusd_epi32(s67, av, _mm512_load_si512(wc + 3));
    }
    __m256i sums = red8(_mm512_castsi512_si256(s01), _mm512_extracti64x4_epi64(s01, 1), _mm512_castsi512_si256(s23),
                        _mm512_extracti64x4_epi64(s23, 1), _mm512_castsi512_si256(s45), _mm512_extracti64x4_epi64(s45, 1),
                        _mm512_castsi512_si256(s67), _mm512_extracti64x4_epi64(s67, 1));
    sums = _mm256_sub_epi32(sums, _mm256_load_si256(reinterpret_cast<const __m256i*>(gp)));
    epi(y, g, sums, reinterpret_cast<const float*>(gp + 32));
  }
}
// D5: the packed layout of D; one zmm holds the packed vectors of two consecutive rows
template <int NP>
static inline void runD5(const uint8_t* arena, int ng, const uint8_t* act, float* y) {
  const __m512i m0F = _mm512_set1_epi8(0x0F);
  const uint8_t* gp = arena;
  constexpr ptrdiff_t GB = 64 + NP * 256;
  for (int g = 0; g < ng; g++, gp += GB) {
    const __m512i* w = reinterpret_cast<const __m512i*>(gp + 64);
    __m512i s01 = _mm512_setzero_si512(), s23 = s01, s45 = s01, s67 = s01;
    for (int p = 0; p < NP; p++) {
      const __m512i* wc = w + p * 4;
      for (int k = 0; k < 4; k++) _mm_prefetch(reinterpret_cast<const char*>(gp) + p * 256 + 6144 + 64 * k, _MM_HINT_T0);
      const __m512i a0 = _mm512_broadcast_i64x4(_mm256_load_si256(reinterpret_cast<const __m256i*>(act + 64 * p)));
      const __m512i a1 = _mm512_broadcast_i64x4(_mm256_load_si256(reinterpret_cast<const __m256i*>(act + 64 * p + 32)));
#define D5_STEP(S, k)                                                                 \
  {                                                                                   \
    const __m512i wv = _mm512_load_si512(wc + (k));                                   \
    S = _mm512_dpbusd_epi32(S, a0, _mm512_and_si512(wv, m0F));                        \
    S = _mm512_dpbusd_epi32(S, a1, _mm512_and_si512(_mm512_srli_epi16(wv, 4), m0F));  \
  }
      D5_STEP(s01, 0) D5_STEP(s23, 1) D5_STEP(s45, 2) D5_STEP(s67, 3)
#undef D5_STEP
    }
    __m256i sums = red8(_mm512_castsi512_si256(s01), _mm512_extracti64x4_epi64(s01, 1), _mm512_castsi512_si256(s23),
                        _mm512_extracti64x4_epi64(s23, 1), _mm512_castsi512_si256(s45), _mm512_extracti64x4_epi64(s45, 1),
                        _mm512_castsi512_si256(s67), _mm512_extracti64x4_epi64(s67, 1));
    sums = _mm256_sub_epi32(sums, _mm256_load_si256(reinterpret_cast<const __m256i*>(gp)));
    epi(y, g, sums, reinterpret_cast<const float*>(gp + 32));
  }
}
#endif

// ---- one site with kernel k ----
enum { K_A, K_A0, K_A12, K_B, K_C, K_D, K_E, K_C5, K_D5, K_N };
static const char* kName[K_N] = {"A  entry (AVX2, unpacked)", "A0 entry, no prefetch", "A12 entry, prefetch 12 KB",
                                 "B  run_packed (AVX2, 4 rows)", "C  AVX-VNNI, unpacked", "D  AVX-VNNI, packed",
                                 "E  AVX2, packed 8 rows", "C5 AVX512-VNNI, unpacked", "D5 AVX512-VNNI, packed"};
static bool kHave[K_N];
alignas(64) static uint8_t g_act[4][192];      // biased u8
alignas(64) static int8_t g_acts[4][192];      // the same as signed int8
static int32_t g_corr7[4][2];                  // 7 * sum(qa) over 192 and over 64 columns
alignas(64) static float g_y[K_N][1024];

__attribute__((noinline)) static void call(int k, const Site& s, int ai) {
  const uint8_t* a = g_act[ai];
  float* y = g_y[k];
  const int ng8 = (s.dout + 7) / 8, ng4 = (s.dout + 3) / 4;
  const bool w = s.din == 192;
  switch (k) {
    case K_A: w ? runA<6, 6144>(poolA + s.oA, ng8, a, y) : runA<2, 6144>(poolA + s.oA, ng8, a, y); break;
    case K_A0: w ? runA<6, 0>(poolA + s.oA, ng8, a, y) : runA<2, 0>(poolA + s.oA, ng8, a, y); break;
    case K_A12: w ? runA<6, 12288>(poolA + s.oA, ng8, a, y) : runA<2, 12288>(poolA + s.oA, ng8, a, y); break;
    case K_B: w ? runB<3>(poolB + s.oB, ng4, g_acts[ai], g_corr7[ai][0], y) : runB<1>(poolB + s.oB, ng4, g_acts[ai], g_corr7[ai][1], y); break;
#if defined(__AVXVNNI__)
    case K_C: w ? runC<6>(poolA + s.oA, ng8, a, y) : runC<2>(poolA + s.oA, ng8, a, y); break;
    case K_D: {
      const __m256i c7 = _mm256_set1_epi32(7 * act_sum(a, s.din));
      w ? runD<3>(poolD + s.oD, ng8, a, y) : runD<1>(poolD + s.oD, ng8, a, y);
      (void)c7;
      break;
    }
#endif
    case K_E: w ? runE<3>(poolD + s.oD, ng8, a, y) : runE<1>(poolD + s.oD, ng8, a, y); break;
#if defined(BENCH_AVX512) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
    case K_C5: w ? runC5<6>(poolA + s.oA, ng8, a, y) : runC5<2>(poolA + s.oA, ng8, a, y); break;
    case K_D5: w ? runD5<3>(poolD + s.oD, ng8, a, y) : runD5<1>(poolD + s.oD, ng8, a, y); break;
#endif
    default: break;
  }
}

int main(int argc, char** argv) {
  // sites in consumption order
  for (int l = 0; l < 16; l++) {
    if (l % 4 != 3) {
      for (int k = 0; k < 3; k++) sites.push_back({192, 192, 1});
      sites.push_back({64, 192, 2}); sites.push_back({192, 64, 2});
      sites.push_back({64, 192, 2}); sites.push_back({192, 64, 2});
      sites.push_back({192, 192, 3});
    } else {
      sites.push_back({192, 192, 1}); sites.push_back({64, 192, 1}); sites.push_back({64, 192, 1});
      sites.push_back({192, 192, 3});
    }
    sites.push_back({1024, 192, 0});
  }
  size_t tA = 0, tB = 0, tD = 0;
  for (auto& s : sites) { s.oA = tA; s.oB = tB; s.oD = tD; tA += bytesA(s.dout, s.din); tB += bytesB(s.dout, s.din); tD += bytesD(s.dout, s.din); }
  poolA = static_cast<uint8_t*>(aligned_alloc(4096, (tA + 32768 + 4095) & ~size_t(4095)));
  poolB = static_cast<uint8_t*>(aligned_alloc(4096, (tB + 32768 + 4095) & ~size_t(4095)));
  poolD = static_cast<uint8_t*>(aligned_alloc(4096, (tD + 32768 + 4095) & ~size_t(4095)));
  memset(poolA, 0, tA + 32768); memset(poolB, 0, tB + 32768); memset(poolD, 0, tD + 32768);
  printf("weight bytes per row: unpacked %.2f MB, run_packed %.2f MB, packed 8-row %.2f MB; %zu calls per row\n", tA / 1048576.0,
         tB / 1048576.0, tD / 1048576.0, sites.size());
  std::vector<int8_t> wq(1024 * 192);
  for (auto& s : sites) {
    const int din = s.din, dout = s.dout;
    for (int i = 0; i < dout * din; i++) wq[i] = int8_t(int(rnd() % 15) - 7);
    auto W = [&](int r, int c) -> int { return (r < dout && c < din) ? wq[r * din + c] : 0; };
    const float sc = 1.0f / 1024.0f;
    // A
    for (int g = 0; g < (dout + 7) / 8; g++) {
      uint8_t* gp = poolA + s.oA + size_t(g) * (64 + size_t(din) * 8);
      for (int r = 0; r < 8; r++) {
        int sum = 0;
        for (int c = 0; c < din; c++) sum += W(g * 8 + r, c);
        reinterpret_cast<int32_t*>(gp)[r] = 128 * sum;
        reinterpret_cast<float*>(gp + 32)[r] = g * 8 + r < dout ? sc : 0.0f;
      }
      int8_t* w = reinterpret_cast<int8_t*>(gp + 64);
      for (int c = 0; c < din / 32; c++) for (int r = 0; r < 8; r++) for (int j = 0; j < 32; j++) *w++ = int8_t(W(g * 8 + r, 32 * c + j));
    }
    // B (qpacked_build)
    for (int g = 0; g < (dout + 3) / 4; g++) {
      uint8_t* gp = poolB + s.oB + size_t(g) * (32 + size_t(din) * 2);
      for (int r = 0; r < 4; r++) reinterpret_cast<float*>(gp)[r] = g * 4 + r < dout ? sc : 0.0f;
      uint8_t* w = gp + 32;
      for (int p = 0; p < din / 64; p++) for (int r = 0; r < 4; r++) for (int j = 0; j < 32; j++)
        *w++ = uint8_t(((W(g * 4 + r, 64 * p + 32 + j) + 7) << 4) | (W(g * 4 + r, 64 * p + j) + 7));
    }
    // D: corr[row] = 128 * sum_i (w + 7) = what the biased activation adds; 7 * sum(act) comes off per call
    for (int g = 0; g < (dout + 7) / 8; g++) {
      uint8_t* gp = poolD + s.oD + size_t(g) * (64 + size_t(din) * 4);
      for (int r = 0; r < 8; r++) {
        int sum = 0;
        for (int c = 0; c < din; c++) sum += W(g * 8 + r, c);
        reinterpret_cast<int32_t*>(gp)[r] = 128 * sum;   // + the per-call term below
        reinterpret_cast<float*>(gp + 32)[r] = g * 8 + r < dout ? sc : 0.0f;
      }
      uint8_t* w = gp + 64;
      for (int p = 0; p < din / 64; p++) for (int r = 0; r < 8; r++) for (int j = 0; j < 32; j++)
        *w++ = uint8_t(((W(g * 8 + r, 64 * p + 32 + j) + 7) << 4) | (W(g * 8 + r, 64 * p + j) + 7));
    }
  }
  for (int ai = 0; ai < 4; ai++) {
    int s192 = 0, s64 = 0;
    for (int i = 0; i < 192; i++) {
      const int qa = int(rnd() % 256) - 128;
      g_act[ai][i] = uint8_t(qa + 128);
      g_acts[ai][i] = int8_t(qa);
      s192 += qa;
      if (i < 64) s64 += qa;
    }
    g_corr7[ai][0] = 7 * s192;
    g_corr7[ai][1] = 7 * s64;
  }
  for (int k = 0; k < K_N; k++) kHave[k] = true;
#if !defined(__AVXVNNI__)
  kHave[K_C] = kHave[K_D] = false;
#endif
#if !(defined(BENCH_AVX512) && defined(__AVX512VNNI__) && defined(__AVX512BW__))
  kHave[K_C5] = kHave[K_D5] = false;
#endif
  // D, E, D5 compute acc - 128 sum(w); the dot also needs - 7 * sum(act bytes) ... put it into the
  // comparison: y_D = (dot + 7 * S) * scale.  For the bit check use integer-exact float values.
  for (size_t si = 0; si < sites.size(); si += 7) {
    const Site& s = sites[si];
    const int rows = (s.dout + 7) / 8 * 8;
    call(K_A, s, 1);
    for (int k = 1; k < K_N; k++) {
      if (!kHave[k]) continue;
      call(k, s, 1);
      bool same = true;
      if (k == K_D || k == K_E || k == K_D5) {
        const float add = float(7 * act_sum(g_act[1], s.din)) / 1024.0f;   // exact: sums < 2^24, scale 2^-10
        for (int r = 0; r < s.dout; r++) same = same && (g_y[k][r] - add == g_y[K_A][r]);
      } else {
        same = memcmp(g_y[k], g_y[K_A], sizeof(float) * (k == K_B ? s.dout : rows)) == 0;
      }
      if (!same) { printf("MISMATCH kernel %s site %zu (%d x %d)\n", kName[k], si, s.dout, s.din); return 1; }
    }
  }
  printf("all kernels agree with A on the checked sites\n");

  std::vector<int> junks;
  for (int i = 1; i < argc; i++) junks.push_back(atoi(argv[i]));
  if (junks.empty()) junks = {0, 16, 48};
  const size_t JMAX = size_t(256) << 20;
  uint8_t* junk = static_cast<uint8_t*>(aligned_alloc(4096, JMAX));
  memset(junk, 1, JMAX);
  volatile unsigned sink = 0;
  for (int jm : junks) {
    const size_t JB = size_t(jm) << 20;
    const int REP = jm ? 120 : 400;
    std::vector<double> tot[K_N];
    double cls[K_N][4] = {};
    for (int rep = 0; rep < REP; rep++)
      for (int k = 0; k < K_N; k++) {
        if (!kHave[k]) continue;
        unsigned sacc = 0;
        for (size_t i = 0; i < JB; i += 64) sacc += junk[i]++;
        sink = sink + sacc;
        double t = 0;
        unsigned long long c0 = __rdtsc();
        const auto t0 = std::chrono::steady_clock::now();
        for (size_t si = 0; si < sites.size(); si++) {
          call(k, sites[si], int(si & 3));
          const unsigned long long c1 = __rdtsc();
          cls[k][sites[si].cls] += double(c1 - c0);
          c0 = c1;
        }
        t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        tot[k].push_back(t);
      }
    printf("\n== %d MB of other memory touched before every row ==\n", jm);
    printf("%-30s %9s %9s %8s | time-stamp ticks per row by site class (mean)\n", "kernel", "median us", "mean us", "to A");
    double medA = 0;
    for (int k = 0; k < K_N; k++) {
      if (!kHave[k]) continue;
      std::sort(tot[k].begin(), tot[k].end());
      const double med = tot[k][tot[k].size() / 2];
      double mean = 0;
      for (double v : tot[k]) mean += v;
      mean /= tot[k].size();
      if (k == K_A) medA = med;
      printf("%-30s %9.1f %9.1f %8.3f |", kName[k], med * 1e6, mean * 1e6, med / medA);
      for (int c = 0; c < 4; c++) printf(" %s %.0f", kCls[c], cls[k][c] / REP);
      printf("\n");
    }
  }
  return 0;
}
