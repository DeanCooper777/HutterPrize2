// NEIF_TEST only: experiment "16-bit activations".  NOT part of the entry: this
// header is included only under `#ifdef NEIF_TEST` (model_opt.cpp).
//
// The entry quantizes the input of every weight matmul to 8 bits with a
// per-site scale s.  With the environment variable NEIF_ACT16 the chosen site
// groups use 16-bit codes with the scale s/256 instead (same weights, same
// rounding rule: IEEE division, clamp in float, round half to even).  The
// paths here are slow, exact reference code for a go/no-go measurement:
// integer dot products in int32 (no overflow, see the bounds below), one
// float multiply per output.  With NEIF_ACT16 unset or 0 none of this code
// runs.
//
// NEIF_ACT16 = a number (bit mask) or a comma list of group names:
//   bit    name     site (input of ...)                      sites x width
//   1      mlpup    mlp.up                                    16 x 192
//   2      mlpdn    mlp.down (relu^2 code, sparse)            16 x 1024
//   4      kqkv     KDA query/key/value projections           36 x 192
//   8      kgup     KDA gate up projections (forget, output)  24 x 192
//   16     kgdn     KDA gate down projections                 24 x 64
//   32     kop      KDA output projection                     12 x 192
//   64     vqkv     window-attention q/k/v projections        12 x 192
//   128    vop      window-attention output projection         4 x 192
//   256    unemb    unembedding (first head)                   1 x 192
//   512    fwunemb  unembedding of the fast-weight head        1 x 192
//   1024   prior    prior embedding (sparse)                   1 x 208
//   2048   paket    side-channel projections (last, match)     2 x 32
//   all = 4095.
// Left at 8 bits on purpose: the per-head q/k/v codes of the window attention
// behind RoPE (4 layers x (3 + 1 + 1) x 64; the KV ring stores int8), and the
// 8-bit relu^2 code the fast-weight head reads (it is still computed from the
// same float value when mlpdn is on).
#pragma once
#ifdef NEIF_TEST

#include <immintrin.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "qmat.h"

namespace fx2 {
namespace opt {
namespace act16 {

enum : int {
  MLPUP = 1, MLPDN = 2, KQKV = 4, KGUP = 8, KGDN = 16, KOP = 32, VQKV = 64, VOP = 128,
  UNEMB = 256, FWUNEMB = 512, PRIOR = 1024, PAKET = 2048, ALL = 4095
};
inline int g_mask = 0;
inline int g_fast = 0;              // the groups of g_mask that run on the fast kernels (0: reference path)
inline int g_dn_bits = 16;          // width of the mlp.down code
inline float g_dn_mult = 1.0f / 256.0f;   // 2^-(g_dn_bits - 8)
inline float g_dn_max = 32767.0f;   // largest mlp.down code
inline int g_dn_min = 1;            // mlp.down codes below this become 0 (NEIF_ACT16_DNMIN; 1: none)

// read once, in the constructor of the engine (before any fork of a pipeline)
inline void init() {
  static const char* const kName[12] = {"mlpup", "mlpdn", "kqkv", "kgup", "kgdn", "kop",
                                        "vqkv", "vop", "unemb", "fwunemb", "prior", "paket"};
  const char* v = std::getenv("NEIF_ACT16");
  int m = 0;
  if (v && *v) {
    if (*v >= '0' && *v <= '9') {
      m = static_cast<int>(std::strtol(v, nullptr, 0)) & ALL;
    } else {
      while (*v) {
        const char* e = v;
        while (*e && *e != ',') e++;
        const size_t n = static_cast<size_t>(e - v);
        bool hit = false;
        if (n == 3 && std::memcmp(v, "all", 3) == 0) { m |= ALL; hit = true; }
        for (int k = 0; k < 12; k++)
          if (std::strlen(kName[k]) == n && std::memcmp(v, kName[k], n) == 0) { m |= 1 << k; hit = true; }
        if (!hit) { std::fprintf(stderr, "neif_act16: unknown group in NEIF_ACT16\n"); std::exit(2); }
        v = *e ? e + 1 : e;
      }
    }
  }
  g_mask = m;
  // NEIF_ACT16_DNBITS = 9..16: width of the mlp.down code (default 16): scale
  // s / 2^(bits - 8), codes in [0, 2^(bits - 1) - 1].  Reference and fast path.
  const char* b = std::getenv("NEIF_ACT16_DNBITS");
  if (b && *b) {
    const int nb = std::atoi(b);
    if (nb < 9 || nb > 16) { std::fprintf(stderr, "neif_act16: NEIF_ACT16_DNBITS must be 9..16\n"); std::exit(2); }
    g_dn_bits = nb;
    g_dn_mult = 1.0f / static_cast<float>(1 << (nb - 8));
    g_dn_max = static_cast<float>((1 << (nb - 1)) - 1);
  }
  // NEIF_ACT16_DNMIN = t: mlp.down codes below t are set to 0 (fewer columns for
  // the sparse kernel; t = 2^(bits - 9) leaves about the columns of the 8-bit code)
  const char* mn = std::getenv("NEIF_ACT16_DNMIN");
  if (mn && *mn) {
    g_dn_min = std::atoi(mn);
    if (g_dn_min < 1 || g_dn_min > 32767) { std::fprintf(stderr, "neif_act16: NEIF_ACT16_DNMIN must be 1..32767\n"); std::exit(2); }
  }
  // fast kernels (below) for kop, vop and mlpdn when no other group is chosen;
  // NEIF_ACT16_SLOW=1 keeps the reference path for checking
  const char* sl = std::getenv("NEIF_ACT16_SLOW");
  const bool slow = sl && *sl && *sl != '0';
  g_fast = (!slow && (m & ~(KOP | VOP | MLPDN)) == 0) ? m : 0;
#ifdef NEIF_PACKED
  if (m && !g_fast) { std::fprintf(stderr, "neif_act16: a NEIF_PACKED build has only the fast kernels (kop, vop, mlpdn)"); std::fputc(10, stderr); std::exit(2); }
#endif
  if (m) std::fprintf(stderr, "neif_act16: mask %d, fast kernels for %d, mlp.down code %d bits, codes below %d dropped\n", m, g_fast, g_dn_bits, g_dn_min);
}

constexpr float K256 = 1.0f / 256.0f;   // exact: a scale times K256 is the scale with exponent - 8

// 16-bit code of x[0, n) with scale s/256: q = rne(clamp(x / (s/256), -32768, 32767)); n % 8 == 0
inline void quant(const float* x, float s, int16_t* q, int n) {
  const __m256 vs = _mm256_set1_ps(s * K256);
  const __m256 vlo = _mm256_set1_ps(-32768.0f), vhi = _mm256_set1_ps(32767.0f);
  for (int i = 0; i < n; i += 8) {
    __m256 t = _mm256_div_ps(_mm256_loadu_ps(x + i), vs);
    t = _mm256_min_ps(_mm256_max_ps(t, vlo), vhi);
    const __m256i v = _mm256_cvtps_epi32(t);
    const __m128i w = _mm_packs_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(q + i), w);
  }
}
// the 8-bit code of the entry (biased u8) as signed int16, so that an 8-bit
// input can go through dense() where the OUTPUT of a matmul is a 16-bit site
inline void from_u8(const uint8_t* q8, int16_t* q, int n) {
  for (int i = 0; i < n; i++) q[i] = static_cast<int16_t>(static_cast<int>(q8[i]) - 128);
}

// y[r] = float(sum_i a[i] * w[r][i]) * (scale[r] * mult) for r in [0, rows_padded),
// over the entry's dense arena (8-row groups: 8 x int32 corr | 8 x fp32 scale |
// weights chunk-major, 32 columns x 8 rows per chunk).  a: stride int16, pads 0.
// mult = K256 for a 16-bit input, 1 for an 8-bit one (then y is bit-identical
// to the entry's epilogue (a): the same int32 dot, the same one multiply).
// Bound: |a| <= 32768, |w| <= 7, 256 columns: |dot| < 5.9e7.
__attribute__((noinline)) inline void dense(const QDense& m, const int16_t* a, float mult, float* y) {
  const int NC = m.nchunk;
  const ptrdiff_t GB = 64 + static_cast<ptrdiff_t>(NC) * 256;
  const uint8_t* gp = m.arena;
  const __m256 vm = _mm256_set1_ps(mult);
  for (int g = 0; g < m.ngroups; g++, gp += GB) {
    alignas(32) int32_t d[8];
    for (int r = 0; r < 8; r++) {
      __m256i acc = _mm256_setzero_si256();
      for (int c = 0; c < NC; c++) {
        const uint8_t* w = gp + 64 + c * 256 + r * 32;
        const __m256i w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w)));
        const __m256i w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + 16)));
        const __m256i a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 32 * c));
        const __m256i a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 32 * c + 16));
        acc = _mm256_add_epi32(acc, _mm256_add_epi32(_mm256_madd_epi16(w0, a0), _mm256_madd_epi16(w1, a1)));
      }
      __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
      s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
      s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
      d[r] = _mm_cvtsi128_si32(s);
    }
    const __m256 sc = _mm256_mul_ps(_mm256_loadu_ps(reinterpret_cast<const float*>(gp + 32)), vm);
    _mm256_storeu_ps(y + 8 * g,
                     _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_load_si256(reinterpret_cast<const __m256i*>(d))), sc));
  }
}

// out[r] (+)= float(sum_c q[c] * w[r][c]) * (fold[r] * mult), r in [0, 192), over
// the entry's int4 column arena (column c: 96 bytes, low nibble = row b + 7,
// high nibble = row 96 + b + 7); q[c] in [0, 32767], zeros skipped.
// Bound: 14 * 32767 * 1024 < 4.7e8.
__attribute__((noinline)) inline void sparse(const QSparse4& m, const int16_t* q, int d_in, float mult, float* out,
                                             bool add) {
  alignas(32) int32_t acc[192];
  std::memset(acc, 0, sizeof(acc));
  const __m128i m0F = _mm_set1_epi8(0x0F);
  int64_t sum = 0;
  for (int c = 0; c < d_in; c++) {
    if (q[c] == 0) continue;
    sum += q[c];
    const __m256i vv = _mm256_set1_epi32(q[c]);   // int16 pairs (q, 0)
    const uint8_t* col = m.cols + static_cast<size_t>(c) * 128;
    for (int b = 0; b < 96; b += 8) {
      const __m128i x = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(col + b));
      const __m256i lo = _mm256_cvtepu8_epi32(_mm_and_si128(x, m0F));
      const __m256i hi = _mm256_cvtepu8_epi32(_mm_and_si128(_mm_srli_epi16(x, 4), m0F));
      __m256i* al = reinterpret_cast<__m256i*>(acc + b);
      __m256i* ah = reinterpret_cast<__m256i*>(acc + 96 + b);
      _mm256_store_si256(al, _mm256_add_epi32(_mm256_load_si256(al), _mm256_madd_epi16(lo, vv)));
      _mm256_store_si256(ah, _mm256_add_epi32(_mm256_load_si256(ah), _mm256_madd_epi16(hi, vv)));
    }
  }
  const __m256i corr = _mm256_set1_epi32(static_cast<int32_t>(7 * sum));
  const __m256 vm = _mm256_set1_ps(mult);
  for (int k = 0; k < 192; k += 8) {
    const __m256i d = _mm256_sub_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(acc + k)), corr);
    __m256 y = _mm256_mul_ps(_mm256_cvtepi32_ps(d), _mm256_mul_ps(_mm256_loadu_ps(m.fold + k), vm));
    if (add) y = _mm256_add_ps(_mm256_loadu_ps(out + k), y);
    _mm256_storeu_ps(out + k, y);
  }
}

// the entry's 8-bit quantizers on a float row, as the fused epilogues compute
// them: (d) t = clamp(y / s, -128, 127), q = rne(t) + 128
inline void quant_bias_u8(const float* y, float s, uint8_t* q, int n) {
  const __m256 vs = _mm256_set1_ps(s);
  const __m256 vlo = _mm256_set1_ps(-128.0f), vhi = _mm256_set1_ps(127.0f);
  for (int i = 0; i < n; i += 8) {
    __m256 t = _mm256_div_ps(_mm256_loadu_ps(y + i), vs);
    t = _mm256_min_ps(_mm256_max_ps(t, vlo), vhi);
    alignas(32) int32_t v[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(v), _mm256_cvtps_epi32(t));
    for (int k = 0; k < 8; k++) q[i + k] = static_cast<uint8_t>(v[k] + 128);
  }
}
// (c) h = max(y, 0)^2, t = min(h / s, 127), q = rne(t); the ascending index
// list of the nonzero codes (16 in-range slack entries behind it); returns nnz
inline int relu2_u8(const float* y, float s, uint8_t* q, uint16_t* idx, int n) {
  const __m256 vs = _mm256_set1_ps(s);
  const __m256 v127 = _mm256_set1_ps(127.0f), vz = _mm256_setzero_ps();
  int nnz = 0;
  for (int i = 0; i < n; i += 8) {
    const __m256 h = _mm256_max_ps(_mm256_loadu_ps(y + i), vz);
    const __m256 t = _mm256_min_ps(_mm256_div_ps(_mm256_mul_ps(h, h), vs), v127);
    alignas(32) int32_t v[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(v), _mm256_cvtps_epi32(t));
    for (int k = 0; k < 8; k++) {
      q[i + k] = static_cast<uint8_t>(v[k]);
      if (v[k]) idx[nnz++] = static_cast<uint16_t>(i + k);
    }
  }
  const uint16_t fill = nnz ? idx[nnz - 1] : 0;
  for (int k = 0; k < 16; k++) idx[nnz + k] = fill;
  return nnz;
}
// the 16-bit relu^2 code: h = max(y, 0)^2, q = rne(min(h / (s/256), 32767))
inline void relu2_16(const float* y, float s, int16_t* q, int n) {
  const __m256 vs = _mm256_set1_ps(s * g_dn_mult);
  const __m256 vhi = _mm256_set1_ps(g_dn_max), vz = _mm256_setzero_ps();
  for (int i = 0; i < n; i += 8) {
    const __m256 h = _mm256_max_ps(_mm256_loadu_ps(y + i), vz);
    const __m256 t = _mm256_min_ps(_mm256_div_ps(_mm256_mul_ps(h, h), vs), vhi);
    const __m256i v = _mm256_cvtps_epi32(t);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(q + i),
                     _mm_packs_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1)));
  }
  for (int i = 0; i < n; i++)   // NEIF_ACT16_DNMIN
    if (q[i] < g_dn_min) q[i] = 0;
}


// ===========================================================================
// Fast kernels (AVX2) for kop / vop (dense 192 x 192, added to the residual)
// and mlpdn (sparse int4 columns).  Same int32 dots as dense() / sparse()
// above and the same float epilogue, so the results are bit-identical to the
// reference path (checked by the coder output of a chunk run).
// ===========================================================================

// 16-bit code of x[0, 192) (as quant()), split into its low byte (unsigned)
// and its high byte + 128 (the biased form the entry's dense arenas correct
// for with corr = 128 * sum of the row's weights)
inline void quant_split192(const float* x, float s, uint8_t* lo, uint8_t* hib) {
  const __m256 vs = _mm256_set1_ps(s * K256);
  const __m256 vlo = _mm256_set1_ps(-32768.0f), vhi = _mm256_set1_ps(32767.0f);
  const __m128i mff = _mm_set1_epi16(0x00FF), b80 = _mm_set1_epi8(static_cast<char>(0x80));
  for (int i = 0; i < 192; i += 16) {
    const __m256i v0 = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(_mm256_div_ps(_mm256_loadu_ps(x + i), vs), vlo), vhi));
    const __m256i v1 = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(_mm256_div_ps(_mm256_loadu_ps(x + i + 8), vs), vlo), vhi));
    const __m128i h0 = _mm_packs_epi32(_mm256_castsi256_si128(v0), _mm256_extracti128_si256(v0, 1));
    const __m128i h1 = _mm_packs_epi32(_mm256_castsi256_si128(v1), _mm256_extracti128_si256(v1, 1));
    _mm_store_si128(reinterpret_cast<__m128i*>(lo + i), _mm_packus_epi16(_mm_and_si128(h0, mff), _mm_and_si128(h1, mff)));
    _mm_store_si128(reinterpret_cast<__m128i*>(hib + i),
                    _mm_xor_si128(_mm_packs_epi16(_mm_srai_epi16(h0, 8), _mm_srai_epi16(h1, 8)), b80));
  }
}

// the int32 sums of one 8-row group (6 chunks of 32 columns) for the unsigned
// activation bytes act: the vpmaddubsw chain of the entry's run_g8.  16-bit
// bound: 6 * 2 * 255 * 7 = 21420.
inline __m256i dense_acc8(const uint8_t* gp, const uint8_t* act, bool pf) {
  const __m256i ones = _mm256_set1_epi16(1);
  const __m256i* w = reinterpret_cast<const __m256i*>(gp + 64);
  __m256i av = _mm256_load_si256(reinterpret_cast<const __m256i*>(act));
  __m256i s0 = _mm256_maddubs_epi16(av, w[0]), s1 = _mm256_maddubs_epi16(av, w[1]);
  __m256i s2 = _mm256_maddubs_epi16(av, w[2]), s3 = _mm256_maddubs_epi16(av, w[3]);
  __m256i s4 = _mm256_maddubs_epi16(av, w[4]), s5 = _mm256_maddubs_epi16(av, w[5]);
  __m256i s6 = _mm256_maddubs_epi16(av, w[6]), s7 = _mm256_maddubs_epi16(av, w[7]);
  if (pf)
    for (int k = 0; k < 4; k++) _mm_prefetch(reinterpret_cast<const char*>(gp) + 6144 + 64 * k, _MM_HINT_T0);
#pragma clang loop unroll(disable)
  for (int c = 1; c < 6; c++) {
    const __m256i* wc = w + c * 8;
    if (pf)
      for (int k = 0; k < 4; k++)
        _mm_prefetch(reinterpret_cast<const char*>(gp) + c * 256 + 6144 + 64 * k, _MM_HINT_T0);
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
  const __m256i p01 = _mm256_hadd_epi32(_mm256_madd_epi16(s0, ones), _mm256_madd_epi16(s1, ones));
  const __m256i p23 = _mm256_hadd_epi32(_mm256_madd_epi16(s2, ones), _mm256_madd_epi16(s3, ones));
  const __m256i p45 = _mm256_hadd_epi32(_mm256_madd_epi16(s4, ones), _mm256_madd_epi16(s5, ones));
  const __m256i p67 = _mm256_hadd_epi32(_mm256_madd_epi16(s6, ones), _mm256_madd_epi16(s7, ones));
  const __m256i q0 = _mm256_hadd_epi32(p01, p23), q1 = _mm256_hadd_epi32(p45, p67);
  return _mm256_add_epi32(_mm256_permute2x128_si256(q0, q1, 0x20), _mm256_permute2x128_si256(q0, q1, 0x31));
}

// kop / vop: out[r] += float(sum_c q[c] * w[r][c]) * (scale[r] / 256), q the
// 16-bit code of x with the site scale s.  Two vpmaddubsw passes per 8-row
// group over the entry's arena: low bytes, then biased high bytes (the group
// is in L1 by then); dot = lo + 256 * (hi_biased - corr).
#ifdef NEIF_PACKED
// NEIF_PACKED build (qmat_dense.cpp): the same two byte passes over the
// nibble-packed arena (8-row groups, 3 x 64 columns).  A pass gives
// acc = sum_c byte[c] * (w[r][c] + 7); with S = sum of the pass's bytes
//   low bytes:          sum lo * w = acc - 7 S_lo
//   biased high bytes:  sum hi * w = acc - 7 S_hib - corr[r]   (corr = 128 sum w)
// so the int32 dot, and with it every float, is the one of the unpacked form.
inline __m256i dense_acc8p(const uint8_t* gp, const uint8_t* act, bool pf) {
  const __m256i m0F = _mm256_set1_epi8(0x0F);
  const __m256i ones = _mm256_set1_epi16(1);
  const __m256i* w = reinterpret_cast<const __m256i*>(gp + 64);
  __m256i s[8];
  for (int r = 0; r < 8; r++) s[r] = _mm256_setzero_si256();
  for (int p = 0; p < 3; p++) {
    const __m256i* wc = w + p * 8;
    if (pf)
      for (int k = 0; k < 4; k++)
        _mm_prefetch(reinterpret_cast<const char*>(gp) + p * 256 + 6144 + 64 * k, _MM_HINT_T0);
    const __m256i a0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(act + 64 * p));
    const __m256i a1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(act + 64 * p + 32));
    for (int r = 0; r < 8; r++) {
      const __m256i wv = _mm256_load_si256(wc + r);
      const __m256i t = _mm256_add_epi16(_mm256_maddubs_epi16(a0, _mm256_and_si256(wv, m0F)),
                                         _mm256_maddubs_epi16(a1, _mm256_and_si256(_mm256_srli_epi16(wv, 4), m0F)));
      s[r] = _mm256_add_epi32(s[r], _mm256_madd_epi16(t, ones));
    }
  }
  const __m256i q0 = _mm256_hadd_epi32(_mm256_hadd_epi32(s[0], s[1]), _mm256_hadd_epi32(s[2], s[3]));
  const __m256i q1 = _mm256_hadd_epi32(_mm256_hadd_epi32(s[4], s[5]), _mm256_hadd_epi32(s[6], s[7]));
  return _mm256_add_epi32(_mm256_permute2x128_si256(q0, q1, 0x20), _mm256_permute2x128_si256(q0, q1, 0x31));
}
inline int32_t bytes_sum192(const uint8_t* a) {
  __m256i s = _mm256_setzero_si256();
  for (int i = 0; i < 192; i += 32)
    s = _mm256_add_epi64(s, _mm256_sad_epu8(_mm256_load_si256(reinterpret_cast<const __m256i*>(a + i)),
                                            _mm256_setzero_si256()));
  const __m128i t = _mm_add_epi64(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
  return static_cast<int32_t>(_mm_cvtsi128_si64(t) + _mm_extract_epi64(t, 1));
}
__attribute__((noinline)) inline void op_fast(const QDense& m, float s, const float* x, float* out) {
  alignas(32) uint8_t lo[192], hib[192];
  quant_split192(x, s, lo, hib);
  const __m256 vm = _mm256_set1_ps(K256);
  const __m256i cl = _mm256_set1_epi32(7 * bytes_sum192(lo)), ch = _mm256_set1_epi32(7 * bytes_sum192(hib));
  const uint8_t* gp = m.arena;
  for (int g = 0; g < m.ngroups; g++, gp += 64 + 3 * 256) {
    const __m256i sl = _mm256_sub_epi32(dense_acc8p(gp, lo, true), cl);
    const __m256i sh = _mm256_sub_epi32(_mm256_sub_epi32(dense_acc8p(gp, hib, false), ch),
                                        _mm256_load_si256(reinterpret_cast<const __m256i*>(gp)));
    const __m256i sums = _mm256_add_epi32(sl, _mm256_slli_epi32(sh, 8));
    const __m256 y = _mm256_mul_ps(_mm256_cvtepi32_ps(sums), _mm256_mul_ps(_mm256_load_ps(reinterpret_cast<const float*>(gp + 32)), vm));
    _mm256_storeu_ps(out + 8 * g, _mm256_add_ps(_mm256_loadu_ps(out + 8 * g), y));
  }
}
#else
__attribute__((noinline)) inline void op_fast(const QDense& m, float s, const float* x, float* out) {
  alignas(32) uint8_t lo[192], hib[192];
  quant_split192(x, s, lo, hib);
  const __m256 vm = _mm256_set1_ps(K256);
  const uint8_t* gp = m.arena;
  for (int g = 0; g < m.ngroups; g++, gp += 64 + 6 * 256) {
    const __m256i sl = dense_acc8(gp, lo, true);
    const __m256i sh = _mm256_sub_epi32(dense_acc8(gp, hib, false), _mm256_load_si256(reinterpret_cast<const __m256i*>(gp)));
    const __m256i sums = _mm256_add_epi32(sl, _mm256_slli_epi32(sh, 8));
    const __m256 y = _mm256_mul_ps(_mm256_cvtepi32_ps(sums), _mm256_mul_ps(_mm256_load_ps(reinterpret_cast<const float*>(gp + 32)), vm));
    _mm256_storeu_ps(out + 8 * g, _mm256_add_ps(_mm256_loadu_ps(out + 8 * g), y));
  }
}
#endif

// the entry's 8-bit relu^2 code with its index list (the arithmetic of
// relu2q_phase2 in qmat_dense.cpp), for the fast-weight head; n % 32 == 0
inline int relu2_u8_fast(const float* y, float s, uint8_t* q, uint16_t* idx, int n) {
  const __m256 vs = _mm256_set1_ps(s), v127 = _mm256_set1_ps(127.0f), vz = _mm256_setzero_ps();
  for (int i = 0; i < n; i += 16) {
    const __m256 y0 = _mm256_max_ps(_mm256_load_ps(y + i), vz), y1 = _mm256_max_ps(_mm256_load_ps(y + i + 8), vz);
    const __m256i q0 = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_div_ps(_mm256_mul_ps(y0, y0), vs), v127));
    const __m256i q1 = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_div_ps(_mm256_mul_ps(y1, y1), vs), v127));
    const __m256i q16 = _mm256_permute4x64_epi64(_mm256_packs_epi32(q0, q1), _MM_SHUFFLE(3, 1, 2, 0));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(q + i),
                     _mm_packus_epi16(_mm256_castsi256_si128(q16), _mm256_extracti128_si256(q16, 1)));
  }
  int nnz = 0;
  for (int b = 0; b < n; b += 32) {
    const __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(q + b));
    const uint32_t mk = ~static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_setzero_si256())));
    for (int k = 0; k < 4; k++) {
      const uint32_t m8 = (mk >> (8 * k)) & 0xFF;
      const __m128i lut = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(g_bitpos.t[m8]));
      _mm_storeu_si128(reinterpret_cast<__m128i*>(idx + nnz),
                       _mm_add_epi16(_mm_cvtepu8_epi16(lut), _mm_set1_epi16(static_cast<short>(b + 8 * k))));
      nnz += __builtin_popcount(m8);
    }
  }
  return nnz;
}

// the wide relu^2 code (as relu2_16) with two ascending index lists: the
// positions whose low byte is not zero and those whose high byte is not zero
// (each with 16 in-range entries behind it; room for n + 32), and the sum of
// the codes.  n % 16 == 0.
inline void relu2_wide(const float* y, float s, int n, uint16_t* c, uint16_t* idx_lo, int* n_lo, uint16_t* idx_hi,
                       int* n_hi, int32_t* sum) {
  const __m256 vs = _mm256_set1_ps(s * g_dn_mult), vmax = _mm256_set1_ps(g_dn_max), vz = _mm256_setzero_ps();
  const __m256i ones = _mm256_set1_epi16(1);
  const __m256i vmin = _mm256_set1_epi16(static_cast<short>(g_dn_min - 1));
  __m256i vsum = _mm256_setzero_si256();
  int nl = 0, nh = 0;
  for (int i = 0; i < n; i += 16) {
    const __m256 y0 = _mm256_max_ps(_mm256_load_ps(y + i), vz), y1 = _mm256_max_ps(_mm256_load_ps(y + i + 8), vz);
    const __m256i q0 = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_div_ps(_mm256_mul_ps(y0, y0), vs), vmax));
    const __m256i q1 = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_div_ps(_mm256_mul_ps(y1, y1), vs), vmax));
    __m256i v = _mm256_permute4x64_epi64(_mm256_packs_epi32(q0, q1), _MM_SHUFFLE(3, 1, 2, 0));   // 16 codes
    v = _mm256_and_si256(v, _mm256_cmpgt_epi16(v, vmin));   // codes below g_dn_min -> 0
    _mm256_store_si256(reinterpret_cast<__m256i*>(c + i), v);
    vsum = _mm256_add_epi32(vsum, _mm256_madd_epi16(v, ones));
    const uint32_t nz = ~static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_setzero_si256())));
    if (nz == 0) continue;
    const uint32_t ml = _pext_u32(nz, 0x55555555u), mh = _pext_u32(nz, 0xAAAAAAAAu);
    for (int k = 0; k < 2; k++) {
      const uint32_t m8 = (ml >> (8 * k)) & 0xFF;
      const __m128i lut = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(g_bitpos.t[m8]));
      _mm_storeu_si128(reinterpret_cast<__m128i*>(idx_lo + nl),
                       _mm_add_epi16(_mm_cvtepu8_epi16(lut), _mm_set1_epi16(static_cast<short>(i + 8 * k))));
      nl += __builtin_popcount(m8);
    }
    if (mh)
      for (int k = 0; k < 2; k++) {
        const uint32_t m8 = (mh >> (8 * k)) & 0xFF;
        const __m128i lut = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(g_bitpos.t[m8]));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(idx_hi + nh),
                         _mm_add_epi16(_mm_cvtepu8_epi16(lut), _mm_set1_epi16(static_cast<short>(i + 8 * k))));
        nh += __builtin_popcount(m8);
      }
  }
  const __m128i f0 = _mm_setzero_si128();
  _mm_storeu_si128(reinterpret_cast<__m128i*>(idx_lo + nl), f0);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(idx_lo + nl + 8), f0);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(idx_hi + nh), f0);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(idx_hi + nh + 8), f0);
  __m128i t = _mm_add_epi32(_mm256_castsi256_si128(vsum), _mm256_extracti128_si256(vsum, 1));
  t = _mm_add_epi32(t, _mm_shuffle_epi32(t, 0x4E));
  t = _mm_add_epi32(t, _mm_shuffle_epi32(t, 0xB1));
  *sum = _mm_cvtsi128_si32(t);
  *n_lo = nl;
  *n_hi = nh;
}

// One byte plane of the wide code over the int4 column arena: for the listed
// columns i, acc[r] += (byte(i) * u[r][i]) << SHIFT with u = w + 7 the stored
// nibble.  The kernel of the entry's sparse4_accum with the operands swapped:
// the activation byte pair is the UNSIGNED operand of vpmaddubsw (so a full
// byte 0..255 fits) and the nibbles 0..14 the signed one.  Per column pair a
// lane grows by at most 2 * 255 * 14 = 7140 (low bytes) or 2 * 127 * 14 = 3556
// (high bytes); the int16 accumulators start at -32768 and are widened every
// FLUSH pairs (9 * 7140 = 64260, 18 * 3556 = 64008, both <= 65535).
// byte(i) = bytes[2 * i] (bytes points at the low or the high byte of code 0).
#define A16_BLOCK(b, AL0, AL1, AH0, AH1)                               \
  {                                                                    \
    __m256i w = _mm256_load_si256(c0 + (b));                           \
    __m256i x = _mm256_load_si256(c1 + (b));                           \
    const __m256i wl = _mm256_and_si256(w, m0F);                       \
    const __m256i xl = _mm256_and_si256(x, m0F);                       \
    AL0 = _mm256_add_epi16(AL0, _mm256_maddubs_epi16(av, _mm256_unpacklo_epi8(wl, xl))); \
    AL1 = _mm256_add_epi16(AL1, _mm256_maddubs_epi16(av, _mm256_unpackhi_epi8(wl, xl))); \
    w = _mm256_and_si256(_mm256_srli_epi16(w, 4), m0F);                \
    x = _mm256_and_si256(_mm256_srli_epi16(x, 4), m0F);                \
    AH0 = _mm256_add_epi16(AH0, _mm256_maddubs_epi16(av, _mm256_unpacklo_epi8(w, x))); \
    AH1 = _mm256_add_epi16(AH1, _mm256_maddubs_epi16(av, _mm256_unpackhi_epi8(w, x))); \
  }
#define A16_PAIR()                 \
  {                                \
    A16_BLOCK(0, a0, a1, a2, a3)   \
    A16_BLOCK(1, a4, a5, a6, a7)   \
    A16_BLOCK(2, a8, a9, a10, a11) \
  }
#define A16_FLUSH1(A, base_lo, base_hi)                                                         \
  {                                                                                             \
    const __m256i ux = _mm256_xor_si256(A, vbias);                                              \
    const __m256i lo32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(ux)), SHIFT); \
    const __m256i hi32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(ux, 1)), SHIFT); \
    _mm256_store_si256(reinterpret_cast<__m256i*>(acc + (base_lo)),                             \
                       _mm256_add_epi32(lo32, _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + (base_lo))))); \
    _mm256_store_si256(reinterpret_cast<__m256i*>(acc + (base_hi)),                             \
                       _mm256_add_epi32(hi32, _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + (base_hi))))); \
    A = vbias;                                                                                  \
  }
#define A16_FLUSH_ALL()         \
  {                             \
    A16_FLUSH1(a0, 0, 16)       \
    A16_FLUSH1(a1, 8, 24)       \
    A16_FLUSH1(a2, 96, 112)     \
    A16_FLUSH1(a3, 104, 120)    \
    A16_FLUSH1(a4, 32, 48)      \
    A16_FLUSH1(a5, 40, 56)      \
    A16_FLUSH1(a6, 128, 144)    \
    A16_FLUSH1(a7, 136, 152)    \
    A16_FLUSH1(a8, 64, 80)      \
    A16_FLUSH1(a9, 72, 88)      \
    A16_FLUSH1(a10, 160, 176)   \
    A16_FLUSH1(a11, 168, 184)   \
  }
template <int FLUSH, int SHIFT, bool PF>
inline void plane_accum(const uint8_t* __restrict cols, const uint8_t* __restrict bytes, const uint16_t* __restrict idx,
                        int nnz, int32_t* __restrict acc) {
  const __m256i m0F = _mm256_set1_epi8(0x0F);
  const __m256i vbias = _mm256_set1_epi16(static_cast<short>(0x8000));
  __m256i a0 = vbias, a1 = vbias, a2 = vbias, a3 = vbias, a4 = vbias, a5 = vbias, a6 = vbias, a7 = vbias, a8 = vbias,
          a9 = vbias, a10 = vbias, a11 = vbias;
  const int npairs = nnz / 2;
  int p = 0;
  while (p < npairs) {
    const int end = p + FLUSH < npairs ? p + FLUSH : npairs;
    for (; p < end; p++) {
      const uint32_t i0 = idx[2 * p], i1 = idx[2 * p + 1];
      if (PF) {
        const char* f0 = reinterpret_cast<const char*>(cols) + 128u * idx[2 * (p + 4)];
        const char* f1 = reinterpret_cast<const char*>(cols) + 128u * idx[2 * (p + 4) + 1];
        _mm_prefetch(f0, _MM_HINT_T0);
        _mm_prefetch(f0 + 64, _MM_HINT_T0);
        _mm_prefetch(f1, _MM_HINT_T0);
        _mm_prefetch(f1 + 64, _MM_HINT_T0);
      }
      const __m256i* c0 = reinterpret_cast<const __m256i*>(cols + 128u * i0);
      const __m256i* c1 = reinterpret_cast<const __m256i*>(cols + 128u * i1);
      const __m256i av = _mm256_set1_epi16(static_cast<short>(bytes[2 * i0] | (bytes[2 * i1] << 8)));
      A16_PAIR()
    }
    A16_FLUSH_ALL()
  }
  if (nnz & 1) {   // the last column with itself, second byte 0
    const uint32_t i0 = idx[nnz - 1];
    const __m256i* c0 = reinterpret_cast<const __m256i*>(cols + 128u * i0);
    const __m256i* c1 = c0;
    const __m256i av = _mm256_set1_epi16(static_cast<short>(bytes[2 * i0]));
    A16_PAIR()
    A16_FLUSH_ALL()
  }
}
#undef A16_BLOCK
#undef A16_PAIR
#undef A16_FLUSH1
#undef A16_FLUSH_ALL

// mlpdn: out[r] += float(sum_c q[c] * w[r][c]) * (fold[r] * g_dn_mult) for the
// wide code c of relu2_wide: the low-byte plane over its list (columns fetched
// from memory, with prefetch), then the high-byte plane over its shorter list,
// shifted by 8; the nibble offset 7 * sum(q) comes off once.
// Bound: 14 * 32767 * 1024 < 4.7e8.
__attribute__((noinline)) inline void sparse_fast(const QSparse4& m, const uint16_t* c, const uint16_t* idx_lo, int n_lo,
                                                  const uint16_t* idx_hi, int n_hi, int32_t sum, float* out) {
  alignas(32) int32_t acc[192];
  std::memset(acc, 0, sizeof(acc));
  for (int k = 0; k < 12; k++) {   // the epilogue's inputs, as the entry's kernel warms them
    _mm_prefetch(reinterpret_cast<const char*>(m.fold) + 64 * k, _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(out) + 64 * k, _MM_HINT_T0);
  }
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(c);
  plane_accum<9, 0, true>(m.cols, bytes, idx_lo, n_lo, acc);
  if (n_hi) plane_accum<18, 8, false>(m.cols, bytes + 1, idx_hi, n_hi, acc);
  const __m256i corr = _mm256_set1_epi32(7 * sum);
  const __m256 vm = _mm256_set1_ps(g_dn_mult);
  for (int k = 0; k < 192; k += 8) {
    const __m256i d = _mm256_sub_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(acc + k)), corr);
    const __m256 y = _mm256_mul_ps(_mm256_cvtepi32_ps(d), _mm256_mul_ps(_mm256_loadu_ps(m.fold + k), vm));
    _mm256_storeu_ps(out + k, _mm256_add_ps(_mm256_loadu_ps(out + k), y));
  }
}

}  // namespace act16
}  // namespace opt
}  // namespace fx2

#endif  // NEIF_TEST
