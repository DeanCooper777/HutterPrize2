// NEIF_TEST: state of the chunk ("split") evaluation mode.  NOT part of the
// entry: this header is included only under `#ifdef NEIF_TEST`, and a build
// without that macro is byte-identical to the published comp9.
//
// The mode (`cmix -t STREAM OUT`, src/runner.cpp NeifTestMain) codes the coder
// stream of a `-e enwik9` run from a page boundary in the middle of the
// stream:
//   * bytes [0, warm_from): a structure-only pass.  Every automaton that reads
//     the stream (metadata block reader, text start, author/census/alias
//     injections, grammar automaton, the transformer's prompt, gate and
//     link-context automata, redirect and alias tables) sees every byte and
//     every injected byte exactly as in a real run; no model is run and
//     nothing is coded (g_skip).
//   * bytes [warm_from, end): the complete predictor and the arithmetic coder,
//     starting from cold models; the cost in bits of every coded bit is taken
//     from the 32-bit probability handed to the interval split and summed
//     for [warm_from, end) and for [start, end) (g_count).
#ifndef NEIF_TEST_H_
#define NEIF_TEST_H_
#ifdef NEIF_TEST

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <signal.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <immintrin.h>
#include <atomic>

namespace neif_test {

inline bool g_on = false;         // -t mode is running
inline bool g_skip = false;       // structure-only pass: no model step, nothing coded
inline bool g_count = false;      // the stream byte being coded lies in [start, end)
inline bool g_hash_on = false;    // structure events of this byte enter g_hash
inline unsigned long long g_start = 0, g_end = 0;
inline unsigned long long g_warm_target = 0;   // skip at least up to here, then to the next page start
inline long long g_warm_from = -1;             // first byte the model ran on
inline double g_bits_all = 0.0, g_bits_kept = 0.0;        // sum of -log2 P(coded bit)
inline unsigned long long g_n_all = 0, g_n_kept = 0;      // coded bits
inline unsigned long long g_zw_mismatch = 0;   // forced byte != stream byte (must stay 0)
inline unsigned long long g_tf_rows = 0;       // transformer rows fed in the hashed range
// all rows since the start of the stream (structure-only pass included), the rows the network ran on, and the read rows of those
inline unsigned long long g_tf_rows_all = 0, g_tf_rows_model = 0, g_tf_rows_read = 0;
inline uint64_t g_hash = 1469598103934665603ull;          // FNV-1a over structure events

// one structure event (a value that must not depend on any model state)
inline void H(unsigned int v) {
  if (!g_hash_on) return;
  g_hash = (g_hash ^ (uint64_t)(v & 0xffffu)) * 1099511628211ull;
}

// cost of one coded bit; p32 = P(bit == 1) in units of 2^-32, exactly the
// value floor_stage::Mid() splits the interval with.  noinline: as a plain
// call it adds no branch to Encoder::Encode(), whose control flow must stay
// the one of the entry for the profile-guided build (a changed flow voids the
// profile of that function, and the test build then no longer computes the
// same bits as comp9)
#ifdef NEIF_PAR
namespace par { inline void CostBit(uint32_t p32, int bit, double c); }   // NEIF_PAR, below
#endif
__attribute__((noinline)) inline void Cost(uint32_t p32, int bit) {
  const double p1 = (double)p32 * (1.0 / 4294967296.0);
  const double c = -log2(bit ? p1 : 1.0 - p1);
  g_bits_all += c;
  ++g_n_all;
  if (g_count) { g_bits_kept += c; ++g_n_kept; }
#ifdef NEIF_PAR
  par::CostBit(p32, bit, c);   // the coded bits of the stream byte: cost, and whether the more probable value came
#endif
}

// Second cost sum, "tap": the cost of the same coded bits at the return value
// of Predictor::Predict(), that is behind the mixers, SSE and the invariants
// and in front of BITLSTM (bit_head::Adjust), the floor stage and the
// semantic-cache correction.  Called from Encoder::Encode() only, for exactly
// the bits Cost() is called for, so bits that are not coded (ChrForced,
// ObserveKnownBit) cost nothing in both sums and the two bit counts are equal.
// p = P(bit == 1) as the float Predict() returned; a probability of exactly 0
// for the coded bit is counted in g_tap_clamped and costs 32 bits (the
// resolution of the coder) instead of infinity.  noinline for the reason
// given at Cost().
inline double g_tap_all = 0.0, g_tap_kept = 0.0;          // sum of -log2 P_tap(coded bit)
inline unsigned long long g_tap_n_all = 0, g_tap_n_kept = 0;   // bits in the tap sums
inline unsigned long long g_tap_clamped = 0;              // bits whose tap probability was 0
__attribute__((noinline)) inline void CostTap(float p, int bit) {
  const double p1 = (double)p;
  double q = bit ? p1 : 1.0 - p1;
  if (!(q >= 1.0 / 4294967296.0)) { q = 1.0 / 4294967296.0; ++g_tap_clamped; }
  const double c = -log2(q);
  g_tap_all += c;
  ++g_tap_n_all;
  if (g_count) { g_tap_kept += c; ++g_tap_n_kept; }
}

// ---- profile (NEIF_PROF=1): where the time of the model bytes goes ----------
// T(k) is called where section k begins (the marks in Predictor::Predict(),
// Predictor::Perceive(), Encoder::Encode() and Compress()); the time since the
// last mark goes to the section that mark began.  A plain call of a noinline
// function: no branch in the caller, so the caller keeps its PGO profile entry.
#ifdef NEIF_PAR
// NEIF_PAR: section 16 is split off section 11 (src/coder/encoder.cpp), section
// 17 off section 1 (src/predictor.cpp), which is then fxcm alone
constexpr int kProfN = 18;
#else
constexpr int kProfN = 16;
#endif
inline const char* const kProfName[kProfN] = {
    "coder, runner, stream automata", "predict: fxcm first half, selectors, cache readout",
    "predict: model outputs to layer-0 row", "predict: mixers, SSE",
    "perceive: cmix models", "perceive: mixer and SSE update",
    "perceive: adaptive depth, contexts, byte updates", "PPMd byte update",
    "transformer driver and network", "transformer slots, path state", "fxcm perceive",
    "BITLSTM", "floor stage, end stages", "end stages learn", "semantic cache byte start",
    "pipeline: waiting for the queue"
#ifdef NEIF_PAR
    , "BITLSTM behind the coded bit (Observe)", "predict: selectors from the row, cache readout (off the second line)"
#endif
    };
inline bool g_prof = false;
inline unsigned long long g_prof_t[kProfN] = {0}, g_prof_last = 0;
inline int g_prof_cur = 0;

#ifdef NEIF_PAR
// ---- NEIF_PAR=1 (mode -t, one process; a build with -DNEIF_TEST=1 -DNEIF_PAR=1.
// A tool of its own: without the macro NEIF_PAR the chunk test build is not
// changed by any of these blocks) ----------------------------------------------
// Measurements for the design of a multi-threaded decoder (our project,
// runs/neif-par/DESIGN.md).  The decoder knows a stream byte only behind its
// eight bits, and the transformer step of that byte gives the row the bits of
// the NEXT byte are mixed with; so the questions are which work of a byte
// could run beside which other work, and how often the next byte can be
// guessed.  The compressor of mode -t walks the same call sequence as the
// decoder and knows every byte, so it can measure both.  What is taken:
//   * time: every profile mark T(k) also books its ticks by the position of
//     the bit in its model byte (slot 0..7; slot 8 = behind the last
//     Perceive() of a model byte up to the next Predict()), separately for
//     stream bytes and for injected model bytes (g_sum);
//   * per stream byte (a "cycle": from the byte-start hook to the next one)
//     the time stamps at which each bit is known and of its transformer step,
//     the fxcm time and the other main-thread time between two bits, the PPMd
//     time, and the rows that step fed (Row(), from Tf1Engine::Impl::Feed);
//   * guesses: the byte-level predictions that exist at the byte start
//     (ByteStart(), from Predictor::NeifParByteStart) against the byte that
//     came, as hit counts by region; and for six guess policies the number K
//     of decoded bits behind which the policy holds the right byte when it
//     guesses again each time a decoded bit rules its guess out;
//   * a simulation per cycle with these measured times: the length of the
//     cycle if the transformer step of the next byte started at the right
//     guess on other threads (factors kFv on its speed), with and without
//     fxcm on a thread of its own (EndCycle()).
// Nothing here changes a value the coder computes.  NEIF_PAR_JSON names the
// result file (default ./neif_par.json), NEIF_PAR_H the cost of one hand-off
// between two threads in time-stamp ticks (default 1500), NEIF_PAR_COMMIT the
// work that stays behind a right guess (default 10000 ticks: the fast-weight
// steps and the second head need the byte itself).
namespace par {
constexpr int kSlot = 9;      // bits 0..7 of a model byte; 8 = behind its last Perceive()
constexpr int kPol = 6;       // guess policies (kPolName)
constexpr int kFn = 9;        // speed factors of the transformer step in the simulation
constexpr int kReg = 7;       // 0 = no gate zone and a live row, 1..5 = gate zone, 6 = no zone, no live row
constexpr int kSrc = 11;      // byte-level predictions at the byte start (kSrcName)
constexpr int kCell = 128;    // contexts of the selector
constexpr int kBlk = 256;     // series: blocks of 65536 stream bytes
constexpr int kHb = 8;        // histograms: bins per octave of ticks
constexpr int kHn = 48 * kHb;
constexpr int kStep = 5;      // classes of a driver step (kStepName)
inline const double kFv[kFn] = {1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0};
inline const char* const kPolName[kPol] = {"tf_top1", "tfxfw_top1", "selector_top1", "tf_top2", "selector_top2",
                                           "oracle"};
inline const char* const kSrcName[kSrc] = {"tf_top1", "tf_top2", "fw_top1", "ppmd_top1", "match_byte", "cache_top1",
                                           "tfxfw_top1", "tfxfw_top2", "selector_top1", "selector_top2",
                                           "any_of_first_six"};
inline const char* const kStepName[kStep] = {"no row (gate, timestamp)", "one read stream row (plain)",
                                             "one other row", "two or more rows", "alias row (no bit loop)"};

inline bool g_par = false;                 // NEIF_PAR=1
inline unsigned long long g_h = 1500;      // NEIF_PAR_H: one hand-off, ticks
inline unsigned long long g_cm = 10000;    // NEIF_PAR_COMMIT: the work that stays behind a right guess, ticks
inline unsigned long long g_ab = 15000;    // NEIF_PAR_ABORT: giving up a wrong guess (stop the step, undo PPMd), ticks

// --- time by slot and section ---
inline int g_slot = 8;                     // slot the running time belongs to
inline int g_kind = 1;                     // model byte the running time belongs to: 0 stream byte, 1 injected
inline bool g_next_stream = false;         // the next model byte is the stream byte of the hook
inline unsigned long long g_sum[2][kSlot][kProfN];
inline unsigned long long g_nmodel[2];     // model bytes by kind
inline unsigned long long g_sync_err = 0;  // a Predict() where none was due (must stay 0)
inline unsigned long long g_hook_ticks = 0;   // time inside ByteStart(), taken out of the sections

// --- rows of the transformer (Row()) ---
inline unsigned long long g_rows[2][2];    // [inserted][read]
inline unsigned long long g_unread = 0;    // unread rows since the last read row (the engine's queue, or more)
inline unsigned int g_st_rows = 0, g_st_read = 0, g_st_ins = 0;   // of the running driver step
inline unsigned long long g_step_n[kStep], g_step_t[kStep], g_step_rows[kStep];

// --- one stream byte ---
struct Cyc {
  unsigned long long t0;          // end of the byte-start hook: the cycle begins
  unsigned long long t_bit[8];    // start of Perceive() of bit j: the bit is known
  unsigned long long t_s0, t_s1;  // the driver step of this byte (in Perceive() of bit 7)
  unsigned long long tf_extra;    // transformer time of the cycle outside that step
  unsigned long long inj;         // injected model bytes of the cycle, without their transformer time
  unsigned long long ppmd;        // PPMd byte update of this byte
  unsigned long long row0;        // layer-0 row of bit 0 (model outputs)
  unsigned long long fx[8];       // fxcm in front of the mixing of bit j: Perceive of the bit before, first half of Predict
  unsigned long long mp[8];       // other main-thread work between "bit j-1 known" and Predict of bit j
  unsigned long long sem14;       // of mp[0]: the cache's byte start in front of this byte
  unsigned long long tprev, ppmd_prev;   // transformer step and PPMd update of the stream byte in front
  double cost;                    // bits
  int K[kPol];
  int reg, byte, forced, elig, have_step, plain, rows, valid, nbit;
  unsigned char hit[kSrc];
  unsigned char pair[3];          // top-2 of tf, tf x fw, selector
};
inline Cyc g_cyc;
inline unsigned long long g_fx_carry = 0, g_mp_carry = 0, g_sem_carry = 0;

// --- the coded bits of the running stream byte (CostBit()) ---
inline int g_cb_n = 0, g_cb_ok = 1, g_cb_first = 8;
inline double g_cb_cost = 0.0;
inline void CostBit(uint32_t p32, int bit, double c) {
  if (!g_par) return;
  // the more probable value of the final probability: 1 from 2^31 on
  if (g_cb_ok && bit != (p32 >= 0x80000000u ? 1 : 0)) { g_cb_ok = 0; g_cb_first = g_cb_n; }
  ++g_cb_n;
  g_cb_cost += c;
}

// one row fed to the network (not in the structure-only pass)
inline void Row(bool inserted, bool gelesen) {
  if (!g_par) return;
  ++g_rows[inserted ? 1 : 0][gelesen ? 1 : 0];
  ++g_st_rows;
  if (gelesen) ++g_st_read;
  if (inserted) ++g_st_ins;
  if (gelesen) g_unread = 0; else ++g_unread;
}

// a profile mark: dt ticks end here; they belong to section g_prof_cur, slot
// g_slot of a model byte of kind g_kind; then the mark k begins
inline void Tick(int k, unsigned long long t, unsigned long long dt) {
  const int sec = g_prof_cur, slot = g_slot;
  g_sum[g_kind][slot][sec] += dt;
  Cyc& c = g_cyc;
  if (g_kind == 0 && slot < 8) {            // the bits of the stream byte
    if (sec == 1) c.fx[slot] += dt;
    else if (sec == 10) { if (slot < 7) c.fx[slot + 1] += dt; else g_fx_carry += dt; }
    else if (sec == 4 || sec == 5 || sec == 6 || sec == 9 || sec == 13 || sec == 16) {
      if (slot < 7) c.mp[slot + 1] += dt; else g_mp_carry += dt;
    }
    else if (sec == 7) c.ppmd += dt;
    else if (sec == 2 && slot == 0) c.row0 += dt;
  } else if (sec == 8) {                    // a transformer step outside the step of the stream byte
    c.tf_extra += dt;
  } else if (slot == 8) {                   // between two model bytes: automata, the cache's next byte start
    g_mp_carry += dt;
    if (sec == 14) g_sem_carry += dt;
  } else {                                  // the bits of an injected model byte
    c.inj += dt;
  }
  if (sec == 8) {                           // a driver step ends: its class
    int cls = g_st_rows == 0 ? 0 : (g_st_rows == 1 ? ((g_st_read == 1 && g_st_ins == 0) ? 1 : 2) : 3);
    if (k != 9) cls = 4;                    // not the step in Perceive(): Predictor::ObserveTfOnlyByte
    ++g_step_n[cls];
    g_step_t[cls] += dt;
    g_step_rows[cls] += g_st_rows;
    if (g_kind == 0 && slot == 7 && k == 9) {
      c.t_s1 = t;
      c.have_step = 1;
      c.plain = cls == 1;
      c.rows = (int)g_st_rows;
    }
  }
  if (k == 8) {                             // a driver step begins
    g_st_rows = g_st_read = g_st_ins = 0;
    if (g_kind == 0 && slot == 7) c.t_s0 = t;
  } else if (k == 4) {                      // Perceive() begins: the bit is known
    if (g_kind == 0 && slot < 8) { c.t_bit[slot] = t; c.nbit = slot + 1; }
  } else if (k == 1) {                      // Predict() begins: the next bit, or the next model byte
    if (slot == 8) {
      g_kind = g_next_stream ? 0 : 1;
      g_next_stream = false;
      g_slot = 0;
      ++g_nmodel[g_kind];
    } else if (slot < 7) {
      g_slot = slot + 1;
    } else {
      ++g_sync_err;
    }
  } else if (k == 0 && sec == 10 && slot == 7) {   // Perceive() of the last bit ends
    g_slot = 8;
  }
}
inline void Reset() {
  g_slot = 8;
  g_kind = 1;
  g_next_stream = false;
  memset(g_sum, 0, sizeof g_sum);
  g_nmodel[0] = g_nmodel[1] = 0;
  memset(&g_cyc, 0, sizeof g_cyc);
  g_fx_carry = g_mp_carry = g_sem_carry = 0;
  g_hook_ticks = 0;
}
}  // namespace par
#endif

__attribute__((noinline)) inline void T(int k) {
  if (!g_prof) return;
  const unsigned long long t = __builtin_ia32_rdtsc();
#ifdef NEIF_PAR
  if (par::g_par) par::Tick(k, t, t - g_prof_last);
#endif
  g_prof_t[g_prof_cur] += t - g_prof_last;
  g_prof_last = t;
  g_prof_cur = k;
}
inline void ProfReset() {
  for (int k = 0; k < kProfN; ++k) g_prof_t[k] = 0;
  g_prof_last = __builtin_ia32_rdtsc();
  g_prof_cur = 0;
#ifdef NEIF_PAR
  par::Reset();   // the measurements start with the profile, at the first byte the model runs on
#endif
}
// the table, scaled so that the sections add up to `seconds`
inline void ProfPrint(FILE* f, const char* who, double seconds) {
  if (!g_prof) return;
  T(0);
  unsigned long long sum = 0;
  for (int k = 0; k < kProfN; ++k) sum += g_prof_t[k];
  if (!sum) return;
  for (int k = 0; k < kProfN; ++k)
    if (g_prof_t[k])
      fprintf(f, "neif_prof %s %5.1f%% %8.1f s  %s\n", who, 100.0 * (double)g_prof_t[k] / (double)sum,
              seconds * (double)g_prof_t[k] / (double)sum, kProfName[k]);
}

#ifdef NEIF_PAR
// ---- NEIF_PAR, second part: the stream byte (guesses, cycle, result file) ----
}  // namespace neif_test
namespace fx2 { namespace opt { void prof_reset(); } }   // src/tf1/opt/model_opt.cpp: the per-site table starts again
namespace neif_test {
namespace par {
// what the predictor hands over at the start of a stream byte
// (Predictor::NeifParByteStart, behind Predictor::SemByteStart)
struct Src {
  const float* tf;    // row of the transformer slot, 256 entries by byte value
  const float* fw;    // row of the fast-weight slot
  const float* pp;    // PPMd row
  int live;           // the transformer slot was set from a row (Tf1Slot::KtxLive), not neutral, not off
  unsigned int zone;  // gate zone of this byte (0 = none)
  int mbyte, mlen;    // longest match of the transformer's match models: its byte, its length in bits (-1, 0: none)
  int cache;          // top-1 byte of the semantic cache (-1: no neighbours)
  int forced;         // the byte the prompt automaton forces (-1: none): known without decoding
};

inline unsigned long long g_ncyc = 0, g_n_step = 0, g_n_plain = 0, g_n_elig = 0, g_n_spec = 0;
inline unsigned long long g_coded_hist[9];         // stream bytes by their number of coded bits
inline unsigned long long g_reg_n[kReg], g_reg_forced[kReg], g_reg_free[kReg], g_reg_greedy[kReg];
inline double g_reg_cost[kReg];                    // bits
inline unsigned long long g_src_hit[kReg][kSrc], g_pair_hit[kReg][3];   // bytes that are not forced
inline unsigned long long g_khist[kReg][kPol][9], g_khist_spec[kPol][9];
inline unsigned long long g_gmiss[9];              // bytes with 8 coded bits: first bit the less probable value came (8: none)
inline double g_bitknown[8];                       // cycles with a plain step: sum of (bit j known - cycle start)
inline double g_sim_base = 0, g_sim_a = 0, g_sim_f_opt = 0, g_sim_f_cons = 0;
inline double g_sim_d[kPol][kFn], g_sim_df[kPol][kFn];
inline double g_t_step = 0, g_t_spec = 0, g_t_extra = 0, g_w_spec = 0, g_ppmd = 0, g_inj = 0, g_row0 = 0, g_sem14 = 0;
inline double g_fx[8], g_mp[8], g_ovl[8];
inline unsigned long long g_h_cyc[kHn], g_h_step[kHn], g_h_w[kHn];
struct Blk { unsigned long long n, tf1, x1, sel1, tf2, ticks, tf_ticks; double cost; };
inline Blk g_blk[kBlk];
inline float g_rate[kCell][6];                     // selector: running hit rate of the first six sources per context
inline bool g_rate_init = false;
inline bool g_started = false;                     // the first kept byte has come
inline unsigned long long g_warm_ticks = 0;        // ticks of the model bytes in front of it

// histogram bin of v ticks: kHb bins per octave
inline int HBin(unsigned long long v) {
  if (v < 2) return 0;
  const int e = 63 - __builtin_clzll(v);
  const int fr = e >= 3 ? (int)((v >> (e - 3)) & 7u) : (int)((v << (3 - e)) & 7u);
  const int b = e * kHb + fr;
  return b < kHn ? b : kHn - 1;
}
// the best byte of the score s among the bytes that share the first k bits
// with c, other than ex (-1: none excluded); ties to the lower byte; -1 if
// no byte is left
inline int Best(const float* s, int c, int k, int ex) {
  const int lo = (c >> (8 - k)) << (8 - k), hi = lo + (256 >> k);
  int b = -1;
  for (int i = lo; i < hi; ++i)
    if (i != ex && (b < 0 || s[i] > s[b])) b = i;
  return b;
}
// first bit (0 = most significant) in which two bytes differ; 8: equal
inline int Diff(int a, int b) {
  const unsigned int x = (unsigned int)(a ^ b) & 255u;
  return x ? __builtin_clz(x) - 24 : 8;
}
// One guesser on the score s: it starts with the best byte and guesses again
// (the best byte the decoded bits allow) when a decoded bit rules its guess
// out.  Returns the number of decoded bits at its last guess, the right one.
inline int SimK1(const float* s, int c) {
  int k = 0;
  for (;;) {
    const int g = Best(s, c, k, -1);
    if (g == c) return k;
    k = Diff(g, c) + 1;
  }
}
// Two guessers that never hold the same byte; the one whose guess is ruled
// out first guesses again.
inline int SimK2(const float* s, int c) {
  int ga = Best(s, c, 0, -1), gb = Best(s, c, 0, ga), ka = 0, kb = 0;
  for (;;) {
    if (ga == c) return ka;
    if (gb == c) return kb;
    const int da = Diff(ga, c), db = Diff(gb, c);
    if (da <= db) { ka = da + 1; ga = Best(s, c, ka, gb); }
    else { kb = db + 1; gb = Best(s, c, kb, ga); }
  }
}

// The stream byte in g_cyc is complete (the next byte-start hook, or the end
// of the run): its counts, and the simulation of its cycle.  All times in
// ticks.  The measured cycle is  W + T + post:  W from the cycle start to the
// transformer step of this byte (the eight bits, the updates of bit 7 up to
// PPMd), T the step, post the rest (slots, fxcm, automata, injected bytes,
// the cache's byte start of the next byte).
//   A     the fxcm part and the layer-0 row of bit 0 run under the step of
//         the byte in front (both read nothing of the transformer)
//   F     fxcm on a thread of its own: between two bits the shorter of
//         {fxcm} and {the other work of the main thread} runs under the
//         longer; "opt": for bit 0 the other work includes PPMd and the
//         step, "cons": only the cache's byte start (fxcm starts behind the
//         stream automata)
//   D     the step of this byte runs on other threads at 1/f of its time; it
//         starts at the cycle start if the policy's first guess is the byte,
//         else when the bit that gives the right guess is known, and it needs
//         the PPMd update of the guessed byte first; the main thread then
//         takes PPMd from there.  Only where the step is one read stream row
//         and no unread row waited at the cycle start; every other step runs
//         at 1/f where it stands.
//   DF    D and fxcm on a thread of its own (the bit times shorten in
//         proportion)
// h = one hand-off, cm = the work behind a right guess.
inline void EndCycle(unsigned long long t_end) {
  Cyc& c = g_cyc;
  if (!c.valid) return;
  c.valid = 0;
  const double cyc = (double)(t_end - c.t0), h = (double)g_h, cm = (double)g_cm;
  const int reg = c.reg;
  ++g_ncyc;
  ++g_coded_hist[g_cb_n > 8 ? 8 : g_cb_n];
  ++g_reg_n[reg];
  g_reg_cost[reg] += g_cb_cost;
  if (c.forced >= 0) {
    ++g_reg_forced[reg];
  } else {
    for (int q = 0; q < kSrc; ++q) g_src_hit[reg][q] += c.hit[q];
    for (int q = 0; q < 3; ++q) g_pair_hit[reg][q] += c.pair[q];
  }
  if (g_cb_n == 8) {
    ++g_reg_free[reg];
    if (g_cb_ok) ++g_reg_greedy[reg];
    ++g_gmiss[g_cb_ok ? 8 : g_cb_first];
  }
  for (int p = 0; p < kPol; ++p) ++g_khist[reg][p][c.K[p]];

  const bool step = c.have_step && c.nbit == 8 && c.t_s0 > c.t0;
  const double T = step ? (double)(c.t_s1 - c.t_s0) : 0.0;
  const double W = step ? (double)(c.t_s0 - c.t0) : 0.0;
  const double te = (double)c.tf_extra, pm = (double)c.ppmd;
  const bool spec = step && c.plain && c.elig;
  g_n_step += step ? 1 : 0;
  g_n_plain += (step && c.plain) ? 1 : 0;
  g_n_elig += c.elig ? 1 : 0;
  g_n_spec += spec ? 1 : 0;
  g_t_step += T;
  g_t_extra += te;
  g_ppmd += pm;
  g_inj += (double)c.inj;
  g_row0 += (double)c.row0;
  g_sem14 += (double)c.sem14;
  if (spec) {
    g_t_spec += T;
    g_w_spec += W;
    for (int j = 0; j < 8; ++j) g_bitknown[j] += (double)(c.t_bit[j] - c.t0);
  }
  ++g_h_cyc[HBin(t_end - c.t0)];
  if (step) { ++g_h_step[HBin(c.t_s1 - c.t_s0)]; ++g_h_w[HBin(c.t_s0 - c.t0)]; }

  // fxcm beside the main thread, bit by bit
  double sv = 0.0;
  for (int j = 1; j < 8; ++j) {
    const double o = (double)(c.fx[j] < c.mp[j] ? c.fx[j] : c.mp[j]);
    sv += o;
    g_ovl[j] += o;
  }
  const double f0 = (double)c.fx[0], m0 = (double)c.mp[0];
  const double m0b = m0 + (double)c.ppmd_prev + (double)c.tprev;
  const double o0d = f0 < m0 ? f0 : m0;                               // the step is off the main thread
  const double o0b = f0 < m0b ? f0 : m0b;                             // the step is on the main thread
  const double o0c = f0 < (double)c.sem14 ? f0 : (double)c.sem14;     // fxcm starts behind the automata
  g_ovl[0] += o0d;
  for (int j = 0; j < 8; ++j) { g_fx[j] += (double)c.fx[j]; g_mp[j] += (double)c.mp[j]; }

  g_sim_base += cyc;
  {
    double a = f0 + (double)c.row0;
    if (a > (double)c.tprev) a = (double)c.tprev;
    a -= 2.0 * h;
    g_sim_a += cyc - (a > 0.0 ? a : 0.0);
  }
  {
    double a = sv + o0b - 16.0 * h;
    g_sim_f_opt += cyc - (a > 0.0 ? a : 0.0);
    a = sv + o0c - 16.0 * h;
    g_sim_f_cons += cyc - (a > 0.0 ? a : 0.0);
  }
  double svd = sv + o0d - 16.0 * h;
  if (svd < 0.0) svd = 0.0;
  for (int p = 0; p < kPol; ++p) {
    const int K = c.K[p];
    if (spec) ++g_khist_spec[p][K];
    // a guess that was not the first one starts behind its bit and behind the end of the wrong one
    const double L = (!spec || K == 0) ? 0.0 : (double)(c.t_bit[K - 1] - c.t0) + (double)g_ab;
    for (int i = 0; i < kFn; ++i) {
      const double f = kFv[i];
      double d, d2;
      if (spec) {
        const double post = cyc - W - T - te;
        double a = W - pm, b = L + pm + h + T / f;
        d = (a > b ? a : b) + h + cm + post + te / f;
        const double Wd = W - svd > 0.25 * W ? W - svd : 0.25 * W;
        a = Wd - pm;
        b = L * (Wd / W) + pm + h + T / f;
        d2 = (a > b ? a : b) + h + cm + post + te / f;
      } else {
        d = cyc - T - te + (T + te) / f;
        d2 = d - svd;
      }
      g_sim_d[p][i] += d;
      g_sim_df[p][i] += d2;
    }
  }

  const unsigned long long b = (g_ncyc - 1) >> 16;
  if (b < (unsigned long long)kBlk) {
    Blk& k = g_blk[b];
    ++k.n;
    if (c.forced < 0) { k.tf1 += c.hit[0]; k.x1 += c.hit[6]; k.sel1 += c.hit[8]; k.tf2 += c.pair[0]; }
    k.ticks += t_end - c.t0;
    k.tf_ticks += (unsigned long long)(T + te);
    k.cost += g_cb_cost;
  }
}

// At the start of a stream byte, behind the cache's byte start and in front
// of the first Predict(): the cycle of the byte in front ends, the guesses for
// this byte (the byte is `byte`; the decoder would not know it).
__attribute__((noinline)) inline void ByteStart(const Src& s, int byte) {
  if (!g_par || !g_prof) return;
  // The measurement covers the kept bytes [start, end) only: at the first of
  // them the profile and all counts start again, so with NEIF_PAR=1 the
  // "neif_prof" table and the per-site table of an FX2_PROF build are tables
  // of the kept bytes, behind the warm-up.
  if (!g_count) return;
  if (!g_started) {
    g_started = true;
    T(g_prof_cur);
    for (int k = 0; k < kProfN; ++k) g_warm_ticks += g_prof_t[k];
    memset(g_rows, 0, sizeof g_rows);
    memset(g_step_n, 0, sizeof g_step_n);
    memset(g_step_t, 0, sizeof g_step_t);
    memset(g_step_rows, 0, sizeof g_step_rows);
    fx2::opt::prof_reset();
    ProfReset();
  }
  T(g_prof_cur);                                   // the time up to here belongs to the running section
  const unsigned long long t_in = g_prof_last;
  EndCycle(t_in);
  const Cyc& o = g_cyc;                            // the byte in front (zero at the first)
  Cyc n;
  memset(&n, 0, sizeof n);
  n.valid = 1;
  n.byte = byte;
  n.forced = s.forced;
  n.reg = s.zone ? (s.zone < 5u ? (int)s.zone : 5) : (s.live ? 0 : 6);
  n.elig = g_unread == 0 ? 1 : 0;
  n.fx[0] = g_fx_carry;
  n.mp[0] = g_mp_carry;
  n.sem14 = g_sem_carry;
  n.tprev = o.have_step ? o.t_s1 - o.t_s0 : 0;
  n.ppmd_prev = o.ppmd;

  // scores: the transformer row (the PPMd row where the slot has none), and
  // the product of the transformer row and the fast-weight row
  float s_tf[256], s_x[256], s_sel[256];
  const float* row = s.live ? s.tf : s.pp;
  float mx = 0.0f;
  double sum = 0.0;
  for (int i = 0; i < 256; ++i) {
    s_tf[i] = row[i];
    sum += (double)row[i];
    s_x[i] = s.live ? s.tf[i] * s.fw[i] : s.pp[i];
    if (s_x[i] > mx) mx = s_x[i];
  }
  if (!(mx > 0.0f)) {   // no fast-weight row: the first score
    for (int i = 0; i < 256; ++i) { s_x[i] = s_tf[i]; if (s_x[i] > mx) mx = s_x[i]; }
  }
  int cand[6];
  cand[0] = Best(s_tf, 0, 0, -1);
  cand[1] = Best(s_tf, 0, 0, cand[0]);
  cand[2] = s.live ? Best(s.fw, 0, 0, -1) : -1;
  cand[3] = Best(s.pp, 0, 0, -1);
  cand[4] = s.mlen > 0 ? s.mbyte : -1;
  cand[5] = s.cache;
  const int x1 = Best(s_x, 0, 0, -1), x2 = Best(s_x, 0, 0, x1);

  // selector: the context is the certainty class of the first score's best
  // byte, the match length class and whether the fast-weight and the PPMd
  // top-1 agree with it; per context the running hit rate of each source.
  // The score puts the sources in front of all other bytes, in the order of
  // their rates.  It only steers which byte is guessed, so it need not be the
  // same on two machines.
  static const float kLim[7] = {0.3f, 0.5f, 0.7f, 0.8f, 0.9f, 0.95f, 0.98f};
  const float p1 = sum > 0.0 ? (float)((double)s_tf[cand[0]] / sum) : 0.0f;
  int conf = 0;
  while (conf < 7 && p1 >= kLim[conf]) ++conf;
  const int mcl = s.mlen <= 0 ? 0 : (s.mlen < 32 ? 1 : (s.mlen < 96 ? 2 : 3));
  const int cell = ((conf * 4 + mcl) * 2 + (cand[2] == cand[0] ? 1 : 0)) * 2 + (cand[3] == cand[0] ? 1 : 0);
  if (!g_rate_init) {
    static const float kStart[6] = {0.6f, 0.1f, 0.5f, 0.4f, 0.4f, 0.3f};
    for (int a = 0; a < kCell; ++a)
      for (int q = 0; q < 6; ++q) g_rate[a][q] = kStart[q];
    g_rate_init = true;
  }
  float* r = g_rate[cell];
  for (int i = 0; i < 256; ++i) s_sel[i] = mx > 0.0f ? s_x[i] / mx : 0.0f;   // at most 1
  for (int q = 0; q < 6; ++q)
    if (cand[q] >= 0 && 2.0f + r[q] > s_sel[cand[q]]) s_sel[cand[q]] = 2.0f + r[q];
  const int e1 = Best(s_sel, 0, 0, -1), e2 = Best(s_sel, 0, 0, e1);

  int any = 0;
  for (int q = 0; q < 6; ++q) { n.hit[q] = cand[q] == byte ? 1 : 0; any |= n.hit[q]; }
  n.hit[6] = x1 == byte ? 1 : 0;
  n.hit[7] = x2 == byte ? 1 : 0;
  n.hit[8] = e1 == byte ? 1 : 0;
  n.hit[9] = e2 == byte ? 1 : 0;
  n.hit[10] = (unsigned char)any;
  n.pair[0] = n.hit[0] | n.hit[1];
  n.pair[1] = n.hit[6] | n.hit[7];
  n.pair[2] = n.hit[8] | n.hit[9];
  if (s.forced < 0) {
    n.K[0] = SimK1(s_tf, byte);
    n.K[1] = SimK1(s_x, byte);
    n.K[2] = SimK1(s_sel, byte);
    n.K[3] = SimK2(s_tf, byte);
    n.K[4] = SimK2(s_sel, byte);
    n.K[5] = 0;
    for (int q = 0; q < 6; ++q)   // the selector learns behind its guess
      if (cand[q] >= 0) r[q] += ((cand[q] == byte ? 1.0f : 0.0f) - r[q]) * (1.0f / 32.0f);
  }

  g_cyc = n;
  g_cb_n = 0;
  g_cb_ok = 1;
  g_cb_first = 8;
  g_cb_cost = 0.0;
  g_fx_carry = g_mp_carry = g_sem_carry = 0;
  g_next_stream = true;
  // the time of this hook is no time of the coder
  const unsigned long long t_out = __builtin_ia32_rdtsc();
  g_hook_ticks += t_out - t_in;
  g_prof_last = t_out;
  g_cyc.t0 = t_out;
}

inline void JU(FILE* f, const unsigned long long* v, int n) {
  fputc('[', f);
  for (int i = 0; i < n; ++i) fprintf(f, "%s%llu", i ? ", " : "", v[i]);
  fputc(']', f);
}
inline void JD(FILE* f, const double* v, int n) {
  fputc('[', f);
  for (int i = 0; i < n; ++i) fprintf(f, "%s%.3f", i ? ", " : "", v[i]);
  fputc(']', f);
}
inline void JS(FILE* f, const char* const* v, int n) {
  fputc('[', f);
  for (int i = 0; i < n; ++i) fprintf(f, "%s\"%s\"", i ? ", " : "", v[i]);
  fputc(']', f);
}
inline void JH(FILE* f, const unsigned long long* hst) {   // the bins that are not empty: [bin, count]
  fputc('[', f);
  bool first = true;
  for (int i = 0; i < kHn; ++i)
    if (hst[i]) { fprintf(f, "%s[%d, %llu]", first ? "" : ", ", i, hst[i]); first = false; }
  fputc(']', f);
}

// of `seconds` (the wall time of all model bytes) the part of the kept bytes,
// which is what the sections of the profile hold in this mode
inline double KeptSeconds(double seconds) {
  if (!g_par || !g_started) return seconds;
  T(g_prof_cur);
  unsigned long long s = 0;
  for (int k = 0; k < kProfN; ++k) s += g_prof_t[k];
  const double all = (double)g_warm_ticks + (double)s + (double)g_hook_ticks;
  return all > 0.0 ? seconds * (double)s / all : seconds;
}

// the result file; seconds = wall time of the model bytes (the ticks of the
// warm-up, of the sections and of the hook add up to it)
inline void Json(FILE* f, double seconds, unsigned long long warm_from, unsigned long long end) {
  if (!g_par) return;
  T(g_prof_cur);
  EndCycle(g_prof_last);
  unsigned long long sec_sum = 0;
  for (int k = 0; k < kProfN; ++k) sec_sum += g_prof_t[k];
  fprintf(f, "{\"neif_par\": 1, \"warm_from\": %llu, \"end\": %llu, \"seconds_model\": %.3f,\n", warm_from, end, seconds);
  // seconds_model covers ticks_warm (the model bytes in front of the kept ones) + ticks_sections + ticks_hook
  fprintf(f, " \"ticks_warm\": %llu, \"ticks_sections\": %llu, \"ticks_hook\": %llu, \"handoff_ticks\": %llu,\n"
             " \"commit_ticks\": %llu, \"abort_ticks\": %llu, \"sync_errors\": %llu,\n",
          g_warm_ticks, sec_sum, g_hook_ticks, g_h, g_cm, g_ab, g_sync_err);
  fprintf(f, " \"sec_names\": "); JS(f, kProfName, kProfN);
  fprintf(f, ",\n \"sec_ticks\": "); JU(f, g_prof_t, kProfN);
  // ticks by [kind: stream byte, injected model byte][slot: bit 0..7, behind the byte][section]
  fprintf(f, ",\n \"slot_sec_ticks\": [");
  for (int a = 0; a < 2; ++a) {
    fprintf(f, "%s[", a ? ",\n  " : "");
    for (int sl = 0; sl < kSlot; ++sl) { if (sl) fprintf(f, ",\n   "); JU(f, g_sum[a][sl], kProfN); }
    fputc(']', f);
  }
  fprintf(f, "],\n \"model_bytes\": "); JU(f, g_nmodel, 2);
  fprintf(f, ",\n \"stream_bytes\": %llu, \"coded_bits_hist\": ", g_ncyc); JU(f, g_coded_hist, 9);
  fprintf(f, ",\n \"rows_stream_unread_read\": "); JU(f, g_rows[0], 2);
  fprintf(f, ", \"rows_inserted_unread_read\": "); JU(f, g_rows[1], 2);
  fprintf(f, ",\n \"step_names\": "); JS(f, kStepName, kStep);
  fprintf(f, ",\n \"step_n\": "); JU(f, g_step_n, kStep);
  fprintf(f, ", \"step_ticks\": "); JU(f, g_step_t, kStep);
  fprintf(f, ", \"step_rows\": "); JU(f, g_step_rows, kStep);
  // cycles: with a step in Perceive(), of them plain (one read stream row),
  // cycles without an unread row waiting at their start, both ("spec")
  fprintf(f, ",\n \"cycles\": {\"n\": %llu, \"with_step\": %llu, \"plain\": %llu, \"eligible\": %llu, \"spec\": %llu,\n"
             "  \"ticks\": %.0f, \"step_ticks\": %.0f, \"spec_step_ticks\": %.0f, \"spec_w_ticks\": %.0f,\n"
             "  \"tf_extra_ticks\": %.0f, \"ppmd_ticks\": %.0f, \"injected_ticks\": %.0f, \"row0_ticks\": %.0f,\n"
             "  \"cache_start_ticks\": %.0f,\n",
          g_ncyc, g_n_step, g_n_plain, g_n_elig, g_n_spec, g_sim_base, g_t_step, g_t_spec, g_w_spec, g_t_extra,
          g_ppmd, g_inj, g_row0, g_sem14);
  fprintf(f, "  \"bit_known_ticks\": "); JD(f, g_bitknown, 8);
  fprintf(f, ",\n  \"fxcm_ticks\": "); JD(f, g_fx, 8);
  fprintf(f, ",\n  \"main_ticks\": "); JD(f, g_mp, 8);
  fprintf(f, ",\n  \"overlap_ticks\": "); JD(f, g_ovl, 8);
  fprintf(f, "},\n \"region_names\": [\"no zone, live row\", \"zone 1\", \"zone 2\", \"zone 3\", \"zone 4\", "
             "\"zone 5 and above\", \"no zone, no live row\"],\n");
  fprintf(f, " \"region_n\": "); JU(f, g_reg_n, kReg);
  fprintf(f, ",\n \"region_forced\": "); JU(f, g_reg_forced, kReg);
  fprintf(f, ",\n \"region_8_coded_bits\": "); JU(f, g_reg_free, kReg);
  fprintf(f, ",\n \"region_greedy_path\": "); JU(f, g_reg_greedy, kReg);
  fprintf(f, ",\n \"region_cost_bits\": "); JD(f, g_reg_cost, kReg);
  fprintf(f, ",\n \"greedy_first_miss_hist\": "); JU(f, g_gmiss, 9);
  fprintf(f, ",\n \"src_names\": "); JS(f, kSrcName, kSrc);
  fprintf(f, ",\n \"src_hits\": [");   // [region][source], bytes that are not forced
  for (int a = 0; a < kReg; ++a) { if (a) fprintf(f, ",\n   "); JU(f, g_src_hit[a], kSrc); }
  fprintf(f, "],\n \"pair_names\": [\"tf_top2\", \"tfxfw_top2\", \"selector_top2\"],\n \"pair_hits\": [");
  for (int a = 0; a < kReg; ++a) { if (a) fprintf(f, ", "); JU(f, g_pair_hit[a], 3); }
  fprintf(f, "],\n \"pol_names\": "); JS(f, kPolName, kPol);
  fprintf(f, ",\n \"k_hist\": [");     // [region][policy][K = 0..8]
  for (int a = 0; a < kReg; ++a) {
    fprintf(f, "%s[", a ? ",\n   " : "");
    for (int p = 0; p < kPol; ++p) { if (p) fprintf(f, ", "); JU(f, g_khist[a][p], 9); }
    fputc(']', f);
  }
  fprintf(f, "],\n \"k_hist_spec\": [");   // [policy][K], cycles with a plain step
  for (int p = 0; p < kPol; ++p) { if (p) fprintf(f, ", "); JU(f, g_khist_spec[p], 9); }
  fprintf(f, "],\n \"f_values\": "); JD(f, kFv, kFn);
  fprintf(f, ",\n \"sim\": {\"base\": %.0f, \"A\": %.0f, \"F_opt\": %.0f, \"F_cons\": %.0f,\n  \"D\": [", g_sim_base,
          g_sim_a, g_sim_f_opt, g_sim_f_cons);
  for (int p = 0; p < kPol; ++p) { if (p) fprintf(f, ",\n   "); JD(f, g_sim_d[p], kFn); }
  fprintf(f, "],\n  \"DF\": [");
  for (int p = 0; p < kPol; ++p) { if (p) fprintf(f, ",\n   "); JD(f, g_sim_df[p], kFn); }
  fprintf(f, "]},\n \"hist_bins_per_octave\": %d,\n \"hist_cycle\": ", kHb); JH(f, g_h_cyc);
  fprintf(f, ",\n \"hist_step\": "); JH(f, g_h_step);
  fprintf(f, ",\n \"hist_w\": "); JH(f, g_h_w);
  // series: blocks of 65536 stream bytes [n, tf_top1, tfxfw_top1, selector_top1, tf_top2, ticks, transformer ticks, bits]
  fprintf(f, ",\n \"series\": [");
  for (int b = 0; b < kBlk && g_blk[b].n; ++b) {
    const Blk& k = g_blk[b];
    fprintf(f, "%s[%llu, %llu, %llu, %llu, %llu, %llu, %llu, %.1f]", b ? ",\n   " : "", k.n, k.tf1, k.x1, k.sel1, k.tf2,
            k.ticks, k.tf_ticks, k.cost);
  }
  fprintf(f, "]}\n");
}
}  // namespace par
#endif

// ---- pipeline (NEIF_PIPE=N: mode -t, the validation decode of mode -u, and,
// as an opt-in convenience that is not the official run, the real compression
// path `-e`: src/runner.cpp NeifEStart) -----------------------------------------
// In compression every byte is known in advance, so a component whose state
// depends on the stream bytes alone can run ahead of the coder.  The run is
// cut into stages, one PROCESS each (fork() at the first byte the model runs
// on, src/runner.cpp NeifPipeStart): every stage walks the same byte loop of
// Compress() with all stream automata, runs only the components it owns and
// hands their results to the next stage through a bounded queue in shared
// memory.  Processes, not threads: the automata and many models keep their
// state in globals, and after a fork every stage has its own copy of all of
// them, so no stage can read a value another stage is writing.
//   N = 2:  [PPMd, transformer]                       -> [cmix models, mixers, coder]
//   N >= 3: [PPMd, transformer layers below g_tf_split] -> [layers from g_tf_split on,
//           output heads, fast weights]               -> [cmix models, mixers, coder]
// The coder stage does everything that reads the coder's own probabilities
// (fxcm reads the cost of the last byte, the mixers, SSE, BITLSTM, the floor
// and end stages, the semantic cache), so it cannot be cut the same way; it is
// about 40 % of the time of one process in page text, which bounds the gain.
// Floating point: a stage is a forked copy of the process, with the same
// rounding mode, MXCSR and tables; every value is computed by the code that
// computes it in one process (the PPMd row twice, see 'P').
// What crosses a queue (records, in stream order; a tag guards the order):
//   'P'  per model byte: the PPMd result (symbol masses and the two bit-tree
//        arrays, integers) and the normalised float row.  The receiving stage
//        normalises the integers with its own code and stops if its row is not
//        bit-identical to the sender's.
//   'A'  per transformer row: the depth-mix state behind layer g_tf_split - 1.
//   'F'  per driver step: the transformer row, the fast-weight row, the hidden
//        vector and the key of the semantic cache.
enum { kOwnPpmd = 1, kOwnTfLo = 2, kOwnTfHi = 4, kOwnMix = 8, kOwnAll = 15 };
inline int g_own = kOwnAll;   // the components this process runs (all: no pipeline)
inline int g_tf_split = 8;    // first transformer layer of the second stage (N >= 3; NEIF_PIPE_SPLIT)
inline bool OwnPpmd() { return (g_own & kOwnPpmd) != 0; }
inline bool OwnTf() { return (g_own & (kOwnTfLo | kOwnTfHi)) != 0; }
inline bool Front() { return (g_own & kOwnMix) == 0; }   // a stage in front of the coder

struct Ring {   // single writer, single reader; positions count bytes and never wrap
  std::atomic<unsigned long long> w;
  char pad0[56];
  std::atomic<unsigned long long> r;
  char pad1[56];
  unsigned long long cap;   // bytes of d (a power of two)
  unsigned char d[64];
};
struct Ctl {   // one per run, shared by all stages
  std::atomic<int> dict_done;   // the first stage has written ./.dict (end of slot 1)
  // written by a front stage when it has passed the last byte
  std::atomic<int> done[4];
  double cpu_s[4], wall_s[4];
  long hwm_kb[4], pss_kb[4];
};
inline Ring* g_in = nullptr;    // from the stage in front
inline Ring* g_out = nullptr;   // to the next stage
inline Ctl* g_ctl = nullptr;
inline int g_stage = 0;         // index of this stage (0 = first)
inline int g_kids[4] = {0, 0, 0, 0};   // coder stage: the pids of the front stages
inline unsigned long long g_waits = 0; // times this stage found a queue empty or full
inline unsigned long long g_pos = 0;   // stream byte this stage works on (for messages)
// entries [0], [1]) of the allocation registry (page_registry.h) that a front
// stage keeps: the tables of the transformer's match models, the transformer
inline unsigned int g_reg_match[2] = {0, 0}, g_reg_tf[2] = {0, 0};

// The validation decode (mode -u with NEIF_PIPE=N, src/runner.cpp): the front
// stages walk the EXPECTED stream, the coder stage decodes the coded input and
// compares every decoded byte with the expected one.  A decoded byte that
// differs is found by that comparison; if the wrong byte first makes the coder
// stage ask for another record than the front stages wrote for the expected
// byte, the run ends here, at the same stream byte (`left` below).  Every
// other stop of the pipeline (a front stage that died, no memory, a PPMd row
// that differs between stages) says nothing about the data: the run failed.
inline bool g_sv = false;
// The pipeline in the real compression path (`cmix -e` with NEIF_PIPE,
// src/runner.cpp NeifEStart): the coder output file.  A pipeline that breaks
// must not leave a file that looks like an output: the coder stage removes it.
inline const char* g_out_path = nullptr;
[[noreturn]] inline void PipeFail(const char* what, bool left = false) {
  fprintf(stderr, "\nneif_pipe: stage %d: %s\n", g_stage, what);
  if (g_out_path && !Front()) {
    unlink(g_out_path);
    fprintf(stderr, "neif_pipe: MULTI-PROCESS COMPRESSION FAILED at stream byte %llu (%s): the output file %s was\n"
                    "neif_pipe: removed and no archive is written. Exit code 3.\n", g_pos, what, g_out_path);
  }
  if (g_sv && !Front()) {
    if (left)
      fprintf(stderr, "neif_sv: FAILED at stream byte %llu: the decoded data leave the expected stream there\n", g_pos);
    else
      fprintf(stderr, "neif_sv: NO VERDICT: the run broke at stream byte %llu (%s); the bytes in front of it were equal\n",
              g_pos, what);
  }
  fflush(stderr);
  if (!Front())
    for (int k = 0; k < 4; ++k) if (g_kids[k] > 0) kill(g_kids[k], SIGKILL);
  _exit(left && g_sv && !Front() ? 4 : 3);   // 4: the data differ (as the comparison in the runner); 3: the run broke
}
// one round of waiting (n = 0 at the first): spin for `spin` time-stamp
// ticks, then sleep in short naps, so a waiting stage costs little CPU time
// (the slowest stage sets the speed and never waits; a nap that ends late
// only lets the queue fill a little).  The coder stage looks after its
// children.
inline void PipeWait(unsigned& n, unsigned long long spin) {
  static unsigned long long t0 = 0;
  if (n == 0) t0 = __builtin_ia32_rdtsc();
  ++n;
  if (__builtin_ia32_rdtsc() - t0 < spin) { _mm_pause(); return; }
  usleep(50);
  if ((n & 1023u) != 0 || Front()) return;
  for (int k = 0; k < 4; ++k) {
    int st = 0;
    if (g_kids[k] > 0 && waitpid(g_kids[k], &st, WNOHANG) == g_kids[k]) {
      if (!WIFEXITED(st) || WEXITSTATUS(st) != 0 || !g_ctl->done[k].load()) PipeFail("a front stage ended early");
      g_kids[k] = 0;
    }
  }
}
inline Ring* RingNew(unsigned long long cap) {
  void* p = mmap(nullptr, sizeof(Ring) + cap, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) return nullptr;
  Ring* q = (Ring*)p;   // zero pages: w = r = 0
  q->cap = cap;
  return q;
}
inline void RingCopy(Ring* q, unsigned long long pos, void* dst, const void* src, size_t n) {
  const unsigned long long o = pos & (q->cap - 1);
  const size_t a = (size_t)(q->cap - o) < n ? (size_t)(q->cap - o) : n;
  if (src) { memcpy(q->d + o, src, a); memcpy(q->d, (const char*)src + a, n - a); }
  else { memcpy(dst, q->d + o, a); memcpy((char*)dst + a, q->d, n - a); }
}
// one record: 8 bytes (tag, length), then the payload
__attribute__((noinline)) inline void Put(int tag, const void* p, size_t n) {
  Ring* q = g_out;
  const unsigned long long w = q->w.load(std::memory_order_relaxed);
  if (w + 8 + n - q->r.load(std::memory_order_acquire) > q->cap) {
    const int sec = g_prof_cur;
    T(15);
    ++g_waits;
    unsigned k = 0;
    // full: this stage is far ahead; it naps until half of the queue is free
    while (w + 8 + n - q->r.load(std::memory_order_acquire) > q->cap / 2) PipeWait(k, 0);
    T(sec);
  }
  const unsigned int h[2] = {(unsigned int)tag, (unsigned int)n};
  RingCopy(q, w, nullptr, h, 8);
  RingCopy(q, w + 8, nullptr, p, n);
  q->w.store(w + 8 + n, std::memory_order_release);
}
__attribute__((noinline)) inline void Get(int tag, void* p, size_t n) {
  Ring* q = g_in;
  const unsigned long long r = q->r.load(std::memory_order_relaxed);
  if (q->w.load(std::memory_order_acquire) - r < 8 + n) {
    const int sec = g_prof_cur;
    T(15);
    ++g_waits;
    unsigned k = 0;
    // empty: the stage in front is the slower one; a short spin (about 20 us), then naps
    while (q->w.load(std::memory_order_acquire) - r < 8 + n) {
      // validation decode: the stage in front has passed its last byte and
      // written everything (done is stored behind its last record), and this
      // stage wants more: it has left the expected stream
      if (g_sv && g_ctl->done[g_stage - 1].load() && q->w.load(std::memory_order_acquire) - r < 8 + n)
        PipeFail("the stage in front has ended and this stage wants another record", true);
      PipeWait(k, 60000ull);
    }
    T(sec);
  }
  unsigned int h[2];
  RingCopy(q, r, h, nullptr, 8);
  if (h[0] != (unsigned int)tag || h[1] != (unsigned int)n) {
    fprintf(stderr, "\nneif_pipe: stage %d at stream byte %llu wants record '%c' of %llu B, the queue has '%c' of %u B\n",
            g_stage, g_pos, tag, (unsigned long long)n, (int)(h[0] & 127u), h[1]);
    PipeFail("the stages are out of step", g_sv && !Front());
  }
  RingCopy(q, r + 8, p, nullptr, n);
  q->r.store(r + 8 + n, std::memory_order_release);
}

// 'P': the PPMd result of one model byte
struct PpmdRec { unsigned int sqp[256], trF[256], trT[256]; float row[256]; };
inline PpmdRec g_ppmd_rec;
// a stage without the PPMd model, in place of the model step (ppmd_PrepareByte)
__attribute__((noinline)) inline void PpmdGet(unsigned int* sqp, unsigned int* trF, unsigned int* trT) {
  Get('P', &g_ppmd_rec, sizeof g_ppmd_rec);
  memcpy(sqp, g_ppmd_rec.sqp, sizeof g_ppmd_rec.sqp);
  memcpy(trF, g_ppmd_rec.trF, sizeof g_ppmd_rec.trF);
  memcpy(trT, g_ppmd_rec.trT, sizeof g_ppmd_rec.trT);
}
// every stage, behind PPMD::ByteUpdate(): the owner sends its result, the
// others compare the row they normalised themselves and pass the record on
__attribute__((noinline)) inline void PpmdRow(const unsigned int* sqp, const unsigned int* trF,
                                              const unsigned int* trT, const float* row) {
  if (g_own == kOwnAll) return;
  if (OwnPpmd()) {
    memcpy(g_ppmd_rec.sqp, sqp, sizeof g_ppmd_rec.sqp);
    memcpy(g_ppmd_rec.trF, trF, sizeof g_ppmd_rec.trF);
    memcpy(g_ppmd_rec.trT, trT, sizeof g_ppmd_rec.trT);
    memcpy(g_ppmd_rec.row, row, sizeof g_ppmd_rec.row);
  } else if (memcmp(g_ppmd_rec.row, row, sizeof g_ppmd_rec.row) != 0) {
    PipeFail("the PPMd row of this stage differs from the row of the stage in front");
  }
  if (g_out) Put('P', &g_ppmd_rec, sizeof g_ppmd_rec);
}

}  // namespace neif_test

#endif  // NEIF_TEST
#endif  // NEIF_TEST_H_
