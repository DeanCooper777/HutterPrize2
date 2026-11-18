#include "meld.h"
#include <string>
#include <ctime>
#include <stdio.h>
#include <cstdlib>
#include <vector>
#include <string.h>
#include <sys/stat.h>
#include <malloc.h>

#include "preprocess/preprocessor.h"
#include "coder/encoder.h"
#include "coder/decoder.h"
#include "predictor.h"

#include "readalike_prepr/article_reorder.h"
#include "readalike_prepr/self_extract.h"
#include "readalike_prepr/phda9_preprocess.h"
#include "readalike_prepr/misc.h"
#include "readalike_prepr/komm_transform.h"
#include "r1_reorder_transform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "isa_floor.h"
#include "page_contexts.h"
#include "word_recency.h"
#include "simd_scan.h"
#include "invariants.h"
#include "huge_pages.h"
#include "mixer_lr.h"
#include "mixer_decay.h"
#include "fx_mixer_lr.h"
#include "mixer_bank.h"
#include "logit_table.h"
#include "mixer_weights.h"
#include "vocab_frozen.h"
#include "run_mode.h"

static_assert(54857 <= 65360,
    "exceeds the capacity of the code plan "
    "(dictionary.cpp kBoundary4 or kBoundary5): the words above it "
    "would get NO code word and be skipped silently.  Use a smaller "
    "dictionary, or open the fifth band and update EVERY reader of the "
    "band plan.");

static_assert(54857 <= run_mode::kBand4,
    "exceeds the bands the grammar automaton knows (kBand4): "
    "Idx3() would return a negative index for the fifth band and "
    "BuildMasks() would reject its second byte.  The grammar automaton "
    "needs a fifth band first.");

static_assert(54857 <= (80 + 3840 + 40960 + 20480),
    "exceeds the bands the PPMd warm-up knows "
    "(Code(), kBandKap4): its last branch would write a wrong code word for "
    "the fifth band and Material() would drop the words above.  Add the "
    "fifth band there first.");

// Run-time parameters: each one returns its compiled default.  The
// manifest writer is empty in this build (see its call in main()).
namespace run_options {
static double ParamD(const char* name, double def) {
  (void)name;
  return def;
}
static long ParamI(const char* name, long def) {
  double v = ParamD(name, (double)def);
  return (long)(v < 0 ? v - 0.5 : v + 0.5);
}
static void WriteManifest(const char* path, int argc, char** argv) {
  (void)path; (void)argc; (void)argv;
}
}

#include "attributes.h"

#include "run_mode.h"

#include "transformer.h"

#include "alias_context.h"
#include "alias_title_subst.h"
#include "author_block.h"
#include "metablock.h"
#include "census_block.h"
#include "computed_values.h"
static bool census_block_active = false;

#include "ip_pack.h"
#include "isbn_check.h"
#include "metablock_form.h"
#include "text_start_pack.h"
#include "rule_stages_file.h"
#include "ts_progress.h"


#include "no_args.h"

#ifdef NEIF_TEST
// Chunk ("split") evaluation mode `-t`, see src/neif_test.h.  Everything of it
// in this file stands under this macro; without it the file is the entry's.
#include "neif_test.h"
#ifdef NEIF_MT
#include "neif_mt.h"   // threads for the transformer step (NEIF_MT=n)
#endif
#include <sys/resource.h>
static double neif_t_model = 0;   // wall clock at the first byte the model ran on
#endif


#include <malloc.h>
#ifndef AUX_BUILD
#define AUX_BUILD 0
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>

namespace {
  constexpr long io_buffer_kb = 256;
  long progress_steps = 10000;  // progress ticks per run (they also drive the huge-page collapse)
}

#if !(1 && !AUX_BUILD)
static bool provide_dict_file(const char* src) {
  FILE* in = fopen(src, "rb");
  if (!in) return false;
  FILE* out = fopen(".dict", "wb");
  if (!out) { fclose(in); return false; }
  char buf[1 << 16];
  size_t n;
  bool ok = true;
  while ((n = fread(buf, 1, sizeof(buf), in)) > 0) {
    if (fwrite(buf, 1, n, out) != n) { ok = false; break; }
  }
  fclose(in);
  if (fclose(out) != 0) ok = false;
  return ok;
}
#endif

int Help() {
  msg_note(256);
  return -1;
}

size_t getFileSize(const char* path) {
  FILE *f = fopen(path, "rb");
  if (f == NULL) {
    msg_note(170);
    return 0;
  }
  fseek(f, 0, SEEK_END);
  size_t output_size = ftell(f);
  fclose(f);
  return output_size;
}

void WriteStorageHeader(FILE* out, bool dictionary_used) {
  for (int i = 4; i >= 0; --i) {
    char c = 0;
    if (i == 4 && dictionary_used) c = 0x80;
    putc(c, out);
  }
}

void ClearOutput() {

}

#include "preprocess/dict_derive.h"
#include "huge_pages.h"
#include "vocab_frozen.h"
#include "console_progress.h"

#include "selfx_clean.h"

void WriteHeaderVocfix(unsigned long long length, bool dictionary_used,
    FILE* os) {
  for (int i = 4; i >= 0; --i) {
    char c = length >> (8*i);
    if (i == 4) {
      c &= 0x7F;
      if (dictionary_used) c |= 0x80;
      if (vocab_frozen::g_aux) c |= 0x40;
    }
    putc(c, os);
  }
}

void ReadHeaderVocfix(FILE* is, unsigned long long* length,
    bool* dictionary_used, std::vector<bool>* vocab) {
  *length = 0;
  for (int i = 0; i <= 4; ++i) {
    *length <<= 8;
    unsigned char c = getc(is);
    if (i == 0) {
      if (c&0x80) *dictionary_used = true;
      else *dictionary_used = false;
      c &= 0x7F;
      vocab_frozen::g_aux = (c & 0x40) != 0;
      c &= 0x3F;
    }
    *length += c;
  }
  vocab_frozen::FillVocab(vocab);
}

static inline bool AuthorBlockInert() {
  if (transformer::BlobForm()) return true;
  if (vocab_frozen::g_aux) return true;
  return false;
}

static inline bool MetablockInert() {
  if (transformer::BlobForm()) return true;
  if (vocab_frozen::g_aux) return true;
  return false;
}

static inline bool TimestampProgressInert() {
  if (transformer::BlobForm()) return true;
  if (vocab_frozen::g_aux) return true;
  return false;
}

// The known bytes b[0..n) of the model side, coder and decoder alike: every
// model observes their bits, none is coded.  One function for both
// directions, so the hooks below need no coder object.
__attribute__((noinline)) static void ObserveKnownBytes(Predictor* p, const unsigned char* b, int n) {
#ifdef NEIF_TEST
  // structure events: the injected bytes; in the structure-only pass they go
  // to the automata of the transformer driver alone (no model observes them)
  for (int k = 0; k < n; ++k) neif_test::H(0x100u | b[k]);
  if (neif_test::g_skip) {
    for (int k = 0; k < n; ++k) p->NeifSkipByte(b[k]);
    return;
  }
  // a front stage of the pipeline: the models of that stage observe them
  if (neif_test::Front()) {
    for (int k = 0; k < n; ++k) p->NeifFrontByte(b[k]);
    return;
  }
#endif
  invariants::g_injected_byte = 1;
  for (int k = 0; k < n; ++k)
    for (int bit = 7; bit >= 0; --bit) p->ObserveKnownBit((b[k] >> bit) & 1);
  invariants::g_injected_byte = 0;
}

__attribute__((noinline)) static void CensusBlockByte(Predictor* p, unsigned char c) {
  if (MetablockInert()) return;
  if (metablock::mk.TaByte()) {
    // slot 3 is the pseudo article: at its end mark the automaton takes it
    if (metablock::mk.TaSlot() == 4 && metablock::mk.TaSlotEndet()) {
      size_t n = 0;
      const unsigned char* q = metablock::mk.TaPuffer(4, &n);
      census_block::aut.Slot3(q, n);
    }
    return;
  }
  census_block::aut.Observe(c);
  const unsigned char* zb = nullptr;
  const int zn = census_block::aut.TakeBlock(&zb);
  if (!zn) return;
  // of the census injections the transformer sees only the computed text (TfSieht)
  const bool tf = census_block::aut.TfSieht();
  if (!tf) timestamp_skip::Beginn();
  if (tf) tf_author_prompt::Beginn(zn);
  {
    const int kn = census_block::aut.EndeNach(zn);
    tf_copy_end::g_nach = tf ? kn : 0;
  }
  ObserveKnownBytes(p, zb, zn);
  tf_copy_end::g_nach = 0;
  if (tf) tf_author_prompt::Ende();
  if (!tf) timestamp_skip::Ende();
  (void)tf;
}

__attribute__((noinline)) static void TextStartHead(Predictor* p, int slot) {
  static const char kKopf2[] = "neif\tpre authors\n";
  static const char kKopf3[] = "neif\tpre census\n";
  static const char kKopf1[] = "neif\tpre dictionary\n";
  const char* k = slot == 2 ? kKopf2 : (slot == 4 ? kKopf3 : (slot == 1 ? kKopf1 : ""));
  const int n = (int)strlen(k);
  if (!n) return;
  tf_author_prompt::Beginn(n);
  ObserveKnownBytes(p, (const unsigned char*)k, n);
  tf_author_prompt::Ende();
}

static_assert(54857 == 54857, "must equal DICT_WORDS");
namespace fxcmv1 { int DictNachladen(); }
__attribute__((noinline)) static void DictArticleSlotEnd() {
  size_t n = 0;
  const unsigned char* b = metablock::mk.TaPuffer(1, &n);
#ifdef NEIF_TEST
  // Pipeline: every stage comes here.  The first stage writes ./.dict, the
  // others wait until it is complete (a second write would truncate the file
  // under a stage that reads it); fxcm lives in the coder stage alone.
  if (neif_test::g_ctl && neif_test::g_stage > 0) {
    unsigned k = 0;
    while (!neif_test::g_ctl->dict_done.load()) neif_test::PipeWait(k, 0);
  } else {
    dict_article::Schreiben(b, n);
    if (neif_test::g_ctl) neif_test::g_ctl->dict_done.store(1);
  }
  const int w = neif_test::Front() ? 0 : fxcmv1::DictNachladen();
#else
  dict_article::Schreiben(b, n);
  const int w = fxcmv1::DictNachladen();
#endif
  if (dict_article::wrt) {
    run_mode::gram.Init(NULL, dict_article::strom, 0);
    run_mode::gram.ZoneSperren();
  }
  computed_values::g_dict_da = true;
  (void)w;
}

namespace dict_runs {
constexpr bool kRegel = 1 == 1;
// fixed bounds, checked against ./.dict by the coder (E551)
constexpr unsigned kTxt = dict_codes::kLaufTxt, kWort = dict_codes::kLaufWort;
struct RunState {                      // static (.bss), lives for the run
  size_t gez;                         // bytes of b followed
  unsigned nt, nw, nc;                // letters in txt, words finished, letters of the current word
  int np;                             // bytes of an open code word
  bool hinter;                        // a run mark in front of the current word
  unsigned char pend[3];              // the open code word
  unsigned char cur[kWort];           // the current word decoded so far
  unsigned ab[54857 + 1];       // start of word r in txt
  unsigned char txt[kTxt];            // the decoded words back to back
};
static RunState z;
static const unsigned char* Text(unsigned r, size_t* l) {
  *l = (r + 1 < z.nw ? z.ab[r + 1] : z.nt) - z.ab[r];
  return z.txt + z.ab[r];
}
// the rank intervals [lo[q], hi[q]] of the code words that begin with pre[0..k)
static int Intervalle(const unsigned char* pre, int k, unsigned* lo, unsigned* hi) {
  const unsigned b0 = pre[0];
  if (k == 1) {
    if (b0 <= 0xCF) { lo[0] = hi[0] = b0 - 0x80; return 1; }
    lo[0] = 80 + (b0 - 0xD0) * 80; hi[0] = lo[0] + 79;
    if (b0 < 0xF0) return 1;
    lo[1] = 3920 + (b0 - 0xF0) * 2560; hi[1] = lo[1] + 2559;
    lo[2] = 44880 + (b0 - 0xF0) * 1280; hi[2] = lo[2] + 1279;
    return 3;
  }
  const unsigned b1 = pre[1];
  if (b1 <= 0xCF) { lo[0] = hi[0] = 80 + (b0 - 0xD0) * 80 + (b1 - 0x80); return 1; }
  lo[0] = b1 <= 0xEF ? 3920 + ((b0 - 0xF0) * 32 + (b1 - 0xD0)) * 80
                     : 44880 + ((b0 - 0xF0) * 16 + (b1 - 0xF0)) * 80;
  if (k == 2) { hi[0] = lo[0] + 79; return 1; }
  lo[0] += pre[2] - 0x80; hi[0] = lo[0];
  return 1;
}
static void Add(const unsigned char* t, size_t l) {   // to the current word
  if (l > kWort - z.nc) l = kWort - z.nc;
  memcpy(z.cur + z.nc, t, l);
  z.nc += (unsigned)l;
}
static void Folge(unsigned char c) {  // the state behind byte c
  if (c == 0x0B) return;
  if (z.np || c >= 0x80) {
    z.pend[z.np++] = c;
    const int L = z.pend[0] <= 0xCF ? 1 : z.np < 2 ? 0 : z.pend[1] <= 0xCF ? 2 : 3;
    if (L && z.np == L) {
      size_t k = 0;
      const long r = dict_codes::Rang(z.pend, (size_t)L, &k);
      if (r >= 0 && (size_t)r < z.nw) {
        size_t l = 0;
        const unsigned char* t = Text((unsigned)r, &l);
        Add(t, l);
      }
      z.np = 0;
    }
    return;
  }
  if (c == 0x0A) {
    if (!z.nc) {
      z.hinter = true;
    } else if (z.nw <= 54857) {   // the current word is finished
      z.ab[z.nw++] = z.nt;
      const unsigned lc = z.nc < kTxt - z.nt ? z.nc : kTxt - z.nt;
      memcpy(z.txt + z.nt, z.cur, lc);
      z.nt += lc;
      z.nc = 0;
      z.hinter = false;
    }
    return;
  }
  Add(&c, 1);
}
// some code word beginning with pre[0..k) names a rank < i; with genau (exact)
// (inside an open code word, k >= 2: at most 80 ranks) one of them names a
// word T that keeps the order, S[0..ls) the rest of the previous word (equal
// so far): behind a mark T < S, in a run T >= S or T a proper prefix of S
static bool CodeOk(const unsigned char* pre, int k, unsigned long long i, bool genau,
                   const unsigned char* S, size_t ls) {
  unsigned lo[3], hi[3];
  const int n = Intervalle(pre, k, lo, hi);
  for (int q = 0; q < n; ++q) {
    if (lo[q] >= i) continue;
    if (!genau) return true;
    const unsigned e = hi[q] < i - 1 ? hi[q] : (unsigned)(i - 1);
    for (unsigned r = lo[q]; r <= e; ++r) {
      size_t lt = 0;
      const unsigned char* t = Text(r, &lt);
      const int c = memcmp(t, S, lt < ls ? lt : ls);
      if (z.hinter ? (c < 0 || (!c && lt < ls)) : c >= 0) return true;
    }
  }
  return false;
}
__attribute__((noinline)) static int Menge(const unsigned char* b, size_t m, unsigned char* v, int* art) {
  if (m < z.gez) z.gez = z.nt = z.nw = z.nc = 0, z.np = 0, z.hinter = false;   // a new slot 1
  for (; z.gez < m; ++z.gez) Folge(b[z.gez]);
  int n = 0;
  *art = 3;
  if (m && b[m - 1] == 0x0B) {
    v[n++] = 0x0B;
    return n;
  }
  const unsigned long long i = z.nw;             // index of the current word
  const bool regel = kRegel && i > dict_codes::kLaufAb;
  size_t lp = 0;
  const unsigned char* p = i ? Text((unsigned)(i - 1), &lp) : nullptr;
  const size_t k = z.nc;
  const bool eq = regel && k <= lp && (!k || !memcmp(z.cur, p, k));
  int lo = 0x61, hi = 0x7A, clo = 0x80, chi = 0xFF;
  bool lf = false, ende = false, genau = false;
  if (z.np) {
    lo = 1, hi = 0;                              // inside a code word: no letter
    chi = (z.np == 1 && z.pend[0] >= 0xF0) ? 0xFF : 0xCF;
    if (eq) genau = true, *art = 4;
  } else if (!m) {
  } else if (!k && b[m - 1] == 0x0A) {           // the start of a word
    if (z.hinter) {
      if (regel) hi = lp > 1 ? p[0] : p[0] - 1, *art = 4;
    } else {
      ende = true;
      lf = i > dict_codes::kLaufAb;           // a run mark
      if (regel) lo = p[0], *art = 4;
    }
  } else {
    // inside a word: no code word (the prefix rule); the line feed only
    // behind a letter
    clo = 1, chi = 0;
    lf = b[m - 1] < 0x80;
    if (eq && !z.hinter) {
      lf = false, *art = 4;
      if (k < lp) lo = p[k];
    } else if (eq && k < lp) {
      hi = k + 1 < lp ? p[k] : p[k] - 1, *art = 4;
    }
  }
  if (lf) v[n++] = 0x0A;
  if (ende) v[n++] = 0x0B;
  for (int c = lo; c <= hi; ++c) v[n++] = (unsigned char)c;
  unsigned char pre[3] = {z.pend[0], z.pend[1], 0};
  for (int c = clo; c <= chi; ++c) {
    pre[z.np] = (unsigned char)c;
    if (kRegel && !CodeOk(pre, z.np + 1, i, genau, p + k, lp - k)) continue;
    v[n++] = (unsigned char)c;
  }
  return n;
}
}  // namespace dict_runs
__attribute__((noinline)) static bool DictArticleSep() {
  unsigned char v[156];
  int n = 0;
  if (metablock::mk.TaNaechsterSlot() == 1) {
    n = dict_runs::Menge(nullptr, 0, v, &metablock::zw::art);
  } else if (metablock::mk.TaByte() && metablock::mk.TaSlot() == 1 &&
             !metablock::mk.TaSlotEndet()) {
    size_t m = 0;
    const unsigned char* b = metablock::mk.TaPuffer(1, &m);
    n = dict_runs::Menge(b, m, v, &metablock::zw::art);
  } else {
    return false;
  }
  return metablock::zw::Setze(v, n);
}

__attribute__((noinline)) static void TextStartByte(Predictor* p) {
  if (MetablockInert()) return;
  metablock::zw::an = false;
  if (!metablock::mk.Endet() && !metablock::mk.TaByte()) return;
  {
    bool jetzt = metablock::mk.TaSlotEndet() && metablock::mk.TaSlot() == 1;
    if (jetzt) DictArticleSlotEnd();
#ifdef NEIF_TEST
    if (!neif_test::OwnPpmd()) jetzt = false;   // pipeline: the PPMd model lives in another stage
#endif
    if (jetzt) p->PpmdWaermen();
  }
  if (metablock::mk.TaSlotEndet() && metablock::mk.TaSlot() == 2) {
    p->WordlblBlockEnde();
  }
  const int s = metablock::mk.TaNaechsterSlot();
  if (s) TextStartHead(p, s);
  const int j = metablock::mk.TaZeileFolgt();
  if (j >= 0) {
    const unsigned char* q = nullptr;
    const int n = metablock::mk.TaOkb(j, &q);
    // the transformer gets the context as well - an author
    // prompt block like the slot head (fed, not predicted, masked in the dump)
    if (n) tf_author_prompt::Beginn(n);
    ObserveKnownBytes(p, q, n);
    if (n) tf_author_prompt::Ende();
  }
  // stage 2: a byte of a repetition line follows - its known bits
  metablock::zw::an = metablock::mk.TaZwSetze();
  if (!metablock::zw::an) metablock::zw::an = DictArticleSep();   // slot 1
  if (metablock::mk.TaFertig()) {
    // the transform grammar holds from the first byte of page 0 on (the slots
    // carry text outside it: the dictionary in plain text, the census texts)
    run_mode::gram.ZoneOeffnen(metablock::mk.Pos());
    p->MetalrBlockEnde();
  }
  (void)p;
}

ENC_ONLY static void TextStartSepEnc(Encoder& e, unsigned int c, unsigned long long pos) {
  unsigned int ctx = 1;
  const int a = metablock::zw::art;
  for (int j = 7; j >= 0; --j) {
    const int bit = (int)((c >> j) & 1u);
    const int f = metablock::zw::line_buf[ctx & 255u];
    if (f < 0) {
      e.Encode(bit);
    } else {
      if (f != bit) {
        msg_stop(542);
        exit(3);
      }
      e.ObserveKnownBit(bit);
    }
    ctx += ctx + (unsigned int)bit;
  }
}
__attribute__((noinline)) static int TextStartSepDec(Decoder& d) {
  int byte = 1;
  while (byte < 256) {
    const int f = metablock::zw::line_buf[byte];
    int bit;
    if (f < 0) {
      bit = d.Decode();
    } else {
      bit = f;
      d.ObserveKnownBit(bit);
    }
    byte += byte + bit;
  }
  return byte;
}

// After every coded (Compress) or decoded (Decompress) stream byte c: the
// units and automata that read the stream, and the bytes they inject.  ONE
// function for both directions, so coder and decoder run the same code in
// the same order; noinline keeps it out of the two byte loops.
__attribute__((noinline)) static void AfterStreamByte(Predictor* p, unsigned char c) {
  if (!MetablockInert()) {
    metablock::mk.Byte(c);
    if (metablock::mk.MetaEndet()) {
      ppmd_meta::block_fertig = true;
    }
    if (metablock::mk.Endet()) {
      p->WordlblBlockEnde();
    }
  }
  // The byte is fully coded, so it is fixed on both sides.  If it closed a
  // title span, the alias rows of this article are pushed through the
  // transformer here, and through no other model: observed, never coded.
  // Beginn/Ende (begin/end) bracket them for it.  Mirrored in Decompress().
  alias_context::aut.Observe(c);
  {
    const unsigned char* blk = nullptr;
    const int n = alias_context::aut.TakeBlock(&blk);
    if (n) alias_prompt::Beginn(n);
    if (n) alias_title_subst::Beginn(alias_context::aut.PraefixLen());
    for (int k = 0; k < n; ++k) p->ObserveTfOnlyByte(blk[k]);
    alias_prompt::Ende();
    alias_title_subst::Ende();
  }
  if (!AuthorBlockInert()) author_block::aut.Observe(c);
  {
    const unsigned char* ab = nullptr;
    const int an = author_block::aut.TakeBlock(&ab);
    // The transformer must classify these bytes as a PROMPT block (feed 1
    // / predict 0), exactly as the alias prompt form does - mirrored, or
    // the archive is unreadable.
    if (an) tf_author_prompt::Beginn(an);
    ObserveKnownBytes(p, ab, an);
    tf_author_prompt::Ende();
  }
  CensusBlockByte(p, c);
  if (census_block::aut.ZeileEndet() && !metablock::mk.TaByte()) {
    p->MetalrBlockEnde();
    ppmd_meta::zeile_fertig = true;
  }
  if (!TimestampProgressInert()) {
    ts_progress::aut.Observe(c);
    const unsigned char* tb = nullptr;
    const int tn = ts_progress::aut.TakeBlock(&tb);
    if (tn) timestamp_skip::Beginn();
    ObserveKnownBytes(p, tb, tn);
    timestamp_skip::Ende();
  }
  TextStartByte(p);
  (void)p; (void)c;
}

#ifdef NEIF_TEST
// ---- pipeline of the chunk mode (NEIF_PIPE=N, see src/neif_test.h) ---------
#include <sys/prctl.h>
#include <sys/mman.h>
namespace ppmd_page { void neif_fork_child(void); }
static int neif_pipe_n = 0;                    // NEIF_PIPE: number of stages (0, 1: none)
static bool neif_e_real = false;               // the pipeline runs in the real compression path (`-e`, NeifEStart)
static unsigned long long neif_e_stop = 0;     // NEIF_E_STOP: `-e` stops behind this many stream bytes (0: off)
static const char* neif_stream_path = nullptr;
static double neif_fork_wall = 0, neif_fork_cpu = 0;   // at the fork (the coder stage: its own)
static double neif_cpu_model = 0;              // CPU time of this process at the first model byte
static long neif_pss_peak_kb = 0;              // largest sum of the stages' proportional set sizes seen
static double NeifCpuNow() {                   // user + system time of this process
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  return (double)ru.ru_utime.tv_sec + (double)ru.ru_stime.tv_sec +
         1e-6 * ((double)ru.ru_utime.tv_usec + (double)ru.ru_stime.tv_usec);
}
// proportional set size of a process (0 = this one), KiB: a page shared by k
// processes counts 1/k, so the sum over the stages is the memory they use
static long NeifPssKb(int pid) {
  char name[64], line[256];
  if (pid) snprintf(name, sizeof name, "/proc/%d/smaps_rollup", pid);
  else snprintf(name, sizeof name, "/proc/self/smaps_rollup");
  FILE* f = fopen(name, "r");
  if (!f) return 0;
  long kb = 0;
  while (fgets(line, sizeof line, f))
    if (!strncmp(line, "Pss:", 4)) { kb = strtol(line + 4, nullptr, 10); break; }
  fclose(f);
  return kb;
}
// the coder stage, about every five seconds: the memory of all stages
static void NeifPssSample() {
  if (!neif_test::g_ctl || neif_test::Front()) return;
  static double next = 0;
  const double now = console_progress::Now();
  if (now < next) return;
  next = now + 5.0;
  long kb = NeifPssKb(0);
  for (int k = 0; k < 4; ++k) if (neif_test::g_kids[k] > 0) kb += NeifPssKb(neif_test::g_kids[k]);
  if (kb > neif_pss_peak_kb) neif_pss_peak_kb = kb;
}
// Called once, in front of the first byte the model runs on (behind the
// structure-only pass, so every stage inherits the automata in step).  Forks
// the front stages; the calling process stays the coder stage.  A front stage
// returns into the byte loop of Compress() with its own handle of the stream,
// positioned behind the block the loop is working on (the handle of the coder
// stage shares its file position with the copies in the children).
static void NeifPipeStart(Predictor* p, FILE** is, unsigned long long read_total) {
  using namespace neif_test;
  const int n = neif_pipe_n > 3 ? 3 : neif_pipe_n;
  static const int own[2][3] = {
      {kOwnPpmd | kOwnTfLo | kOwnTfHi, kOwnMix, 0},
      {kOwnPpmd | kOwnTfLo, kOwnTfHi, kOwnMix}};
  p->NeifPipePrepare();
  g_ctl = (Ctl*)mmap(nullptr, sizeof(Ctl), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  Ring* q[2] = {RingNew(32ull << 20), n == 3 ? RingNew(32ull << 20) : nullptr};
  if ((void*)g_ctl == MAP_FAILED || !q[0] || (n == 3 && !q[1])) { g_ctl = nullptr; PipeFail("no shared memory"); }
  fflush(nullptr);
  prctl(PR_SET_TIMERSLACK, 1000UL);   // naps of a waiting stage (PipeWait) end on time; inherited
  for (int s = 0; s + 1 < n; ++s) {
    const pid_t pid = fork();
    if (pid < 0) PipeFail("fork failed");
    if (pid == 0) {
      prctl(PR_SET_PDEATHSIG, SIGKILL);   // a front stage does not outlive the coder stage
      g_stage = s;
      g_own = own[n - 2][s];
      g_in = s ? q[s - 1] : nullptr;
      g_out = q[s];
      for (int k = 0; k < 4; ++k) g_kids[k] = 0;
      *is = fopen(neif_stream_path, "rb");
      if (!*is || fseeko(*is, (off_t)read_total, SEEK_SET) != 0) PipeFail("cannot reopen the stream");
      if (OwnPpmd()) ppmd_page::neif_fork_child();
      else p->NeifPpmdDisown();
      p->NeifFrontDrop();
      neif_fork_wall = console_progress::Now();
      neif_fork_cpu = NeifCpuNow();
      ProfReset();
      return;
    }
    g_kids[s] = (int)pid;
  }
  g_stage = n - 1;
  g_own = kOwnMix;
  g_in = q[n - 2];
  p->NeifPpmdDisown();
  neif_fork_wall = console_progress::Now();
  neif_fork_cpu = NeifCpuNow();
  ProfReset();
}
// A front stage behind the last byte of the chunk: its figures go to the
// shared block, and the process ends without the end-of-run work of the coder
// (no output, no destructors: the files and the PPMd heap are not its own).
[[noreturn]] static void NeifFrontEnd() {
  using namespace neif_test;
  const int s = g_stage;
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  g_ctl->wall_s[s] = console_progress::Now() - neif_fork_wall;
  g_ctl->cpu_s[s] = NeifCpuNow() - neif_fork_cpu;
  g_ctl->hwm_kb[s] = ru.ru_maxrss;
  g_ctl->pss_kb[s] = NeifPssKb(0);
  char who[16];
  snprintf(who, sizeof who, "stage%d", s);
  ProfPrint(stderr, who, g_ctl->wall_s[s]);
  fprintf(stderr, "neif_pipe stage %d: %.1f s wall, %.1f s cpu, %llu queue waits\n", s, g_ctl->wall_s[s],
          g_ctl->cpu_s[s], g_waits);
  fflush(stderr);
  g_ctl->done[s].store(1);
  _exit(0);
}
#endif

#if defined(NEIF_TEST) && defined(NEIF_MT)
// NEIF_MT (src/neif_mt.h): the guess of a model byte for the speculative
// transformer step.  `known` is the byte if it is known before its bits (the
// compressor; a forced byte; in mode -u with NEIF_MT_SPEC=2 the stream byte),
// -1 otherwise (no speculation at this stage).  NEIF_MT_FAULT=p makes p per
// mille of the guesses wrong on purpose: another byte of the transformer's
// byte set, so that every undo path runs.
static unsigned long long neif_mt_faults = 0;
static int NeifSpecGuess(int known) {
  if (known < 0) return -1;
  static unsigned long long rng = 0x9E3779B97F4A7C15ull;
  rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
  if (neif_mt::g_fault > 0 && (int)(rng % 1000u) < neif_mt::g_fault) {
    for (int k = 1; k < 256; ++k) {
      const int g = (known + k * (int)(1 + (rng >> 20) % 7)) & 255;
      if (g != known && transformer::InSet(g)) { ++neif_mt_faults; return g; }
    }
  }
  return known;
}
#endif

void Compress(unsigned long long input_bytes, FILE* is,
    FILE* os, unsigned long long* output_bytes, Predictor* p) {
  Encoder e(os, p);

  unsigned long long percent =
      1 + (input_bytes / (unsigned long long)progress_steps);
  unsigned long long prog_left = 1;
  ClearOutput();
  // the read block (BSS, one coder at a time per process)
  static char buffer[(size_t)io_buffer_kb * 1024];
  const size_t buffer_size = sizeof buffer;
  unsigned long long pos = 0;
  int sem_pb = 0;   // the previous stream byte (0 before the first)
#ifdef NEIF_TEST
  unsigned long long neif_tail = 0;   // the last 8 stream bytes
  unsigned long long neif_read = 0;   // stream bytes read so far (whole blocks)
  const char* neif_tr = getenv("NEIF_T_TRACE");
  const unsigned long long neif_trace = (neif_tr && *neif_tr) ? strtoull(neif_tr, nullptr, 10) : 0;
#endif
  while (pos < input_bytes) {
    const size_t chunk = static_cast<size_t>(
        std::min<unsigned long long>(buffer_size, input_bytes - pos));
    const size_t got = fread(buffer, 1, chunk, is);
    if (got == 0) break;
#ifdef NEIF_TEST
    neif_read += got;
#endif
    for (size_t i = 0; i < got; ++i) {
      unsigned char c = static_cast<unsigned char>(buffer[i]);
#ifdef NEIF_TEST
      if (neif_test::g_on) {
        // The structure-only pass ends at the first page start at or behind
        // the warm-up target: the 8 bytes in front are the page anchor
        // `  </page>\n` (author_block::kAnker), so the model starts on the
        // first title byte of a page, where the transformer starts a document.
        if (neif_test::g_skip && pos >= neif_test::g_warm_target &&
            neif_tail == 0x20204c2fdf984e0aull) {
          neif_test::g_skip = false;
          neif_test::g_warm_from = (long long)pos;
          neif_t_model = console_progress::Now();
          neif_cpu_model = NeifCpuNow();
          p->NeifSkipEnd();
          neif_test::ProfReset();
        }
        // pipeline: the stages part in front of the first byte the model runs on;
        // a front stage goes on in this loop with its own handle of the stream
        if (!neif_test::g_skip && neif_pipe_n > 1 && !neif_test::g_ctl) NeifPipeStart(p, &is, neif_read);
        neif_test::g_pos = pos;
        neif_tail = (neif_tail << 8) | c;
        // NEIF_T_TRACE=n: a line every n stream bytes with the bits and the
        // coder output so far (before this byte), for comparison with
        // doc/stream_rate.tsv
        if (neif_trace && !neif_test::g_skip && !neif_test::Front() && pos % neif_trace == 0)
          // (fourth field: the tap sum, the cost at the return value of Predict())
          fprintf(stderr, "neif_trace %llu %.3f %llu %.3f\n", pos, neif_test::g_bits_all,
                  (unsigned long long)e.OutputSize(), neif_test::g_tap_all);
        neif_test::g_hash_on = pos >= neif_test::g_start;
        neif_test::g_count = !neif_test::g_skip && pos >= neif_test::g_start;
        // structure events of this stream byte, the same in both passes: the
        // byte, a forced byte, the forced-bit row of the text start, the mask
        // of the grammar automaton, the gate zone and the page count
        const int nzw = p->Tf1ZwangByte();
        neif_test::H(c);
        neif_test::H((unsigned)(nzw + 1));
        neif_test::H(metablock::zw::an ? 1u : 0u);
        neif_test::H(run_mode::gram.ByteActive(pos) ? (unsigned)run_mode::gram.DbgMask() : 0xffu);
        neif_test::H(tf_gate::ctx.zone);
        neif_test::H(tf_gate::ctx.pages);
        if (nzw >= 0 && nzw != (int)c) ++neif_test::g_zw_mismatch;
        if (neif_test::g_skip || neif_test::Front()) {
          // the byte as the branches below treat it, without the coder: in the
          // structure-only pass without any model, in a front stage of the
          // pipeline with the models of that stage
          if (nzw >= 0) {
            if (run_mode::gram.ByteActive(pos)) run_mode::gram.NextByte((unsigned int)nzw);
          } else if (metablock::zw::an) {
          } else if (run_mode::gram.ByteActive(pos)) {
            run_mode::gram.NextByte(c);
          }
          const unsigned int mb = nzw >= 0 ? (unsigned int)nzw : (unsigned int)c;
          if (neif_test::g_skip) p->NeifSkipByte(mb); else p->NeifFrontByte(mb);
          sem_pb = c;
          AfterStreamByte(p, c);
          if (neif_test::g_skip) p->NeifSkipSync();
          ++pos;
          continue;
        }
      }
#endif
#ifdef NEIF_TEST
      neif_test::T(14);   // profile mark: semantic cache, byte start
#endif
      p->SemByteStart(sem_pb);
#ifdef NEIF_TEST
      neif_test::T(0);
#ifdef NEIF_PAR
      // NEIF_PAR (src/neif_test.h): the bit loop of this byte begins.  The
      // measurement closes the byte in front and takes the guesses a decoder
      // could make for this one (which is c; reads only).
      p->NeifParByteStart(c);
#endif
#endif
      sem_pb = c;
      const int zw = p->Tf1ZwangByte();
#if defined(NEIF_TEST) && defined(NEIF_MT)
      // NEIF_MT: the compressor knows the byte, so the transformer step behind it can start now
      if (neif_mt::g_spec) p->NeifSpecLaunch(NeifSpecGuess(zw >= 0 ? zw : (int)c));
#endif
      if (zw >= 0) {
        for (int j = 7; j >= 0; --j) e.ObserveKnownBit((zw >> j) & 1);
        if (run_mode::gram.ByteActive(pos)) run_mode::gram.NextByte((unsigned int)zw);
      } else
      if (metablock::zw::an) TextStartSepEnc(e, c, pos);
      else
      if (run_mode::gram.ByteActive(pos)) {
        unsigned int ctx = 1;
        for (int j = 7; j >= 0; --j) {
          const int bit = (c >> j) & 1;
          const int f = run_mode::gram.Forced(ctx);
          if (f < 0) {
            run_mode::gram.CountFree(ctx);
            e.Encode(bit);
          } else {
            run_mode::gram.CountKnown(ctx);
            if (f != bit) run_mode::gram.Violation(pos, ctx, f, bit);
            e.ObserveKnownBit(bit);
          }
          ctx += ctx + bit;
        }
        run_mode::gram.NextByte(c);
      } else
      {
      for (int j = 7; j >= 0; --j) {
        e.Encode((c>>j)&1);
      }
      }
      AfterStreamByte(p, c);   // the readers of the stream, mirrored in Decompress()
      if (--prog_left == 0) {
        prog_left = percent;
        double frac = 100.0 * pos / input_bytes;

        HUGEPAGE_TICK(pos);
        console_progress::Tick(pos, input_bytes);
#ifdef NEIF_TEST
        NeifPssSample();
#endif
      }
      ++pos;
    }
  }
#ifdef NEIF_TEST
  if (neif_test::Front()) NeifFrontEnd();   // a front stage of the pipeline ends here
#endif
  e.Flush();
#ifdef NEIF_TEST
  // a chunk ends in the middle of the stream: the end-of-stream checks of the
  // units below do not apply
  // (the pipeline of the real path, neif_e_real, runs to the end of the stream
  // and takes the checks; a run cut short by NEIF_E_STOP does not)
  if ((neif_test::g_on && !neif_e_real) || neif_e_stop) { *output_bytes = ftell(os); return; }
#endif
  run_mode::gram.Report("compress");
  if (!MetablockInert()) {
    metablock::mk.Report("compress");
    census_block::aut.Report("compress");
    metablock::mk.ByteWacht("compress", pos);
  }
  redirect_table::tab.Schluss();
  alias_table::tab.Schluss();
  alias_context::aut.Finish();
  alias_context::aut.Report("compress");
  if (!AuthorBlockInert()) {
    author_block::aut.Finish();
    author_block::aut.Report("compress");
  }
  if (!TimestampProgressInert()) {
    ts_progress::aut.Finish();
    ts_progress::aut.Report("compress");
  }
  *output_bytes = ftell(os);
}

void Decompress(unsigned long long output_length, FILE* is,
                FILE* os, Predictor* p) {
  Decoder d(is, p);
  unsigned long long percent = 1 + (output_length / 10000);
  unsigned long long prog_left = 1;
  ClearOutput();
  int sem_pb = 0;   // the previous stream byte (0 before the first)
  for(unsigned long long pos = 0; pos < output_length; ++pos) {
    int byte = 1;
    p->SemByteStart(sem_pb);
    const int zw = p->Tf1ZwangByte();
    if (zw >= 0) {
      for (int j = 7; j >= 0; --j) d.ObserveKnownBit((zw >> j) & 1);
      byte = 256 + zw;
      if (run_mode::gram.ByteActive(pos)) run_mode::gram.NextByte((unsigned int)zw);
    } else
    if (metablock::zw::an) byte = TextStartSepDec(d);
    else
    if (run_mode::gram.ByteActive(pos)) {
      while (byte < 256) {
        const int f = run_mode::gram.Forced((unsigned int)byte);
        int bit;
        if (f < 0) {
          run_mode::gram.CountFree((unsigned int)byte);
          bit = d.Decode();
        } else {
          run_mode::gram.CountKnown((unsigned int)byte);
          bit = f;
          d.ObserveKnownBit(bit);
        }
        byte += byte + bit;
      }
      run_mode::gram.NextByte((unsigned int)byte & 255);
    } else
    {
    while (byte < 256) {
      byte += byte + d.Decode();
    }
    }
    putc(byte, os);
    sem_pb = byte & 255;   // the stream byte for the next SemByteStart, as in Compress()
    AfterStreamByte(p, (unsigned char)(byte & 255));   // as in Compress()
    if (--prog_left == 0) {
      prog_left = percent;
      double frac = 100.0 * pos / output_length;
      HUGEPAGE_TICK(pos);
      console_progress::Tick(pos, output_length);
    }
  }
  run_mode::gram.Report("decompress");
  if (!MetablockInert()) {
    metablock::mk.Report("decompress");
    census_block::aut.Report("decompress");
    metablock::mk.ByteWacht("decompress", output_length);
  }
  redirect_table::tab.Schluss();
  alias_table::tab.Schluss();
  alias_context::aut.Finish();
  alias_context::aut.Report("decompress");
  if (!AuthorBlockInert()) {
    author_block::aut.Finish();
    author_block::aut.Report("decompress");
  }
  if (!TimestampProgressInert()) {
    ts_progress::aut.Finish();
    ts_progress::aut.Report("decompress");
  }
}

bool Store(const std::string& input_path, const std::string& temp_path,
    const std::string& output_path, FILE* dictionary,
    unsigned long long* input_bytes, unsigned long long* output_bytes) {
  FILE* data_in = fopen(input_path.c_str(), "rb");
  if (!data_in) return false;
  FILE* data_out = fopen(output_path.c_str(), "wb");
  if (!data_out) return false;
  fseek(data_in, 0L, SEEK_END);
  *input_bytes = ftell(data_in);
  fseek(data_in, 0L, SEEK_SET);
  WriteStorageHeader(data_out, dictionary != NULL);
  preprocessor::Encode(data_in, data_out, *input_bytes, dictionary);
  fseek(data_out, 0L, SEEK_END);
  *output_bytes = ftell(data_out);
  fclose(data_in);
  fclose(data_out);
  return true;
}

#ifdef NEIF_TEST
// per-site time table of the transformer (src/tf1/opt/model_opt.cpp; only in a build with -DFX2_PROF)
namespace fx2 { namespace opt { bool prof_enabled(); void prof_print(double tsc_to_core, long tokens); } }
static unsigned long long NeifEnv(const char* name, unsigned long long def) {
  const char* v = getenv(name);
  return (v && *v) ? strtoull(v, nullptr, 10) : def;
}
// `cmix -t STREAM OUT`: codes bytes [warm_from, end) of the coder stream STREAM
// (the file `-e enwik9` hands to Compress(); training/tf/1_dump: cmix_stream
// writes it as out.cmix.temp) and writes the coder output to OUT.
// Environment: NEIF_T_START, NEIF_T_END (the chunk, stream bytes; default the
// whole stream), NEIF_T_WARM (warm-up bytes in front of START; the model starts
// at the first page start at or behind START - WARM, or at byte 0 if WARM >=
// START), NEIF_T_JSON (result file).  The run mirrors RunCompression() from
// the header on: as in the decoder, everything the coder needs comes from the
// stream itself (the dictionary is slot 1 of the text start), so no file of
// the preprocessing is read.  The weights container must lie in the working
// directory; ./.dict and ./ppm.page are written there (one directory per
// process).
static int NeifTestMain(const char* stream_path, const char* out_path) {
  FILE* in = fopen(stream_path, "rb");
  if (!in) { fprintf(stderr, "neif_test: cannot open %s\n", stream_path); return 2; }
  fseek(in, 0L, SEEK_END);
  const unsigned long long stream_bytes = ftell(in);
  fseek(in, 0L, SEEK_SET);
  FILE* out = fopen(out_path, "wb");
  if (!out) { fprintf(stderr, "neif_test: cannot write %s\n", out_path); return 2; }
  unsigned long long start = NeifEnv("NEIF_T_START", 0);
  unsigned long long end = NeifEnv("NEIF_T_END", stream_bytes);
  const unsigned long long warm = NeifEnv("NEIF_T_WARM", 0);
  if (end > stream_bytes) end = stream_bytes;
  if (start >= end) { fprintf(stderr, "neif_test: empty chunk\n"); return 2; }
  neif_test::g_on = true;
  neif_test::g_start = start;
  neif_test::g_end = end;
  neif_test::g_warm_target = start > warm ? start - warm : 0;
  neif_test::g_skip = neif_test::g_warm_target > 0;
  neif_test::g_prof = NeifEnv("NEIF_PROF", 0) != 0;
#ifdef NEIF_MT
  neif_mt::Init();   // NEIF_MT=n: n threads for the regions of the transformer step (src/neif_mt.h)
#endif
  // NEIF_PIPE=N: the model bytes run as a pipeline of N processes (2, or 3 and
  // more = 3; src/neif_test.h); NEIF_PIPE_SPLIT: the first transformer layer
  // of the second stage for N = 3 (1..12: the fast weights sit in the last four layers)
  neif_pipe_n = (int)NeifEnv("NEIF_PIPE", 0);
  neif_test::g_tf_split = (int)NeifEnv("NEIF_PIPE_SPLIT", 8);
  if (neif_test::g_tf_split < 1 || neif_test::g_tf_split > 12) { fprintf(stderr, "neif_test: NEIF_PIPE_SPLIT must be 1..12\n"); return 2; }
#ifdef NEIF_PAR
  // NEIF_PAR=1: the measurements for a multi-threaded decoder (src/neif_test.h).
  // They ride on the profile marks, so the profile is on; one process only.
  // NEIF_PAR_H: the cost of one hand-off between threads in time-stamp ticks;
  // NEIF_PAR_COMMIT: the work that stays behind a right guess, in ticks.
  neif_test::par::g_par = NeifEnv("NEIF_PAR", 0) != 0;
  neif_test::par::g_h = NeifEnv("NEIF_PAR_H", 1500);
  neif_test::par::g_cm = NeifEnv("NEIF_PAR_COMMIT", 10000);
  neif_test::par::g_ab = NeifEnv("NEIF_PAR_ABORT", 15000);   // a guess given up: stop the step, undo PPMd (ticks)
  if (neif_test::par::g_par) {
    if (neif_pipe_n > 1) { fprintf(stderr, "neif_test: NEIF_PAR measures one process, not a pipeline\n"); return 2; }
    neif_test::g_prof = true;
  }
#endif
  neif_stream_path = stream_path;
  if (!neif_test::g_skip) neif_test::g_warm_from = 0;

  // from here as RunCompression() behind the preprocessing
  std::vector<bool> vocab(256, false);
  vocab_frozen::FillVocab(&vocab);
  WriteHeaderVocfix(stream_bytes, true, out);   // the length of the whole stream, as in a real run
  dict_article::strom = stream_bytes;
  dict_article::wrt = true;
  dict_article::Verbergen();
  Predictor p(vocab);
  HUGEPAGE_COLLAPSE("c-init");
  malloc_trim(0);
  const double t0 = console_progress::Now();
  neif_t_model = t0;
  neif_cpu_model = NeifCpuNow();
  neif_test::ProfReset();
  unsigned long long output_bytes = 0;
  Compress(end, in, out, &output_bytes, &p);
  const double t1 = console_progress::Now();
#ifdef NEIF_PAR
  // NEIF_PAR: the sections cover the kept bytes only, so the table is scaled to their share of the time
  neif_test::ProfPrint(stderr, neif_test::g_ctl ? "coder" : "single", neif_test::par::KeptSeconds(t1 - neif_t_model));
#else
  neif_test::ProfPrint(stderr, neif_test::g_ctl ? "coder" : "single", t1 - neif_t_model);
#endif
#ifdef NEIF_PAR
  // NEIF_PAR: the result file of the measurement (NEIF_PAR_JSON, default ./neif_par.json)
  if (neif_test::par::g_par) {
    const char* pj = getenv("NEIF_PAR_JSON");
    FILE* pf = fopen((pj && *pj) ? pj : "neif_par.json", "w");
    if (pf) {
      neif_test::par::Json(pf, t1 - neif_t_model, (unsigned long long)neif_test::g_warm_from, end);
      fclose(pf);
    } else {
      fprintf(stderr, "neif_par: cannot write the result file\n");
    }
  }
#endif
#ifdef NEIF_MT
  if (neif_mt::On()) fprintf(stderr, "neif_mt: %d threads, %llu regions on the pool\n", neif_mt::g_n, neif_mt::g_forks);
  if (neif_mt::g_spec) { p.NeifSpecReport(stderr); fprintf(stderr, "neif_mt spec: %llu guesses made wrong on purpose\n", neif_mt_faults); }
#endif
  fprintf(stderr, "neif_rows: transformer rows up to byte %llu: %llu, of them with the network (from byte %lld): %llu, read rows %llu\n",
          end, neif_test::g_tf_rows_all, neif_test::g_warm_from, neif_test::g_tf_rows_model, neif_test::g_tf_rows_read);
  // FX2_PROF build, one process: raw time-stamp ticks per site (tokens = 1: totals and call counts)
  if (fx2::opt::prof_enabled() && !neif_test::g_ctl) { fflush(stderr); fx2::opt::prof_print(1.0, 1); fflush(stdout); }
  // pipeline: the front stages have passed the last byte before the coder
  // stage; take their figures and wait for them (before the Predictor goes:
  // the PPMd heap is theirs while they live)
  const int n_front = neif_test::g_ctl ? neif_test::g_stage : 0;
  double model_cpu = NeifCpuNow() - neif_cpu_model, front_cpu = 0;
  long pss_end_kb = 0;
  if (neif_test::g_ctl) {
    pss_end_kb = NeifPssKb(0);
    for (int k = 0; k < n_front; ++k) {
      int st = 0;
      if (neif_test::g_kids[k] > 0 && (waitpid(neif_test::g_kids[k], &st, 0) != neif_test::g_kids[k] ||
                                       !WIFEXITED(st) || WEXITSTATUS(st) != 0)) {
        fprintf(stderr, "neif_test: front stage %d failed\n", k);
        return 3;
      }
      front_cpu += neif_test::g_ctl->cpu_s[k];
      pss_end_kb += neif_test::g_ctl->pss_kb[k];
    }
    if (pss_end_kb > neif_pss_peak_kb) neif_pss_peak_kb = pss_end_kb;
    fprintf(stderr, "neif_pipe coder stage: %.1f s wall, %.1f s cpu, %llu queue waits\n", t1 - neif_t_model,
            model_cpu, neif_test::g_waits);
  }
  fclose(in);
  fclose(out);
  remove(".dict.ta");
  if (neif_test::g_warm_from < 0) { fprintf(stderr, "neif_test: no page start behind the warm-up target\n"); return 2; }

  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  const unsigned long long wf = (unsigned long long)neif_test::g_warm_from;
  const unsigned long long kept_from = start > wf ? start : wf;   // START itself unless the page start fell behind it
  const char* jp = getenv("NEIF_T_JSON");
  FILE* j = (jp && *jp) ? fopen(jp, "w") : stdout;
  if (!j) j = stdout;
  fprintf(j, "{\"start\": %llu, \"end\": %llu, \"warm_from\": %llu, \"kept_from\": %llu,\n"
             " \"stream_bytes\": %llu, \"seconds\": %.1f, \"skip_seconds\": %.1f, \"model_seconds\": %.1f,\n"
             " \"model_bytes_per_second\": %.1f,\n"
             " \"coded_bytes\": %llu, \"total_bits\": %.3f, \"kept_bits\": %.3f, \"kept_bytes\": %.3f,\n"
             " \"coded_bit_count\": %llu, \"kept_bit_count\": %llu, \"coded_vs_bits_ratio\": %.6f,\n"
             " \"kept_bits_per_byte\": %.5f, \"forced_byte_mismatch\": %llu,\n"
             " \"structure_hash\": \"%016llx\", \"tf_rows_hashed\": %llu, \"pages_at_end\": %u, \"peak_rss_kb\": %ld,\n",
          start, end, wf, kept_from, stream_bytes, t1 - t0, neif_t_model - t0, t1 - neif_t_model,
          (double)(end - wf) / (t1 - neif_t_model > 0 ? t1 - neif_t_model : 1.0),
          output_bytes - 5, neif_test::g_bits_all, neif_test::g_bits_kept, neif_test::g_bits_kept / 8.0,
          neif_test::g_n_all, neif_test::g_n_kept,
          neif_test::g_bits_all > 0 ? (double)(output_bytes - 5) * 8.0 / neif_test::g_bits_all : 0.0,
          end > kept_from ? neif_test::g_bits_kept / (double)(end - kept_from) : 0.0,
          neif_test::g_zw_mismatch, (unsigned long long)neif_test::g_hash, neif_test::g_tf_rows,
          tf_gate::ctx.pages, ru.ru_maxrss);
  // the tap sums (neif_test::CostTap): the cost of the same coded bits at the
  // return value of Predictor::Predict(), in front of BITLSTM, the floor stage
  // and the semantic-cache correction; split as the full cost is (all model
  // bytes from warm_from, and the kept part from kept_from)
  fprintf(j, " \"tap_total_bits\": %.3f, \"tap_kept_bits\": %.3f, \"tap_kept_bytes\": %.3f,\n"
             " \"tap_bit_count\": %llu, \"tap_kept_bit_count\": %llu, \"tap_kept_bits_per_byte\": %.5f,\n"
             " \"tap_clamped_bits\": %llu,\n",
          neif_test::g_tap_all, neif_test::g_tap_kept, neif_test::g_tap_kept / 8.0,
          neif_test::g_tap_n_all, neif_test::g_tap_n_kept,
          end > kept_from ? neif_test::g_tap_kept / (double)(end - kept_from) : 0.0,
          neif_test::g_tap_clamped);
  // one text line with both costs side by side (stderr, as the other reports of the mode)
  fprintf(stderr, "neif_test cost: full %.3f bits (kept %.3f), at Predict() %.3f bits (kept %.3f), "
                  "%llu coded bits (kept %llu)\n",
          neif_test::g_bits_all, neif_test::g_bits_kept, neif_test::g_tap_all, neif_test::g_tap_kept,
          neif_test::g_n_all, neif_test::g_n_kept);
  // pipe: number of stages (0 = one process).  model_cpu_seconds: CPU time of
  // all stages over the model bytes; cpu_seconds: the same plus start-up and
  // structure pass.  pipe_peak_pss_kb: the largest sum of the stages'
  // proportional set sizes, sampled every 5 s and at the end (peak_rss_kb
  // above is the coder stage alone and counts pages shared with the others).
  fprintf(j, " \"pipe\": %d, \"tf_split\": %d, \"model_cpu_seconds\": %.1f, \"cpu_seconds\": %.1f,\n"
             " \"front_cpu_seconds\": [%.1f, %.1f], \"pipe_peak_pss_kb\": %ld}\n",
          n_front ? n_front + 1 : 0, n_front == 2 ? neif_test::g_tf_split : 0, model_cpu + front_cpu,
          NeifCpuNow() + front_cpu, n_front > 0 ? neif_test::g_ctl->cpu_s[0] : 0.0,
          n_front > 1 ? neif_test::g_ctl->cpu_s[1] : 0.0, neif_pss_peak_kb);
  if (j != stdout) fclose(j);
  return 0;
}

// ---- chunk decoder (`cmix -u STREAM CHUNK`) ---------------------------------
// The decoding counterpart of `-t`: CHUNK is the output of a `-t` run with the
// same NEIF_T_START / NEIF_T_END / NEIF_T_WARM (its 5-byte header, then the
// coder output of the stream bytes [warm_from, end)).  The stream bytes in
// front of warm_from get the same structure-only pass as in `-t` (they are
// read from STREAM); from warm_from on every byte is DECODED from CHUNK with
// the byte loop of Decompress() and compared with the byte of STREAM.  The
// decoder is lost at the first wrong byte, so the run stops there and reports
// the position.  Used to check that a chunk decodes (the test bed of the
// multi-threaded decoder) and to time decoding against coding on one window.
// NEIF_T_JSON names the result file.
static unsigned long long neif_u_bad = 0;      // decoded bytes that differ from the stream (0 or 1: the run stops)
static long long neif_u_first_bad = -1;        // stream position of the first
static unsigned long long neif_u_decoded = 0;  // bytes decoded and found equal
static void NeifDecompressChunk(unsigned long long end, FILE* stream, FILE* chunk, Predictor* p) {
  Decoder d(chunk, p);   // before the structure-only pass, where `-t` makes its Encoder
  ClearOutput();
  static unsigned char buffer[1 << 20];
  unsigned long long percent = 1 + (end / (unsigned long long)progress_steps);
  unsigned long long prog_left = 1;
  unsigned long long pos = 0, neif_tail = 0;
  unsigned long long neif_read = 0;   // stream bytes read so far (whole blocks)
  int sem_pb = 0;
  while (pos < end && !neif_u_bad) {
    const size_t want = (size_t)std::min<unsigned long long>(sizeof buffer, end - pos);
    const size_t got = fread(buffer, 1, want, stream);
    if (got == 0) break;
    neif_read += got;
    for (size_t i = 0; i < got && !neif_u_bad; ++i) {
      const unsigned char c = buffer[i];
      // the same lines as in Compress(): end of the structure-only pass, structure events
      if (neif_test::g_skip && pos >= neif_test::g_warm_target && neif_tail == 0x20204c2fdf984e0aull) {
        neif_test::g_skip = false;
        neif_test::g_warm_from = (long long)pos;
        neif_t_model = console_progress::Now();
        neif_cpu_model = NeifCpuNow();
        p->NeifSkipEnd();
        neif_test::ProfReset();
      }
      // validation decode (NEIF_PIPE=N): the stages part here, as in Compress().
      // A front stage goes on in this loop with its own handle of the EXPECTED
      // stream and never touches the decoder; this process stays the coder stage.
      if (!neif_test::g_skip && neif_pipe_n > 1 && !neif_test::g_ctl) NeifPipeStart(p, &stream, neif_read);
      neif_test::g_pos = pos;
      neif_tail = (neif_tail << 8) | c;
      neif_test::g_hash_on = pos >= neif_test::g_start;
      neif_test::g_count = !neif_test::g_skip && pos >= neif_test::g_start;
      const int nzw = p->Tf1ZwangByte();
      neif_test::H(c);
      neif_test::H((unsigned)(nzw + 1));
      neif_test::H(metablock::zw::an ? 1u : 0u);
      neif_test::H(run_mode::gram.ByteActive(pos) ? (unsigned)run_mode::gram.DbgMask() : 0xffu);
      neif_test::H(tf_gate::ctx.zone);
      neif_test::H(tf_gate::ctx.pages);
      if (nzw >= 0 && nzw != (int)c) ++neif_test::g_zw_mismatch;
      if (neif_test::g_skip || neif_test::Front()) {   // as in Compress(): no coder
        if (nzw >= 0) {
          if (run_mode::gram.ByteActive(pos)) run_mode::gram.NextByte((unsigned int)nzw);
        } else if (metablock::zw::an) {
        } else if (run_mode::gram.ByteActive(pos)) {
          run_mode::gram.NextByte(c);
        }
        const unsigned int mb = nzw >= 0 ? (unsigned int)nzw : (unsigned int)c;
        if (neif_test::g_skip) p->NeifSkipByte(mb); else p->NeifFrontByte(mb);
        sem_pb = c;
        AfterStreamByte(p, c);
        if (neif_test::g_skip) p->NeifSkipSync();
        ++pos;
        continue;
      }
      // one decoded byte: the body of the loop of Decompress()
      int byte = 1;
      p->SemByteStart(sem_pb);
      const int zw = p->Tf1ZwangByte();
#ifdef NEIF_MT
      // NEIF_MT: a forced byte is known before its bits; with NEIF_MT_SPEC=2
      // (a test of the machinery, not a decoder) the guess is the stream byte
      if (neif_mt::g_spec) p->NeifSpecLaunch(NeifSpecGuess(zw >= 0 ? zw : (neif_mt::g_spec == 2 ? (int)c : -1)));
#endif
      if (zw >= 0) {
        for (int j = 7; j >= 0; --j) d.ObserveKnownBit((zw >> j) & 1);
        byte = 256 + zw;
        if (run_mode::gram.ByteActive(pos)) run_mode::gram.NextByte((unsigned int)zw);
      } else
      if (metablock::zw::an) byte = TextStartSepDec(d);
      else
      if (run_mode::gram.ByteActive(pos)) {
        while (byte < 256) {
          const int f = run_mode::gram.Forced((unsigned int)byte);
          int bit;
          if (f < 0) {
            run_mode::gram.CountFree((unsigned int)byte);
            bit = d.Decode();
          } else {
            run_mode::gram.CountKnown((unsigned int)byte);
            bit = f;
            d.ObserveKnownBit(bit);
          }
          byte += byte + bit;
        }
        run_mode::gram.NextByte((unsigned int)byte & 255);
      } else
      {
      while (byte < 256) {
        byte += byte + d.Decode();
      }
      }
      if ((byte & 255) != (int)c) {   // the decoder has left the stream
        ++neif_u_bad;
        neif_u_first_bad = (long long)pos;
        break;
      }
      ++neif_u_decoded;
      sem_pb = byte & 255;
      AfterStreamByte(p, (unsigned char)(byte & 255));
      if (--prog_left == 0) {
        prog_left = percent;
        HUGEPAGE_TICK(pos);
        console_progress::Tick(pos, end);
        NeifPssSample();
        // validation decode: a line every NEIF_SV_LOG seconds (default 600), so
        // that a run that breaks late still certifies a prefix and leaves its
        // speed and memory over the stream
        if (neif_test::g_sv) {
          static double next = 0;
          static const double step = (double)NeifEnv("NEIF_SV_LOG", 600);
          const double now = console_progress::Now();
          if (now >= next) {
            if (next > 0)
              fprintf(stderr, "\nneif_sv progress: stream bytes [%lld, %llu) equal so far, %.0f bytes/s, peak %ld MiB in all stages\n",
                      neif_test::g_warm_from, pos + 1, (double)neif_u_decoded / (now - neif_t_model > 0 ? now - neif_t_model : 1.0),
                      neif_pss_peak_kb >> 10);
            next = now + step;
          }
        }
      }
      ++pos;
    }
  }
  if (neif_test::Front()) NeifFrontEnd();   // a front stage of the validation decode ends here
}

static int NeifUntestMain(const char* stream_path, const char* chunk_path) {
  FILE* in = fopen(stream_path, "rb");
  if (!in) { fprintf(stderr, "neif_test: cannot open %s\n", stream_path); return 2; }
  fseek(in, 0L, SEEK_END);
  const unsigned long long stream_bytes = ftell(in);
  fseek(in, 0L, SEEK_SET);
  FILE* chunk = fopen(chunk_path, "rb");
  if (!chunk) { fprintf(stderr, "neif_test: cannot open %s\n", chunk_path); return 2; }
  unsigned long long start = NeifEnv("NEIF_T_START", 0);
  unsigned long long end = NeifEnv("NEIF_T_END", stream_bytes);
  const unsigned long long warm = NeifEnv("NEIF_T_WARM", 0);
  if (end > stream_bytes) end = stream_bytes;
  if (start >= end) { fprintf(stderr, "neif_test: empty chunk\n"); return 2; }
  neif_test::g_on = true;
  neif_test::g_start = start;
  neif_test::g_end = end;
  neif_test::g_warm_target = start > warm ? start - warm : 0;
  neif_test::g_skip = neif_test::g_warm_target > 0;
  neif_test::g_prof = NeifEnv("NEIF_PROF", 0) != 0;
#ifdef NEIF_MT
  neif_mt::Init();   // NEIF_MT=n: n threads for the regions of the transformer step (src/neif_mt.h)
#endif
  // NEIF_PIPE=N (2, or 3 and more = 3): the validation decode.  The front
  // stages of the pipeline of mode -t walk STREAM, the expected stream; this
  // process becomes the coder stage, decodes CHUNK and compares.  See
  // src/neif_test.h (g_sv) and runs/neif-par/PROGRESS.md.
  neif_pipe_n = (int)NeifEnv("NEIF_PIPE", 0);
  neif_test::g_tf_split = (int)NeifEnv("NEIF_PIPE_SPLIT", 8);
  if (neif_test::g_tf_split < 1 || neif_test::g_tf_split > 12) { fprintf(stderr, "neif_test: NEIF_PIPE_SPLIT must be 1..12\n"); return 2; }
  neif_test::g_sv = neif_pipe_n > 1;
#ifdef NEIF_MT
  if (neif_test::g_sv && (neif_mt::On() || neif_mt::g_spec)) { fprintf(stderr, "neif_test: NEIF_PIPE and NEIF_MT do not go together\n"); return 2; }
#endif
  neif_stream_path = stream_path;
  if (!neif_test::g_skip) neif_test::g_warm_from = 0;

  // as RunDecompression(): the header, then the predictor
  std::vector<bool> vocab(256, false);
  unsigned long long length = 0;
  bool dictionary_used = false;
  ReadHeaderVocfix(chunk, &length, &dictionary_used, &vocab);
  if (length != (stream_bytes & 0x3FFFFFFFFFull) || !dictionary_used) {
    fprintf(stderr, "neif_test: %s is not a chunk of this stream (header %llu, stream %llu)\n", chunk_path, length,
            stream_bytes);
    return 2;
  }
  dict_article::strom = stream_bytes;
  dict_article::wrt = true;
  dict_article::Verbergen();
  Predictor p(vocab);
  HUGEPAGE_COLLAPSE("c-init");
  malloc_trim(0);
  const double t0 = console_progress::Now();
  neif_t_model = t0;
  neif_cpu_model = NeifCpuNow();
  neif_test::ProfReset();
  NeifDecompressChunk(end, in, chunk, &p);
  const double t1 = console_progress::Now();
  neif_test::ProfPrint(stderr, "decode", t1 - neif_t_model);
#ifdef NEIF_MT
  if (neif_mt::On()) fprintf(stderr, "neif_mt: %d threads, %llu regions on the pool\n", neif_mt::g_n, neif_mt::g_forks);
  if (neif_mt::g_spec) { p.NeifSpecReport(stderr); fprintf(stderr, "neif_mt spec: %llu guesses made wrong on purpose\n", neif_mt_faults); }
#endif
  double model_cpu = NeifCpuNow() - neif_cpu_model, front_cpu = 0;
  // validation decode: the front stages.  Behind a wrong byte they are stopped
  // (they walk on through the expected stream); otherwise they have passed the
  // last byte before this stage and must have ended in order.
  const int n_front = neif_test::g_ctl ? neif_test::g_stage : 0;
  bool front_bad = false;
  if (neif_test::g_ctl) {
    long pss_end_kb = NeifPssKb(0);
    for (int k = 0; k < n_front; ++k) {
      int st = 0;
      if (neif_test::g_kids[k] <= 0) { front_cpu += neif_test::g_ctl->cpu_s[k]; continue; }   // reaped in PipeWait, in order
      if (neif_u_bad) { kill(neif_test::g_kids[k], SIGKILL); waitpid(neif_test::g_kids[k], &st, 0); continue; }
      if (waitpid(neif_test::g_kids[k], &st, 0) != neif_test::g_kids[k] || !WIFEXITED(st) || WEXITSTATUS(st) != 0) {
        fprintf(stderr, "neif_test: front stage %d failed\n", k);
        front_bad = true;
        continue;
      }
      front_cpu += neif_test::g_ctl->cpu_s[k];
      pss_end_kb += neif_test::g_ctl->pss_kb[k];
    }
    if (pss_end_kb > neif_pss_peak_kb) neif_pss_peak_kb = pss_end_kb;
    fprintf(stderr, "neif_pipe coder stage: %.1f s wall, %.1f s cpu, %llu queue waits\n", t1 - neif_t_model,
            model_cpu, neif_test::g_waits);
  }
  fclose(in);
  fclose(chunk);
  remove(".dict.ta");
  if (neif_test::g_warm_from < 0) { fprintf(stderr, "neif_test: no page start behind the warm-up target\n"); return 2; }
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  const unsigned long long wf = (unsigned long long)neif_test::g_warm_from;
  // a pass needs every byte of [warm_from, end): a stream that ends early is not one
  const bool short_run = !neif_u_bad && neif_u_decoded != end - wf;
  const char* jp = getenv("NEIF_T_JSON");
  FILE* j = (jp && *jp) ? fopen(jp, "w") : stdout;
  if (!j) j = stdout;
  fprintf(j, "{\"mode\": \"decode\", \"start\": %llu, \"end\": %llu, \"warm_from\": %llu,\n"
             " \"seconds\": %.1f, \"skip_seconds\": %.1f, \"model_seconds\": %.1f, \"model_cpu_seconds\": %.1f,\n"
             " \"model_bytes_per_second\": %.1f,\n"
             " \"decoded_bytes\": %llu, \"wrong_bytes\": %llu, \"first_wrong_byte\": %lld,\n"
             " \"forced_byte_mismatch\": %llu, \"structure_hash\": \"%016llx\", \"peak_rss_kb\": %ld,\n"
             " \"pipe\": %d, \"tf_split\": %d, \"front_cpu_seconds\": [%.1f, %.1f], \"pipe_peak_pss_kb\": %ld,\n"
             " \"short_run\": %d, \"front_failed\": %d}\n",
          start, end, wf, t1 - t0, neif_t_model - t0, t1 - neif_t_model, model_cpu + front_cpu,
          (double)neif_u_decoded / (t1 - neif_t_model > 0 ? t1 - neif_t_model : 1.0),
          neif_u_decoded, neif_u_bad, neif_u_first_bad, neif_test::g_zw_mismatch,
          (unsigned long long)neif_test::g_hash, ru.ru_maxrss,
          n_front ? n_front + 1 : 0, n_front == 2 ? neif_test::g_tf_split : 0,
          n_front > 0 ? neif_test::g_ctl->cpu_s[0] : 0.0, n_front > 1 ? neif_test::g_ctl->cpu_s[1] : 0.0,
          neif_pss_peak_kb, short_run ? 1 : 0, front_bad ? 1 : 0);
  if (j != stdout) fclose(j);
  fprintf(stderr, "neif_test decode: %llu bytes decoded from stream byte %llu, %s\n", neif_u_decoded, wf,
          neif_u_bad ? "then a WRONG byte" : (short_run ? "then the stream ENDED EARLY" : "all equal to the stream"));
  if (neif_test::g_sv) {
    if (neif_u_bad)
      fprintf(stderr, "neif_sv: FAILED at stream byte %lld: the decoded byte differs from the expected stream\n", neif_u_first_bad);
    else if (short_run || front_bad)
      fprintf(stderr, "neif_sv: NO VERDICT: %s; %llu bytes from stream byte %llu were equal\n",
              short_run ? "the expected stream could not be read to the end of the range" : "a front stage failed",
              neif_u_decoded, wf);
    else
      fprintf(stderr, "neif_sv: PASS: stream bytes [%llu, %llu) decoded and equal to the expected stream\n", wf, end);
  }
  return neif_u_bad ? 4 : (short_run ? 5 : (front_bad ? 3 : 0));
}
#endif

#ifdef NEIF_TEST
// ---- multi-process compression in the real path (`cmix -e`, NEIF_PIPE=2|3) ---
// A convenience for development, NOT the official run (the contest is one
// core): default off; with NEIF_PIPE=2 or 3 in the environment the coder of
// the normal compress command runs as the pipeline of mode -t (src/neif_test.h)
// from stream byte 0 to the end of the stream, with the end-of-stream checks
// of Compress().  The coder output must be, and on everything compared is,
// byte-identical to the one-process run of the same build
// (runs/neif-par/PROGRESS.md, "Multi-process compression").
// A broken pipeline never leaves an output: PipeFail() (a front stage that
// died, no shared memory, stages out of step) and a front stage that did not
// end in order remove the coder output and end the process with exit code 3,
// before archive9 is put together.
// NEIF_E_STOP=n (with or without NEIF_PIPE; for tests): the coder stops behind
// n stream bytes, flushes, and the process ends there: the output file is the
// header and the coder output of the first n stream bytes, and no archive9 is
// written.
static bool neif_e_on = false;
static bool NeifEStart(const char* temp_path, const char* out_path, unsigned long long temp_bytes,
                       unsigned long long* run_bytes) {
  const int n = (int)NeifEnv("NEIF_PIPE", 0);
  neif_e_stop = NeifEnv("NEIF_E_STOP", 0);
  if (neif_e_stop >= temp_bytes) neif_e_stop = 0;
  if (neif_e_stop) *run_bytes = neif_e_stop;
  if (n > 1) {
    neif_test::g_tf_split = (int)NeifEnv("NEIF_PIPE_SPLIT", 8);
    if (neif_test::g_tf_split < 1 || neif_test::g_tf_split > 12) { fprintf(stderr, "neif_test: NEIF_PIPE_SPLIT must be 1..12\n"); return false; }
    if (NeifEnv("NEIF_MT", 0)) { fprintf(stderr, "neif_test: NEIF_PIPE and NEIF_MT do not go together\n"); return false; }
    neif_pipe_n = n;
    neif_e_real = true;
    // the byte loop of Compress() as mode -t runs it from stream byte 0: no structure-only pass
    neif_test::g_on = true;
    neif_test::g_start = 0;
    neif_test::g_end = *run_bytes;
    neif_test::g_warm_target = 0;
    neif_test::g_skip = false;
    neif_test::g_warm_from = 0;
    neif_stream_path = temp_path;
    neif_test::g_out_path = out_path;
    fprintf(stderr, "\nneif_pipe: MULTI-PROCESS COMPRESSION (NEIF_PIPE=%d, %d processes; test build): a development\n"
                    "neif_pipe: convenience, NOT the official single-core run.\n", n, n > 3 ? 3 : n);
  }
  if (neif_e_stop)
    fprintf(stderr, "\nneif_e: NEIF_E_STOP=%llu: the coder stops behind stream byte %llu of %llu; no archive9 will be written\n",
            neif_e_stop, neif_e_stop, temp_bytes);
  if (n > 1 || neif_e_stop) {
    neif_e_on = true;
    neif_test::g_prof = NeifEnv("NEIF_PROF", 0) != 0;
    neif_t_model = console_progress::Now();
    neif_cpu_model = NeifCpuNow();
    neif_test::ProfReset();
  }
  return true;
}
// behind Compress(), while the Predictor lives (the PPMd heap belongs to the first stage while it lives)
static void NeifEEnd(unsigned long long run_bytes, unsigned long long output_bytes, FILE* temp_in, FILE* data_out,
                     const char* temp_path) {
  if (!neif_e_on) return;
  const double t1 = console_progress::Now();
  if (neif_test::g_prof) neif_test::ProfPrint(stderr, neif_test::g_ctl ? "coder" : "single", t1 - neif_t_model);
  const int n_front = neif_test::g_ctl ? neif_test::g_stage : 0;
  const double model_cpu = NeifCpuNow() - neif_cpu_model;
  double front_cpu = 0;
  if (neif_test::g_ctl) {
    long pss_end_kb = NeifPssKb(0);
    for (int k = 0; k < n_front; ++k) {
      int st = 0;
      // (a stage with g_kids[k] == 0 was taken in PipeWait(), ended in order)
      if (neif_test::g_kids[k] > 0) {
        const int kid = neif_test::g_kids[k];
        const bool ok = waitpid(kid, &st, 0) == kid && WIFEXITED(st) && WEXITSTATUS(st) == 0 &&
                        neif_test::g_ctl->done[k].load();
        neif_test::g_kids[k] = 0;
        if (!ok) neif_test::PipeFail("a front stage did not end in order behind the last byte");
      }
      front_cpu += neif_test::g_ctl->cpu_s[k];
      pss_end_kb += neif_test::g_ctl->pss_kb[k];
    }
    if (pss_end_kb > neif_pss_peak_kb) neif_pss_peak_kb = pss_end_kb;
    fprintf(stderr, "neif_pipe coder stage: %.1f s wall, %.1f s cpu, %llu queue waits\n", t1 - neif_t_model, model_cpu,
            neif_test::g_waits);
  }
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  fprintf(stderr, "neif_e: %llu stream bytes -> %llu B, coder %.1f s wall, %.1f s cpu in all processes, "
                  "processes %d, peak memory %ld MiB (%s)\n",
          run_bytes, output_bytes, t1 - neif_t_model, model_cpu + front_cpu, n_front + 1,
          (n_front ? neif_pss_peak_kb : ru.ru_maxrss) >> 10,
          n_front ? "largest sum of the processes' proportional set sizes" : "peak resident set");
  if (neif_e_real)
    fprintf(stderr, "neif_pipe: MULTI-PROCESS COMPRESSION ended in order (%d processes). NOT the official single-core run;\n"
                    "neif_pipe: the output is meant to equal the one-process output of this build byte for byte.\n",
            n_front + 1);
  if (neif_e_stop) {
    fclose(temp_in);
    if (fclose(data_out) != 0) { fprintf(stderr, "neif_e: the output could not be written\n"); _exit(3); }
    remove(temp_path);
    remove(".dict.ta");
    fprintf(stderr, "neif_e: STOPPED behind stream byte %llu (NEIF_E_STOP): the output is a PREFIX run, not an archive\n",
            run_bytes);
    fflush(nullptr);
    _exit(0);
  }
}
#endif

bool RunCompression(bool enable_preprocess, const std::string& input_path,
    const std::string& temp_path, const std::string& output_path,
    FILE* dictionary, unsigned long long* input_bytes,
    unsigned long long* output_bytes,
    const char* post_wrt_side_path = nullptr) {
  FILE* data_in = fopen(input_path.c_str(), "rb");
  if (!data_in) return false;
  FILE* temp_out = fopen(temp_path.c_str(), "wb");
  if (!temp_out) return false;

  fseek(data_in, 0L, SEEK_END);
  *input_bytes = ftell(data_in);
  fseek(data_in, 0L, SEEK_SET);

  if (enable_preprocess) {
    preprocessor::Encode(data_in, temp_out, *input_bytes, dictionary);
  } else {
    preprocessor::NoPreprocess(data_in, temp_out, *input_bytes);
  }
  fclose(data_in);
  fclose(temp_out);

  if (census_block_active && !census_block::Einfuegen(temp_path, ".censblock", dictionary)) {
    msg_note(407);
    return false;
  }
  if (!metablock_form::Pack(temp_path)) {
    msg_note(171);
    return false;
  }

  if (!ip_pack::Pack(temp_path)) {
    msg_note(172);
    return false;
  }
  if (!MetablockInert() && !dict_article::SlotDatei()) {
    msg_note(547);
    return false;
  }
  if (!MetablockInert() && !rule_stages::DateiEnc(temp_path)) {
    msg_note(553);
    return false;
  }
  if (!text_start::Pack(temp_path, !MetablockInert(), !enable_preprocess)) {
    msg_note(536);
    return false;
  }

  console_progress::PhaseEnd();   // "preprocessing" ends here

  FILE* temp_in = fopen(temp_path.c_str(), "rb");
  if (!temp_in) return false;

  FILE* data_out = fopen(output_path.c_str(), "wb");
  if (!data_out) return false;

  fseek(temp_in, 0L, SEEK_END);
  unsigned long long temp_bytes = ftell(temp_in);
  fseek(temp_in, 0L, SEEK_SET);

  std::vector<bool> vocab(256, false);
#if AUX_BUILD
  vocab_frozen::g_aux = true;
#endif
  vocab_frozen::FillVocab(&vocab);
  WriteHeaderVocfix(temp_bytes, dictionary != NULL, data_out);

  dict_article::strom = temp_bytes;
  dict_article::wrt = dictionary != NULL;
  if (!MetablockInert()) dict_article::Verbergen();
  Predictor p(vocab);
  HUGEPAGE_COLLAPSE("c-init");
  malloc_trim(0);
  console_progress::PhaseEnd();   // "pretraining ..." ends here
#ifdef NEIF_TEST
  // NEIF_PIPE / NEIF_E_STOP (default off: then nothing changes), see NeifEStart()
  unsigned long long neif_run_bytes = temp_bytes;
  if (!NeifEStart(temp_path.c_str(), output_path.c_str(), temp_bytes, &neif_run_bytes)) return false;
  Compress(neif_run_bytes, temp_in, data_out, output_bytes, &p);
  NeifEEnd(neif_run_bytes, *output_bytes, temp_in, data_out, temp_path.c_str());
#else
  Compress(temp_bytes, temp_in, data_out, output_bytes, &p);
#endif
  if (!MetablockInert()) dict_article::Pruefen();
  fclose(temp_in);
  fclose(data_out);
  remove(temp_path.c_str());
  return true;
}

bool RunDecompression(const std::string& input_path,
    const std::string& temp_path, const std::string& output_path,
    FILE* dictionary, unsigned long long* input_bytes,
    unsigned long long* output_bytes,
    const char* post_wrt_side_path = nullptr) {
  FILE* data_in = fopen(input_path.c_str(), "rb");
  if (!data_in) return false;

  fseek(data_in, 0L, SEEK_END);
  *input_bytes = ftell(data_in);
  fseek(data_in, 0L, SEEK_SET);
  std::vector<bool> vocab(256, false);
  bool dictionary_used;
  ReadHeaderVocfix(data_in, output_bytes, &dictionary_used, &vocab);
  if (!dictionary_used && dictionary != NULL) return false;

  {
    dict_article::strom = *output_bytes;
    dict_article::wrt = dictionary_used;
    if (!MetablockInert()) dict_article::Verbergen();
    Predictor p(vocab);
    HUGEPAGE_COLLAPSE("d-init");

    FILE* temp_out = fopen(temp_path.c_str(), "wb");
    if (!temp_out) return false;

    console_progress::PhaseEnd();   // "pretraining ..." ends here
    Decompress(*output_bytes, data_in, temp_out, &p);
    // the inverse transforms of the main run take some minutes
    if (dictionary_used) console_progress::Phase("postprocessing");
    p.FreeFxcmMemory();
    p.FreeFxcmRest();
    fclose(data_in);
    fclose(temp_out);
  }
  malloc_trim(0);
  if (!text_start::Unpack(temp_path, !MetablockInert())) {
    msg_note(537);
    return false;
  }
  if (!MetablockInert()) {
    if (!dict_article::DateiAusSlot()) {
      msg_note(548);
      return false;
    }
    if (dict_article::wrt) {
      if (dictionary) fclose(dictionary);
      dictionary = fopen(".dict", "rb");
    }
  }
  if (!MetablockInert() && !rule_stages::DateiDec(temp_path)) {
    msg_note(554);
    return false;
  }

  if (!ip_pack::Unpack(temp_path)) {
    msg_note(173);
    return false;
  }

  if (!metablock_form::Unpack(temp_path)) {
    msg_note(174);
    return false;
  }
  if (census_block_active && !census_block::Entnehmen(temp_path, ".censblock", dictionary)) {
    msg_note(408);
    return false;
  }

  FILE* temp_in = fopen(temp_path.c_str(), "rb");
  if (!temp_in) return false;
  FILE* data_out = fopen(output_path.c_str(), "wb");
  if (!data_out) return false;

  preprocessor::Decode(temp_in, data_out, dictionary);
  isbn_check::after_at.Put(-1, data_out);
  fseek(data_out, 0L, SEEK_END);
  *output_bytes = ftell(data_out);
  fclose(temp_in);
  fclose(data_out);
  remove(temp_path.c_str());
  return true;
}

namespace ppmd_page { extern int hilfs; }
int main(int argc, char** argv) {
  console_progress::Begin();
  if (argc == 1 && no_args::IsCompressor()) {
    argc = 4;
    argv = (char**)no_args::kArgVector.argv;
  }
#if !AUX_BUILD
  if (argc > 1 && argv[1][0] == '-' && (argv[1][1] | 0x20) == 'd')   // -d or -D
#else
  if (argc > 1 && argv[1][0] == '-' && (argv[1][1] == 'c' || argv[1][1] == 'd'
      || argv[1][1] == 'C' || argv[1][1] == 'D'))
#endif
    ppmd_page::hilfs = 1;
  mallopt(M_MMAP_THRESHOLD, 1 << 20);
  if (PredictorLayoutTagFast() != ((unsigned long long)sizeof(Predictor) * 4ULL + (unsigned long long)(1) * 2ULL + (unsigned long long)(0))) {
    msg_note(175)

;
    fflush(stderr);
    return -1;
  }
#if !AUX_BUILD
  // The release form takes -e (compress), -d (decompress; a fifth argument, a
  // dictionary path, is accepted and not read) and -D (an embedded block of
  // the self-extraction).  Builds without this restriction, the build helper
  // among them, also take -c/-n/-s/-x and -C.
  if ((argc != 1) && (argc < 4 || argc > 5 || strlen(argv[1]) != 2 || argv[1][0] != '-' ||
      (argv[1][1] != 'd' && argv[1][1] != 'e'
      && argv[1][1] != 'D'
#ifdef NEIF_TEST
      && argv[1][1] != 't'   // chunk evaluation mode
      && argv[1][1] != 'u'   // chunk decoder
#endif
      ))) {
    return Help();
  }
#else
  if ((argc != 1) && (argc < 4 || argc > 5 || strlen(argv[1]) != 2 || argv[1][0] != '-' ||
      (argv[1][1] != 'c' && argv[1][1] != 'd' && argv[1][1] != 'x' && argv[1][1] != 's' &&
      argv[1][1] != 'n' && argv[1][1] != 'e'
      && argv[1][1] != 'C' && argv[1][1] != 'D'
      ))) {
    return Help();
  }
#endif
#if !AUX_BUILD
  if (argc > 1 && argv[1][1] == 'D') {
#else
  if (argc > 1 && (argv[1][1] == 'C' || argv[1][1] == 'D')) {
#endif
    transformer::SetBlobForm();
#if !AUX_BUILD
    argv[1][1] = 'd';
#else
    argv[1][1] = (argv[1][1] == 'C') ? 'c' : 'd';
#endif
  }
  srand((unsigned int)run_options::ParamI("seed", SEED));
  progress_steps = run_options::ParamI("progress_steps", 10000);
  if (progress_steps < 1) progress_steps = 10000;

#if !1 || AUX_BUILD
  clock_t start = clock();   // only for the end line of the build helper
#endif

  bool enable_preprocess = true;
  std::string input_path ;
  std::string output_path;
  FILE* dictionary = NULL;

  if (argc > 1)  {
#if !AUX_BUILD
    // the last two arguments; a dictionary argument of -d is not read, the
    // dictionary is slot 1 of the stream
    input_path = argv[argc - 2];
    output_path = argv[argc - 1];
#else
    input_path = argv[2];
    output_path = argv[3];
    if (argc == 5) {
      if (argv[1][1] == 'n') return Help();
      dictionary = fopen(argv[2], "rb");
      if (!dictionary) return Help();
      input_path = argv[3];
      output_path = argv[4];
    }
#endif
  }

  std::string temp_path = output_path + ".cmix.temp";
  const bool progress_encoding = argc > 1 && (argv[1][1] == 'c' || argv[1][1] == 'e');
  console_progress::Start(progress_encoding, output_path.c_str());

  unsigned long long input_bytes = 0, output_bytes = 0;

  if (argc == 1) {
    //Decompress enwik9
    // unpack a) header b) cmix dictionary, c) new order of articles, d) actual cmix binary
    census_block_active = true;
    if (selfextract_decomp() != 0) {
      return Help();
    }

    // run compression
    input_path = ".ready4cmix_decomp";
    output_path = ".input_decomp" ;
    dictionary = fopen(".dict", "rb");
    isbn_check::after_at.an = true;

    if (!RunDecompression(input_path, temp_path, output_path, dictionary,
        &input_bytes, &output_bytes,
        nullptr
        )) {
      return Help();
    }

    split4Decomp();

    // apply phda9 preprocessor
    phda9_resto();
    if (!census_block::Setze(".main_decomp_restored", ".censblock")) {
      msg_note(409);
      return Help();
    }
    if (!comment_form::Dekodiere(".main_decomp_restored")) {
      msg_note(484);
      return Help();
    }

    // change the order of articles in the input
    sort();

    if (!cmtidx_inplace<0>(".main_decomp_restored_sorted", ".cmtidx_resto")) {
      msg_note(176);
      return Help();
    }

    // merge all input parts after preprocessing; the census inverse transform
    // follows as the last step - mirror image of compression, where it ran first.
    {
      static const char* const teile[] = {".intro_decomp", ".main_decomp_restored_sorted",
                                          ".coda_decomp", nullptr};
      if (!cat(".census_resto", teile)) return Help();
    }
    if (!census_file<0>(".census_resto", "enwik9_uncompressed")) {
      msg_note(177);
      return Help();
    }
    remove(".census_resto");

    selfx_clean::AfterDecompress();

    goto print_end_message;
  }

#if !(1 && !AUX_BUILD)   // the package form has no -c
  if (argv[1][1] == 'c') {
      if (argv[1][1] == 'c' && argc == 5) {
        if (!provide_dict_file(argv[2])) return Help();
      }
    if (!RunCompression(enable_preprocess, input_path, temp_path, output_path,
        dictionary, &input_bytes, &output_bytes)) {
      return Help();
    }
  } else
#endif
#ifdef NEIF_TEST
  if (argv[1][1] == 't') return NeifTestMain(input_path.c_str(), output_path.c_str());
  if (argv[1][1] == 'u') return NeifUntestMain(input_path.c_str(), output_path.c_str());
#endif
  if (argv[1][1] == 'e') {
#define ENC_FAIL return Help()

    if (selfextract_comp() != 0) {
      ENC_FAIL;
    }

    {
      console_progress::Phase("dictionary");
      const bool derived = dict_derive::BuildDictFromInput(
          input_path.c_str(), ".hints", ".dict");
      console_progress::PhaseEnd();
      if (!derived) {
        msg_note(178);
        ENC_FAIL;
      }
      remove(".hints");
      remove(".hints.comp");
    }

    console_progress::Phase("preprocessing");
    if (!census_file_enc(input_path.c_str(), ".census_prepr")) {
      msg_note(181);
      ENC_FAIL;
    }
    split4Comp(".census_prepr");
    remove(".census_prepr");
    if (!cmtidx_inplace_enc(".main", ".cmtidx_main")) {
      msg_note(182);
      ENC_FAIL;
    }

    // change the order of articles in the input
    reorder();
    if (!comment_form::Kodiere(".main_reordered")) {
      msg_note(483);
      ENC_FAIL;
    }
    if (!census_block::Zerlege(".main_reordered", ".censblock")) {
      msg_note(406);
      ENC_FAIL;
    }
    census_block_active = true;

    // apply phda9 preprocessor
    phda9_prepr();

    // merge all input parts after preprocessing
    {
      static const char* const teile[] = {".main_phda9prepr", ".intro", ".coda", nullptr};
      if (!cat(".ready4cmix", teile)) ENC_FAIL;
    }
    isbn_check::Pack();

    // run compression
    input_path = ".ready4cmix";
    dictionary = fopen(".dict", "rb");
    if (!RunCompression(enable_preprocess, input_path, temp_path, output_path,
        dictionary, &input_bytes, &output_bytes,
        nullptr
        )) {
      ENC_FAIL;
    }

    // construct a selfextracting decompressor binary; in the release
    // configuration
    //   archive9 = [decoder][container][payload][trailer]
    // (the dictionary travels as slot 1 of the stream; a build without the
    // dictionary article puts .dict.comp behind the decoder, one without the
    // embedded container leaves the container out).
    // It is written under the temporary name first: the payload may be the
    // file archive9 itself (the jury form writes its output there).
    {
      ArchivKopf header;
      header.weights_size = (int)getFileSize("weights-poc8m_w2048_tf2t3_pruned_v5k_np2.c5");
      header.decomp_input_size = (int)getFileSize(output_path.c_str());
      const char* const teile[] = {".decomp_bin",
                                   "weights-poc8m_w2048_tf2t3_pruned_v5k_np2.c5",
                                   output_path.c_str(), nullptr};
      if (!cat(temp_path.c_str(), teile, &header, sizeof(header)) ||
          rename(temp_path.c_str(), "archive9") != 0)
        ENC_FAIL;
    }
    // make the decompressor binary executable
    chmod("archive9", 0777);

    selfx_clean::AfterCompress();

#undef ENC_FAIL
  }
  else
  {
#if !(1 && !AUX_BUILD)
    if (argc == 5) {
      if (!provide_dict_file(argv[2])) return Help();
    }
#endif
    if (!RunDecompression(input_path, temp_path, output_path, dictionary,
        &input_bytes, &output_bytes)) {
      return Help();
    }
  }

print_end_message:
#if !AUX_BUILD
  {
    const double progress_s = console_progress::End(progress_encoding);
    printf("%lld bytes -> %lld bytes in %1.2f s.\n", input_bytes, output_bytes,
           progress_s);
  }
#else
  printf("\r%lld bytes -> %lld bytes in %1.2f s.\n",
      input_bytes, output_bytes,
      ((double)clock() - start) / CLOCKS_PER_SEC);
#endif

  return 0;
}
