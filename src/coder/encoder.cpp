#include "../invariants.h"
#include "encoder.h"
#ifdef NEIF_TEST
#include "../neif_test.h"   // chunk evaluation: bit costs
#endif
__attribute__((cold)) Encoder::Encoder(FILE* os, Predictor* p) : os_(os), x1_(0),
    x2_(0xffffffff), p_(p) {
  // Register the per-byte channel BEFORE Init(): Init() builds the
  // head and sizes its projections from these widths.  Every pointer is stable
  // for the whole run, so this is the only registration there is: no setter
  // call per coded bit, across a translation-unit boundary, into a unit built
  // without -flto.  Encoder and Decoder register the identical things, which
  // is what keeps the head in lockstep.
  bit_head::SetAux(p_->bitl_hid_, p_->bitl_n_hid_);
  bit_head::SetTf(p_->bitl_tf_, p_->bitl_n_tf_);
  bit_head::SetZone(p_->bitl_zone_);
  bit_head::Init();
}

// cold: once per coded byte, off the bit path; the PGO profile has no record
// of this form of the function.
__attribute__((cold)) void Encoder::WriteByte(unsigned int byte) {
  out_[out_n_++] = (char)byte;
  if (out_n_ == sizeof out_) {
    fwrite(out_, 1, out_n_, os_);
    total_out_ += out_n_;
    out_n_ = 0;
  }
}

unsigned int Encoder::Discretize(float p) {
  return 1 + 65534 * p;
}

void Encoder::Encode(int bit) {
  if (p_->ChrForced()) {
    p_->Predict();
    p_->Perceive(bit);
    return;
  }
#ifdef NEIF_TEST
  // chunk evaluation, second cost sum: the same statement as below with the
  // return value of Predict() kept, so that its cost can be summed in front
  // of BITLSTM, the floor stage and the semantic-cache correction
  const float neif_tap_p = p_->Predict();
  neif_test::CostTap(neif_tap_p, bit);
  unsigned int p = Discretize(neif_tap_p);
#else
  unsigned int p = Discretize(p_->Predict());
#endif
  int bb_bitl = 0;
  if (bit_head::Enabled()
      && !bit_head::AuxOff()
      && !bit_head::MetaOff()
      && !p_->AdepthSkipped()
      ) {
#ifdef NEIF_TEST
    neif_test::T(11);   // profile mark: BITLSTM
#endif
    p = bit_head::Adjust(p, p_->bitl_stage1_, p_->bitl_n_stage1_,
                          p_->bitl_m1raw_, p_->bitl_override_);
    bb_bitl = p_->bitl_override_ ? 0 : 1;
  }
#ifdef NEIF_TEST
  neif_test::T(12);   // profile mark: floor stage, end stages
  // chunk evaluation: the same call sequence as below, with the final 32-bit
  // probability kept in a variable so that its cost can be summed
  const uint32_t neif_p32 = p_->SemCorrect(floor_.P32(p,
      p_->bitl_override_,
      p_->AdepthSkipped() ? 1 : 0,
      bb_bitl, p_->bb_z_, p_->bitl_m1raw_, bit_head::MetaOff() ? 1 : 0,
      (31 - __builtin_clz(p_->BbBitCtx() | 1u)) & 7));
  neif_test::Cost(neif_p32, bit);
  const unsigned int xmid = floor_stage::Mid(x1_, x2_, neif_p32);
#else
  const unsigned int xmid = floor_stage::Mid(x1_, x2_, p_->SemCorrect(floor_.P32(p,
      p_->bitl_override_,
      p_->AdepthSkipped() ? 1 : 0,
      bb_bitl, p_->bb_z_, p_->bitl_m1raw_, bit_head::MetaOff() ? 1 : 0,
      (31 - __builtin_clz(p_->BbBitCtx() | 1u)) & 7)));
#endif
  if (bit) {
    x2_ = xmid;
  } else {
    x1_ = xmid + 1;
  }
  floor_.Update(bit);
#ifdef NEIF_TEST
  neif_test::T(13);   // profile mark: end stages learn
#endif
  p_->SemLearn(bit);
#ifdef NEIF_TEST
#ifdef NEIF_PAR
  neif_test::T(16);   // NEIF_PAR: BITLSTM behind the coded bit, apart from its part in front (section 11)
#else
  neif_test::T(11);   // profile mark: BITLSTM
#endif
#endif
  if (bit_head::Enabled()
      && !bit_head::AuxOff()
      && !bit_head::MetaOff()
      && !p_->AdepthSkipped()
      ) bit_head::Observe(bit);
  p_->Perceive(bit);

  while (((x1_^x2_) & 0xff000000) == 0) {
    WriteByte(x2_ >> 24);
    x1_ <<= 8;
    x2_ = (x2_ << 8) + 255;
  }
}

void Encoder::EncodeRawBit(int bit, unsigned int p) {
  if (p < 1) p = 1;
  if (p > 65534) p = 65534;
  const unsigned int xmid = x1_ + ((x2_ - x1_) >> 16) * p +
      (((x2_ - x1_) & 0xffff) * p >> 16);
  if (bit) {
    x2_ = xmid;
  } else {
    x1_ = xmid + 1;
  }

  while (((x1_^x2_) & 0xff000000) == 0) {
    WriteByte(x2_ >> 24);
    x1_ <<= 8;
    x2_ = (x2_ << 8) + 255;
  }
}

void Encoder::ObserveKnownByte(unsigned int byte) {
  invariants::g_injected_byte = 1;
  for (int bit = 7; bit >= 0; --bit) {
    ObserveKnownBit((byte >> bit) & 1);
  }
  invariants::g_injected_byte = 0;
}

void Encoder::BeginTraceByte(unsigned long long offset, unsigned int actual_byte,
    unsigned int prev4) {
  (void)offset;
  (void)actual_byte;
  (void)prev4;
}

void Encoder::EndTraceByte() {}

__attribute__((cold)) void Encoder::Flush() {   // cold: once per run, no profile record
  while (((x1_^x2_) & 0xff000000) == 0) {
    WriteByte(x2_ >> 24);
    x1_ <<= 8;
    x2_ = (x2_ << 8) + 255;
  }
  WriteByte(x2_ >> 24);

  fwrite(out_, 1, out_n_, os_);
  total_out_ += out_n_;
  out_n_ = 0;
}
