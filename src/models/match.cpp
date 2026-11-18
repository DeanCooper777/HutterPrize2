#include "match.h"
#include "../page_registry.h"

Match::Match(const std::vector<unsigned char>& history,
    const unsigned long long& byte_context, const unsigned int& bit_context,
    int limit, float delta, unsigned long long map_size,
    unsigned long long* longest_match) : history_(history),
    byte_context_(byte_context), bit_context_(bit_context), history_pos_(0),
    cur_match_(0), cur_byte_(0), bit_pos_(128), match_length_(0),
    longest_match_(longest_match), limit_(limit),delta_(delta),
    divisor_(1.0 / (limit + delta)), map_(map_size, 0) {
  map_div_.Init(map_.size());
  hist_div_.Init(history_.size());
  for (int i = 0; i < 256; ++i) {
    predictions_[i] = 0.5 + (i + 0.5) / 512;
  }
  counts_.fill(0);
  PAGE_REGISTRY_ADD(map_.data(), map_.size() * sizeof(map_[0]));
}

const std::valarray<float>& Match::Predict() const {
  if (cur_byte_ & bit_pos_) outputs_[0] = predictions_[match_length_];
  else outputs_[0] = 1 - predictions_[match_length_];
  return outputs_;
}

void Match::Perceive(int bit) {
  int match = 0;
  if (bit == ((cur_byte_ & bit_pos_) != 0)) match = 1;
  bit_pos_ /= 2;

  float divisor = divisor_;
  if (counts_[match_length_] < limit_) {
    ++counts_[match_length_];
    divisor = 1.0 / (counts_[match_length_] + delta_);
  }
  predictions_[match_length_] +=
      (match - predictions_[match_length_]) * divisor;

  if (match) {
    if (match_length_ < 255) ++match_length_;
  } else {
    match_length_ = 0;
  }

  if (bit_context_ >= 128) {
    map_[map_div_.Mod(byte_context_)] = history_pos_;
    ++history_pos_;
  }
}

void Match::ByteUpdate() {
  if (match_length_ < 8) {
    cur_match_ = map_[map_div_.Mod(byte_context_)];
  } else {
    ++cur_match_;
  }
  cur_match_ = hist_div_.Mod(cur_match_);
  cur_byte_ = history_.at(cur_match_);
  bit_pos_ = 128;

  unsigned long long match_context = match_length_ / 32;
  *longest_match_ = std::max(*longest_match_, match_context);
}


#if defined(NEIF_TEST) && defined(NEIF_MT)
// NEIF_MT (match.h): the state behind the byte g, without writing.  Follows
// Perceive() bit by bit (a matching bit lengthens the match, another one ends
// it), then ByteUpdate(): a match shorter than 8 bits is looked up anew under
// the next context, where the entry Perceive() will write on the 8th bit
// (under the present context) is taken into account, a longer one moves on by
// one byte; the byte at the match position is g itself if that is the
// position g will be written to.
void Match::Peek(int g, unsigned int order, unsigned int hash_bits, unsigned long long mgr_hist_pos,
                 unsigned char* byte, unsigned char* len) const {
  unsigned char l = match_length_;
  for (int j = 7; j >= 0; --j) {
    if (((g >> j) & 1) == ((cur_byte_ >> j) & 1)) {
      if (l < 255) ++l;
    } else {
      l = 0;
    }
  }
  unsigned long long cm;
  if (l < 8) {
    const unsigned long long next =
        order ? ((byte_context_ * (1ull << hash_bits) + (unsigned long long)(g & 255)) &
                 (((unsigned long long)1 << (hash_bits * order)) - 1))
              : 0;
    const unsigned long long k2 = map_div_.Mod(next), k1 = map_div_.Mod(byte_context_);
    cm = k2 == k1 ? (unsigned long long)(unsigned int)history_pos_ : (unsigned long long)map_[k2];
  } else {
    cm = cur_match_ + 1;
  }
  cm = hist_div_.Mod(cm);
  *byte = cm == mgr_hist_pos ? (unsigned char)g : history_[cm];
  *len = l;
}
#endif
