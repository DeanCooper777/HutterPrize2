#ifndef PPMD_H
#define PPMD_H

#include "byte-model.h"

#include <array>
#include <memory>
#include <vector>

namespace PPMD {

struct ppmd_Model;

class PPMD : public ByteModel {
 public:
  PPMD(int order, int memory, const unsigned int& bit_context,
      const std::vector<bool>& vocab);
  ~PPMD();
  std::valarray<float>& Predict();
  void Perceive(int bit);
  void ByteUpdate();
  void Waermen();
#ifdef NEIF_TEST
  void NeifSync();   // chunk evaluation: the pending restarts (src/predictor.cpp NeifSkipSync)
  void NeifRow();    // pipeline: send or check the result of this byte (src/neif_test.h)
  void NeifDisown(); // pipeline: this process no longer runs the model
#ifdef NEIF_MT
  // NEIF_MT (src/neif_mt.h): the model update of a GUESSED byte, on another
  // thread while this object serves the bits of the current byte.
  //   * a shadow wrapper is a second PPMD object on the SAME model; its
  //     ByteUpdate() is the code of this class, so its bit-tree arrays and
  //     its row are the ones this object would have computed;
  //   * SpecBegin() saves the model object and starts the undo journal of
  //     the heap (ppmd_page.h), SpecEnd() stops the journal;
  //   * if the guess was the byte, SpecKeep() and SpecAdopt(shadow) put this
  //     object where its own ByteUpdate() would have put it; if not,
  //     SpecUndo() puts model and heap back and ByteUpdate() runs as usual.
  // Only one thread is inside the model at any time (the caller's protocol).
  struct Shadow {};
  PPMD(const PPMD& real, const unsigned int& byte, Shadow);
  bool SpecSafe() const;   // no restart of the model is pending
  void SpecBegin();
  void SpecEnd();
  void SpecKeep();
  void SpecUndo();
  void SpecAdopt(const PPMD& shadow);
#endif
#endif
 private:
  const unsigned int& byte_;
  std::unique_ptr<ppmd_Model> ppmd_model_;
  std::array<unsigned int, 256> tree_zero_;
  std::array<unsigned int, 256> tree_total_;
  // the byte values outside the vocabulary, ascending; a fixed array instead
  // of a vector (no heap block, no destructor), iterated like one
  struct Bytes {
    unsigned char b[256];
    unsigned int n = 0;
    const unsigned char* begin() const { return b; }
    const unsigned char* end() const { return b + n; }
  } disabled_bytes_;
  unsigned int tree_context_ = 1;
  bool vocab_full_ = false;
};

} // namespace PPMD

#endif
