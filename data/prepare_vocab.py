import os
import json
import pickle
import argparse


class SimpleVocab:
    """Simple vocab object compatible with src.dataset.Vocabulary for pickling."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, w):
        if w not in self.word2idx:
            self.word2idx[w] = self.idx
            self.idx2word[self.idx] = w
            self.idx += 1

    def __call__(self, word):
        return self.word2idx.get(word, self.word2idx.get('<unk>', 0))

    def __len__(self):
        return len(self.word2idx)


def build_pickle_from_json(vocab_json_path, out_pkl_path):
    if not os.path.exists(vocab_json_path):
        raise FileNotFoundError(f"vocab json not found: {vocab_json_path}")

    with open(vocab_json_path, 'r', encoding='utf-8') as f:
        j = json.load(f)

    # Expecting structure with "word2idx" and optionally "idx2word"
    word2idx = j.get("word2idx", {})
    idx2word = j.get("idx2word", {})


    vocab = SimpleVocab()

    # Fill maps (word2idx keys are words; values may be ints or strings)
    for w, idx in word2idx.items():
        try:
            i = int(idx)
        except Exception:
            i = idx
        vocab.word2idx[w] = i
        vocab.idx2word[int(i)] = w

    # map possible alternate tokens
    if '<sos>' in vocab.word2idx and '<start>' not in vocab.word2idx:
        vocab.word2idx['<start>'] = vocab.word2idx['<sos>']
    if '<eos>' in vocab.word2idx and '<end>' not in vocab.word2idx:
        vocab.word2idx['<end>'] = vocab.word2idx['<eos>']

    if len(vocab.idx2word) > 0:
        vocab.idx = max(vocab.idx2word.keys()) + 1
    else:
        vocab.idx = 0

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_pkl_path), exist_ok=True)
    with open(out_pkl_path, 'wb') as f:
        pickle.dump(vocab, f)

    print(f"Saved pickle vocab to: {out_pkl_path}")
    print(f"Vocab size: {len(vocab)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_json", type=str, default=os.path.join(os.path.dirname(__file__), "vocab.json"),
                        help="Path to data/vocab.json")
    parser.add_argument("--out_pkl", type=str, default=os.path.join(os.path.dirname(__file__), "vocab.pkl"),
                        help="Output pickle path (data/vocab.pkl)")
    args = parser.parse_args()

    build_pickle_from_json(args.vocab_json, args.out_pkl)


if __name__ == "__main__":
    main()


