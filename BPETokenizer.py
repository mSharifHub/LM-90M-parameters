import re
from collections import defaultdict

class BPETokenizer:
    SPECIAL_TOKENS = ["<|unk|>", "<|endoftext|>"]


    def __init__(self, pattern, num_merges=1000):
        self.regex = re.compile(pattern)
        self.num_merges = num_merges
        self.merges = []
        self.token_to_id = {}
        self.id_to_token = {}

    def build_vocab(self,dataset):
        word_freqs = self._get_word_freqs(dataset)
        self.merges = self._train_bpe(word_freqs)

        base_chars = sorted({ch for word in word_freqs for ch in word})
        merged_tokens = [''.join(pair) for pair in self.merges]

        all_tokens = self.SPECIAL_TOKENS + base_chars + merged_tokens
        self.token_to_id = {tok: idx for idx, tok in enumerate(all_tokens)}
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}


    def _get_word_freqs(self,dataset):
        word_freqs = defaultdict(int)
        for row in dataset:
            for word in re.findall(self.regex, row['text']):
                word_freqs[tuple(word)] +=1
        return word_freqs

    def _get_pairs(self, word_freqs):
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            for i in range(len(word) -1):
                pairs[(word[i]), word[i + 1]] += freq
        return pairs

    def _merge_pair(self,pair, word_freqs):
        a,b, merged = pair[0], pair[1], pair[0] + pair[1]
        new_freqs = {}
        for word, freq, in word_freqs.items():
            new_word, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i +1] == b:
                    new_word.append(merged)
                    i +=2
                else:
                    new_word.append(word[i])
                    i +=1
            new_freqs[tuple(new_word)] = freq
        return new_freqs

    def _train_bpe(self, word_freqs):
        merges = []
        for _ in range(self.num_merges):
            pairs = self._get_pairs(word_freqs)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            merges.append(best)
            word_freqs = self._merge_pair(best, word_freqs)
        return merges


    def _apply_merges(self, chars):
        tokens = list(chars)
        for a, b in self.merges:
            merged, i = a + b, 0
            while i < len(tokens) - 1:
                if tokens[i] == a and tokens[i + 1] == b:
                    tokens[i] = merged
                    del tokens[i + 1]
                else:
                    i += 1
        return tokens

    def encode(self, text):
        unk_id = self.token_to_id["<|unk|>"]
        ids = []
        for word in re.findall(self.regex, text):
            for tok in self._apply_merges(list(word)):
                ids.append(self.token_to_id.get(tok, unk_id))
        return ids

    def encode_row(self, row):
        row['token_ids'] = self.encode(row['text'])
        return row

    def decode(self, ids):
        return "".join(self.id_to_token[i] for i in ids)

    @property
    def vocab_size(self):
        return len(self.token_to_id)


