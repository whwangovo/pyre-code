"""Byte-Pair Encoding (BPE) task."""

TASK = {
    "title": "Byte-Pair Encoding (BPE)",
    "title_zh": "字节对编码（BPE）",
    "difficulty": "Hard",
    "description_en": "Implement Byte-Pair Encoding (BPE) tokenization.\n\nBPE iteratively merges the most frequent adjacent symbol pairs to build a subword vocabulary, widely used in modern language models.\n\n**Signature:** `SimpleBPE()` (class)\n\n**Methods:**\n- `train(corpus: list[str], num_merges: int)` — learn merge rules from a word list\n- `encode(text: str) -> list[str]` — tokenize text into subword tokens\n\n**Constraints:**\n- Append `</w>` to mark word boundaries\n- `self.merges` stores learned pairs in order\n- `encode` applies merges sequentially",
    "description_zh": "实现字节对编码（BPE）分词。\n\nBPE 通过迭代合并最频繁的相邻符号对来构建子词词表，广泛用于现代语言模型。\n\n**签名:** `SimpleBPE()`（类）\n\n**方法:**\n- `train(corpus: list[str], num_merges: int)` — 从词列表中学习合并规则\n- `encode(text: str) -> list[str]` — 将文本分词为子词 token\n\n**约束:**\n- 使用 `</w>` 标记词边界\n- `self.merges` 按顺序存储学习到的合并对\n- `encode` 按顺序应用合并规则",
    "function_name": "SimpleBPE",
    "hint": "train: split each word → chars + `</w>`, count adjacent pairs, merge most frequent, repeat\nencode: apply learned merges in order to split text into subword tokens",
    "hint_zh": "train：每个词拆分为字符 + `</w>`，统计相邻对频率，合并最高频对，重复\nencode：按学习到的合并顺序将文本拆分为子词 token",
    "tests": [
        {
            "name": "Correct number of merges",
            "code": "\nbpe = {fn}()\nbpe.train(['low', 'low', 'low', 'lower', 'newest', 'widest'], num_merges=5)\nassert len(bpe.merges) == 5, f'Expected 5 merges, got {len(bpe.merges)}'\n"
        },
        {
            "name": "Most frequent pair merged first",
            "code": "\nbpe = {fn}()\nbpe.train(['aaa', 'aaa', 'aaa', 'bbb'], num_merges=1)\nassert bpe.merges[0] == ('a', 'a'), f'First merge: {bpe.merges[0]}'\n"
        },
        {
            "name": "Encode returns list of strings",
            "code": "\nbpe = {fn}()\nbpe.train(['low', 'lower', 'lowest'] * 3, num_merges=10)\ntokens = bpe.encode('low')\nassert isinstance(tokens, list), 'encode must return a list'\nassert all(isinstance(t, str) for t in tokens), 'tokens must be strings'\nreconstructed = ''.join(t.replace('</w>', '') for t in tokens)\nassert reconstructed == 'low', f'Reconstruction: {reconstructed}'\n"
        },
        {
            "name": "More merges -> fewer tokens",
            "code": "\nbpe1 = {fn}()\nbpe1.train(['hello'] * 10, num_merges=2)\nbpe2 = {fn}()\nbpe2.train(['hello'] * 10, num_merges=10)\nassert len(bpe2.encode('hello')) <= len(bpe1.encode('hello')), 'More merges should reduce tokens'\n"
        },
        {
            "name": "Encode compresses known corpus",
            "code": "\n# Corpus where 'aa' is the dominant pair; after 3 merges 'aab' should encode to fewer tokens than chars\nbpe = {fn}()\nbpe.train(['aaa', 'aaa', 'aaa', 'aab', 'aab'], num_merges=3)\nencoded = bpe.encode('aab')\nassert isinstance(encoded, list), 'encode should return a list'\nassert len(encoded) > 0, 'encode should return non-empty list'\n# 'aa' must be a learned merge (most frequent pair), so 'aab' -> ['aa', 'b', '</w>'] or similar\nassert len(encoded) < len('aab') + 1, f'BPE should compress: \"aab\" encoded to {encoded} ({len(encoded)} tokens) but expected fewer than {len(\"aab\") + 1}'\n"
        },
        {
            "name": "Encode output reconstructs original word",
            "code": "\nbpe = {fn}()\nbpe.train(['lowest', 'newer', 'wider'] * 5, num_merges=6)\nfor word in ['low', 'new', 'wide']:\n    tokens = bpe.encode(word)\n    reconstructed = ''.join(t.replace('</w>', '') for t in tokens)\n    assert reconstructed == word, f'Reconstruction failed: {word!r} -> {tokens} -> {reconstructed!r}'\n"
        }
    ],
    "solution": '''class SimpleBPE:
    def __init__(self):
        self.merges = []

    def train(self, corpus, num_merges):
        vocab = {}
        for word in corpus:
            symbols = tuple(word) + ('</w>',)
            vocab[symbols] = vocab.get(symbols, 0) + 1
        self.merges = []
        for _ in range(num_merges):
            pairs = {}
            for word, freq in vocab.items():
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    pairs[pair] = pairs.get(pair, 0) + freq
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.merges.append(best)
            new_vocab = {}
            for word, freq in vocab.items():
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                        new_word.append(word[i] + word[i + 1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_vocab[tuple(new_word)] = freq
            vocab = new_vocab

    def encode(self, text):
        all_tokens = []
        for word in text.split():
            symbols = list(word) + ['</w>']
            for a, b in self.merges:
                i = 0
                while i < len(symbols) - 1:
                    if symbols[i] == a and symbols[i + 1] == b:
                        symbols = symbols[:i] + [a + b] + symbols[i + 2:]
                    else:
                        i += 1
            all_tokens.extend(symbols)
        return all_tokens''',
    "demo": """bpe = SimpleBPE()
bpe.train(['low', 'low', 'low', 'lower', 'newest', 'widest'], num_merges=10)
print('Merges:', bpe.merges)
print('Encode:', bpe.encode('low lower newest'))""",

}