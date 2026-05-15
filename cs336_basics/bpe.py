import itertools
import json
import re

from collections.abc import Iterable, Iterator
from functools import lru_cache

from cs336_basics.train_bpe import TIKTOKEN_PATTERN


class BPETokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.token_to_id = {token: token_id for token_id, token in self.vocab.items()}

        self.merges = merges
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

        if special_tokens:
            sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
            self.special_pattern = f"({'|'.join(re.escape(t) for t in sorted_special_tokens)})"
            self.special_tokens_set = set(special_tokens)
        else:
            self.special_pattern = ""
            self.special_tokens_set = set()

    # Port from get_tokenizer_from_vocab_merges_path() in tests/test_tokenizer.py
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_filepath) as vocab_f:
            gpt2_vocab = json.load(vocab_f)
        gpt2_bpe_merges = []

        with open(merges_filepath) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))

        # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
        # just return the original bytes, so we don't force students to use
        # any particular encoding scheme.
        vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }
        # If any of the special tokens don't exist in the vocab, append them to the vocab.
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        encoded_text: list[int] = []

        # Pre-tokenize
        if self.special_pattern:
            split_segments = re.split(self.special_pattern, text)
        else:
            split_segments = [text]

        for split_segment in split_segments:
            if not split_segment:
                continue

            if split_segment in self.special_tokens_set:
                token_id = self.token_to_id[split_segment.encode("utf-8")]
                encoded_text.append(token_id)
                continue

            for pre_token_str in TIKTOKEN_PATTERN.findall(split_segment):
                pre_tokens = [bytes([b]) for b in pre_token_str.encode("utf-8")]

                # Apply merges
                while True:
                    if len(pre_tokens) < 2:
                        break

                    best_pair = min(
                        itertools.pairwise(pre_tokens), key=lambda pair: self.merge_ranks.get(pair, float("inf"))
                    )

                    if self.merge_ranks.get(best_pair, float("inf")) == float("inf"):
                        break

                    new_pre_tokens = []
                    i = 0
                    while i < len(pre_tokens):
                        if (
                            i < len(pre_tokens) - 1
                            and pre_tokens[i] == best_pair[0]
                            and pre_tokens[i + 1] == best_pair[1]
                        ):
                            new_pre_tokens.append(pre_tokens[i] + pre_tokens[i + 1])
                            i += 2
                        else:
                            new_pre_tokens.append(pre_tokens[i])
                            i += 1

                    pre_tokens = new_pre_tokens

                encoded_text.extend([self.token_to_id[pre_token] for pre_token in pre_tokens])

        return encoded_text

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        tokens = b"".join(self.vocab[id] for id in ids)
        return tokens.decode("utf-8", errors="replace")


# Port from tests/common.py
@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d
