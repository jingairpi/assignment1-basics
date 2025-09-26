import json
import regex as re

from collections.abc import Iterable
from typing import Iterator

from cs336_basics.train_bpe import PRETOKENIZATION_RE


class Tokenizer:
    """Byte-pair encoding (BPE) tokenizer that mirrors the GPT-2 training pipeline.

    The runtime pipeline mirrors :func:`cs336_basics.train_bpe.train_bpe`:
    1. Split around registered special tokens so they remain intact.
    2. Apply GPT-2 style regex pretokenization to the remaining spans.
    3. Convert bytes to initial token ids from the base vocabulary.
    4. Iteratively merge adjacent ids using the learned BPE ranks.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: Iterable[tuple[bytes, bytes]],
        special_tokens: Iterable[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.byte_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}
        self.special_tokens = tuple(special_tokens or [])
        self.special_token_set = set(self.special_tokens)
        self._special_split_pattern = self._compile_special_split_pattern()

        # Cache merge metadata to mirror the ranks produced during training.
        self._merge_ranks: dict[tuple[int, int], int] = {}
        self._merged_pair_to_id: dict[tuple[int, int], int] = {}
        self._initialize_merges(merges)

        # Precompute base byte -> token id lookups for faster pretokenization handling.
        self._byte_to_token_id = [self._token_id_for_bytes(bytes([value])) for value in range(256)]

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Iterable[str] | None = None,
    ):
        """Load vocabulary and merges from disk using the GPT-2 serialization format."""
        with open(vocab_filepath) as vocab_f:
            vocab_json = json.load(vocab_f)
            vocab = {int(k): v.encode("utf-8") if isinstance(v, str) else v for k, v in vocab_json.items()}

        with open(merges_filepath) as merges_f:
            merges: list[tuple[bytes, bytes]] = []
            for line in merges_f:
                parts = line.rstrip().split(" ")
                if len(parts) == 2:
                    token1_bytes = parts[0].encode("utf-8")
                    token2_bytes = parts[1].encode("utf-8")
                    merges.append((token1_bytes, token2_bytes))

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode ``text`` into token ids following the same stages as training.

        Args:
            text: The string to tokenize.

        Returns:
            list[int]: Token ids representing ``text``.
        """
        if not text:
            return []

        encoded_ids: list[int] = []
        # Step 1: split on special tokens so they remain intact, mirroring training's handling.
        parts = self._special_split_pattern.split(text) if self._special_split_pattern else [text]

        for part in parts:
            if not part:
                continue

            if part in self.special_token_set:
                encoded_ids.append(self._token_id_for_bytes(part.encode("utf-8")))
                continue

            # Step 2: apply regex pretokenization within regular text spans.
            encoded_ids.extend(self._encode_regular_text(part))

        return encoded_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Yield token ids lazily for each string in ``iterable``.

        Args:
            iterable: Stream of strings to tokenize.

        Yields:
            int: Next token id in sequence order.
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to a UTF-8 string.

        Args:
            ids: Sequence of token ids to detokenize.

        Returns:
            str: The decoded string.
        """
        decoded_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return decoded_bytes.decode("utf-8", errors="replace")

    def _initialize_merges(self, merges: Iterable[tuple[bytes, bytes]]) -> None:
        """Populate merge rank tables so encoding can pick the lowest-rank merge first.

        Args:
            merges: Ordered merge list produced during training.
        """

        for rank, (left_bytes, right_bytes) in enumerate(merges):
            left_id = self._token_id_for_bytes(left_bytes)
            right_id = self._token_id_for_bytes(right_bytes)
            merged_bytes = left_bytes + right_bytes
            merged_id = self.byte_to_id.get(merged_bytes)
            if merged_id is None:
                raise ValueError(f"Merged token {merged_bytes!r} missing from vocabulary")

            pair_to_merge = (left_id, right_id)
            self._merge_ranks[pair_to_merge] = rank
            self._merged_pair_to_id[pair_to_merge] = merged_id

    def _compile_special_split_pattern(self):
        """Compile a regex that keeps special tokens intact during pretokenization.

        Returns:
            re.Pattern | None: Pattern used to split around special tokens.
        """
        if not self.special_tokens:
            return None

        tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "(" + "|".join(map(re.escape, tokens_sorted)) + ")"
        return re.compile(pattern)

    def _encode_regular_text(self, text: str) -> list[int]:
        """Convert text without special tokens into token ids using regex pretokenization.

        Args:
            text: Text span without special tokens.

        Returns:
            list[int]: Token ids for the span.
        """
        encoded: list[int] = []
        for match in PRETOKENIZATION_RE.finditer(text):
            token_bytes = match.group().encode("utf-8")
            initial_ids = [self._byte_to_token_id[byte] for byte in token_bytes]
            # Step 3: run BPE merges on the byte-level token ids.
            encoded.extend(self._apply_bpe_merges(initial_ids))
        return encoded

    def _apply_bpe_merges(self, token_ids: list[int]) -> list[int]:
        """Apply learned BPE merges to a sequence of token ids.

        Args:
            token_ids: Pretokenized token id sequence to merge.

        Returns:
            list[int]: Token ids after applying merges.
        """
        if len(token_ids) <= 1:
            return token_ids

        tokens = list(token_ids)

        while True:
            best_rank = None
            best_index = None

            for idx in range(len(tokens) - 1):
                pair_to_merge = (tokens[idx], tokens[idx + 1])
                # Match the training loop by choosing the lowest-ranked eligible pair.
                rank = self._merge_ranks.get(pair_to_merge)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_index = idx

            if best_index is None:
                break

            pair_to_merge = (tokens[best_index], tokens[best_index + 1])
            # Replace the pair with the merged token id, just like _merge_pair_in_word.
            tokens[best_index : best_index + 2] = [self._merged_pair_to_id[pair_to_merge]]

        return tokens

    def _token_id_for_bytes(self, token_bytes: bytes) -> int:
        """Lookup the token id for ``token_bytes`` in the vocabulary.

        Args:
            token_bytes: Token byte sequence.

        Returns:
            int: Token id corresponding to ``token_bytes``.
        """
        try:
            return self.byte_to_id[token_bytes]
        except KeyError as exc:
            raise ValueError(f"Unknown token bytes: {token_bytes!r}") from exc
