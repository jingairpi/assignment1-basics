import itertools
import os
import regex as re

from collections import defaultdict
from multiprocessing import Pool
from typing import Final

from cs336_basics.pretokenization_example import find_chunk_boundaries

# From github.com/openai/tiktoken/pull/234/files
TIKTOKEN_PATTERN: Final = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []
    special_pattern = f"({'|'.join(re.escape(t) for t in special_tokens)})" if special_tokens else ""
    special_tokens_set = set(special_tokens)

    # 1: Vocab initialization

    ## Initialize the default vocab (ASCII)
    for i in range(256):
        vocab[i] = bytes([i])
    initial_vocab_size = len(vocab)

    ## Append the special tokens to the vocab
    for index, special_token in enumerate(special_tokens):
        vocab[initial_vocab_size + index] = special_token.encode("utf-8")

    # 2: Pre-tokenization in parallel

    pre_token_freqs: dict[tuple[int, ...], int] = {}
    with open(input_path, "rb") as f:
        num_processes = os.cpu_count()
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunks.append(f.read(end - start).decode("utf-8", errors="ignore"))

        args = [(chunk, special_pattern, special_tokens_set) for chunk in chunks]
        with Pool(processes=num_processes) as pool:
            chunk_pre_token_freqs = pool.starmap(_count_pre_tokens, args)

        for sharded_freq_table in chunk_pre_token_freqs:
            for pre_token, count in sharded_freq_table.items():
                pre_token_freqs[pre_token] = pre_token_freqs.get(pre_token, 0) + count

    # 3: Merges

    pair_counts, pair_to_pretokens = _build_pair_index(pre_token_freqs)
    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        # Lexicographically greater pair wins
        best_pair, _ = max(pair_counts.items(), key=lambda item: (item[1], vocab[item[0][0]], vocab[item[0][1]]))

        first_token = vocab[best_pair[0]]
        second_token = vocab[best_pair[1]]
        new_token = first_token + second_token
        merges.append((first_token, second_token))
        new_token_id = len(vocab)
        vocab[new_token_id] = new_token

        _apply_merge_to_freqs(pre_token_freqs, best_pair, new_token_id, pair_counts, pair_to_pretokens)

    return (vocab, merges)


def _count_pre_tokens(chunk: str, special_pattern: str, special_tokens_set: set[str]) -> dict[tuple[int, ...], int]:
    pre_token_freqs: dict[tuple[int, ...], int] = {}
    if special_pattern:
        split_segments = re.split(special_pattern, chunk)
    else:
        split_segments = [chunk]

    for part in split_segments:
        if not part:
            continue

        # Skip special tokens so they don't get shredded by the regex
        if part in special_tokens_set:
            continue

        for pre_token_str in TIKTOKEN_PATTERN.findall(part):
            pre_token_ids = tuple(pre_token_str.encode("utf-8"))
            pre_token_freqs[pre_token_ids] = pre_token_freqs.get(pre_token_ids, 0) + 1
    return pre_token_freqs


def _build_pair_index(
    pre_token_freqs: dict[tuple[int, ...], int],
) -> tuple[dict[tuple[int, int], int], defaultdict[tuple[int, int], set[tuple[int, ...]]]]:
    pair_counts: dict[tuple[int, int], int] = {}
    pair_to_pretokens: defaultdict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)

    for pre_token_ids, count in pre_token_freqs.items():
        for token_id, next_token_id in itertools.pairwise(pre_token_ids):
            pair_counts[(token_id, next_token_id)] = pair_counts.get((token_id, next_token_id), 0) + count
            pair_to_pretokens[(token_id, next_token_id)].add(pre_token_ids)
    return pair_counts, pair_to_pretokens


def _apply_merge_to_freqs(
    pre_token_freqs: dict[tuple[int, ...], int],
    best_pair: tuple[int, int],
    new_token_id: int,
    pair_counts: dict[tuple[int, int], int],
    pair_to_pretokens: defaultdict[tuple[int, int], set[tuple[int, ...]]],
) -> None:
    dirty_pairs = set()

    for pre_token_ids in list(pair_to_pretokens[best_pair]):
        count = pre_token_freqs[pre_token_ids]
        for token_id, next_token_id in itertools.pairwise(pre_token_ids):
            key = (token_id, next_token_id)
            pair_counts[key] -= count
            pair_to_pretokens[key].discard(pre_token_ids)
            dirty_pairs.add(key)

        pre_token_list = []
        i = 0

        while i < len(pre_token_ids):
            if i < len(pre_token_ids) - 1 and pre_token_ids[i] == best_pair[0] and pre_token_ids[i + 1] == best_pair[1]:
                pre_token_list.append(new_token_id)
                i += 2
            else:
                pre_token_list.append(pre_token_ids[i])
                i += 1

        pre_token_tuple = tuple(pre_token_list)
        for i in range(len(pre_token_list) - 1):
            key = (pre_token_list[i], pre_token_list[i + 1])
            pair_counts[key] = pair_counts.get(key, 0) + count
            pair_to_pretokens[key].add(pre_token_tuple)

        del pre_token_freqs[pre_token_ids]
        pre_token_freqs[pre_token_tuple] = pre_token_freqs.get(pre_token_tuple, 0) + count

    for key in dirty_pairs:
        if key in pair_counts and pair_counts[key] <= 0:
            del pair_counts[key]
