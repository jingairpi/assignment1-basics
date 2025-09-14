import os
import multiprocessing
import regex as re
from collections import defaultdict
from collections.abc import Iterable

from cs336_basics.pretokenization_example import find_chunk_boundaries

# Regex pattern for pretokenization (matches GPT-2 style tokenization)
PRETOKENIZATION_RE = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str], **kwargs) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a BPE tokenizer on the given corpus (GPT-2 style).

    This implements modern byte-level BPE with regex pretokenization, not naive 
    character-level BPE. The algorithm:
    1. Pretokenizes text using GPT-2 style regex
    2. Initializes vocabulary with all 256 bytes + special tokens  
    3. Iteratively merges most frequent adjacent pairs within pretokenized units
    4. Never merges across pretokenization boundaries
    
    Args:
        input_path: Path to BPE tokenizer training data.
        vocab_size: Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens: List of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the input_path,
            they are treated as any other string.

    Returns:
        tuple containing:
            vocab: The trained tokenizer vocabulary, mapping token ID to token bytes
            merges: BPE merges as list of (token1_bytes, token2_bytes) tuples,
                   ordered by creation time
    """
    # Initialize vocabulary with all 256 possible bytes plus special tokens
    vocab = _initialize_vocabulary(special_tokens)
    merges = []

    # Get pretokenized word counts from corpus
    # Forward optional keyword arguments for flexibility (e.g. num_processes)
    word_counts = _get_pretokenized_word_counts(input_path, special_tokens, **kwargs)

    # Perform BPE merges until we reach target vocabulary size
    while len(vocab) < vocab_size:
        pair_counts = _count_adjacent_pairs(word_counts)

        if not pair_counts:
            break  # No more pairs to merge

        # Find most frequent pair (with lexicographic tiebreaking)
        best_pair = _find_best_pair_to_merge(pair_counts, vocab)

        # Create new token and update vocabulary
        new_token_id = _add_merged_token_to_vocab(vocab, best_pair)
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))

        # Update word counts to reflect the merge
        word_counts = _apply_merge_to_words(word_counts, best_pair, new_token_id)

    return vocab, merges


def _initialize_vocabulary(special_tokens: list[str]) -> dict[int, bytes]:
    """Initialize vocabulary with all 256 bytes plus special tokens."""
    vocab = {i: bytes([i]) for i in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")
    return vocab


def _get_pretokenized_word_counts(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int | None = None,
    split_special_token: bytes | None = None,
) -> dict[tuple[int, ...], int]:
    """Get word counts from pretokenized corpus using multiprocessing.

    Args:
        input_path: Path to the corpus.
        special_tokens: List of tokens that should not be split.
        num_processes: Number of processes to use. Defaults to available CPUs.
        split_special_token: Token used to find chunk boundaries. Defaults to the
            first special token if provided.
    """

    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    if split_special_token is None:
        if special_tokens:
            split_special_token = special_tokens[0].encode("utf-8")
        else:
            split_special_token = b"<|endoftext|>"

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_special_token)

    # Create chunk jobs
    chunk_jobs = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    # If there's only one chunk (or none), avoid multiprocessing overhead
    if len(chunk_jobs) <= 1:
        chunk_results = [
            _preprocess_chunk(input_path, start, end, special_tokens)
            for (input_path, start, end, special_tokens) in chunk_jobs
        ]
    else:
        # Cap processes to the actual number of chunks to avoid spawning idle workers
        procs = min(num_processes, len(chunk_jobs))
        with multiprocessing.Pool(processes=procs) as pool:
            chunk_results = pool.starmap(_preprocess_chunk, chunk_jobs)

    # Merge results from all chunks
    return _merge_word_counts(chunk_results)


def _preprocess_chunk(input_path: str | os.PathLike, start: int, end: int, special_tokens: list[str]) -> dict[tuple[int, ...], int]:
    """Preprocess a chunk of the input data, returning word counts as tuples of byte values.

    Args:
        input_path: Path to BPE tokenizer training data.
        start: Start position of the chunk.
        end: End position of the chunk.
        special_tokens: List of special tokens that should not be split.

    Returns:
        Dictionary mapping word tuples (as byte sequences) to their counts in the chunk.
    """
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)
        chunk = chunk.decode("utf-8", errors="ignore")

        # Split on special tokens to avoid tokenizing them.
        if special_tokens:
            parts = re.split("|".join(map(re.escape, special_tokens)), chunk)
        else:
            parts = [chunk]

        word_counts = defaultdict(int)
        for part in parts:
            # Apply pretokenization pattern
            for match in PRETOKENIZATION_RE.finditer(part):
                token_bytes = match.group().encode("utf-8")
                # Convert bytes to tuple of individual byte values for consistency
                word_tuple = tuple(token_bytes)
                word_counts[word_tuple] += 1

    return dict(word_counts)


def _merge_word_counts(chunk_results: Iterable[dict[tuple[int, ...], int]]) -> dict[tuple[int, ...], int]:
    """Merge word counts from all chunks.

    Args:
        chunk_results: Iterable of dictionaries mapping word tuples to counts.

    Returns:
        Dictionary mapping word tuples to their total counts across all chunks.
    """
    merged_counts = defaultdict(int)
    for chunk_result in chunk_results:
        for word_tuple, count in chunk_result.items():
            merged_counts[word_tuple] += count
    return dict(merged_counts)


def _count_adjacent_pairs(word_counts: dict[tuple[int, ...], int]) -> dict[tuple[int, int], int]:
    """Count all adjacent token pairs in the word counts."""
    pair_counts = defaultdict(int)
    for word, count in word_counts.items():
        for a, b in zip(word, word[1:]):
            pair_counts[(a, b)] += count
    return dict(pair_counts)


def _find_best_pair_to_merge(pair_counts: dict[tuple[int, int], int], vocab: dict[int, bytes]) -> tuple[int, int]:
    """Find the most frequent pair, using lexicographic tiebreaking on byte content."""
    return max(pair_counts, key=lambda pair: (pair_counts[pair], (vocab[pair[0]], vocab[pair[1]])))


def _add_merged_token_to_vocab(vocab: dict[int, bytes], pair: tuple[int, int]) -> int:
    """Add a new merged token to the vocabulary and return its ID."""
    new_token_bytes = vocab[pair[0]] + vocab[pair[1]]
    new_token_id = len(vocab)
    vocab[new_token_id] = new_token_bytes
    return new_token_id


def _apply_merge_to_words(
    word_counts: dict[tuple[int, ...], int],
    pair_to_merge: tuple[int, int],
    new_token_id: int,
) -> dict[tuple[int, ...], int]:
    """Apply a merge operation to all words, replacing the specified pair with the new token."""
    new_word_counts = defaultdict(int)

    for word, count in word_counts.items():
        new_word = _merge_pair_in_word(word, pair_to_merge, new_token_id)
        new_word_counts[new_word] += count

    return dict(new_word_counts)


def _merge_pair_in_word(
    word: tuple[int, ...], pair_to_merge: tuple[int, int], new_token_id: int
) -> tuple[int, ...]:
    """Replace all occurrences of pair_to_merge in word with new_token_id."""
    if len(word) < 2:
        return word
    
    a, b = pair_to_merge
    result = []
    i = 0
    
    while i < len(word):
        # Check if we have a pair match at current position
        if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
            result.append(new_token_id)
            i += 2  # Skip both elements of the pair
        else:
            result.append(word[i])
            i += 1
    
    return tuple(result)
