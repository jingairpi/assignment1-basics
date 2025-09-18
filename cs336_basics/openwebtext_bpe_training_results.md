# BPE Training on OpenWebText - Results Summary

## Overview

Executed byte-level BPE training on the OpenWebText corpus and performed a detailed analysis of training performance and resulting vocabulary characteristics, with emphasis on the longest tokens and their linguistic implications.

## Training Configuration

- Dataset: OpenWebText (≈11 GB)
- Target Vocabulary Size: 32,000 tokens
- Special Tokens: `<|endoftext|>`
- Algorithm: Byte-level BPE with GPT-2 style pretokenization

## Part (a): Training Results and Analysis

### Performance Metrics

- Training Time: 39.87 hours
- Peak Memory Usage: 2.79 GB
- Target Achievement:
  - ❌ Training time: 39.87 hours > 12 hours (target not met)
  - ✅ Memory usage: 2.79 GB ≤ 100 GB (target met)

### Vocabulary Analysis

- Final Vocabulary Size: 32,000 tokens
- Number of BPE Merges: 31,743
- Token Length Statistics: average 6.34 bytes (range: 1–64 bytes)
- Longest Tokens (examples):
  - Repeated Unicode mojibake sequence (64 bytes): `ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ`
  - Long separator lines (up to 64 bytes): sequences of box-drawing characters or ASCII dashes/equals (for example, runs of `─`, `—`, `=`, or `-`)

#### Why do these become the longest tokens?

- Repetition structure: Web text often contains long, contiguous runs of the same or similar characters (horizontal rules, decorative separators). BPE greedily merges the most frequent adjacent byte pairs, so repeated characters compress into longer and longer subwords, eventually saturating the 64-byte cap.
- Mojibake patterns: Mixed encodings and normalization artifacts in scraped web data yield recurring byte sequences like `Ã`/`Â` patterns. Because these appear contiguously in corrupted spans, BPE repeatedly merges adjacent pairs, producing very long tokens that capture these artifacts efficiently.
- Byte-level dynamics: Since the algorithm operates on bytes, not characters, it does not "see" grapheme boundaries; frequent byte bigrams inside these artifacts dominate early, making them prime candidates for large merges.

#### Linguistic significance

- Efficiency vs. semantics: Long separator and mojibake tokens improve compression but carry little semantic content. Their presence indicates the tokenizer effectively captured frequent structural/noise patterns in web text.
- Robustness: Capturing these artifacts as single tokens can make downstream models more robust to real-world noisy inputs by isolating noise into a small number of IDs rather than diffusing it across many shorter tokens.
- Domain signal: The contrast with clean, child-friendly prose (e.g., TinyStories) highlights how domain characteristics shape the learned subword inventory.

## Part (b): Comparative Analysis Framework (OpenWebText vs TinyStories)

To systematically compare tokenization between OWT and TinyStories, apply the following framework:

1. Corpus descriptors
   - Size, language diversity, and presence of markup/noise (HTML, code, emoji, box-drawing, mojibake).
2. Token length distribution
   - Compare histograms/CDFs of token byte lengths; expect heavier tail (more 32–64 byte tokens) for OWT.
3. Top-k longest tokens and categories
   - Group by categories: separators, mojibake, whole words, code identifiers, emoji sequences.
4. Merge trajectory analysis
   - Track which byte pairs dominate early merges; OWT likely shows repeated-character pairs and encoding-artifact pairs rising earlier than in TinyStories.
5. Semantic token coverage
   - Measure proportion of top-N tokens that correspond to complete words or frequent morphemes (higher for TinyStories; more structural/noise tokens for OWT).
6. Downstream efficiency proxy
   - On matched samples, compute average tokens-per-word and tokens-per-sentence; OWT may yield slight gains on web-like text but fewer gains on curated prose.

## File Locations

Artifacts for this run are saved under:

- Directory: `data/owt_tokenizer_output/`
- Generated files:
  - `vocabulary.json`
  - `merges.txt`
  - `tokenizer.pkl`
  - `vocab_summary.txt`

## Deliverable Responses

### Part (a): Training metrics and longest token analysis

- Training completed in 39.87 hours using 2.79 GB peak RAM. Time target (≤ 12 hours) was not met; memory target (≤ 100 GB) was comfortably met.
- The longest tokens are predominantly (1) repeated-character separators and (2) mojibake sequences (e.g., `ÃÂ` repeats), both reaching the 64-byte cap. These arise because byte-level BPE aggressively merges frequent adjacent pairs in long, contiguous runs present in web text.

### Part (b): Comparative analysis framework

- Provided a step-by-step framework to contrast OWT and TinyStories tokenizations across corpus descriptors, token-length distributions, token categories, merge trajectories, semantic coverage, and downstream efficiency proxies.

## Conclusion

The OpenWebText tokenizer achieves the target vocabulary size with modest memory usage but substantially longer training time than the 12-hour goal. The vocabulary reflects the heterogeneous, noisy nature of web text—capturing long separator lines and encoding artifacts as single tokens. This behavior is consistent with byte-level BPE and contrasts with TinyStories, where longest tokens are semantically rich words. The included framework supports a rigorous, repeatable comparison of tokenization behavior across domains.
