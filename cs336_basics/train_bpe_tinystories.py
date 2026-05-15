import argparse
import json

from cs336_basics.train_bpe import train_bpe
from cs336_basics.bpe import gpt2_bytes_to_unicode


def main():
    parser = argparse.ArgumentParser(description="Train a BPE Tokenizer")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the training data text file")
    parser.add_argument(
        "--vocab_output_path", type=str, default="data/tinystories_vocab.json", help="Where to save the vocab JSON"
    )
    parser.add_argument(
        "--merges_output_path",
        type=str,
        default="data/tinystories_merges.txt",
        help="Where to save the merges text file",
    )

    args = parser.parse_args()

    print(f"Starting BPE training on: {args.input_path}")

    vocab, merges = train_bpe(
        input_path=args.input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )

    # Serialize the resulting vocabulary and merges to disk
    print(f"Training complete! Saving {args.vocab_output_path} and {args.merges_output_path} ...")

    byte_to_unicode = gpt2_bytes_to_unicode()
    with open(args.vocab_output_path, "w") as f:
        reversed_vocab = {
            "".join([byte_to_unicode[token] for token in tokens]): token_id for token_id, tokens in vocab.items()
        }

        json.dump(reversed_vocab, f, indent=2, ensure_ascii=False)

    with open(args.merges_output_path, "w") as f:
        for first_token, second_token in merges:
            first_str = "".join([byte_to_unicode[b] for b in first_token])
            second_str = "".join([byte_to_unicode[b] for b in second_token])
            f.write(f"{first_str} {second_str}\n")


if __name__ == "__main__":
    main()
