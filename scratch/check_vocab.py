import json

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

byte_to_unicode_map = bytes_to_unicode()
unicode_to_byte_map = {v: b for b, v in byte_to_unicode_map.items()}

with open('tests/fixtures/gpt2_vocab.json', 'r') as f:
    vocab = json.load(f)

# Try to decode some tokens
sample_tokens = ["!", "\u0100", "\u0120t"]
for t in sample_tokens:
    if t in vocab:
        token_id = vocab[t]
        try:
            # Convert string to bytes using the mapping
            b_val = bytes([unicode_to_byte_map[char] for char in t])
            print(f"Token: {t!r}, ID: {token_id}, Bytes: {b_val!r}")
        except KeyError as e:
            print(f"Token: {t!r}, ID: {token_id}, Error: {e}")
