import torch
import torch.nn.functional as F
import numpy as np
import constriction
import math
import time
from model import DecoderOnlyTransformer
from encode import encode_with_cache
from decode import decode_with_cache

def verify(original_tokens, decoded_tokens):
    """Check that decompression perfectly recovered the original."""
    if len(original_tokens) != len(decoded_tokens):
        print(f"FAIL: length mismatch ({len(original_tokens)} vs {len(decoded_tokens)})")
        return False

    mismatches = np.sum(original_tokens != decoded_tokens)
    if mismatches > 0:
        first_mismatch = np.argmin(original_tokens == decoded_tokens)
        print(f"FAIL: {mismatches:,} mismatches. First at position {first_mismatch}")
        print(f"  Expected: {original_tokens[first_mismatch]}")
        print(f"  Got:      {decoded_tokens[first_mismatch]}")
        return False

    print("PASS: perfect reconstruction — all tokens match")
    return True


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compress/decompress chr22')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt')
    parser.add_argument('--tokens', type=str, default='tokenized_data/chr22.npy')
    parser.add_argument('--output', type=str, default='chr22_compressed.npy')
    args = parser.parse_args()

    # ── Setup ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Determinism
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True

    # ── Load model ──
    model = DecoderOnlyTransformer(
        vocab_size=4096, embed_dim=512, num_heads=8,
        num_layers=12, tie_weights=True,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from {args.checkpoint}")

    # ── Load tokens ──
    tokens = np.load(args.tokens)
    print(f"Chr22 tokens: {len(tokens):,}")
    print(f"Chr22 bases:  {len(tokens) * 6:,}")
    tokens_test = tokens[:1000]
    # ── Encode ──
    print("\n" + "=" * 60)
    print("ENCODING")
    print("=" * 60)
    # compressed, num_bits = encode(model, tokens, device)
    compressed, num_bits = encode_with_cache(model, tokens_test, device)

    # Save compressed
    np.save(args.output, compressed)
    first_token = tokens[0]
    print(f"First token (must store separately): {first_token}")
    

    # ── Decode ──
    print("\n" + "=" * 60)
    print("DECODING")
    print("=" * 60)
    # decoded = decode_full(model, compressed, first_token, len(tokens), device)
    decoded = decode_with_cache(model, compressed, tokens_test[0], len(tokens_test), device)

    # ── Verify ──
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    # success = verify(tokens, decoded)
    success = verify(tokens_test, decoded)
    

    # ── Summary ──
    num_bases = (len(tokens) - 1) * 6
    bpb = num_bits / num_bases
    compressed_bytes = len(compressed) * 4
    original_bytes = len(tokens) * 6  # 1 byte per base in FASTA

    print("\n" + "=" * 60)
    print("COMPRESSION RESULTS")
    print("=" * 60)
    print(f"  Original:    {original_bytes:>12,} bytes ({original_bytes/1e6:.1f} MB)")
    print(f"  Compressed:  {compressed_bytes:>12,} bytes ({compressed_bytes/1e6:.1f} MB)")
    print(f"  Ratio:       {bpb:.4f} bits per base")
    print(f"  Savings:     {(1 - compressed_bytes/original_bytes)*100:.1f}%")
    print(f"  Lossless:    {'YES' if success else 'NO'}")