# Genomic Compression with Decoder-Only Transformers

Lossless compression of the human reference genome using a small autoregressive transformer paired with an arithmetic coder. The model predicts the next 6-mer token of DNA conditioned on the preceding sequence; the predicted distribution feeds a range coder that compresses the actual token to its information content. Reconstruction is exact.

On chromosome 22 (held out from training and validation) we reach **1.450 bits per base**, beating gzip, xz, Genozip, and GeCo3.

| Compressor | Bits per base |
|------------|--------------:|
| gzip       | 2.120 |
| xz         | 1.677 |
| Genozip    | 1.705 |
| GeCo3      | 1.539 |
| **Ours**   | **1.450** |


## Architecture

A 12-layer decoder-only transformer (~43M parameters):

- 6-mer tokenization, vocabulary size 4096
- Embedding dimension 512, 8 attention heads
- Rotary positional embeddings
- SwiGLU feedforward, RMSNorm
- Tied input and output embeddings
- Causal self-attention with KV caching for fast streaming compression


## Repository layout

```
.
├── preprocess.py              Split FASTA into per-chromosome text files
├── tokenizer.py               6-mer encode/decode
├── data.py                    GenomeDataset + train/val dataloaders
├── model.py                   Decoder-only transformer
├── train.py                   End-to-end training loop
├── encode.py                  Arithmetic encoder (greedy, KV-cached, batched)
├── decode.py                  Arithmetic decoder (matches encoder variants)
├── verify.py                  Compress + decompress + bit-exact check
├── extract_hidden_states.py   Pull layer-11 activations on chr22
└── notebooks/
    ├── extract_hidden_states.ipynb   Colab version of the extraction script
    └── visualizations.ipynb          All paper figures + SAE interpretability
```


## Setup

```bash
git clone https://github.com/dariussattari/genomic-compression.git
cd genomic-compression
pip install -r requirements.txt
```

A CUDA GPU with at least 16 GB of memory is recommended for training. Inference (encode / decode) runs on CPU but is much slower. The authors used Google Colab Pro for compute resources.


## Reproducing the results

### 1. Download the reference genome

Get GRCh38 from NCBI:

```
https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.26_GRCh38/
```

Save the unzipped FASTA as `GCF_000001405.26_GRCh38_genomic.txt` in the repo root.

### 2. Preprocess into per-chromosome text files

```bash
python preprocess.py
```

Produces `chr1.txt` through `chr24.txt`, one ASCII letter per base, no newlines, with `N` characters dropped.

### 3. Tokenize

```python
from data import tokenize_chromosomes
tokenize_chromosomes('tokenized_data')
```

Place the `chr*.txt` files in `tokenized_data/` first. This writes `chr1.npy` ... `chr24.npy` alongside them.

### 4. Train

```bash
python train.py --token-dir tokenized_data --checkpoint-dir checkpoints
```

Default hyperparameters reproduce the reported result (12 layers, embed 512, 8 heads, sequence length 2048, batch size 32, AdamW lr 3e-4, 5 epochs with cosine decay). Validation uses chr21; chr22 is never seen during training.

The script saves `checkpoints/last_model.pt` after every epoch and `checkpoints/best_model.pt` whenever validation bpb improves.

### 5. Verify lossless compression on chr22

```bash
python verify.py \
    --checkpoint checkpoints/best_model.pt \
    --tokens tokenized_data/chr22.npy \
    --output chr22_compressed.npy
```

Encodes a slice of chr22, decodes it back, and checks token-level equality.

### 6. Extract hidden states for interpretability

```bash
python extract_hidden_states.py
```

Runs chr22 through the model and saves layer-11 activations to `chr22_hidden_states.pt`. Use `notebooks/extract_hidden_states.ipynb` to do the same on Colab.

### 7. Reproduce the figures

Open `notebooks/visualizations.ipynb` in Colab and run top to bottom. The notebook expects a Drive layout under `MyDrive/genomic_compressor/` containing the checkpoint, tokenized chromosomes, hidden states, SAE checkpoint, and an `annotations.py` helper for genomic annotation lookup. See the notebook header for details.


## Data split

| Split           | Chromosomes |
|-----------------|-------------|
| Training        | chr1-20, chr23, chr24 |
| Validation      | chr21 |
| Held-out test   | chr22 |

`chr22` is the smallest autosome and is never touched by the training pipeline, which makes it a clean target for compression benchmarks and interpretability work.


## Citation

If you use this work, please cite:

> Boateng, K, Sattari, D. *Genomic Compression with Decoder-Only Transformers.* 2026.

A link to the paper will be added once available.


## License

MIT. See [LICENSE](LICENSE).
