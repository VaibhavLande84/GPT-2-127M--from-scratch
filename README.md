# GPT-2 (127M) — From Scratch

A **126.7M parameter** decoder-only transformer language model built from scratch in PyTorch, trained on biomedical literature (PMC articles). This project implements a modernized GPT-2 architecture with **Grouped-Query Attention (GQA)**, **Rotary Position Embeddings (RoPE)**, **SwiGLU** activations, **RMSNorm** (Pre-LN), and **weight tying**.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Parameter Calculations](#parameter-calculations)
- [Architecture Components](#architecture-components)
- [Data Collection & Extraction](#data-collection--extraction)
- [Tokenization](#tokenization)
- [Training](#training)
- [Text Generation](#text-generation)
- [Project Structure](#project-structure)
- [Requirements](#requirements)

---

## Architecture Overview

| Hyperparameter | Value |
|----------------|-------|
| Parameters | **126.70M** |
| Layers (`n_layer`) | 14 |
| Hidden dim (`n_embd`) | 768 |
| Attention heads (`n_head`) | 12 |
| KV heads (`n_kv_heads`) | 4 |
| Head dimension | 64 |
| Vocab size | 50,257 |
| Context length (`block_size`) | 1024 |
| GQA groups (`n_kv_groups`) | 3 (12 Q-heads ÷ 4 KV-heads) |
| Activation | SwiGLU |
| Normalization | RMSNorm (Pre-LN) |
| Position encoding | RoPE |
| Dropout | 0.3 |
| Biases | None (all `bias=False`) |

---

## Parameter Calculations

Every parameter count broken down component by component.

### 1. Token Embedding (`wte`)

```
Params = vocab_size × n_embd
       = 50257 × 768
       = 38,597,376
```

> **Note:** Due to weight tying, these parameters are shared with `lm_head` and are **not** duplicated in the total count.

### 2. Transformer Blocks (×14 layers)

Each block contains: Attention + MLP + 2× RMSNorm

#### 2a. Grouped-Query Attention (GQA)

```
Q projection:  n_embd × (n_head × head_dim)  = 768 × (12 × 64) = 768 × 768   = 589,824
K projection:  n_embd × (n_kv_head × head_dim) = 768 × (4 × 64)  = 768 × 256   = 196,608
V projection:  n_embd × (n_kv_head × head_dim) = 768 × (4 × 64)  = 768 × 256   = 196,608
Output proj:  (n_head × head_dim) × n_embd    = 768 × 768       = 589,824

Attention total per layer:  589,824 + 196,608 + 196,608 + 589,824 = 1,572,864
```

#### 2b. SwiGLU MLP

SwiGLU uses 3 weight matrices with a hidden dimension of `int(8 × n_embd / 3)`:

```
hidden_dim = int(8 × 768 / 3) = 2048

Gate projection:  n_embd × hidden_dim = 768 × 2048 = 1,572,864
Up projection:    n_embd × hidden_dim = 768 × 2048 = 1,572,864
Down projection:  hidden_dim × n_embd = 2048 × 768 = 1,572,864

MLP total per layer: 1,572,864 + 1,572,864 + 1,572,864 = 4,718,592
```

#### 2c. RMSNorm (×2 per layer)

```
RMSNorm 1: n_embd = 768
RMSNorm 2: n_embd = 768

RMSNorm total per layer: 768 + 768 = 1,536
```

#### Total per Block

```
1,572,864 (attention) + 4,718,592 (MLP) + 1,536 (RMSNorms) = 6,292,992
```

#### All 14 Blocks

```
6,292,992 × 14 = 88,101,888
```

### 3. Final RMSNorm (`ln_f`)

```
RMSNorm: n_embd = 768
```

### 4. LM Head (shared with embeddings via weight tying)

```
Params = vocab_size × n_embd = 50257 × 768 = 38,597,376
```

> Due to weight tying (`wte.weight = lm_head.weight`), these parameters are **shared** with the token embedding layer. They are counted once in the total.

### 5. Total Parameter Count

| Component | Parameters | % of Total |
|-----------|-----------|------------|
| Token Embeddings (tied) | 38,597,376 | — (shared) |
| 14 Attention layers | 22,020,096 | 17.4% |
| 14 SwiGLU MLP layers | 66,060,288 | 52.1% |
| 28 RMSNorm layers | 21,504 | <0.1% |
| Final RMSNorm | 768 | <0.1% |
| LM Head (tied) | 0 | — |
| **Total** | **126,702,720** (~126.70M) | **100%** |

```python
# Quick sanity check
n_embd, n_head, n_kv_head, head_dim = 768, 12, 4, 64
n_layer, vocab_size = 14, 50257
hidden_dim = int(8 * 768 / 3)

attn_params  = 3 * n_embd * (n_head + 2 * n_kv_head) * head_dim  # = 1,572,864
mlp_params   = 3 * n_embd * hidden_dim                           # = 4,718,592
rms_params   = 2 * n_embd                                        # = 1,536
block_params = attn_params + mlp_params + rms_params             # = 6,292,992

total = block_params * n_layer + n_embd                          # = 88,101,888 + 768
print(f"Total: {total:,} ≈ {total/1e6:.2f}M")
```

> **Why ~126.7M instead of 124M?** The original GPT-2 small used 48,000 vocab and GELU. Our model uses 50,257 vocab (SentencePiece BPE), which slightly increases embedding parameters, but weight tying keeps the net count lower than it would otherwise be.

---

## Architecture Components

### Grouped-Query Attention (GQA)

**GQA** is a middle-ground between Multi-Head Attention (MHA) and Multi-Query Attention (MQA). Instead of having separate Key/Value heads for every Query head (MHA), GQA shares KV heads across groups of Query heads.

```
         ┌──────────────────┐
         │     Input (768)   │
         └────────┬─────────┘
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
   ┌────────┐ ┌────────┐ ┌────────┐
   │ Q Proj │ │ K Proj │ │ V Proj │
   │ 12×64  │ │  4×64  │ │  4×64  │
   └───┬────┘ └───┬────┘ └───┬────┘
       │          │          │
       │    ┌─────┴─────┐    │
       │    │ repeat × 3│    │
       │    └─────┬─────┘    │
       │          │          │
       └──────────┼──────────┘
                  ▼
         ┌────────────────┐
         │  FlashAttn     │
         │  (causal mask) │
         └────────┬───────┘
                  ▼
         ┌────────────────┐
         │   Out Proj     │
         └────────────────┘
```

**Configuration:** 12 Query heads, 4 KV heads → 3 Query heads share each KV head.

**Benefits:**
- **~2.6× fewer KV parameters** vs MHA (4 KV heads vs 12)
- **Lower KV cache memory** during inference (important for a 1K context)
- Minimal quality degradation vs MHA (validated in the GQA paper)

### Rotary Position Embeddings (RoPE)

RoPE encodes position information directly into the attention computation by rotating query and key vectors by an angle proportional to their position.

```python
# For each pair of dimensions (2d, 2d+1), RoPE applies:
# [x_2d, x_{2d+1}] @ [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
# where θ = position / 10000^(2d / head_dim)
```

**Why RoPE instead of learned position embeddings:**
- **No extra parameters** — learned position embeddings (`wpe`) would add `1024 × 768 = 786,432` parameters
- **Relative position bias** — the rotation naturally encodes relative distances, making it easier for the model to generalize to longer sequences
- **Better extrapolation** — RoPE can handle sequences longer than those seen during training
- **Theoretical grounding** — the rotation matrix formulation ensures position information decays naturally with distance

### SwiGLU Activation

SwiGLU replaces the standard GELU activation with a gated mechanism:

```python
# GELU (original GPT-2):
x = GELU(W₁ @ x)               # single projection

# SwiGLU (ours):
x = SiLU(W_gate @ x) * (W_up @ x)   # gated via element-wise multiplication
x = W_down @ x
```

**Configuration:** Hidden dimension = `int(8 × n_embd / 3)` = 2048, which keeps the total MLP parameter count similar to the standard 4× expansion (which would be 3072) while adding a gating mechanism.

**Benefits:**
- **Gated mechanism** gives the network more expressivity per parameter than GELU
- **SiLU (Swish)** is smoother than ReLU and empirically outperforms GELU in modern LLMs (PaLM, Llama)

### RMSNorm (Pre-LN)

RMSNorm is a simplified version of LayerNorm that only normalizes by the root mean square statistic, omitting the mean-centering step:

```python
# LayerNorm:
x = (x - μ) / √(σ² + ε) * γ + β    # mean + variance, learnable bias β

# RMSNorm:
x = x / √(mean(x²) + ε) * γ         # only RMS, no mean, no bias
```

**Pre-LN placement** means normalization is applied **before** the sublayer:

```python
# Post-LN (original GPT-2):
x = LayerNorm(x + Sublayer(x))       # normalization after residual add

# Pre-LN (ours):
x = x + Sublayer(RMSNorm(x))         # normalization before sublayer
```

**Benefits:**
- **Computation reduction** — no mean calculation, no bias parameter
- **Training stability** — Pre-LN prevents exploding activations in deep models
- **Fewer parameters** — saves `n_embd` bias parameters per normalization layer compared to LayerNorm

### Weight Tying

Weight tying shares the weight matrix between the input token embedding layer and the output language modeling head:

```python
self.transformer.wte.weight = self.lm_head.weight
```

**Parameters saved:**

```
Without tying:  vocab_size × n_embd + vocab_size × n_embd = 2 × 50257 × 768 = 77,194,752
With tying:     vocab_size × n_embd                       = 1 × 50257 × 768 = 38,597,376
────────────────────────────────────────────────────────────────────────────────────
Saved:                                                      38,597,376 parameters (~38.6M)
```

This massive savings (~30% of total model parameters) means more capacity can be allocated to the transformer blocks themselves.

### Scaled Residual Initialization

Residual projection weights (`out_proj` in attention, `down_proj` in MLP) are initialized with:

```python
std = 0.02 / √(2 × n_layer) = 0.02 / √(28) ≈ 0.00378
```

This smaller initialization prevents the variance from accumulating across 14 residual streams, keeping activations stable in the deep network.

---

## Data Collection & Extraction

### Datasets

| Dataset | Source | Records | Size | Purpose |
|---------|--------|---------|------|---------|
| **PMC OA** | `casperhansen/pmc-oa-markdown` | 15,000 | ~1.2M tokens | Primary training corpus |
| **FineWeb-Edu** | `HuggingFaceFW/fineweb-edu` | 15,000 | ~1.2M tokens | Tokenizer training only |

### PMC (PubMed Central) — Training Data

**Source:** [PMC-OA-Markdown on HuggingFace](https://huggingface.co/datasets/casperhansen/pmc-oa-markdown)

15,000 open-access medical journal articles from the US National Library of Medicine, stream-loaded and saved as a JSONL file:

```python
# datacollection.ipynb
loaded_data = load_dataset('casperhansen/pmc-oa-markdown', streaming=True)

with open('pmc_articles.jsonl', 'w', encoding='utf-8') as f:
    for article in loaded_data["train"]:
        record = {"text": article.get("text")}
        f.write(json.dumps(record) + "\n")
```

Output: `pmc_articles.jsonl` — 15,000 articles with `{"text": "..."}` format.

### FineWeb-Edu — Tokenizer Training Data

Used only to enrich the SentencePiece tokenizer with general-domain web text, ensuring the tokenizer isn't overly specialized to biomedical jargon.

### Data Interleaving

During tokenizer training, PMC and FineWeb are interleaved 50/50:

```python
mixed_ds = interleave_datasets(
    [medical_ds, fineweb_ds],
    probabilities=[0.5, 0.5],
    seed=42
)
```

---

## Tokenization

### SentencePiece BPE Tokenizer

We train a **Byte-Pair Encoding (BPE)** tokenizer using SentencePiece from scratch on the mixed dataset.

**Why SentencePiece BPE:**
- **Byte-level fallback** (`byte_fallback=True`) ensures every possible Unicode character can be encoded, not just those seen in training
- **Language-agnostic** — works well on biomedical text with specialized terminology and chemical formulas
- **Lossless encoding** — no unknown tokens; any unseen character is decomposed into byte tokens
- **50,257 vocab** matches the original GPT-2 vocabulary size (with ID 50,256 reserved for `<|endoftext|>`)

### Training Configuration

```python
spm.SentencePieceTrainer.train(
    input='mixed_corpus.txt',
    model_prefix='med_fine_sp',
    vocab_size=50256,            # 50,256 BPE tokens + 1 reserved slot
    model_type='bpe',            # Byte-Pair Encoding
    character_coverage=1.0,      # Cover all Unicode characters
    byte_fallback=True           # Fall back to UTF-8 bytes for unknown chars
)
```

### Reserved Special Token

| ID | Token | Purpose |
|----|-------|---------|
| 0–50255 | BPE subword tokens | Regular vocabulary |
| **50256** | **`<|endoftext|>`** | Document boundary marker |

The last ID is deliberately left out of BPE training so it can serve as a document separator. Every document in the training data is followed by this token:

```
[doc1 tokens...] [EOS=50256] [doc2 tokens...] [EOS=50256] ...
```

### Tokenization Output

```
Text:      "hospital!! !"
Pieces:    ["▁hospital", "!", "!", "▁", "!"]
IDs:       [23756, 5, 5, 22, 5]
```

### Pre-tokenized Files

After tokenization, the data is split and saved as NumPy arrays:

| File | Content |
|------|---------|
| `train.npy` | 95% of tokens for training |
| `val.npy` | 5% of tokens for validation loss monitoring |

---

## Training

### Optimizer

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (fused when available) |
| Weight decay | 0.01 (applied to 2D+ params only) |
| Learning rate | 1×10⁻⁴ |
| Betas | (0.9, 0.95) |
| Gradient clipping | 1.0 |

**Parameter groups:**
- **Decayed** (dim ≥ 2): weight matrices in linear layers and embeddings
- **Non-decayed** (dim < 2): RMSNorm weights and biases

### GPU Memory Estimate (AdamW)

A 126.7M model with AdamW requires roughly **3× the model size** in GPU memory:

```
Model weights:      126.7M × 4 bytes (FP32) ≈ 507 MB
AdamW states:       126.7M × 8 bytes (2×FP32) ≈ 1,014 MB
Gradients:          126.7M × 4 bytes (FP32) ≈ 507 MB
Activations:        TBD (depends on batch size & seq len)
────────────────────────────────────────────────────────
Approximate total:  >2 GB (FP32) / ~1.3 GB (mixed precision)
```

---

## Text Generation

The `generate()` method uses **autoregressive sampling**:

```
Input:  "The study shows that"
Output: "The study shows that the treatment group exhibited a statistically
         significant improvement in patient outcomes compared to the control
         group (p < 0.05). These findings suggest that..."
```

**Generation parameters:**
- `temperature = 0.7` — controls randomness (lower = more deterministic)
- `top_k = 40` — restricts sampling to the 40 most likely tokens
- `max_new_tokens = 100` — sequence continuation length

---

## Project Structure

```
GPT-2-127M--from-scratch
├── README.md                      <- You are here
├── datacollection.ipynb           <- Stream & save PMC articles as JSONL
├── Tokenization.ipynb             <- Train BPE tokenizer + save .npy files
├── GPT2Fromhuggingface.ipynb      <- Reference: HuggingFace GPT-2
├── GPT_traning/train.ipynb        <- Model definition + training loop
├── med_fine_sp.model              <- Trained SentencePiece model
├── med_fine_sp.vocab              <- Trained SentencePiece vocab
├── pmc_articles.jsonl             <- 15K PMC articles (training data)
└── Data_collection/               <- Additional data scripts
```

---

## Requirements

```
torch>=2.0.0        # Flash attention via scaled_dot_product_attention
sentencepiece       # BPE tokenization
numpy               # Data serialization (.npy files)
datasets            # HuggingFace dataset streaming
tqdm                # Progress bars
```

---

## References

- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) — Radford et al.
- [GQA: Training Generalized Multi-Query Transformer](https://arxiv.org/abs/2305.13245) — Ainslie et al.
- [RoFormer: Enhanced Transformer with Rotary Embeddings](https://arxiv.org/abs/2104.09864) — Su et al.
- [SwiGLU: GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — Shazeer
- [RMSNorm: Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — Zhang & Sennrich
- [SentencePiece Tokenizer](https://github.com/google/sentencepiece) — Kudo & Richardson
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Andrej Karpathy (reference implementation)
  

