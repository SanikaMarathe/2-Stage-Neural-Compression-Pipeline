# 2-Stage Neural Compression Pipeline

Two-microservice pipeline: CNN-based OCR → Adaptive Huffman compression.

## Architecture

An image containing handwritten digits is sent to the OCR service, which segments individual digits using Otsu thresholding + connected components, classifies each with a noise-robust CNN (~468K parameters), and returns the digit string. That string is forwarded to the Huffman service, which compresses it using FGK Adaptive Huffman (pure Python, built from scratch) and returns the compressed payload with compression metrics.

Full system diagram and all endpoint contracts: [docs/architecture.md](docs/architecture.md)

## Quick Start

### Prerequisites

- Docker + Docker Compose, **or** Python 3.11+

### With Docker

```bash
docker compose up --build
```

Services:
- OCR: http://localhost:8001
- Huffman: http://localhost:8002

### Without Docker (local dev)

#### 1. Train the OCR model (run once)

`model.pt` is not checked in. Run the training script once to reproduce it — `eval_metrics.json` (per-noise-profile accuracy) is written alongside it.

```bash
cd ocr-service
pip install -r requirements.txt
python app/train.py
# Outputs: model.pt, eval_metrics.json
# Takes ~2 min on M3 MPS
```

#### 2. Start services

```bash
# Terminal 1
cd ocr-service && uvicorn app.main:app --port 8001

# Terminal 2
cd huffman-service && uvicorn app.main:app --port 8002
```

#### 3. Run the pipeline

```bash
# Step 1: OCR
curl -F "image=@sample.png" http://localhost:8001/ocr
# → {"text": "4 7 2 9", "digits": [4,7,2,9], "count": 4, "inference_ms": 12.3}

# Step 2: Compress
curl -X POST http://localhost:8002/compress \
  -H "Content-Type: application/json" \
  -d '{"text": "4 7 2 9"}'
# → {"compressed_b64": "...", "ratio": 0.82, "encoding_efficiency": 0.78, ...}
```

## Running Benchmarks

```bash
pip install requests
python benchmarks/accuracy_by_profile.py    # OCR accuracy by noise profile
python benchmarks/e2e_latency.py            # End-to-end latency (p50/p95/p99)
python benchmarks/compression_quality.py    # Compression ratio/entropy/efficiency
```

## Running Tests

```bash
cd neural-compression
pip install pytest
pytest tests/
```

## Performance (M3 MPS, post-training)

### OCR Accuracy by Noise Profile

| Profile | Accuracy |
|---------|----------|
| Clean | 99.4% |
| Gaussian (σ=0.3) | 99.2% |
| Salt & Pepper (5%) | 99.2% |

### Compression Quality by Input Length

| Input Length | Ratio | Entropy (bits/sym) | Efficiency |
|-------------|-------|--------------------|------------|
| Short (5 chars) | 1.40 | 2.32 | 0.41 |
| Medium (19 chars) | 0.82 | 3.10 | 0.78 |
| Long (99 chars) | 0.31 | 3.12 | 0.91 |

> Note: ratio > 1.0 for short inputs is expected — FGK pays a startup cost while the tree is sparse. Efficiency converges to ~0.91 at 99 characters.

### End-to-End Latency (N=100 requests, M3 MPS, local)

| Metric | p50 | p95 | p99 |
|--------|-----|-----|-----|
| Full pipeline (OCR + compress + decompress) | 9.2ms | 11.7ms | 39.8ms |
| OCR only | 5.3ms | — | — |
| Compress only | 2.0ms | — | — |
| Decompress only | 1.8ms | — | — |

## CNN Architecture & Design Justification

```
Input (1×28×28)
 │
 ▼
[Conv(1→32, 3×3, pad=1) → ReLU → Conv(32→32, 3×3, pad=1) → ReLU → MaxPool(2×2) → Dropout(0.25)]
 │  Block 1: 32 feature maps, 14×14
 ▼
[Conv(32→64, 3×3, pad=1) → ReLU → Conv(64→64, 3×3, pad=1) → ReLU → MaxPool(2×2) → Dropout(0.25)]
 │  Block 2: 64 feature maps, 7×7
 ▼
Flatten (3136) → Linear(3136→128) → ReLU → Dropout(0.50) → Linear(128→10) → logits
```

![DigitCNN Architecture Diagram](docs/cnn_architecture_diagram.png)

### Parameter Count

| Layer | Output Shape | Parameters |
|-------|-------------|------------|
| Conv1 | 32×28×28 | 320 |
| Conv2 | 32×28×28 | 9,248 |
| Conv3 | 64×14×14 | 18,496 |
| Conv4 | 64×14×14 | 36,928 |
| FC1   | 128        | 401,664 |
| FC2   | 10         | 1,290 |
| **Total** | | **~468K** |

### 3×3 Kernels, Stacked in Pairs

Two stacked 3×3 convolutions produce the same 5×5 receptive field as a single 5×5 kernel but require fewer parameters: `2 × 3² × C²` vs `5² × C²`. For C=32, that is 18,432 vs 25,600 — a 28% reduction. The additional benefit is an extra ReLU between them, giving the network a deeper composition of non-linearities for the same spatial reach. This is the core VGGNet insight (Simonyan & Zisserman, 2014).

### padding=1

Same-padding on every 3×3 conv keeps the spatial dimensions unchanged through the convolutional layers. Spatial downsampling happens only at explicit MaxPool operations. This makes shape arithmetic trivial and avoids unintentional edge-cropping artifacts — relevant because our OCR segmentation crops may not produce perfectly centered 28×28 patches.

### Channel Doubling (32 → 64)

Each MaxPool halves both spatial dimensions, reducing the total feature volume by 4×. Doubling channels at each block recovers half that capacity, so the network can represent increasingly abstract features per spatial location without a net collapse in representational bandwidth. This is standard practice in VGG-style networks.

### MaxPool over Strided Convolutions

MaxPool has zero learnable parameters and is fully deterministic. It preserves the strongest activation from each 2×2 spatial region — exactly the right operation for digit recognition where you want to know whether a stroke feature is present in a region, not a learned weighted blend. Strided convolution would add 36,864 additional parameters to learn an operation MaxPool performs for free and arguably better.

### Dropout Rates — 0.25 in Conv Blocks, 0.50 in FC

The FC1 layer (3136→128) contains 401,408 parameters — roughly 86% of the entire network. It is the most overfit-prone component. Dropout at 0.50 forces the classifier to learn redundant, distributed representations. The convolutional layers are far smaller, spatially local, and naturally share weights across positions, so lighter 0.25 dropout is sufficient without degrading their ability to learn local stroke features.

### No BatchNorm

BatchNorm adds per-layer computational overhead, introduces batch-size dependency during training, and can destabilise training when batch statistics are noisy. MNIST is a small, clean, well-normalised dataset — training is stable without it. Dropout alone provides sufficient regularisation.

### No Softmax in `forward()`

PyTorch's `CrossEntropyLoss` internally applies `LogSoftmax` followed by `NLLLoss` using the numerically stable log-sum-exp formulation. Adding an explicit softmax in `forward()` before passing to `CrossEntropyLoss` creates double-softmax: the loss receives already-normalised probabilities, squashes them a second time into a near-uniform distribution, and produces incorrect near-zero gradients. The model outputs raw logits during training. Softmax is applied only at inference time.

### Noise Augmentation Strategy

Training data is augmented on-the-fly with three equal-probability strata:

| Profile | Description |
|---------|-------------|
| Clean | Original MNIST pixel values |
| Gaussian | Additive noise σ=0.3, clipped to [0,1] |
| Salt & Pepper | 5% of pixels randomly set to 0 or 1 |

This gives the model noise-invariant internal representations without a separate denoising preprocessing step.

---

## FGK Adaptive Huffman Algorithm

### Why Adaptive?

Static Huffman coding requires two passes over the input: count symbol frequencies, build the tree, then encode. The receiver also needs the tree transmitted as overhead before the compressed payload begins.

**Adaptive Huffman** (Faller 1973, Gallager 1978, Knuth 1985 — hence FGK) eliminates both constraints. The tree is built incrementally as each symbol is encoded. Because encoder and decoder process the same symbols in the same order and apply identical update rules, their trees remain bit-for-bit identical throughout. No tree transmission is needed; the compressed bitstream is self-contained.

### Core Data Structures

- **Nodes** carry a weight (symbol frequency seen so far) and an implicit **node number**. Nodes are numbered in breadth-first, left-to-right order from the root downward: the root has the highest number, leaves near the bottom have the lowest.
- **NYT node** ("Not Yet Transmitted") — a special leaf with weight 0. It represents all symbols not yet seen.

### Encoding a Symbol

**First occurrence:**
1. Emit the current path from root to the NYT node.
2. Emit the raw 8-bit ASCII value of the symbol.
3. Split the NYT leaf into two children: a new NYT leaf (left) and a new symbol leaf (right), both with weight 0.

**Subsequent occurrences:**
1. Emit the current path to the symbol's leaf.
2. Apply the update rule (swap walk).

### Sibling Property & Swap Walk

A binary tree is a valid Huffman tree if and only if its nodes, listed in non-decreasing order of weight, have siblings adjacent in that list. FGK maintains this dynamically via the swap rule:

```
current_node = symbol's leaf
while current_node != root:
    candidate = highest-numbered node with same weight as current_node
                that is not current_node's parent
    if candidate != current_node:
        swap(current_node, candidate)   # exchange tree positions + node numbers
    increment current_node.weight
    current_node = current_node.parent
increment root.weight
```

Swapping with the **highest-numbered** node of equal weight ensures the least-frequent nodes remain deepest (longest codes) while high-frequency nodes rise toward the root (shorter codes) — the precise condition that maintains the sibling property after each update.

### Decoder Symmetry

The decoder traverses the current tree from root to leaf on each received bit. When it reaches a leaf:
- If NYT: reads the next 8 bits as a raw symbol, applies the same split.
- Otherwise: identifies the symbol, applies the same swap walk.

Both sides apply identical operations to an identical sequence of symbols, so the trees stay synchronised with no explicit tree data in the bitstream.

### Compression Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Compression ratio | compressed\_bytes / original\_bytes | <1.0 = compression; >1.0 = expansion |
| Shannon entropy | −Σ p(c) log₂ p(c) | Theoretical minimum bits/symbol |
| Encoding efficiency | entropy / avg\_bits\_per\_symbol | 1.0 = Shannon-optimal |

> Short inputs expand (ratio > 1.0) because FGK pays a startup cost while the tree is sparse. Efficiency converges toward 1.0 as the symbol stream grows and the tree stabilises — reaching 0.91 at 99 characters in our benchmarks.

---

## Design Documentation

- [System Architecture & Endpoint Contracts](docs/architecture.md)

## Repository Structure

```
neural-compression/
├── ocr-service/app/
│   ├── main.py        # FastAPI: /ocr, /metrics, /healthz
│   ├── model.py       # DigitCNN (~468K params)
│   ├── train.py       # MNIST training + noise augmentation
│   └── segment.py     # Otsu + connected-component segmentation
├── huffman-service/app/
│   ├── main.py        # FastAPI: /compress, /decompress, /healthz
│   └── huffman.py     # FGK adaptive Huffman (pure Python)
├── benchmarks/
│   ├── accuracy_by_profile.py
│   ├── e2e_latency.py
│   └── compression_quality.py
├── tests/
├── docs/
└── docker-compose.yml
```

## Graduate Requirements Checklist

- [x] Two noise profiles (Gaussian σ=0.3 + salt-and-pepper 5%) with per-profile accuracy reporting
- [x] Compression metrics: ratio, Shannon entropy, encoding efficiency
- [x] End-to-end latency benchmarked (see `benchmarks/e2e_latency.py`)
- [x] CNN architecture documented with full design justification (`docs/cnn_justification.md`)
- [x] FGK algorithm explained with sibling property and swap-walk derivation (`docs/fgk_algorithm.md`)
- [x] System architecture diagram + all endpoint contracts (`docs/architecture.md`)
