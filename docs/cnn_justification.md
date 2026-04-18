# CNN Architecture & Design Justification

## Architecture Overview

```
Input (1×28×28)
 │
 ▼
[Conv(1→32, 3×3, pad=1) → ReLU → Conv(32→32, 3×3, pad=1) → ReLU → MaxPool(2) → Dropout(0.25)]
 │  Block 1: 32 feature maps, 14×14
 ▼
[Conv(32→64, 3×3, pad=1) → ReLU → Conv(64→64, 3×3, pad=1) → ReLU → MaxPool(2) → Dropout(0.25)]
 │  Block 2: 64 feature maps, 7×7
 ▼
Flatten (3136) → Linear(128) → ReLU → Dropout(0.5) → Linear(10) → logits
```

## Parameter Count

| Layer | Output Shape | Parameters |
|-------|-------------|------------|
| Conv1 | 32×28×28 | 320 |
| Conv2 | 32×28×28 | 9,248 |
| Conv3 | 64×14×14 | 18,496 |
| Conv4 | 64×14×14 | 36,928 |
| FC1   | 128        | 401,664 |
| FC2   | 10         | 1,290 |
| **Total** | | **~468K** |

## Design Justifications

### 1. 3×3 Kernels, Stacked in Pairs

Two stacked 3×3 convolutions produce the same 5×5 receptive field as a single 5×5 kernel but require fewer parameters: `2 × 3² × C²` vs `5² × C²`. For C=32, that is 18,432 vs 25,600 — a 28% reduction. The additional benefit is an extra ReLU between them, giving the network a deeper composition of non-linearities for the same spatial reach. This is the core VGGNet insight (Simonyan & Zisserman, 2014).

### 2. padding=1

Same-padding on every 3×3 conv keeps the spatial dimensions unchanged through the convolutional layers. Spatial downsampling happens only at explicit MaxPool operations. This makes shape arithmetic trivial and avoids unintentional edge-cropping artifacts — relevant because our OCR segmentation crops may not produce perfectly centered 28×28 patches.

### 3. Channel Doubling (32 → 64)

Each MaxPool halves both spatial dimensions, reducing the total feature volume by 4×. Doubling channels at each block recovers half that capacity, so the network can represent increasingly abstract features per spatial location without a net collapse in representational bandwidth. This is standard practice in VGG-style networks.

### 4. MaxPool over Strided Convolutions

MaxPool is parameter-free and translation-tolerant: it preserves the strongest activation within each 2×2 window regardless of exact position. For a digit OCR task where segmented crops may be slightly shifted, this local invariance is preferable to strided convolutions, which learn a fixed sampling pattern.

### 5. Dropout Placement

- **0.25 in conv blocks** — Light regularization after spatial pooling. Conv layers share weights spatially, so they are already constrained; aggressive dropout would destroy too many feature maps.
- **0.5 before final FC** — The FC1 layer (3136×128 = ~401K weights) accounts for 86% of all parameters. This is where overfitting pressure is highest, so dropout is set to its heaviest value.

### 6. No BatchNorm

BatchNorm adds two learned parameters per channel plus running statistics, and benefits are marginal on MNIST-scale inputs where the signal is clean and batches are large relative to class count. Omitting it reduces implementation complexity without measurable accuracy loss, and simplifies the explanation for a 24-hour build context.

### 7. No Softmax in forward()

`CrossEntropyLoss` in PyTorch fuses log-softmax and negative log-likelihood internally using the numerically stable log-sum-exp trick. Applying an explicit softmax before `CrossEntropyLoss` would negate this stability guarantee. Raw logits are returned from `forward()` and softmax is applied only at inference time where needed.

### 8. Adam Optimizer

Adam maintains per-parameter adaptive learning rates via first and second moment estimates. This removes the need for manual learning-rate tuning or a scheduling sweep — critical when build time is limited to 24 hours. SGD with momentum achieves comparable final accuracy on MNIST but requires careful LR and momentum tuning.

### 9. Noise Augmentation Strategy

Training data is augmented on-the-fly with three equal probability strata (each ~1/3 of batches):

| Profile | Description |
|---------|-------------|
| Clean | Original MNIST pixel values |
| Gaussian | Additive noise σ=0.3, clipped to [0,1] |
| Salt & Pepper | 5% of pixels randomly set to 0 or 1 |

This gives the model noise-invariant internal representations without a separate denoising preprocessing step. At evaluation, three fixed held-out splits — one per noise profile — are used to report per-profile accuracy, meeting the graduate requirement for multi-profile robustness reporting.
