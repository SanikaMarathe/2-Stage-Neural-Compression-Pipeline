# System Architecture

## Pipeline Overview

```
Client
│
│  POST /ocr  (multipart image)
▼
┌─────────────────────────────────┐
│        OCR Service :8001        │
│  ┌──────────┐  ┌─────────────┐ │
│  │ segment  │→ │  DigitCNN   │ │
│  │ .py      │  │  (PyTorch)  │ │
│  └──────────┘  └─────────────┘ │
│    Otsu+CC        ~468K params  │
└────────────┬────────────────────┘
            │  {"text": "4 7 2 9", ...}
            │
            │  POST /compress  {"text": "4 7 2 9"}
            ▼
┌─────────────────────────────────┐
│     Huffman Service :8002       │
│      FGK Adaptive Huffman       │
│    (pure Python, from scratch)  │
└────────────┬────────────────────┘
            │  {"compressed_b64": "...", "ratio": 0.82, ...}
            ▼
        Client
```

## Service Details

### OCR Service (:8001)

- **Segmentation** (`segment.py`): Otsu thresholding + connected-component analysis to isolate individual digit bounding boxes. Each bounding box is padded and resized to 28×28 grayscale.
- **Classification** (`model.py`): Two-block VGG-style CNN (~468K parameters). See [CNN Architecture & Justification](cnn_justification.md).
- **Framework**: FastAPI + PyTorch (MPS/CUDA/CPU autodetect)
- **Model artifact**: `ocr-service/model.pt` (produced by `train.py`, not checked in)

### Huffman Service (:8002)

- **Algorithm**: FGK Adaptive Huffman, pure Python, no external compression libraries. See [FGK Algorithm](fgk_algorithm.md).
- **Framework**: FastAPI
- **State**: Stateless per-request — tree is built fresh for each compress/decompress call.

## Endpoint Contracts

### GET /healthz (both services)

**Response 200**
```json
{"status": "ok"}
```

---

### POST /ocr (OCR Service :8001)

**Request**: `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `image` | file | PNG or JPEG image containing handwritten digits |

**Response 200**
```json
{
"text": "4 7 2 9",
"digits": [4, 7, 2, 9],
"count": 4,
"inference_ms": 12.3
}
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Space-separated digit string |
| `digits` | int[] | Ordered list of classified digits |
| `count` | int | Number of digits detected |
| `inference_ms` | float | End-to-end inference time in milliseconds |

**Response 422**: No digits detected or segmentation failure.

---

### GET /metrics (OCR Service :8001)

Returns per-noise-profile accuracy metrics loaded from `eval_metrics.json` (written by `train.py`).

**Response 200**
```json
{
"clean_accuracy": 0.992,
"gaussian_accuracy": 0.973,
"salt_pepper_accuracy": 0.978,
"model_parameters": 468026,
"training_epochs": 10
}
```

---

### POST /compress (Huffman Service :8002)

**Request**: `application/json`

```json
{"text": "4 7 2 9"}
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Plaintext string to compress |

**Response 200**
```json
{
"compressed_b64": "SGVsbG8...",
"original_bytes": 7,
"compressed_bytes": 6,
"ratio": 0.857,
"entropy_bits_per_symbol": 2.807,
"avg_bits_per_symbol": 3.43,
"encoding_efficiency": 0.818
}
```

| Field | Type | Description |
|-------|------|-------------|
| `compressed_b64` | string | Base64-encoded compressed bitstream |
| `original_bytes` | int | Byte length of input text |
| `compressed_bytes` | int | Byte length of compressed output |
| `ratio` | float | compressed / original (< 1.0 = compression) |
| `entropy_bits_per_symbol` | float | Shannon entropy H for the input symbol distribution |
| `avg_bits_per_symbol` | float | Actual average code length achieved |
| `encoding_efficiency` | float | entropy / avg_bits (1.0 = Shannon-optimal) |

---

### POST /decompress (Huffman Service :8002)

**Request**: `application/json`

```json
{
"compressed_b64": "SGVsbG8...",
"original_length": 7
}
```

| Field | Type | Description |
|-------|------|-------------|
| `compressed_b64` | string | Base64-encoded compressed bitstream from `/compress` |
| `original_length` | int | Original byte count (used to stop decoding at exact boundary) |

**Response 200**
```json
{"text": "4 7 2 9"}
```

**Response 422**: Malformed base64 or length mismatch.

## Repository Layout

```
neural-compression/
├── ocr-service/
│   ├── app/
│   │   ├── main.py          # FastAPI app, /ocr and /metrics endpoints
│   │   ├── model.py         # DigitCNN definition
│   │   ├── train.py         # MNIST training + eval_metrics.json output
│   │   └── segment.py       # Otsu + connected-component segmentation
│   ├── Dockerfile
│   └── requirements.txt
├── huffman-service/
│   ├── app/
│   │   ├── main.py          # FastAPI app, /compress and /decompress endpoints
│   │   └── huffman.py       # FGK adaptive Huffman implementation
│   ├── Dockerfile
│   └── requirements.txt
├── benchmarks/
│   ├── accuracy_by_profile.py    # Per-noise-profile OCR accuracy
│   ├── e2e_latency.py            # End-to-end latency (p50/p95/p99)
│   └── compression_quality.py   # Ratio, entropy, efficiency by input length
├── tests/
│   ├── test_ocr.py
│   └── test_huffman.py
├── docs/
│   ├── architecture.md          # This file
│   ├── cnn_justification.md
│   └── fgk_algorithm.md
├── docker-compose.yml
└── README.md
```
