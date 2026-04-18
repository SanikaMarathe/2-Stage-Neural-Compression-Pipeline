"""
Compression quality benchmark: tests FGK Huffman across OCR output lengths.
Reports ratio, entropy, efficiency at different character counts.
"""

import requests

COMPRESS_URL = "http://localhost:8002/compress"

# Sample texts mimicking OCR output at different lengths
SAMPLES = [
    ("short",  "4 7"),
    ("short",  "3"),
    ("short",  "9 1 2"),
    ("medium", "4 7 2 9 1 0 3 5 8 6"),
    ("medium", "0 1 2 3 4 5 6 7 8 9 0 1 2"),
    ("medium", "5 3 8 2 7 4 9 1 6 0 5 3 8 2"),
    ("long",   "1 2 3 4 5 6 7 8 9 0 " * 5),
    ("long",   "0 1 2 3 4 5 6 7 8 9 " * 5),
    ("long",   "3 7 2 9 1 0 3 5 8 6 " * 5),
]


def run_benchmark():
    print(f"{'Length':<10} {'Chars':<8} {'Ratio':<10} {'Entropy':<12} {'Efficiency'}")
    print("-" * 55)

    seen_labels: dict[str, int] = {}

    for label, text in SAMPLES:
        text = text.strip()
        chars = len(text)

        # Deduplicate display label
        count = seen_labels.get(label, 0)
        seen_labels[label] = count + 1
        display_label = label if count == 0 else f"{label}_{count + 1}"

        try:
            resp = requests.post(
                COMPRESS_URL,
                json={"text": text},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            ratio = data.get("ratio", 0.0)
            entropy = data.get("entropy", 0.0)
            efficiency = data.get("efficiency", 0.0)

            print(
                f"{display_label:<10} {chars:<8} {ratio:<10.2f} {entropy:<12.2f} {efficiency:.2f}"
            )

        except requests.RequestException as e:
            print(f"{display_label:<10} {chars:<8} {'ERROR':<10} — {e}")


if __name__ == "__main__":
    run_benchmark()
