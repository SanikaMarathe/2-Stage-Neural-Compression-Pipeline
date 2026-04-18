import requests

COMPRESS_URL = "http://localhost:8002/compress"

# sample texts mimicking ocr output at different lengths
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
    seen: dict[str,int] = {}  # track how many times each label seen

    for label,text in SAMPLES:
        text = text.strip()
        nch = len(text)  # char count

        cnt = seen.get(label,0)
        seen[label] = cnt + 1
        dlbl = label if cnt == 0 else f"{label}_{cnt+1}"  # deduplicate label

        try:
            resp = requests.post(COMPRESS_URL, json={"text": text}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            r  = data.get("ratio",0.0)
            e  = data.get("entropy",0.0)
            ef = data.get("efficiency",0.0)
            print(f"{dlbl} chars={nch} ratio={r:.2f} ent={e:.2f} eff={ef:.2f}")
        except requests.RequestException as ex:
            print(f"{dlbl} chars={nch} ERROR: {ex}")


if __name__ == "__main__":
    run_benchmark()
