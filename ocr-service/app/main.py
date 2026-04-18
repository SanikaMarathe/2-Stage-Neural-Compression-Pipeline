from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io, json, os, sys, time

sys.path.insert(0, os.path.dirname(__file__))  # so local imports work

import torch
from PIL import Image, UnidentifiedImageError
from model import DigitCNN, NUM_CLASSES
from segment import segment_digits

app = FastAPI(title="OCR Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080","http://127.0.0.1:8080"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MDL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model.pt')
MET_PATH = os.path.join(os.path.dirname(__file__), '..', 'eval_metrics.json')

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # pick gpu if available

model = DigitCNN().to(device)
mdl_ok = False  # model ready

if os.path.exists(MDL_PATH):
    model.load_state_dict(torch.load(MDL_PATH, map_location=device))  # load saved weights
    model.eval()
    mdl_ok = True


class OCRResponse(BaseModel):
    text: str
    per_digit_confidence: list[float]
    digit_count: int
    latency_ms: float


@app.get("/healthz")
def health():
    return {"status": "ok", "model_loaded": mdl_ok}


@app.post("/ocr", response_model=OCRResponse)
async def ocr(file: UploadFile = File(...)):
    if not mdl_ok:
        raise HTTPException(status_code=503, detail="Model not trained yet. Run train.py first.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except (UnidentifiedImageError, Exception) as e:
        raise HTTPException(status_code=422, detail=f"Invalid image file: {str(e)}")

    t0 = time.time()
    dig_tens = segment_digits(image)  # list of (1,28,28) tensors

    if not dig_tens:
        lat = (time.time() - t0) * 1000
        return OCRResponse(text="", per_digit_confidence=[], digit_count=0, latency_ms=round(lat,2))

    batch = torch.stack(dig_tens).to(device)  # (N,1,28,28)

    # run all digits in a single forward pass
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        pred_cls = probs.argmax(dim=1)       # highest prob class per digit
        confs = probs.max(dim=1).values      # confidence score per digit

    lat = (time.time() - t0) * 1000
    pred_digs = pred_cls.cpu().tolist()  # pull off gpu
    conf_list = [round(float(c),4) for c in confs.cpu().tolist()]
    text = " ".join(str(d) for d in pred_digs)

    return OCRResponse(
        text=text,
        per_digit_confidence=conf_list,
        digit_count=len(pred_digs),
        latency_ms=round(lat,2),
    )


@app.get("/metrics")
def metrics():
    if not os.path.exists(MET_PATH):
        return {"error": "not trained yet"}
    with open(MET_PATH,"r") as f:
        return json.load(f)


@app.get("/test-sample")
def test_sample(n: int = 8):
    import random, base64
    import numpy as np
    from torchvision import datasets, transforms
    from PIL import Image as PILImage

    data_root = os.path.join(os.path.dirname(__file__), '..', 'data')
    v_ds = datasets.MNIST(
        root=data_root, train=False, download=False,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    idxs = random.sample(range(len(v_ds)), min(n,len(v_ds)))
    dsize=28  # mnist native size
    gap=8
    sw = dsize*len(idxs) + gap*(len(idxs)-1)  # strip width
    strip = PILImage.new('L', (sw,dsize), color=0)  # blank black canvas

    lbls = []
    for i,idx in enumerate(idxs):
        img_t,label = v_ds[idx]
        arr = (img_t.squeeze().numpy() * 255).astype(np.uint8)
        dimg = PILImage.fromarray(arr)
        strip.paste(dimg, (i*(dsize+gap),0))  # place digit at its x offset
        lbls.append(int(label))

    buf = io.BytesIO()
    strip.save(buf, format='PNG')
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    return {"image_b64": img_b64, "ground_truth": " ".join(str(l) for l in lbls)}


@app.get("/live-accuracy")
def live_accuracy():
    if not mdl_ok:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from augment import apply_gaussian, apply_salt_and_pepper

    data_root = os.path.join(os.path.dirname(__file__), '..', 'data')
    tfm = transforms.Compose([transforms.ToTensor()])
    v_ds = datasets.MNIST(root=data_root, train=False, download=False, transform=tfm)
    loader = DataLoader(v_ds, batch_size=500, shuffle=True)
    x,y = next(iter(loader))  # grab one batch

    model.eval()
    with torch.no_grad():
        yd = y.to(device)
        acc_c  = (model(x.to(device)).argmax(dim=1) == yd).float().mean().item()
        acc_g  = (model(apply_gaussian(x).to(device)).argmax(dim=1) == yd).float().mean().item()
        acc_sp = (model(apply_salt_and_pepper(x).to(device)).argmax(dim=1) == yd).float().mean().item()

    return {
        "accuracy_clean": round(acc_c,4),
        "accuracy_gaussian": round(acc_g,4),
        "accuracy_salt_and_pepper": round(acc_sp,4),
    }
