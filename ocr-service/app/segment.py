from PIL import Image
import numpy as np
from scipy.ndimage import label, find_objects
from skimage.filters import threshold_otsu
import torch
from torchvision.transforms.functional import to_tensor


def segment_digits(image: Image.Image) -> list:
    g_img = image.convert("L")  # to grayscale
    g_arr = np.array(g_img, dtype=np.float32)

    thresh = threshold_otsu(g_arr)
    binary = g_arr > thresh  # pixels above threshold = foreground

    if g_arr.mean() > 127:  # dark-on-light, invert so digits are foreground
        binary = ~binary

    lbl_arr,n_feat = label(binary)  # connected components
    slices = find_objects(lbl_arr)  # bounding box per component

    res = []  # (x_start, tensor)

    for i,sl in enumerate(slices):
        if sl is None:
            continue

        r_sl,c_sl = sl
        comp_mask = (lbl_arr[sl] == (i+1))  # isolate this blob

        area = comp_mask.sum()
        if area < 20:  # skip tiny noise
            continue

        crop = g_arr[r_sl,c_sl].copy()
        crop[~comp_mask] = 0.0  # zero out other blobs in this box

        # pad shorter side to make it square
        h,w = crop.shape
        if h > w:
            ptot = h - w
            pl = ptot // 2
            pr = ptot - pl
            crop = np.pad(crop, ((0,0),(pl,pr)), mode='constant', constant_values=0)
        elif w > h:
            ptot = w - h
            pt = ptot // 2
            pb = ptot - pt
            crop = np.pad(crop, ((pt,pb),(0,0)), mode='constant', constant_values=0)

        pc = Image.fromarray(crop.astype(np.uint8) if crop.max() > 1 else (crop*255).astype(np.uint8))
        pc = pc.resize((28,28), Image.LANCZOS)  # down to 28x28
        tensor = to_tensor(pc)  # (1,28,28) float [0,1]

        xs = c_sl.start
        res.append((xs,tensor))

    res.sort(key=lambda item: item[0])  # left to right
    return [t for _,t in res]
