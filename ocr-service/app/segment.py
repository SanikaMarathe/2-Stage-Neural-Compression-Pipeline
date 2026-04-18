"""
Digit segmentation: given a PIL Image, find each digit blob and return
a list of 28x28 tensors ready for the CNN.
"""

from PIL import Image
import numpy as np
from scipy.ndimage import label, find_objects
from skimage.filters import threshold_otsu
import torch
from torchvision.transforms.functional import to_tensor


def segment_digits(image: Image.Image) -> list:
    """
    Segment a horizontal digit strip into individual 28x28 tensors.

    Steps:
    1. Convert to grayscale
    2. Convert to numpy array
    3. Otsu threshold to binarize — pixels above threshold are foreground (digits)
       For dark-on-light images, invert so digits are bright on dark background
    4. Label connected components
    5. For each component:
       a. Get bounding box from find_objects
       b. Skip components with area < 20 pixels
       c. Crop the component
       d. Pad to square
       e. Resize to 28x28 using PIL LANCZOS
       f. Normalize to [0,1] float tensor, shape (1, 28, 28)
    6. Sort tensors left-to-right by x-coordinate of bounding box
    7. Return list of tensors
    """
    # Step 1: grayscale
    gray_img = image.convert("L")

    # Step 2: numpy array
    gray_array = np.array(gray_img, dtype=np.float32)

    # Step 3: Otsu threshold — pixels above threshold are foreground
    thresh = threshold_otsu(gray_array)
    binary = gray_array > thresh

    # If the mean pixel value is high (light background), the above-threshold
    # pixels are the lighter background. We want digits as foreground.
    # For dark-on-light: digits are dark (below threshold), so invert binary.
    if gray_array.mean() > 127:
        binary = ~binary

    # Step 4: label connected components
    labeled_array, num_features = label(binary)

    # Step 5: extract each component
    slices = find_objects(labeled_array)

    results = []  # list of (x_start, tensor)

    for i, sl in enumerate(slices):
        if sl is None:
            continue

        row_slice, col_slice = sl

        # Get the component mask and crop
        component_mask = (labeled_array[sl] == (i + 1))

        # Compute area
        area = component_mask.sum()
        if area < 20:
            continue

        # Crop grayscale image to bounding box
        cropped = gray_array[row_slice, col_slice].copy()

        # Zero out pixels not in this component (set to background = 0)
        cropped[~component_mask] = 0.0

        # Step d: pad to square
        h, w = cropped.shape
        if h > w:
            pad_total = h - w
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            cropped = np.pad(cropped, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
        elif w > h:
            pad_total = w - h
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            cropped = np.pad(cropped, ((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=0)

        # Step e: resize to 28x28 using PIL LANCZOS
        pil_crop = Image.fromarray(cropped.astype(np.uint8) if cropped.max() > 1 else (cropped * 255).astype(np.uint8))
        pil_crop = pil_crop.resize((28, 28), Image.LANCZOS)

        # Step f: normalize to [0,1] float tensor shape (1, 28, 28)
        tensor = to_tensor(pil_crop)  # to_tensor converts PIL L image to (1, 28, 28) in [0,1]

        x_start = col_slice.start
        results.append((x_start, tensor))

    # Step 6: sort left-to-right by x-coordinate
    results.sort(key=lambda item: item[0])

    # Step 7: return list of tensors
    return [t for _, t in results]
