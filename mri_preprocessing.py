"""MRI preprocessing shared by prepare_mri.py (training data) and the app (predictions).

Each patient has three DCE-MRI scans of the breast, taken before the contrast agent
is injected ("pre"), shortly after ("early") and a few minutes later ("late"), plus
a tumour mask marking which voxels are tumour. From these we cut out small square
images centred on the tumour, on the slices where the tumour is largest, and stack
the three scans as the three colour channels of one image. Keeping this in one
module guarantees the app prepares an uploaded scan exactly as the training data was.
"""
import numpy as np
from scipy.ndimage import zoom

IMAGE_SIZE = 128        # pixels per side of each image given to the model
SLICES_PER_PATIENT = 5  # use the 5 slices where the tumour is largest
CROP_MARGIN = 1.5       # crop 1.5x the tumour's size, to include its surroundings
MIN_CROP = 48           # pixels; stops very small tumours being zoomed in too far
CLIP = 1.2              # brightness cap after scaling (see make_images)


def tumour_geometry(mask):
    """From a 3D tumour mask (slices, rows, cols): which slices to use and where to crop.

    Returns the chosen slice numbers, the tumour's centre (row, col) and the crop side
    in pixels. The crop is sized to the tumour, so small and large tumours end up
    filling a similar share of the final image.
    """
    mask = np.asarray(mask) > 0
    area = mask.sum(axis=(1, 2))
    if area.max() == 0:
        raise ValueError("The tumour mask is empty.")
    largest = np.argsort(area)[::-1][:SLICES_PER_PATIENT]
    slices = np.sort(largest[area[largest] > 0])

    rows = np.where(mask.any(axis=(0, 2)))[0]
    cols = np.where(mask.any(axis=(0, 1)))[0]
    centre = ((rows[0] + rows[-1]) / 2, (cols[0] + cols[-1]) / 2)
    side = max(rows[-1] - rows[0] + 1, cols[-1] - cols[0] + 1) * CROP_MARGIN
    side = int(np.clip(side, MIN_CROP, max(mask.shape[1:])))
    return slices, centre, side


def crop_slices(volume, slices, centre, side):
    """Cut the same square around the tumour out of each chosen slice and resize it.

    Parts of the square that fall outside the scan are filled with zeros.
    Returns an array of shape (number of slices, IMAGE_SIZE, IMAGE_SIZE).
    """
    volume = np.asarray(volume, dtype=np.float32)
    top = int(round(centre[0] - side / 2))
    left = int(round(centre[1] - side / 2))
    crops = []
    for z in slices:
        square = np.zeros((side, side), dtype=np.float32)
        image = volume[z]
        r0, r1 = max(top, 0), min(top + side, image.shape[0])
        c0, c1 = max(left, 0), min(left + side, image.shape[1])
        square[r0 - top:r1 - top, c0 - left:c1 - left] = image[r0:r1, c0:c1]
        crops.append(zoom(square, IMAGE_SIZE / side, order=1)[:IMAGE_SIZE, :IMAGE_SIZE])
    return np.stack(crops)


def make_images(pre, early, late):
    """Combine the three cropped scans into 3-channel images (red = pre, green = early, blue = late).

    Different scanners use very different brightness scales, so every patient is divided
    by the brightness of their own early scan (99.5th percentile). After that, the
    images show how strongly and how quickly the tissue took up the contrast agent,
    which is what radiologists look at. Returns uint8 images of shape (slices, 128, 128, 3).
    """
    scale = float(np.percentile(early, 99.5))
    if scale <= 0:
        scale = 1.0
    stacked = np.stack([pre, early, late], axis=-1) / scale
    return (np.clip(stacked, 0, CLIP) / CLIP * 255).astype(np.uint8)


def prepare_patient(pre_volume, early_volume, late_volume, mask_volume):
    """Full preprocessing for one patient's three scans and tumour mask."""
    shapes = {np.shape(v) for v in (pre_volume, early_volume, late_volume, mask_volume)}
    if len(shapes) != 1:
        raise ValueError("The three scans and the tumour mask must all have the same size.")
    slices, centre, side = tumour_geometry(mask_volume)
    pre, early, late = (crop_slices(v, slices, centre, side) for v in (pre_volume, early_volume, late_volume))
    return make_images(pre, early, late)
