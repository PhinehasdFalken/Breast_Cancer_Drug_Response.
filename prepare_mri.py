"""Build the MRI training data from the BreastDCEDL downloads in data/raw/mri/.

Reads the archives directly, without extracting them, to save disk space, and
saves small tumour-centred images for every I-SPY1 and I-SPY2 patient to
data/raw/mri/processed/. Takes 30-60 minutes, mostly reading the 12 GB archive.
"""
import gzip
import os
import re
import sys
import tarfile
import time
import zipfile

import nibabel as nib
import numpy as np
import pandas as pd

from mri_preprocessing import crop_slices, make_images, tumour_geometry

RAW = "data/raw/mri"
ROLES = ["pre", "post_early", "post_late"]

# "python prepare_mri.py --test" processes only 10 patients per trial, as a quick check
TEST_RUN = "--test" in sys.argv
LIMIT = 10 if TEST_RUN else None
OUT = f"{RAW}/processed_test" if TEST_RUN else f"{RAW}/processed"

meta = pd.read_csv(f"{RAW}/BreastDCEDL_metadata_min_crop.csv")
meta = meta[meta["dataset"].isin(["spy1", "spy2"]) & meta["pCR"].notna()].set_index("pid")
# The dataset's own benchmark split: 0 = training, 2 = validation, 1 = test
meta["split"] = meta["test"].map({0: "train", 2: "val", 1: "test"})


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def load_nifti(raw_bytes):
    """Read a .nii.gz file held in memory."""
    return np.asarray(nib.Nifti1Image.from_bytes(gzip.decompress(raw_bytes)).dataobj)


def phase_role(pid, scan_number):
    """Which of pre / early / late this scan is, from the metadata (or None if unused)."""
    row = meta.loc[pid]
    for role in ROLES:
        if int(row[role]) == scan_number:
            return role
    return None


images, image_pids, problems = [], [], []


def finish_patient(pid, phases, geometry):
    if len(phases) != 3:
        problems.append(f"{pid}: found {sorted(phases)} instead of pre, early and late")
        return
    batch = make_images(phases["pre"], phases["post_early"], phases["post_late"])
    images.append(batch)
    image_pids.extend([pid] * len(batch))


# ---------- I-SPY1: one zip with full scans, tumour masks and metadata ----------
log("Reading I-SPY1")
with zipfile.ZipFile(f"{RAW}/BreastDCEDL_spy1.zip") as archive:
    names = archive.namelist()
    for pid in meta.index[meta["dataset"] == "spy1"][:LIMIT]:
        mask_name = next((n for n in names if n.endswith(f"/{pid}_spy1_vis1_mask.nii.gz")), None)
        if mask_name is None:
            problems.append(f"{pid}: no tumour mask")
            continue
        mask = load_nifti(archive.read(mask_name))
        geometry = tumour_geometry(mask)
        phases = {}
        for name in names:
            found = re.search(rf"/{pid}_spy1_vis1_acq(\d+)\.nii\.gz$", name)
            role = found and phase_role(pid, int(found.group(1)))
            if role:
                volume = load_nifti(archive.read(name))
                if volume.shape != mask.shape:
                    problems.append(f"{pid}: scan and mask sizes differ")
                    break
                phases[role] = crop_slices(volume, *geometry)
        finish_patient(pid, phases, geometry)
log(f"I-SPY1 done: {len(set(image_pids))} patients")

# ---------- I-SPY2: masks first (small archive), then stream the 12 GB scan archive ----------
log("Reading I-SPY2 tumour masks")
geometries, mask_shapes = {}, {}
with tarfile.open(f"{RAW}/spy2_crop_min_mask.tar.gz", "r|gz") as archive:
    for member in archive:
        found = re.search(r"([^/]+)_spy2_vis1_mask\.nii\.gz$", member.name)
        if found and found.group(1) in meta.index:
            mask = load_nifti(archive.extractfile(member).read())
            try:
                geometries[found.group(1)] = tumour_geometry(mask)
                mask_shapes[found.group(1)] = mask.shape
            except ValueError:
                problems.append(f"{found.group(1)}: empty tumour mask")
log(f"{len(geometries)} I-SPY2 masks read")

log("Reading I-SPY2 scans (the slow part; progress is shown every 50 patients)")
current, phases, spy2_count = None, {}, 0
with tarfile.open(f"{RAW}/spy2_crop_min_dce.tar.gz", "r|gz") as archive:
    for member in archive:
        found = re.search(r"([^/]+)_spy2_vis1_dce_aqc_(\d+)\.nii\.gz$", member.name)
        if not found or found.group(1) not in geometries:
            continue
        pid = found.group(1)
        if pid != current:
            if current is not None:
                finish_patient(current, phases, geometries[current])
                spy2_count += 1
                if spy2_count % 50 == 0:
                    log(f"{spy2_count} of {len(geometries)} I-SPY2 patients done")
                if LIMIT and spy2_count >= LIMIT:
                    current = None
                    break
            current, phases = pid, {}
        role = phase_role(pid, int(found.group(2)))
        if role:
            volume = load_nifti(archive.extractfile(member).read())
            if volume.shape != mask_shapes[pid]:
                problems.append(f"{pid}: scan and mask sizes differ")
                continue
            phases[role] = crop_slices(volume, *geometries[pid])
    if current is not None:
        finish_patient(current, phases, geometries[current])

# ---------- Save ----------
images = np.concatenate(images)
image_pids = np.array(image_pids)
done = sorted(set(image_pids))
patients = meta.loc[done, ["dataset", "pCR", "split", "HR", "HER2", "HR_HER2_STATUS", "age"]]
patients["pCR"] = patients["pCR"].astype(int)
patients["n_images"] = pd.Series(image_pids).value_counts().reindex(done).to_numpy()

os.makedirs(OUT, exist_ok=True)
np.savez_compressed(f"{OUT}/mri_images.npz", images=images, pids=image_pids)
patients.to_csv(f"{OUT}/mri_patients.csv", index_label="pid")

log(f"Saved {len(images)} images from {len(done)} patients")
print(patients.groupby(["split", "pCR"]).size().unstack().to_string())
if problems:
    print(f"\n{len(problems)} patients skipped:")
    print("\n".join(problems))
