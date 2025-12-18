# preprocess_infer.py
import numpy as np
import SimpleITK as sitk
import torch
import tempfile
import os


# ============================================================
# 1. Load .mha from binary bytes (API upload)
# ============================================================
def load_mha_from_bytes(binary_data):
    """
    Load .mha file from raw binary (e.g. UploadFile.read()).
    Returns:
        volume : np.ndarray (D,H,W), float32
        spacing, origin, direction
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mha") as tmp:
        tmp.write(binary_data)
        tmp_path = tmp.name

    img = sitk.ReadImage(tmp_path)
    os.remove(tmp_path)  # clean tmp file

    vol = sitk.GetArrayFromImage(img).astype(np.float32)
    spacing = img.GetSpacing()        # (x,y,z)
    origin = img.GetOrigin()          # (x,y,z)
    direction = img.GetDirection()    # tuple len=9

    return vol, spacing, origin, direction


# ============================================================
# 2. HU normalization
# ============================================================
def normalize_hu(vol, clip=(-1024, 400)):
    vmin, vmax = clip
    vol = np.clip(vol, vmin, vmax)
    mean = vol.mean()
    std = vol.std() if vol.std() > 0 else 1.0
    return (vol - mean) / std


# ============================================================
# 3. Compute voxel index
# ============================================================
def world_to_voxel(coordX, coordY, coordZ, origin, spacing):
    vx = int((coordX - origin[0]) / spacing[0])
    vy = int((coordY - origin[1]) / spacing[1])
    vz = int((coordZ - origin[2]) / spacing[2])
    return vz, vy, vx  # (z, y, x)


# ============================================================
# 4. Crop 3D patch
# ============================================================
def extract_patch_3d(volume, center, size=64):
    zc, yc, xc = center
    half = size // 2

    z1, z2 = zc - half, zc + half
    y1, y2 = yc - half, yc + half
    x1, x2 = xc - half, xc + half

    pad_before = [
        max(0, -z1),
        max(0, -y1),
        max(0, -x1),
    ]
    pad_after = [
        max(0, z2 - volume.shape[0]),
        max(0, y2 - volume.shape[1]),
        max(0, x2 - volume.shape[2]),
    ]

    if any(pad_before) or any(pad_after):
        volume = np.pad(
            volume,
            (
                (pad_before[0], pad_after[0]),
                (pad_before[1], pad_after[1]),
                (pad_before[2], pad_after[2]),
            ),
            mode="constant",
            constant_values=0,
        )

    z1 = max(z1, 0)
    y1 = max(y1, 0)
    x1 = max(x1, 0)

    patch = volume[z1:z1+size, y1:y1+size, x1:x1+size]
    return patch


# ============================================================
# 5. Crop 2D slices for inference
# ============================================================
def extract_slices_2d(patch3d, center_z=None, k=1):
    """
    patch3d: (D, H, W)
    center_z: index z trong patch (nếu None -> lấy giữa)
    k=1 -> lấy [z-1, z, z+1] => 3 slices
    return: (3, H, W)
    """
    D = patch3d.shape[0]
    if center_z is None:
        center_z = D // 2

    zs = [max(0, min(D - 1, center_z + dz)) for dz in range(-k, k + 1)]
    slices = np.stack([patch3d[z] for z in zs], axis=0)  # (3, H, W)
    return slices

# ============================================================
# 5. Main inference preprocessing function
# ============================================================
def preprocess_mha_for_inference(
    binary_data,
    coordX, coordY, coordZ,
    age,
    gender,
    patch_size=64,
    device="cpu"
):
    """
    This function reproduces EXACT training preprocessing.

    Input:
        - binary_data: raw bytes of .mha file
        - coordX, coordY, coordZ: world coordinates from CSV
        - age: int/float
        - gender_str: str ("Male"/"Female")
    
    Return:
        vol_tensor  : (1,1,64,64,64) torch.float32  (ready for model)
        meta_tensor : (1,2) torch.float32          (age, gender_code)
    """

    # 1) Load MHA
    volume, spacing, origin, direction = load_mha_from_bytes(binary_data)

    # 2) Convert world coords → voxel coords
    center = world_to_voxel(coordX, coordY, coordZ, origin, spacing)

    # 3) Crop patch
    patch = extract_patch_3d(volume, center, size=patch_size)

    # 4) Normalize (same as training)
    patch = normalize_hu(patch)

    # 5) Extract 3 slices to feed to model
    img = extract_slices_2d(patch, center_z=patch.shape[0]//2, k=1)  # (3,H,W)

    # 6) Convert to tensor (batch dimension added by API code)
    img = torch.from_numpy(img).float()

    # 7) Meta processing (same as training dataset)
    gender_code = 1.0 if str(gender).lower().startswith("m") else 0.0
    meta_vec = np.array([float(age), gender_code], dtype=np.float32)
    meta_tensor = torch.from_numpy(meta_vec).float().unsqueeze(0)  # (1,2)

    return img.unsqueeze(0).to(device), meta_tensor.to(device)
