import SimpleITK as sitk
import tempfile
import numpy as np
import os

def load_mha(binary_data):
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mha") as tmp:
        tmp.write(binary_data)
        tmp_path = tmp.name

    try:
        img = sitk.ReadImage(tmp_path)
        arr = sitk.GetArrayFromImage(img)

        spacing = img.GetSpacing()
        origin = img.GetOrigin()
        direction = img.GetDirection()

        return arr, spacing, origin, direction

    except Exception as e:
        raise e

    finally:
        # Clean up temporary file
        os.remove(tmp_path)
