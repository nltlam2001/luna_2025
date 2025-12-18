from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from utils.preprocess import preprocess_mha_for_inference
from models.model import ConvNeXtTiny_2D_Meta_CBAM, load_flexible_state_dict

from utils.error_codes import (
    ERR_INVALID_FILE_FORMAT,
    ERR_PROCESSING_ERROR,
    ERR_INTERNAL_SERVER_ERROR,
    ERR_GATEWAY_TIMEOUT,
)
from middlewares.auth import auth_middleware
from utils.logger import setup_logger

logger = setup_logger(
    name="api",
    log_dir="logs/api",
)

logger.info("Inference service started")

import time
import torch
import traceback
import numpy as np

PRETRAIN_CKPT = "models/best_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# meta_mean = np.load("models/meta_mean.npy")  # shape (17,)
# meta_std = np.load("models/meta_std.npy") 

app = FastAPI(title="CT Lesion Classification Service")

# Apply middleware
app.middleware("http")(auth_middleware)

# Load model…
model = ConvNeXtTiny_2D_Meta_CBAM(
    meta_dim=2,
    in_channels=3,
    pretrained=True,
).to(device)

# load MedicalNet pretrained backbone
model = load_flexible_state_dict(model, PRETRAIN_CKPT, device)
model = model.to(device)
model.eval()

# ---------- GLOBAL TIMEOUT CHECK (600 seconds) ----------
MAX_TIME_SEC = 600

@app.post("/api/v1/predict/lesion")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    seriesInstanceUID: str = Form(...),
    patientID: str = Form(...),
    studyDate: str = Form(...),
    lesionID: int = Form(...),
    coordX: float = Form(...),
    coordY: float = Form(...),
    coordZ: float = Form(...),
    ageAtStudyDate: int = Form(...),
    gender: str = Form(...)
):
    start = time.time()

    try:
        # 1) Check định dạng file
        if not (file.filename.endswith(".mha") or file.filename.endswith(".mhd")):
            return ERR_INVALID_FILE_FORMAT()

        binary = await file.read()

        # 2) Preprocess giống Dataset
        try:
            vol_tensor, meta_tensor = preprocess_mha_for_inference(
                binary_data=binary,
                coordX=coordX,
                coordY=coordY,
                coordZ=coordZ,
                age=ageAtStudyDate,
                gender=gender,
                device=device
            )
        except Exception as e:
            logger.error("Preprocess error:", e)
            return ERR_PROCESSING_ERROR("Lỗi khi tiền xử lý ảnh/metadata.")

        # 3) Inference
        try:
            with torch.no_grad():
                logits = model(vol_tensor, meta_tensor)
                prob = torch.sigmoid(logits).item()
                logger.info(f'Probability: {prob}')
        except Exception as e:
            logger.error("Model error:", e)
            import traceback
            traceback.print_exc()
            return ERR_PROCESSING_ERROR("Lỗi nội tại khi chạy mô hình.")
        
        elapsed = time.time() - start
        if elapsed > 600:
            return ERR_GATEWAY_TIMEOUT()

        return JSONResponse({
            "status": "success",
            "data": {
                "seriesInstanceUID": seriesInstanceUID,
                "lesionID": lesionID,
                "probability": round(prob, 4),
                "predictionLabel": 1 if prob >= 0.2828 else 0,
                "processingTimeMs": int(elapsed * 1000),
            }
        })

    except Exception as e:
        logger.error("Unexpected server error:", e)
        import traceback
        traceback.print_exc()
        return ERR_INTERNAL_SERVER_ERROR()