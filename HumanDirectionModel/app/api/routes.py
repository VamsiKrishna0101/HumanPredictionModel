from fastapi import APIRouter, UploadFile, File, HTTPException
from app.core.logger import get_logger
from app.model.pose import PoseEstimator
from app.services.classify import classify_orientation
from app.schemas.predict import PredictResponse
from app.utils.images import read_image_bytes_to_bgr

router = APIRouter()
log = get_logger(__name__)
pose = PoseEstimator("yolov8n-pose-tiny.pt")

@router.get("/health")
def health():
    return {"status": "ok"}

@router.post("/predict", response_model=PredictResponse)
async def predict(image: UploadFile = File(...)):
    try:
        data = await image.read()
        img = read_image_bytes_to_bgr(data)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image error during processing into bytes")

        ok, bbox, kpts = pose.detect_single_person(img)
        if not ok:
            return PredictResponse(label="N/A", meta={"reason": "zero or multiple persons found"})

        label = classify_orientation(kpts, bbox, img.shape)
        meta = {
            "bbox": bbox,
            "image_shape": img.shape,
            "kpts_visible": int((kpts[:,2] > 0.25).sum()) if kpts is not None else 0
        }
        return PredictResponse(label=label, meta=meta)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
