from ultralytics import YOLO
import numpy as np
from app.core.config import settings

class PoseEstimator:
    def __init__(self, model_name: str | None = None):
        self.model = YOLO(model_name or settings.POSE_MODEL)

    def detect_single_person(self, image_bgr: np.ndarray):
        results = self.model(image_bgr, verbose=False)
        if not results or len(results) == 0:
            return False, None, None

        r = results[0]
        if r.boxes is None or r.keypoints is None:
            return False, None, None

        dets = []
        for i, box in enumerate(r.boxes):
            cls = int(box.cls.item()) if box.cls is not None else -1
            conf = float(box.conf.item()) if box.conf is not None else 0.0
            if cls == 0 and conf >= settings.PERSON_CONF_THR:
                dets.append((i, conf))

        if len(dets) != 1:
            return False, None, None

        idx = dets[0][0]
        bbox = r.boxes.xyxy[idx].cpu().numpy().tolist()
        kpts = r.keypoints.xy[idx].cpu().numpy()
        kconf = r.keypoints.conf[idx].cpu().numpy()
        kpts = np.concatenate([kpts, kconf[:, None]], axis=1)
        return True, bbox, kpts
