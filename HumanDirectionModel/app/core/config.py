from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    POSE_MODEL: str = "yolov8n-pose-tiny.pt"
    PERSON_CONF_THR: float = 0.35
    KEYPOINT_CONF_THR: float = 0.25
    FULLBODY_MIN_VISIBLE_KPTS: int = 12
    FULLBODY_REQUIRE_FEET: bool = True
    FULLBODY_MIN_BBOX_COVERAGE: float = 0.55

settings = Settings()
