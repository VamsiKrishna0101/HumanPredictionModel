import cv2
import numpy as np
from io import BytesIO

def read_image_bytes_to_bgr(data: bytes):
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img
