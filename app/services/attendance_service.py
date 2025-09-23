import logging
from app.config import db
import base64
import cv2
import torch
import requests
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

unknown_notify_url = os.getenv("THREAT_API", None)

def frame_to_base64(frame, target_size=(640, 480)):
    if frame is None or (hasattr(frame, 'size') and frame.size == 0):
        raise ValueError("Input frame is empty or None")

    if hasattr(frame, "to_ndarray"):
        try:
            frame = frame.to_ndarray(format="bgr24")
        except Exception:
            frame = frame.to_ndarray()  

    if isinstance(frame, torch.Tensor):
        if frame.ndim == 3:
            if frame.shape[0] <= 4:
                frame = frame.permute(1, 2, 0)
        frame = frame.contiguous().cpu().numpy()

    frame = np.asarray(frame)

    print(f"[DEBUG] Input frame shape: {frame.shape}, dtype: {frame.dtype}")

    if frame.ndim == 2:

        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3:
        channels = frame.shape[2]
        if channels == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif channels == 2:
            frame = frame[..., 0]  # drop 2nd channel
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif channels == 3:
            pass 
        elif channels == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        else:
            raise ValueError(f"Unsupported channel count: {channels}")
    else:
        raise ValueError(f"Unsupported frame dimensions: {frame.shape}")

    frame = cv2.resize(frame, target_size)


    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    print(f"[DEBUG] Final frame shape: {frame.shape}, dtype: {frame.dtype}")

    success, encoded_img = cv2.imencode(".jpg", frame)
    if not success:
        raise ValueError("Image encoding failed")

    return base64.b64encode(encoded_img).decode("utf-8")


def mark_attendance(
    unique_user_id,
    camera_id,
    company_id=None,
    frame=None,
):
    try:
        try:
            user_oid = str(unique_user_id)
        except Exception:
            return None, "Invalid user ID format"
        user = db.users.find_one({"unique_user_id": unique_user_id})
        if not user:
            return None, "User not found"
          
        b64_string = frame_to_base64(frame=frame)   
        if b64_string == None:
            logging.error("Frame BASE 64 FAILS")           
        payload = {
            "camera_id": camera_id,
            "company_id": user["company_id"],
            "user_id": user["company_user_id"],
            "file": b64_string,
            "type": "known",
        }
        resp = requests.post(
            unknown_notify_url,
            json=payload,
        )
        if resp.status_code == 200:
            return True, None
        else:
            return None, "SIR API ERROR"
    except Exception as e:
        logging.error(f"Error in mark_attendance: {e}")
        return None, "Internal server error"
