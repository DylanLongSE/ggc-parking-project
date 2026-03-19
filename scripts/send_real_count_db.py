"""
vehicle_detect_and_send.py

- Runs YOLOv8 ONNX vehicle detection (OpenCV DNN) on a webcam feed
- Periodically POSTs the detected vehicle count to a Spring Boot API
- Keeps running even if the API is temporarily down (retry/backoff)

Install deps:
  pip install opencv-python numpy requests

Run:
  python vehicle_detect_and_send.py

Configure (env vars optional):
  API_BASE_URL   e.g. http://100.79.126.126:8080
  LOT_ID         e.g. W
  SEND_INTERVAL  seconds, e.g. 60
  CAMERA_INDEX   e.g. 0
  MODEL_PATH     override path to yolov8n.onnx
"""

import cv2
import os
import json
import time
import numpy as np
import requests
from typing import List, Tuple, Optional

# ADDED: Load .env file so config works without systemd's EnvironmentFile.
# Place a .env file next to this script with API_BASE_URL, PI_SHARED_SECRET, etc.
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

cv2.setUseOptimized(True)
cv2.setNumThreads(4)

# COCO class IDs: person=0, bicycle=1, car=2, motorcycle=3, bus=5, truck=7
# If you want multiple vehicle types, use ITEM_CLASS_IDS = {2, 3, 5, 7}
ITEM_CLASS_IDS = {2}  # default: car only

DEFAULT_INPUT_SIZE = (640, 640)
DEFAULT_CONF_THRESH = 0.20
DEFAULT_IOU_THRESH = 0.45

INGESTION_TOKEN = os.getenv("PI_SHARED_SECRET", "").strip()

# Choose ONE header style (default uses Authorization: Bearer ...)
TOKEN_HEADER_NAME = os.getenv("TOKEN_HEADER_NAME", "Authorization")
TOKEN_HEADER_PREFIX = os.getenv("TOKEN_HEADER_PREFIX", "Bearer ")  # set "" if not using Bearer

# ---- API config (adjust to match your Spring Boot endpoint) ----
API_BASE_URL = os.getenv("API_BASE_URL", "http://100.79.126.126:8080").rstrip("/")
LOT_ID = os.getenv("LOT_ID", "W")
SEND_INTERVAL = float(os.getenv("SEND_INTERVAL", "60"))  # seconds
CONNECT_TIMEOUT = float(os.getenv("CONNECT_TIMEOUT", "2.5"))
READ_TIMEOUT = float(os.getenv("READ_TIMEOUT", "4.0"))

# Your earlier example: /api/v1/lots/W/counts
# If yours is different, change this.
API_PATH_TEMPLATE = os.getenv("API_PATH_TEMPLATE", "/api/v1/lots/{lotId}/counts")

# Camera config
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
FRAME_W = int(os.getenv("FRAME_W", "640"))
FRAME_H = int(os.getenv("FRAME_H", "360"))

# UI / display
SHOW_WINDOW = os.getenv("SHOW_WINDOW", "1") == "1"
SHOW_EVERY = int(os.getenv("SHOW_EVERY", "3"))  # draw every N frames


def letterbox(im: np.ndarray, new_shape: Tuple[int, int] = (640, 640), color=(114, 114, 114)):
    h, w = im.shape[:2]
    nh, nw = new_shape
    scale = min(nw / w, nh / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w, pad_h = nw - new_w, nh - new_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return out, scale, left, top
 
def nms_xywh(boxes: List[List[int]], scores: List[float], iou_thresh: float = 0.45) -> List[int]:
    idxs = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=iou_thresh)
    if len(idxs) == 0:
        return []
    return idxs.flatten().tolist()


def build_model_path() -> str:
    # Keeps your original relative structure: scripts/vehicle_detect_and_send.py -> ../models/yolov8n.onnx
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.abspath(os.path.join(base_dir, "..", "models", "yolov8n.onnx"))
    return os.getenv("MODEL_PATH", default_path)


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "Content-Type": "application/json",
            "Accept": "application/json",
            # Optional: identify your Pi in server logs
            "User-Agent": "ggc-parking-pi/1.0",
        }
    )
    if INGESTION_TOKEN:
        s.headers[TOKEN_HEADER_NAME] = f"{TOKEN_HEADER_PREFIX}{INGESTION_TOKEN}"
    else:
        print("[WARN] INGESTION_TOKEN is empty. Requests will likely 401.")
    return s
    
def send_count(
    session: requests.Session,
    car_count: int,
    lot_id: str,
    api_base_url: str,
    api_path_template: str,
    ts: Optional[float] = None,
) -> bool:
    """
    Sends JSON like:
      { "count": 12, "timestamp": "2026-02-27T00:12:34Z", "source": "pi", "classIds": [2] }

    If your Spring Boot expects a different payload, edit `payload`.
    """
    if ts is None:
        ts = time.time()

    # ISO8601-ish UTC timestamp
    timestamp_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))

    url = f"{api_base_url}{api_path_template.format(lotId=lot_id)}"
    payload = {
    "occupied": int(car_count),
    "timestamp": timestamp_utc,
    }
    
    try:
        resp = session.post(url, json=payload, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        if 200 <= resp.status_code < 300:
            print(f"[OK] Sent count={car_count} to {url}")
            return True
        else:
            print(f"[HTTP {resp.status_code}] Failed to send to {url}: {resp.text[:200]}")
            return False
    except requests.RequestException as e:
        print(f"[ERR] Send failed to {url}: {e}")
        return False
        
def main():
    model_path = build_model_path()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model not found at: {model_path}")

    # Load ONNX once (your original script loaded twice by mistake)
    net = cv2.dnn.readNetFromONNX(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam on index {CAMERA_INDEX}. Try CAMERA_INDEX=1")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    input_size = DEFAULT_INPUT_SIZE
    conf_thresh = DEFAULT_CONF_THRESH
    iou_thresh = DEFAULT_IOU_THRESH

    session = make_session()
    
    prev_time = time.time()
    fps = 0.0
    frame_id = 0

    # Sending cadence / retry state
    last_sent_at = 0.0
    backoff = 1.0  # seconds (only used after failures)
    max_backoff = 60.0

    # Welll send the most recent computed count when it's time to send
    latest_count = 0

    print("Starting detection + API sender")
    print(f"Model: {model_path}")
    print(f"API: {API_BASE_URL}{API_PATH_TEMPLATE.format(lotId=LOT_ID)}")
    print(f"SEND_INTERVAL: {SEND_INTERVAL}s, CAMERA_INDEX: {CAMERA_INDEX}")
    print("Press 'q' in the preview window to quit." if SHOW_WINDOW else "Running headless (SHOW_WINDOW=0).")
    print("using token:", INGESTION_TOKEN)
    
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Camera read failed; retrying...")
            time.sleep(0.2)
            continue
        frame_id += 1
        h0, w0 = frame.shape[:2]

        img, scale, pad_x, pad_y = letterbox(frame, input_size)
        blob = cv2.dnn.blobFromImage(
            img, scalefactor=1 / 255.0, size=input_size, swapRB=True, crop=False
        )

        net.setInput(blob)
        preds = net.forward()  # typically (1, 84, 8400) for COCO YOLOv8n

        preds = np.squeeze(preds)
        if preds.ndim != 2:
            preds = preds.reshape(preds.shape[0], -1)
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T  # -> (8400, 84)

        boxes: List[List[int]] = []
        scores: List[float] = []

        for det in preds:
            x, y, bw, bh = det[0:4]
            cls_scores = det[4:]
            cls_id = int(np.argmax(cls_scores))
            conf = float(cls_scores[cls_id])
            
            if cls_id not in ITEM_CLASS_IDS:
                continue
            if conf < conf_thresh:
                continue

            # letterboxed -> original coords
            x1 = (x - bw / 2) - pad_x
            y1 = (y - bh / 2) - pad_y
            x2 = (x + bw / 2) - pad_x
            y2 = (y + bh / 2) - pad_y

            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)

            x1 = max(0, min(w0 - 1, x1))
            y1 = max(0, min(h0 - 1, y1))
            x2 = max(0, min(w0 - 1, x2))
            y2 = max(0, min(h0 - 1, y2))

            boxes.append([x1, y1, x2 - x1, y2 - y1])  # xywh
            scores.append(conf)

        keep = nms_xywh(boxes, scores, iou_thresh=iou_thresh) if boxes else []
        latest_count = len(keep)
        
        # FPS smoothing
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - prev_time))
        prev_time = now

        # ---- periodic send ----
        due_to_send = (now - last_sent_at) >= SEND_INTERVAL
        failed_recently = backoff > 1.0
        due_to_retry = failed_recently and (now - last_sent_at) >= backoff

        if due_to_send or due_to_retry:
            success = send_count(
                session=session,
                car_count=latest_count,
                lot_id=LOT_ID,
                api_base_url=API_BASE_URL,
                api_path_template=API_PATH_TEMPLATE,
                ts=now,
            )
            last_sent_at = now
            if success:
                backoff = 1.0
            else:
                backoff = min(max_backoff, backoff * 2.0)
                print(f"[WARN] Will retry with backoff={backoff:.1f}s")
		# ---- optional preview window ----
        if SHOW_WINDOW and (frame_id % SHOW_EVERY == 0):
            for i in keep:
                x, y, bw, bh = boxes[i]
                conf = scores[i]
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"vehicle {conf:.2f}",
                    (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Cars: {latest_count}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("YOLOv8 ONNX Vehicle Detection (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
                
