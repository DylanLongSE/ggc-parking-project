import json
import time
import os
import requests
from collections import deque
from datetime import datetime
from typing import List, Tuple, Optional


import cv2
import numpy as np
import onnxruntime as ort

# ADDED: Load .env file so config works without systemd's EnvironmentFile.
# Uses env file next to this script with API_BASE_URL, PI_SHARED_SECRET, etc.
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

cv2.setUseOptimized(True)
cv2.setNumThreads(4)

INGESTION_TOKEN = os.getenv("PI_SHARED_SECRET", "").strip()

# Default header uses Authorization: Bearer <token>
TOKEN_HEADER_NAME = os.getenv("TOKEN_HEADER_NAME", "Authorization")
TOKEN_HEADER_PREFIX = os.getenv("TOKEN_HEADER_PREFIX", "Bearer ")  # set "" if not using Bearer

# ---- API config (matches Spring Boot endpoint) ----
API_BASE_URL = os.getenv("API_BASE_URL", "http://100.79.126.126:8080").rstrip("/")
LOT_ID = os.getenv("LOT_ID", "W")
SEND_INTERVAL = float(os.getenv("SEND_INTERVAL", "60"))  # 60 seconds
HEARTBEAT_INTERVAL = float(os.getenv("HEARTBEAT_INTERVAL", "300")) #every 5 mins to check if Pi is alive
CONNECT_TIMEOUT = float(os.getenv("CONNECT_TIMEOUT", "2.5"))
READ_TIMEOUT = float(os.getenv("READ_TIMEOUT", "4.0"))
API_PATH_TEMPLATE = os.getenv("API_PATH_TEMPLATE", "/api/v1/lots/{lotId}/counts")

# Only send during campus hours
SEND_START_HOUR = int(os.getenv("SEND_START_HOUR", "7"))   # 7:00 AM
SEND_END_HOUR   = int(os.getenv("SEND_END_HOUR",   "19"))  # 7:00 PM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def in_send_window() -> bool:
    return SEND_START_HOUR <= datetime.now().hour < SEND_END_HOUR

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
    occupied_count: int,
    occupied_ids: [int],
    lot_id: str,
    api_base_url: str,
    api_path_template: str,
    reason: str,
    ts: Optional[float] = None,
) -> bool:
    """
    Sends JSON like:
      { "occupied": 12, "timestamp": "2026-02-27T00:12:34Z", "source": "pi", "classIds": [2] }.
    """
    if ts is None:
        ts = time.time()

    # ISO8601-ish UTC timestamp
    timestamp_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))

    url = f"{api_base_url}{api_path_template.format(lotId=lot_id)}"
    payload = {
    "occupied": int(occupied_count),
    "timestamp": timestamp_utc,
    "reason": reason,
    "reason": reason,
    "occupied_ids": occupied_ids,
    }
    
    try:
        resp = session.post(url, json=payload, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        if 200 <= resp.status_code < 300:
            print(f"[OK] Sent count={occupied_count} to {url}")
            return True
        else:
            print(f"[HTTP {resp.status_code}] Failed to send to {url}: {resp.text[:200]}")
            return False
    except requests.RequestException as e:
        print(f"[ERR] Send failed to {url}: {e}")
        return False

#-------Detection config--------------
DEFAULT_INPUT_SIZE = (640, 640)
DEFAULT_CONF_THRESH = 0.25 #lower conf threshold better detects cars in low light condi, can increase false pos
DEFAULT_IOU_THRESH = 0.45 #controls how aggresively overlapping detects are removed

# ---- COCO classes for the parking lot
COCO_CAR = 2
COCO_MOTORCYCLE = 3
COCO_TRUCK = 7

VEHICLE_CLASS_IDS = {COCO_CAR, COCO_MOTORCYCLE, COCO_TRUCK}

#--------CAMERA AND WINDOW CONFIG
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
FRAME_W = int(os.getenv("FRAME_W", "640"))
FRAME_H = int(os.getenv("FRAME_H", "360"))

# UI / display
SHOW_WINDOW = os.getenv("SHOW_WINDOW", "1") == "0"
WINDOW = "Parking Occupancy (q=quit)"

def letterbox(im, new_shape=(640, 640), color=(114,114,114)):
	h, w = im.shape[:2]
	nh, nw = new_shape
	scale = min(nw / w, nh / h)
	new_w, new_h = int(round(w * scale)), int(round(h*scale))
	resized = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
	pad_w, pad_h = nw - new_w, nh - new_h
	top, bottom = pad_h // 2, pad_h - pad_h //2
	left, right = pad_w // 2, pad_w - pad_w // 2
	out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
	return out, scale, left, top
	
def bbox_bottom_center_in_poly(xyxy, poly_pts):
    x1, y1, x2, y2 = xyxy
    px = int((x1 + x2) / 2)
    h = y2 - y1
    py = int(y1 + 0.8 * h)
    return cv2.pointPolygonTest(np.array(poly_pts, dtype=np.int32), (px, py), False) >= 0
    
def get_occupied_space_ids(boxes_xyxy, spaces):
    occupied_ids = set()

    for (x1, y1, x2, y2) in boxes_xyxy:        
        # bottom-center of detection box
        px = int((x1+ x2) / 2)
        py = int(y2)

        for s in spaces:
            poly = np.array(s["points"], dtype=np.int32)
            if cv2.pointPolygonTest(poly, (px, py), False) >= 0:
                occupied_ids.add(s["id"])
                break

    return sorted(occupied_ids)
    
class SpaceState:
    def __init__(self, history_len=12):
        self.hist = deque(maxlen=history_len)
        self.occupied = False

    def update(self, yolo_hit):
        self.hist.append(bool(yolo_hit))

        # Hysteresis / smoothing:
        # - become occupied quickly
        # - become free more slowly
        last5 = list(self.hist)[-5:]
        last11 = list(self.hist)[-11:]

        if len(last5) >= 5 and sum(last5) >= 3:         # 3-5 positives -> occupied
            self.occupied = True
        elif (len(last11) == 11) and (sum(last11) <= 1):  # only becomes free after a long stretch of almost no evidence
            self.occupied = False

        return self.occupied
        
def load_spaces(path="spaces.json"):
    with open(path, "r") as f:
        data = json.load(f)
    return data["spaces"]
    
    
def main():
    # --- Load spaces
    spaces = load_spaces("spaces.json")

    # --- Load YOLO ONNX
    model_path = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "yolov8n.onnx"))
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"]) # creates ONNX Runtime inference (predictions) session using the CPU
    in_name = sess.get_inputs()[0].name # gets the input tensor name expected by ONNX model. needed for sess.run

    cap = cv2.VideoCapture(CAMERA_INDEX) # opens webcam
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try index 1.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    session = make_session()  # makes http session for api calls

    # reading dimensions of one frame capture
    ok, frame = cap.read()
    #crashes if frame couldn't be read
    if not ok:
        raise RuntimeError("Could not read initial frame.")
    h0, w0 = frame.shape[:2] # storing original frame dimensions to map detection boxes from letterboxed coords to original frame space

    states = [SpaceState(history_len=12) for _ in spaces] # creates one SpaceState obj per parking space in spaces for occupancy history

    #----frame timing variables
    prev = time.time() # prev frame timestamp
    fps = 0.0 # smoothed fps estimate
    
    # API sending / retry state
    last_sent_at = 0.0
    last_successful_send_at = 0.0
    backoff = 1.0  # seconds (only used after failures)
    max_backoff = 60.0

    # most recent occupied count that gets sent to API
    latest_count = 0
    last_sent_count = None # last successfully sent occupancy
    pending_count = None # changed count not yet delivered

    print("Starting detection + API sender")
    print(f"Model: {model_path}")
    print(f"API: {API_BASE_URL}{API_PATH_TEMPLATE.format(lotId=LOT_ID)}")
    print(f"SEND_INTERVAL: {SEND_INTERVAL}s")
    print("Press 'q' in the preview window to quit." if SHOW_WINDOW else "Running headless (SHOW_WINDOW=0).")
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # ---- YOLO inference
        img, scale, pad_x, pad_y = letterbox(frame, DEFAULT_INPUT_SIZE)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = np.transpose(img_rgb, (2, 0, 1))[None, ...]  # (1,3,640,640)

        preds = sess.run(None, {in_name: inp})[0] # running inference on model and get output tensor
        preds = np.squeeze(preds) # removes extra singleton dimensions from prediction output

        # Ensure shape (N, 84-ish)
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T

        boxes_xyxy = [] # bounding boxes for spaces
        boxes_scores = [] # confidence scores
        boxes_cls = [] # class ids

        for det in preds:
            x, y, bw, bh = det[0:4]
            cls_scores = det[4:]        # YOLOv8 ONNX: class scores directly
            cls_id = int(np.argmax(cls_scores))
            conf = float(cls_scores[cls_id])

            if conf < DEFAULT_CONF_THRESH:
                continue
            if cls_id not in VEHICLE_CLASS_IDS:
                continue

        # map back from letterbox -> original
            x1 = (x - bw/2) - pad_x
            y1 = (y - bh/2) - pad_y
            x2 = (x + bw/2) - pad_x
            y2 = (y + bh/2) - pad_y

            x1 = int(x1 / scale); y1 = int(y1 / scale)
            x2 = int(x2 / scale); y2 = int(y2 / scale)

            x1 = max(0, min(w0 - 1, x1))
            y1 = max(0, min(h0 - 1, y1))
            x2 = max(0, min(w0 - 1, x2))
            y2 = max(0, min(h0 - 1, y2))

            boxes_xyxy.append((x1, y1, x2, y2))
            boxes_scores.append(conf)
            boxes_cls.append(cls_id)

        # NMS (on vehicle boxes)
        # OpenCV wants [x,y,w,h]
        nms_boxes = [[x1, y1, x2-x1, y2-y1] for (x1,y1,x2,y2) in boxes_xyxy]
        # runs non-max suppression to remove overlapping duplicate detections
        keep = cv2.dnn.NMSBoxes(nms_boxes, boxes_scores, score_threshold=0.0, nms_threshold=DEFAULT_IOU_THRESH)
        keep = [] if len(keep) == 0 else keep.flatten().tolist()

        kept_boxes = [boxes_xyxy[i] for i in keep]
        kept_scores = [boxes_scores[i] for i in keep]
    
        occupied_list = get_occupied_space_ids(boxes_xyxy, spaces)
        
        
        #blue detection boxes on every vehicle
        for (x1, y1, x2, y2), score in zip(kept_boxes, kept_scores):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, f"{score:.2f}", (x1, max(20, y1 - 5)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
       
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - prev)) # FPS smoothing 
        prev = now
        
        # ---- Evaluate each space: YOLO + ROI
        for i, s in enumerate(spaces):
            poly = s["points"]

            # YOLO hit: any vehicle box center inside polygon
            yolo_hit = any(bbox_bottom_center_in_poly(b, poly) for b in kept_boxes)
            
            occupied = states[i].update(yolo_hit)

            # Draw polygon (in BGR not RGB)
            pts = np.array(poly, dtype=np.int32)
            color = (0, 0, 255) if occupied else (0, 255, 0)
            cv2.polylines(frame, [pts], True, color, 2)

            # Label with debug info
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            label_x = cx - 20
            label_y = cy - 10
            cv2.putText(frame, f"#{s['id']} {'OCC' if occupied else 'FREE'}",
                        (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 2)
        
        latest_count = sum(1 for st in states if st.occupied)
        
        #queue a changed count for sending
        if latest_count != last_sent_count:
            pending_count = latest_count
            
        # ---- send on change or heartbeat----
        failed_recently = backoff > 1.0
        due_to_retry = failed_recently and (now - last_sent_at) >= backoff
        due_to_heartbeat = (
            last_successful_send_at == 0.0 or (now - last_successful_send_at) >= HEARTBEAT_INTERVAL
        )
        
        # priority is:
        # 1. send pending changed value
        # 2. otherwise, send heartbeat with current count
        send_value = None
        send_reason = None
        
        if pending_count is not None:
            if (not failed_recently) or due_to_retry or last_sent_at == 0.0:
                send_value = pending_count
                send_reason = "change"
        elif due_to_heartbeat:
            if (not failed_recently) or due_to_retry or last_sent_at == 0.0:
                send_value = latest_count
                send_reason = "heartbeat"
                
        if send_value is not None and not in_send_window():
            print("[SKIP] Outside send window (7am-7pm), not sending to DB")
            send_value = None

        if send_value is not None:
            success = send_count(
                session=session,
                occupied_count=send_value,
                occupied_ids = occupied_list,
                lot_id=LOT_ID,
                api_base_url=API_BASE_URL,
                api_path_template=API_PATH_TEMPLATE,
                reason=send_reason,
                ts=now,
            )
            last_sent_at = now
            
            if success:
                last_successful_send_at = now
                last_sent_count = send_value
                backoff = 1.0
                
                if pending_count == send_value:
                    pending_count = None
                    
                print(f"[OK] Send reason: {send_reason}")
            else:
                backoff = min(max_backoff, backoff * 2.0)
                print(f"[WARN] Will retry with backoff={backoff:.1f}s")
                

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        k = 255
        if SHOW_WINDOW:
            cv2.imshow(WINDOW, frame)
            k = cv2.waitKey(1) & 0xFF
                
        

        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
