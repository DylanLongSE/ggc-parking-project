import json
import time
import os
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- COCO classes for the parking lot
COCO_CAR = 2
COCO_MOTORCYCLE = 3
COCO_TRUCK = 7

VEHICLE_CLASS_IDS = {COCO_CAR, COCO_MOTORCYCLE, COCO_TRUCK}

WINDOW = "Parking Occupancy (b=baseline, q=quit)"

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
	
def bbox_center_in_poly(xyxy, poly_pts):
	x1, y1, x2, y2, = xyxy
	cx, cy = int((x1+x2)/2), int((y1+y2)/2)
	return cv2.pointPolygonTest(np.array(poly_pts, dtype=np.int32), (cx, cy), False) >= 0
	
def make_mask(shape_hw, poly_pts):
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(poly_pts, dtype=np.int32)], 255)
    return mask

def roi_occupied(frame_gray, baseline_gray, mask, diff_thresh=25, frac_thresh=0.08):
    # abs diff inside ROI
    diff = cv2.absdiff(frame_gray, baseline_gray)
    diff_roi = cv2.bitwise_and(diff, diff, mask=mask)
    
 # threshold changed pixels
    _, changed = cv2.threshold(diff_roi, diff_thresh, 255, cv2.THRESH_BINARY)

    changed_px = cv2.countNonZero(changed)
    area_px = cv2.countNonZero(mask)
    frac = changed_px / max(area_px, 1)

    return (frac > frac_thresh), frac
    
class SpaceState:
    def __init__(self, history_len=10):
        self.hist = deque(maxlen=history_len)
        self.occupied = False

    def update(self, fused_hit):
        self.hist.append(bool(fused_hit))

        # Hysteresis / smoothing:
        # - become occupied quickly
        # - become free more slowly
        last5 = list(self.hist)[-5:]
        last7 = list(self.hist)[-7:]

        if sum(last5) >= 3:         # 3/5 positives -> occupied
            self.occupied = True
        elif (len(last7) == 7) and (sum(last7) <= 1):  # 0-1/7 positives -> free
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
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try index 1.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    input_size = (640, 640)
    conf_thresh = 0.25
    iou_thresh = 0.45  # NMS threshold

    # Build masks once (after we know frame size)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read initial frame.")
    h0, w0 = frame.shape[:2]

    masks = []
    for s in spaces:
        masks.append(make_mask((h0, w0), s["points"]))

    states = [SpaceState(history_len=12) for _ in spaces]

    baseline_gray = None

    prev = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
           
        # grayscale for ROI method
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # default baseline if not set: use first frame (not ideal, but avoids crash)
        if baseline_gray is None:
            baseline_gray = frame_gray.copy()

        # ---- YOLO inference
        img, scale, pad_x, pad_y = letterbox(frame, input_size)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = np.transpose(img_rgb, (2, 0, 1))[None, ...]  # (1,3,640,640)

        preds = sess.run(None, {in_name: inp})[0]
        preds = np.squeeze(preds)

        # Ensure shape (N, 84-ish)
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T

        boxes_xyxy = []
        boxes_scores = []
        boxes_cls = []

        for det in preds:
            x, y, bw, bh = det[0:4]
            cls_scores = det[4:]        # YOLOv8 ONNX: class scores directly
            cls_id = int(np.argmax(cls_scores))
            conf = float(cls_scores[cls_id])

            if conf < conf_thresh:
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
        keep = cv2.dnn.NMSBoxes(nms_boxes, boxes_scores, score_threshold=0.0, nms_threshold=iou_thresh)
        keep = [] if len(keep) == 0 else keep.flatten().tolist()

        kept_boxes = [boxes_xyxy[i] for i in keep]
        kept_scores = [boxes_scores[i] for i in keep]
        
        
         # ---- Evaluate each space: YOLO + ROI
        for i, s in enumerate(spaces):
            poly = s["points"]
            mask = masks[i]

            # YOLO hit: any vehicle box center inside polygon
            yolo_hit = any(bbox_center_in_poly(b, poly) for b in kept_boxes)

            # ROI hit: difference vs baseline inside polygon
            roi_hit, roi_frac = roi_occupied(
                frame_gray,
                baseline_gray,
                mask,
                diff_thresh=25,     # tune
                frac_thresh=0.08    # tune
            )

            # Fuse
            fused = yolo_hit or roi_hit
            occupied = states[i].update(fused)

            # Draw polygon
            pts = np.array(poly, dtype=np.int32)
            color = (0, 0, 255) if occupied else (0, 255, 0)
            cv2.polylines(frame, [pts], True, color, 2)

            # Label with debug info
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            cv2.putText(frame, f"#{s['id']} {'OCC' if occupied else 'FREE'}",
                        (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.putText(frame, f"Y:{int(yolo_hit)} R:{int(roi_hit)} ({roi_frac:.2f})",
                        (cx, cy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

			# Draw vehicle boxes (optional)
        for (x1, y1, x2, y2), conf in zip(kept_boxes, kept_scores):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, f"veh {conf:.2f}", (x1, max(0, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

        # FPS
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - prev))
        prev = now
        cv2.putText(frame, f"FPS: {fps:.1f}  (press b to set baseline)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow(WINDOW, frame)
        k = cv2.waitKey(1) & 0xFF

        if k == ord('q'):
            break
        elif k == ord('b'):
            baseline_gray = frame_gray.copy()
            print("Baseline captured (b).")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
