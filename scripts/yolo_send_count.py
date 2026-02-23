import time
import os
import requests
from datetime import datetime, timezone

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
API_BASE = "http://100.71.153.22:8080"
LOT_ID = "W"
ITEM_CLASS_ID = 2  # COCO car = 2

cv2.setUseOptimized(True)
cv2.setNumThreads(4)


def post_count(count: int):
    url = f"{API_BASE}/api/v1/lots/{LOT_ID}/counts"
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "carCount": int(count),
        "sourceDeviceId": "pi5-cam-01",
        "avgConfidence": 0.75,
    }
    r = requests.post(url, json=payload, timeout=5)
    r.raise_for_status()
    print("POST ok:", payload)


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = im.shape[:2]
    nh, nw = new_shape
    scale = min(nw / w, nh / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w, pad_h = nw - new_w, nh - new_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    out = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return out, scale, left, top


def nms_xywh(boxes, scores, iou_thresh=0.45):
    idxs = cv2.dnn.NMSBoxes(
        boxes, scores, score_threshold=0.0, nms_threshold=iou_thresh
    )
    if len(idxs) == 0:
        return []
    return idxs.flatten().tolist()


def main():
    model_path = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "yolov8n.onnx"))
    net = cv2.dnn.readNetFromONNX(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam on index 0. Try index 1.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    input_size = (640, 640)
    conf_thresh = 0.20
    iou_thresh = 0.45

    # UI / perf knobs
    show_every = 3
    frame_id = 0

    # FPS
    prev = time.time()
    fps = 0.0

    # POST throttling
    post_interval_s = 10
    last_post_t = 0.0
    last_sent_count = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1

        h0, w0 = frame.shape[:2]

        img, scale, pad_x, pad_y = letterbox(frame, input_size)
        blob = cv2.dnn.blobFromImage(
            img, scalefactor=1 / 255.0, size=input_size, swapRB=True, crop=False
        )

        net.setInput(blob)
        preds = net.forward()

        preds = np.squeeze(preds)
        if preds.ndim != 2:
            preds = preds.reshape(preds.shape[0], -1)
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T

        boxes = []
        scores = []

        for det in preds:
            x, y, bw, bh = det[0:4]
            cls_scores = det[4:]
            cls_id = int(np.argmax(cls_scores))
            conf = float(cls_scores[cls_id])

            if cls_id != ITEM_CLASS_ID or conf < conf_thresh:
                continue

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

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(conf)

        keep = nms_xywh(boxes, scores, iou_thresh=iou_thresh) if boxes else []
        car_count = len(keep)

        # POST every N seconds, and only if changed (optional but nice)
        now_t = time.time()
        if (now_t - last_post_t) >= post_interval_s:
            if last_sent_count is None or car_count != last_sent_count:
                try:
                    post_count(car_count)
                    last_sent_count = car_count
                except Exception as e:
                    print("POST failed:", e)
            last_post_t = now_t

        # FPS calc
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now_t - prev))
        prev = now_t

        # UI
        draw = frame_id % show_every == 0
        if draw:
            for i in keep:
                x, y, bw, bh = boxes[i]
                conf = scores[i]
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"car {conf:.2f}",
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
                f"Cars: {car_count}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("YOLOv8 ONNX Vehicle Detection (q to quit)", frame)

        # Always process events + quit key (prevents freezing)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
