import cv2
import numpy as np
import time

PERSON_CLASS_ID = 0  # COCO: person, car = 2

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
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

def nms_xywh(boxes, scores, iou_thresh=0.45):
    # OpenCV NMSBoxes expects boxes=[x,y,w,h]
    idxs = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=iou_thresh)
    if len(idxs) == 0:
        return []
    return idxs.flatten().tolist()

def main():
    model_path = "models/yolov8n.onnx"  # update if needed
    net = cv2.dnn.readNetFromONNX(model_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam on index 0. Try index 1.")

    # Pi-friendly resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    input_size = (640, 640)
    conf_thresh = 0.20
    iou_thresh = 0.45

    prev = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h0, w0 = frame.shape[:2]

        img, scale, pad_x, pad_y = letterbox(frame, input_size)
        blob = cv2.dnn.blobFromImage(img, scalefactor=1/255.0, size=input_size, swapRB=True, crop=False)

        net.setInput(blob)
        preds = net.forward()  # shape usually (1, 84, 8400) for COCO
        # print("raw preds shape:", preds.shape) #troubleshooting


        preds = np.squeeze(preds)
        if preds.ndim != 2:
            # Safety check in case your output shape differs
            preds = preds.reshape(preds.shape[0], -1)
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T  # -> (8400, 84)

        boxes = []
        scores = []

        for det in preds:
            x, y, bw, bh = det[0:4]

            # YOLOv8 ONNX: det[4:] are class scores directly (no separate objectness)
            cls_scores = det[4:]
            cls_id = int(np.argmax(cls_scores))
            conf = float(cls_scores[cls_id])

            print(f"class_id={cls_id}, confidence={conf:.6f}")


            if cls_id != PERSON_CLASS_ID:
                continue
            if conf < conf_thresh:
                continue

            # from letterboxed coords -> original coords
            x1 = (x - bw / 2) - pad_x
            y1 = (y - bh / 2) - pad_y
            x2 = (x + bw / 2) - pad_x
            y2 = (y + bh / 2) - pad_y

            x1 = int(x1 / scale); y1 = int(y1 / scale)
            x2 = int(x2 / scale); y2 = int(y2 / scale)

            x1 = max(0, min(w0 - 1, x1))
            y1 = max(0, min(h0 - 1, y1))
            x2 = max(0, min(w0 - 1, x2))
            y2 = max(0, min(h0 - 1, y2))

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(float(conf))

       # print(f"detections this frame: {len(scores)}") #troubleshooting

        keep = nms_xywh(boxes, scores, iou_thresh=iou_thresh)

        for i in keep:
            x, y, bw, bh = boxes[i]
            conf = scores[i]
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(frame, f"person {conf:.2f}", (x, max(0, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - prev))
        prev = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("YOLOv8 ONNX Person Detection (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
