import cv2
import json
import numpy as np
from datetime import datetime

WINDOW = "Label Parking Spaces (p=prev, n=next, u=undo, r=reset current, s=save, q=quit)"

# Usage:
# - Left click to add points for current space polygon
# - Press 'n' to finish current polygon and start next
# - Press 'p' to go back to previous space (and edit its points)
# - Press 'u' to undo last point
# - Press 'r' to reset current polygon
# - Press 's' to save spaces.json
# - Press 'q' to quit

class Labeler:
    def __init__(self, frame):
        self.frame0 = frame.copy()
        self.h, self.w = frame.shape[:2]
        self.spaces = []  # list of dicts: {"id": int, "points":[[x,y],...]}
        self.idx = 0
        self.current_points = []
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw(self):
        canvas = self.frame0.copy()

        # draw existing spaces
        for s in self.spaces:
            pts = np.array(s["points"], dtype=np.int32)
            if len(pts) >= 3:
                cv2.polylines(canvas, [pts], True, (0, 255, 0), 2)
                # label at centroid-ish
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))
                cv2.putText(canvas, f"#{s['id']}", (cx, cy), self.font, 0.7, (0, 255, 0), 2)

        # draw current points/poly
        pts = np.array(self.current_points, dtype=np.int32)
        for (x, y) in self.current_points:
            cv2.circle(canvas, (x, y), 3, (0, 0, 255), -1)

        if len(pts) >= 2:
            cv2.polylines(canvas, [pts], False, (0, 0, 255), 2)

        # HUD
        cv2.putText(canvas, f"Current space: {self.idx}  Points: {len(self.current_points)}", (10, 25),
                    self.font, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, "Click points. n=next polygon, u=undo, r=reset, s=save, q=quit",
                    (10, 50), self.font, 0.55, (255, 255, 255), 2)
        return canvas

    def add_point(self, x, y):
        self.current_points.append([int(x), int(y)])

    def undo(self):
        if self.current_points:
            self.current_points.pop()

    def reset_current(self):
        self.current_points = []

    def next_space(self):
        if len(self.current_points) < 3:
            print("Need at least 3 points for a polygon.")
            return
        self.spaces.append({"id": self.idx, "points": self.current_points})
        self.idx += 1
        self.current_points = []

    def prev_space(self):
        # If currently drawing a new one, discard its in-progress points and go back
        if self.current_points:
            self.current_points = []
            return
        if not self.spaces:
            return
        last = self.spaces.pop()
        self.idx = last["id"]
        self.current_points = last["points"]

    def save(self, path="spaces.json"):
        payload = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "image_width": self.w,
            "image_height": self.h,
            "spaces": self.spaces
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved {len(self.spaces)} spaces -> {path}")

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try index 1.")

    # grab a single frame to label
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read a frame from webcam.")

    labeler = Labeler(frame)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            labeler.add_point(x, y)

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)

    while True:
        canvas = labeler.draw()
        cv2.imshow(WINDOW, canvas)
        k = cv2.waitKey(20) & 0xFF

        if k == ord('q'):
            break
        elif k == ord('u'):
            labeler.undo()
        elif k == ord('r'):
            labeler.reset_current()
        elif k == ord('n'):
            labeler.next_space()
        elif k == ord('p'):
            labeler.prev_space()
        elif k == ord('s'):
            labeler.save("spaces.json")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
