import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from mlvot.tp2_3.tp2 import load_data, get_current_list_bbox, save_video


# Helper function for IoU calculation
def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    xi1, yi1 = max(x1, x1_p), max(y1, y1_p)
    xi2, yi2 = min(x2, x2_p), min(y2, y2_p)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


# Helper function to extract bounding box properties
def get_bbox_properties(bbox):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return cx, cy, w, h


def generate_frames(tracked_bboxs):
    frames = []

    for i in range(len(tracked_bboxs)):
        img = cv2.imread(f"files/img1/{str(i+1).zfill(6)}.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for j, bbox in enumerate(tracked_bboxs[i]):
            x, y, w, h, index = bbox
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 5)
            cv2.putText(img, str(j), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        frames.append(img)

    return frames


def save_tracking_results(tracked_bboxs, output_file="ADL-Rundle-6.txt"):
    results = []

    for frame, content in enumerate(tracked_bboxs):
        for id, bbox in enumerate(content):
            x, y, w, h, index = bbox
            results.append(f"{frame+1},{index},{x},{y},{w},{h},1,1,1,1")

    with open(output_file, "w") as f:
        f.write("\n".join(results))

    print(f"Tracking results saved to {output_file}")


# Kalman Tracker Class
class KalmanTracker:
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=6, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ])
        self.kf.R *= 10
        self.kf.P *= 1000
        self.kf.Q *= 0.01

        cx, cy, w, h = get_bbox_properties(bbox)
        self.kf.x[:4] = np.array([cx, cy, w, h]).reshape((4, 1))

    def predict(self):
        self.kf.predict()
        cx, cy, w, h = self.kf.x[:4].flatten()
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    def update(self, bbox):
        cx, cy, w, h = get_bbox_properties(bbox)
        self.kf.update([cx, cy, w, h])


# Tracking system
class Tracker:
    def __init__(self):
        self.tracks = []
        self.track_ids = []
        self.next_id = 0

    def update(self, detections):
        # Predict step for existing tracks
        predicted_positions = [track.predict() for track in self.tracks]

        # Compute IoU and use Hungarian Algorithm
        if len(predicted_positions) > 0 and len(detections) > 0:
            iou_matrix = np.zeros((len(predicted_positions), len(detections)))
            for i, track_bbox in enumerate(predicted_positions):
                for j, det_bbox in enumerate(detections):
                    iou_matrix[i, j] = iou(track_bbox, det_bbox)

            row_ind, col_ind = linear_sum_assignment(-iou_matrix)

            assigned_tracks = set()
            assigned_detections = set()

            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] > 0.3:
                    self.tracks[r].update(detections[c])
                    assigned_tracks.add(r)
                    assigned_detections.add(c)

            # Handle unassigned tracks
            unassigned_tracks = set(range(len(self.tracks))) - assigned_tracks
            for track_idx in sorted(unassigned_tracks, reverse=True):
                del self.tracks[track_idx]
                del self.track_ids[track_idx]

            # Handle new detections
            unassigned_detections = set(range(len(detections))) - assigned_detections
            for det_idx in unassigned_detections:
                new_tracker = KalmanTracker(detections[det_idx])
                self.tracks.append(new_tracker)
                self.track_ids.append(self.next_id)
                self.next_id += 1

        elif len(detections) > 0:
            # No existing tracks, create new ones
            for det in detections:
                new_tracker = KalmanTracker(det)
                self.tracks.append(new_tracker)
                self.track_ids.append(self.next_id)
                self.next_id += 1

        elif len(predicted_positions) > 0:
            # No detections, remove existing tracks
            self.tracks = []
            self.track_ids = []

        # Return active track bounding boxes and IDs
        return [(tracker.kf.x[:4].flatten().tolist(), tid) for tracker, tid in zip(self.tracks, self.track_ids)]


def main():
    # Load detection data
    path_det = "files/det/Yolov5s/det.txt"
    df_det_s = load_data(path_det, sep=" ")

    tracker = Tracker()
    all_tracked_bboxs = []

    frames = sorted(df_det_s['frame'].unique())
    for frame_idx in frames:
        detections = get_current_list_bbox(df_det_s, frame_idx)

        if len(detections) == 0:
            continue

        tracked_data = tracker.update(detections)
        frame_results = []
        for (position, track_id) in tracked_data:
            cx, cy, w, h = position
            bbox = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
            frame_results.append((*bbox, track_id))

        all_tracked_bboxs.append(frame_results)

    # Generate and save video
    frames_with_tracks = generate_frames(all_tracked_bboxs)
    save_video(frames_with_tracks, "output_tp3.avi", fps=30, frame_size=(1920, 1080))

    # Save tracking results
    save_tracking_results(all_tracked_bboxs)

if __name__ == "__main__":
    main()