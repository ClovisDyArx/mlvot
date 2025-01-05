import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

# Track management
track_id_dict = {}
next_track_id = 1


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def load_data(path, sep=","):
    return pd.read_csv(path, sep=sep, names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z'])


def check_data(df):
    print(bcolors.HEADER + "General info :" + bcolors.ENDC)
    print(df.info())

    print(bcolors.HEADER + "\n\nStats :" + bcolors.ENDC)
    print(df.describe())

    print(bcolors.HEADER + "\n\nNull entries :" + bcolors.ENDC)
    print(df.isnull().sum())

    print(bcolors.HEADER + "\n\nTypes :" + bcolors.ENDC)
    print(df.dtypes)


def plot_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)


def plot_img_bboxs(path, df, frame):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    temp_df = df[df['frame'] == frame]

    for _, row in temp_df.iterrows():
        x, y, w, h = int(row['bb_left']), int(row['bb_top']), int(row['bb_width']), int(row['bb_height'])
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 10)

    plt.imshow(img)


def get_current_list_bbox(df, frame):
    res = []

    temp_df = df[df['frame'] == frame]
    for _, row in temp_df.iterrows():
        x, y, w, h = row['bb_left'], row['bb_top'], row['bb_width'], row['bb_height']
        res.append([x, y, w, h])

    return res


def get_IoU(bbox1, bbox2):
    x, y, w, h = bbox1
    _x, _y, _w, _h = bbox2

    xA = max(x, _x)
    yA = max(y, _y)
    xB = min(x+w, _x+_w)
    yB = min(y+h, _y+_h)

    interArea = max(0, xB - xA) * max(0, yB - yA)
    trackedArea = bbox1[2] * bbox1[3]
    newArea = bbox2[2] * bbox2[3]
    IoU = interArea / float(trackedArea + newArea - interArea)

    return IoU


def get_sim_matrix(bbox1, bbox2):
    matrix = np.zeros((len(bbox1), len(bbox2)))

    for i, tracked_bbox in enumerate(bbox1):
        for j, new_detection in enumerate(bbox2):
            matrix[i][j] = get_IoU(tracked_bbox, new_detection)

    return matrix


def get_assignments(sim_matrix):
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)

    return row_ind, col_ind


def update_tracks_with_ids(tracked_bboxs, new_detections, assignments):
    global next_track_id
    matches = []
    unmatched_tracks = []
    unmatched_detections = []

    # Update matches and unmatched tracks
    for i, tracked_bbox in enumerate(tracked_bboxs):
        if i not in assignments[0]:
            unmatched_tracks.append(i)
        else:
            match_index = assignments[0].tolist().index(i)
            if assignments[1][match_index] < len(new_detections):
                matches.append([i, assignments[1][match_index]])
            else:
                unmatched_tracks.append(i)

    # Unmatched detections
    for j, new_detection in enumerate(new_detections):
        if j not in assignments[1]:
            unmatched_detections.append(j)

    # Update track IDs
    updated_tracks = {}
    for match in matches:
        track_idx, detection_idx = match
        # Retain the ID of the matched track
        for track_id, bbox in track_id_dict.items():
            if list(bbox) == list(tracked_bboxs[track_idx]):
                updated_tracks[track_id] = new_detections[detection_idx]

    # Add new IDs for unmatched detections
    for detection_idx in unmatched_detections:
        updated_tracks[next_track_id] = new_detections[detection_idx]
        next_track_id += 1

    # Update global track ID dictionary
    track_id_dict.clear()
    track_id_dict.update(updated_tracks)

    return matches, unmatched_tracks, unmatched_detections


def get_all_tracked_bboxs(df):
    frames = df['frame'].unique()
    all_tracked_bboxs = []
    all_new_detections = []
    all_sim_matrices = []
    all_assignments = []
    all_matches = []
    all_unmatched_tracks = []
    all_unmatched_detections = []

    for current_frame in frames:
        tracked_bboxs = get_current_list_bbox(df, current_frame)
        new_detections = get_current_list_bbox(df, current_frame + 1)

        sim_matrix = get_sim_matrix(tracked_bboxs, new_detections)
        assignments = get_assignments(sim_matrix)

        matches, unmatched_tracks, unmatched_detections = update_tracks_with_ids(tracked_bboxs, new_detections, assignments)

        all_tracked_bboxs.append(tracked_bboxs)
        all_new_detections.append(new_detections)
        all_sim_matrices.append(sim_matrix)
        all_assignments.append(assignments)
        all_matches.append(matches)
        all_unmatched_tracks.append(unmatched_tracks)
        all_unmatched_detections.append(unmatched_detections)

    return all_tracked_bboxs


def save_video(frames, output_path, fps=10, frame_size=(1920, 1080)):
    result = cv2.VideoWriter(output_path,
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             fps, frame_size)

    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        result.write(frame_bgr)

    result.release()
    print("The video was successfully saved")


def generate_frames(tracked_bboxs):
    frames = []

    for i in range(len(tracked_bboxs)):
        img = cv2.imread(f"files/img1/{str(i+1).zfill(6)}.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for j, bbox in enumerate(tracked_bboxs[i]):
            x, y, w, h = bbox
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 5)
            cv2.putText(img, str(j), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        frames.append(img)

    return frames


def save_tracking_results(tracked_bboxs, output_file="ADL-Rundle-6.txt"):
    results = []

    for frame, content in enumerate(tracked_bboxs):
        for id, bbox in enumerate(content):
            x, y, w, h = bbox
            results.append(f"{frame+1},{id+1},{x},{y},{w},{h},1,1,1,1")

    with open(output_file, "w") as f:
        f.write("\n".join(results))

    print(f"Tracking results saved to {output_file}")


def main_loop_full(path_det):
    df_det_s = load_data(path_det, sep=" ")

    check_data(df_det_s)

    plot_img("files/img1/000001.jpg")
    plot_img_bboxs("files/img1/000001.jpg", df_det_s, 1)

    current_frame = 1
    tracked_bboxs = get_current_list_bbox(df_det_s, current_frame)
    new_detections = get_current_list_bbox(df_det_s, current_frame + 1)

    sim_matrix = get_sim_matrix(tracked_bboxs, new_detections)
    plt.imshow(sim_matrix, cmap='hot', interpolation='nearest')
    plt.show()

    assignments = get_assignments(sim_matrix)

    all_tracked_bboxs = get_all_tracked_bboxs(df_det_s)

    frames = generate_frames(all_tracked_bboxs)
    save_video(frames, 'output_tp2.avi', fps=30, frame_size=(1920, 1080))

    save_tracking_results(all_tracked_bboxs)


def main_loop(path_det):
    df_det_s = load_data(path_det, sep=" ")

    all_tracked_bboxs = get_all_tracked_bboxs(df_det_s)

    frames = generate_frames(all_tracked_bboxs)
    save_video(frames, 'output_tp2.avi', fps=30, frame_size=(1920, 1080))

    save_tracking_results(all_tracked_bboxs)


if __name__ == "__main__":
    path_det = "files/det/Yolov5s/det.txt"
    main_loop(path_det)
