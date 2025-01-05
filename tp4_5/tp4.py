import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from mlvot.tp2_3.tp2 import load_data, get_current_list_bbox, save_video, save_tracking_results
import onnxruntime
from mlvot.tp2_3.tp3 import iou, generate_frames, Tracker
from PIL import Image


class FeatureExtractor:
    def __init__(self, model_path="reid_osnet_x025_market1501.onnx", input_size=(128, 64)):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def preprocess_patch(self, patch):
        if patch is None or patch.size == 0:
            raise ValueError("Empty patch provided for preprocessing")
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        patch = Image.fromarray(patch)
        patch = self.transform(patch).unsqueeze(0)
        return patch.numpy()

    def extract_features(self, im_crops):
        features = []
        for patch in im_crops:
            preprocessed_patch = self.preprocess_patch(patch)
            feature = self.session.run(None, {self.input_name: preprocessed_patch})[0]
            features.append(feature)
        if not features:
            return np.empty((0, self.session.get_outputs()[0].shape[1]))
        return np.vstack(features)


def compute_similarity(features1, features2, metric="cosine"):
    return 1 - cdist(features1, features2, metric)


def pad_matrix(matrix, target_shape):
    padded_matrix = np.zeros(target_shape)
    padded_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix
    return padded_matrix


def get_combined_sim_matrix(bbox1, features1, bbox2, features2, alpha=0.5, beta=0.5):
    if features1.size == 0 or features2.size == 0:
        return np.empty((len(bbox1), len(bbox2)))

    iou_matrix = get_sim_matrix(bbox1, bbox2)
    feature_similarity_matrix = compute_similarity(features1, features2, metric="cosine")

    if feature_similarity_matrix.size == 0:
        return np.empty((len(bbox1), len(bbox2)))

    feature_similarity_matrix = (feature_similarity_matrix - feature_similarity_matrix.min()) / (
        feature_similarity_matrix.max() - feature_similarity_matrix.min()
    )

    if iou_matrix.shape != feature_similarity_matrix.shape:
        max_shape = (max(iou_matrix.shape[0], feature_similarity_matrix.shape[0]),
                     max(iou_matrix.shape[1], feature_similarity_matrix.shape[1]))
        iou_matrix = pad_matrix(iou_matrix, max_shape)
        feature_similarity_matrix = pad_matrix(feature_similarity_matrix, max_shape)
    return alpha * iou_matrix + beta * feature_similarity_matrix


def get_sim_matrix(bbox1, bbox2):
    matrix = np.zeros((len(bbox1), len(bbox2)))

    for i, tracked_bbox in enumerate(bbox1):
        for j, new_detection in enumerate(bbox2):
            matrix[i][j] = iou(tracked_bbox, new_detection)

    return matrix


def get_assignments(sim_matrix):
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)
    return row_ind, col_ind


def main():
    path_det = "../tp2_3/files/det/Yolov5s/det.txt"
    df_det_s = load_data(path_det, sep=" ")

    feature_extractor = FeatureExtractor()

    tracker = Tracker()
    all_tracked_bboxs = []

    frames = sorted(df_det_s['frame'].unique())

    img1_dir = "../tp2_3/files/img1"

    for frame_idx in frames:
        detections = get_current_list_bbox(df_det_s, frame_idx)

        frame_path = f"{img1_dir}/{frame_idx:06d}.jpg"
        frame = cv2.imread(frame_path)

        im_crops = []
        for d in detections:
            x, y, w, h = int(d[0]), int(d[1]), int(d[2]), int(d[3])
            if x >= 0 and y >= 0 and x + w <= frame.shape[1] and y + h <= frame.shape[0]:
                im_crops.append(frame[y:y + h, x:x + w])

        features = feature_extractor.extract_features(im_crops)

        tracked_data = tracker.update(detections)
        tracked_bboxs = [data[0] for data in tracked_data]

        valid_patches = []
        for d in tracked_bboxs:
            x, y, w, h = int(d[0]), int(d[1]), int(d[2]), int(d[3])
            if x >= 0 and y >= 0 and x + w <= frame.shape[1] and y + h <= frame.shape[0]:
                patch = frame[y:y + h, x:x + w]
                if patch.size > 0:
                    valid_patches.append(patch)

        tracked_features = feature_extractor.extract_features(valid_patches)

        sim_matrix = get_combined_sim_matrix(tracked_bboxs, tracked_features, detections, features)

        # row_ind, col_ind = get_assignments(sim_matrix)

        tracker.update(detections)

        tracked_bboxs = [data[0] for data in tracker.update(detections)]
        all_tracked_bboxs.append(tracked_bboxs)

    save_tracking_results(all_tracked_bboxs, "output_tp4.txt")

    frames = generate_frames(all_tracked_bboxs)

    save_video(frames, "output_tp4.avi")


if __name__ == "__main__":
    main()
