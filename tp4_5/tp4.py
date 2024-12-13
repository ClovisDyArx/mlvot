"""
Objective: Extend IoU-Kalman tracker to include object re-identification (ReID)
Instead of relying solely on geometric information (IoU) consider also the appearance similarity between the
predicted object state (form frame t-1) and the detected objects (from frame t).
1.Implement Object Re-Identification
 Feature Extraction: Use a pre-trained lightweight deep learning model (EfficientNet, OSNet or
MobileNet) to extract features from detected objects (image patches). You can download
checkpoints of REiD model from obtained link or search in the Git repository.
 Patch preprocessing:
o Generate a patch for each detected object. The size of each patch is defined by its
bounding box => im_crops
o Patch Resizing: Resize each patch (im_crops) to the size of the image patches used to
train the ReID model you are using. If the provided checkpoints are utilized, resize to
(64, 128), following the approach of the Market1501 dataset used to train this OSNet
model.
o Convert BGR to RGB
o Normalize
Here is an example function to preprocess image patches:
def preprocess_patch(self, im_crops):
roi_input = cv2.resize(im_crops, (self.roi_width, self.roi_height))
roi_input = cv2.cvtColor(roi_input, cv2.COLOR_BGR2RGB)
roi_input = (np.asarray(roi_input).astype(np.float32) -
/self.roi_stds
roi_input = np.moveaxis(roi_input, -1, 0)
object_patch = roi_input.astype('float32')
return object_patch
self.roi_means)
2.
 Compute the ReID vector for each patch

 Distance Metrics for ReID: Use metrics like cosine similarity or Euclidean distance to compare
feature vectors of detected objects with those of tracked objects
Combine IoU and Feature Similarity to make the association more robust. One common approach is to
use a weighted sum. You could define a combined score S as follows:
𝑆 = 𝛼 ∙ 𝐼𝑜𝑈 + 𝛽 ∙ 𝑁𝑜𝑟𝑚𝑎𝑙𝑖𝑧𝑒𝑑 𝑆𝑖𝑚𝑖𝑙𝑎𝑟𝑖𝑡𝑦
where: α and β are weights that you can tune based on your application
Normalized Similarity is obtained by normalizing the feature similarity score (e.g., for cosine similarity,
this could be directly used; for Euclidean distance, you would need to invert it to get a similarity score).
1
𝑁𝑜𝑟𝑚𝑎𝑙𝑖𝑧𝑒𝑑 𝑆𝑖𝑚𝑖𝑙𝑎𝑟𝑖𝑡𝑦 = 1 / (1 + 𝐸𝑢𝑐𝑙𝑖𝑑𝑒𝑎𝑛 𝐷𝑖𝑠𝑡𝑎𝑛𝑐𝑒)
This ensures that a lower distance results in a higher similarity score.
"""
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from mlvot.tp2_3.tp2 import load_data, get_current_list_bbox, save_video
import onnxruntime
from mlvot.tp2_3.tp3 import iou, generate_frames, save_tracking_results, Tracker
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
        return np.vstack(features)


def compute_similarity(features1, features2, metric="cosine"):
    return 1 - cdist(features1, features2, metric)


def get_combined_sim_matrix(bbox1, features1, bbox2, features2, alpha=0.5, beta=0.5):
    iou_matrix = get_sim_matrix(bbox1, bbox2)
    feature_similarity_matrix = compute_similarity(features1, features2, metric="cosine")

    feature_similarity_matrix = (feature_similarity_matrix - feature_similarity_matrix.min()) / (
        feature_similarity_matrix.max() - feature_similarity_matrix.min()
    )

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
    # Load detection data
    path_det = "../tp2_3/files/det/Yolov5s/det.txt"
    df_det_s = load_data(path_det, sep=" ")

    # Load ReID model
    feature_extractor = FeatureExtractor()

    tracker = Tracker()
    all_tracked_bboxs = []

    frames = sorted(df_det_s['frame'].unique())

    # Print the list of files in the img1 directory
    img1_dir = "../tp2_3/files/img1"

    for frame_idx in frames:
        print(frame_idx)
        # Get detections for current frame
        detections = get_current_list_bbox(df_det_s, frame_idx)

        # Load the corresponding frame image
        frame_path = f"{img1_dir}/{frame_idx:06d}.jpg"
        frame = cv2.imread(frame_path)

        print(frame.shape)

        # Extract image patches from the frame using bounding boxes
        im_crops = []
        for d in detections:
            x, y, w, h = int(d[0]), int(d[1]), int(d[2]), int(d[3])
            if x >= 0 and y >= 0 and x + w <= frame.shape[1] and y + h <= frame.shape[0]:
                im_crops.append(frame[y:y + h, x:x + w])

        print(len(im_crops))
        # Get features for detections
        features = feature_extractor.extract_features(im_crops)

        print(features.shape)

        # Get tracked bounding boxes and features
        tracked_data = tracker.update(detections)
        tracked_bboxs = [data[0] for data in tracked_data]
        tracked_features = feature_extractor.extract_features(
            [frame[int(d[1]):int(d[1] + d[3]), int(d[0]):int(d[0] + d[2])] for d in tracked_bboxs if int(d[0]) >= 0 and int(d[1]) >= 0 and int(d[0]) + int(d[2]) <= frame.shape[1] and int(d[1]) + int(d[3]) <= frame.shape[0]])

        # Compute combined similarity matrix
        sim_matrix = get_combined_sim_matrix(tracked_bboxs, tracked_features, detections, features)

        # Get assignments
        row_ind, col_ind = get_assignments(sim_matrix)

        # Update tracks
        tracker.update(detections)

        # Get tracked bounding boxes
        tracked_bboxs = [data[0] for data in tracker.update(detections)]
        all_tracked_bboxs.append(tracked_bboxs)

    # Save tracking results
    save_tracking_results(all_tracked_bboxs, "output_tp4.txt")

    # Generate frames
    frames = generate_frames(all_tracked_bboxs)

    # Save video
    save_video(frames, "output_tp4.avi")


if __name__ == "__main__":
    main()
