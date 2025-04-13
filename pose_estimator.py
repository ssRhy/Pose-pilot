#pose_estimator.py
''''
# YOLOv8 Pose estimator wrapper. Provides 17 COCO keypoints:
# 0=nose,1=left_eye,2=right_eye,3=left_ear,4=right_ear,
# 5=left_shoulder,6=right_shoulder,7=left_elbow,8=right_elbow,
# 9=left_wrist,10=right_wrist,11=left_hip,12=right_hip,
# 13=left_knee,14=right_knee,15=left_ankle,16=right_ankle
'''
from ultralytics import YOLO
import numpy as np
import cv2

class PoseEstimator:
    """
    YOLOv8 Pose estimator wrapper. Provides 17 COCO keypoints:
      0=nose,1=left_eye,2=right_eye,3=left_ear,4=right_ear,
      5=left_shoulder,6=right_shoulder,7=left_elbow,8=right_elbow,
      9=left_wrist,10=right_wrist,11=left_hip,12=right_hip,
      13=left_knee,14=right_knee,15=left_ankle,16=right_ankle
    """

    def __init__(self):
        self.model = YOLO("yolov8n-pose.pt")

    def get_pose(self, frame):
        """
        Runs YOLOv8 pose estimation on the input frame.
        Returns: (keypoints, annotated_frame)
          keypoints: List[(x_norm, y_norm), ...] for first person
          annotated_frame: the frame with skeleton drawn
        """
        results = self.model.predict(frame, verbose=False)[0]
        annotated_frame = results.plot()

        if len(results.keypoints) == 0:
            return [], annotated_frame

        # For simplicity, take the first person's keypoints
        raw_kp = results.keypoints.xy[0].cpu().numpy()  # shape (17,2)

        # Convert to normalized coords (0..1)
        kp_normalized = [(x / frame.shape[1], y / frame.shape[0]) for x, y in raw_kp]
        return kp_normalized, annotated_frame
