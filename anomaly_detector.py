#anomaly_detector.py
'''
# Anomaly Detector for Posture Detection
# This module implements an anomaly detector for posture detection using angles between keypoints.
# It provides two modes:
# 1) Normal angle checks (fall-like / bad posture).
# 2) Deviation from a user-defined baseline posture.
# 
# The detector computes angles between keypoints and checks if they are within specified thresholds.'''
import numpy as np
import time

class AnomalyDetector:
    """
    A posture anomaly detector with two modes:
      1) Normal angle checks ("fall-like" / "bad" posture).
      2) Deviation from a user-defined baseline posture.
    """

    def __init__(
        self,
        body_angle_threshold=55,
        neck_angle_threshold=50,
        history_window=10,
        deviation_angle_threshold=15
    ):
        """
        Args:
            body_angle_threshold      (float): If (shoulder-hip-knee) < this => bad posture.
            neck_angle_threshold      (float): If neck angle < this => bad posture.
            history_window            (int)  : Rolling time window (seconds) for 'bad' posture.
            deviation_angle_threshold (float): If user sets a baseline, differences larger
                                              than this trigger an alert.
        """
        self.timestamp_history = []
        self.body_angle_threshold = body_angle_threshold
        self.neck_angle_threshold = neck_angle_threshold
        self.window = history_window  # seconds

        # For baseline mode:
        self.baseline_angles = None
        self.deviation_angle_threshold = deviation_angle_threshold

    def set_baseline(self, keypoints):
        """
        Save the user's current posture as the 'ideal' baseline,
        by computing the main angles (left shoulder-hip-knee, right shoulder-hip-knee, neck).
        """
        if len(keypoints) < 15:
            print("Not enough keypoints to set a baseline.")
            return

        angles = self._compute_angles(keypoints)
        self.baseline_angles = angles
        print(f"[AnomalyDetector] Baseline angles set: {angles}")

    def has_baseline(self):
        return self.baseline_angles is not None

    def is_deviated_from_baseline(self, keypoints):
        """
        Compare current posture angles to the baseline angles.
        If difference is > deviation_angle_threshold in ANY angle => 'bad' posture.
        """
        if not self.has_baseline():
            return False

        if len(keypoints) < 15:
            return False

        current_angles = self._compute_angles(keypoints)
        # Compare each angle
        angle_names = ["left_body", "right_body", "avg_body", "neck"]
        for name in angle_names:
            base_val = self.baseline_angles.get(name, 9999)
            curr_val = current_angles.get(name, 9999)
            diff = abs(base_val - curr_val)
            # Debug:
            # print(f"[DEBUG] Angle {name}: baseline={base_val:.1f}, current={curr_val:.1f}, diff={diff:.1f}")
            if diff > self.deviation_angle_threshold:
                return True

        return False

    def is_fall_like(self, keypoints):
        """
        The older angle-threshold logic, used if no baseline is set.
        We'll check if posture angles are 'bad' within the last X seconds.
        """
        if len(keypoints) < 15:
            return False

        def angle(a, b, c):
            a, b, c = map(np.array, (a, b, c))
            ba = a - b
            bc = c - b
            denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
            cosine = np.dot(ba, bc) / denom
            deg = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
            return deg

        # YOLOv8 keypoints: 5=left_shoulder, 6=right_shoulder,
        # 11=left_hip, 12=right_hip, 13=left_knee, 14=right_knee, 0=nose
        left_angle = angle(keypoints[5], keypoints[11], keypoints[13])
        right_angle = angle(keypoints[6], keypoints[12], keypoints[14])
        avg_body_angle = (left_angle + right_angle) / 2
        neck_angle = angle(keypoints[0], keypoints[5], keypoints[6])

        is_bad_now = (
            (avg_body_angle < self.body_angle_threshold)
            or (neck_angle < self.neck_angle_threshold)
        )

        now = time.time()
        if is_bad_now:
            self.timestamp_history.append(now)

        # Remove timestamps older than self.window
        self.timestamp_history = [t for t in self.timestamp_history if now - t < self.window]

        return len(self.timestamp_history) >= 2

    # ----------------------------
    # Internal utility
    # ----------------------------
    def _compute_angles(self, keypoints):
        """
        Return a dict of the main angles:
          left_body, right_body, avg_body, neck
        """
        def angle(a, b, c):
            a, b, c = map(np.array, (a, b, c))
            ba = a - b
            bc = c - b
            denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
            cosine = np.dot(ba, bc) / denom
            deg = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
            return deg

        left_body = angle(keypoints[5], keypoints[11], keypoints[13])   # L shoulder-hip-knee
        right_body = angle(keypoints[6], keypoints[12], keypoints[14]) # R shoulder-hip-knee
        avg_body = (left_body + right_body) / 2
        neck = angle(keypoints[0], keypoints[5], keypoints[6])         # nose ~ shoulders

        return {
            "left_body": left_body,
            "right_body": right_body,
            "avg_body": avg_body,
            "neck": neck
        }
