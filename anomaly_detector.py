#anomaly_detector.py
'''
# 姿态检测异常检测器
# 该模块使用关键点之间的角度实现姿态检测的异常检测器。
# 它提供两种模式：
# 1) 常规角度检查（跌倒类 / 不良姿态）。
# 2) 与用户定义的基准姿态的偏差。
# 
# 检测器计算关键点之间的角度，并检查它们是否在指定的阈值范围内。'''
import numpy as np
import time

class AnomalyDetector:
    """
    具有两种模式的姿态异常检测器：
      1) 常规角度检查（"跌倒类" / "不良"姿态）。
      2) 与用户定义的基准姿态的偏差。
    """

    def __init__(
        self,
        body_angle_threshold=55,
        neck_angle_threshold=50,
        history_window=10,
        deviation_angle_threshold=15
    ):
        """
        参数:
            body_angle_threshold      (float): 如果(肩-髋-膝)角度 < 此值 => 不良姿态。
            neck_angle_threshold      (float): 如果颈部角度 < 此值 => 不良姿态。
            history_window            (int)  : '不良'姿态的滚动时间窗口（秒）。
            deviation_angle_threshold (float): 如果用户设置了基准，差异大于
                                              此值将触发警报。
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
        通过计算主要角度（左肩-髋-膝、右肩-髋-膝、颈部），
        将用户当前姿态保存为'理想'基准。
        """
        if len(keypoints) < 15:
            print("没有足够的关键点来设置基准。")
            return

        angles = self._compute_angles(keypoints)
        self.baseline_angles = angles
        print(f"[异常检测器] 基准角度已设置: {angles}")

    def has_baseline(self):
        return self.baseline_angles is not None

    def is_deviated_from_baseline(self, keypoints):
        """
        将当前姿态角度与基准角度进行比较。
        如果任何角度的差异 > deviation_angle_threshold => '不良'姿态。
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
            # 调试:
            # print(f"[调试] 角度 {name}: 基准={base_val:.1f}, 当前={curr_val:.1f}, 差异={diff:.1f}")
            if diff > self.deviation_angle_threshold:
                return True

        return False

    def is_fall_like(self, keypoints):
        """
        如果没有设置基准，则使用较旧的角度阈值逻辑。
        我们将检查在过去X秒内姿态角度是否'不良'。
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

        # YOLOv8关键点: 5=左肩, 6=右肩,
        # 11=左髋, 12=右髋, 13=左膝, 14=右膝, 0=鼻子
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
    # 内部工具函数
    # ----------------------------
    def _compute_angles(self, keypoints):
        """
        返回主要角度的字典：
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

        left_body = angle(keypoints[5], keypoints[11], keypoints[13])   # 左肩-髋-膝
        right_body = angle(keypoints[6], keypoints[12], keypoints[14]) # 右肩-髋-膝
        avg_body = (left_body + right_body) / 2
        neck = angle(keypoints[0], keypoints[5], keypoints[6])         # 鼻子~肩膀

        return {
            "left_body": left_body,
            "right_body": right_body,
            "avg_body": avg_body,
            "neck": neck
        }
