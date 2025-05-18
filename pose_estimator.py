#pose_estimator.py
''''
# YOLOv8 姿态估计器封装。提供17个COCO关键点：
# 0=鼻子,1=左眼,2=右眼,3=左耳,4=右耳,
# 5=左肩,6=右肩,7=左肘,8=右肘,
# 9=左手腕,10=右手腕,11=左髋,12=右髋,
# 13=左膝,14=右膝,15=左踝,16=右踝
'''
from ultralytics import YOLO
import numpy as np
import cv2
import os
import logging

logger = logging.getLogger(__name__)

class PoseEstimator:
    """
    YOLOv8 姿态估计器封装。提供17个COCO关键点：
      0=鼻子,1=左眼,2=右眼,3=左耳,4=右耳,
      5=左肩,6=右肩,7=左肘,8=右肘,
      9=左手腕,10=右手腕,11=左髋,12=右髋,
      13=左膝,14=右膝,15=左踝,16=右踝
    """

    def __init__(self, model_path="yolov8n-pose.pt"):
            # 尝试优雅地处理网络问题
        try:
            if os.path.exists(model_path):
                logger.info(f"从{model_path}加载本地姿态模型")
                self.model = YOLO(model_path)
            else:
                # 如果本地文件不存在且网络连接有问题
                # 使用备用方案或抛出清晰的错误
                logger.warning(f"未找到模型文件{model_path}。尝试下载...")
                self.model = YOLO(model_path)
        except Exception as e:
            logger.error(f"加载姿态模型失败: {e}")
            raise RuntimeError(f"无法加载姿态模型: {e}。请检查网络连接或手动下载模型。")

    def get_pose(self, frame):
        """
        在输入帧上运行YOLOv8姿态估计。
        返回: (keypoints, annotated_frame)
          keypoints: 第一个人的关键点列表[(x_norm, y_norm), ...]
          annotated_frame: 带有骨架绘制的帧
        """
        results = self.model.predict(frame, verbose=False)[0]
        annotated_frame = results.plot()

        if len(results.keypoints) == 0:
            return [], annotated_frame

        # 为简单起见，只取第一个人的关键点
        raw_kp = results.keypoints.xy[0].cpu().numpy()  # shape (17,2)

        # 转换为归一化坐标 (0..1)
        kp_normalized = [(x / frame.shape[1], y / frame.shape[0]) for x, y in raw_kp]
        return kp_normalized, annotated_frame
