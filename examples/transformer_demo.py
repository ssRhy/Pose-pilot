#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transformer 优化 YOLOv8 姿态估计器使用示例

运行方式：
  python examples/transformer_demo.py --camera 0 --type swin
或：
  USE_TRANSFORMER=1 TRANSFORMER_TYPE=swin python main.py
"""

import argparse
import cv2
import os
from transformer_pose_estimator import TransformerPoseEstimator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="摄像头索引")
    parser.add_argument("--type", type=str, default="swin", choices=["swin", "vit", "detr"], help="Transformer 类型")
    args = parser.parse_args()

    estimator = TransformerPoseEstimator(use_transformer=True, transformer_type=args.type)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头")

    print("按 q 退出")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        keypoints, annotated = estimator.get_pose(frame)
        cv2.imshow("Transformer Pose Estimation", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


