# -*- coding: utf-8 -*-
"""
yolo_rtsp_cn.py - 使用YOLOv8的简单RTSP测试脚本（中文版）

依赖:
    pip install opencv-python ultralytics pillow
"""

import cv2
import sys
import time
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 中文标签映射字典
CHINESE_LABELS = {
    'person': '人',
    'bicycle': '自行车',
    'car': '汽车',
    'motorcycle': '摩托车',
    'airplane': '飞机',
    'bus': '公交车',
    'train': '火车',
    'truck': '卡车',
    'boat': '船',
    'traffic light': '红绿灯',
    'fire hydrant': '消防栓',
    'stop sign': '停止标志',
    'parking meter': '停车计时器',
    'bench': '长椅',
    'bird': '鸟',
    'cat': '猫',
    'dog': '狗',
    'horse': '马',
    'sheep': '羊',
    'cow': '牛',
    'elephant': '大象',
    'bear': '熊',
    'zebra': '斑马',
    'giraffe': '长颈鹿',
    'backpack': '背包',
    'umbrella': '雨伞',
    'handbag': '手提包',
    'tie': '领带',
    'suitcase': '手提箱',
    'frisbee': '飞盘',
    'skis': '滑雪板',
    'snowboard': '滑雪板',
    'sports ball': '运动球',
    'kite': '风筝',
    'baseball bat': '棒球棒',
    'baseball glove': '棒球手套',
    'skateboard': '滑板',
    'surfboard': '冲浪板',
    'tennis racket': '网球拍',
    'bottle': '瓶子',
    'wine glass': '酒杯',
    'cup': '杯子',
    'fork': '叉子',
    'knife': '刀',
    'spoon': '勺子',
    'bowl': '碗',
    'banana': '香蕉',
    'apple': '苹果',
    'sandwich': '三明治',
    'orange': '橙子',
    'broccoli': '西兰花',
    'carrot': '胡萝卜',
    'hot dog': '热狗',
    'pizza': '披萨',
    'donut': '甜甜圈',
    'cake': '蛋糕',
    'chair': '椅子',
    'couch': '沙发',
    'potted plant': '盆栽',
    'bed': '床',
    'dining table': '餐桌',
    'toilet': '马桶',
    'tv': '电视',
    'laptop': '笔记本电脑',
    'mouse': '鼠标',
    'remote': '遥控器',
    'keyboard': '键盘',
    'cell phone': '手机',
    'microwave': '微波炉',
    'oven': '烤箱',
    'toaster': '烤面包机',
    'sink': '水槽',
    'refrigerator': '冰箱',
    'book': '书',
    'clock': '时钟',
    'vase': '花瓶',
    'scissors': '剪刀',
    'teddy bear': '泰迪熊',
    'hair drier': '吹风机',
    'toothbrush': '牙刷'
}

def put_chinese_text(img, text, pos, color=(0, 255, 0)):
    """在图片上绘制中文文本"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    font_size = int(min(img.shape[0], img.shape[1]) / 30)  # 自适应字体大小
    try:
        font = ImageFont.truetype("simhei.ttf", font_size)  # 使用系统的黑体
    except:
        # 如果找不到系统字体，使用OpenCV默认字体
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return img
    
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=color[::-1])  # OpenCV使用BGR，PIL使用RGB
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def main():
    # 获取RTSP URL
    if len(sys.argv) > 1:
        rtsp_url = sys.argv[1]
    else:
        # 默认RTSP地址
        rtsp_url = "rtsp://10.148.165.1:8554/live"
        print(f"未提供RTSP URL，使用默认: {rtsp_url}")
        print("你可以提供自定义RTSP URL作为参数:")
        print("示例: python yolo_rtsp_cn.py <你的RTSP地址>")
    
    # 加载YOLOv8模型
    print("正在加载YOLOv8模型...")
    try:
        model = YOLO("yolov8n-pose.pt")  # 使用最小的YOLOv8模型
        print("YOLOv8模型加载成功!")
    except Exception as e:
        print(f"YOLOv8模型加载失败: {e}")
        print("请确保已正确安装ultralytics并下载模型")
        return
    
    print(f"正在连接RTSP流: {rtsp_url}")
    
    # 设置RTSP选项
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    # 设置缓冲区大小
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    
    # 检查连接状态
    if not cap.isOpened():
        print(f"错误: 无法连接到RTSP流 {rtsp_url}")
        print("可能的原因:")
        print("  1. RTSP服务器未运行")
        print("  2. URL格式错误")
        print("  3. 用户名/密码错误")
        print("  4. 网络连接问题")
        return
    
    print("成功连接到RTSP流!")
    
    try:
        frame_count = 0
        start_time = time.time()
        fps_update_interval = 30  # 每30帧更新一次FPS
        
        while True:
            # 读取一帧
            ret, frame = cap.read()
            
            # 如果读取失败
            if not ret:
                print("帧读取失败，尝试重新连接...")
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                time.sleep(0.5)
                continue
            
            frame_count += 1
            
            # 使用YOLOv8进行目标检测
            results = model(frame)
            
            # 获取原始帧的副本
            display_frame = frame.copy()
            
            # 在帧上绘制检测结果
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # 获取置信度
                    conf = float(box.conf[0])
                    # 获取类别
                    cls = int(box.cls[0])
                    # 获取类别名称
                    name = r.names[cls]
                    # 转换为中文名称
                    chinese_name = CHINESE_LABELS.get(name, name)
                    
                    # 绘制边界框
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # 添加中文标签
                    label = f"{chinese_name} {conf:.2f}"
                    display_frame = put_chinese_text(display_frame, label, (x1, y1-10))
            
            # 计算并显示FPS
            if frame_count % fps_update_interval == 0:
                end_time = time.time()
                fps = fps_update_interval / (end_time - start_time)
                start_time = end_time
                # 在帧上添加FPS文本
                display_frame = put_chinese_text(display_frame, f"FPS: {fps:.1f}", (20, 40))
            
            # 显示带有检测结果的帧
            cv2.imshow("YOLOv8目标检测", display_frame)
            
            # 按'q'键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"错误: {e}")
    
    finally:
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        print("已关闭RTSP流和窗口")

if __name__ == "__main__":
    main()