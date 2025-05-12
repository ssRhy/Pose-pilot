'''
简化版的RTSP姿势检测器
提供独立于Flask的姿势检测功能
'''
import os
import cv2
import time
import argparse
import threading
import queue
import base64
import logging

# 导入自定义模块
from yolo_detector import YOLODetector
from pose_estimator import PoseEstimator
from anomaly_detector import AnomalyDetector

# 可选: 导入TTS功能
from speaker.ip_speaker import send_tts

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

class RTSPPoseDetector:
    def __init__(self, rtsp_url="rtsp://192.168.3.242:8554/live", use_tts=True):
        """
        初始化RTSP姿势检测器
        
        Args:
            rtsp_url: RTSP流URL
            use_tts: 是否使用TTS功能
        """
        self.rtsp_url = rtsp_url
        self.use_tts = use_tts
        self.running = False
        self.thread = None
        
        # 队列用于存储最新帧和结果
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        
        # 初始化检测器
        self.detector = YOLODetector(
            model_path="yolov8n-pose.pt",
            device="auto",
            conf_thres=0.7,
            iou_thres=0.45
        )
        
        # 初始化姿态估计
        try:
            self.pose_estimator = PoseEstimator()
            logger.info("成功初始化姿态估计器")
        except Exception as e:
            logger.error(f"初始化姿态估计器失败: {e}")
            raise RuntimeError(f"无法初始化姿态估计器: {e}")
            
        # 初始化异常检测器
        self.anomaly_detector = AnomalyDetector()
        
        # 姿势报告计数器和间隔
        self.frame_count = 0
        self.report_interval = 30  # 每30帧生成一次报告
        self.last_report_time = 0
        self.min_report_interval = 10  # 最短报告间隔10秒

    def start(self):
        """启动RTSP检测线程"""
        if self.running:
            logger.info("检测线程已经在运行")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._detection_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"RTSP检测线程已启动, URL: {self.rtsp_url}")
        
    def stop(self):
        """停止RTSP检测线程"""
        if not self.running:
            logger.info("检测线程未在运行")
            return
            
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        logger.info("RTSP检测线程已停止")
        
        # 清空队列
        while not self.frame_queue.empty():
            self.frame_queue.get()
        while not self.result_queue.empty():
            self.result_queue.get()
            
    def _scale_keypoints(self, kp_normalized, shape):
        """
        将归一化关键点转换为绝对像素坐标
        """
        h, w, _ = shape
        return [(x * w, y * h) for x, y in kp_normalized]
        
    def _detection_loop(self):
        """RTSP检测循环"""
        try:
            # 设置RTSP连接
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;15000000"
            
            logger.info(f"尝试连接到RTSP流: {self.rtsp_url}")
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            
            # 检查连接是否成功
            if not cap.isOpened():
                logger.error(f"无法连接到RTSP流: {self.rtsp_url}")
                self.running = False
                return
                
            logger.info("RTSP连接成功")
            
            while self.running:
                # 读取帧
                ret, frame = cap.read()
                
                # 如果读取失败，尝试重新连接
                if not ret:
                    logger.warning("帧读取失败，尝试重新连接...")
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                    continue
                
                # 更新帧队列
                if self.frame_queue.full():
                    self.frame_queue.get()
                self.frame_queue.put(frame)
                
                # 检测人物
                boxes, confs = self.detector.detect(frame)
                
                # 如果检测到人物，进行姿态估计
                if boxes:
                    # 选择置信度最高的检测框
                    max_conf_idx = max(range(len(confs)), key=lambda i: confs[i])
                    
                    # 估计姿态
                    kp_norm, annotated_frame = self.pose_estimator.get_pose(frame)
                    kp_abs = self._scale_keypoints(kp_norm, frame.shape)
                    
                    # 计算角度
                    angles = self.anomaly_detector._compute_angles(kp_abs)
                    
                    # 确定姿态状态
                    if self.anomaly_detector.has_baseline():
                        is_deviated = self.anomaly_detector.is_deviated_from_baseline(kp_abs)
                        posture_status = "bad" if is_deviated else "good"
                    else:
                        is_bad = self.anomaly_detector.is_fall_like(kp_abs)
                        posture_status = "bad" if is_bad else "good"
                    
                    # 更新结果队列
                    try:
                        # 转换为JPEG格式
                        _, buffer = cv2.imencode('.jpg', annotated_frame)
                        img_str = base64.b64encode(buffer).decode('utf-8')
                        
                        result = {
                            'image': f"data:image/jpeg;base64,{img_str}",
                            'posture_status': posture_status,
                            'angles': angles
                        }
                        
                        if self.result_queue.full():
                            self.result_queue.get()
                        self.result_queue.put(result)
                        
                        # 记录帧计数
                        self.frame_count += 1
                        
                        # 检查是否需要生成报告
                        current_time = time.time()
                        time_since_last_report = current_time - self.last_report_time
                        
                        if (posture_status == "bad" and 
                            self.frame_count % self.report_interval == 0 and
                            time_since_last_report >= self.min_report_interval and
                            self.use_tts):
                            
                            # 生成姿势报告
                            logger.info("生成姿势报告...")
                            advice = self.generate_advice(angles)
                            
                            # 使用TTS播报
                            logger.info(f"播报建议: {advice}")
                            try:
                                send_tts(advice)
                                self.last_report_time = current_time
                            except Exception as e:
                                logger.error(f"TTS播报失败: {e}")
                    
                    except Exception as e:
                        logger.error(f"处理检测结果时出错: {e}")
                else:
                    # 即使没有检测到人物，也更新结果队列
                    if self.result_queue.full():
                        self.result_queue.get()
                    
                    self.result_queue.put({
                        'posture_status': "unknown",
                        'angles': {},
                        'message': "未检测到人物"
                    })
                
                # 短暂休眠以减少CPU使用率
                time.sleep(0.01)
            
            # 关闭RTSP连接
            cap.release()
            logger.info("RTSP连接已关闭")
            
        except Exception as e:
            logger.error(f"RTSP检测线程出错: {e}")
            import traceback
            traceback.print_exc()
            self.running = False
    
    def generate_advice(self, angles):
        """生成简单的姿势建议"""
        # 简单的建议生成逻辑，实际项目中可能会调用LLM API
        neck_angle = angles.get('neck', 0)
        body_angle = angles.get('avg_body', 0)
        
        if neck_angle > 40:
            return "请注意颈部姿势，抬头减少低头角度，避免长期低头造成颈椎压力。适当活动颈部，做些简单的颈部伸展运动。"
        elif body_angle > 30:
            return "您的坐姿可能需要调整，请保持背部挺直，调整椅子高度和靠背，让脊柱自然伸展。每隔一小时起身活动一下。"
        else:
            return "您的姿势不太理想，建议调整坐姿，保持背部挺直，肩膀放松，颈部自然。定期起身活动，避免长时间保持同一姿势。"
    
    def set_baseline(self):
        """设置当前姿势为基准"""
        if self.frame_queue.empty():
            logger.error("帧队列为空，无法设置基准")
            return False
            
        frame = self.frame_queue.queue[0]  # 获取但不移除
        
        # 估计姿势
        kp_norm, _ = self.pose_estimator.get_pose(frame)
        
        if not kp_norm:
            logger.error("未检测到姿势关键点，无法设置基准")
            return False
            
        kp_abs = self._scale_keypoints(kp_norm, frame.shape)
        
        # 设置基准
        self.anomaly_detector.set_baseline(kp_abs)
        baseline_angles = self.anomaly_detector._compute_angles(kp_abs)
        
        logger.info(f"已设置基准姿势，角度: {baseline_angles}")
        return True
    
    def get_latest_result(self):
        """获取最新的检测结果"""
        if self.result_queue.empty():
            return None
        return self.result_queue.queue[0]  # 获取但不移除

def main():
    parser = argparse.ArgumentParser(description='RTSP姿势检测')
    parser.add_argument('--rtsp_url', type=str, default="rtsp://192.168.3.242:8554/live",
                      help='RTSP URL')
    parser.add_argument('--use_tts', action='store_true', help='使用TTS功能')
    parser.add_argument('--set_baseline', action='store_true', help='设置基准姿势')
    parser.add_argument('--run_time', type=int, default=0, 
                      help='运行时间(秒)，0表示一直运行直到按Ctrl+C')
    
    args = parser.parse_args()
    
    # 创建检测器
    detector = RTSPPoseDetector(rtsp_url=args.rtsp_url, use_tts=args.use_tts)
    
    try:
        # 启动检测
        detector.start()
        
        # 如果需要设置基准
        if args.set_baseline:
            print("等待3秒以获取稳定帧...")
            time.sleep(3)
            if detector.set_baseline():
                print("成功设置基准姿势!")
            else:
                print("设置基准姿势失败!")
        
        # 显示实时结果
        start_time = time.time()
        while True:
            # 检查是否达到运行时间限制
            if args.run_time > 0 and time.time() - start_time > args.run_time:
                print(f"已运行 {args.run_time} 秒，退出")
                break
                
            # 获取最新结果
            result = detector.get_latest_result()
            if result:
                posture = result.get('posture_status', 'unknown')
                angles = result.get('angles', {})
                
                # 打印结果
                print("\r", end="")  # 清除当前行
                print(f"姿势: {posture}, 颈部: {angles.get('neck', '--')}°, 身体: {angles.get('avg_body', '--')}°", end="")
            
            time.sleep(0.1)  # 短暂休眠以减少CPU使用率
            
    except KeyboardInterrupt:
        print("\n用户中断，停止检测")
    finally:
        # 停止检测
        detector.stop()

if __name__ == "__main__":
    main() 