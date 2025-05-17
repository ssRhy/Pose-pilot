'''
# Smart Risk Detector Flask App
# This Flask app provides endpoints for posture detection using YOLOv8 and pose estimation.
# It also integrates with Baidu Wenxin Yiyan API for generating posture reports.
'''


# 导入 RTSP 相关模块
import threading
import queue
import os
import base64
import time
import cv2
import numpy as np
import requests
import json
import logging
import sys
import io
import random

from flask import Flask, request, jsonify, send_from_directory, redirect, url_for
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.plot([0, 1], [0, 1])
plt.savefig('test.png')

# Import your custom modules
from yolo_detector import YOLODetector
from anomaly_detector import AnomalyDetector
from ultralytics import YOLO

# 导入IP音箱模块
from speaker.ip_speaker import send_tts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Fix console encoding for Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

########################################
# 1) Flask App Setup
########################################

from flask_cors import CORS

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

# Instantiate YOLO, Pose, and Anomaly modules
detector = YOLODetector(
    model_path="yolov8n-pose.pt",
    device="auto",
    conf_thres=0.7,
    iou_thres=0.45,
    gamma=1.0,
    advanced_box_filter=True
)

# Create dummy pose estimator if the real one fails due to network issues
try:
    from pose_estimator import PoseEstimator
    pose_estimator = PoseEstimator()
    logger.info("Successfully initialized PoseEstimator")
except Exception as e:
    logger.warning(f"Failed to initialize real PoseEstimator: {e}")
    logger.warning("Using dummy PoseEstimator for TTS testing")
    
    # Create a dummy pose estimator class for testing TTS functionality
    class DummyPoseEstimator:
        def get_pose(self, frame):
            # Return dummy keypoints and the original frame
            h, w = frame.shape[:2]
            # Create dummy keypoints in normalized coordinates (17 COCO keypoints)
            dummy_keypoints = [
                (0.5, 0.1),   # 0=nose
                (0.45, 0.1),  # 1=left_eye
                (0.55, 0.1),  # 2=right_eye
                (0.4, 0.12),  # 3=left_ear
                (0.6, 0.12),  # 4=right_ear
                (0.4, 0.3),   # 5=left_shoulder
                (0.6, 0.3),   # 6=right_shoulder
                (0.35, 0.45), # 7=left_elbow
                (0.65, 0.45), # 8=right_elbow
                (0.3, 0.6),   # 9=left_wrist
                (0.7, 0.6),   # 10=right_wrist
                (0.45, 0.6),  # 11=left_hip
                (0.55, 0.6),  # 12=right_hip
                (0.45, 0.8),  # 13=left_knee
                (0.55, 0.8),  # 14=right_knee
                (0.45, 0.95), # 15=left_ankle
                (0.55, 0.95)  # 16=right_ankle
            ]
            
            # Draw circles for keypoints on a copy of the frame
            annotated_frame = frame.copy()
            for x_norm, y_norm in dummy_keypoints:
                x, y = int(x_norm * w), int(y_norm * h)
                cv2.circle(annotated_frame, (x, y), 5, (0, 255, 0), -1)
            
            return dummy_keypoints, annotated_frame
    
    pose_estimator = DummyPoseEstimator()

# 创建队列来存储最新的检测结果
rtsp_frame_queue = queue.Queue(maxsize=1)
rtsp_result_queue = queue.Queue(maxsize=1)
rtsp_thread_running = False
rtsp_thread = None

# Create the anomaly detector instance
anomaly_detector = AnomalyDetector()

rtsp_url = "rtsp://10.148.165.1:8554/live"  # 默认使用localhost而非0.0.0.0

########################################
# 2) Helper Functions
#######################################

def scale_keypoints(kp_normalized, shape):
    """
    Convert normalized keypoints (range 0..1) to absolute pixel coordinates.
    """
    h, w, _ = shape
    return [(x * w, y * h) for x, y in kp_normalized]

def run_posture_detection(frame):
    """
    Run YOLO detection, pose estimation, convert keypoints to absolute values,
    and then check for deviation if a baseline is set.
    
    Returns:
       (annotated_frame, posture_status, angles)
    """
    # 1) Detect persons using YOLO.
    boxes, confs = detector.detect(frame)
    if not boxes:
        return frame, "no_person", {}

    # 2) Choose the detection with highest confidence.
    max_conf_idx = max(range(len(confs)), key=lambda i: confs[i])
    # (x1, y1, x2, y2) = boxes[max_conf_idx]  # Not used further in this example

    # 3) Estimate the pose.
    kp_norm, annotated_frame = pose_estimator.get_pose(frame)
    kp_abs = scale_keypoints(kp_norm, frame.shape)

    # 4) Compute the main angles using the anomaly detector helper.
    angles = anomaly_detector._compute_angles(kp_abs)
    
    # 5) Determine posture status:
    # If a baseline exists, compare against it.
    if anomaly_detector.has_baseline():
        is_deviated = anomaly_detector.is_deviated_from_baseline(kp_abs)
        posture_status = "bad" if is_deviated else "good"
    else:
        # Fallback: use the "fall_like" detection.
        is_bad = anomaly_detector.is_fall_like(kp_abs)
        posture_status = "bad" if is_bad else "good"

    return annotated_frame, posture_status, angles

########################################
# 3) Baidu API Functions
########################################

def get_access_token(api_key, secret_key):
    """获取百度API访问令牌"""
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": api_key, "client_secret": secret_key}
    # 显式禁用代理
    response = requests.post(url, params=params, proxies={"http": None, "https": None})
    token = response.json().get("access_token")
    print(f"获取令牌响应: {response.json()}")
    return token

def generate_posture_advice(token, posture_status, angles):
    """使用百度文心一言生成姿势建议"""
    # 如果没有有效令牌，使用备用建议
    if token is None:
        print("没有有效的LLM令牌，使用备用建议")
        return "未能连接到AI服务，请检查您的姿势，确保背部挺直，颈部保持自然角度，双肩放松。"
    
    # 只有当token有效时才定义和使用llm_url
    llm_url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-4.0-turbo-128k?access_token={token}"
    
    # 改进的提示词，生成更自然、更有个性的建议
    user_prompt = (
        f"你是一位关心办公人员健康的虚拟助手，你需要安慰崩溃的学生，安慰的语气且个性化。请根据以下姿势数据，给出一条简短、"
        f"有效且容易理解的改进建议(70-100字)。建议应该具体、实用，语气友好自然：\n\n"
        f"姿势状态：{posture_status}\n"
        f"颈部角度：{angles.get('neck', '--')}度\n"
        f"左侧身体角度：{angles.get('left_body', '--')}度\n"
        f"右侧身体角度：{angles.get('right_body', '--')}度\n"
        f"平均身体角度：{angles.get('avg_body', '--')}度\n\n"
        f"重要提示：根据度数实时给出建议，不要播报度数和数据，而是根据数据智能化但是专业的提出建议。"
    )
    
    print(f"发送给LLM的提示: {user_prompt}")
    
    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "penalty_score": 1,
        "enable_system_memory": False,
        "disable_search": True,
        "enable_citation": False,
        "enable_trace": False
    }, ensure_ascii=False)
    
    headers = {'Content-Type': 'application/json'}
    
    print("正在调用文心一言生成建议...")
    response = requests.post(llm_url, headers=headers, data=payload.encode("utf-8"), proxies={"http": None, "https": None})
    print(f"文心一言原始响应: {response.text}")
    
    try:
        result = response.json()
        print(f"文心一言解析响应: {result}")
        
        if 'result' in result:
            advice = result['result']
            print(f"成功从LLM获取建议: {advice}")
            return advice
        else:
            print(f"文心一言响应异常: {result}")
            # 使用备用建议
            return "未能获取个性化建议，请注意您的姿势，保持背部挺直，颈部放松。定期起身活动可减轻久坐带来的不适。"
    except Exception as e:
        print(f"解析文心一言响应时出错: {e}")
        return "系统暂时无法生成建议，请尝试保持良好坐姿，每小时起身活动5-10分钟，双眼注视远处放松视力。"

def get_posture_report(angles, posture_status):
    """整合姿势分析，生成文本建议"""
    try:
        # Baidu API credentials
        llm_api_key = "2YcoX4HqbcA6pMEolmNknwTQ"
        llm_secret_key = "3eeNrmrpVcKnBEes3MZrcQJeMxqfLhEH"
        
        # 清除可能存在的代理环境变量
        if 'http_proxy' in os.environ:
            del os.environ['http_proxy']
        if 'https_proxy' in os.environ:
            del os.environ['https_proxy']
        
        print("======== 开始生成姿势报告 ========")
        print(f"姿势数据: 状态={posture_status}, 角度={angles}")
        
        # 1. 获取LLM令牌
        print("\n--- 步骤1: 获取LLM访问令牌 ---")
        llm_token = get_access_token(llm_api_key, llm_secret_key)
        if not llm_token:
            print("警告: 无法获取LLM访问令牌，将使用备用建议")
        else:
            print(f"成功获取LLM令牌: {llm_token[:10]}...")
        
        # 2. 生成姿势建议
        print("\n--- 步骤2: 生成姿势建议 ---")
        ai_advice = generate_posture_advice(llm_token, posture_status, angles)
        print(f"最终生成的建议: {ai_advice}")
        
        # 3. 通过IP音箱输出语音
        try:
            print("\n--- 步骤3: 通过IP音箱输出语音 ---")
            send_tts(ai_advice)
            print("成功发送文本到IP音箱")
        except Exception as e:
            print(f"发送到IP音箱失败: {e}")
        
        print("\n--- 报告生成完成 ---")
        result = {
            "text": ai_advice,
            "ai_advice": ai_advice
        }
        print(f"最终结果: {result}")
        return result
        
    except Exception as e:
        print(f"Error in get_posture_report: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Error calling API: {e}", "text": "无法生成姿势报告"}

########################################
# 4) Baseline Capture Endpoint
########################################

@app.route("/capture_baseline", methods=["POST"])
def capture_baseline():
    """
    修改为直接从RTSP流获取当前帧并设置姿势基准线
    不再需要从请求中获取图像数据
    """
    try:
        # 检查RTSP是否在运行
        if not rtsp_thread_running:
            return jsonify({"success": False, "error": "RTSP stream not running"}), 400
        
        # 添加重试逻辑和更详细的日志
        max_attempts = 5
        attempt = 0
        
        while attempt < max_attempts:
            # 检查队列是否为空
            if rtsp_frame_queue.empty():
                print(f"尝试 #{attempt+1}: rtsp_frame_queue 为空，等待...")
                time.sleep(0.5)  # 等待0.5秒后重试
                attempt += 1
                continue
            
            # 从RTSP帧队列获取当前帧
            frame = rtsp_frame_queue.queue[0]  # 获取但不移除最新帧
            print(f"成功从队列获取帧，大小: {frame.shape}")
            
            # 估计姿势并获取关键点
            kp_norm, annotated_frame = pose_estimator.get_pose(frame)
            kp_abs = scale_keypoints(kp_norm, frame.shape)
            
            print(f"获取到的关键点数量: {len(kp_abs)}")
            
            if len(kp_abs) < 15:
                print(f"关键点不足 ({len(kp_abs)}/15)，重试...")
                time.sleep(0.5)
                attempt += 1
                continue
            
            # 设置anomaly detector中的基准姿势
            anomaly_detector.set_baseline(kp_abs)
            baseline_angles = anomaly_detector._compute_angles(kp_abs)
            
            # 将图像帧编码为base64以便返回给前端
            _, buffer = cv2.imencode(".jpg", annotated_frame)
            annotated_b64 = base64.b64encode(buffer).decode("utf-8")
            annotated_b64_url = f"data:image/jpeg;base64,{annotated_b64}"
            
            print(f"成功设置基准，角度: {baseline_angles}")
            
            return jsonify({
                "success": True,
                "baseline": kp_abs,
                "angles": baseline_angles,
                "annotated_image": annotated_b64_url
            })
        
        # 如果重试了多次仍然失败
        return jsonify({"success": False, "error": "Failed to get valid frame after multiple attempts"}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Error setting baseline: {str(e)}"}), 500

########################################
# 5) Posture Detection Endpoint
########################################

@app.route("/detect_posture", methods=["POST"])
def detect_posture():
    """
    Expects JSON:
      { "image": "data:image/jpeg;base64,..." }
    Runs detection, pose estimation, and generates a posture report using Baidu API
    if the detected posture is bad.
    Returns JSON with posture status, angles, annotated image, and text advice.
    """
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No 'image' field in request"}), 400

    try:
        header, b64_data = data["image"].split(",", 1)
    except ValueError:
        return jsonify({"error": "Invalid image format"}), 400

    img_bytes = base64.b64decode(b64_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Invalid image data"}), 400

    annotated_frame, posture_status, angles = run_posture_detection(frame)

    # Encode the annotated frame back to a base64 URL
    _, buffer = cv2.imencode(".jpg", annotated_frame)
    annotated_b64 = base64.b64encode(buffer).decode("utf-8")
    annotated_b64_url = f"data:image/jpeg;base64,{annotated_b64}"

    # Generate a report using Baidu API if posture is bad
    report_data = {"text": ""}
    if posture_status == "bad":
        report_data = get_posture_report(angles, posture_status)

    return jsonify({
        "posture": posture_status,
        "annotated_image": annotated_b64_url,
        "angles": angles,
        "report": report_data.get("text", "")
    })

########################################
# 6) Test TTS Endpoint
########################################

@app.route("/test_tts", methods=["GET"])
def test_tts_endpoint():
    """测试端点，模拟坏姿势并生成文本报告，同时通过IP音箱输出"""
    # 模拟角度数据
    angles = {
        'neck': 100,
        'left_body': 70,
        'right_body': 68,
        'avg_body': 90
    }
    posture_status = "bad"
    
    try:
        # 生成报告
        print("\n\n========== 处理新请求 ==========")
        report_data = get_posture_report(angles, posture_status)
        
        response = {
            "posture": posture_status,
            "angles": angles,
            "report": report_data.get("text", ""),
            "ai_advice": report_data.get("ai_advice", "")
        }
        
        print(f"返回给客户端的响应: {response}")
        return jsonify(response)
    except Exception as e:
        error_msg = f"处理请求时出错: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return jsonify({"error": error_msg})

########################################
# 7) RTSP Stream Processing
########################################
def rtsp_detection_thread():
    """
    后台线程，从 RTSP 流读取帧并进行检测
    增强版：自动检测不良姿势并生成语音建议
    """
    global rtsp_thread_running
    global rtsp_url  # 使用全局变量，允许通过API更改
    
    # 添加姿势状态跟踪和建议生成频率控制
    last_posture_status = None
    last_advice_time = 0
    continuous_bad_posture_time = 0  # 连续不良姿势的时长(秒)
    
    if not rtsp_url:
        rtsp_url = "rtsp://10.148.165.1:8554/live"  # 默认使用localhost而非0.0.0.0
    
    print(f"RTSP检测线程启动，连接到: {rtsp_url}")
    print("已启用自动姿势建议生成功能")
    
    try:
        # 检查RTSP URL是否有效
        if not rtsp_url or not rtsp_url.startswith('rtsp://'):
            print(f"错误: 无效的RTSP URL: {rtsp_url}")
            print("RTSP URL必须以'rtsp://'开头")
            rtsp_thread_running = False
            return
            
        print(f"尝试连接到RTSP流: {rtsp_url}")
        
        # 设置RTSP连接
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;15000000"  # 设置15秒超时
        
        # 尝试打开RTSP流
        print("创建VideoCapture对象...")
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        
        print("设置缓冲区大小...")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        
        # 设置读取超时
        if hasattr(cv2, 'CAP_PROP_OPEN_TIMEOUT'):
            print("设置连接超时为15秒...")
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT, 15000)  # 15秒连接超时
        if hasattr(cv2, 'CAP_PROP_READ_TIMEOUT'):
            print("设置读取超时为5秒...")
            cap.set(cv2.CAP_PROP_READ_TIMEOUT, 5000)   # 5秒读取超时
        
        if not cap.isOpened():
            print(f"错误: 无法连接到RTSP流 {rtsp_url}")
            print("请检查: 1) 设备是否开启 2) IP地址是否正确 3) 端口是否正确 4) 路径是否正确")
            rtsp_thread_running = False
            return
        
        print("RTSP detection thread started successfully!")
        
        while rtsp_thread_running:
            # 记录当前时间用于跟踪不良姿势持续时间
            current_time = time.time()
            
            # 读取帧
            ret, frame = cap.read()
            
            # 如果读取失败
            if not ret:
                print("帧读取失败，尝试重新连接...")
                # 尝试关闭旧连接
                cap.release()
                # 等待一小段时间
                time.sleep(1)
                # 重新连接
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                cap.set(cv2.CAP_PROP_RTSP_TRANSPORT, cv2.CAP_RTSP_TCP)
                # 重置姿势追踪状态
                last_posture_status = None
                continuous_bad_posture_time = 0
                continue
            
            # 检测人物
            boxes, confs = detector.detect(frame)
            
            # 如果检测到人物，进行姿态估计
            if boxes:
                # 选择置信度最高的检测框
                max_conf_idx = max(range(len(confs)), key=lambda i: confs[i])
                
                # 估计姿态
                kp_norm, annotated_frame = pose_estimator.get_pose(frame)
                kp_abs = scale_keypoints(kp_norm, frame.shape)
                
                # 计算角度
                angles = anomaly_detector._compute_angles(kp_abs)
                print(f"姿态角度: 颈部={angles.get('neck', '--')}°, 身体={angles.get('avg_body', '--')}°")
                
                # 确定姿态状态
                if anomaly_detector.has_baseline():
                    is_deviated = anomaly_detector.is_deviated_from_baseline(kp_abs)
                    posture_status = "bad" if is_deviated else "good"
                else:
                    is_bad = anomaly_detector.is_fall_like(kp_abs)
                    posture_status = "bad" if is_bad else "good"
                
                print(f"姿态状态: {posture_status}")
                
                # ===== 新增：自动姿势建议生成逻辑 =====
                # 1. 检查姿势状态变化
                if posture_status == "bad":
                    # 当姿势不良时，更新连续不良姿势的时长
                    if last_posture_status == "bad":
                        continuous_bad_posture_time += time.time() - current_time
                    else:
                        continuous_bad_posture_time = 0
                    
                    # 在以下情况生成建议:
                    # a) 姿势状态从"good"变为"bad"
                    # b) 连续保持不良姿势超过60秒
                    # c) 距离上次建议已超过3分钟
                    should_generate_advice = (
                        (last_posture_status != "bad") or 
                        (continuous_bad_posture_time > 60) or
                        (current_time - last_advice_time > 180)
                    )
                    
                    if should_generate_advice:
                        try:
                            print("\n===== 检测到不良姿势，自动生成建议 =====")
                            report_data = get_posture_report(angles, posture_status)
                            advice_text = report_data.get("text", "")
                            
                            if advice_text:
                                print(f"建议内容: {advice_text}")
                                # 更新结果队列中的建议字段，以便前端可以显示
                                if not rtsp_result_queue.empty():
                                    current_result = rtsp_result_queue.queue[0]
                                    # 如果当前队列中有结果，更新它的建议字段
                                    current_result['advice'] = advice_text
                            
                            # 更新上次建议时间
                            last_advice_time = current_time
                            continuous_bad_posture_time = 0  # 重置连续不良姿势时长
                            
                        except Exception as e:
                            print(f"生成自动建议时出错: {e}")
                else:
                    # 姿势良好时，重置连续不良姿势时长
                    continuous_bad_posture_time = 0
                
                # 更新上一次姿势状态
                last_posture_status = posture_status
                # ===== 自动姿势建议生成逻辑结束 =====
                
                # 将结果放入队列
                try:
                    # 转换为 JPEG 格式
                    _, buffer = cv2.imencode('.jpg', annotated_frame)
                    img_str = base64.b64encode(buffer).decode('utf-8')
                    
                    # 更新结果队列，如果队列已满则移除旧的结果
                    if rtsp_result_queue.full():
                        rtsp_result_queue.get()
                    
                    # 创建结果字典
                    result_dict = {
                        'image': f"data:image/jpeg;base64,{img_str}",
                        'posture_status': posture_status,
                        'angles': angles
                    }
                    
                    # 如果最近有生成建议，添加到结果中
                    if posture_status == "bad" and current_time - last_advice_time < 60:
                        # 尝试从之前的结果中获取建议
                        if not rtsp_result_queue.empty() and 'advice' in rtsp_result_queue.queue[0]:
                            result_dict['advice'] = rtsp_result_queue.queue[0]['advice']
                    
                    rtsp_result_queue.put(result_dict)
                    
                    # 更新帧队列
                    print(f"当前rtsp_frame_queue大小: {rtsp_frame_queue.qsize()}/{rtsp_frame_queue.maxsize}")
                    if rtsp_frame_queue.full():
                        rtsp_frame_queue.get()
                    rtsp_frame_queue.put(frame)
                except Exception as e:
                    print(f"处理RTSP帧时出错: {e}")
            else:
                # 未检测到人物，重置跟踪状态
                last_posture_status = None
                continuous_bad_posture_time = 0
                
                # 即使没有检测到人物，也至少更新一下帧队列，以便baseline等功能使用
                if rtsp_frame_queue.full():
                    rtsp_frame_queue.get()
                rtsp_frame_queue.put(frame)
                
                # 也更新结果队列，防止"No RTSP detection results available"错误
                try:
                    # 转换为 JPEG 格式
                    _, buffer = cv2.imencode('.jpg', frame)
                    img_str = base64.b64encode(buffer).decode('utf-8')
                    
                    # 更新结果队列，如果队列已满则移除旧的结果
                    if rtsp_result_queue.full():
                        rtsp_result_queue.get()
                    
                    rtsp_result_queue.put({
                        'image': f"data:image/jpeg;base64,{img_str}",
                        'posture_status': "unknown",
                        'angles': {},
                        'message': "未检测到人物"
                    })
                except Exception as e:
                    print(f"处理RTSP帧时出错: {e}")
            
            # 短暂休眠以减少 CPU 使用率
            time.sleep(0.01)
        
        # 关闭 RTSP 连接
        cap.release()
        print("RTSP detection thread stopped")
    
    except Exception as e:
        print(f"RTSP检测线程错误: {e}")
        import traceback
        traceback.print_exc()
        rtsp_thread_running = False

@app.route("/rtsp/start", methods=["POST"])
def start_rtsp():
    """
    启动 RTSP 检测线程
    """
    global rtsp_thread, rtsp_thread_running, rtsp_url
    
    # 如果线程已经在运行，返回成功
    if rtsp_thread_running and rtsp_thread and rtsp_thread.is_alive():
        return jsonify({"success": True, "message": "RTSP detection already running"})
    
    # 更新 RTSP URL（如果提供）
    try:
        data = request.get_json()
        if data and 'rtsp_url' in data:
            rtsp_url = data['rtsp_url']
            print(f"RTSP URL已更新为: {rtsp_url}")
    except Exception as e:
        # 如果没有JSON数据或解析失败，忽略错误并继续
        print(f"No JSON data in request or parse error: {e}")
    
    # 启动线程
    rtsp_thread_running = True
    rtsp_thread = threading.Thread(target=rtsp_detection_thread)
    rtsp_thread.daemon = True
    rtsp_thread.start()
    
    return jsonify({"success": True, "message": "RTSP detection started"})

@app.route("/rtsp/stop", methods=["POST"])
def stop_rtsp():
    """
    停止 RTSP 检测线程
    """
    global rtsp_thread_running
    
    if rtsp_thread_running:
        rtsp_thread_running = False
        # 等待线程结束
        if rtsp_thread:
            rtsp_thread.join(timeout=5.0)
        
        # 清空队列
        while not rtsp_frame_queue.empty():
            rtsp_frame_queue.get()
        while not rtsp_result_queue.empty():
            rtsp_result_queue.get()
        
        return jsonify({"success": True, "message": "RTSP detection stopped"})
    else:
        return jsonify({"success": True, "message": "RTSP detection not running"})

@app.route("/rtsp/status", methods=["GET"])
def rtsp_status():
    """
    获取 RTSP 检测状态
    """
    return jsonify({
        "running": rtsp_thread_running and rtsp_thread and rtsp_thread.is_alive()
    })

@app.route("/rtsp/latest", methods=["GET"])
def rtsp_latest():
    """
    获取最新的 RTSP 检测结果
    """
    global rtsp_thread_running, rtsp_thread
    
    # 检查RTSP线程状态，提供更详细的错误信息
    if not rtsp_thread_running:
        return jsonify({
            "success": False, 
            "message": "RTSP detection is not running",
            "error_code": "not_running"
        })
    
    if rtsp_thread and not rtsp_thread.is_alive():
        return jsonify({
            "success": False, 
            "message": "RTSP detection thread has stopped unexpectedly",
            "error_code": "thread_stopped"
        })
    
    if rtsp_result_queue.empty():
        # 提供更具体的错误信息
        return jsonify({
            "success": False, 
            "message": "No RTSP detection results available",
            "error_code": "no_results",
            "details": "Camera may be connecting or no frames have been processed yet"
        })
    
    result = rtsp_result_queue.queue[0]  # 获取但不移除
    return jsonify({"success": True, "result": result})

@app.route("/rtsp/set_url", methods=["POST"])
def set_rtsp_url():
    """
    设置 RTSP URL
    """
    data = request.get_json()
    if not data or "rtsp_url" not in data:
        return jsonify({"error": "No 'rtsp_url' field in request"}), 400
    
    # 需要重新启动线程才能应用新的URL
    restart_required = rtsp_thread_running
    
    # 如果线程正在运行，先停止
    if restart_required:
        stop_rtsp()
    
    # 如果需要重启，启动线程
    if restart_required:
        start_rtsp()
    
    return jsonify({"success": True, "message": f"RTSP URL updated, thread restarted: {restart_required}"})

@app.route("/rtsp/report", methods=["GET"])
def rtsp_report():
    """
    获取最新的 RTSP 检测结果，并生成姿势报告，同时通过IP音箱输出
    """
    if rtsp_result_queue.empty():
        return jsonify({"success": False, "message": "No RTSP detection results available"})
    
    result = rtsp_result_queue.queue[0]  # 获取但不移除
    posture_status = result.get('posture_status')
    angles = result.get('angles', {})
    
    # 只有当姿势状态为 "bad" 时才生成报告
    if posture_status == "bad":
        # 生成报告
        report_data = get_posture_report(angles, posture_status)
        
        # 添加到结果中
        result['advice'] = report_data.get('text', '')
    # 添加对"unknown"状态的处理
    elif posture_status == "unknown":
        # 不生成报告，但添加一个信息
        result['advice'] = "未检测到人物，无法生成姿势报告"
    
    return jsonify({"success": True, "result": result})

@app.route("/rtsp/pose_data", methods=["GET"])
def rtsp_pose_data():
    """
    获取最新的姿态数据，格式化为易读的HTML
    """
    if rtsp_result_queue.empty():
        return "<h1>无可用姿态数据</h1><p>请确保RTSP检测线程正在运行且已检测到人物</p>"
    
    result = rtsp_result_queue.queue[0]  # 获取但不移除
    posture_status = result.get('posture_status', '未知')
    angles = result.get('angles', {})
    
    # 构建HTML响应
    html = f"""
    <html>
    <head>
        <title>姿态数据</title>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; max-width: 800px; margin: 0 auto; }}
            h1 {{ color: #333; }}
            .data-container {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
            .status {{ font-size: 24px; margin: 10px 0; }}
            .good {{ color: green; }}
            .bad {{ color: red; }}
            .unknown {{ color: gray; }}
            .angles {{ margin-top: 20px; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>实时姿态数据</h1>
        <div class="data-container">
            <div class="status {posture_status}">
                姿势状态: {posture_status.upper()}
            </div>
            <div class="angles">
                <h2>角度数据:</h2>
                <table>
                    <tr>
                        <th>指标</th>
                        <th>角度值</th>
                    </tr>
                    <tr>
                        <td>颈部角度</td>
                        <td>{angles.get('neck', '--')}°</td>
                    </tr>
                    <tr>
                        <td>左侧身体角度</td>
                        <td>{angles.get('left_body', '--')}°</td>
                    </tr>
                    <tr>
                        <td>右侧身体角度</td>
                        <td>{angles.get('right_body', '--')}°</td>
                    </tr>
                    <tr>
                        <td>平均身体角度</td>
                        <td>{angles.get('avg_body', '--')}°</td>
                    </tr>
                </table>
            </div>
        </div>
    </body>
    </html>
    """
    return html

@app.route("/get_advice", methods=["GET"])
def get_advice():
    """
    简单端点，返回AI生成的文本建议，使用随机生成的数据模拟实时效果
    不需要RTSP数据
    """
    # 生成随机变化的姿势数据，模拟实时数据
    
    # 使用随机数模拟实时数据变化
    # 颈部角度范围(80-110)，90以下通常为良好姿势
    neck_angle = random.randint(80, 110)
    # 身体角度范围(55-80)，70以下通常为良好姿势
    left_body = random.randint(55, 80)
    right_body = random.randint(55, 80)
    # 计算平均身体角度
    avg_body = (left_body + right_body) // 2
    
    # 基于角度确定姿势状态
    posture_status = "bad" if (neck_angle > 95 or avg_body > 70) else "good"
    
    angles = {
        'neck': neck_angle,
        'left_body': left_body,
        'right_body': right_body,
        'avg_body': avg_body
    }
    
    try:
        # 生成报告
        report_data = get_posture_report(angles, posture_status)
        advice = report_data.get("text", "无法生成建议")
        
        # 返回纯文本或JSON
        format_type = request.args.get('format', 'json')
        if format_type == 'text':
            return advice
        else:
            return jsonify({
                "advice": advice,
                "posture": posture_status,
                "angles": angles,
                "timestamp": time.time()  # 添加时间戳表示实时性
            })
    except Exception as e:
        return jsonify({"error": f"生成建议时出错: {str(e)}"})

########################################
# 8) Serve the Index HTML
########################################

@app.route("/")
def serve_index():
    """
    Redirect to the advice page instead of serving index.html
    """
    return redirect(url_for('serve_advice'))

@app.route("/advice")
def serve_advice():
    """
    Serve the advice.html file for a simplified interface focused only on text advice.
    """
    return send_from_directory(app.static_folder, "advice.html")

# 静态文件服务
@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory(app.static_folder, path)

########################################
# 9) Run the Flask App
########################################

if __name__ == "__main__":
    # 确保static文件夹存在
    os.makedirs(os.path.join(app.static_folder, "audio"), exist_ok=True)
    
    # 自动启动RTSP检测线程
    print("自动启动RTSP检测线程...")
    rtsp_thread_running = True
    rtsp_thread = threading.Thread(target=rtsp_detection_thread)
    rtsp_thread.daemon = True
    rtsp_thread.start()
    
    app.run(host="0.0.0.0", port=5000, debug=True)
