'''
# Smart Risk Detector Flask App
# This Flask app provides endpoints for posture detection using YOLOv8 and pose estimation.
# It also integrates with Baidu Wenxin Yiyan API for generating posture reports and Baidu TTS for audio.
'''
from dotenv import load_dotenv
load_dotenv()

# 导入 RTSP 相关模块

import threading
import queue
import os
import base64
import time
import cv2
import numpy as np
import requests
import urllib.parse
import json
import logging

from flask import Flask, request, jsonify, send_from_directory, redirect, url_for
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.plot([0, 1], [0, 1])
plt.savefig('test.png')
# Import your custom modules
from yolo_detector import YOLODetector
from anomaly_detector import AnomalyDetector

# 添加简单RTSP测试功能
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

########################################
# 1) Flask App Setup
########################################

app = Flask(__name__, static_folder='static')

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
}

# 简单RTSP测试队列和线程控制变量
simple_rtsp_frame_queue = queue.Queue(maxsize=1)
simple_rtsp_thread_running = False
simple_rtsp_thread = None

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

def simple_rtsp_thread(rtsp_url):
    """简单RTSP检测线程，使用YOLOv8进行目标检测"""
    global simple_rtsp_thread_running
    
    print(f"简单RTSP线程启动，连接到: {rtsp_url}")
    
    try:
        # 加载YOLOv8模型
        model = YOLO("yolov8n-pose.pt")
        
        # 设置RTSP连接
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        
        if not cap.isOpened():
            print(f"错误: 无法连接到RTSP流 {rtsp_url}")
            simple_rtsp_thread_running = False
            return
        
        frame_count = 0
        start_time = time.time()
        fps_update_interval = 30
        
        while simple_rtsp_thread_running:
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
            
            # 将处理后的帧放入队列
            try:
                # 转换为JPEG格式
                _, buffer = cv2.imencode('.jpg', display_frame)
                
                # 更新帧队列
                if simple_rtsp_frame_queue.full():
                    simple_rtsp_frame_queue.get()
                simple_rtsp_frame_queue.put(buffer)
            except Exception as e:
                print(f"处理RTSP帧时出错: {e}")
            
            # 短暂休眠以减少CPU使用率
            time.sleep(0.01)
    
    except Exception as e:
        print(f"简单RTSP线程错误: {e}")
    
    finally:
        # 释放资源
        if 'cap' in locals():
            cap.release()
        simple_rtsp_thread_running = False
        print("简单RTSP线程已停止")

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

# 添加简单RTSP接口
@app.route("/simple_rtsp/start", methods=["POST"])
def start_simple_rtsp():
    """启动简单RTSP检测线程"""
    global simple_rtsp_thread, simple_rtsp_thread_running
    
    # 如果线程已经在运行，返回成功
    if simple_rtsp_thread_running and simple_rtsp_thread and simple_rtsp_thread.is_alive():
        return jsonify({"success": True, "message": "简单RTSP检测已在运行"})
    
    # 获取RTSP URL
    data = request.get_json()
    rtsp_url = data.get("rtsp_url", "rtsp://10.148.165.1:8554/live")
    
    # 启动线程
    simple_rtsp_thread_running = True
    simple_rtsp_thread = threading.Thread(target=simple_rtsp_thread, args=(rtsp_url,))
    simple_rtsp_thread.daemon = True
    simple_rtsp_thread.start()
    
    return jsonify({"success": True, "message": "简单RTSP检测已启动"})

@app.route("/simple_rtsp/stop", methods=["POST"])
def stop_simple_rtsp():
    """停止简单RTSP检测线程"""
    global simple_rtsp_thread_running
    
    if simple_rtsp_thread_running:
        simple_rtsp_thread_running = False
        # 等待线程结束
        if simple_rtsp_thread:
            simple_rtsp_thread.join(timeout=5.0)
        
        # 清空队列
        while not simple_rtsp_frame_queue.empty():
            simple_rtsp_frame_queue.get()
        
        return jsonify({"success": True, "message": "简单RTSP检测已停止"})
    else:
        return jsonify({"success": True, "message": "简单RTSP检测未运行"})

@app.route("/simple_rtsp/status", methods=["GET"])
def simple_rtsp_status():
    """获取简单RTSP检测状态"""
    return jsonify({
        "running": simple_rtsp_thread_running and simple_rtsp_thread and simple_rtsp_thread.is_alive()
    })

@app.route("/simple_rtsp/frame", methods=["GET"])
def get_simple_rtsp_frame():
    """获取最新的简单RTSP检测帧"""
    if simple_rtsp_frame_queue.empty():
        return jsonify({"success": False, "message": "没有可用的RTSP帧"})
    
    # 获取最新帧
    buffer = simple_rtsp_frame_queue.queue[0]  # 获取但不移除
    img_str = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        "success": True, 
        "image": f"data:image/jpeg;base64,{img_str}"
    })

########################################
# 2) Helper Functions
########################################

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
        f"你是一位关心办公人员健康的虚拟助手，语气友好且个性化。请根据以下姿势数据，给出一条简短、"
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

def generate_audio(token, text):
    """使用百度TTS API生成音频"""
    if not token:
        print("没有有效的TTS令牌，无法生成音频")
        return None
        
    tts_url = "http://tsn.baidu.com/text2audio"
    
    # URL编码文本内容
    encoded_text = urllib.parse.quote_plus(text)
    
    # 准备表单数据
    data = {
        "tex": encoded_text,
        "lan": "zh",
        "cuid": "posepilot",
        "ctp": "1",
        "aue": "3",
        "tok": token,
        "audio_ctrl": '{"sampling_rate":16000}'
    }
    
    print(f"发送TTS请求，文本内容: {text}")
    response = requests.post(tts_url, data=data, proxies={"http": None, "https": None})
    
    # 检查响应内容类型
    content_type = response.headers.get("Content-Type", "")
    print(f"TTS响应内容类型: {content_type}")
    
    if content_type.startswith("audio/"):
        # 保存音频文件
        timestamp = int(time.time())
        audio_dir = os.path.join(app.static_folder, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        
        audio_file = f"report_{timestamp}.mp3"
        audio_path = os.path.join(audio_dir, audio_file)
        
        with open(audio_path, "wb") as f:
            f.write(response.content)
        
        print(f"音频文件已保存到: {audio_path}")
        return f"/static/audio/{audio_file}"
    else:
        try:
            error_text = response.text[:200]  # 只打印前200个字符，避免日志过长
            print(f"TTS API非音频响应: {error_text}...")
            
            try:
                error_json = response.json()
                print(f"TTS API错误: {error_json}")
            except:
                pass
                
            return None
        except:
            print(f"无法解析TTS API响应")
            return None

def get_posture_report(angles, posture_status):
    """整合姿势分析，生成文本建议和语音报告"""
    try:
        # Baidu API credentials
        llm_api_key = "2YcoX4HqbcA6pMEolmNknwTQ"
        llm_secret_key = "3eeNrmrpVcKnBEes3MZrcQJeMxqfLhEH"
        tts_api_key = "UONsI3GbC0ABHkuWS2b8coxG" 
        tts_secret_key = "FpvrKuHJjVJrf1K6HEuhM7wtXsTqFO6K"
        
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
        
        # 3. 组合完整报告文本
        print("\n--- 步骤3: 组合完整报告文本 ---")
        text_content = ai_advice
        print(f"完整报告文本: {text_content}")
        
        # 4. 获取TTS令牌并生成音频
        print("\n--- 步骤4: 获取TTS令牌 ---")
        tts_token = get_access_token(tts_api_key, tts_secret_key)
        if not tts_token:
            print("警告: 无法获取TTS访问令牌，将只返回文本报告")
            audio_url = ""
        else:
            print(f"成功获取TTS令牌: {tts_token[:10]}...")
            
            print("\n--- 步骤5: 生成语音 ---")
            audio_url = generate_audio(tts_token, text_content) or ""
            print(f"生成的音频URL: {audio_url}")
        
        print("\n--- 报告生成完成 ---")
        result = {
            "audio_url": audio_url,
            "text": text_content,
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
        # 检查RTSP是否连接
        if not rtsp_thread_running or rtsp_frame_queue.empty():
            return jsonify({"success": False, "error": "RTSP stream not running or no frames available"}), 400
        
        # 从RTSP帧队列获取当前帧
        frame = rtsp_frame_queue.queue[0]  # 获取但不移除最新帧
        
        # 估计姿势并获取关键点
        kp_norm, annotated_frame = pose_estimator.get_pose(frame)
        kp_abs = scale_keypoints(kp_norm, frame.shape)

        if len(kp_abs) < 15:
            return jsonify({"success": False, "error": "Not enough keypoints to set baseline."})

        # 设置anomaly detector中的基准姿势
        anomaly_detector.set_baseline(kp_abs)
        baseline_angles = anomaly_detector._compute_angles(kp_abs)
        
        # 将图像帧编码为base64以便返回给前端
        _, buffer = cv2.imencode(".jpg", annotated_frame)
        annotated_b64 = base64.b64encode(buffer).decode("utf-8")
        annotated_b64_url = f"data:image/jpeg;base64,{annotated_b64}"

        return jsonify({
            "success": True,
            "baseline": kp_abs,
            "angles": baseline_angles,
            "annotated_image": annotated_b64_url
        })
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
    Returns JSON with posture status, angles, annotated image, and audio.
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
    report_data = {"text": "", "audio_url": ""}
    if posture_status == "bad":
        report_data = get_posture_report(angles, posture_status)

    return jsonify({
        "posture": posture_status,
        "annotated_image": annotated_b64_url,
        "angles": angles,
        "report": report_data.get("text", ""),
        "audio_url": report_data.get("audio_url", "")
    })

########################################
# 6) Test TTS Endpoint
########################################

@app.route("/test_tts", methods=["GET"])
def test_tts_endpoint():
    """测试端点，模拟坏姿势并生成语音报告"""
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
            "ai_advice": report_data.get("ai_advice", ""),
            "audio_url": report_data.get("audio_url", "")
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
    使用简单RTSP实现而非RTSPDetector
    """
    global rtsp_thread_running
    rtsp_url = "rtsp://10.148.165.1:8554/live"  # 默认RTSP地址，请根据MaixCAM输出的URL进行修改
    
    print(f"RTSP检测线程启动，连接到: {rtsp_url}")
    
    try:
        # 设置RTSP连接
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
       
        
        if not cap.isOpened():
            print(f"错误: 无法连接到RTSP流 {rtsp_url}")
            rtsp_thread_running = False
            return
        
        print("RTSP detection thread started successfully!")
        
        while rtsp_thread_running:
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
                continue
            
            # 检测人物
            boxes, confs = detector.detect(frame)
            
            # 打印检测结果
            if boxes:
                print(f"检测到 {len(boxes)} 个物体:")
                for i, (box, conf) in enumerate(zip(boxes, confs)):
                    print(f"  物体 {i+1}: 置信度 {conf:.2f}, 位置 {box}")
            else:
                print("未检测到物体")
            
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
                
                # 将结果放入队列
                try:
                    # 转换为 JPEG 格式
                    _, buffer = cv2.imencode('.jpg', annotated_frame)
                    img_str = base64.b64encode(buffer).decode('utf-8')
                    
                    # 更新结果队列，如果队列已满则移除旧的结果
                    if rtsp_result_queue.full():
                        rtsp_result_queue.get()
                    
                    rtsp_result_queue.put({
                        'image': f"data:image/jpeg;base64,{img_str}",
                        'posture_status': posture_status,
                        'angles': angles
                    })
                    
                    # 更新帧队列
                    if rtsp_frame_queue.full():
                        rtsp_frame_queue.get()
                    rtsp_frame_queue.put(annotated_frame)
                except Exception as e:
                    print(f"处理RTSP帧时出错: {e}")
            else:
                # 即使没有检测到人物，也至少更新一下帧队列，以便baseline等功能使用
                if rtsp_frame_queue.full():
                    rtsp_frame_queue.get()
                rtsp_frame_queue.put(frame)
            
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
    global rtsp_thread, rtsp_thread_running
    
    # 如果线程已经在运行，返回成功
    if rtsp_thread_running and rtsp_thread and rtsp_thread.is_alive():
        return jsonify({"success": True, "message": "RTSP detection already running"})
    
    # 更新 RTSP URL（如果提供）
    data = request.get_json()
    
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
    if rtsp_result_queue.empty():
        return jsonify({"success": False, "message": "No RTSP detection results available"})
    
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
    获取最新的 RTSP 检测结果，并生成姿势报告
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
        result['audio_url'] = report_data.get('audio_url', '')
    
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
    
    # 生成HTML显示
    html = f"""
    <html>
    <head>
        <title>姿态数据</title>
        <meta http-equiv="refresh" content="2"> <!-- 每2秒刷新 -->
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            .data {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .good {{ color: green; }}
            .bad {{ color: red; }}
            .angle {{ font-size: 18px; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>实时姿态数据</h1>
        <div class="data">
            <h2>姿态状态: <span class="{posture_status}">{posture_status}</span></h2>
            <div class="angle">颈部角度: {angles.get('neck', '--')}°</div>
            <div class="angle">平均身体角度: {angles.get('avg_body', '--')}°</div>
            <div class="angle">左侧身体角度: {angles.get('left_body', '--')}°</div>
            <div class="angle">右侧身体角度: {angles.get('right_body', '--')}°</div>
        </div>
        <p><a href="/rtsp/pose_data">手动刷新</a> | <a href="/">返回主页</a></p>
    </body>
    </html>
    """
    
    return html

########################################
# 8) Serve the Index HTML
########################################

@app.route("/")
def serve_index():
    """
    Serve the index.html file from the 'static' folder.
    """
    return send_from_directory(app.static_folder, "index.html")

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
    
    # 默认运行在localhost:5000
    app.run(host="0.0.0.0", port=5000, debug=True)