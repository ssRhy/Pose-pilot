#main.py
'''
# Smart Risk Detector Flask App
# This Flask app provides endpoints for posture detection using YOLOv8 and pose estimation.
# It also integrates with the Gemini API for generating posture reports.
'''
from dotenv import load_dotenv
load_dotenv()

import os
import base64
import time
import cv2
import numpy as np

from flask import Flask, request, jsonify, send_from_directory
from google import genai  # Import the Google Gen AI client

# Import your custom modules
from yolo_detector import YOLODetector
from pose_estimator import PoseEstimator
from anomaly_detector import AnomalyDetector

########################################
# 1) Flask App Setup
########################################

app = Flask(__name__, static_folder='static')

# Instantiate YOLO, Pose, and Anomaly modules
detector = YOLODetector(
    model_path="yolov8n.pt",
    device="auto",
    conf_thres=0.7,
    iou_thres=0.45,
    gamma=1.0,
    advanced_box_filter=True
)

pose_estimator = PoseEstimator()

# Instantiate your anomaly detector using the expected argument names.
anomaly_detector = AnomalyDetector(
    body_angle_threshold=55, 
    neck_angle_threshold=50,
    history_window=10,
    deviation_angle_threshold=15
)

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

def get_posture_report(angles, posture_status):
    """
    Build a prompt from the detected angles and posture status and
    call the Gemini API (using Google Gen AI) to generate a report.
    
    Ensure that the GEMINI_API_KEY environment variable is set.
    """
    gemini_api_key = os.environ.get("")
    if not gemini_api_key:
        return "Gemini API key not set."

    # Instantiate the Gen AI client with your API key.
    client = genai.Client(api_key=gemini_api_key)

    prompt = (
        f"Posture Report:\n"
        f"Status: {posture_status}\n"
        f"Neck Angle: {angles.get('neck', '--')}°\n"
        f"Left Body Angle: {angles.get('left_body', '--')}°\n"
        f"Right Body Angle: {angles.get('right_body', '--')}°\n"
        f"Average Body Angle: {angles.get('avg_body', '--')}°\n"
        "Generate actionable and easy-to-understand tips for correcting posture "
        "for someone who works or plays games all day."
    )
    
    payload = {
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.7,
    }
    
    try:
        # Use the Gemini flash model (update the model name if needed)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        print("Error calling Gemini API:", e)
        return "Error generating report."

########################################
# 3) Baseline Capture Endpoint
########################################

@app.route("/capture_baseline", methods=["POST"])
def capture_baseline():
    """
    Expects a JSON payload with:
      { "image": "data:image/jpeg;base64,..." }
    Processes the image to obtain keypoints and sets the baseline posture.
    Returns the computed baseline angles.
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

    # Estimate pose and get keypoints
    kp_norm, annotated_frame = pose_estimator.get_pose(frame)
    kp_abs = scale_keypoints(kp_norm, frame.shape)

    if len(kp_abs) < 15:
        return jsonify({"success": False, "error": "Not enough keypoints to set baseline."})

    # Set the baseline posture in the anomaly detector.
    anomaly_detector.set_baseline(kp_abs)
    baseline_angles = anomaly_detector._compute_angles(kp_abs)

    return jsonify({
        "success": True,
        "baseline": kp_abs,
        "angles": baseline_angles
    })

########################################
# 4) Posture Detection Endpoint
########################################

@app.route("/detect_posture", methods=["POST"])
def detect_posture():
    """
    Expects JSON:
      { "image": "data:image/jpeg;base64,..." }
    Runs detection, pose estimation, and generates a posture report using Gemini
    if the detected posture is bad.
    Returns JSON with posture status, angles, and annotated image.
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

    # Generate a report using Gemini API if posture is bad
    report = ""
    if posture_status == "bad":
        report = get_posture_report(angles, posture_status)

    return jsonify({
        "posture": posture_status,
        "annotated_image": annotated_b64_url,
        "angles": angles,
        "report": report
    })

########################################
# 5) Serve the Index HTML
########################################

@app.route("/")
def serve_index():
    """
    Serve the index.html file from the 'static' folder.
    """
    return send_from_directory(app.static_folder, "index.html")

########################################
# 6) Run the Flask App
########################################

if __name__ == "__main__":
    # By default, run on localhost:5000
    app.run(host="0.0.0.0", port=5000, debug=True)
