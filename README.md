# Pose-pilot

一个基于计算机视觉的智能姿态监测系统，用于检测和纠正不良坐姿，提供实时反馈和语音提醒。

## 项目概述

Pose-pilot 是一个智能姿态监测系统，利用 YOLOv8 进行人体检测和姿态估计，结合自定义的异常检测算法，实时监测用户的坐姿状态。当检测到不良姿势时，系统会通过 IP 音箱发出语音提醒，并使用百度文心一言 API 生成个性化的姿势改进建议。

主要功能：
- 实时姿态监测和分析
- 基于 RTSP 视频流的连续监控
- 自定义姿势基准线设置
- 不良姿势检测和警报
- AI 生成的个性化姿势改进建议
- IP 音箱语音提醒

## 环境配置

### 系统要求
- Python 3.9 或更高版本
- Windows/Linux/MacOS 操作系统
- 摄像头或 RTSP 视频流源
- （可选）IP 音箱设备

### 安装依赖
1. 克隆仓库
```bash
git clone https://github.com/yourusername/Pose-pilot.git
cd Pose-pilot
```

2. 安装依赖包
```bash
pip install -r requirements.txt
```

### 依赖包列表
创建 `requirements.txt` 文件，包含以下依赖：
```
ultralytics>=8.0.0
opencv-python>=4.7.0
numpy>=1.24.0
flask>=2.2.0
flask-cors>=3.0.0
matplotlib>=3.7.0
requests>=2.28.0
torch>=2.0.0
```

### 模型文件
项目使用 YOLOv8 姿态估计模型。首次运行时会自动下载模型，或者您可以手动下载并放置在项目根目录：
- `yolov8n-pose.pt`：YOLOv8 姿态估计模型（轻量版）

## 配置说明

### RTSP 流配置
在 `main.py` 中，默认的 RTSP 流地址为：
```python
rtsp_url = "rtsp://192.168.3.242:8554/live"
```
您可以根据实际环境修改此地址。

### IP 音箱配置
在 `speaker/ip_speaker.py` 中，默认的 IP 音箱地址为：
```python
host = "192.168.3.29"
port = 80
```
请根据您的 IP 音箱设备配置修改这些参数。

### 百度文心一言 API 配置
如需使用 AI 生成的姿势建议功能，请在运行时提供百度文心一言 API 的密钥：
```python
# 在 main.py 中设置
api_key = "您的百度API密钥"
secret_key = "您的百度密钥"
```

## 使用方法

### 启动服务
```bash
python main.py
```
服务启动后，访问 `http://localhost:5000` 可打开 Web 界面。

### Web 界面功能
- `/advice.html`：显示姿势建议和状态
- `/rtsp/start`：启动 RTSP 流监测
- `/rtsp/stop`：停止 RTSP 流监测
- `/rtsp/status`：查看 RTSP 监测状态
- `/rtsp/latest`：获取最新的监测结果
- `/rtsp/report`：获取姿势报告并通过 IP 音箱播报
- `/rtsp/pose_data`：获取格式化的姿态数据
- `/baseline/capture`：捕获当前姿势作为基准线

### 姿势基准线设置
1. 保持正确的坐姿
2. 访问 `/baseline/capture` 端点捕获当前姿势作为基准线
3. 系统将根据此基准线检测姿势偏差

## 项目结构

- `main.py`：主程序和 Flask 服务器
- `yolo_detector.py`：YOLOv8 人体检测模块
- `pose_estimator.py`：姿态估计模块
- `anomaly_detector.py`：姿势异常检测模块
- `speaker/ip_speaker.py`：IP 音箱通信模块
- `static/`：Web 界面静态文件
  - `advice.html`：姿势建议页面
  - `rtsp-manager.js`：RTSP 流管理 JavaScript
  - `styles.css`：样式表

## 技术原理

### 姿态检测
使用 YOLOv8 姿态估计模型检测 17 个 COCO 关键点：
- 0=鼻子, 1=左眼, 2=右眼, 3=左耳, 4=右耳
- 5=左肩, 6=右肩, 7=左肘, 8=右肘
- 9=左手腕, 10=右手腕, 11=左髋, 12=右髋
- 13=左膝, 14=右膝, 15=左踝, 16=右踝

### 姿势分析
系统计算关键点之间的角度来分析姿势：
- 左肩-髋-膝角度
- 右肩-髋-膝角度
- 颈部角度（鼻子-左肩-右肩）

### 异常检测模式
1. **常规角度检查**：检查角度是否在预设阈值范围内
2. **基准偏差检测**：与用户定义的基准姿势比较角度差异

## 常见问题

1. **模型下载失败**
   - 检查网络连接
   - 手动下载模型文件并放置在项目根目录

2. **RTSP 流连接问题**
   - 确认 RTSP 地址正确
   - 检查网络连接和防火墙设置

3. **IP 音箱连接失败**
   - 确认 IP 音箱地址和端口正确
   - 检查网络连接和防火墙设置

4. **百度 API 连接问题**
   - 确认 API 密钥正确
   - 检查网络连接和代理设置

## 许可证

[MIT License](LICENSE)

## 联系方式

如有问题或建议，请提交 Issue 或联系项目维护者。
