# 姿势检测系统调试指南

本文档提供了几种不同的方法来测试和调试姿势检测系统的主要功能，无需使用前端界面。

## 1. RTSP 姿势检测器 (推荐)

`rtsp_detector.py` 是一个独立的 RTSP 姿势检测程序，它提供了简洁的命令行界面来测试 RTSP 流处理、姿势检测和 TTS 功能。

### 使用方法

```bash
python rtsp_detector.py [参数]
```

### 参数

- `--rtsp_url RTSP_URL`: RTSP 流地址，默认为 "rtsp://192.168.3.242:8554/live"
- `--use_tts`: 启用 TTS 功能，检测到不良姿势时会播报提示
- `--set_baseline`: 启动后设置当前姿势为基准姿势
- `--run_time SECONDS`: 运行指定秒数后退出，0 表示一直运行直到按 Ctrl+C

### 示例

```bash
# 基本用法，连接到默认RTSP URL
python rtsp_detector.py

# 指定RTSP URL，启用TTS功能
python rtsp_detector.py --rtsp_url rtsp://192.168.3.242:8554/live --use_tts

# 运行30秒，同时设置基准姿势
python rtsp_detector.py --run_time 30 --set_baseline
```

## 2. 调试后端工具

`debug_backend.py` 提供了更接近实际应用的调试体验，可以测试主应用（main.py）中的 RTSP 和 TTS 功能。

### 使用方法

```bash
python debug_backend.py [参数]
```

### 参数

- `--rtsp_url RTSP_URL`: RTSP 流地址
- `--mode {rtsp,tts_test}`: 调试模式，rtsp 用于测试流处理，tts_test 用于测试 TTS 功能
- `--frames N`: 要处理的帧数（0 表示无限制）
- `--report_interval N`: 每 N 帧生成一次姿势报告

### 示例

```bash
# 处理RTSP流并每10帧检查一次是否需要生成报告
python debug_backend.py --mode rtsp --report_interval 10

# 仅测试TTS功能
python debug_backend.py --mode tts_test
```

## 3. TTS 测试工具

`tts_test.py` 是一个专门用于测试 IP 音箱 TTS 功能的简单工具。

### 使用方法

```bash
python tts_test.py [参数]
```

### 参数

- `--text TEXT`: 要播报的文本
- `--host HOST`: IP 音箱的 IP 地址，默认为 "192.168.3.29"
- `--port PORT`: IP 音箱的端口，默认为 80

### 示例

```bash
# 使用默认文本和IP地址
python tts_test.py

# 指定文本和IP地址
python tts_test.py --text "测试语音播报功能" --host 192.168.3.29
```

## 4. 直接运行 main.py

如果需要运行完整的应用程序，但希望通过日志输出进行调试：

```bash
python main.py > app_output.log 2>&1
```

然后可以在另一个终端中查看日志文件：

```bash
tail -f app_output.log
```

## 常见问题

1. **RTSP 连接失败**：确保摄像头已开启，IP 地址正确，并且可以访问。你可以使用 VLC 等工具测试 RTSP 流是否可访问。

2. **TTS 功能不工作**：检查 IP 音箱的 IP 地址是否正确，设备是否开启，并且在同一网络中。

3. **姿势检测不准确**：尝试调整照明条件，确保人物在摄像头视野中清晰可见。
