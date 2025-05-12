# RTSP YOLOv8 Stream Demo

This script demonstrates how to use YOLOv8 for real-time object detection or tracking on RTSP streams, webcam feeds, or video files.

## Prerequisites

- Python 3.6+
- Required packages:
  ```
  pip install ultralytics opencv-python
  ```

## Usage

### Basic Usage

```bash
python rtsp_test.py [OPTIONS]
```

### Options

- `--source`, `-s`: Source for detection (RTSP URL, video file path, or '0' for webcam)
  - Default: `rtsp://0.0.0.0:8554/live`
  
- `--use-webcam`, `-c`: Use webcam (equivalent to `--source 0`)

- `--weights`, `-w`: Path to YOLOv8 weights file
  - Default: `yolov8n.pt`
  
- `--stride`, `-t`: Frame reading stride
  - Default: `1`
  
- `--buffer`, `-b`: Enable reading buffer (to prevent stuttering)

- `--track`, `-k`: Enable multi-object tracking (uses model.track)

### Examples

1. Use default RTSP stream:
   ```bash
   python rtsp_test.py
   ```

2. Use webcam:
   ```bash
   python rtsp_test.py --use-webcam
   ```

3. Use a video file:
   ```bash
   python rtsp_test.py --source path/to/video.mp4
   ```

4. Use a different RTSP URL:
   ```bash
   python rtsp_test.py --source rtsp://username:password@192.168.1.100:554/stream
   ```

5. Enable object tracking:
   ```bash
   python rtsp_test.py --source path/to/video.mp4 --track
   ```

## Troubleshooting

- If you get a connection error with RTSP:
  - Check if the RTSP server is running
  - Verify the RTSP URL is correct
  - Try using a webcam or video file instead

- If the webcam doesn't work:
  - Make sure your webcam is properly connected
  - Try a different webcam index (e.g., `--source 1`)

- If you get "model not found" errors:
  - Make sure the YOLOv8 weights file exists in the specified path
  - Download YOLOv8 weights if needed: `pip install ultralytics` will automatically download the default weights

## Controls

- Press 'q' to quit the detection window
