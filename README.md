# 🧠 Pose Pilot: Intelligent Posture Risk Detection for Gamers 🎮

Welcome to **Pose Pilot**, your AI-powered real-time posture assistant designed for gamers and high-performance users. This project was developed as part of **Bitcamp 2025**, combining computer vision, pose estimation, and conversational AI to help users avoid poor posture and related health risks.

## 🚀 Features

- ⚙️ **YOLOv8 + YOLO-Pose** for fast, real-time human detection and keypoint extraction
- 🧍 **Posture Anomaly Detection** using custom angle-based metrics and AI feedback
- 🎙️ **Dynamic Voice Feedback** with varied human-like TTS responses (via `pyttsx3`)
- 🧠 **Gemini AI Integration**: Understand posture and respond to questions visually and contextually
- 💬 **Live Chat UI** for posture Q&A, tips, and Gemini-powered safety suggestions
- 📊 **Confidence Control** with real-time slider tuning

## 📦 Installation

1. **Clone the Repository**

```bash
git clone https://github.com/cravotics/Pose-pilot.git
cd Pose-pilot
```

2. **Set Up Virtual Environment** (Recommended)

```bash
python -m venv scene_env
source scene_env/bin/activate  # On Windows: scene_env\Scripts\activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

> ⚠️ Note: You may need to install system dependencies for `pyttsx3` and `mediapipe`.

4. **Download YOLO Weights** (Optional)

By default, the model downloads `yolov8n.pt` on first run.

5. **Add Your Gemini API Key**

Replace `"YOUR_REAL_GEMINI_2_KEY"` in `main.py` with your actual Gemini API key.

## 🧪 Run the App

```bash
python main.py
```

> The camera index is set to `1` in `main.py`. You can change it to `0` if your default webcam is on index 0.

## 📂 Project Structure

```bash
Pose-pilot/
├── main.py                  # Main PyQt5 GUI with TTS, Gemini & pose detection
├── yolo_detector.py         # YOLOv8 object detection module
├── pose_estimator.py        # Pose keypoint detector (e.g., using YOLO-Pose)
├── anomaly_detector.py      # Custom rule-based posture risk analyzer
├── requirements.txt         # Required Python packages
└── README.md                # This file
```

## 🧠 How It Works

Pose Pilot continuously captures webcam video and detects human posture using keypoints like shoulders, hips, and knees. It checks for dangerous lean angles using:

- 👎 Rule-based angle filters
- 🧠 Gemini Vision: Uploads annotated frame to Gemini for AI interpretation

If a risk is detected:
- You'll **hear a dynamic spoken alert**
- See it displayed in the **PyQt5 GUI**
- And can chat with **Gemini AI** for advice

## 🎯 Use Case: Gamers & Streamers

Perfect for streamers, e-sports athletes, or remote workers who forget to monitor their posture while focused on gameplay.

## 🛠️ Future Improvements

- Use LLM-generated dynamic suggestions for pose correction
- Add exercise/stretch suggestions based on long-term posture data
- Export posture timeline reports in CSV/JSON
- Add Twitch overlay alerts

## 🧾 License

MIT License. Feel free to use, modify, and build upon Pose Pilot.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- Google Gemini API
- Bitcamp 2025 Hackathon @ University of Maryland

---

> Made with passion by [Sai Jagadeesh Muralikrishnan](https://github.com/cravotics) 👨‍💻

Stay upright. Stay sharp. Stay in the game. 🎮🔥
