# 🎯 Object Detection and Tracking — CodeAlpha 

A real-time Object Detection and Tracking system built with Python as part of the **CodeAlpha AI Internship**. Uses YOLOv8 and OpenCV to detect and track 80 object classes live from a webcam.


## ✨ Features

- 🎥 **Real-time detection** from webcam or video file
- 🏷️ Detects **80 object classes** — person, car, phone, laptop, bottle and more
- 🔢 **Object tracking** with unique IDs per object
- 📊 Shows **confidence score** for every detection
- 🎨 **Unique color** per object class
- 📺 Info panel showing object count and frame number
- ⚡ Runs at ~30 FPS using YOLOv8 Nano model
- ⌨️ Press **Q** to quit

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core programming language |
| YOLOv8 (Ultralytics) | Pre-trained object detection model |
| OpenCV | Video capture and frame processing |
| cvzone | Drawing utilities for bounding boxes |

---

## 📦 Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/CodeAlpha_ObjectDetection.git
cd CodeAlpha_ObjectDetection
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
python task4_object_detection.py
```

> The YOLOv8 model (~6MB) will be downloaded automatically on first run.

---

## 🚀 How It Works

1. OpenCV captures live frames from the webcam
2. Each frame is passed to the **YOLOv8 Nano model**
3. YOLO detects all objects and returns bounding box coordinates
4. Detections with confidence below **40%** are filtered out
5. A simple **position-based tracker** assigns consistent IDs to objects
6. Bounding boxes, labels, IDs and confidence scores are drawn on the frame
7. The processed frame is displayed in real time

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| Q | Quit the application |

---

## 🔧 Configuration

To use a **video file** instead of webcam, change line 44 in the code:
```python
# Webcam
cap = cv2.VideoCapture(0)

# Video file
cap = cv2.VideoCapture("your_video.mp4")
```

To adjust **confidence threshold** (default 40%):
```python
if confidence < 0.4:   # Change 0.4 to any value between 0 and 1
```

---

## 📁 Project Structure

```
CodeAlpha_ObjectDetection/
├── task4_object_detection.py    # Main detection script
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── screenshot.png               # App screenshot
```

---
 Author

SIMRAN DAYMA
AI Intern at CodeAlpha
[LinkedIn](https://www.linkedin.com/in/simran-dayma-54151131a?utm_source=share_via&utm_content=profile&utm_medium=member_android) 
---

## 🏢 About CodeAlpha

CodeAlpha is a leading software development company providing internship opportunities in AI, web development, and more.
🌐 [www.codealpha.tech](https://www.codealpha.tech)
