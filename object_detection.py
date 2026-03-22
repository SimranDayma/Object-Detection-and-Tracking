"""
CodeAlpha Internship - Task 4: Object Detection and Tracking
Author: Your Name
Description: Real-time object detection using YOLOv8 and OpenCV.
             Detects 80 object classes from webcam or video file.
             Press Q to quit.
"""

# ── IMPORTS ──────────────────────────────────────────────────────────────────
import cv2                          # OpenCV — for reading webcam/video frames
from ultralytics import YOLO        # YOLOv8 — pre-trained object detection model
import cvzone                       # Helper library for drawing nice boxes
import math                         # For rounding confidence scores


# ── OBJECT CLASS NAMES ────────────────────────────────────────────────────────
# YOLOv8 can detect these 80 object types
CLASS_NAMES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# ── COLORS FOR DIFFERENT CLASSES ──────────────────────────────────────────────
# Each object class gets a unique color for its bounding box
import random
random.seed(42)
COLORS = [(random.randint(100, 255),
           random.randint(100, 255),
           random.randint(100, 255)) for _ in CLASS_NAMES]


# ── LOAD YOLO MODEL ───────────────────────────────────────────────────────────
print("Loading YOLOv8 model...")
print("(First time will download ~6MB model — please wait)")
model = YOLO("yolov8n.pt")   # 'n' = nano (smallest + fastest)
print("Model loaded! Starting detection...")


# ── OPEN VIDEO SOURCE ─────────────────────────────────────────────────────────
# Change 0 to a video file path like "video.mp4" if no webcam
cap = cv2.VideoCapture(0)

# Set webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Check if webcam opened successfully
if not cap.isOpened():
    print("ERROR: Could not open webcam!")
    print("Try changing VideoCapture(0) to VideoCapture(1) in the code")
    print("Or use a video file: VideoCapture('your_video.mp4')")
    exit()

print("Webcam opened! Press Q to quit.")
print("-" * 40)


# ── TRACKING STATE ────────────────────────────────────────────────────────────
# Simple tracker — assigns IDs to detected objects
tracked_objects = {}
next_id         = 1


def get_tracking_id(cx, cy, label):
    """
    Assigns a consistent ID to a detected object based on its position.
    Objects that stay in roughly the same spot keep the same ID.
    """
    global next_id
    threshold = 80   # pixels — how close counts as "same object"

    for obj_id, (ox, oy, olabel) in tracked_objects.items():
        if olabel == label and abs(cx - ox) < threshold and abs(cy - oy) < threshold:
            tracked_objects[obj_id] = (cx, cy, label)
            return obj_id

    # New object — assign new ID
    tracked_objects[next_id] = (cx, cy, label)
    new_id   = next_id
    next_id += 1
    return new_id


# ── MAIN DETECTION LOOP ───────────────────────────────────────────────────────
frame_count  = 0
object_count = 0

while True:
    # Read one frame from webcam
    success, frame = cap.read()

    if not success:
        print("Could not read frame. Exiting...")
        break

    frame_count += 1
    object_count = 0

    # ── Run YOLO detection on this frame ──
    results = model(frame, stream=True, verbose=False)

    # ── Process each detection ──
    for result in results:
        boxes = result.boxes   # All detected boxes in this frame

        for box in boxes:
            # ── Get bounding box coordinates ──
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # ── Get confidence score (how sure the model is) ──
            confidence = math.ceil(box.conf[0] * 100) / 100

            # Skip low confidence detections (less than 40% sure)
            if confidence < 0.4:
                continue

            # ── Get class name ──
            class_id = int(box.cls[0])
            label    = CLASS_NAMES[class_id]
            color    = COLORS[class_id]

            # ── Get center point for tracking ──
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # ── Assign tracking ID ──
            track_id = get_tracking_id(cx, cy, label)

            object_count += 1

            # ── Draw bounding box ──
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # ── Draw label background ──
            label_text = f"{label} #{track_id} {int(confidence*100)}%"
            (text_w, text_h), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(frame,
                          (x1, y1 - text_h - 10),
                          (x1 + text_w + 8, y1),
                          color, -1)   # Filled rectangle

            # ── Draw label text ──
            cv2.putText(frame, label_text,
                        (x1 + 4, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)

            # ── Draw center dot ──
            cv2.circle(frame, (cx, cy), 4, color, -1)

    # ── Draw info panel at top of frame ──
    # Background bar
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 45), (20, 20, 20), -1)

    # Title
    cv2.putText(frame, "CodeAlpha - Object Detection & Tracking",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 255, 150), 2)

    # Object count
    cv2.putText(frame, f"Objects: {object_count}  |  Frame: {frame_count}  |  Press Q to quit",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (180, 180, 180), 1)

    # ── Show the frame ──
    cv2.imshow("CodeAlpha - Object Detection", frame)

    # ── Press Q to quit ──
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break

    # ── Clean up old tracked objects every 30 frames ──
    if frame_count % 30 == 0:
        tracked_objects.clear()


# ── CLEANUP ───────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
print(f"Done! Processed {frame_count} frames.")