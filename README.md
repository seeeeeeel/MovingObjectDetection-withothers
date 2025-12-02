# Moving Object Detection

Real‑time object detection & recording using YOLOv8 + OpenCV, with a clean GUI.  

## 🚀 What is this

This project detects moving objects (people, cars, etc.) from a webcam or video file — draws bounding boxes + confidence scores, auto‑records video **only when a person is present**, and lets you manually capture snapshots.  
It’s ideal for projects like surveillance, monitoring, or just learning computer vision + GUI programming.

## ✅ Features

- Real‑time detection using YOLOv8 + OpenCV.  
- Support for **Webcam** or **Video file** as input.  
- Auto‑recording: video saved automatically when a person is detected; stops recording when no person.  
- Snapshot capture with a key press (`C`).  
- REC indicator badge that blinks during active recording.  
- Graphical user interface (GUI) for easy control: choose input, start detection, stop, capture.  
- Dark-themed UI — easier on the eyes, no distracting white backgrounds.  
- Object‑specific bounding box colors (e.g. person = red, vehicles = blue/other) for clarity.  
- Organized output:  
  - `recordings/` folder — saved video clips  
  - `captures/` folder — saved snapshots  

## 🛠 Requirements & Setup

- Python 3.10 or newer  
- Install dependencies:

  ```bash
  pip install opencv-python pillow ultralytics customtkinter
