import os
import cv2
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
from datetime import datetime
from ultralytics import YOLO
import numpy as np

# ---------------------------
# Setup folders
# ---------------------------
os.makedirs("recordings", exist_ok=True)
os.makedirs("captures", exist_ok=True)

# ---------------------------
# Load YOLO
# ---------------------------
WEIGHTS = "yolov8n.pt"
if not os.path.exists(WEIGHTS):
    print("Downloading YOLO weights...")
    YOLO(WEIGHTS)
model = YOLO(WEIGHTS)

# ---------------------------
# Theme and palette
# ---------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

PALETTE = {
    "bg": "#1C1E26",
    "panel": "#252733",
    "accent": "#FF4D4D",  # person
    "accent2": "#4DA6FF", # car/object
    "text": "#E1E1E1",
    "muted": "#555555"
}

# ---------------------------
# Globals
# ---------------------------
app = None
detection_win = None
video_label = None
rec_badge = None
cap = None
running = False
recording = False
out = None
current_frame = None
blink_state = False

# ---------------------------
# Helpers
# ---------------------------
def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# ---------------------------
# Detection loop
# ---------------------------
def update_frame_loop():
    global cap, running, recording, out, current_frame

    if not running or cap is None:
        return

    ret, frame = cap.read()
    if not ret:
        stop_and_close_detection()
        return

    current_frame = frame.copy()

    # YOLO detection
    results = model(frame, verbose=False)[0]
    person_detected = False
    if hasattr(results, "boxes") and results.boxes is not None:
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], "cpu") else box.xyxy[0]
            x1, y1, x2, y2 = map(int, xyxy)
            conf = float(box.conf[0]) if hasattr(box, "conf") else float(box.conf)
            cls_idx = int(box.cls[0]) if hasattr(box, "cls") else int(box.cls)
            cls_name = model.names[cls_idx] if cls_idx in model.names else str(cls_idx)

            # Dark color scheme
            if cls_name.lower() == "person":
                color = (255,0,0)  # red
                person_detected = True
            elif cls_name.lower() in ["car","truck","bus"]:
                color = (0,128,255) # blue
            else:
                color = (85,85,85)  # dark gray

            cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Auto-record only when person detected
    if person_detected and not recording:
        start_recording(frame)
    if not person_detected and recording:
        stop_recording()
    if recording and out is not None:
        out.write(frame)

    # Convert to Tkinter image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    if running:
        video_label.after(10, update_frame_loop)

# ---------------------------
# Recording
# ---------------------------
def start_recording(frame):
    global recording, out
    if recording:
        return
    h,w = frame.shape[:2]
    fname = f"recordings/record_{timestamp()}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(fname, fourcc, 20.0, (w,h))
    recording=True
    rec_badge.configure(text="● REC", fg_color=PALETTE["accent"])
    print("[REC START]", fname)

def stop_recording():
    global recording, out
    if out:
        out.release()
        out=None
    recording=False
    rec_badge.configure(text="", fg_color=PALETTE["panel"])
    print("[REC STOP]")

# ---------------------------
# Capture / Stop
# ---------------------------
def capture_current_frame(event=None):
    global current_frame
    if current_frame is None:
        return
    fname = f"captures/capture_{timestamp()}.png"
    cv2.imwrite(fname, current_frame)
    print("[CAPTURE]", fname)

def stop_and_close_detection(event=None):
    global running, cap, detection_win
    running=False
    if recording:
        stop_recording()
    if cap:
        try: cap.release()
        except: pass
    if detection_win and detection_win.winfo_exists():
        detection_win.destroy()
    app.deiconify()

# ---------------------------
# Blinking REC badge
# ---------------------------
def blink_recursively():
    global blink_state
    if detection_win is None or not detection_win.winfo_exists():
        return
    blink_state = not blink_state
    if recording:
        color = PALETTE["accent"] if blink_state else "#FF6B6B"
        rec_badge.configure(text="● REC", fg_color=color, text_color="#0B0B0B")
    else:
        rec_badge.configure(text="", fg_color=PALETTE["panel"])
    detection_win.after(500, blink_recursively)

# ---------------------------
# Detection window
# ---------------------------
def open_detection_window(source="webcam"):
    global detection_win, video_label, cap, running, rec_badge

    if detection_win and detection_win.winfo_exists():
        return

    app.withdraw()
    detection_win = ctk.CTkToplevel()
    detection_win.title("Moving Object Detection")
    detection_win.geometry("1024x576")
    detection_win.configure(fg_color=PALETTE["bg"])
    detection_win.resizable(True, True)

    # top panel
    top_panel = ctk.CTkFrame(detection_win, fg_color=PALETTE["panel"], corner_radius=12)
    top_panel.pack(padx=10, pady=10, fill="x")

    rec_badge = ctk.CTkButton(top_panel, text="", width=80, height=30,
                              fg_color=PALETTE["panel"], hover=False, corner_radius=10)
    rec_badge.pack(side="left", padx=10, pady=8)

    lbl_title = ctk.CTkLabel(top_panel, text="Moving Object Detection", font=("Arial", 18, "bold"), text_color=PALETTE["text"])
    lbl_title.pack(side="right", padx=10)

    # controls top-left
    btn_capture = ctk.CTkButton(top_panel, text="Capture (C)", width=120, height=30,
                                fg_color=PALETTE["accent"], command=capture_current_frame)
    btn_capture.pack(side="left", padx=10)
    btn_stop = ctk.CTkButton(top_panel, text="Stop (S)", width=120, height=30,
                             fg_color=PALETTE["accent2"], command=stop_and_close_detection)
    btn_stop.pack(side="left", padx=10)

    # video frame
    video_label = ctk.CTkLabel(detection_win, text="", fg_color=PALETTE["panel"], corner_radius=8)
    video_label.pack(expand=True, fill="both", padx=10, pady=(0,10))

    detection_win.bind("<Key>", key_handler)

    # placeholder image
    ph = np.zeros((480,854,3), dtype=np.uint8)
    ph_img = ImageTk.PhotoImage(Image.fromarray(ph))
    video_label.imgtk = ph_img
    video_label.configure(image=ph_img)

    # video source
    if source=="webcam":
        cap = cv2.VideoCapture(1)
    else:
        filep = filedialog.askopenfilename(title="Select video", filetypes=[("Video files","*.mp4;*.avi;*.mov")])
        if filep:
            cap = cv2.VideoCapture(filep)
        else:
            detection_win.destroy()
            app.deiconify()
            return

    global running
    running = True
    detection_win.after(50, update_frame_loop)
    detection_win.after(500, blink_recursively)

# ---------------------------
# Keyboard handler
# ---------------------------
def key_handler(event):
    key = event.keysym.lower()
    if key=='c':
        capture_current_frame()
    elif key=='s':
        stop_and_close_detection()
    elif key=='q' and app and app.winfo_exists():
        app.destroy()

# ---------------------------
# Launcher
# ---------------------------
def build_launcher():
    global app
    app = ctk.CTk()
    app.title("Launcher")
    app.geometry("220x200")
    app.resizable(False, False)
    app.configure(fg_color=PALETTE["bg"])

    # Ekis close → hide
    def on_close():
        app.withdraw()
    app.protocol("WM_DELETE_WINDOW", on_close)

    frame = ctk.CTkFrame(app, fg_color=PALETTE["panel"], corner_radius=12, width=230, height=180)
    frame.place(relx=0.5, rely=0.5, anchor="center")

    lbl = ctk.CTkLabel(frame, text="MOD Launcher", font=("Arial", 16, "bold"), text_color=PALETTE["text"])
    lbl.pack(pady=(10,6))

    # buttons stacked vertically, aligned left
    btn_cam = ctk.CTkButton(frame, text="Open Webcam", width=160, height=36,
                            fg_color=PALETTE["accent"], command=lambda: open_detection_window("webcam"))
    btn_cam.pack(pady=(6,4), anchor="w", padx=12)

    btn_vid = ctk.CTkButton(frame, text="Open Video", width=160, height=36,
                            fg_color=PALETTE["accent2"], command=lambda: open_detection_window("video"))
    btn_vid.pack(pady=(4,12), anchor="w", padx=12)

    app.bind("<q>", lambda e: app.destroy())
    app.mainloop()

# ---------------------------
# Run
# ---------------------------
if __name__=="__main__":
    build_launcher()
