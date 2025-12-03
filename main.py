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
    "accent": "#FF4D4D",  # person/REC
    "accent2": "#4DA6FF", # car/object
    "text": "#E1E1E1",
    "muted": "#555555"
}

# ---------------------------
# Globals
# ---------------------------
app = None
multi_win = None
full_win = None
video_win = None
labels = []
caps = []
current_frames = [None, None, None, None]
running = False
recording = False
out = None
full_index = None
CAM_SOURCES = [0,1,2,3]
rec_badge = None

# ---------------------------
# Helpers
# ---------------------------
def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def draw_yolo_on(frame):
    try:
        results = model(frame, verbose=False)[0]
    except:
        return frame
    if hasattr(results, "boxes") and results.boxes is not None:
        for box in results.boxes:
            try:
                xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], "cpu") else box.xyxy[0]
            except:
                xyxy = box.xyxy if hasattr(box, "xyxy") else None
            if xyxy is None: continue
            x1, y1, x2, y2 = map(int, xyxy)
            try: conf = float(box.conf[0]) if hasattr(box, "conf") else float(box.conf)
            except: conf = 0.0
            try: cls_idx = int(box.cls[0]) if hasattr(box, "cls") else int(box.cls)
            except: cls_idx = 0
            cls_name = model.names[cls_idx] if cls_idx in model.names else str(cls_idx)

            if cls_name.lower() == "person": color = (0,0,255)
            elif cls_name.lower() in ["car","truck","bus"]: color = (255,128,0)
            else: color = (85,85,85)

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,f"{cls_name} {conf:.2f}",(x1,y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
    return frame

# ---------------------------
# Recording
# ---------------------------
def start_recording(frame):
    global recording, out
    if recording: return
    h,w = frame.shape[:2]
    fname = f"recordings/record_{timestamp()}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(fname, fourcc, 20.0, (w,h))
    recording=True
    try: rec_badge.configure(text="● REC", fg_color=PALETTE["accent"])
    except: pass
    print("[REC START]", fname)

def stop_recording():
    global recording, out
    if out: out.release(); out=None
    recording=False
    try: rec_badge.configure(text="", fg_color=PALETTE["panel"])
    except: pass
    print("[REC STOP]")

# ---------------------------
# Capture frame
# ---------------------------
def capture_current_frame(frame):
    if frame is None:
        print("[CAPTURE] No frame available")
        return
    fname = f"captures/capture_{timestamp()}.png"
    cv2.imwrite(fname, frame)
    print("[CAPTURE]", fname)

# ---------------------------
# Camera init/release
# ---------------------------
def init_caps():
    global caps
    caps=[]
    for src in CAM_SOURCES:
        try:
            c = cv2.VideoCapture(src)
            ok,_ = c.read()
            caps.append(c)
        except: caps.append(None)

def release_caps():
    global caps
    for c in caps:
        try:
            if c and c.isOpened(): c.release()
        except: pass
    caps=[]

# ---------------------------
# Webcam 4-grid update
# ---------------------------
def update_all_cams():
    global current_frames, running
    if not running or multi_win is None or not multi_win.winfo_exists():
        return
    for i, c in enumerate(caps):
        if c is None:
            ph = np.zeros((240,320,3), np.uint8)
            cv2.putText(ph,f"CAM {i} (no feed)",(10,120),cv2.FONT_HERSHEY_SIMPLEX,0.7,(200,200,200),2)
            frame = ph
        else:
            ret, frame = c.read()
            if not ret or frame is None:
                ph = np.zeros((240,320,3), np.uint8)
                cv2.putText(ph,f"CAM {i} (no frame)",(10,120),cv2.FONT_HERSHEY_SIMPLEX,0.7,(200,200,200),2)
                frame = ph
        current_frames[i]=frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
        labels[i].imgtk=imgtk
        labels[i].configure(image=imgtk)

        # Recording for 4-grid (only fullscreen)
        if recording and full_index==i:
            out.write(frame)

    multi_win.after(30, update_all_cams)

# ---------------------------
# Fullscreen webcam with YOLO
# ---------------------------
def open_fullscreen(cam_index):
    global full_win, full_index, multi_win, rec_badge
    if full_win and full_win.winfo_exists(): return
    full_index=cam_index

    # Hide 4-grid window
    if multi_win and multi_win.winfo_exists():
        multi_win.withdraw()

    full_win=ctk.CTkToplevel()
    full_win.title(f"Camera {cam_index} Fullscreen")
    full_win.geometry("1000x700")
    full_win.configure(fg_color=PALETTE["bg"])

    top=ctk.CTkFrame(full_win, fg_color=PALETTE["panel"], corner_radius=8)
    top.pack(fill="x", padx=8, pady=8)
    lbl_title=ctk.CTkLabel(top,text=f"Camera {cam_index}",font=("Arial",16,"bold"),text_color=PALETTE["text"])
    lbl_title.pack(side="right", padx=6)
    btn_back=ctk.CTkButton(top,text="BACK",fg_color=PALETTE["accent2"],command=close_fullscreen)
    btn_back.pack(side="left", padx=6)
    btn_capture=ctk.CTkButton(top,text="Capture (C)",fg_color=PALETTE["accent"],command=lambda: capture_current_frame(current_frames[cam_index]))
    btn_capture.pack(side="left", padx=6)
    btn_rec=ctk.CTkButton(top,text="REC",fg_color=PALETTE["accent"],command=lambda: start_recording(current_frames[cam_index]))
    btn_rec.pack(side="left", padx=6)

    rec_badge = ctk.CTkLabel(top, text="", fg_color=PALETTE["panel"], width=50, corner_radius=8)
    rec_badge.pack(side="left", padx=6)

    lbl=ctk.CTkLabel(full_win,text="",fg_color=PALETTE["panel"],corner_radius=8)
    lbl.pack(expand=True,fill="both", padx=10, pady=(0,10))

    def loop_full():
        if not full_win or not full_win.winfo_exists(): return
        c = caps[cam_index]
        if c is None:
            ph=np.zeros((480,854,3),np.uint8)
            cv2.putText(ph,"No feed",(20,240),cv2.FONT_HERSHEY_SIMPLEX,1.0,(200,200,200),2)
            frame=ph
        else:
            ret, frame=c.read()
            if not ret or frame is None:
                frame=current_frames[cam_index] if current_frames[cam_index] is not None else np.zeros((480,854,3),np.uint8)
            # YOLO detection only in fullscreen
            frame=draw_yolo_on(frame)
            current_frames[cam_index]=frame

        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        imgtk=ImageTk.PhotoImage(Image.fromarray(rgb))
        lbl.imgtk=imgtk
        lbl.configure(image=imgtk)

        if recording:
            out.write(frame)

        full_win.after(30, loop_full)

    loop_full()

def close_fullscreen():
    global full_win, full_index
    stop_recording()
    try: full_win.destroy()
    except: pass
    full_win=None
    full_index=None

    # Show 4-grid again
    if multi_win and multi_win.winfo_exists():
        multi_win.deiconify()

# ---------------------------
# 4-grid webcam window
# ---------------------------
def open_multicam_window():
    global multi_win, labels, running
    if multi_win and multi_win.winfo_exists(): return
    running=True
    init_caps()
    labels=[]
    multi_win=ctk.CTkToplevel()
    multi_win.title("4-Camera Grid")
    multi_win.geometry("1200x700")
    multi_win.configure(fg_color=PALETTE["bg"])

    top_panel=ctk.CTkFrame(multi_win, fg_color=PALETTE["panel"], corner_radius=8)
    top_panel.pack(fill="x", padx=8, pady=8)
    lbl_title=ctk.CTkLabel(top_panel,text="Moving Object Detection",font=("Arial",16,"bold"),text_color=PALETTE["text"])
    lbl_title.pack(side="right", padx=6)
    btn_capture=ctk.CTkButton(top_panel,text="Capture (C)",width=120,height=30,fg_color=PALETTE["accent"],command=lambda: capture_current_frame(None))
    btn_capture.pack(side="left", padx=10)
    btn_stop=ctk.CTkButton(top_panel,text="Stop (S)",width=120,height=30,fg_color=PALETTE["accent2"],command=stop_and_close_detection)
    btn_stop.pack(side="left", padx=10)

    grid_frame=ctk.CTkFrame(multi_win, fg_color=PALETTE["bg"])
    grid_frame.pack(expand=True, fill="both", padx=10, pady=(0,10))

    for i in range(4):
        lbl=ctk.CTkLabel(grid_frame,text="",fg_color=PALETTE["panel"],corner_radius=8)
        r=i//2; ccol=i%2
        lbl.grid(row=r,column=ccol,padx=8,pady=8,sticky="nsew")
        lbl.bind("<Button-1>",lambda e, idx=i: open_fullscreen(idx))
        labels.append(lbl)

    for col in range(2): grid_frame.grid_columnconfigure(col,weight=1)
    for row in range(2): grid_frame.grid_rowconfigure(row,weight=1)

    multi_win.bind("<Key>", lambda e: key_handler(e))
    multi_win.after(50, update_all_cams)

# ---------------------------
# Open video
# ---------------------------
def open_video_file():
    global video_win, recording, out
    filep=filedialog.askopenfilename(title="Select video",filetypes=[("Video files","*.mp4;*.avi;*.mov")])
    if not filep: return
    cap=cv2.VideoCapture(filep)
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_win=ctk.CTkToplevel()
    video_win.title("Video Detection")
    video_win.geometry(f"{w}x{h}")
    video_win.configure(fg_color=PALETTE["bg"])

    top_panel=ctk.CTkFrame(video_win, fg_color=PALETTE["panel"], corner_radius=8)
    top_panel.pack(fill="x", padx=8, pady=8)
    btn_capture=ctk.CTkButton(top_panel,text="Capture (C)",fg_color=PALETTE["accent"],command=lambda: capture_current_frame(current_frames[0]))
    btn_capture.pack(side="left", padx=6)
    btn_rec=ctk.CTkButton(top_panel,text="REC",fg_color=PALETTE["accent"],command=lambda: start_recording(current_frames[0]))
    btn_rec.pack(side="left", padx=6)

    global rec_badge
    rec_badge = ctk.CTkLabel(top_panel, text="", fg_color=PALETTE["panel"], width=50, corner_radius=8)
    rec_badge.pack(side="left", padx=6)

    lbl=ctk.CTkLabel(video_win,text="",fg_color=PALETTE["panel"],corner_radius=8)
    lbl.pack(expand=True,fill="both", padx=10, pady=10)

    def loop_vid():
        if not video_win or not video_win.winfo_exists():
            stop_recording()
            cap.release()
            return
        ret, frame = cap.read()
        if not ret:
            stop_recording()
            cap.release()
            return
        # Always detect on video
        frame=draw_yolo_on(frame)
        current_frames[0]=frame

        if recording:
            out.write(frame)

        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        imgtk=ImageTk.PhotoImage(Image.fromarray(rgb))
        lbl.imgtk=imgtk
        lbl.configure(image=imgtk)
        video_win.after(30, loop_vid)

    loop_vid()
    video_win.bind("<Key>", lambda e: key_handler(e))

# ---------------------------
# Stop / cleanup
# ---------------------------
def stop_and_close_detection(event=None):
    global running, caps, multi_win, full_win, video_win
    stop_recording()
    running=False
    release_caps()
    for win in [multi_win, full_win, video_win]:
        try:
            if win and win.winfo_exists(): win.destroy()
        except: pass
    multi_win=None
    full_win=None
    video_win=None
    app.deiconify()

# ---------------------------
# Keyboard
# ---------------------------
def key_handler(event):
    key = event.keysym.lower()
    if key=='c':
        if full_index is not None:
            capture_current_frame(current_frames[full_index])
        elif video_win and video_win.winfo_exists():
            capture_current_frame(current_frames[0])
    elif key=='s':
        if full_win and full_win.winfo_exists():
            close_fullscreen()
        elif multi_win and multi_win.winfo_exists():
            stop_and_close_detection()
        elif video_win and video_win.winfo_exists():
            stop_and_close_detection()
    elif key=='q' and app and app.winfo_exists():
        app.destroy()

# ---------------------------
# Launcher
# ---------------------------
def build_launcher():
    global app
    app=ctk.CTk()
    app.title("MOD Launcher")
    app.geometry("320x220")
    app.resizable(False,False)
    app.configure(fg_color=PALETTE["bg"])

    frame=ctk.CTkFrame(app,fg_color=PALETTE["panel"],corner_radius=12,width=330,height=220)
    frame.place(relx=0.5,rely=0.5,anchor="center")

    lbl=ctk.CTkLabel(frame,text="Moving Object Detection",font=("Arial",16,"bold"),text_color=PALETTE["text"])
    lbl.pack(pady=(10,6))

    btn_video=ctk.CTkButton(frame,text="Open Video",width=220,height=36,fg_color=PALETTE["accent2"],command=open_video_file)
    btn_video.pack(pady=(4,6),anchor="w",padx=12)
    btn_webcam=ctk.CTkButton(frame,text="Open Webcam",width=220,height=36,fg_color=PALETTE["accent"],command=open_multicam_window)
    btn_webcam.pack(pady=(4,6),anchor="w",padx=12)

    app.bind("<q>",lambda e: app.destroy())
    app.mainloop()

# ---------------------------
# Run
# ---------------------------
if __name__=="__main__":
    build_launcher()
