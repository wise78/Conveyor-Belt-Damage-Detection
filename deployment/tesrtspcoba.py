#!/usr/bin/env python3
# YOLO + EdgeTPU (TFLite) — RTSP / Video / Image / Webcam
# - Konfigurasi di blok KONFIG (tidak pakai CLI)
# - RTSP threaded (drop frame lama → low-latency)
# - Diagnostik (backend, resolusi, FPS)
# - Simpan output (gambar/video) opsional
# - ROI (Region of Interest) untuk fokus deteksi conveyor belt

import os
import cv2
import time
import queue
import threading
import numpy as np
from collections import deque
from pathlib import Path
from ultralytics import YOLO

# =======================
# KONFIG
# =======================
# Pilih salah satu: "rtsp", "video", "image", "webcam"
SOURCE_TYPE = "webcam"

# RTSP
RTSP_URL = "rtsp://username:password@IP:PORT/Streaming/Channels/101"
USE_GSTREAMER = False    # True di Linux/RPi; Windows/Laptop → False (FFMPEG)

# Video / Gambar
VIDEO_PATH = r"C:\path\to\input.mp4"
IMAGE_PATH = r"/home/admin/yolo/contohsobek1.jpg"

# Webcam
WEBCAM_INDEX = 0        # 0=default cam; ganti 1/2 sesuai device

# Model & Inference
MODEL_PATH = "/home/admin/yolo/256best_detect_databox_balanced_full_integer_quant_edgetpu.tflite"  # *_edgetpu.tflite (INT8)
DEVICE = "tpu:0"       # EdgeTPU = "tpu:0", debug CPU = "cpu"
IMGSZ = 256            # sesuaikan dgn model (256/320/480 umum)
CONF = 0.25            # confidence threshold

# ROI Configuration
ENABLE_ROI = True       # Aktifkan ROI untuk fokus deteksi conveyor belt
ROI_MODE = "manual"     # "manual" = set manual, "auto" = full frame, "interactive" = klik untuk set
ROI_SHAPE = "rectangle" # "rectangle", "polygon", "circle"
# ROI coordinates - akan diupdate jika interactive mode
ROI_COORDS = [100, 100, 500, 400]  # Default ROI area (rectangle)
ROI_POLYGON_POINTS = []  # Untuk polygon mode
ROI_CIRCLE_CENTER = (300, 250)  # Center point untuk circle
ROI_CIRCLE_RADIUS = 150  # Radius untuk circle
ROI_COLOR = (0, 255, 255)  # Cyan color untuk ROI
ROI_THICKNESS = 2

# I/O & Tampilan
DISPLAY = True
DRAW_THICK = 2
WARMUP_FRAMES = 10     # RTSP/Video/Webcam
SAVE_OUTPUT = False    # Set False untuk mematikan save ke folder
OUTPUT_DIR = "outputs"
OUTPUT_NAME = "result"

# =======================
# ROI Management
# =======================
class ROIManager:
    def __init__(self, mode="manual", coords=None, shape="rectangle"):
        self.mode = mode
        self.shape = shape
        self.coords = coords or [100, 100, 500, 400]  # rectangle coords
        self.polygon_points = []  # untuk polygon
        self.circle_center = (300, 250)  # untuk circle
        self.circle_radius = 150  # untuk circle
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.roi_set = False
        self.polygon_drawing = False  # untuk polygon mode
        
    def set_roi_from_frame(self, frame):
        """Set ROI berdasarkan ukuran frame"""
        h, w = frame.shape[:2]
        if self.mode == "auto":
            # Full frame ROI
            if self.shape == "rectangle":
                self.coords = [0, 0, w, h]
            elif self.shape == "circle":
                self.circle_center = (w//2, h//2)
                self.circle_radius = min(w, h) // 3
            elif self.shape == "polygon":
                # Buat polygon default (persegi)
                margin = min(w, h) // 4
                self.polygon_points = [
                    [margin, margin],
                    [w-margin, margin],
                    [w-margin, h-margin],
                    [margin, h-margin]
                ]
            self.roi_set = True
        elif self.mode == "manual":
            # Manual ROI dengan default 50% dari frame
            margin_x = w // 4
            margin_y = h // 4
            if self.shape == "rectangle":
                self.coords = [margin_x, margin_y, w - margin_x, h - margin_y]
            elif self.shape == "circle":
                self.circle_center = (w//2, h//2)
                self.circle_radius = min(w, h) // 3
            elif self.shape == "polygon":
                # Buat polygon default (persegi)
                self.polygon_points = [
                    [margin_x, margin_y],
                    [w-margin_x, margin_y],
                    [w-margin_x, h-margin_y],
                    [margin_x, h-margin_y]
                ]
            self.roi_set = True
            
    def get_roi_mask(self, frame):
        """Buat mask untuk ROI area"""
        if not self.roi_set:
            return None
            
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        if self.shape == "rectangle":
            x1, y1, x2, y2 = self.coords
            mask[y1:y2, x1:x2] = 255
        elif self.shape == "circle":
            cv2.circle(mask, self.circle_center, self.circle_radius, 255, -1)
        elif self.shape == "polygon" and len(self.polygon_points) >= 3:
            points = np.array(self.polygon_points, dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
            
        return mask
        
    def is_point_in_roi(self, x, y):
        """Cek apakah point berada dalam ROI"""
        if not self.roi_set:
            return True
            
        if self.shape == "rectangle":
            x1, y1, x2, y2 = self.coords
            return x1 <= x <= x2 and y1 <= y <= y2
        elif self.shape == "circle":
            center_x, center_y = self.circle_center
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            return distance <= self.circle_radius
        elif self.shape == "polygon" and len(self.polygon_points) >= 3:
            point = np.array([x, y], dtype=np.int32)
            points = np.array(self.polygon_points, dtype=np.int32)
            return cv2.pointPolygonTest(points, (x, y), False) >= 0
        return True
        
    def is_bbox_in_roi(self, bbox):
        """Cek apakah bounding box berada dalam ROI"""
        if not self.roi_set:
            return True
            
        x1, y1, x2, y2 = bbox
        
        if self.shape == "rectangle":
            roi_x1, roi_y1, roi_x2, roi_y2 = self.coords
            
            # Cek overlap antara bbox dan ROI
            overlap_x1 = max(x1, roi_x1)
            overlap_y1 = max(y1, roi_y1)
            overlap_x2 = min(x2, roi_x2)
            overlap_y2 = min(y2, roi_y2)
            
            if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                # Hitung area overlap
                bbox_area = (x2 - x1) * (y2 - y1)
                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                # Return True jika lebih dari 50% bbox berada dalam ROI
                return overlap_area / bbox_area > 0.5
            return False
            
        elif self.shape == "circle":
            # Cek apakah center bbox berada dalam circle
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            return self.is_point_in_roi(center_x, center_y)
            
        elif self.shape == "polygon":
            # Cek apakah center bbox berada dalam polygon
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            return self.is_point_in_roi(center_x, center_y)
            
        return True
        
    def draw_roi(self, frame):
        """Gambar ROI pada frame"""
        if not self.roi_set:
            return frame
            
        if self.shape == "rectangle":
            x1, y1, x2, y2 = self.coords
            cv2.rectangle(frame, (x1, y1), (x2, y2), ROI_COLOR, ROI_THICKNESS)
            label_x, label_y = x1, max(y1 - 10, 20)
            
        elif self.shape == "circle":
            cv2.circle(frame, self.circle_center, self.circle_radius, ROI_COLOR, ROI_THICKNESS)
            label_x = self.circle_center[0] - 50
            label_y = max(self.circle_center[1] - self.circle_radius - 10, 20)
            
        elif self.shape == "polygon" and len(self.polygon_points) >= 3:
            points = np.array(self.polygon_points, dtype=np.int32)
            cv2.polylines(frame, [points], True, ROI_COLOR, ROI_THICKNESS)
            # Gambar titik-titik polygon
            for point in self.polygon_points:
                cv2.circle(frame, tuple(point), 3, ROI_COLOR, -1)
            label_x = min([p[0] for p in self.polygon_points])
            label_y = max(min([p[1] for p in self.polygon_points]) - 10, 20)
        
        # Tambah label "ROI - Conveyor Belt"
        label = f"ROI - Conveyor Belt ({self.shape})"
        cv2.putText(frame, label, (label_x, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, ROI_COLOR, ROI_THICKNESS)
        
        return frame

# =======================
# Interactive ROI Setup
# =======================
def mouse_callback(event, x, y, flags, param):
    """Mouse callback untuk interactive ROI setup"""
    roi_manager = param
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if roi_manager.shape == "rectangle":
            roi_manager.drawing = True
            roi_manager.start_point = (x, y)
            roi_manager.end_point = (x, y)
        elif roi_manager.shape == "circle":
            roi_manager.circle_center = (x, y)
            roi_manager.drawing = True
        elif roi_manager.shape == "polygon":
            roi_manager.polygon_points.append([x, y])
            roi_manager.polygon_drawing = True
            
    elif event == cv2.EVENT_MOUSEMOVE:
        if roi_manager.drawing:
            if roi_manager.shape == "rectangle":
                roi_manager.end_point = (x, y)
            elif roi_manager.shape == "circle":
                # Hitung radius dari center ke mouse position
                center_x, center_y = roi_manager.circle_center
                radius = int(np.sqrt((x - center_x)**2 + (y - center_y)**2))
                roi_manager.circle_radius = radius
                
    elif event == cv2.EVENT_LBUTTONUP:
        if roi_manager.shape == "rectangle" and roi_manager.drawing:
            roi_manager.drawing = False
            if roi_manager.start_point and roi_manager.end_point:
                x1 = min(roi_manager.start_point[0], roi_manager.end_point[0])
                y1 = min(roi_manager.start_point[1], roi_manager.end_point[1])
                x2 = max(roi_manager.start_point[0], roi_manager.end_point[0])
                y2 = max(roi_manager.start_point[1], roi_manager.end_point[1])
                roi_manager.coords = [x1, y1, x2, y2]
                roi_manager.roi_set = True
                print(f"[ROI] Rectangle set: {roi_manager.coords}")
        elif roi_manager.shape == "circle" and roi_manager.drawing:
            roi_manager.drawing = False
            roi_manager.roi_set = True
            print(f"[ROI] Circle set: center={roi_manager.circle_center}, radius={roi_manager.circle_radius}")
        elif roi_manager.shape == "polygon":
            # Double click untuk finish polygon
            if len(roi_manager.polygon_points) >= 3:
                roi_manager.roi_set = True
                print(f"[ROI] Polygon set: {len(roi_manager.polygon_points)} points")

def setup_interactive_roi(reader, roi_manager):
    """Setup ROI secara interactive dengan mouse"""
    shape_instructions = {
        "rectangle": "Click and drag to set rectangle ROI",
        "circle": "Click center, then drag to set radius",
        "polygon": "Click to add points, press Enter when done"
    }
    
    print(f"[ROI] Mode interactive - {shape_instructions.get(roi_manager.shape, 'Set ROI area')}")
    print("[ROI] Tekan 'Enter' untuk konfirmasi, 'r' untuk reset, 'q' untuk keluar")
    
    window_name = "Set ROI - Conveyor Belt Detection"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback, roi_manager)
    
    while True:
        frame = reader()
        if frame is None:
            time.sleep(0.01)
            continue
            
        # Set ROI default jika belum ada
        if not roi_manager.roi_set:
            roi_manager.set_roi_from_frame(frame)
            
        # Gambar ROI sementara jika sedang drag
        display_frame = frame.copy()
        
        if roi_manager.shape == "rectangle" and roi_manager.drawing and roi_manager.start_point and roi_manager.end_point:
            cv2.rectangle(display_frame, roi_manager.start_point, roi_manager.end_point, 
                         ROI_COLOR, ROI_THICKNESS)
        elif roi_manager.shape == "circle" and roi_manager.drawing:
            cv2.circle(display_frame, roi_manager.circle_center, roi_manager.circle_radius, 
                      ROI_COLOR, ROI_THICKNESS)
        elif roi_manager.shape == "polygon":
            # Gambar polygon yang sedang dibuat
            if len(roi_manager.polygon_points) > 0:
                for i, point in enumerate(roi_manager.polygon_points):
                    cv2.circle(display_frame, tuple(point), 3, ROI_COLOR, -1)
                    if i > 0:
                        cv2.line(display_frame, tuple(roi_manager.polygon_points[i-1]), 
                               tuple(point), ROI_COLOR, ROI_THICKNESS)
        else:
            roi_manager.draw_roi(display_frame)
            
        # Tambah instruksi
        instruction = shape_instructions.get(roi_manager.shape, "Set ROI area")
        cv2.putText(display_frame, instruction, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, "Press Enter to confirm, R to reset, Q to quit", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow(window_name, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyWindow(window_name)
            return False
        elif key == ord('r'):
            roi_manager.roi_set = False
            roi_manager.polygon_points = []
            if roi_manager.shape == "rectangle":
                roi_manager.coords = [100, 100, 500, 400]
            elif roi_manager.shape == "circle":
                roi_manager.circle_center = (300, 250)
                roi_manager.circle_radius = 150
        elif key == 13:  # Enter key
            if roi_manager.roi_set:
                cv2.destroyWindow(window_name)
                return True
    
    cv2.destroyWindow(window_name)
    return False

# =======================
# Util: Text overlay
# =======================
def put_text(img, txt, org, scale=0.6, color=(0,255,255), thick=2):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

# =======================
# Threaded RTSP Reader
# =======================
class RTSPStream:
    def __init__(self, url, use_gst=False, width=None, height=None, queue_size=2):
        self.url = url
        self.use_gst = use_gst
        self.width = width
        self.height = height
        self.q = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.cap = None
        self.thread = None

    def _gst_pipeline(self):
        # H264 → BGR (sesuaikan bila perlu)
        return (
            f"rtspsrc location={self.url} latency=0 drop-on-late=true ! "
            f"rtph264depay ! h264parse ! avdec_h264 ! "
            f"videoconvert ! videoscale ! "
            f"video/x-raw,format=BGR"
            f"{',width='+str(self.width) if self.width else ''}"
            f"{',height='+str(self.height) if self.height else ''} ! "
            f"appsink sync=false max-buffers=1 drop=true"
        )

    def start(self):
        if self.use_gst:
            pipeline = self._gst_pipeline()
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            backend = "GSTREAMER"
        else:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            backend = "FFMPEG"
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # reduce buffering

        if not self.cap.isOpened():
            raise RuntimeError("Gagal membuka RTSP stream.")

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 0
        print(f"[RTSP] Backend: {backend}, Resolusi: {w}x{h}, Kamera FPS (lapor): {fps:.2f}")

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            # drop frame lama untuk low-latency
            if not self.q.empty():
                try: self.q.get_nowait()
                except queue.Empty: pass
            self.q.put(frame)

        if self.cap is not None:
            self.cap.release()

    def get_frame(self, timeout=0.5):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.stopped = True
        if self.thread is not None:
            self.thread.join(timeout=1.0)

# =======================
# Video Reader (file)
# =======================
class VideoReader:
    def __init__(self, path):
        self.path = path
        self.cap = None

    def start(self):
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Gagal membuka video: {self.path}")
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 0
        print(f"[VIDEO] {self.path}\n       Resolusi: {w}x{h}, FPS(file): {fps:.2f}")
        return self

    def get_frame(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def stop(self):
        if self.cap is not None:
            self.cap.release()

# =======================
# Image Reader (file)
# =======================
class ImageReader:
    def __init__(self, path):
        self.path = path
        self._done = False

    def start(self):
        if not Path(self.path).exists():
            raise RuntimeError(f"Gambar tidak ditemukan: {self.path}")
        return self

    def get_frame(self):
        if self._done:
            return None
        img = cv2.imread(self.path)
        if img is None:
            raise RuntimeError("Gagal memuat gambar (format rusak?)")
        self._done = True
        print(f"[IMAGE] {self.path} Resolusi: {img.shape[1]}x{img.shape[0]}")
        return img

    def stop(self): pass

# =======================
# Draw Detections with ROI Filter
# =======================
def draw_detections(frame, results, names, roi_manager=None, thick=2):
    detection_count = 0
    roi_detection_count = 0
    
    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        for b in boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            conf = float(b.conf[0])
            cls_id = int(b.cls[0])
            label = f"{names.get(cls_id, str(cls_id))}: {conf:.2f}"
            
            # Cek apakah detection berada dalam ROI
            in_roi = True
            if roi_manager and ENABLE_ROI:
                in_roi = roi_manager.is_bbox_in_roi([x1, y1, x2, y2])
            
            if in_roi:
                # Detection dalam ROI - warna hijau
                color = (0, 255, 0)
                roi_detection_count += 1
            else:
                # Detection di luar ROI - warna merah (optional, bisa di-disable)
                color = (0, 0, 255)
            
            detection_count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
            cv2.putText(frame, label, (x1, max(0, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thick)
    
    # Tambah info deteksi
    if ENABLE_ROI and roi_manager:
        put_text(frame, f"Total Detections: {detection_count}", (10, 60))
        put_text(frame, f"ROI Detections: {roi_detection_count}", (10, 90))
    
    return frame

# =======================
# Writer util
# =======================
def build_videowriter(out_path, w, h, fps=25):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(out_path), fourcc, max(1.0, fps), (w, h))

# =======================
# Main
# =======================
def main():
    # Output dir
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"[MODEL] Load: {MODEL_PATH}, device={DEVICE}, imgsz={IMGSZ}")
    model = YOLO(MODEL_PATH)
    names = model.names

    # Initialize ROI Manager
    roi_manager = ROIManager(mode=ROI_MODE, coords=ROI_COORDS, shape=ROI_SHAPE)
    
    # Siapkan source
    out_video = None

    if SOURCE_TYPE == "rtsp":
        reader = RTSPStream(RTSP_URL, use_gst=USE_GSTREAMER).start()
        print(f"[INFO] Warmup {WARMUP_FRAMES} frames...")
        for _ in range(WARMUP_FRAMES):
            reader.get_frame()

        read_fn = lambda: reader.get_frame()

    elif SOURCE_TYPE == "video":
        vr = VideoReader(VIDEO_PATH).start()
        print(f"[INFO] Warmup {WARMUP_FRAMES} frames...")
        for _ in range(WARMUP_FRAMES):
            if vr.get_frame() is None: break
        read_fn = lambda: vr.get_frame()

    elif SOURCE_TYPE == "image":
        ir = ImageReader(IMAGE_PATH).start()
        read_fn = lambda: ir.get_frame()

    elif SOURCE_TYPE == "webcam":
        cap = cv2.VideoCapture(WEBCAM_INDEX)
        if not cap.isOpened():
            raise RuntimeError(f"Gagal membuka webcam index {WEBCAM_INDEX}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        print(f"[WEBCAM] index {WEBCAM_INDEX} Resolusi: {w}x{h}, FPS (lapor): {fps:.2f}")

        print(f"[INFO] Warmup {WARMUP_FRAMES} frames...")
        for _ in range(WARMUP_FRAMES):
            cap.read()

        read_fn = lambda: (cap.read()[1] if cap.read()[0] else None)

    else:
        raise ValueError("SOURCE_TYPE harus salah satu: 'rtsp' | 'video' | 'image' | 'webcam'")

    # Setup ROI
    if ENABLE_ROI:
        print(f"[ROI] Mode: {ROI_MODE}")
        if ROI_MODE == "interactive":
            # Setup ROI secara interactive
            if not setup_interactive_roi(read_fn, roi_manager):
                print("[INFO] ROI setup dibatalkan. Keluar...")
                return
        else:
            # Auto atau manual ROI setup
            first_frame = read_fn()
            if first_frame is not None:
                roi_manager.set_roi_from_frame(first_frame)
                print(f"[ROI] Set otomatis: {roi_manager.coords}")
        
        if roi_manager.roi_set:
            print(f"[ROI] Aktif - Area: {roi_manager.coords}")
        else:
            print("[ROI] Tidak aktif - deteksi seluruh frame")

    print("[INFO] Mulai inferensi...")
    t_prev = time.time()
    fps_deque = deque(maxlen=30)

    try:
        while True:
            frame = read_fn()
            if frame is None:
                # Habis (video/image) atau belum ada (rtsp/webcam)
                if SOURCE_TYPE in ("video", "image"):
                    break
                else:
                    time.sleep(0.001)
                    continue

            # Gambar ROI jika aktif
            if ENABLE_ROI and roi_manager.roi_set:
                roi_manager.draw_roi(frame)

            # Inference
            results = model.predict(
                source=frame,
                imgsz=IMGSZ,
                device=DEVICE,
                conf=CONF,
                verbose=False
            )
            
            # Draw detections dengan ROI filter
            draw_detections(frame, results, names, roi_manager, thick=DRAW_THICK)

            # FPS overlay
            now = time.time()
            fps_deque.append(1.0 / max(1e-6, (now - t_prev)))
            t_prev = now
            fps = sum(fps_deque) / len(fps_deque)
            put_text(frame, f"FPS: {fps:.1f}", (10, 30))

            # Tampilkan
            if DISPLAY:
                window_title = "YOLO EdgeTPU - Conveyor Belt Detection"
                if ENABLE_ROI:
                    window_title += " (ROI Active)"
                cv2.imshow(window_title, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Simpan
            if SAVE_OUTPUT:
                if SOURCE_TYPE == "image":
                    out_img = Path(OUTPUT_DIR) / f"{OUTPUT_NAME}.jpg"
                    cv2.imwrite(str(out_img), frame)
                else:  # video/rtsp/webcam
                    if out_video is None:
                        h, w = frame.shape[:2]
                        # ambil fps sumber jika video; default 25 utk lainnya
                        src_fps = 25.0
                        if SOURCE_TYPE == "video":
                            cap_tmp = cv2.VideoCapture(VIDEO_PATH)
                            src_fps = cap_tmp.get(cv2.CAP_PROP_FPS) or 25.0
                            cap_tmp.release()
                        out_path = Path(OUTPUT_DIR) / f"{OUTPUT_NAME}.mp4"
                        out_video = build_videowriter(out_path, w, h, fps=src_fps)
                        print(f"[SAVE] Rekam ke: {out_path} @ {src_fps:.2f} fps")
                    out_video.write(frame)

    finally:
        # cleanup
        if SOURCE_TYPE == "rtsp":
            reader.stop()
        elif SOURCE_TYPE == "video":
            vr.stop()
        elif SOURCE_TYPE == "image":
            ir.stop()
        elif SOURCE_TYPE == "webcam":
            cap.release()

        if out_video is not None:
            out_video.release()
        cv2.destroyAllWindows()
        print("[INFO] Selesai.")

if __name__ == "__main__":
    main()
