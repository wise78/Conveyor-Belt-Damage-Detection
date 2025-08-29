#!/usr/bin/env python3
# RTSP CCTV Dataset Collector + ROI (rect/poly), interactive & config-based
# - Threaded RTSP (low-latency, auto-reconnect)
# - Snapshot tiap N detik (configurable)
# - Quality filter (blur/brightness) & anti-duplicate (pHash) pada ROI
# - ROI: rect / polygon; apply mode: crop/mask
# - Interactive ROI: R (rect), P (poly), C (clear), S (save), L (load), Q (quit), BACKSPACE (undo 1 titik)
# - Kompatibel Python 3.9

import os
import cv2
import csv
import json
import time
import queue
import threading
import numpy as np
from pathlib import Path
from datetime import datetime

# =======================
# KONFIG
# =======================
RTSP_URL = "rtsp://username:password@IP:554/Streaming/Channels/101"
USE_GSTREAMER = False           # True untuk Linux/RPi + GStreamer; Windows -> False
READ_WIDTH = None               # contoh 1280 (opsional)
READ_HEIGHT = None              # contoh 720 (opsional)

SAVE_ROOT = "dataset_snapshots"
INTERVAL_SEC = 2.0              # target interval simpan
JPEG_QUALITY = 95
WARMUP_FRAMES = 15

DISPLAY_PREVIEW = True
SHOW_FPS = True
SHOW_HINTS = True               # tampilkan petunjuk tombol

# ---- Quality Filters pada ROI (set None untuk disable) ----
MIN_LAP_VAR = 60.0              # minimal ketajaman (variance of Laplacian)
MIN_BRIGHT = 25.0               # rata-rata brightness min (0-255)
MAX_BRIGHT = 235.0              # rata-rata brightness max (0-255)

# ---- Anti-duplicate (pHash) pada ROI ----
ANTI_DUPLICATE = True
MIN_HAMMING_DIST = 6            # 0..64 (lebih besar = lebih ketat)

# ---- Kontrol interval saat skip ----
UPDATE_TIMER_ON_SKIP = False    # False: jadwal tidak maju jika skip (default disarankan)
FORCE_SAVE_MAX_GAP_SEC = 8.0    # paksa simpan (bypass anti-dup) jika terakhir save > nilai ini (set None untuk disable)

# ---- Reconnect ----
AUTO_RECONNECT = True
RECONNECT_COOLDOWN_SEC = 3.0
MAX_READ_FAIL_BEFORE_REOPEN = 50

# ---- ROI SETTINGS ----
ROI_MODE = "none"               # "none" | "rect" | "poly"
ROI_RECT = (400, 200, 800, 400) # x,y,w,h (kalau ROI_MODE="rect")
ROI_POLY = [                    # [(x1,y1), (x2,y2), ...] jika "poly"
    # (100,100), (600,120), (620,400), (140,380)
]
ROI_APPLY = "crop"              # "crop" (simpan area ROI saja) atau "mask" (full-frame, luar ROI hitam)

# Simpan/muat ROI ke file (agar persist)
ROI_CONFIG_FILE = "roi_config.json"
ALLOW_INTERACTIVE_ROI = True    # tekan R/P/C/S/L saat preview

# =======================
# RTSP Threaded Reader
# =======================
class RTSPStream:
    def __init__(self, url, use_gst=False, width=None, height=None, queue_size=2, auto_reconnect=True):
        self.url = url
        self.use_gst = use_gst
        self.width = width
        self.height = height
        self.q = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.cap = None
        self.thread = None
        self.auto_reconnect = auto_reconnect
        self._fail_count = 0

    def _gst_pipeline(self):
        return (
            f"rtspsrc location={self.url} latency=0 drop-on-late=true ! "
            f"rtph264depay ! h264parse ! avdec_h264 ! "
            f"videoconvert ! videoscale ! "
            f"video/x-raw,format=BGR"
            f"{',width='+str(self.width) if self.width else ''}"
            f"{',height='+str(self.height) if self.height else ''} ! "
            f"appsink sync=false max-buffers=1 drop=true"
        )

    def _open(self):
        if self.use_gst:
            pipeline = self._gst_pipeline()
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            backend = "GSTREAMER"
        else:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            backend = "FFMPEG"
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok = self.cap.isOpened()
        if ok:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0)
            print(f"[RTSP] Opened via {backend} | {w}x{h} @ {fps:.2f} fps")
        else:
            print("[RTSP] FAILED to open stream")
        return ok

    def start(self):
        if not self._open():
            raise RuntimeError("Gagal membuka RTSP stream.")
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self._fail_count += 1
                time.sleep(0.005)
                if self.auto_reconnect and self._fail_count >= MAX_READ_FAIL_BEFORE_REOPEN:
                    print("[RTSP] Too many read failures. Reconnecting...")
                    self._reconnect()
                continue
            self._fail_count = 0
            if not self.q.empty():
                try: self.q.get_nowait()
                except queue.Empty: pass
            self.q.put(frame)
        if self.cap is not None:
            self.cap.release()

    def _reconnect(self):
        if self.cap is not None:
            self.cap.release()
        time.sleep(RECONNECT_COOLDOWN_SEC)
        self._open()

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
# Utils: quality & hash (Py 3.9 compatible)
# =======================
def variance_of_laplacian(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def mean_brightness(gray):
    return float(gray.mean())

def average_hash(gray, hash_size=8):
    small = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    avg = small.mean()
    bits = (small > avg).astype(np.uint8).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return int(value)

def _popcount(x: int) -> int:
    x = int(x)
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c

def hamming_distance_hash(a: int, b: int) -> int:
    x = int(a) ^ int(b)
    if hasattr(x, "bit_count"):
        return x.bit_count()
    return _popcount(x)

# =======================
# Polygon State Manager
# =======================
class PolygonState:
    def __init__(self):
        self.drawing = False
        self.points = []
    
    def start_drawing(self):
        self.drawing = True
        self.points = []
        print("[ROI] Polygon mode started")
        print(f"[DEBUG] Reset points, length now: {len(self.points)}")
    
    def add_point(self, x, y):
        if self.drawing:
            self.points.append((x, y))
            print(f"[ROI] Added point {len(self.points)}: ({x}, {y})")
            print(f"[DEBUG] Total points now: {len(self.points)}")
            return True
        return False
    
    def finish_drawing(self):
        if len(self.points) >= 3:
            result = self.points.copy()
            self.drawing = False
            self.points = []
            print(f"[ROI] Set POLY with {len(result)} points")
            return result
        else:
            print("[ROI] Butuh >=3 titik untuk polygon; dibatalkan")
            self.drawing = False
            self.points = []
            return None
    
    def cancel_drawing(self):
        self.drawing = False
        self.points = []
        print("[ROI] Polygon dibatalkan")
    
    def remove_last_point(self):
        if self.drawing and len(self.points) > 0:
            removed = self.points.pop()
            print(f"[ROI] Removed point: {removed}")
    
    def get_points(self):
        return self.points.copy()
    
    def is_drawing(self):
        return self.drawing

# =======================
# ROI helpers
# =======================
def clamp_rect(x, y, w, h, W, H):
    x = max(0, min(x, W-1))
    y = max(0, min(y, H-1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return int(x), int(y), int(w), int(h)

def apply_roi(frame, mode, rect, poly, apply_mode):
    H, W = frame.shape[:2]
    if mode == "none":
        return frame, None, None

    if mode == "rect":
        x, y, w, h = rect
        x, y, w, h = clamp_rect(int(x), int(y), int(w), int(h), W, H)
        roi = frame[y:y+h, x:x+w]
        if apply_mode == "crop":
            return roi, None, (x, y, w, h)
        else:
            out = np.zeros_like(frame)
            out[y:y+h, x:x+w] = roi
            return out, None, (x, y, w, h)

    if mode == "poly" and len(poly) >= 3:
        pts = np.array(poly, dtype=np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, W-1)
        pts[:, 1] = np.clip(pts[:, 1], 0, H-1)

        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        if apply_mode == "crop":
            x, y, w, h = cv2.boundingRect(pts)
            crop = frame[y:y+h, x:x+w].copy()
            crop_mask = mask[y:y+h, x:x+w]
            crop[crop_mask == 0] = 0
            return crop, pts.tolist(), None
        else:
            out = frame.copy()
            out[mask == 0] = 0
            return out, pts.tolist(), None

    return frame, None, None

def draw_roi_preview(img, vis_poly, vis_rect):
    if vis_rect is not None:
        x,y,w,h = vis_rect
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,255), 2)
    if vis_poly is not None and len(vis_poly) >= 2:
        pts = np.array(vis_poly, dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=(0,255,255), thickness=2)

def save_roi_config(path, mode, rect, poly, apply_mode):
    data = {"mode": mode, "rect": list(rect) if rect else None, "poly": poly if poly else [], "apply": apply_mode}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[ROI] Saved config -> {path}")

def load_roi_config(path):
    p = Path(path)
    if not p.exists():
        print(f"[ROI] Config file not found: {path}")
        return None
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    mode = data.get("mode", "none")
    rect = tuple(data.get("rect", (0,0,0,0))) if data.get("rect") else (0,0,0,0)
    poly = data.get("poly", [])
    apply_mode = data.get("apply", "crop")
    print(f"[ROI] Loaded: mode={mode}, rect={rect}, poly_points={len(poly)}, apply={apply_mode}")
    return mode, rect, poly, apply_mode

# =======================
# IO helpers
# =======================
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)
def timestamp_str():
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
def today_folder(root: Path):
    return root / datetime.now().strftime("%Y-%m-%d")

# =======================
# Main
# =======================
def main():
    global ROI_MODE, ROI_RECT, ROI_POLY, ROI_APPLY
    
    # Polygon state manager
    polygon_state = PolygonState()

    # Load ROI dari file jika ada
    if Path(ROI_CONFIG_FILE).exists():
        loaded = load_roi_config(ROI_CONFIG_FILE)
        if loaded is not None:
            ROI_MODE, ROI_RECT, ROI_POLY, ROI_APPLY = loaded

    # init save dirs
    save_root = Path(SAVE_ROOT)
    ensure_dir(save_root)
    day_dir = today_folder(save_root)
    ensure_dir(day_dir)
    meta_path = day_dir / "metadata.csv"

    meta_file = open(meta_path, "a", newline="", encoding="utf-8")
    meta_writer = csv.writer(meta_file)
    if meta_file.tell() == 0:
        meta_writer.writerow([
            "timestamp","filename","orig_w","orig_h",
            "roi_mode","roi_apply","roi_rect","roi_poly_points",
            "lap_var","brightness","phash"
        ])

    # start stream
    stream = RTSPStream(
        RTSP_URL, use_gst=USE_GSTREAMER,
        width=READ_WIDTH, height=READ_HEIGHT,
        auto_reconnect=AUTO_RECONNECT
    ).start()

    # warmup
    print(f"[INFO] Warmup {WARMUP_FRAMES} frames...")
    pulled = 0
    while pulled < WARMUP_FRAMES:
        if stream.get_frame() is not None:
            pulled += 1

    print(f"[INFO] Start capturing target every {INTERVAL_SEC}s")
    last_save_wall = time.monotonic() - INTERVAL_SEC
    next_due_t = last_save_wall + INTERVAL_SEC
    last_hash = None
    fps_est, fps_alpha = None, 0.9
    t_prev = time.monotonic()

    # Polygon drawing state sudah diinisialisasi sebagai global di atas

    if DISPLAY_PREVIEW:
        cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
        
        # Pastikan window aktif dan dapat menerima input mouse
        cv2.setWindowProperty("Preview", cv2.WND_PROP_TOPMOST, 1)

        def on_mouse(event, x, y, flags, param):
            # Debug: print semua event mouse untuk troubleshooting
            if event in [cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN]:
                print(f"[DEBUG] Mouse event: {event} at ({x}, {y})")
                print(f"[DEBUG] Current points length: {len(polygon_state.get_points())}")
            
            if event == cv2.EVENT_LBUTTONDOWN:
                polygon_state.add_point(x, y)
            elif event == cv2.EVENT_RBUTTONDOWN:
                # Klik kanan juga bisa digunakan untuk menambah titik
                polygon_state.add_point(x, y)
        
        cv2.setMouseCallback("Preview", on_mouse)
        print("[ROI] Mouse callback registered for polygon drawing")

    try:
        while True:
            frame = stream.get_frame()
            if frame is None:
                if DISPLAY_PREVIEW:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue

            H, W = frame.shape[:2]

            # FPS preview
            now = time.monotonic()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                cur_fps = 1.0 / dt
                fps_est = cur_fps if fps_est is None else (fps_alpha*fps_est + (1-fps_alpha)*cur_fps)

            # ROI apply untuk preview dan evaluasi kualitas
            roi_img, vis_poly, vis_rect = apply_roi(frame, ROI_MODE, ROI_RECT, ROI_POLY, ROI_APPLY)

            # draw preview: ROI borders + hints
            preview = frame.copy()
            draw_roi_preview(preview, vis_poly, vis_rect)
            if DISPLAY_PREVIEW:
                if SHOW_FPS and fps_est is not None:
                    cv2.putText(preview, f"FPS~{fps_est:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
                if SHOW_HINTS:
                    y0 = 60
                    hints = [
                        f"ROI: mode={ROI_MODE}, apply={ROI_APPLY}",
                        "Keys: R=Rect | P=Poly | C=Clear | S=Save ROI | L=Load ROI | Q=Quit",
                        "Poly: LEFT/RIGHT click titik-titik, ENTER selesai, ESC batal, BACKSPACE hapus 1 titik"
                    ]
                    for i, h in enumerate(hints):
                        cv2.putText(preview, h, (10, y0 + i*22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2, cv2.LINE_AA)
                # Draw polygon points if drawing
                if polygon_state.is_drawing():
                    points = polygon_state.get_points()
                    if len(points) >= 1:
                        # Tampilkan status polygon drawing
                        cv2.putText(preview, f"POLYGON MODE - Points: {len(points)}", 
                                    (10, H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
                        
                        for i, pt in enumerate(points):
                            cv2.circle(preview, pt, 5, (0, 255, 255), -1)
                            cv2.putText(preview, str(i+1), (pt[0]+8, pt[1]-8), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
                        
                        if len(points) >= 2:
                            cv2.polylines(preview, [np.array(points, dtype=np.int32)],
                                          False, (0,255,255), 2)
                cv2.imshow("Preview", preview)
                # Pastikan window tetap aktif untuk menerima input mouse
                cv2.setWindowProperty("Preview", cv2.WND_PROP_TOPMOST, 1)

            # cek apakah sudah waktunya attempt save
            now = time.monotonic()
            due = (now >= next_due_t)
            force_due = (FORCE_SAVE_MAX_GAP_SEC is not None and (now - last_save_wall) >= float(FORCE_SAVE_MAX_GAP_SEC))

            if not due and not force_due:
                # belum waktunya
                if DISPLAY_PREVIEW and ALLOW_INTERACTIVE_ROI:
                    # Gunakan waitKey yang lebih responsif untuk input mouse
                    key = cv2.waitKey(10) & 0xFF
                    if key == ord('q') or key == ord('Q'):
                        break
                    elif key == ord('c') or key == ord('C'):
                        ROI_MODE = "none"; ROI_POLY = []; polygon_state.cancel_drawing(); print("[ROI] Cleared")
                    elif key == ord('l') or key == ord('L'):
                        loaded = load_roi_config(ROI_CONFIG_FILE)
                        if loaded is not None:
                            ROI_MODE, ROI_RECT, ROI_POLY, ROI_APPLY = loaded
                    elif key == ord('s') or key == ord('S'):
                        save_roi_config(ROI_CONFIG_FILE, ROI_MODE, ROI_RECT, ROI_POLY, ROI_APPLY)
                    elif key == ord('r') or key == ord('R'):
                        sel = cv2.selectROI("Preview", frame, fromCenter=False, showCrosshair=True)
                        x,y,w,h = sel
                        if w > 0 and h > 0:
                            ROI_MODE = "rect"
                            ROI_RECT = (int(x), int(y), int(w), int(h))
                            ROI_POLY = []
                            polygon_state.cancel_drawing()
                            print(f"[ROI] Set RECT: {ROI_RECT}")
                        try: cv2.destroyWindow("ROI selector")
                        except Exception: pass
                    elif key == ord('p') or key == ord('P'):
                        if not polygon_state.is_drawing():
                            ROI_MODE = "poly"
                            ROI_POLY = []
                            polygon_state.start_drawing()
                            print("[ROI] Polygon: LEFT/RIGHT click titik-titik, ENTER selesai, ESC batal, BACKSPACE undo")
                    elif key in (8, 255):  # BACKSPACE (beberapa platform 8, ada juga 255)
                        polygon_state.remove_last_point()
                    elif key in (13, 10):  # ENTER (Linux bisa 10)
                        if polygon_state.is_drawing():
                            result = polygon_state.finish_drawing()
                            if result is not None:
                                ROI_POLY = result
                    elif key == 27:  # ESC
                        if polygon_state.is_drawing():
                            polygon_state.cancel_drawing()
                continue

            # --- Saatnya attempt save ---
            # Quality checks pada ROI
            gray_src = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
            lap_var = variance_of_laplacian(gray_src)
            bright = mean_brightness(gray_src)

            quality_ok = True
            if MIN_LAP_VAR is not None and lap_var < MIN_LAP_VAR:
                quality_ok = False
            if (MIN_BRIGHT is not None and bright < MIN_BRIGHT) or \
               (MAX_BRIGHT is not None and bright > MAX_BRIGHT):
                quality_ok = False

            dup_ok = True
            phash = None
            if ANTI_DUPLICATE:
                ph = average_hash(gray_src, hash_size=8)
                phash = ph
                if last_hash is not None:
                    try:
                        dist = hamming_distance_hash(last_hash, ph)
                    except Exception:
                        dist = 999
                else:
                    dist = 999
                dup_ok = (dist >= MIN_HAMMING_DIST)

            # force-save rules: bypass anti-duplicate bila gap terlalu lama
            if force_due and not dup_ok:
                dup_ok = True  # override anti-dup
                # (filter kualitas tetap dipakai agar tidak menimbun blur/gelap)

            will_save = (quality_ok and dup_ok)

            if will_save:
                # rotate folder harian jika berganti hari
                cur_day_dir = today_folder(Path(SAVE_ROOT))
                if cur_day_dir != day_dir:
                    day_dir = cur_day_dir
                    ensure_dir(day_dir)
                    meta_file.close()
                    new_meta = day_dir / "metadata.csv"
                    meta_file = open(new_meta, "a", newline="", encoding="utf-8")
                    meta_writer = csv.writer(meta_file)
                    if meta_file.tell() == 0:
                        meta_writer.writerow([
                            "timestamp","filename","orig_w","orig_h",
                            "roi_mode","roi_apply","roi_rect","roi_poly_points",
                            "lap_var","brightness","phash"
                        ])

                ts = timestamp_str()
                fname = f"{ts}.jpg"
                out_path = day_dir / fname
                ok = False
                try:
                    ok = cv2.imwrite(str(out_path), roi_img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                except Exception:
                    ok = False
                if ok:
                    meta_writer.writerow([
                        ts, fname, W, H, ROI_MODE, ROI_APPLY,
                        list(ROI_RECT) if ROI_MODE=="rect" else "",
                        len(ROI_POLY) if ROI_MODE=="poly" else 0,
                        f"{lap_var:.2f}", f"{bright:.2f}", str(phash) if phash is not None else ""
                    ])
                    meta_file.flush()
                    print(f"[SAVE] {out_path} | lap={lap_var:.1f} bright={bright:.1f}")
                    last_hash = phash if phash is not None else last_hash
                    last_save_wall = now
                    next_due_t = last_save_wall + INTERVAL_SEC
                else:
                    # gagal save: tetap majukan jadwal agar tidak macet
                    if UPDATE_TIMER_ON_SKIP:
                        next_due_t = now + INTERVAL_SEC
            else:
                # skip
                if UPDATE_TIMER_ON_SKIP:
                    next_due_t = now + INTERVAL_SEC
                # jika tidak, kita akan mencoba lagi pada frame berikutnya (hingga lolos filter atau force_due)

            # handle keys (interactive ROI) setelah attempt
            if DISPLAY_PREVIEW and ALLOW_INTERACTIVE_ROI:
                # Gunakan waitKey yang lebih responsif untuk input mouse
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('c') or key == ord('C'):
                    ROI_MODE = "none"; ROI_POLY = []; polygon_state.cancel_drawing(); print("[ROI] Cleared")
                elif key == ord('l') or key == ord('L'):
                    loaded = load_roi_config(ROI_CONFIG_FILE)
                    if loaded is not None:
                        ROI_MODE, ROI_RECT, ROI_POLY, ROI_APPLY = loaded
                elif key == ord('s') or key == ord('S'):
                    save_roi_config(ROI_CONFIG_FILE, ROI_MODE, ROI_RECT, ROI_POLY, ROI_APPLY)
                elif key == ord('r') or key == ord('R'):
                    sel = cv2.selectROI("Preview", frame, fromCenter=False, showCrosshair=True)
                    x,y,w,h = sel
                    if w > 0 and h > 0:
                        ROI_MODE = "rect"
                        ROI_RECT = (int(x), int(y), int(w), int(h))
                        ROI_POLY = []
                        polygon_state.cancel_drawing()
                        print(f"[ROI] Set RECT: {ROI_RECT}")
                    try: cv2.destroyWindow("ROI selector")
                    except Exception: pass
                elif key == ord('p') or key == ord('P'):
                    if not polygon_state.is_drawing():
                        ROI_MODE = "poly"
                        ROI_POLY = []
                        polygon_state.start_drawing()
                        print("[ROI] Polygon: LEFT/RIGHT click titik-titik, ENTER selesai, ESC batal, BACKSPACE undo")
                elif key in (8, 255):  # BACKSPACE
                    polygon_state.remove_last_point()
                elif key in (13, 10):  # ENTER
                    if polygon_state.is_drawing():
                        result = polygon_state.finish_drawing()
                        if result is not None:
                            ROI_POLY = result
                elif key == 27:  # ESC
                    if polygon_state.is_drawing():
                        polygon_state.cancel_drawing()

    finally:
        stream.stop()
        meta_file.close()
        if DISPLAY_PREVIEW:
            cv2.destroyAllWindows()
        print("[INFO] Stopped.")

if __name__ == "__main__":
    main()
