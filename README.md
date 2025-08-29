# Conveyor Belt Damage Detection (Raspberry Pi + Google Coral)

Sistem end-to-end untuk mendeteksi kerusakan pada conveyor belt, meliputi pengumpulan data dari site (RTSP/IP Camera), persiapan dataset, pelatihan model YOLO11 (di Google Colab, GPU A100), konversi model ke EdgeTPU TFLite INT8, dan deployment inferensi real-time di Raspberry Pi + Google Coral.

## Fitur Utama
- Pengumpulan data RTSP threaded rendah-latensi dengan filter kualitas (blur/brightness) dan anti-duplikasi (pHash) + ROI interaktif (rect/polygon). (`raspi/data_collector.py`)
- Konversi dataset YOLO11 polygon (segmentation) → bounding box (object detection). (`colab/polygon_to_box.py`)
- Training YOLO11 berbasis Ultralytics untuk object detection. (`colab/Train_Yolo11_A100.ipynb`)
- Export model `.pt` → EdgeTPU TFLite INT8 (`*_edgetpu.tflite`) siap Coral. (`colab/convertedgetpu.py`)
- Inferensi real-time di Raspberry Pi + Coral pada sumber RTSP/Video/Image/Webcam dengan dukungan ROI. (`raspi/tesrtspcoba.py`)

## Arsitektur & Alur
1) Site: Raspberry Pi + IP Camera → kumpulkan snapshot berkualitas menggunakan ROI → `datasets/`  
2) Labeling: Buat label YOLO11 (bbox). Jika dataset awalnya polygon (segmentation), konversi ke bbox.  
3) Training (Colab): Latih YOLO11 dengan dataset format YOLO11 → hasil `best.pt`.  
4) Konversi (Colab): `best.pt` → `*_edgetpu.tflite` (INT8) untuk Coral.  
5) Deploy (Raspberry Pi): Jalankan inferensi dengan EdgeTPU di `raspi/tesrtspcoba.py`.

## Struktur Repository
```
.
├─ colab/
│  ├─ Train_Yolo11_A100.ipynb      # Notebook training YOLO11 di Colab (GPU A100)
│  ├─ convertedgetpu.py            # Script Colab: export best.pt → *_edgetpu.tflite
│  └─ polygon_to_box.py            # Script Colab: konversi polygon → bbox (YOLO11)
├─ raspi/
│  ├─ data_collector.py            # Kolektor dataset dari RTSP/IP Cam dengan ROI & filter kualitas
│  ├─ tesrtspcoba.py               # Inferensi di Raspberry Pi + Coral (RTSP/Video/Image/Webcam)
│  └─ requirements.txt             # Dependensi Python untuk lingkungan Raspberry Pi
├─ datasets/                        # (kosong) tempat dataset Anda
│  └─ .gitkeep
├─ models/                          # taruh model *.pt / *_edgetpu.tflite
│  └─ .gitkeep
├─ outputs/                         # keluaran video/gambar hasil inferensi
│  └─ .gitkeep
├─ LICENSE
└─ README.md
```

## Persyaratan
- Google Colab (untuk training & konversi)
- Raspberry Pi 4/5 (64-bit OS direkomendasikan) + Google Coral USB Accelerator
- Python 3.9 (untuk skrip di Raspberry Pi)
- Kamera RTSP/IP atau sumber video/gambar/webcam

> Catatan Coral: pastikan EdgeTPU runtime terpasang (libedgetpu). Lihat bagian Setup Raspberry Pi.

---

## Setup di Raspberry Pi (Deployment + Data Collection)
1) Siapkan Python 3.9 dan virtual environment:
```bash
sudo apt update
# Jika perlu Python 3.9 dan venv (opsional; sesuaikan OS Anda)
sudo apt install -y python3.9 python3.9-venv
cd /path/to/your/repo
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

2) Install dependensi Python:
```bash
pip install -r raspi/requirements.txt
```

3) Install EdgeTPU runtime (wajib agar Coral berfungsi):
```bash
# Standar (CPU + TPU bersama)
sudo apt install -y libedgetpu1-std
# atau (opsional, performa lebih tinggi): sudo apt install -y libedgetpu1-max
```

4) (Opsional) RTSP via GStreamer:
```bash
sudo apt install -y gstreamer1.0-tools gstreamer1.0-libav gstreamer1.0-plugins-bad
```

> Jika `tflite-runtime`/`pycoral` via pip belum tersedia untuk OS/arch Anda, ikuti petunjuk resmi TensorFlow Lite Python (`https://www.tensorflow.org/lite/guide/python`) dan Coral (`https://coral.ai/software/#debian-packages`).

---

## Pengumpulan Data dari Site (raspi/data_collector.py)
- Ubah konfigurasi di awal file sesuai kebutuhan (RTSP URL, interval simpan, ROI mode, filter kualitas, dll.).
- Jalankan:
```bash
source .venv/bin/activate
python raspi/data_collector.py
```

### Fitur & Output
- Stream RTSP threaded (auto-reconnect).  
- Filter kualitas pada ROI: minimal ketajaman (variance of Laplacian) dan batas brightness.  
- Anti-duplikasi berbasis pHash pada ROI.  
- ROI: rectangle atau polygon, mode apply: crop atau mask.  
- Simpan ke folder harian `dataset_snapshots/YYYY-MM-DD/` + `metadata.csv`.

### Kontrol Interaktif (window Preview)
- R: pilih ROI rectangle (OpenCV ROI selector)
- P: gambar ROI polygon (klik untuk titik; BACKSPACE undo; Enter simpan)
- C: clear ROI
- S: simpan ROI ke `roi_config.json`
- L: load ROI dari `roi_config.json`
- Q: keluar

> Headless: set `DISPLAY_PREVIEW=False` bila tanpa display/SSH.

---

## Menyiapkan Dataset YOLO11
- Label gambar hasil koleksi menjadi format YOLO11 (bbox). Anda bisa memakai labeler apa pun (Ultralytics Label Studio, Roboflow, dll.).
- Jika Anda sudah terlanjur memiliki dataset YOLO11 dengan label polygon (segmentation), gunakan konversi ke bbox:

### Konversi Polygon → BBox (colab/polygon_to_box.py)
1) Buka `colab/polygon_to_box.py` di Google Colab (unggah file atau clone repo di Colab).  
2) Jalankan sel instalasi dan ikuti instruksi upload dataset ZIP.  
3) Jalankan konversi; hasil dataset baru dengan bbox akan tersedia untuk diunduh (sufiks `_bbox`).

> Script ini melakukan validasi, statistik, dan visualisasi contoh konversi.

---

## Training YOLO11 di Colab (colab/Train_Yolo11_A100.ipynb)
1) Buka notebook di Colab dengan GPU (A100 direkomendasikan).  
2) Ikuti sel-sel instalasi dan setup.  
3) Pastikan `data.yaml` menunjuk ke direktori dataset YOLO11 (images/labels dengan split train/val/test).  
4) Jalankan training sampai selesai. Output `best.pt` akan tersimpan di folder `runs/.../weights/best.pt`.

> Sesuaikan parameter `imgsz`, `epochs`, `batch`, `names/nc`, dsb., sesuai dataset Anda.

---

## Konversi ke EdgeTPU TFLite INT8 (colab/convertedgetpu.py)
1) Buka `colab/convertedgetpu.py` di Colab.  
2) Jalankan skrip; Anda akan diminta upload ZIP dataset (untuk referensi `names`) dan `best.pt`.  
3) Skrip akan membuat `data.yaml` otomatis bila perlu, lalu export ke `*_edgetpu.tflite` (INT8).  
4) Unduh file `*_edgetpu.tflite` yang dihasilkan dan pindahkan ke folder `models/` pada repo lokal Anda.

> Pastikan `imgsz` saat export selaras dengan saat training (mis. 256/320/480). Ubah nilai `imgsz` di script bila diperlukan.

---

## Deploy & Inferensi di Raspberry Pi + Coral (raspi/tesrtspcoba.py)
1) Salin model `*_edgetpu.tflite` ke `models/`.  
2) Edit konfigurasi di `raspi/tesrtspcoba.py`:
   - `MODEL_PATH` → arahkan ke file di `models/your_model_edgetpu.tflite`
   - `DEVICE = "tpu:0"` untuk Coral, gunakan `"cpu"` untuk debug CPU
   - Pilih sumber: `SOURCE_TYPE = "rtsp" | "video" | "image" | "webcam"`
   - Sesuaikan `RTSP_URL` atau `VIDEO_PATH`/`IMAGE_PATH`/`WEBCAM_INDEX`
   - ROI: `ENABLE_ROI`, `ROI_MODE` (manual/auto/interactive), `ROI_SHAPE`
   - `IMGSZ` dan `CONF` sesuai model

3) Jalankan:
```bash
source .venv/bin/activate
python raspi/tesrtspcoba.py
```

4) Kontrol saat berjalan:
- Tekan `q` untuk keluar.
- Jika `ROI_MODE="interactive"`, akan muncul window setup ROI (rectangle/circle/polygon) sebelum inferensi.

5) Output:  
- Tampilkan jendela visualisasi (bisa dimatikan dengan `DISPLAY=False`).  
- Simpan hasil ke `outputs/` bila `SAVE_OUTPUT=True`.

---

## Tips & Troubleshooting
- Coral tidak terdeteksi / error delegate: pastikan `libedgetpu1-std` terpasang dan Anda memakai `DEVICE="tpu:0"`.
- `tflite-runtime`/`pycoral` gagal di-install: gunakan wheel yang sesuai arsitektur/OS; rujuk dokumentasi resmi TensorFlow Lite & Coral.
- RTSP macet/lag: coba `USE_GSTREAMER=True`, kurangi resolusi/kualitas feed, atau gunakan FFMPEG (default). Pastikan jaringan stabil.
- Window GUI gagal di server/headless: set `DISPLAY=False` (inference) atau `DISPLAY_PREVIEW=False` (collector).
- Ukuran input mismatch: samakan `IMGSZ` saat inferensi dengan saat export/training.
- CPU debug: set `DEVICE="cpu"` untuk melihat apakah masalah di Coral/delegate.

---

## Lisensi
Lihat file `LICENSE` untuk detail lisensi.

## Kredit
- Ultralytics YOLO11
- Google Coral / EdgeTPU
- OpenCV

Jika Anda menemukan isu/bug atau membutuhkan bantuan, silakan buka issue atau hubungi maintainer repo ini.
