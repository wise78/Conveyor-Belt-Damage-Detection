# 🏭 Conveyor Belt Damage Detection System

Sistem deteksi kerusakan conveyor belt menggunakan Raspberry Pi dengan Google Coral Edge TPU untuk inferensi real-time. Sistem ini terdiri dari pipeline lengkap mulai dari pengumpulan data, training model YOLO11, hingga deployment di edge device.

## 📋 Overview

Sistem ini dirancang untuk mendeteksi kerusakan pada conveyor belt secara real-time menggunakan:
- **Raspberry Pi** + **Google Coral Edge TPU** untuk inferensi
- **IP Camera** untuk pengumpulan data dan monitoring
- **YOLO11** untuk object detection
- **TensorFlow Lite** untuk optimasi edge deployment

## 🏗️ Arsitektur Sistem

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Collection│    │   Model Training │    │   Edge Deployment│
│   (Raspberry Pi) │    │   (Google Colab) │    │   (Raspberry Pi) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  data_collector │    │ train_yolo11_   │    │ tesrtspcoba.py  │
│      .py        │    │     a100.ipynb  │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Dataset (Images)│    │   best.pt       │    │ Real-time       │
│                 │    │   (YOLO Model)  │    │ Detection       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       │
┌─────────────────┐    ┌─────────────────┐               │
│ polygon_to_box  │    │ convertedgetpu  │               │
│      .py        │    │      .py        │               │
└─────────────────┘    └─────────────────┘               │
         │                       │                       │
         ▼                       ▼                       │
┌─────────────────┐    ┌─────────────────┐               │
│ YOLO Format     │    │ *_edgetpu.tflite│               │
│ Dataset         │    │ (Edge TPU Model)│               │
└─────────────────┘    └─────────────────┘               │
                                                         │
                                                         ▼
                                              ┌─────────────────┐
                                              │ Damage Detection│
                                              │ Results         │
                                              └─────────────────┘
```

## 📁 Struktur Repository

```
conveyor-belt-damage-detection/
├── README.md                           # Dokumentasi utama
├── requirements.txt                    # Dependencies
├── config/                            # Konfigurasi sistem
│   ├── camera_config.json             # Konfigurasi IP camera
│   └── model_config.json              # Konfigurasi model
├── data_collection/                   # Pengumpulan data
│   ├── data_collector.py              # Program pengumpulan data dari IP camera
│   └── polygon_to_box.py              # Konversi polygon ke bounding box
├── training/                          # Training model
│   ├── train_yolo11_a100.ipynb        # Notebook training YOLO11
│   └── convertedgetpu.py              # Konversi model ke Edge TPU
├── deployment/                        # Deployment di Raspberry Pi
│   ├── tesrtspcoba.py                 # Program inferensi real-time
│   └── requirements_rpi.txt           # Dependencies untuk Raspberry Pi
├── utils/                             # Utility functions
│   ├── image_utils.py                 # Fungsi bantuan untuk image processing
│   └── roi_utils.py                   # Fungsi bantuan untuk ROI
└── docs/                              # Dokumentasi tambahan
    ├── setup_guide.md                 # Panduan setup
    └── troubleshooting.md             # Troubleshooting
```

## 🚀 Quick Start

### 1. Setup Environment

#### Untuk Google Colab (Training):
```bash
# Install dependencies untuk training
pip install ultralytics==8.3.185 tensorflow==2.19.0 onnx onnxslim
pip install opencv-python-headless pillow tqdm pyyaml matplotlib
```

#### Untuk Raspberry Pi (Data Collection & Inference):
```bash
# Buat virtual environment Python 3.9
python3.9 -m venv conveyor_env
source conveyor_env/bin/activate

# Install dependencies
pip install -r requirements_rpi.txt
```

### 2. Pengumpulan Data

1. **Setup IP Camera** di `data_collection/data_collector.py`:
   ```python
   RTSP_URL = "rtsp://username:password@IP:554/Streaming/Channels/101"
   ```

2. **Jalankan data collector**:
   ```bash
   cd data_collection
   python data_collector.py
   ```

3. **Konversi polygon ke bounding box** (jika diperlukan):
   ```bash
   python polygon_to_box.py
   ```

### 3. Training Model

1. **Upload dataset** ke Google Colab
2. **Jalankan notebook** `training/train_yolo11_a100.ipynb`
3. **Export model** ke Edge TPU:
   ```bash
   python convertedgetpu.py
   ```

### 4. Deployment

1. **Transfer model** `*_edgetpu.tflite` ke Raspberry Pi
2. **Update konfigurasi** di `deployment/tesrtspcoba.py`
3. **Jalankan inferensi**:
   ```bash
   cd deployment
   python tesrtspcoba.py
   ```

## 📖 Panduan Detail

### 🔧 Data Collection (`data_collector.py`)

Program untuk mengumpulkan data dari IP camera dengan fitur:
- **RTSP streaming** dengan auto-reconnect
- **ROI (Region of Interest)** untuk fokus area conveyor
- **Quality filter** untuk menghindari gambar blur/gelap
- **Anti-duplicate** menggunakan perceptual hashing
- **Interactive ROI** untuk setup area deteksi

**Konfigurasi utama:**
```python
RTSP_URL = "rtsp://username:password@IP:554/Streaming/Channels/101"
SAVE_ROOT = "dataset_snapshots"
INTERVAL_SEC = 2.0
ROI_MODE = "rect"  # "none", "rect", "poly"
```

### 🔄 Polygon to Box Converter (`polygon_to_box.py`)

Konversi dataset dengan label polygon menjadi format bounding box:
- **Auto-detect** struktur dataset YOLO11
- **Validasi data** dan error handling
- **Statistik konversi** detail
- **Visualisasi hasil** konversi

### 🎯 Model Training (`train_yolo11_a100.ipynb`)

Notebook untuk training model YOLO11:
- **Optimized** untuk Google Colab A100
- **Auto-detect** format dataset
- **Hyperparameter tuning**
- **Model evaluation** dan metrics

### ⚡ Edge TPU Conversion (`convertedgetpu.py`)

Konversi model YOLO11 ke format Edge TPU:
- **INT8 quantization** untuk optimasi
- **Tensor shape validation**
- **Compatibility check** dengan Edge TPU

### 🚀 Real-time Inference (`tesrtspcoba.py`)

Program inferensi real-time di Raspberry Pi:
- **Multi-source input** (RTSP, video, image, webcam)
- **ROI filtering** untuk fokus deteksi
- **Performance optimization** untuk Edge TPU
- **Output visualization** dan recording

## ⚙️ Konfigurasi

### Camera Configuration
```json
{
  "rtsp_url": "rtsp://username:password@IP:554/Streaming/Channels/101",
  "use_gstreamer": false,
  "read_width": 1280,
  "read_height": 720
}
```

### Model Configuration
```json
{
  "model_path": "best_edgetpu.tflite",
  "device": "tpu:0",
  "img_size": 256,
  "confidence": 0.25,
  "enable_roi": true
}
```

## 🔧 Troubleshooting

### Common Issues

1. **RTSP Connection Failed**
   - Periksa URL dan credentials
   - Pastikan network connectivity
   - Coba ganti backend (GStreamer/FFMPEG)

2. **Edge TPU Not Detected**
   - Pastikan Google Coral terpasang dengan benar
   - Install Edge TPU runtime
   - Periksa USB connection

3. **Low FPS Performance**
   - Kurangi resolusi input
   - Optimasi ROI area
   - Periksa thermal throttling

4. **Model Conversion Failed**
   - Pastikan model YOLO11 compatible
   - Periksa input tensor shape
   - Update Ultralytics version

## 📊 Performance Metrics

- **Inference Speed**: ~30-50 FPS pada Raspberry Pi 4 + Coral
- **Model Size**: ~5-10 MB (quantized)
- **Accuracy**: 85-95% (tergantung dataset quality)
- **Latency**: <100ms end-to-end

## 🤝 Contributing

1. Fork repository
2. Buat feature branch
3. Commit changes
4. Push ke branch
5. Buat Pull Request

## 📄 License

MIT License - lihat file [LICENSE](LICENSE) untuk detail.

## 📞 Support

Untuk pertanyaan dan dukungan:
- Buat issue di GitHub
- Email: [your-email@domain.com]
- Dokumentasi lengkap: [docs/](docs/)

---

**Note**: Pastikan untuk mengikuti panduan setup dengan teliti dan melakukan testing di environment yang sesuai sebelum deployment ke production.