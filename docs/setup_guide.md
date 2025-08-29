# 📋 Setup Guide - Conveyor Belt Damage Detection System

Panduan lengkap untuk setup dan konfigurasi sistem deteksi kerusakan conveyor belt.

## 🎯 Overview

Sistem ini terdiri dari 3 komponen utama:
1. **Data Collection** - Raspberry Pi + IP Camera
2. **Model Training** - Google Colab
3. **Real-time Inference** - Raspberry Pi + Google Coral

## 🔧 Prerequisites

### Hardware Requirements

#### Untuk Data Collection & Inference:
- **Raspberry Pi 4** (4GB RAM minimum, 8GB recommended)
- **Google Coral USB Accelerator** atau **Coral Dev Board**
- **IP Camera** (Hikvision, Dahua, atau kompatibel RTSP)
- **MicroSD Card** (32GB minimum, Class 10)
- **Power Supply** (5V/3A untuk Pi 4)
- **Ethernet Cable** atau **WiFi** untuk koneksi network

#### Untuk Training (Opsional):
- **Google Colab Pro** (untuk akses GPU A100)
- **Dataset** dengan label kerusakan conveyor belt

### Software Requirements

#### Raspberry Pi:
- **Raspberry Pi OS** (Bullseye atau Bookworm)
- **Python 3.9** (wajib)
- **OpenCV** dengan GStreamer support
- **Edge TPU Runtime**

#### Google Colab:
- **Google Account**
- **Colab Pro** (untuk akses GPU A100)

## 🚀 Installation Steps

### Step 1: Setup Raspberry Pi

#### 1.1 Install Raspberry Pi OS
```bash
# Download Raspberry Pi Imager
# Flash Raspberry Pi OS Bullseye ke microSD
# Enable SSH dan WiFi saat setup
```

#### 1.2 Update System
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.9 python3.9-venv python3.9-dev
sudo apt install -y build-essential cmake pkg-config
sudo apt install -y libjpeg-dev libpng-dev libtiff-dev
sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt install -y libxvidcore-dev libx264-dev
sudo apt install -y libgtk-3-dev
sudo apt install -y libatlas-base-dev gfortran
sudo apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
```

#### 1.3 Install Edge TPU Runtime
```bash
# Download Edge TPU runtime
wget https://packages.cloud.google.com/apt/pool/edgetpu-stable/edgetpu_2.0.0_all.deb
sudo dpkg -i edgetpu_2.0.0_all.deb

# Install Python runtime
pip3 install pycoral
```

#### 1.4 Setup Python Virtual Environment
```bash
# Buat virtual environment
python3.9 -m venv ~/conveyor_env
source ~/conveyor_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r deployment/requirements_rpi.txt
```

### Step 2: Setup IP Camera

#### 2.1 Konfigurasi IP Camera
1. **Connect IP Camera** ke network yang sama dengan Raspberry Pi
2. **Note IP Address** camera
3. **Set username/password** untuk akses RTSP
4. **Enable RTSP streaming** di settings camera

#### 2.2 Test RTSP Connection
```bash
# Test dengan VLC atau ffmpeg
ffmpeg -i "rtsp://username:password@IP:554/Streaming/Channels/101" -t 10 test.mp4
```

### Step 3: Konfigurasi Sistem

#### 3.1 Update Camera Configuration
```bash
# Edit config/camera_config.json
nano config/camera_config.json
```

Update parameter berikut:
```json
{
  "rtsp_url": "rtsp://username:password@IP:554/Streaming/Channels/101",
  "use_gstreamer": true,
  "read_width": 1280,
  "read_height": 720
}
```

#### 3.2 Test Data Collection
```bash
cd data_collection
python data_collector.py
```

**Expected Output:**
- Preview window dengan stream dari IP camera
- Gambar tersimpan di folder `dataset_snapshots`
- ROI dapat diatur dengan interaktif

### Step 4: Model Training (Google Colab)

#### 4.1 Upload Dataset
1. **Buka Google Colab**
2. **Upload dataset** dalam format ZIP
3. **Upload notebook** `training/train_yolo11_a100.ipynb`

#### 4.2 Run Training
```python
# Install dependencies
!pip install ultralytics==8.3.185 tensorflow==2.19.0 onnx onnxslim

# Run training notebook
# Follow instructions di notebook
```

#### 4.3 Export Model
```python
# Run convertedgetpu.py
# Download *_edgetpu.tflite file
```

### Step 5: Deployment

#### 5.1 Transfer Model
```bash
# Copy model ke Raspberry Pi
scp best_edgetpu.tflite pi@raspberrypi_ip:/home/pi/conveyor-belt-damage-detection/
```

#### 5.2 Update Model Configuration
```bash
# Edit config/model_config.json
nano config/model_config.json
```

Update parameter:
```json
{
  "model_path": "best_edgetpu.tflite",
  "device": "tpu:0",
  "img_size": 256,
  "confidence": 0.25
}
```

#### 5.3 Test Inference
```bash
cd deployment
python tesrtspcoba.py
```

**Expected Output:**
- Real-time detection dari IP camera
- Bounding box untuk kerusakan terdeteksi
- FPS display dan performance metrics

## ⚙️ Configuration Details

### Camera Configuration (`config/camera_config.json`)

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `rtsp_url` | RTSP stream URL | - | Sesuai IP camera |
| `use_gstreamer` | Use GStreamer backend | false | true (Linux/RPi) |
| `read_width` | Input width | 1280 | 1280 |
| `read_height` | Input height | 720 | 720 |
| `interval_sec` | Save interval | 2.0 | 1.0-5.0 |
| `min_lap_var` | Min sharpness | 60.0 | 50.0-80.0 |
| `min_bright` | Min brightness | 25.0 | 20.0-40.0 |
| `max_bright` | Max brightness | 235.0 | 220.0-250.0 |

### Model Configuration (`config/model_config.json`)

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `model_path` | Path to Edge TPU model | - | best_edgetpu.tflite |
| `device` | Inference device | tpu:0 | tpu:0 |
| `img_size` | Input image size | 256 | 256/320/480 |
| `confidence` | Detection threshold | 0.25 | 0.2-0.5 |
| `enable_roi` | Enable ROI filtering | true | true |

## 🔍 Troubleshooting

### Common Issues

#### 1. RTSP Connection Failed
```bash
# Check network connectivity
ping IP_CAMERA_IP

# Test RTSP with ffmpeg
ffmpeg -i "rtsp://..." -t 5 test.mp4

# Check firewall settings
sudo ufw status
```

#### 2. Edge TPU Not Detected
```bash
# Check USB connection
lsusb | grep Google

# Check Edge TPU runtime
python3 -c "import tflite_runtime.interpreter as tflite; print('OK')"

# Reinstall Edge TPU runtime
sudo apt remove edgetpu
sudo apt install edgetpu
```

#### 3. Low FPS Performance
```bash
# Check CPU temperature
vcgencmd measure_temp

# Check memory usage
free -h

# Optimize ROI area
# Reduce input resolution
# Use smaller model
```

#### 4. Model Conversion Failed
```bash
# Check Ultralytics version
pip show ultralytics

# Update to compatible version
pip install ultralytics==8.3.185

# Check model format
file best.pt
```

### Performance Optimization

#### 1. Raspberry Pi Optimization
```bash
# Enable GPU memory split
sudo raspi-config
# Advanced Options > Memory Split > 128

# Overclock (optional)
sudo raspi-config
# Performance Options > Overclock > Pi 4 2GHz

# Disable unnecessary services
sudo systemctl disable bluetooth
sudo systemctl disable avahi-daemon
```

#### 2. Network Optimization
```bash
# Use Ethernet instead of WiFi
# Set static IP for Raspberry Pi
# Optimize RTSP settings di IP camera
```

#### 3. Model Optimization
```bash
# Use smaller input size (256 instead of 480)
# Reduce confidence threshold
# Optimize ROI area
# Use quantized model (INT8)
```

## 📊 Performance Benchmarks

### Raspberry Pi 4 + Coral USB

| Model Size | Input Size | FPS | Memory Usage | Accuracy |
|------------|------------|-----|--------------|----------|
| YOLO11n | 256x256 | 35-45 | ~800MB | 85-90% |
| YOLO11s | 320x320 | 25-35 | ~1GB | 88-93% |
| YOLO11m | 480x480 | 15-25 | ~1.5GB | 90-95% |

### Optimization Tips

1. **Use YOLO11n** untuk real-time performance
2. **Input size 256x256** untuk balance speed/accuracy
3. **ROI filtering** untuk fokus area conveyor
4. **Ethernet connection** untuk stable RTSP
5. **Adequate cooling** untuk prevent throttling

## 🔄 Maintenance

### Regular Tasks

#### Daily:
- Check system logs: `journalctl -u conveyor-detection`
- Monitor disk space: `df -h`
- Check temperature: `vcgencmd measure_temp`

#### Weekly:
- Update system: `sudo apt update && sudo apt upgrade`
- Clean old logs: `sudo journalctl --vacuum-time=7d`
- Backup configurations

#### Monthly:
- Test full pipeline
- Update model jika diperlukan
- Performance review

### Monitoring Scripts

```bash
#!/bin/bash
# monitor.sh - System monitoring script

echo "=== System Status ==="
echo "CPU Temp: $(vcgencmd measure_temp)"
echo "Memory: $(free -h | grep Mem)"
echo "Disk: $(df -h / | tail -1)"
echo "Network: $(ping -c 1 8.8.8.8 >/dev/null && echo 'OK' || echo 'FAIL')"
```

## 📞 Support

### Getting Help

1. **Check logs**: `tail -f /var/log/syslog`
2. **Test components** individually
3. **Search issues** di GitHub repository
4. **Create issue** dengan detail error

### Useful Commands

```bash
# Check Edge TPU status
lsusb | grep Google

# Test camera connection
ffmpeg -i "rtsp://..." -t 5 test.mp4

# Monitor system resources
htop

# Check Python environment
which python
python --version
pip list

# Test model inference
python -c "import tflite_runtime; print('OK')"
```

---

**Note**: Pastikan untuk melakukan testing di environment yang aman sebelum deployment ke production. Backup semua konfigurasi dan data penting.