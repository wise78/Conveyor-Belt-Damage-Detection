# 🔧 Troubleshooting Guide - Conveyor Belt Damage Detection System

Panduan lengkap untuk mengatasi masalah umum dalam sistem deteksi kerusakan conveyor belt.

## 🚨 Quick Diagnosis

### System Health Check
```bash
#!/bin/bash
# health_check.sh - Quick system health check

echo "=== System Health Check ==="
echo "1. CPU Temperature: $(vcgencmd measure_temp)"
echo "2. Memory Usage: $(free -h | grep Mem | awk '{print $3"/"$2}')"
echo "3. Disk Usage: $(df -h / | tail -1 | awk '{print $5}')"
echo "4. Network: $(ping -c 1 8.8.8.8 >/dev/null && echo 'OK' || echo 'FAIL')"
echo "5. Edge TPU: $(lsusb | grep Google >/dev/null && echo 'OK' || echo 'NOT FOUND')"
echo "6. Python: $(python3 --version 2>/dev/null || echo 'NOT FOUND')"
echo "7. OpenCV: $(python3 -c 'import cv2; print(cv2.__version__)' 2>/dev/null || echo 'NOT FOUND')"
```

## 🔍 Common Issues & Solutions

### 1. RTSP Connection Issues

#### Problem: Cannot connect to IP camera
```
Error: Failed to connect to RTSP stream
```

#### Solutions:

**A. Check Network Connectivity**
```bash
# Test basic connectivity
ping IP_CAMERA_IP

# Check if port 554 is open
telnet IP_CAMERA_IP 554

# Test with curl
curl -I rtsp://username:password@IP:554/Streaming/Channels/101
```

**B. Verify RTSP URL Format**
```bash
# Common RTSP URL formats:
# Hikvision: rtsp://username:password@IP:554/Streaming/Channels/101
# Dahua: rtsp://username:password@IP:554/cam/realmonitor?channel=1&subtype=0
# Generic: rtsp://username:password@IP:554/stream1
```

**C. Test with FFmpeg**
```bash
# Test RTSP connection
ffmpeg -i "rtsp://username:password@IP:554/Streaming/Channels/101" -t 10 test.mp4

# Check stream info
ffprobe "rtsp://username:password@IP:554/Streaming/Channels/101"
```

**D. Firewall Issues**
```bash
# Check firewall status
sudo ufw status

# Allow RTSP traffic
sudo ufw allow 554/tcp
sudo ufw allow 554/udp

# Check iptables
sudo iptables -L
```

#### Problem: High latency or dropped frames
```
Warning: High latency detected
```

#### Solutions:

**A. Optimize Network Settings**
```bash
# Set network interface to performance mode
sudo ethtool -s eth0 speed 1000 duplex full autoneg off

# Optimize TCP settings
echo 'net.core.rmem_max = 16777216' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max = 16777216' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

**B. Use GStreamer Backend**
```python
# In camera_config.json
{
  "use_gstreamer": true,
  "read_width": 1280,
  "read_height": 720
}
```

**C. Reduce Stream Quality**
```python
# Lower resolution for better performance
{
  "read_width": 640,
  "read_height": 480
}
```

### 2. Edge TPU Issues

#### Problem: Edge TPU not detected
```
Error: No Edge TPU device found
```

#### Solutions:

**A. Check Hardware Connection**
```bash
# Check USB devices
lsusb | grep Google

# Expected output:
# Bus 001 Device 004: ID 18d1:9302 Google Inc. Edge TPU
```

**B. Check Edge TPU Runtime**
```bash
# Check if runtime is installed
dpkg -l | grep edgetpu

# Reinstall if needed
sudo apt remove edgetpu
sudo apt install edgetpu
```

**C. Check Python Runtime**
```bash
# Test Python runtime
python3 -c "import tflite_runtime.interpreter as tflite; print('OK')"

# Install if missing
pip3 install pycoral
```

**D. USB Power Issues**
```bash
# Check USB power
lsusb -t

# Use powered USB hub if needed
# Ensure adequate power supply (5V/3A)
```

#### Problem: Model loading failed
```
Error: Failed to load Edge TPU model
```

#### Solutions:

**A. Check Model Format**
```bash
# Verify model file
file best_edgetpu.tflite

# Expected: data
# Should contain: "TensorFlow Lite model"
```

**B. Check Model Compatibility**
```python
# Test model loading
import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(
    model_path="best_edgetpu.tflite",
    experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
)
```

**C. Reconvert Model**
```python
# Use convertedgetpu.py to reconvert
# Ensure INT8 quantization
# Check input tensor shape
```

### 3. Performance Issues

#### Problem: Low FPS
```
Warning: FPS below threshold (current: 15, target: 30)
```

#### Solutions:

**A. Check System Resources**
```bash
# Monitor CPU usage
htop

# Check temperature
vcgencmd measure_temp

# Check memory
free -h
```

**B. Optimize Model Settings**
```python
# Reduce input size
{
  "img_size": 256  # Instead of 480
}

# Lower confidence threshold
{
  "confidence": 0.2  # Instead of 0.5
}
```

**C. Enable ROI Filtering**
```python
# Focus detection on conveyor area
{
  "enable_roi": true,
  "roi_mode": "manual",
  "roi_coords": [100, 100, 500, 400]
}
```

**D. System Optimization**
```bash
# Disable unnecessary services
sudo systemctl disable bluetooth
sudo systemctl disable avahi-daemon
sudo systemctl disable triggerhappy

# Increase GPU memory
sudo raspi-config
# Advanced Options > Memory Split > 128
```

#### Problem: High memory usage
```
Warning: Memory usage high (current: 85%, threshold: 80%)
```

#### Solutions:

**A. Monitor Memory Usage**
```bash
# Check memory usage
free -h

# Check swap usage
swapon --show

# Check memory by process
ps aux --sort=-%mem | head -10
```

**B. Optimize Application**
```python
# Reduce batch size
# Clear unused variables
# Use garbage collection
import gc
gc.collect()
```

**C. Add Swap Space**
```bash
# Create swap file
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 4. Data Collection Issues

#### Problem: No images saved
```
Warning: No images saved in last 60 seconds
```

#### Solutions:

**A. Check Save Directory**
```bash
# Check directory permissions
ls -la dataset_snapshots/

# Create directory if missing
mkdir -p dataset_snapshots
chmod 755 dataset_snapshots
```

**B. Check Quality Filters**
```python
# Adjust quality thresholds
{
  "min_lap_var": 30.0,  # Lower sharpness requirement
  "min_bright": 15.0,   # Lower brightness requirement
  "max_bright": 240.0   # Higher brightness tolerance
}
```

**C. Check Anti-duplicate Settings**
```python
# Disable anti-duplicate temporarily
{
  "anti_duplicate": false
}

# Or adjust threshold
{
  "min_hamming_dist": 3  # Less strict
}
```

#### Problem: Poor image quality
```
Warning: Image quality below threshold
```

#### Solutions:

**A. Check Camera Settings**
```bash
# Test camera with different settings
ffmpeg -i "rtsp://..." -vf "scale=1280:720" -q:v 2 test.mp4
```

**B. Adjust Quality Parameters**
```python
# Optimize for your environment
{
  "jpeg_quality": 90,
  "min_lap_var": 40.0,
  "min_bright": 20.0,
  "max_bright": 230.0
}
```

**C. Improve Lighting**
- Add additional lighting
- Adjust camera exposure
- Clean camera lens

### 5. Training Issues

#### Problem: Model conversion failed
```
Error: Failed to convert model to Edge TPU format
```

#### Solutions:

**A. Check Ultralytics Version**
```python
# Use compatible version
pip install ultralytics==8.3.185
```

**B. Check Model Format**
```python
# Verify model is YOLO11
from ultralytics import YOLO
model = YOLO('best.pt')
print(model.info())
```

**C. Check Tensor Shape**
```python
# Ensure compatible input shape
# YOLO11 typically uses multiples of 32
# Common sizes: 256, 320, 480, 640
```

#### Problem: Poor detection accuracy
```
Warning: Detection accuracy below threshold
```

#### Solutions:

**A. Improve Dataset**
- Add more diverse samples
- Balance class distribution
- Improve annotation quality
- Augment training data

**B. Adjust Training Parameters**
```python
# Optimize hyperparameters
{
  "epochs": 100,
  "batch_size": 16,
  "learning_rate": 0.01,
  "patience": 20
}
```

**C. Use Transfer Learning**
```python
# Start with pre-trained model
model = YOLO('yolo11n.pt')  # Pre-trained
model.train(data='dataset.yaml', epochs=50)
```

## 🔧 Advanced Troubleshooting

### Debug Mode

#### Enable Debug Logging
```python
# Add to configuration
{
  "debug": true,
  "log_level": "DEBUG"
}
```

#### Check System Logs
```bash
# Check system logs
sudo journalctl -u conveyor-detection -f

# Check application logs
tail -f /var/log/conveyor-detection.log
```

### Performance Profiling

#### CPU Profiling
```bash
# Install profiling tools
sudo apt install python3-cProfile

# Profile application
python3 -m cProfile -o profile.stats tesrtspcoba.py

# Analyze results
python3 -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(10)
"
```

#### Memory Profiling
```bash
# Install memory profiler
pip install memory-profiler

# Profile memory usage
python3 -m memory_profiler tesrtspcoba.py
```

### Network Diagnostics

#### RTSP Stream Analysis
```bash
# Analyze RTSP stream
ffprobe -v quiet -print_format json -show_format -show_streams "rtsp://..."

# Check stream statistics
ffmpeg -i "rtsp://..." -f null - 2>&1 | grep "frame="
```

#### Network Performance
```bash
# Test network bandwidth
iperf3 -c IP_CAMERA_IP

# Check packet loss
ping -c 100 IP_CAMERA_IP | grep "packet loss"
```

## 📊 Monitoring & Alerts

### System Monitoring Script
```bash
#!/bin/bash
# monitor_system.sh

# Check system health
TEMP=$(vcgencmd measure_temp | cut -d'=' -f2 | cut -d"'" -f1)
MEMORY=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
DISK=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')

# Alert thresholds
TEMP_THRESHOLD=70
MEMORY_THRESHOLD=85
DISK_THRESHOLD=90

# Generate alerts
if (( $(echo "$TEMP > $TEMP_THRESHOLD" | bc -l) )); then
    echo "ALERT: High temperature: ${TEMP}°C"
fi

if (( $(echo "$MEMORY > $MEMORY_THRESHOLD" | bc -l) )); then
    echo "ALERT: High memory usage: ${MEMORY}%"
fi

if [ "$DISK" -gt "$DISK_THRESHOLD" ]; then
    echo "ALERT: High disk usage: ${DISK}%"
fi
```

### Automated Recovery
```bash
#!/bin/bash
# auto_recovery.sh

# Restart service if needed
if ! pgrep -f "tesrtspcoba.py" > /dev/null; then
    echo "Restarting detection service..."
    cd /home/pi/conveyor-belt-damage-detection/deployment
    python3 tesrtspcoba.py &
fi

# Clear old logs
find /var/log -name "*.log" -mtime +7 -delete

# Restart if temperature too high
TEMP=$(vcgencmd measure_temp | cut -d'=' -f2 | cut -d"'" -f1)
if (( $(echo "$TEMP > 80" | bc -l) )); then
    echo "Critical temperature, restarting system..."
    sudo reboot
fi
```

## 📞 Getting Help

### Before Asking for Help

1. **Collect Information**
   ```bash
   # System info
   uname -a
   cat /etc/os-release
   
   # Hardware info
   vcgencmd get_mem gpu
   lsusb
   
   # Software versions
   python3 --version
   pip list | grep -E "(ultralytics|tensorflow|opencv)"
   ```

2. **Check Logs**
   ```bash
   # Recent logs
   tail -100 /var/log/syslog
   
   # Application logs
   tail -100 /var/log/conveyor-detection.log
   ```

3. **Test Components**
   ```bash
   # Test camera
   ffmpeg -i "rtsp://..." -t 5 test.mp4
   
   # Test Edge TPU
   python3 -c "import tflite_runtime; print('OK')"
   
   # Test model
   python3 -c "import tflite_runtime.interpreter as tflite; tflite.Interpreter('best_edgetpu.tflite')"
   ```

### Contact Information

- **GitHub Issues**: Create issue with detailed error information
- **Email Support**: [your-email@domain.com]
- **Documentation**: Check [docs/](docs/) for additional guides

### Issue Template

When reporting issues, include:

1. **System Information**
   - Raspberry Pi model and OS version
   - Python version
   - Installed packages

2. **Error Details**
   - Full error message
   - Steps to reproduce
   - Expected vs actual behavior

3. **Logs**
   - System logs
   - Application logs
   - Debug output

4. **Configuration**
   - Camera configuration
   - Model configuration
   - Environment variables

---

**Note**: Always backup your configuration and data before making changes. Test solutions in a safe environment first.