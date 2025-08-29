# 🤝 Contributing to Conveyor Belt Damage Detection System

Terima kasih atas minat Anda untuk berkontribusi pada sistem deteksi kerusakan conveyor belt! Panduan ini akan membantu Anda memahami cara berkontribusi dengan efektif.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)

## 📜 Code of Conduct

### Our Pledge

Kami sebagai kontributor dan maintainer berjanji untuk membuat pengalaman kontribusi yang bebas dari pelecehan untuk semua orang, terlepas dari usia, ukuran tubuh, disabilitas, etnis, karakteristik seksual, identitas dan ekspresi gender, tingkat pengalaman, pendidikan, status sosial-ekonomi, kebangsaan, penampilan pribadi, ras, agama, atau identitas dan orientasi seksual.

### Our Standards

Contoh perilaku yang berkontribusi pada lingkungan yang positif:

- Menggunakan bahasa yang ramah dan inklusif
- Menghormati perspektif dan pengalaman yang berbeda
- Menerima kritik konstruktif dengan baik
- Fokus pada apa yang terbaik untuk komunitas
- Menunjukkan empati terhadap anggota komunitas lainnya

Contoh perilaku yang tidak dapat diterima:

- Penggunaan bahasa atau gambar yang seksual atau kekerasan
- Trolling, komentar yang menghina/menghujat, dan serangan pribadi atau politik
- Pelecehan publik atau pribadi
- Mempublikasikan informasi pribadi orang lain tanpa izin eksplisit
- Perilaku lain yang dapat dianggap tidak pantas dalam pengaturan profesional

## 🚀 How Can I Contribute?

### Reporting Bugs

Sebelum membuat laporan bug, periksa daftar masalah yang sudah ada untuk memastikan masalah belum dilaporkan.

Saat membuat laporan bug, sertakan informasi berikut:

1. **Judul yang jelas dan deskriptif**
2. **Deskripsi langkah demi langkah** tentang bagaimana masalah dapat direproduksi
3. **Perilaku yang diharapkan** dan apa yang sebenarnya terjadi
4. **Screenshot** jika relevan
5. **Informasi sistem**:
   - Raspberry Pi model dan OS version
   - Python version
   - Daftar package yang terinstall
   - Konfigurasi camera dan model

### Suggesting Enhancements

Jika Anda memiliki saran untuk peningkatan, buat issue dengan label "enhancement" dan sertakan:

1. **Deskripsi fitur** yang diinginkan
2. **Use case** yang menjelaskan mengapa fitur ini berguna
3. **Implementasi yang diusulkan** (jika ada)
4. **Alternatif** yang telah dipertimbangkan

### Pull Requests

1. Fork repository
2. Buat branch untuk fitur Anda (`git checkout -b feature/amazing-feature`)
3. Commit perubahan Anda (`git commit -m 'Add some amazing feature'`)
4. Push ke branch (`git push origin feature/amazing-feature`)
5. Buat Pull Request

## 🔧 Development Setup

### Prerequisites

- Python 3.9+
- Git
- Raspberry Pi 4 (untuk testing deployment)
- Google Coral Edge TPU (untuk testing inference)

### Local Development

```bash
# Clone repository
git clone https://github.com/yourusername/conveyor-belt-damage-detection.git
cd conveyor-belt-damage-detection

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r deployment/requirements_rpi.txt

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy
```

### Testing Environment

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=. tests/

# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

## 📝 Coding Standards

### Python Style Guide

Kami mengikuti [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide dengan beberapa penyesuaian:

```python
# Good
def calculate_roi_area(coords: List[int], shape: str = "rectangle") -> float:
    """Calculate ROI area in pixels.
    
    Args:
        coords: ROI coordinates
        shape: ROI shape type
        
    Returns:
        ROI area in pixels
    """
    if shape == "rectangle":
        if len(coords) == 4:
            width, height = coords[2], coords[3]
            return width * height
    return 0.0

# Bad
def calcArea(c, s="rect"):
    if s=="rect":
        if len(c)==4:
            return c[2]*c[3]
    return 0
```

### File Organization

```
project/
├── module_name/
│   ├── __init__.py
│   ├── core.py
│   ├── utils.py
│   └── tests/
│       ├── __init__.py
│       ├── test_core.py
│       └── test_utils.py
├── docs/
├── examples/
└── README.md
```

### Documentation

- Gunakan docstrings untuk semua fungsi dan kelas
- Ikuti format Google style docstrings
- Sertakan type hints
- Dokumentasikan parameter, return values, dan exceptions

```python
def process_image(
    image: np.ndarray,
    roi_coords: Optional[List[int]] = None,
    quality_threshold: float = 0.8
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Process image with optional ROI and quality filtering.
    
    Args:
        image: Input image as numpy array
        roi_coords: Optional ROI coordinates [x, y, width, height]
        quality_threshold: Minimum quality score (0.0 to 1.0)
        
    Returns:
        Tuple of (processed_image, metadata)
        
    Raises:
        ValueError: If image is None or invalid
        RuntimeError: If processing fails
    """
    pass
```

## 🧪 Testing

### Unit Tests

Tulis unit tests untuk semua fungsi baru:

```python
# test_image_utils.py
import pytest
import numpy as np
from utils.image_utils import calculate_image_quality

def test_calculate_image_quality():
    # Create test image
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Test function
    quality = calculate_image_quality(test_image)
    
    # Assertions
    assert isinstance(quality, dict)
    assert "laplacian_variance" in quality
    assert "brightness" in quality
    assert "contrast" in quality
    assert 0 <= quality["brightness"] <= 255
```

### Integration Tests

Test integrasi antar komponen:

```python
# test_integration.py
def test_data_collection_to_inference():
    """Test complete pipeline from data collection to inference."""
    # Setup
    # Run data collection
    # Run inference
    # Verify results
    pass
```

### Performance Tests

```python
# test_performance.py
def test_inference_speed():
    """Test inference speed meets requirements."""
    start_time = time.time()
    # Run inference
    end_time = time.time()
    
    fps = 1.0 / (end_time - start_time)
    assert fps >= 30, f"Inference too slow: {fps:.1f} FPS"
```

## 🔄 Pull Request Process

### Before Submitting

1. **Test your changes**:
   ```bash
   pytest tests/
   black .
   flake8 .
   mypy .
   ```

2. **Update documentation**:
   - Update README.md jika diperlukan
   - Update docstrings
   - Update setup guide jika ada perubahan setup

3. **Check compatibility**:
   - Test di Raspberry Pi
   - Test dengan Edge TPU
   - Test dengan berbagai IP camera

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Tested on Raspberry Pi
- [ ] Tested with Edge TPU

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes
- [ ] Performance impact assessed

## Screenshots (if applicable)
```

### Review Process

1. **Automated checks** harus pass
2. **Code review** oleh maintainer
3. **Testing** di environment target
4. **Documentation** review
5. **Approval** dan merge

## 🐛 Reporting Bugs

### Bug Report Template

```markdown
## Bug Description
Clear and concise description of the bug

## Steps to Reproduce
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Expected Behavior
What you expected to happen

## Actual Behavior
What actually happened

## Environment
- Raspberry Pi: [e.g., Pi 4 8GB]
- OS: [e.g., Raspberry Pi OS Bullseye]
- Python: [e.g., 3.9.2]
- Edge TPU: [e.g., USB Accelerator]
- Camera: [e.g., Hikvision DS-2CD2142FWD-I]

## Configuration
```json
{
  "rtsp_url": "rtsp://...",
  "model_path": "best_edgetpu.tflite"
}
```

## Logs
```
Error message or log output
```

## Additional Context
Any other context about the problem
```

## 💡 Feature Requests

### Feature Request Template

```markdown
## Feature Description
Clear and concise description of the feature

## Problem Statement
What problem does this feature solve?

## Proposed Solution
How should this feature work?

## Alternatives Considered
What other solutions have you considered?

## Additional Context
Any other context, screenshots, or examples
```

## 📚 Documentation

### Contributing to Documentation

1. **Update README.md** untuk perubahan utama
2. **Update setup guide** untuk perubahan setup
3. **Update troubleshooting** untuk masalah baru
4. **Add docstrings** untuk fungsi baru
5. **Create examples** untuk fitur baru

### Documentation Standards

- Gunakan bahasa yang jelas dan mudah dipahami
- Sertakan contoh kode yang dapat dijalankan
- Gunakan diagram jika diperlukan
- Update semua referensi yang relevan

## 🏷️ Labels and Milestones

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `priority: high`: High priority issue
- `priority: low`: Low priority issue
- `status: in progress`: Work in progress
- `status: blocked`: Blocked by another issue

### Milestones

- `v1.0.0`: Initial release
- `v1.1.0`: Performance improvements
- `v1.2.0`: New features
- `v2.0.0`: Major refactoring

## 📞 Getting Help

### Communication Channels

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: For sensitive issues or private communication

### Resources

- [Python Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Edge TPU Documentation](https://coral.ai/docs/)
- [YOLO Documentation](https://docs.ultralytics.com/)

## 🙏 Acknowledgments

Terima kasih kepada semua kontributor yang telah membantu mengembangkan sistem ini. Setiap kontribusi, baik besar maupun kecil, sangat dihargai.

---

**Note**: Panduan ini dapat diperbarui dari waktu ke waktu. Pastikan untuk membaca versi terbaru sebelum berkontribusi.