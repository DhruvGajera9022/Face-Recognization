# 🧠 Enhanced Face Detection System

A powerful real-time face detection system using OpenCV that supports both Haar Cascade and DNN (Deep Neural Network) based detection. It includes FPS tracking, UI overlay, pause/resume controls, and smoothed face count for a stable experience.

---

## 🚀 Features

- ✅ **Real-time Face Detection**
- 🔁 **Switch between Haar Cascade and DNN detection**
- 🧮 **Smoothed FPS & Face Count**
- ⏸️ **Pause/Resume Detection**
- 🎯 **Overlay UI for performance metrics**
- 📷 **Camera Resolution & FPS Configuration**
- 🔁 **Live Frame Mirroring (Selfie-like view)**

---

## 📦 Requirements

- Python 3.6+
- OpenCV (`opencv-python`)
- NumPy

You also need the following DNN model files (optional but recommended for better accuracy):

- `opencv_face_detector_uint8.pb`
- `opencv_face_detector.pbtxt`

> ⚠️ If DNN files are not found, it will automatically fall back to Haar Cascade.

---

## 📥 Installation

```bash
# Clone the repository
git clone https://github.com/DhruvGajera9022/Face-Recognization.git
cd Face-Recognization

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
