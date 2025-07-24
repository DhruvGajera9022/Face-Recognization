import cv2
import time
import numpy as np
from collections import deque
import argparse
import sys


class EnhancedFaceDetector:
    def __init__(self, camera_id=0, width=1280, height=720, confidence_threshold=0.5):
        """
        Initialize the Enhanced Face Detection System

        Args:
            camera_id (int): Camera device ID
            width (int): Frame width
            height (int): Frame height
            confidence_threshold (float): Minimum confidence for face detection
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.confidence_threshold = confidence_threshold

        # Initialize face detection models
        self.init_face_detectors()

        # Performance tracking
        self.fps_history = deque(maxlen=30)  # Track last 30 FPS values
        self.face_count_history = deque(maxlen=10)  # Smooth face count

        # Camera setup
        self.webcam = None
        self.setup_camera()

        # Colors for different detection states
        self.colors = {
            'face': (0, 255, 0),  # Green for faces
            'text': (255, 255, 255),  # White for text
            'fps': (100, 255, 100),  # Light green for FPS
            'info': (255, 200, 0)  # Cyan for info
        }

    def init_face_detectors(self):
        """Initialize multiple face detection methods"""
        # Haar Cascade (fast but less accurate)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Try to load DNN model for better accuracy (optional)
        try:
            self.net = cv2.dnn.readNetFromTensorflow(
                'opencv_face_detector_uint8.pb',
                'opencv_face_detector.pbtxt'
            )
            self.use_dnn = True
            print("DNN face detector loaded successfully")
        except:
            self.use_dnn = False
            print("Using Haar Cascade detector (DNN model not found)")

    def setup_camera(self):
        """Setup camera with optimal settings"""
        self.webcam = cv2.VideoCapture(self.camera_id)

        if not self.webcam.isOpened():
            print(f"Error: Could not access camera {self.camera_id}")
            sys.exit(1)

        # Set camera properties
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.webcam.set(cv2.CAP_PROP_FPS, 30)
        self.webcam.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency

        # Verify actual resolution
        actual_width = int(self.webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera initialized: {actual_width}x{actual_height}")

    def detect_faces_haar(self, gray):
        """Detect faces using Haar Cascade"""
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            maxSize=(300, 300),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces

    def detect_faces_dnn(self, img):
        """Detect faces using DNN model"""
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123])
        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                faces.append((x1, y1, x2 - x1, y2 - y1, confidence))

        return faces

    def smooth_face_count(self, current_count):
        """Smooth face count to reduce flickering"""
        self.face_count_history.append(current_count)
        return int(np.median(self.face_count_history))

    def calculate_fps(self, frame_time):
        """Calculate smoothed FPS"""
        if frame_time > 0:
            current_fps = 1.0 / frame_time
            self.fps_history.append(current_fps)
            return int(np.mean(self.fps_history))
        return 0

    def draw_face_info(self, img, faces, detection_method):
        """Draw face rectangles and additional info"""
        for i, face in enumerate(faces):
            if len(face) == 4:  # Haar cascade format
                x, y, w, h = face
                confidence = None
            else:  # DNN format
                x, y, w, h, confidence = face

            # Draw main rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), self.colors['face'], 2)

            # Draw face number
            cv2.putText(img, f'#{i + 1}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['face'], 2)

            # Draw confidence if available
            if confidence:
                cv2.putText(img, f'{confidence:.2f}', (x, y + h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['info'], 1)

    def draw_ui_overlay(self, img, face_count, fps, detection_method):
        """Draw UI overlay with statistics"""
        h, w = img.shape[:2]

        # Semi-transparent overlay for text background
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

        # Face count
        cv2.putText(img, f'Faces Detected: {face_count}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['text'], 2)

        # FPS
        cv2.putText(img, f'FPS: {fps}', (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['fps'], 2)

        # Detection method
        cv2.putText(img, f'Method: {detection_method}', (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['info'], 2)

        # Controls info
        cv2.putText(img, 'ESC: Exit | S: Switch Method | Space: Pause',
                    (w - 450, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    self.colors['text'], 1)

    def run(self):
        """Main detection loop"""
        print("Starting Enhanced Face Detection...")
        print("Controls:")
        print("  ESC - Exit")
        print("  S - Switch detection method")
        print("  SPACE - Pause/Resume")
        print("  Q - Quit")

        paused = False
        use_dnn_method = self.use_dnn

        try:
            while True:
                if not paused:
                    start_time = time.time()

                    # Read frame
                    ret, img = self.webcam.read()
                    if not ret:
                        print("Error: Failed to read from webcam")
                        break

                    # Flip frame horizontally for mirror effect
                    img = cv2.flip(img, 1)

                    # Convert to grayscale for Haar cascade
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    # Apply histogram equalization for better detection
                    gray = cv2.equalizeHist(gray)

                    # Detect faces based on selected method
                    if use_dnn_method and self.use_dnn:
                        faces = self.detect_faces_dnn(img)
                        method_name = "DNN"
                    else:
                        faces = self.detect_faces_haar(gray)
                        method_name = "Haar Cascade"

                    # Smooth face count
                    smooth_count = self.smooth_face_count(len(faces))

                    # Calculate FPS
                    frame_time = time.time() - start_time
                    fps = self.calculate_fps(frame_time)

                    # Draw face detection results
                    self.draw_face_info(img, faces, method_name)

                    # Draw UI overlay
                    self.draw_ui_overlay(img, smooth_count, fps, method_name)

                # Display frame
                cv2.imshow("Enhanced Face Detection System", img)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF

                if key == 27:  # ESC
                    break
                elif key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('s') or key == ord('S'):
                    if self.use_dnn:
                        use_dnn_method = not use_dnn_method
                        method = "DNN" if use_dnn_method else "Haar Cascade"
                        print(f"Switched to {method}")
                    else:
                        print("DNN model not available")
                elif key == ord(' '):  # Space
                    paused = not paused
                    status = "Paused" if paused else "Resumed"
                    print(f"Detection {status}")

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        if self.webcam:
            self.webcam.release()
        cv2.destroyAllWindows()
        print("Cleanup completed")


def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced Face Detection System')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    parser.add_argument('--width', type=int, default=1280, help='Frame width')
    parser.add_argument('--height', type=int, default=720, help='Frame height')
    parser.add_argument('--confidence', type=float, default=0.5, help='DNN confidence threshold')

    args = parser.parse_args()

    # Create and run face detector
    detector = EnhancedFaceDetector(
        camera_id=args.camera,
        width=args.width,
        height=args.height,
        confidence_threshold=args.confidence
    )

    detector.run()


if __name__ == "__main__":
    main()