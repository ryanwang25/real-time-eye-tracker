# Real-Time Eye Tracking System

A real-time eye tracking system that detects and classifies eye states (open, closed, winking) using computer vision. Built with OpenCV for video processing and MediaPipe Face Mesh for facial landmark detection.

## How It Works

The system captures webcam frames, detects facial landmarks using MediaPipe's Face Mesh model, and extracts 6 key landmarks per eye. It then computes the **Eye Aspect Ratio (EAR)**, a geometric measure of how open each eye is, and classifies the eye state based on a configurable threshold.

**EAR Formula:**

```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```

Where p1/p4 are the horizontal eye corners and p2, p3, p5, p6 are the vertical eye landmarks.

## Features

- Real-time eye state classification (OPEN / CLOSED / LEFT WINK / RIGHT WINK)
- Eye Aspect Ratio (EAR) computation and on-screen display
- Eye contour landmark visualization
- Consecutive frame counting to reduce classification jitter (worked better than temporal smoothing)
- Handling of edge cases (no face detected, camera failure, face partially out of frame)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/eye_tracking_project.git
cd eye_tracking_project
```

2. Create and activate a virtual environment:
```bash
python -m venv myenv
source myenv/bin/activate  # macOS/Linux
myenv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the eye tracker:
```bash
python eye_tracker.py
```

Press `q` to quit the application.

### Configuring the Threshold

The EAR threshold can be adjusted when creating the tracker. The default is 0.21, but values between 0.18–0.25 may work better depending on your face and lighting:

```python
tracker = EyeTracker()
```

## Project Structure

```
eye_tracking_project/
├── eye_tracker.py        # Main implementation (EyeTracker class)
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── tests/
    ├── test_ear.py       # Unit tests for EAR calculation
    ├── test_webcam.py    # Development test: verify webcam setup
    └── test_face_mesh.py # Development test: verify MediaPipe setup
```

## Dependencies

- Python 3.10+
- opencv-python >= 4.8.0
- mediapipe == 0.10.14
- numpy >= 1.24.0
- scipy >= 1.10.0

## Known Limitations

- EAR-based classification can be unreliable when the face is not roughly facing the camera. A face orientation check mitigates this but does not fully solve it at extreme angles.
- The EAR threshold is sensitive to individual face geometry.
- Rapid blinks may occasionally be missed for higher numbers of consecutive frame counting
- Extreme angles of face orientation can cause false positives and negatives.
