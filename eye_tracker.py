import numpy as np
import cv2
import mediapipe as mp

class EyeTracker:
    """
    Real-time eye tracking system that detects and classifies eye states
    (open/closed/winking) using MediaPipe Face Mesh and Eye Aspect Ratio (EAR).
    """
    
    # MediaPipe Face Mesh landmark indices for each eye
    # Order: [p1, p2, p3, p4, p5, p6] mapping to EAR formula
    # p1, p4 = horizontal corners; p2, p3, p5, p6 = vertical points

    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    

    def __init__(self, ear_threshold=0.21, max_faces=1, detection_conf=0.5, tracking_conf=0.5):
        """
        Initialize the EyeTracker with MediaPipe Face Mesh and webcam.

        Args:
            ear_threshold: EAR value below which an eye is classified as closed.
            max_faces: Maximum number of faces to detect (default 1).
            detection_conf: Minimum confidence for initial face detection (0.0-1.0).
            tracking_conf: Minimum confidence for face tracking between frames (0.0-1.0).
        
        Raises:
            RuntimeError: If webcam cannot be opened.
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
        max_num_faces=max_faces,
        refine_landmarks=True,
        min_detection_confidence=detection_conf,
        min_tracking_confidence=tracking_conf
    )
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")
        # Set EAR threshold
        self.ear_threshold = ear_threshold
        
        
        
    def calculate_ear(self, eye_landmarks):
        """
        Compute the Eye Aspect Ratio (EAR) for a single eye.
        
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

        A higher EAR indicates an open eye; a lower EAR indicates a closed eye.

        Args:
            eye_landmarks: List of 6 (x, y) tuples corresponding to
                           [p1, p2, p3, p4, p5, p6] of the EAR formula.

        Returns:
            Float representing the EAR value.
        """
        p1, p2, p3, p4, p5, p6 = eye_landmarks

        # vertical distances (top-to-bottom pairs)
        vertical1 = self.euclidean_dist(np.array(p2), np.array(p6))
        vertical2 = self.euclidean_dist(np.array(p3), np.array(p5))

        # horizontal distance (corner-to-corner)
        horizontal = self.euclidean_dist(np.array(p1), np.array(p4))
        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        return ear
        
    def get_eye_landmarks(self, landmarks, indices, frame_w, frame_h):
        """
        Extract eye landmark pixel coordinates from MediaPipe face landmarks.

        Args:
            landmarks: MediaPipe NormalizedLandmarkList for a detected face.
            indices: List of landmark indices to extract (LEFT_EYE or RIGHT_EYE).
            frame_w: Width of the video frame in pixels.
            frame_h: Height of the video frame in pixels.

        Returns:
            List of (x, y) tuples in pixel coordinates.
        """
        points = []
        for idx in indices:
            lm = landmarks.landmark[idx]
            points.append((int(lm.x * frame_w), int(lm.y * frame_h)))
        return points
        
    def process_frame(self, frame):
        """
        Process a single video frame: detect face, extract eye landmarks,
        compute EAR, classify eye state, and draw annotations.

        Args:
            frame: BGR image (NumPy array) from the webcam.

        Returns:
            Annotated BGR frame with eye contours, EAR values, and state label.
        """
        # mirror the frame for intuitive selfie-view
        frame = cv2.flip(frame, 1)
        frame.flags.writeable = False
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        frame.flags.writeable = True

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape

            left_eye = self.get_eye_landmarks(face_landmarks, self.LEFT_EYE, w, h)
            right_eye = self.get_eye_landmarks(face_landmarks, self.RIGHT_EYE, w, h)

            for point in left_eye:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)
            for point in right_eye:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)

            # calculate EAR
            left_ear = self.calculate_ear(left_eye)
            right_ear = self.calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2

            # calculating face turned
            nose = face_landmarks.landmark[1]
            chin = face_landmarks.landmark[152]
            forehead = face_landmarks.landmark[10]

            left_edge = face_landmarks.landmark[234]
            right_edge = face_landmarks.landmark[454]

            left_dist = abs(nose.x - left_edge.x)
            right_dist = abs(nose.x - right_edge.x)

            top_dist = abs(nose.y - forehead.y)
            bottom_dist = abs(nose.y - chin.y)

            vertical_ratio = min(top_dist, bottom_dist) / max(top_dist, bottom_dist)

            horizontal_ratio = min(left_dist, right_dist) / max(left_dist, right_dist)

            if horizontal_ratio < 0.1 or vertical_ratio < 0.55:
                state = "FACE TURNED"
                color = (255, 255, 0)

             # classify eye state using per-eye EAR to detect winks
            elif left_ear < self.ear_threshold and right_ear < self.ear_threshold:
                state = "CLOSED"
                color = (0, 0, 255)
            elif left_ear < self.ear_threshold and right_ear >= self.ear_threshold:
                state = "RIGHT WINK"
                color = (0, 255, 255)
            elif right_ear < self.ear_threshold and left_ear >= self.ear_threshold:
                state = "LEFT WINK"
                color = (0, 255, 255)
            else:
                state = "OPEN"
                color = (0, 255, 0)

            # overlay EAR values and state on the frame
            cv2.putText(frame, f"L EAR: {left_ear:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"R EAR: {right_ear:.2f}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"State: {state}", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        else:
             # no face found — display warning message
            cv2.putText(frame, "No face detected", (80, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        return frame
        
    def run(self):
        """
        Main loop: continuously captures frames from the webcam, processes
        each frame, and displays the result. Press 'q' to exit.
        """
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = self.process_frame(frame)
            cv2.imshow("Eye Tracker", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        self.face_mesh.close()
        cv2.destroyAllWindows()
        
    
    def euclidean_dist(self, pt1, pt2):
        """
        Compute the Euclidean distance between two points.

        Args:
            pt1: NumPy array of first point.
            pt2: NumPy array of second point.

        Returns:
            Float representing the L2 distance between pt1 and pt2.
        """
        return np.linalg.norm(pt1 - pt2) # returns 2 norm by default
    
if __name__ == "__main__":
    tracker = EyeTracker(ear_threshold=0.18)
    tracker.run()


