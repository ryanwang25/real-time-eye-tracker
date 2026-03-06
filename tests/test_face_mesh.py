# Development test: verify MediaPipe setup
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# landmark indices for each eye
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image = cv2.flip(image, 1)
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      face_landmarks = results.multi_face_landmarks[0]
      h, w, _ = image.shape
      left_eye_points = []
      for idx in LEFT_EYE:
        lm = face_landmarks.landmark[idx]
        left_eye_points.append((int(lm.x * w), int(lm.y * h)))

      right_eye_points = []
      for idx in RIGHT_EYE:
        lm = face_landmarks.landmark[idx]
        right_eye_points.append((int(lm.x * w), int(lm.y * h)))

      # draw eye contours
      for point in left_eye_points:
        cv2.circle(image, point, 2, (0, 255, 0), -1)
      for point in right_eye_points:
        cv2.circle(image, point, 2, (0, 255, 0), -1)


      # for face_landmarks in results.multi_face_landmarks:
      #   mp_drawing.draw_landmarks(
      #       image=image,
      #       landmark_list=face_landmarks,
      #       connections=mp_face_mesh.FACEMESH_TESSELATION,
      #       landmark_drawing_spec=None,
      #       connection_drawing_spec=mp_drawing_styles
      #       .get_default_face_mesh_tesselation_style())
      #   mp_drawing.draw_landmarks(
      #       image=image,
      #       landmark_list=face_landmarks,
      #       connections=mp_face_mesh.FACEMESH_CONTOURS,
      #       landmark_drawing_spec=None,
      #       connection_drawing_spec=mp_drawing_styles
      #       .get_default_face_mesh_contours_style())
      #   mp_drawing.draw_landmarks(
      #       image=image,
      #       landmark_list=face_landmarks,
      #       connections=mp_face_mesh.FACEMESH_IRISES,
      #       landmark_drawing_spec=None,
      #       connection_drawing_spec=mp_drawing_styles
      #       .get_default_face_mesh_iris_connections_style())
    else:
        cv2.putText(image, "No face detected", (80, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    cv2.imshow('MediaPipe Face Mesh', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
      break

# check resolutions
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(cap.get(cv2.CAP_PROP_FPS))
cap.release()


