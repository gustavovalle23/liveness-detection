"""
In this example, blink detection is used to perform liveness detection. 
The idea is that a real person will naturally blink their eyes, while a fake 
face may not have the ability to blink. Blink detection is performed using 
the eye aspect ratio, which measures the relative distance between the eye 
landmarks. If the eye aspect ratio falls below a certain threshold for a 
certain number of consecutive frames, the person is considered to be blinking. 
The liveness score is then determined as the inverse of the blink state - 
if the person is blinking, the score is 0, and if the person is not blinking, 
the score is 1.
"""

import cv2
import dlib
import numpy as np

# Load the face detector
detector = dlib.get_frontal_face_detector()

# Load the facial landmark predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# Define the function for computing the eye aspect ratio
def eye_aspect_ratio(eye):
    # Compute the distance between the vertical eye landmarks
    a = np.linalg.norm(eye[1] - eye[5])
    b = np.linalg.norm(eye[2] - eye[4])

    # Compute the distance between the horizontal eye landmarks
    c = np.linalg.norm(eye[0] - eye[3])

    # Compute the eye aspect ratio
    ear = (a + b) / (2.0 * c)

    return ear


# Define the function for blink detection
def blink_detection(eye_points, landmarks):
    # Extract the coordinates of the eye landmarks
    left_eye = landmarks[eye_points[0] : eye_points[1]]
    right_eye = landmarks[eye_points[2] : eye_points[3]]

    # Compute the eye aspect ratio for each eye
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)

    # Compute the average eye aspect ratio
    ear = (left_ear + right_ear) / 2.0

    # Define the thresholds for blink detection
    eye_aspect_ratio_threshold = 0.2
    eye_aspect_ratio_consecutive_frames = 3

    # Update the blink counter and state
    global blink_counter, blink_state
    if ear < eye_aspect_ratio_threshold:
        blink_counter += 1
        if blink_counter >= eye_aspect_ratio_consecutive_frames:
            blink_state = True
    else:
        blink_counter = 0
        blink_state = False


# Load the video stream
cap = cv2.VideoCapture(0)

# Initialize the blink counter and state
blink_counter = 0
blink_state = False

# Loop through each frame in the video stream
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Detect the face in the frame
    faces = detector(frame)

    # Perform blink detection to determine liveness
    if len(faces) == 1:
        landmarks = np.array([[p.x, p.y] for p in predictor(frame, faces[0]).parts()])
        blink_detection(
            [36, 42, 42, 48], landmarks
        )  # use eye landmarks to detect blink
        alive = not blink_state  # person is considered alive if not blinking
    else:
        alive = False

    # Display the result on the frame
    if alive:
        cv2.putText(
            frame,
            "Liveness: Real",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame,
            "Liveness: Fake",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # Display the frame
    cv2.imshow("Liveness Detection", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()
