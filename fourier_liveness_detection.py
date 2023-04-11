"""
In this example, liveness detection is performed by examining the frequency 
content of the face in the Fourier domain. The idea is that a real face will
have certain frequency content in the low-frequency region of the Fourier 
spectrum, while a fake face may not have the same frequency content. 
The face ROI is extracted from the frame and its two-dimensional 
discrete Fourier transform (DFT) is computed. The power spectrum of the 
DFT is then computed, and the mean power in the low-frequency region is 
calculated. If the mean power in the low-frequency region is above a 
certain threshold, the person is

"""

import cv2
import numpy as np
import dlib
import scipy.fftpack

# Load the face detector
detector = dlib.get_frontal_face_detector()

# Load the facial landmark predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# Define the function for liveness detection using Fourier analysis
def liveness_detection(frame, landmarks):
    # Extract the region of interest (ROI) around the face
    face_x1 = np.min(landmarks[:, 0])
    face_x2 = np.max(landmarks[:, 0])
    face_y1 = np.min(landmarks[:, 1])
    face_y2 = np.max(landmarks[:, 1])
    face_roi = frame[face_y1:face_y2, face_x1:face_x2]

    # Compute the two-dimensional discrete Fourier transform (DFT) of the face ROI
    face_dft = scipy.fftpack.fft2(face_roi)

    # Compute the power spectrum of the face DFT
    face_power_spectrum = np.abs(face_dft) ** 2

    # Compute the mean power in the low-frequency region of the power spectrum
    power_threshold = 1000000  # threshold for power in low-frequency region
    mean_low_power = np.mean(face_power_spectrum[:10, :10])

    # Determine the liveness score based on the mean power in the low-frequency region
    if mean_low_power > power_threshold:
        alive = True
    else:
        alive = False

    return alive


# Load the video stream
cap = cv2.VideoCapture(0)

# Loop through each frame in the video stream
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Detect the face in the frame
    faces = detector(frame)

    # Perform liveness detection using Fourier analysis
    if len(faces) == 1:
        landmarks = np.array([[p.x, p.y] for p in predictor(frame, faces[0]).parts()])
        alive = liveness_detection(frame, landmarks)
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
