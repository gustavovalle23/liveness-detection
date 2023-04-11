"""
In this example, liveness detection is performed by examining the differences
and properties of optical flow generated from 3D objects and 2D planes.
The idea is that 3D objects will produce a more complex and irregular 
pattern of optical flow compared to 2D planes, which will produce a more 
regular and uniform pattern of optical flow. The flow magnitude is 
calculated using the cv2.calcOpticalFlowFarneback function, and the 
average flow magnitude is computed for each row and column of the flow
image. The ratio of the average flow magnitudes in the row and column directions 
is then computed, and if this ratio is above a certain threshold, the object is 
considered a 2D plane and thus fake. Otherwise, it is considered a 3D object and thus real.
"""

import cv2
import numpy as np

# Load the video stream
cap = cv2.VideoCapture(0)

# Initialize variables for liveness detection
frame_count = 0
prev_frame = None
threshold = 1.0

# Loop through each frame in the video stream
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform liveness detection by examining the differences and properties of optical flow generated from 3D objects and 2D planes
    if frame_count > 0:
        # Compute the optical flow between the previous frame and the current frame
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # Compute the average flow magnitude for each row and column
        mag_row = np.mean(np.abs(flow[:, :, 0]), axis=1)
        mag_col = np.mean(np.abs(flow[:, :, 1]), axis=0)

        # Compute the ratio of the average flow magnitudes in the row and column directions
        ratio = np.mean(mag_row) / np.mean(mag_col)

        # Determine if the object is a 2D plane or a 3D object based on the ratio of flow magnitudes
        if ratio > threshold:
            alive = False
        else:
            alive = True
    else:
        alive = True

    # Update variables for liveness detection
    prev_frame = gray
    frame_count += 1

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
