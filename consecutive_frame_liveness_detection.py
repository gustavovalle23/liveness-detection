"""
In this example, liveness detection is performed by examining the variation 
of pixel values between two consecutive frames. The idea is that a real face 
will have some variation in pixel values between two consecutive frames due to 
natural movements and changes in lighting, while a fake face may not have 
the same variation. The difference between two consecutive frames is computed 
using the cv2.absdiff function, and the mean value of the difference image is 
calculated. If the mean difference is above a certain threshold, the person 
is considered fake. Otherwise, they are considered real.
"""

import cv2
import numpy as np

# Load the video stream
cap = cv2.VideoCapture(0)

# Initialize variables for liveness detection
frame_count = 0
prev_frame = None
threshold = 20

# Loop through each frame in the video stream
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Perform liveness detection by examining the variation of pixel values between two consecutive frames
    if frame_count > 0:
        frame_diff = cv2.absdiff(gray, prev_frame)
        diff_mean = np.mean(frame_diff)
        if diff_mean > threshold:
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
        cv2.putText(frame, 'Liveness: Real', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'Liveness: Fake', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow('Liveness Detection', frame)
    
    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()
