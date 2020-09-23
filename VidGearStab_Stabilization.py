# Importing libraries
import cv2
from vidgear.gears.stabilizer import Stabilizer

# Capturing video
cap = cv2.VideoCapture("v1.mp4")

# Capturing video frames from camera on-device
# Initiating Stabilizer object with parameters of our choice
stab = Stabilizer(smoothing_radius=30, crop_n_zoom=True)

while(1):
    # Reading current frames
    ret, frame = cap.read()

    if not ret:
        break
    # Applying stabilizer processing to the current frame
    stabilized_frame = stab.stabilize(frame)

    if stabilized_frame is None:
        continue

    # Display stabilized frame
    stabilized_frame = cv2.resize(stabilized_frame, (500, 300))
    frame = cv2.resize(frame, (500, 300))
    cv2.imshow("Stabilized video", stabilized_frame)
    cv2.imshow("Input video", frame)

    key = cv2.waitKey(5)
    if key == ord('q'):
        break

# Exit control statements
cv2.destroyAllWindows()
stab.clean()
cap.release()