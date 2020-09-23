# Importing libraries
import cv2
from vidgear.gears import VideoGear

# Capturing and stabilizing the frame
# Put stabilize = True to activate stabilization features
cap = VideoGear(source= "v1.mp4", stabilize=True).start()

while (1):
    # Reading stabilized frame
    frame_final = cap.read()

    # To avoid error returned value
    if frame_final is None:
        break

    # Display the stabilized frame
    cv2.imshow("Stabilized video", frame_final)

    key = cv2.waitKey(1)
    if key == 32:
        break

# Exit control statement
cap.stop()
cv2.destroyAllWindows()