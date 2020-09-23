# Importing libraries
import cv2
from vidstab import VidStab, layer_blend

# Initialize stabilizer and video reader
stabilizer = VidStab()
vidcap = cv2.VideoCapture("v1.mp4")

while True:
    # Reading current frame
    ret, frame = vidcap.read()

    # Stabilization process
    # Pass frame to stabilizer even if frame is None
    stabilized_frame = stabilizer.stabilize_frame(input_frame=frame,
                                                  layer_func=layer_blend,
                                                  border_size=30)

    # If stabilized_frame is None then there are no frames left to process
    if stabilized_frame is None:
        break

    # Display stabilized output
    cv2.imshow('Stabilized output', stabilized_frame)

    key = cv2.waitKey(5)
    if key == ord('q'):
        break

vidcap.release()
cv2.destroyAllWindows()