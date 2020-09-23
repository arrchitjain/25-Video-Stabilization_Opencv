# Importing libraries
import numpy as np
import cv2

cap = cv2.VideoCapture("v1.mp4")
# cap = cv2.VideoCapture("v2.mp4")

def slidingAverage(curve, radius):
    winSize = 2 * radius + 1
    f = np.ones(winSize) / winSize
    # padding
    padding = np.lib.pad(curve, (radius, radius), 'edge')
    # Applying convolution
    smoothed = np.convolve(padding, f, mode='same')
    # Removing padding
    smoothed = smoothed[radius:-radius]
    return smoothed

def smoothingTrajectory(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(3):
        smoothed_trajectory[:, i] = slidingAverage(trajectory[:, i], radius=50)

    return smoothed_trajectory

# Accessing number of frames
noFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

wdth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
ht = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Reading the prevFrame(first frame)
retPrev, prevFrame = cap.read()

prev_gray = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)
trnsfm = np.zeros((noFrames - 1, 3), np.float32)

for i in range(noFrames - 2):
    prevPts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)

    retCurr, currFrame = cap.read()

    if not retCurr:
        break

    currGray = cv2.cvtColor(currFrame, cv2.COLOR_BGR2GRAY)

    # Tracking the featured points
    currPts, found, error = cv2.calcOpticalFlowPyrLK(prev_gray, currGray, prevPts, None)

    idx = np.where(found == 1)[0]
    prevPts = prevPts[idx]
    currPts = currPts[idx]
    assert prevPts.shape == currPts.shape

    # rigid transformation
    rigTrnsfm = cv2.estimateAffinePartial2D(prevPts, currPts)[0]

    x = rigTrnsfm[0, 2]
    y = rigTrnsfm[1, 2]
    angle = np.arctan2(rigTrnsfm[1, 0], rigTrnsfm[0, 0])

    trnsfm[i] = [x, y, angle]

    prev_gray = currGray

# Cumulative sum of trnsfm for x, y and angle
trajectory = np.cumsum(trnsfm, axis=0)

smoothed_trajectory = smoothingTrajectory(trajectory)
diff = smoothed_trajectory - trajectory
trnsfmSmooth = trnsfm + diff

# Resetting to first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

for i in range(noFrames - 2):
    # Reading nxtFrame
    retNxt, nxtFrame = cap.read()
    if not retNxt:
        break

    # Accepting values of x, y and angle
    x = trnsfmSmooth[i, 0]
    y = trnsfmSmooth[i, 1]
    angle = trnsfmSmooth[i, 2]

    # Filling transformation values
    rigTrnsfm = np.zeros((2, 3), np.float32)
    rigTrnsfm[0, 0] = np.cos(angle)
    rigTrnsfm[0, 1] = -np.sin(angle)
    rigTrnsfm[1, 0] = np.sin(angle)
    rigTrnsfm[1, 1] = np.cos(angle)
    rigTrnsfm[0, 2] = x
    rigTrnsfm[1, 2] = y

    # wrapAffine function operation on nxtFrame
    fnlFrame = cv2.warpAffine(nxtFrame, rigTrnsfm, (wdth, ht))

    # Display stabilized output
    cv2.imshow("Output video", fnlFrame)
    cv2.imshow("Input video", nxtFrame)

    if cv2.waitKey(50) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()























# import numpy as np
# import cv2
#
# smoothRadius = 50
#
# cap = cv2.VideoCapture("v2.mp4")
#
# def slidingAverage(curve, radius):
#     winSize = 2 * radius + 1
#     f = np.ones(winSize) / winSize
#     # padding
#     padding = np.lib.pad(curve, (radius, radius), 'edge')
#     # Applying convolution
#     smoothed = np.convolve(padding, f, mode='same')
#     # Removing padding
#     smoothed = smoothed[radius:-radius]
#     return smoothed
#
#
# def smoothingTrajectory(trajectory):
#     smoothed_trajectory = np.copy(trajectory)
#     for i in range(3):
#         smoothed_trajectory[:, i] = slidingAverage(trajectory[:, i], radius=smoothRadius)
#
#     return smoothed_trajectory
#
#
# # Accessing number of frames
# noFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
# # To check the number of frames in the video
# print("Number of frames : ", noFrames)
#
# wdth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# ht = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
# print("Width : ", wdth)
# print("height : ", ht)
#
# # Reading the prevFrame(first frame)
# retPrev, prevFrame = cap.read()
#
# prev_gray = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)
# trnsfm = np.zeros((noFrames - 1, 3), np.float32)
#
# for i in range(noFrames - 2):
#     prevPts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
#
#     retCurr, currFrame = cap.read()
#
#     if not retCurr:
#         break
#
#     currGray = cv2.cvtColor(currFrame, cv2.COLOR_BGR2GRAY)
#
#     # Tracking the featured points
#     currPts, found, error = cv2.calcOpticalFlowPyrLK(prev_gray, currGray, prevPts, None)
#
#     idx = np.where(found == 1)[0]
#     prevPts = prevPts[idx]
#     currPts = currPts[idx]
#     assert prevPts.shape == currPts.shape
#
#     # rigid transformation
#     rigTrnsfm = cv2.estimateAffinePartial2D(prevPts, currPts)[0]
#
#     x = rigTrnsfm[0, 2]
#     y = rigTrnsfm[1, 2]
#     angle = np.arctan2(rigTrnsfm[1, 0], rigTrnsfm[0, 0])
#
#     trnsfm[i] = [x, y, angle]
#
#     prev_gray = currGray
#
# # Cumulative sum of trnsfm for x, y and angle
# trajectory = np.cumsum(trnsfm, axis=0)
#
# smoothed_trajectory = smoothingTrajectory(trajectory)
# diff = smoothed_trajectory - trajectory
# trnsfmSmooth = trnsfm + diff
#
# # Resetting to first frame
# cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#
# for i in range(noFrames - 2):
#     # Reading nxtFrame
#     retNxt, nxtFrame = cap.read()
#     if not retNxt:
#         break
#
#     # Accepting values of x, y and angle
#     x = trnsfmSmooth[i, 0]
#     y = trnsfmSmooth[i, 1]
#     angle = trnsfmSmooth[i, 2]
#
#     # Filling transformation values
#     rigTrnsfm = np.zeros((2, 3), np.float32)
#     rigTrnsfm[0, 0] = np.cos(angle)
#     rigTrnsfm[0, 1] = -np.sin(angle)
#     rigTrnsfm[1, 0] = np.sin(angle)
#     rigTrnsfm[1, 1] = np.cos(angle)
#     rigTrnsfm[0, 2] = x
#     rigTrnsfm[1, 2] = y
#
#     # wrapAffine function operation on nxtFrame
#     fnlFrame = cv2.warpAffine(nxtFrame, rigTrnsfm, (wdth, ht))
#
#     cv2.imshow("Output video", fnlFrame)
#     cv2.imshow("Input video", nxtFrame)
#
#     # # Appending nxtFrame
#     # finalFrame = cv2.hconcat([nxtFrame, fnlFrame])
#     #
#     # # Resizing frame
#     # if (finalFrame.shape[1] > 1920):
#     #     finalFrame = cv2.resize(finalFrame, (finalFrame.shape[1] / 2, finalFrame.shape[0] / 2));
#     #
#     # cv2.imshow("Left window : Input   and   Right window : Output", finalFrame)
#
#     if cv2.waitKey(50) == ord("q"):
#         break
#
# cap.release()
# cv2.destroyAllWindows()