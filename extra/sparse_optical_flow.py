import numpy as np
import cv2


# cap = cv2.VideoCapture('/Users/felipemoran/Desktop/TCC/datasets/slow.mp4')

cap = cv2.VideoCapture("/Users/felipemoran/Desktop/TCC/datasets/sala/480_30.mp4")



# params for ShiTomasi corner detection
feature_params = {
    "maxCorners": 200,
    "qualityLevel": 0.3,
    "minDistance": 7,
    "blockSize": 7
}

# Parameters for lucas kanade optical flow
lk_params = {
    "winSize": (15, 15),
    "maxLevel": 2,
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
}

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, first_frame = cap.read()
prev_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
features_prev_frame = cv2.goodFeaturesToTrack(prev_frame_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(first_frame)

frame_counter = 0

while(1):
    frame_counter += 1
    print("Processin frame {}...".format(frame_counter))

    ret, next_frame = cap.read()
    if not ret:
        break

    next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    features_next_frame, status, err = cv2.calcOpticalFlowPyrLK(
        prev_frame_gray,
        next_frame_gray,
        features_prev_frame,
        None,
        **lk_params
    )

    # Select good points
    good_features_next_frame = features_next_frame[status == 1]
    good_features_prev_frame = features_prev_frame[status == 1]

    # draw the tracks
    for feature_index, (point_next, point_prev) in enumerate(zip(good_features_next_frame, good_features_prev_frame)):
        a, b = point_next.ravel()
        c, d = point_prev.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[feature_index].tolist(), 2)
        next_frame = cv2.circle(next_frame, (a, b), 5, color[feature_index].tolist(), -1)

    img = cv2.add(next_frame, mask)

    cv2.imshow('frame', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    prev_frame_gray = next_frame_gray.copy()
    features_prev_frame = good_features_next_frame.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()
