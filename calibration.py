import numpy as np
import cv2
import glob

# from google.colab.patches import cv2_imshow

# ==== CONFIGURATION ======================================================================

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# checkerboard Dimensions
cbrow = 12
cbcol = 10

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((cbrow * cbcol, 3), np.float32)
objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = np.array(glob.glob('/Users/felipemoran/Dropbox/Poli/TCC/calibration/iphone_xr/dataset_5/photos_converted/*'))
# images = images[[0, 1, 2, 3, 4, 6, 7, 8, 9, 14, 17]]

# img = cv2.imread(images[0])
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print('{} images found'.format(len(images)))

for i, fname in enumerate(images):
    print('Processing image {} / {} : '.format(i, len(images)), end='')
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (cbcol, cbrow), None)

    # If found, add object points, image points (after refining them)
    print(ret, end=' ')
    if not ret:
        print(fname)
    else:
        print()

    if ret is True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        # img = cv2.drawChessboardCorners(img, (7,10), corners2,ret)
        # cv2_imshow(img)
        # cv2.waitKey(500)

# cv2.destroyAllWindows()

# ==== CALIBRATE CAMERA ======================================================================

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(mtx)

# ==== REPROJECTION ERROR ======================================================================

mean_error = 0
tot_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    tot_error += error

print("total error: ", tot_error / len(objpoints))

# ==== UNDISTORTION ======================================================================

img = cv2.imread(images[0])
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv2.imwrite('calibresult.png', dst)

# undistort
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv2.imwrite('calibresult.png', dst)

