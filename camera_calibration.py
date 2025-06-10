import numpy as np
import cv2 as cv
import glob
import os
from datetime import datetime
from controller import UR5_controller

def collection():
    ur5 = UR5_controller()
    k4a = ur5.k4a
    capture = k4a.get_capture()
    color_image = capture.color[:, :, :3]
    # depth_image = capture.transformed_depth

    # === Save Images ===
    img_dir = "camera_calib_images"
    os.makedirs(img_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv.imwrite(f"{img_dir}/color_{timestamp}.png", color_image)
    # cv.imwrite(f"{img_dir}/depth_{timestamp}.png", depth_image)
    print("Image Saved.")

def calibration():
  chessboard_size = (11, 8)  # (col, row) inside corner
  square_size = 0.02       # block size, 0.02 means 2cm

  # stop condition
  criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

  # (Z=0 plane)
  objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
  objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
  objp *= square_size

  # Save
  objpoints = []  # 3D : World
  imgpoints = []  # 2D : Image

  # load all images
  images = glob.glob('camera_calib_images/*.png')

  for fname in images:
      img = cv.imread(fname)
      gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

      ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)
      if ret:
          print(f"Found corners in: {fname}")
          objpoints.append(objp)
          corners_subpix = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
          imgpoints.append(corners_subpix)

          cv.drawChessboardCorners(img, chessboard_size, corners_subpix, ret)
          cv.imshow('Corners', img)
          cv.waitKey(100)
      else:
          print(f"Failed to detect corners in: {fname}")

  cv.destroyAllWindows()

  # camera calibration
  ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
      objpoints, imgpoints, gray.shape[::-1], None, None)

  # print
  print("\n=== Calibration Results ===")
  print("Camera Matrix (Intrinsic Parameters):\n", camera_matrix)
  print("Distortion Coefficients:\n", dist_coeffs.ravel())

  # save
  np.savez("calib_intrinsic_result.npz", K=camera_matrix, D=dist_coeffs)

  # calculate error
  mean_error = 0
  for i in range(len(objpoints)):
      imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
      error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
      mean_error += error

  print(f"Mean Reprojection Error: {mean_error / len(objpoints):.4f}")

if __name__ == '__main__':
    # collection()

    calibration()