import os
import sys
import yaml
import cv2
# from cv2.aruco import estimatePoseBoard
import numpy as np
from scipy.spatial.transform import Rotation as R
from datetime import datetime

ROBOT_ONLINE = False  # Set to False if the robot is not online

CALIBRATION = False  # Set to True if you want to perform calibration

def collection():
    """
    Collects data for hand-eye calibration by capturing images and detecting ArUco markers.
    """
    global ROBOT_ONLINE

    if ROBOT_ONLINE:
        # === Initialize ===
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from controller import UR5_controller
        ur5 = UR5_controller()
        tcp_pose = ur5.rtde_r.getActualTCPPose()
        position = tcp_pose[:3]
        rotvec = tcp_pose[3:6]
        R_gripper2base = R.from_rotvec(rotvec).as_matrix()

        k4a = ur5.k4a
        capture = k4a.get_capture()
        color_image = capture.color[:, :, :3]
        depth_image = capture.transformed_depth
        camera_matrix = np.array(k4a.calibration.get_camera_matrix("color"))
        dist_coeffs = np.array(k4a.calibration.get_distortion_coefficients("color"))

        # === Save Images ===
        img_dir = "/home/mainuser/UR5_yc_mh/calibration/hand_eye_clib_images"
        os.makedirs(img_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"{img_dir}/color_{timestamp}.png", color_image)
        cv2.imwrite(f"{img_dir}/depth_{timestamp}.png", depth_image)
        print("Image Saved.")

    # === detect ArUco GridBoard ===
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()

    # Create GridBoard（4x3）
    board = cv2.aruco.GridBoard(
        size=(4, 3),              
        markerLength=0.05,       # Every marker is 5cm
        markerSeparation=0.005,  # interval is 0.5cm
        dictionary=aruco_dict
    )

    exit("ArUco GridBoard created with 4x3 markers.")

    # detect
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, _ = aruco_detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        print("No ArUco marker detected. Sample not saved.")
        sys.exit(1)
    # estimate pose
    retval, rvec, tvec = cv2.aruco.estimatePoseBoard(corners, ids, board, camera_matrix, dist_coeffs, None, None)
    if not retval:
        print("ArUco board detected but pose estimation failed.")
        sys.exit(1)

    R_target2cam, _ = cv2.Rodrigues(rvec)
    t_target2cam = tvec.reshape(-1)

    # === write YAML file ===
    yaml_path = "handeye_data.yaml"
    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
    else:
        data = {
            "R_gripper2base": [],
            "t_gripper2base": [],
            "R_target2cam": [],
            "t_target2cam": []
        }

    data["R_gripper2base"].append(R_gripper2base.tolist())
    data["t_gripper2base"].append(position)
    data["R_target2cam"].append(R_target2cam.tolist())
    data["t_target2cam"].append(t_target2cam.tolist())

    with open(yaml_path, "w") as f:
        yaml.dump(data, f)

    print(f"Sample saved to {yaml_path}. Current total: {len(data['t_gripper2base'])}")


def calibration():
    """
    Performs hand-eye calibration using the collected data.
    """
    with open("handeye_data.yaml", "r") as f:
        data = yaml.safe_load(f)

    R_gripper2base = [np.array(R) for R in data["R_gripper2base"]]
    t_gripper2base = [np.array(t) for t in data["t_gripper2base"]]
    R_target2cam = [np.array(R) for R in data["R_target2cam"]]
    t_target2cam = [np.array(t) for t in data["t_target2cam"]]

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )
    print("Calibration Result:")
    print("Rotation matrix (cam2gripper):\n", R_cam2gripper)
    print("Translation vector (cam2gripper):\n", t_cam2gripper)

if __name__ == "__main__":

    if not CALIBRATION:
        collection()
    else:
        calibration()