import json
import cv2
import numpy as np
import pyzed.sl as sl
from xarm.wrapper import XArmAPI


ROBOT_IP = "192.168.1.XXX"

PATTERN_SIZE = (6,7)
SQUARE_SIZE = 0.03

RESULT_FILE = "calib_dataset/handeye_result.json"


def load_handeye():
    data = json.load(open(RESULT_FILE))
    T = np.array(data["T_base_cam"]["matrix_4x4"])
    return T


def detect_board(frame, K, dist):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE)

    if not ret:
        return None

    objp = np.zeros((PATTERN_SIZE[0]*PATTERN_SIZE[1],3))
    objp[:,:2] = np.mgrid[0:PATTERN_SIZE[0],0:PATTERN_SIZE[1]].T.reshape(-1,2)
    objp *= SQUARE_SIZE

    ret, rvec, tvec = cv2.solvePnP(objp,corners,K,dist)

    if not ret:
        return None

    return rvec,tvec


def rvec_to_matrix(rvec,tvec):

    R,_ = cv2.Rodrigues(rvec)

    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = tvec.flatten()

    return T


def main():

    arm = XArmAPI(ROBOT_IP)
    arm.connect()

    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720

    zed.open(init)

    runtime = sl.RuntimeParameters()
    image = sl.Mat()

    T_base_cam = load_handeye()

    print("Loaded T_base_cam")
    print(T_base_cam)

    while True:

        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue

        zed.retrieve_image(image,sl.VIEW.LEFT)
        frame = image.get_data()

        frame = cv2.cvtColor(frame,cv2.COLOR_BGRA2BGR)

        result = detect_board(frame,None,None)

        if result is None:

            cv2.imshow("frame",frame)

            if cv2.waitKey(1)==27:
                break

            continue

        rvec,tvec = result

        T_cam_board = rvec_to_matrix(rvec,tvec)

        T_base_board = T_base_cam @ T_cam_board

        pos = T_base_board[:3,3]

        print("\nDetected board position in base frame (m):")
        print(pos)

        x = pos[0]*1000
        y = pos[1]*1000
        z = pos[2]*1000

        print("Robot target (mm):",x,y,z)

        cmd = input("Move robot to this position? (y/n) ")

        if cmd=="y":

            arm.set_position(
                x=x,
                y=y,
                z=z,
                roll=3.14,
                pitch=0,
                yaw=0,
                speed=50,
                wait=True
            )

        cv2.imshow("frame",frame)

        if cv2.waitKey(1)==27:
            break


if __name__ == "__main__":
    main()