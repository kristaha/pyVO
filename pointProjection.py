import numpy as np
import cv2

from typing import Tuple

fx = 520.9
fy = 521.0
cx = 325.1
cy = 249.7


def project_points(ids: np.ndarray, points: np.ndarray, depth_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects the 2D points to 3D using the depth image and the camera instrinsic parameters.
    :param ids: A N vector point ids.
    :param points: A 2xN matrix of 2D points
    :param depth_img: The depth image. Divide pixel value by 5000 to get depth in meters.
    :return: A tuple containing a N vector and a 3xN vector of all the points that where successfully projected.
    """


    #print(f"depth img shape: {depth_img.shape}")
    #print("points:")
    #print(points)

    x_values = []
    y_values = []
    z_values = []
    valid_point_ids = []

    window_size = 3

    for i in range(points.shape[1]):
        u = int(points[0][i])
        v = int(points[1][i])

        # Need to check if u,v in depth image is of ok value
        # Finds minimum value in window with center [v,u] 
        window_u_start_coordinate = u - window_size + 1
        window_u_end_coordinate = u + window_size
        window_v_start_coordinate = v - window_size + 1
        window_v_end_coordinate = v + window_size

        if u - window_size < 0:
            window_u_start_coordinate = 0
        elif u + window_size > depth_img.shape[1]:
            window_u_end_coordinate = depth_img.shape[1]

        if v - window_size < 0:
            window_v_start_coordinate = 0
        elif v + window_size > depth_img.shape[0]:
            window_v_end_coordinate = depth_img.shape[0]

        window = depth_img[window_u_start_coordinate : window_u_end_coordinate, 
                window_v_start_coordinate : window_v_end_coordinate]


        z = np.amin(window) / 5000

        print(f"z value = {z}")

        if z > 0:
            x = (u - cx)*z / fx
            y = (v - cy)*z / fy
            valid_point_ids.append(ids[i])
    
            x_values.append(x)
            y_values.append(y)
            z_values.append(z)
        

        return (np.asarray(valid_point_ids), np.asarray([x_values, y_values, z_values]))





    

