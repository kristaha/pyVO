import cv2
import numpy as np
from skimage import measure 
from skimage.feature import peak_local_max
from operator import itemgetter
from typing import Tuple, List

def harris_corners(img: np.ndarray, threshold=4.5, blur_sigma=2.0) -> List[Tuple[float, np.ndarray]]:
    """
    Return the harris corners detected in the image.
    :param img: The grayscale image.
    :param threshold: The harris respnse function threshold.
    :param blur_sigma: Sigma value for the image bluring.
    :return: A sorted list of tuples containing response value and image position.
    The list is sorted from largest to smallest response value.
    """

    # Finding gradients in x and y direction for input image using Scharr operator
    dx = cv2.Scharr(img, cv2.CV_64FC1, 1, 0)
    dy = cv2.Scharr(img, cv2.CV_64FC1, 0, 1)

    # Determine matrix M = (dx^2 dx*dy, dx*dy dy^2)
    dx2 = dx*dx
    dxdy =dx*dy
    dy2 = dy*dy

    # Applies Gaussian Blur to reduce noice
    dx2 = cv2.GaussianBlur(dx2, None, sigmaX=blur_sigma, sigmaY=blur_sigma)
    dxdy = cv2.GaussianBlur(dxdy, None, sigmaX=blur_sigma, sigmaY=blur_sigma)
    dy2 = cv2.GaussianBlur(dy2, None, sigmaX=blur_sigma, sigmaY=blur_sigma)
    
    # Finding response image f = det(M) / trace(M)
    epsilon = 1e-8 # Added when finding response to avoid zero-devision
    f = np.divide(dx2*dy2 - dxdy*dxdy, dx2 + dy2 + epsilon)

    # Filter scores below threshold
    f[f < threshold] = 0

    # Non-maximum suppression 
    window_size = 10
    max_indices = peak_local_max(f, window_size)

    max_values = []
    for e in max_indices:
        row, col = e
        #max_values.append((f[row][col], np.asarray([row, col])))
        max_values.append((f[row][col], np.asarray([col, row])))
        img_copy = img
        cv2.circle(img_copy,(int(col),int(row)),3,(100,100,100),-1) # draw circle
    cv2.imshow("harris_corners.png", img_copy)
    cv2.waitKey(1)

    return sorted(max_values, key=itemgetter(0), reverse=True)

