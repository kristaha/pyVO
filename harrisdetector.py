import cv2
import numpy as np
from skimage import measure 
from operator import itemgetter
from typing import Tuple, List


def harris_corners(img: np.ndarray, threshold=1e-2, blur_sigma=2.0) -> List[Tuple[float, np.ndarray]]:
    """
    Return the harris corners detected in the image.
    :param img: The grayscale image.
    :param threshold: The harris respnse function threshold.
    :param blur_sigma: Sigma value for the image bluring.
    :return: A sorted list of tuples containing response value and image position.
    The list is sorted from largest to smallest response value.
    """
    # Applies Gaussian Blur to reduce noice in image
    img = cv2.GaussianBlur(img, ksize=(3,3), sigmaX=blur_sigma, 
            sigmaY=blur_sigma, borderType=cv2.BORDER_DEFAULT)

    # Finding gradients in x and y direction for input image using Scharr operator
    x_gradient = np.zeros(img.shape)
    y_gradient = np.zeros(img.shape)
    x_gradient = cv2.Scharr(img, ddepth=-1, dx=1, dy=0, scale=1,
            delta=0, borderType=cv2.BORDER_DEFAULT)
    y_gradient = cv2.Scharr(img, ddepth=-1, dx=0, dy=1, scale=1,
            delta=0, borderType=cv2.BORDER_DEFAULT)


    # Determine matrix M = (A B, B C)
    a_matrix = np.zeros(img.shape)
    b_matrix = np.zeros(img.shape)
    c_matrix = np.zeros(img.shape)

    a_matrix = x_gradient * x_gradient
    b_matrix = x_gradient * y_gradient
    c_matrix = y_gradient * y_gradient

    # Finding f = det(M) - alpha*trace(M)
    alpha = 0.06
    epsilon = 1e-8
    f = np.zeros(img.shape)
    f = (a_matrix*c_matrix - b_matrix*b_matrix) - alpha*(a_matrix + c_matrix)**2 
    #f = (a_matrix*c_matrix - b_matrix*b_matrix) / (a_matrix + c_matrix + epsilon)
    print("f values 1 " + str(f))

    # Filter scores below threshold and nan's
    #is_nan_indices = np.where(np.isnan(f))
    #f[is_nan_indices] = 0
    below_threshold_indices = f < 0#threshold
    f[below_threshold_indices] = 0
    #print("f values 3 " + str(f))

    # Non-maximum suppression - here same as doing max pooling
    window_size = 4
    max_values = measure.block_reduce(f, (window_size, window_size), np.max).flatten()
    max_indices = np.where(np.greater(max_values, 0))[0]
    max_values = max_values[max_indices]
    print("Max indices " + str(max_indices))
    print("Max values " + str(max_values))

    returnList = zip(max_values, max_indices)

    return sorted(returnList, key=itemgetter(0), reverse=True)





