import cv2
import numpy as np
from operator import itemgetter
from typing import Tuple, List


def harris_corners(img: np.ndarray, threshold=1.0, blur_sigma=2.0) -> List[Tuple[float, np.ndarray]]:
    """
    Return the harris corners detected in the image.
    :param img: The grayscale image.
    :param threshold: The harris respnse function threshold.
    :param blur_sigma: Sigma value for the image bluring.
    :return: A sorted list of tuples containing response value and image position.
    The list is sorted from largest to smallest response value.
    """
    # Applies Gaussian Blur to reduce noice in image
    img = cv2.GaussianBlur(img, ksize=3, sigmaX=blur_sigma, sigmaY=blur_sigma, BORDER_DEFAULT)

    # Finding gradients in x and y direction for input image using Scharr operator
    x_gradient = np.zeros(img.shape)
    y_gradient = np.zeros(img.shape)
    x_gradient = cv2.Scharr(img, ddepth=-1, dx=1, dy=0, scale=1, delta=0, BORDER_DEFAULT)
    y_gradient = cv2.Scharr(img, ddepth=-1, dx=0, dy=1, scale=1, delta=0, BORDER_DEFAULT)


    # Determine matrix M = (A B, B C)
    a_matrix = np.zeros(img.shape)
    b_matrix = np.zeros(img.shape)
    c_matrix = np.zeros(img.shape)

    a_matrix = x_gradient * x_gradient
    b_matrix = x_gradient * y_gradient
    c_matrix = y_gradient * y_gradient

    #TODO Hvordan sette sammen A,B,C til M?


    # Finding f = det(M) - alpha*trace(M)

    raise NotImplementedError

