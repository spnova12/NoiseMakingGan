import os
import cv2


def load_BGR(filepath):
    img_BGR = cv2.imread(filepath, cv2.IMREAD_COLOR)
    return img_BGR


def load_grayscale(filepath):
    img_grayscale = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return img_grayscale


def make_dirs(path):
    """
    경로(폴더) 가 있음을 확인하고 없으면 새로 생성한다.
    :param path: 확인할 경로
    :return: path
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path
