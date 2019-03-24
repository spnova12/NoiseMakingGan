from utils.transforms import *
from math import log10
import csv

import os


def psnr(img1, img2):
    """
    :param img1, img2: numpy uint8 img
    """
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255
    return 10 * log10((PIXEL_MAX ** 2) / mse)


class LogCSV(object):
    def __init__(self, log_dir):
        """
        :param log_dir: log(csv 파일) 가 저장될 dir
        """
        self.log_dir = log_dir
        f = open(self.log_dir, 'a')
        f.close()

    def make_head(self, header):
        with open(self.log_dir, "a") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow(header)

    def __call__(self, log):
        """
        :param log: header 의 각 항목에 해당하는 값들의 list
        """
        with open(self.log_dir, "a") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow(log)


class TorchPaddingForOdd(object):
    """
    OneToMany 의 경우 Down-Sampling 하는 층이 있다. 이때 1/2 크기로 Down-Sampling 하는데
    이미지 사이즈가 홀수이면 2로 나눌 수 없기 때문에 일시적으로 padding 을 하여 짝수로 만들어 준 후 모델을 통과시키고,
    마지막으로 unpadding 을 하여 원래 이미지 크기로 만들어준다.
    """
    def __init__(self):
        self.is_height_even = True
        self.is_width_even = True

    def padding(self, img):
        # 홀수면 패딩을 체워주는 것을 해주자
        if img.shape[2] % 2 != 0:
            self.is_height_even = False
            img_ = torch.zeros(img.shape[0], img.shape[1], img.shape[2] + 1, img.shape[3])
            img_[:img.shape[0], :img.shape[1], :img.shape[2], :img.shape[3]] = img
            img_[:img.shape[0], :img.shape[1], img.shape[2], :img.shape[3]] = img_[:img.shape[0], :img.shape[1],
                                                                              img.shape[2] - 1, :img.shape[3]]
            img = img_
        if img.shape[3] % 2 != 0:
            self.is_width_even = False
            img_ = torch.zeros(img.shape[0], img.shape[1], img.shape[2], img.shape[3] + 1)
            img_[:img.shape[0], :img.shape[1], :img.shape[2], :img.shape[3]] = img
            img_[:img.shape[0], :img.shape[1], :img.shape[2], img.shape[3]] = img_[:img.shape[0], :img.shape[1],
                                                                              :img.shape[2], img.shape[3] - 1]
            img = img_
        return img

    def unpadding(self, img):
        # 홀수였으면 패딩을 제거하는 것을 해주자
        if not self.is_height_even:
            img.data = img.data[:img.shape[0], :img.shape[1], :img.shape[2] - 1, :img.shape[3]]
        if not self.is_width_even:
            img.data = img.data[:img.shape[0], :img.shape[1], :img.shape[2], :img.shape[3] - 1]
        return img

