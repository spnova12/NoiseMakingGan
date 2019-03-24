import numbers
import numpy as np
import torch
import random

class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> Compose([
        >>>     CenterCrop(10),
        >>>     ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img1, img2=None):
        if img2 is None:
            for t in self.transforms:
                img1 = t(img1)
            return img1
        else:
            for t in self.transforms:
                img1, img2 = t(img1, img2)
            return img1, img2


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    default tensor type is (torch.float32).
    """
    def __call__(self, img1, img2=None):
        if len(img1.shape) == 2:  # this image is grayscale
            img1 = np.expand_dims(img1, axis=0)
        elif len(img1.shape) == 3:  # image is either RGB or YCbCr colorspace
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            img1 = img1.transpose((2, 0, 1))
        img1 = torch.from_numpy(img1.copy())
        if img2 is None:
            return img1
        else:
            if len(img2.shape) == 2:
                img2 = np.expand_dims(img2, axis=0)
            elif len(img2.shape) == 3:
                img2 = img2.transpose((2, 0, 1))
            img2 = torch.from_numpy(img2.copy())

            return img1, img2


class ToImage(object):
    """
    torch tensor 을 <class 'numpy.ndarray'> (cv2 에 사용되는 이미지 형태) 로 바꿔준다.
    또한 gpu 에서 cpu 로 데이터를 옮겨준다.
    tensor 가 batch 일 경우와 batch 가 아닐 때 모두 고려한다.
    """
    def __call__(self, input):
        input = input.cpu().numpy()
        if len(input.shape) == 4:  # batch tensor
            input = input.transpose((0, 2, 3, 1))
        elif len(input.shape) == 3:  # single tensor
            input = input.transpose((1, 2, 0))
        input = np.squeeze(input)
        return input


class CenterCrop(object):
    def __init__(self, output_size):
        if isinstance(output_size, numbers.Number):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, img1, img2=None):
        h, w = img1.shape[:2]
        new_h, new_w = self.output_size
        # 가끔 crop 하는 size 보다 워본 이미지 size 가 더 작아서 에러가 날 수 있다.
        top = int(round(h - new_h) * 0.5)
        left = int(round(w - new_w) * 0.5)
        if img2 is None:
            return img1[top: top + new_h, left: left + new_w]
        else:
            return img1[top: top + new_h, left: left + new_w], img2[top: top + new_h, left: left + new_w]


class RandomCrop(object):
    """
    Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, output_size):
        if isinstance(output_size, numbers.Number):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, img1, img2=None):
        h, w = img1.shape[:2]
        new_h, new_w = self.output_size
        # 가끔 crop 하는 size 보다 워본 이미지 size 가 더 작아서 에러가 날 수 있다.
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        if img2 is None:
            return img1[top: top + new_h, left: left + new_w]
        else:
            return img1[top: top + new_h, left: left + new_w], img2[top: top + new_h, left: left + new_w]


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5

    https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.fliplr.html#numpy.fliplr
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img1, img2):
        if random.random() < self.p:
            # Flip array in the left/right direction.
            return np.fliplr(img1), np.fliplr(img2)
        return img1, img2


class RandomRotation90(object):
    """Rotate the image by angle 90.
    https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663
    https://www.geeksforgeeks.org/numpy-rot90-python/
    """
    def __call__(self, img1, img2):
        # 0 -> 0도, 1 -> 90도, 2 -> 180도, 3 -> 270도
        angle = random.randint(0, 3)
        return np.rot90(img1, angle), np.rot90(img2, angle)


class Color0_255to1_1(object):
    """
    [0, 255] to [-1, 1]
    """
    def __call__(self, img1, img2=None):
        if img2 is None:
            return img1/127.5 - 1.

        else:
            return img1/127.5 - 1., img2/127.5 - 1.


class Color1_1to0_255(object):
    """
    [-1, 1] to [0, 255]
    """
    def __call__(self, img):
        return (img + 1.) * 127.5


class Color0_255to0_1(object):
    """
    [0, 255] to [-1, 1]
    """
    def __call__(self, img1, img2=None):
        if img2 is None:
            return img1/255.

        else:
            return img1/255., img2/255 - 1.


class Color0_1to0_255(object):
    """
    [-1, 1] to [0, 255]
    """
    def __call__(self, img):
        return img * 255.


class MergeNP(object):
    """
    batch 를 타일형태의 한장의 이미지로 만들어준다.
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, images):
        """
        :param images: input image 의 shape 은 (batch, h, w, channel) 이다.
        :param size: (a, b) 형태의 튜플 a = 세로 타일 개수, b = 가로 타일 개수.
        :return: color 일 경우 (h, w, 3), 흑백일 경우 (h, w) 인 한장의 이미지.
        """
        h, w = images.shape[1], images.shape[2]
        # color
        if len(images.shape) == 4:
            c = images.shape[3]
            img = np.zeros((h * self.size[0], w * self.size[1], c))
            for idx, image in enumerate(images):
                i = idx % self.size[1]  # 나누기 연산 후 몫이 아닌 나머지를 구함
                j = idx // self.size[1]  # 나누기 연산 후 소수점 이하의 수를 버리고, 정수 부분의 수만 구함
                img[j * h:j * h + h, i * w:i * w + w, :] = image
            return img
        # gray scale
        elif len(images.shape) == 3:
            img = np.zeros((h * self.size[0], w * self.size[1]))
            for idx, image in enumerate(images):
                i = idx % self.size[1]
                j = idx // self.size[1]
                img[j * h:j * h + h, i * w:i * w + w] = image[:, :]
            return img
        else:
            raise ValueError('in merge(images,size) images parameter '
                             'must have dimensions: HxW or HxWx3 or HxWx4')
