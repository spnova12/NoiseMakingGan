from os import listdir
from os.path import join
import os

from utils.ntirecamera import CameraInfo
from utils.transforms import *

import torch.utils.data as data
import torch


class DatasetForNoise(data.Dataset):
    """
    https://www.youtube.com/watch?v=zN49HdDxHi8 참고
    """
    def __init__(self, input_dir, target_dir, loader, patch_size):
        # input_dir : 아무 노이즈 없는 영상들 폴더 경로.
        # target_dir : 노이즈 자체.

        scene_instances_txt = "/home/lab/works/datasets/ssd2/ntire/Scene_Instances.txt"
        cam_info = CameraInfo(scene_instances_txt)
        self.camera_dict = cam_info.get_preprocessed_ntire_dict()

        self.input_filenames = [join(input_dir, x) for x in sorted(listdir(input_dir))]
        self.target_filenames = [join(target_dir, x) for x in sorted(listdir(target_dir))]
        self.loader = loader
        self.patch_size = patch_size

        # image 를 -1과 1사이로 노말라이즈 해준다.
        self.transform = Compose([
            RandomCrop(self.patch_size),
            Color0_255to1_1(),
            #RandomHorizontalFlip(),
            #RandomRotation90(),
            ToTensor()
        ])

    def __getitem__(self, index):
        input_name = self.input_filenames[index]
        target_name = self.target_filenames[index]

        input = self.loader(input_name)
        target = self.loader(target_name)

        # input, target 모두 tensor 로 바꿔준다.
        input, target = self.transform(input, target)

        # 노이즈 그 자체를 target 으로 하기 위해 빼주고.
        target = target - input

        # 카메라 정보 patch 를 생성한다.
        camera_info = self.camera_dict[str(os.path.basename(input_name)).split('_')[0]]
        smartphone_code = torch.full((1, self.patch_size, self.patch_size), camera_info[0])
        ISO_level = torch.full((1, self.patch_size, self.patch_size), camera_info[1])
        shutter_speed = torch.full((1, self.patch_size, self.patch_size), camera_info[2])

        # 카메라 정보를 하나의 tensor 로 합쳐준다.
        camera_info_tensor = torch.cat((smartphone_code, ISO_level, shutter_speed), 0)

        return input.float(), target.float(), camera_info_tensor.float()

    def __len__(self):
        return len(self.input_filenames)
