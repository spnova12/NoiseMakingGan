import time
import tqdm

from utils import *
from utils.eval import LogCSV, TorchPaddingForOdd
from utils.ntirecamera import PhoneCam

from models.base import *
from models.subNets import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import random

device = torch.device('cuda:0')

checkpoint_dir = 'noiseG.pkl'
checkpoint = torch.load(checkpoint_dir)
netG = Generator().to(device)
netG.load_state_dict(checkpoint['netG'])

# training 하는 도중에 eval 을 해주기 위해 camera model 을 생성해준다.
scene_instances_txt = "/home/lab/works/datasets/ssd2/ntire/Scene_Instances.txt"

# 1. 먼저 ntire dataset 의 정보가 담긴 text 파일을 읽는다.
f = open(scene_instances_txt, 'r')
lines = f.readlines()
f.close()
# 2. 읽은 파일에서 폰 종류, iso, shutter speed 만 파싱해서 list 로 만들어준다.
#    list 에서 random 으로 조건을 뽑아서 사용할 수 있다.
ntire_info_list = []
for line in lines:
    line = line.split('_')
    ntire_info_list.append((line[2], line[3], line[4]))

print(ntire_info_list)



phonecam = PhoneCam(scene_instances_txt, netG)

test_imgs_folder_dir = '/home/lab/works/datasets/ssd2/flickr/validation/GroundTruth'
test_imgs_dirs = [join(test_imgs_folder_dir, x) for x in sorted(listdir(test_imgs_folder_dir))]

"""
camera_info 예시 : ('G4', 100, 60) # (기종, iso, shutter_speed)

iso 최대는 10000
shutter_speed 최대는 8460

GP: Google Pixel
IP: iPhone 7
S6: Samsung Galaxy S6 Edge
N6: Motorola Nexus 6
G4: LG G4
"""

folder_dir = make_dirs('/home/lab/works/datasets/ssd2/flickr/validation/Noisy_gan')


for test_img_dir in tqdm.tqdm(test_imgs_dirs):
    camera_info = random.choice(ntire_info_list)
    test_output_img = phonecam.snap(test_img_dir, camera_info=camera_info)
    cv2.imwrite(folder_dir + '/' + os.path.basename(test_img_dir), test_output_img)
