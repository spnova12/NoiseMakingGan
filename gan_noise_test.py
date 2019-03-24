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


scene_instances_txt = "Scene_Instances.txt"

# 1. Read text file containing ntire dataset's information
f = open(scene_instances_txt, 'r')
lines = f.readlines()
f.close()

# 2. Information(smartphone-code, iso, shutter speed) to list.
ntire_info_list = []
for line in lines:
    line = line.split('_')
    ntire_info_list.append((line[2], line[3], line[4]))

print(ntire_info_list)

phonecam = PhoneCam(scene_instances_txt, netG)

# This is folder directory containing clean images(GroundTruth)
test_imgs_folder_dir = 'flickr_test_imgs/GroundTruth'
test_imgs_dirs = [join(test_imgs_folder_dir, x) for x in sorted(listdir(test_imgs_folder_dir))]

"""
camera_info example : ('G4', 100, 60) # (smartphone-code, iso, shutter_speed)

Max iso : 10000
Max shutter_speed : 8460

GP: Google Pixel
IP: iPhone 7
S6: Samsung Galaxy S6 Edge
N6: Motorola Nexus 6
G4: LG G4
"""

# results will be saved in this directory.
folder_dir = make_dirs('flickr_test_imgs/Noisy_gan')


for test_img_dir in tqdm.tqdm(test_imgs_dirs):
    # pick camera information randomly.
    camera_info = random.choice(ntire_info_list)

    # with picked camera information, we add synthetic noises to clean images.
    test_output_img = phonecam.snap(test_img_dir, camera_info=camera_info)

    # save synthetic noisy images.
    cv2.imwrite(folder_dir + '/' + os.path.basename(test_img_dir), test_output_img)
