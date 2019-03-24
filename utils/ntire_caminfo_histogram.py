from os import listdir
from os.path import join
import os
import random


###########################
# text file 을 읽는다.
###########################

scene_instances_txt = "/home/lab/works/datasets/ssd2/ntire/Scene_Instances.txt"
f = open(scene_instances_txt, 'r')
lines = f.readlines()
f.close()


###########################
# ntire 에서 생성한 데이터들의 histogram 을 구한다.
###########################

phone_count = {'GP': 0, 'IP': 0, 'S6': 0, 'N6': 0, 'G4': 0}
iso_count = {}
shutter_speed_count = {}

for i, line in enumerate(lines):
    line = line.split('_')
    scene_instance_number = line[0]

    phone_count[line[2]] += 1

    if line[3] in iso_count.keys():
        iso_count[line[3]] += 1
    else:
        iso_count[line[3]] = 1

    if line[4] in shutter_speed_count.keys():
        shutter_speed_count[line[4]] += 1
    else:
        shutter_speed_count[line[4]] = 1

    #print(i, (line[2], line[3], line[4]))

print(phone_count)
print(iso_count)
print(shutter_speed_count)

print('####################################')
###########################
# 위와는 별개로 그냥 ntire 데이터 셋 정보를 그냥 랜덤으로 뽑을 수 있게 한다.
###########################

ntire_info_list = []
for line in lines:
    line = line.split('_')
    ntire_info_list.append((line[2], line[3], line[4]))

print(random.choice(ntire_info_list))

