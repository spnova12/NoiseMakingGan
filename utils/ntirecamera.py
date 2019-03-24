
from utils import *
from utils.eval import LogCSV, TorchPaddingForOdd
from utils.utils import *
from utils.transforms import *
import torch

class CameraInfo(object):
    """
    ntire 에서 제공한 "scene_instances_txt" 을 읽고,

    <scene-instance-number>
    <scene_number>
    <smartphone-code> 사용.
    <ISO-level> 사용.
    <shutter-speed> 사용.
    <illuminant-temperature>
    <illuminant-brightness-code>

    중에서 <smartphone-code>, <ISO-level>, <shutter-speed> 정보를 추출해서
    <scene-instance-number> 를 input 으로 받으면 세 정보를 return 해주는 dict 를 만든다.
    """
    def __init__(self, scene_instances_txt):
        # scene instances 정보를 list 로 옮겨준다.
        f = open(scene_instances_txt, 'r')
        lines = f.readlines()
        f.close()

        # 최대 iso level, shutter_speed 를 찾는다.
        iso_level_list = []
        shutter_speed_list = []

        for line in lines:
            line = line.split('_')
            iso_level_list.append(float(line[3]))
            shutter_speed_list.append(float(line[4]))

        self.max_iso_level = max(iso_level_list)
        self.max_shutter_speed = max(shutter_speed_list)

        self.phone_dict = {'GP': 0 / 4, 'IP': 1 / 4, 'S6': 2 / 4, 'N6': 3 / 4, 'G4': 4 / 4}

        self.preprocessed_ntire_dict = {}

        # "scene_instances_txt" 를 읽어가면서 정보들을 pre-processing 한 후 "scene instance number" 에 맞게 dict 에 넣어준다.
        for line in lines:
            line = line.split('_')
            scene_instance_number = line[0]
            self.preprocessed_ntire_dict[scene_instance_number] = self.cam_info_preprocessing((line[2], line[3], line[4]))

    def get_preprocessed_ntire_dict(self):
        return self.preprocessed_ntire_dict

    def cam_info_preprocessing(self, cam_info):
        smartphone_code, ISO_level, shutter_speed = cam_info
        smartphone_code = self.phone_dict[smartphone_code]  # 폰 정보는 미리 정해둔 dict 를 이용해서 숫자로 변환.
        ISO_level = float(ISO_level) / self.max_iso_level  # 최댓값으로 나눠줌.
        shutter_speed = float(shutter_speed) / self.max_shutter_speed  # 최댓값으로 나눠줌.

        return smartphone_code, ISO_level, shutter_speed


class PhoneCam(object):
    def __init__(self, scene_instances_txt, netG, crop_size=None):

        self.cam_info = CameraInfo(scene_instances_txt)
        self.preprocessed_dict = self.cam_info.get_preprocessed_ntire_dict()

        self.crop_size = crop_size
        self.device = torch.device('cuda:0')
        self.netG = netG
        self.img_loader = load_BGR

        # image 를 -1과 1사이로 노말라이즈 해준다.
        if crop_size:
            self.transform_for_eval = Compose([
                CenterCrop(self.crop_size),
                Color0_255to1_1(),
                ToTensor()
            ])
        else:
            self.transform_for_eval = Compose([
                Color0_255to1_1(),
                ToTensor()
            ])

        # 모델의 출력(fake camera noise) 은 임의의 범위가 될 것이기 때문에 따로 normalization 을 해주지 않았다.
        self.to_image = ToImage()

    # take_pictures_with_phone
    def snap(self, pure_img_dir, camera_info=None):
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

        # 학습을 위해서 카메라 info 들을 normalization 해줘야 한다.
        if camera_info:
            # camera_info 가 입력되면 cam_info_preprocessing 로 직접 norm 해준다.
            preprocessed_camera_info = self.cam_info.cam_info_preprocessing(camera_info)
        else:
            # camera info 가 none 이라면 pure_img_dir 은 ntire 데어터 셋의 형식이여야 한다. 자동으로 norm 된 info 불러온다.
            scene_instance_number = str(os.path.basename(pure_img_dir)).split('_')[0]  # 이름 파싱
            preprocessed_camera_info = self.preprocessed_dict[scene_instance_number]

        # 이미지를 읽는다.
        input_img = self.img_loader(pure_img_dir)


        if self.crop_size:
            self.crop_size_h = self.crop_size
            self.crop_size_w = self.crop_size
        else:
            self.crop_size_h = input_img.shape[0]
            self.crop_size_w = input_img.shape[1]


        # 카메라 정보 patch 를 tensor 로 만들어준다.
        smartphone_code = torch.full((1, 1, self.crop_size_h, self.crop_size_w), preprocessed_camera_info[0])
        ISO_level = torch.full((1, 1, self.crop_size_h, self.crop_size_w), preprocessed_camera_info[1])
        shutter_speed = torch.full((1, 1, self.crop_size_h, self.crop_size_w), preprocessed_camera_info[2])

        # seed random 값을 만들어준다.
        z = torch.randn(1, 1, self.crop_size_h, self.crop_size_w).float()

        # 카메라 정보를 하나의 tensor 로 합쳐준다.
        camera_info_tensor = torch.cat((smartphone_code, ISO_level, shutter_speed), 1)



        input_tensor = self.transform_for_eval(input_img).float().unsqueeze_(0).to(self.device)

        input_tensor_temp = input_tensor.clone()

        padder = TorchPaddingForOdd()
        input_tensor_temp = padder.padding(input_tensor_temp)
        z_temp = padder.padding(z)
        camera_info_tensor_temp = padder.padding(camera_info_tensor)

        # fake camera noise 생성.
        with torch.no_grad():
            self.netG.eval()
            output_tensor_noise = self.netG(z_temp.to(self.device), camera_info_tensor_temp.to(self.device), input_tensor_temp.to(self.device))
            output_tensor_noise = padder.unpadding(output_tensor_noise)

            # 생성된 fake camera noise (tensor) 를 깨끗한 tensor 에 더해준다.
            output_tensor_img = input_tensor + output_tensor_noise

        # tensor 에서 image 로 전환.
        output_img = self.to_image(output_tensor_img)
        # output_noise = self.to_image(output_tensor_noise)

        # 거꾸로 norm.
        output_img = (output_img + 1.) * 127.5
        # output_noise = output_noise * 127.5

        # 실제 이미지는 8bit 로 저장하므로 아래와 같이 처리해준다.
        output_img = np.around(output_img)
        output_img = output_img.clip(0, 255)
        output_img = output_img.astype(np.uint8)

        return output_img



