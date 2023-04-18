from CarlaApiAsync import CarlaApi
import cv2
import numpy as np
import os


# 讀取資料夾下照片名稱，並返回最大值
def get_image_name(dir_path):
    max_num = 0

    for file_name in os.listdir(dir_path):
        file_name = int(file_name.strip('.jpg'))
        if file_name > max_num:
            max_num = file_name

    return max_num


class Main:
    def __init__(self, rgb_camera=[], seg_camera=[], depth_camera=[]):
        """
        輸入為字典型態, ex:攝影機的名字、角度、位置、寬、高. ex:
            rgb_camera = [{name:top, pitch:0, roll:0, yaw:0, x:0, y:0, z:0, width:640, height:480}]
        :param rgb_camera:
        :param seg_camera:
        :param depth_camera:
        """
        # 建立Carla Api並初始化
        self.carla_api = CarlaApi(rgb_camera, seg_camera, depth_camera)
        self.carla_api.initial_sim()
        self.save_img()

    def save_img(self):
        while True:
            pass

rgb_camera = [
    {'name': '0_degree_rgb', 'path': '0_degree_rgb/', 'pitch': 0, 'roll': 0, 'yaw': 0, 'x': 1.5, 'y': 0, 'z': 2.4,
     'width': 200, 'height': 200},
    {'name': '60_degree_rgb', 'path': '60_degree_rgb/', 'pitch': 0, 'roll': 0, 'yaw': 60, 'x': 0.75, 'y': 1.29,
     'z': 2.4, 'width': 200, 'height': 200},
    {'name': '120_degree_rgb', 'path': '120_degree_rgb/', 'pitch': 0, 'roll': 0, 'yaw': 120, 'x': -0.75, 'y': 1.29,
     'z': 2.4, 'width': 200, 'height': 200},
    {'name': '180_degree_rgb', 'path': '180_degree_rgb/', 'pitch': 0, 'roll': 0, 'yaw': 180, 'x': -1.5, 'y': 0,
     'z': 2.4, 'width': 200, 'height': 200},
    {'name': '240_degree_rgb', 'path': '240_degree_rgb/', 'pitch': 0, 'roll': 0, 'yaw': 240, 'x': -0.75, 'y': -1.29,
     'z': 2.4, 'width': 200, 'height': 200},
    {'name': '300_degree_rgb', 'path': '300_degree_rgb/', 'pitch': 0, 'roll': 0, 'yaw': 300, 'x': 0.75, 'y': -1.29,
     'z': 2.4, 'width': 200, 'height': 200}
]

depth_camera = [
    {'name': '0_degree_depth', 'path': '0_degree_depth/', 'pitch': 0, 'roll': 0, 'yaw': 0, 'x': 1.5, 'y': 0, 'z': 2.4,
     'width': 200, 'height': 200},
    {'name': '60_degree_depth', 'path': '60_degree_depth/', 'pitch': 0, 'roll': 0, 'yaw': 60, 'x': 0.75, 'y': 1.29,
     'z': 2.4, 'width': 200, 'height': 200},
    {'name': '120_degree_depth', 'path': '120_degree_depth/', 'pitch': 0, 'roll': 0, 'yaw': 120, 'x': -0.75, 'y': 1.29,
     'z': 2.4, 'width': 200, 'height': 200},
    {'name': '180_degree_depth', 'path': '180_degree_depth/', 'pitch': 0, 'roll': 0, 'yaw': 180, 'x': -1.5, 'y': 0,
     'z': 2.4, 'width': 200, 'height': 200},
    {'name': '240_degree_depth', 'path': '240_degree_depth/', 'pitch': 0, 'roll': 0, 'yaw': 240, 'x': -0.75, 'y': -1.29,
     'z': 2.4, 'width': 200, 'height': 200},
    {'name': '300_degree_depth', 'path': '300_degree_depth/', 'pitch': 0, 'roll': 0, 'yaw': 300, 'x': 0.75, 'y': -1.29,
     'z': 2.4, 'width': 200, 'height': 200}
]

seg_camera = [
    {'name': 'top_view', 'path': 'top_down_view_seg/', 'pitch': -90, 'roll': 0, 'yaw': 0, 'x': 0, 'y': 0, 'z': 7, 'width': 200, 'height': 200}]


m = Main(rgb_camera=rgb_camera, seg_camera=seg_camera, depth_camera=depth_camera)
