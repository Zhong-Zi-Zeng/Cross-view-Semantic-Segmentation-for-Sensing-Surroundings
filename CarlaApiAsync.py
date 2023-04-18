import carla
import numpy as np
import random
from collections import deque

file_name = 307


# ============處理bgr影像============
def process_rgb_frame(bgr_frame):
    bgr_frame.convert(carla.ColorConverter.Raw)
    array = np.frombuffer(bgr_frame.raw_data, dtype=np.uint8)
    array = np.reshape(array, (bgr_frame.height, bgr_frame.width, 4))
    bgr_frame = array[:, :, :3]

    return bgr_frame


# ============處理seg影像============
def process_seg_frame(seg_frame):
    seg_frame.convert(carla.ColorConverter.CityScapesPalette)
    array = np.frombuffer(seg_frame.raw_data, dtype=np.uint8)
    array = np.reshape(array, (seg_frame.height, seg_frame.width, 4))
    seg_frame = array[:, :, :3]

    return seg_frame


# ============處理depth影像============
def process_depth_frame(depth_frame):
    depth_frame.convert(carla.ColorConverter.Depth)
    array = np.frombuffer(depth_frame.raw_data, dtype=np.uint8)
    array = np.reshape(array, (depth_frame.height, depth_frame.width, 4))
    depth_frame = array[:, :, :3]

    return depth_frame


class CarlaApi:
    def __init__(self, rgb_camera, seg_camera, depth_camera):
        # 變數設置
        self.world = None
        self.map = None
        self.blueprint_library = None
        self.vehicle = None
        self.vehicle_transform = None

        # 攝影機變數
        self.rgb_camera = rgb_camera
        self.seg_camera = seg_camera
        self.depth_camera = depth_camera

        self.actor_list = []
        self.camera_list = []
        self.camera_queue_list = []
        self.camera_info_queue = deque(maxlen=5)

    """初始化虛擬環境"""

    def initial_sim(self):
        self._connect_to_world()
        self._spawn_vehicle(AutoMode=True)
        self._spawn_camera(self.rgb_camera, type='sensor.camera.rgb')
        self._spawn_camera(self.seg_camera, type='sensor.camera.semantic_segmentation')
        self._spawn_camera(self.depth_camera, type='sensor.camera.depth')
        self._build_queue()
        self.world.on_tick(self._callback)

    """連接到模擬環境"""

    def _connect_to_world(self):
        print('Connect to world....')
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        self.world = client.get_world()
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()

    """取出佇列資料"""

    def _pop_queue(self, Q):
        while True:
            if len(Q):
                return Q.pop()

    """on_tick的回調函式"""

    def _callback(self, WorldSnapshot):
        self._camera_callback()

    """相機回調函式"""

    def _camera_callback(self):
        global file_name
        file_name += 1

        for camera_queue, camera_name, camera_path, type in self.camera_queue_list:
            data = self._pop_queue(camera_queue)

            if type == 'sensor.camera.rgb':
                data.save_to_disk(camera_path + str(file_name) + '.jpg',
                                  color_converter=carla.ColorConverter.Raw)
            elif type == 'sensor.camera.semantic_segmentation':
                data.save_to_disk(camera_path + str(file_name) + '.jpg',
                                  color_converter=carla.ColorConverter.CityScapesPalette)
            elif type == 'sensor.camera.depth':
                data.save_to_disk(camera_path + str(file_name) + '.jpg',
                                  color_converter=carla.ColorConverter.Depth)

    """等待模擬開始"""

    def wait_for_sim(self):
        self.world.wait_for_tick()

    """產生車輛"""

    def _spawn_vehicle(self, AutoMode):
        self.vehicle_transform = random.choice(self.world.get_map().get_spawn_points())
        vehicle_bp = self.blueprint_library.filter('vehicle')[0]
        self.vehicle = self.world.spawn_actor(vehicle_bp, self.vehicle_transform)
        self.vehicle.set_autopilot(AutoMode)
        self.actor_list.append(self.vehicle)

    """產生攝影機到車輛上"""

    def _spawn_camera(self, camera_list, type):
        for camera in camera_list:
            camera_bp = self.blueprint_library.find(type)
            camera_bp.set_attribute('image_size_x', str(camera['width']))
            camera_bp.set_attribute('image_size_y', str(camera['height']))
            camera_trans = carla.Transform(
                carla.Location(x=int(camera['x']), y=int(camera['y']), z=int(camera['z'])),
                carla.Rotation(pitch=int(camera['pitch']), roll=int(camera['roll']), yaw=int(camera['yaw']))
            )
            cam = self.world.spawn_actor(camera_bp, camera_trans, attach_to=self.vehicle)
            self.camera_list.append([cam, camera['name'], camera['path'], type])

    """建立佇列"""
    def _build_queue(self):
        for camera, camera_name, camera_path, type in self.camera_list:
            Q = deque(maxlen=5)
            camera.listen(Q.append)
            self.camera_queue_list.append([Q, camera_name, camera_path, type])

    """銷毀生成物件"""

    def destroy(self):
        for actor in self.actor_list:
            actor.destory()



