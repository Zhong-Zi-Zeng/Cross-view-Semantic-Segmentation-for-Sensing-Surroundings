from Network.VPN import VPN
import numpy as np
import tensorflow as tf
import cv2
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# 全部的類別
AllClass = {
    '0': (0, 0, 0),  # 未標記
    '1': (70, 70, 70),  # 建築
    '2': (40, 40, 100),  # 柵欄
    '3': (80, 90, 55),  # 其他
    '4': (60, 20, 220),  # 行人
    '5': (153, 153, 153),  # 桿
    '6': (50, 234, 157),  # 道路線
    '7': (128, 64, 128),  # 馬路
    '8': (232, 35, 244),  # 人行道
    '9': (35, 142, 107),  # 植披
    '10': (142, 0, 0),  # 汽車
    '11': (156, 102, 102),  # 牆
    '12': (0, 220, 220),  # 交通號誌
    '13': (180, 130, 70),  # 天空
    '14': (81, 0, 81),  # 地面
    '15': (100, 100, 150),  # 橋
    '16': (140, 150, 230),  # 鐵路
    '17': (180, 165, 180),  # 護欄
    '18': (30, 170, 250),  # 紅綠燈
    '19': (160, 190, 110),  # 靜止的物理
    '20': (50, 120, 170),  # 動態的
    '21': (150, 60, 45),  # 水
    '22': (100, 170, 145)  # 地形
}

# 感興趣的類別
InterestClass = {
    '0': (0, 0, 0),  # 未標記
    '1': (50, 234, 157),  # 道路線
    '2': (128, 64, 128),  # 馬路
    '3': (232, 35, 244),  # 人行道
    '4': (70, 70, 70),  # 建築
    '5': (142, 0, 0),  # 汽車
}

# 總共有幾類
CLASS_NUM = len(InterestClass.items())

# 讀取訓練資料
rgb_img_dir = ['./0_degree_rgb', './60_degree_rgb', './120_degree_rgb', './180_degree_rgb', './240_degree_rgb',
               './300_degree_rgb']
depth_img_dir = ['./0_degree_depth', './60_degree_depth', './120_degree_depth', './180_degree_depth',
                 './240_degree_depth', './300_degree_depth']
seg_img_dir = './top_down_view_seg'
file_name = os.listdir(seg_img_dir)
file_name.sort(key=lambda x: int(x[:-4]))

# 從資料夾裡讀取訓練圖片
def get_image(img_file):
    rgb_img_list = []
    depth_img_list = []

    # 讀取rgb影像
    for rgb_dir in rgb_img_dir:
        img = cv2.imread(rgb_dir + '/' + img_file)
        rgb_img_list.append(img)

    # 讀取depth影像
    for depth_dir in depth_img_dir:
        img = cv2.imread(depth_dir + '/' + img_file)
        depth_img_list.append(img)

    # 讀取seg影像
    seg_img = cv2.imread(seg_img_dir + '/' + img_file)

    return np.array(rgb_img_list), np.array(depth_img_list), seg_img


def parseOutput(output):
    result_img = np.zeros((200, 200, 3), dtype=np.uint8)
    hot_code = np.argmax(output, axis=2)

    for label in range(CLASS_NUM):
        matrix = np.where((hot_code[:, :] == label), np.ones((200, 200)), np.zeros((200, 200)))
        result_img[matrix > 0] = InterestClass[str(label)]

    return result_img


model = VPN(num_class=CLASS_NUM, V=6, M=2)
model.build(input_shape=(1, 12, 200, 200, 3))
model.load_weights('./weight/vpn_weight.h5')

for img_name in file_name:
    rgb_img, depth_img, seg_img = get_image(img_name)
    x = np.concatenate([rgb_img, depth_img], axis=0, dtype=np.float64)
    x /= 255.
    x = np.array([x])
    x = tf.convert_to_tensor(x)

    pre = model(x)[0]
    output = parseOutput(pre)

    cv2.imshow("", output)
    cv2.imshow("label", seg_img)
    cv2.waitKey(1)
