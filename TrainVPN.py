from Network.VPN import VPN
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
import tensorflow as tf
import numpy as np
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
np.random.shuffle(file_name)


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


# 訓練次數
EPOCH = 200

# 訓練批次
BATCH_SIZE = 2

# 學習率
LR = 0.001

# 生成器
def generate(data, batch_size):
    i = 0
    n = len(data)

    while True:
        X_train = []
        Y_train = []
        for _ in range(batch_size):
            # 圖片名稱
            img_name = data[i]

            # 讀取訓練照片 shape (12, 200, 200, 3)
            rgb_img, depth_img, seg_img = get_image(img_name)
            x = np.concatenate([rgb_img, depth_img], axis=0, dtype=np.float64)

            # 訓練照片遇處理
            x /= 255.
            X_train.append(x)

            # Y_train圖片預處理
            seg_img_height = seg_img.shape[0]
            seg_img_width = seg_img.shape[1]
            label_seg = np.zeros((seg_img_height, seg_img_width, CLASS_NUM))

            for index, obj_bgr in enumerate(AllClass.values()):
                b = obj_bgr[0]
                g = obj_bgr[1]
                r = obj_bgr[2]
                matrix = np.where((seg_img[:, :, 0] == b) & (seg_img[:, :, 1] == g) & (seg_img[:, :, 2] == r),
                                  np.ones((seg_img_height, seg_img_width)), np.zeros((seg_img_height, seg_img_width)))

                if obj_bgr in InterestClass.values() and index != 0:
                    channel = list(InterestClass.values()).index(obj_bgr)
                    label_seg[:, :, channel] = matrix
                else:
                    label_seg[:, :, 0] += matrix
            
            Y_train.append(label_seg)
            i = (i + 1) % n

        yield (np.array(X_train), np.array(Y_train))


model = VPN(num_class=CLASS_NUM, V=6, M=2)
model.build(input_shape=(1, 12, 200, 200, 3))
model.load_weights("./weight/vpn_weight.h5")

model.compile(loss="categorical_crossentropy", metrics=['acc'])
opt = Adam(learning_rate=LR)
Accuracy = 0

gene = generate(file_name, BATCH_SIZE)

for e in range(EPOCH):
    for i in range(len(file_name) // BATCH_SIZE):
        with tf.GradientTape() as tape:
            train_x, label = next(gene)
            train_x = tf.convert_to_tensor(train_x)
            label = tf.convert_to_tensor(label)

            # output shape=(200, 200, 6)
            output = model(train_x)

            # label shape=(200, 200, 6)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=output)
            # loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        loss = tf.reduce_mean(loss)
        acc = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(label, output))

        print("Epoch:{}, loss:{:.5f}, acc:{:.5f}".format(e, loss, acc))

        if acc > Accuracy:
            Accuracy = acc
            model.save_weights("./weight/vpn_weight.h5")

