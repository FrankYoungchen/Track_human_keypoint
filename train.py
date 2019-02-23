import cv2
import os
import json
import math
import numpy as np
import uuid
from keras import optimizers
from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard
import network
import random
def r_json(json_paths):
    with open(json_paths) as rf:
        open_json = json.load(rf)
    return open_json
def check_file_exits(path):
    if os.path.exists(path) == 0:
        os.mkdir(path)
#print(read_valid)
#print(read_train)
def put_heatmap(heatmap, plane_idx, center, sigma):
    center_x = center[0]
    center_y = center[1]
    _, height, width = heatmap.shape[:3]
    th = 1.6052 #这个不知道改多少
    delta = math.sqrt(th * 2)
    x0 = int(max(0, center_x - delta * sigma))
    y0 = int(max(0, center_y - delta * sigma))
    x1 = int(min(width, center_x + delta * sigma))
    y1 = int(min(height, center_y + delta * sigma))
    #左上角，右下角
    # gaussian filter
    for y in range(y0, y1):
        for x in range(x0, x1):
            d = (x - center_x) ** 2 + (y - center_y) ** 2
            exp = d / 2.0 / sigma / sigma
            if exp > th:
                continue
            heatmap[plane_idx][y][x] = max(heatmap[plane_idx][y][x], math.exp(-exp))
            heatmap[plane_idx][y][x] = min(heatmap[plane_idx][y][x], 1.0)
    return heatmap
def input_generator(data, network_w, network_h, s, heatmap_head_channel):
    sigma = 6
    heatmap = np.zeros((heatmap_head_channel, network_w, network_h), dtype=np.float32)  # (4,200,200)
    #heatmap_pic = np.zeros((1, int(network_w/4),int(network_h/4)), dtype=np.float32)  # (1,200,200)
    heatmap_pic = np.zeros((1, int(network_w),int(network_h)), dtype=np.float32)
    img = cv2.imread(data[heatmap_head_channel])
    # print(data[heatmap_head_channel])
    # print('111111111111111')
    #print(img)
    h_ratio = network_h / img.shape[0]
    w_ratio = network_w / img.shape[1]
    img = cv2.resize(img, (network_w, network_h))
    num = []
    for i in range(heatmap_head_channel):
        num.append([data[i+heatmap_head_channel+1][0] * int(w_ratio), data[i+heatmap_head_channel+1][1] * int(h_ratio)])
    for i in range(heatmap_head_channel):
        heatmap = put_heatmap(heatmap, i, num[i], sigma)
    num_1 = []
    #num_1.append(data[2*heatmap_head_channel+1][0] * w_ratio/4)
    #num_1.append(data[2*heatmap_head_channel+1][1] * h_ratio/4)
    #heatmap_pic_reslut = put_heatmap(heatmap_pic, 0, num_1, sigma/4)
    num_1.append(data[2*heatmap_head_channel+1][0] * w_ratio)
    num_1.append(data[2*heatmap_head_channel+1][1] * h_ratio)
    heatmap_pic_reslut = put_heatmap(heatmap_pic, 0, num_1, sigma)
    heatmap = np.transpose(heatmap, [1, 2, 0])
    pic_concat = np.concatenate((heatmap, img), axis=2)
    heatmap_pic_reslut = np.transpose(heatmap_pic_reslut, [1, 2, 0])
    return pic_concat, heatmap_pic_reslut
def data_generator(picture_list, batch_size, network_w, network_h, s, heatmap_head_channel):
    random.shuffle(picture_list)
    print(picture_list[0])
    i = 0
    while True:
        image_data = []
        label_data = []
        for j in range(batch_size):
            #print(picture_list)

            data = picture_list[i]
            input, label = input_generator(data, network_w, network_h, s, heatmap_head_channel)
            image_data.append(input)
            label_data.append(label)
            i = (i + 1) % len(picture_list)
        image_data = np.array(image_data)
        label_data = np.array(label_data)
        yield image_data, label_data
def train(read_train,read_valid):
    batch_size = 32
    w = 200
    h = 200
    heatmap_head_channel = 2 #head heatmap number
    picture_channel =3
    s = heatmap_head_channel + picture_channel
    #print('s:%d',s)
    epoch = 1000
    alt_model='Unet'
    if alt_model == '7_conv':
        model = network.creat_model_7_conv(w, h, s)
    elif alt_model == 'mobilenetv1':
        model = network.creat_model_mobilenetv1(w, h, s)
    elif alt_model == 'mobilenetv2':
        model = network.creat_model_mobilenetv2(w, h, s)
    elif alt_model == 'Unet':
         model = network.creat_model_7_upconv(w, h, s)
    checkpoint = ModelCheckpoint('weights/model_valid_{epoch:02d}_{val_loss:05f}.h5', monitor='val_loss', verbose=0)
    check_file_exits("weights")
    logname = "./logs/pose%s" % str(uuid.uuid4())
    check_file_exits("./logs")
    os.makedirs(logname)
    tensorboard = TensorBoard(log_dir=logname)
    model.fit_generator(
                generator=data_generator(read_train, batch_size, w, h, s, heatmap_head_channel),
                validation_data=data_generator(read_valid, batch_size, w, h, s, heatmap_head_channel),
                validation_steps=len(read_valid) // batch_size,
                epochs=epoch,
                steps_per_epoch=len(read_train) // batch_size,
                verbose=1,
            callbacks=[tensorboard, checkpoint])
    model.save('weights/model.h5')
    print("save over")
if __name__ == "__main__":
    train_json = "./train_1.json"
    valid_json = "./valid_1.json"
    read_train = r_json(train_json)
    #print(read_train)
    read_valid = r_json(valid_json)
    train(read_train,read_valid)
