import cv2
import os
import json
import math
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
from keras.applications.mobilenet import relu6, DepthwiseConv2D
import time
def r_json(json_paths):
    with open(json_paths) as rf:
        open_json = json.load(rf)
    return open_json
def check_file_exits(path):
    if os.path.exists(path) == 0:
        os.mkdir(path)
def put_heatmap(heatmap, plane_idx, center, sigma):
    center_x = center[0]
    center_y = center[1]
    _, height, width = heatmap.shape[:3]
    th = 1.6052 
    delta = math.sqrt(th * 2)
    x0 = int(max(0, center_x - delta * sigma))
    y0 = int(max(0, center_y - delta * sigma))
    x1 = int(min(width, center_x + delta * sigma))
    y1 = int(min(height, center_y + delta * sigma))
    for y in range(y0, y1):
        for x in range(x0, x1):
            d = (x - center_x) ** 2 + (y - center_y) ** 2
            exp = d / 2.0 / sigma / sigma
            if exp > th:
                continue
            heatmap[plane_idx][y][x] = max(heatmap[plane_idx][y][x], math.exp(-exp))
            heatmap[plane_idx][y][x] = min(heatmap[plane_idx][y][x], 1.0)
    return heatmap
def input_generator(data,network_w,network_h, num_ahead, sigma):
    heatmap = np.zeros((num_ahead, network_w, network_h), dtype=np.float32)  # (4,200,200)
    real_heatmap_model = np.zeros((1, network_w, network_h), dtype=np.float32)  # (1,200,200)
    for idx,point in enumerate(data):
        if idx == num_ahead:
            print(point)
            img = cv2.imread(point)
            h_ratio = network_h/img.shape[0]
            w_ratio = network_w/img.shape[1]
            img = cv2.resize(img,(network_w,network_h))
        if num_ahead*2 + 1 > idx > num_ahead:
            point[0] = point[0] * w_ratio
            point[1] = point[1] * h_ratio
            # print(point[0])
            # print('1234321')
            # print(point[1])
            heatmap_concat = put_heatmap(heatmap, idx-(num_ahead+1), point, sigma)
        if idx == num_ahead*2 + 1 :
            point[0] = point[0] * w_ratio
            point[1] = point[1] * h_ratio
            real_heatmap = put_heatmap(real_heatmap_model, 0, point, sigma)
    # print(img.shape)
    # print(heatmap_concat.shape)
    # print(real_heatmap.shape)
    heatmap_concat = np.transpose(heatmap_concat, [1, 2, 0])
    pic_concat = np.concatenate((heatmap_concat, img),axis=2)
    real_heatmap = np.transpose(real_heatmap,[1, 2, 0])

    return pic_concat, real_heatmap

if __name__ =="__main__":
    sigma = 6
    num_ahead = 2
    model_path = "./weights/model_valid_557_0.002429.h5"
    model_load = load_model(model_path, custom_objects={
                'relu6': relu6,
                'DepthwiseConv2D': DepthwiseConv2D})
    #model_load = load_model(model_path)
    print("**************************************load model sucessful")
    one_json = "/media/cyk/7441-63E8/posetrack/posetrack_data/annotations_delete_ignore/annotations_step_3/left_elbow/016215_mpii_train/person_1/NO_9.json"
    open_json = r_json(one_json)
    picture_list = []
    for per_picture in open_json["images"]:
        picture_list.append(per_picture["file_name"])
    for ann in open_json["annotations"]:
        picture_list.append(ann["keypoints"])
    #print("This is my picture list",picture_list)
    network_w = 200
    network_h = 200
    pic_concat,real_heatmap = input_generator(picture_list, network_w, network_w, num_ahead, sigma)
    pic_concat = np.expand_dims(pic_concat,axis=0)#(1,200,200,7)
    time_head = time.time()
    pic_heatmap = model_load.predict(pic_concat) # 得到预测的pic_heatmap
    time_last = time.time()
    print(time_last-time_head)
    real_heatmap = np.squeeze(real_heatmap, axis=2)#(200,200)
    save_pic = "./save_picture/"
    plt.imsave(os.path.join(save_pic,'pic.png'),pic_heatmap[0,:,:,0])
    ori_picture = cv2.imread(picture_list[num_ahead])#读取实际图片
    print(ori_picture.shape)
    plt.imsave(os.path.join(save_pic,'ori.png'),real_heatmap) #real_heatmap
    y_ori, x_ori = np.where(real_heatmap[:,:] == np.max(real_heatmap[:,:]))
    print(x_ori,y_ori)#ori
    y_pic, x_pic = np.where(pic_heatmap[0, :, :, 0] == np.max(pic_heatmap[0, :, :, 0]))
    print(x_pic,y_pic)#这是预测之后热力图的x,y
    h_pic, w_pic = pic_heatmap.shape[1], pic_heatmap.shape[2]
    h_ori, w_ori = ori_picture.shape[0], ori_picture.shape[1]
    pic_x = (w_ori*x_pic)/w_pic
    pic_y = (h_ori*y_pic)/h_pic
    print("pic_x:", pic_x, "pic_y:", pic_y)
    ori_x = (w_ori*x_ori)/network_w
    ori_y = (h_ori*y_ori)/network_h
    print("ori_x:", ori_x, "ori_y:", ori_y)
    ori_x ,ori_y = open_json["annotations"][2]["keypoints"][0],open_json["annotations"][2]["keypoints"][1]
    print("ori_x:", ori_x, "ori_y:", ori_y)
    cv2.circle(ori_picture,(pic_x,pic_y),3,(55,255,155),1)
    save_path = './test'
    test_name = 'a.jpg'
    cv2.imwrite(os.path.join(save_path,test_name),ori_picture)
