import os
import numpy as np
import cv2
import sys
import glob
import pickle
from multiprocessing import Pool
import pdb

label_dic = np.load('label_dir.npy', allow_pickle=True).item()
print(label_dic)

source_dir = 'hmdbjpg/'
target_train_dir = 'hmdbjpg/allpkl'
target_test_dir = 'hmdbjpg/allpkl'
target_val_dir = 'hmdbjpg/allpkl'
if not os.path.exists(target_train_dir):
    os.mkdir(target_train_dir)
#if not os.path.exists(target_test_dir):
    #os.mkdir(target_test_dir)
#if not os.path.exists(target_val_dir):
    #os.mkdir(target_val_dir)

for key in label_dic:
    each_mulu = key + '_jpg'
    print(each_mulu, key)
   
    label_dir = os.path.join(source_dir, each_mulu)
    label_mulu = os.listdir(label_dir)
    tag = 1
    for each_label_mulu in label_mulu:
        image_file = os.listdir(os.path.join(label_dir, each_label_mulu))
        image_file.sort()
        image_name = image_file[0][:-6]
        image_num = len(image_file)
        print(each_label_mulu)
        frame = []
        vid = image_name
        for i in range(image_num):
            image_path = os.path.join(os.path.join(label_dir, each_label_mulu), image_name + '_' + str(i+1) + '.jpg')
            frame.append(image_path)

        output_pkl = vid + '.pkl'
        if tag < 40:
            output_pkl = os.path.join(target_train_dir, output_pkl)
        elif tag >= 40 and tag<45:
            output_pkl = os.path.join(target_test_dir, output_pkl)
        else:
            output_pkl = os.path.join(target_val_dir, output_pkl)
        tag += 1
        f = open(output_pkl, 'wb')
        pickle.dump((vid, label_dic[key], frame), f, -1)
        f.close()
