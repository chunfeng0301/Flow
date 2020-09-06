#'data/hmdb_data_demo/train/Basic_Basketball_Moves_dribble_f_cm_np1_ri_goo_7.pkl'
# txt to pkl
import os
import numpy as np
import cv2
data_dir = 'hmdbjpg/'
filepath = 'hmdbjpg/'

with open(filepath+'train.txt', 'r') as ftxt:
        lines1 = [line.strip() for line in ftxt]
with open(filepath+'test.txt', 'r') as ftxt:
        lines2 = [line.strip() for line in ftxt]
f = open('hmdbjpg/train.list', 'w')
for line in lines1:
    portion = os.path.splitext(line)  
    f.write(data_dir+ 'allpkl/' + portion[0] +'.pkl' + '\n')
    print(data_dir+ 'allpkl/' + portion[0] +'.pkl' + '\n')

f = open('hmdbjpg/test.list', 'w')
for line in lines2:
    portion = os.path.splitext(line) 
    f.write(data_dir+ 'allpkl/' + portion[0] +'.pkl' + '\n')
    print(data_dir+ 'allpkl/' + portion[0] +'.pkl' + '\n')
