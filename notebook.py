#!/usr/bin/env python
# coding: utf-8

# In[ ]:



# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. This directory will be recovered automatically after resetting environment. 
get_ipython().system('ls /home/aistudio/data')


# In[ ]:


# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. All changes under this directory will be kept even after reset. Please clean unnecessary files in time to speed up environment loading.
get_ipython().system('ls /home/aistudio/work')


# In[ ]:


# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, you need to use the persistence path as the following:
get_ipython().system('mkdir /home/aistudio/external-libraries')
get_ipython().system('pip install beautifulsoup4 -t /home/aistudio/external-libraries')


# In[ ]:


get_ipython().system('mkdir /home/aistudio/exernal-libraries')
get_ipython().system('pip install rarfile -t /home/aistudio/external-libraries')
get_ipython().system('pip install unrar -t /home/aistudio/external-libraries')


# In[ ]:


# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可:
# Also add the following code, so that every time the environment (kernel) starts, just run the following code:
import sys
sys.path.append('/home/aistudio/external-libraries')


# In[ ]:


get_ipython().system('unzip -q /home/aistudio/data/data47656/hmdb51_org.rar -d data')


# In[ ]:


#111111 jiang
#建立第一次解压文件夹
import os
video_src_src_path = "/home/aistudio/data/hmdb"

if not os.path.exists(video_src_src_path):
    os.mkdir(video_src_src_path)

get_ipython().system('rar x "/home/aistudio/data/data47656/hmdb51_org.rar" -d data/hmdb')


# In[ ]:





# In[ ]:


#批量打印命令代码
import os
import numpy as np
import cv2


label_name = os.listdir("/home/aistudio/data/hmdb")
#建立第二次解压文件夹
#video_src_src_path = "/home/aistudio/data/hmdb/hmdbavi/"

#if not os.path.exists(video_src_src_path):
    #os.mkdir(video_src_src_path)
video_src_src_path2 = "/home/aistudio/data/hmdb/"
for rarflie in label_name:
    filei =video_src_src_path2+rarflie
    #print(filei)
    
    runcode = '!rar x {} -d hmdbjpg/'.format(filei)
    print(runcode)
    #!rar x  "/home/aistudio/data/sit.rar" -d data/hmdb


# In[ ]:


get_ipython().system('rar x /home/aistudio/data/hmdb/hug.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/draw_sword.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/ride_bike.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/swing_baseball.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/dive.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/kick_ball.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/dribble.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/pour.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/shoot_bow.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/pick.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/smoke.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/clap.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/eat.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/sword_exercise.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/shake_hands.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/pullup.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/flic_flac.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/throw.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/push.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/stand.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/turn.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/cartwheel.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/run.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/smile.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/ride_horse.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/hit.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/sit.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/fall_floor.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/laugh.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/shoot_ball.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/somersault.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/brush_hair.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/golf.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/shoot_gun.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/drink.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/jump.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/sword.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/chew.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/walk.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/kick.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/talk.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/punch.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/climb.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/wave.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/catch.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/kiss.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/handstand.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/pushup.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/fencing.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/situp.rar -d hmdbjpg/')
get_ipython().system('rar x /home/aistudio/data/hmdb/climb_stairs.rar -d hmdbjpg/')


# In[ ]:


#将文件转成jpg
get_ipython().system('python avi2jpg.py')


# In[ ]:


#将文件转换为pkl 
get_ipython().system('python jpg2pkl.py')


# In[ ]:


#将原论文的train列表中的转换为pkl  'hmdbjpg/allpkl/Basic_Basketball_Moves_dribble_f_cm_np1_ri_goo_7.pkl'
get_ipython().system('python data_list_gener_ccf.py')


# In[ ]:


get_ipython().system('python testtesr.py')


# In[22]:


#############训练#########################################
#############训练#########################################
#############训练#########################################
get_ipython().system('python train.py --use_gpu True ')


# 一直训练不出来，找不出什么原因！加班忙竞标，没时间搞了，不好意思！
# [INFO: train.py:  123]: Loss at epoch 0 step 0: [4.079467], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 0 step 100: [4.2944207], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 0 step 200: [4.4078217], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 1 step 0: [3.8298128], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 1 step 100: [4.273285], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 1 step 200: [4.2656593], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 2 step 0: [4.031604], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 2 step 100: [4.1989713], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 2 step 200: [4.3802195], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 3 step 0: [4.3188577], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 3 step 100: [4.3247633], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 3 step 200: [4.3336215], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 4 step 0: [4.1079807], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 4 step 100: [3.8434334], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 4 step 200: [3.9635196], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 5 step 0: [3.9827979], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 5 step 100: [4.1488094], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 5 step 200: [4.081131], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 6 step 0: [4.1455584], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 6 step 100: [4.1515794], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 6 step 200: [4.0413027], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 7 step 0: [3.9229589], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 7 step 100: [4.079381], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 7 step 200: [4.1388535], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 8 step 0: [3.8443344], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 8 step 100: [4.0737543], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 8 step 200: [4.059402], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 9 step 0: [3.736187], acc: [0.1875]
# [INFO: train.py:  123]: Loss at epoch 9 step 100: [4.010389], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 9 step 200: [4.025918], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 10 step 0: [4.0985436], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 10 step 100: [4.150345], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 10 step 200: [4.051855], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 11 step 0: [3.9446325], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 11 step 100: [3.953126], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 11 step 200: [5.369727], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 12 step 0: [5.1749077], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 12 step 100: [10.634393], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 12 step 200: [8.445167], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 13 step 0: [3.840023], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 13 step 100: [3.8882048], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 13 step 200: [13.691071], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 14 step 0: [11.443243], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 14 step 100: [18.957123], acc: [0.125]
# [INFO: train.py:  123]: Loss at epoch 14 step 200: [5.576311], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 15 step 0: [6.172942], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 15 step 100: [5.5378675], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 15 step 200: [4.6583586], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 16 step 0: [5.1759944], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 16 step 100: [5.5600495], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 16 step 200: [4.5540833], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 17 step 0: [5.2831407], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 17 step 100: [4.8479624], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 17 step 200: [5.4225044], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 18 step 0: [5.4713326], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 18 step 100: [5.3898487], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 18 step 200: [5.434576], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 19 step 0: [4.6498365], acc: [0.1875]
# [INFO: train.py:  123]: Loss at epoch 19 step 100: [5.752571], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 19 step 200: [5.3855295], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 20 step 0: [4.608441], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 20 step 100: [5.324392], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 20 step 200: [5.16581], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 21 step 0: [5.9639874], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 21 step 100: [4.916876], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 21 step 200: [4.711916], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 22 step 0: [4.593629], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 22 step 100: [4.961303], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 22 step 200: [4.911063], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 23 step 0: [4.8542404], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 23 step 100: [4.4223623], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 23 step 200: [5.898564], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 24 step 0: [4.7504625], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 24 step 100: [4.411187], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 24 step 200: [4.729386], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 25 step 0: [4.7342644], acc: [0.125]
# [INFO: train.py:  123]: Loss at epoch 25 step 100: [4.685506], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 25 step 200: [4.7100487], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 26 step 0: [5.3674216], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 26 step 100: [4.504959], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 26 step 200: [5.043587], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 27 step 0: [4.6281514], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 27 step 100: [4.665947], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 27 step 200: [4.443961], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 28 step 0: [4.436695], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 28 step 100: [4.2599545], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 28 step 200: [4.6702924], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 29 step 0: [4.2549157], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 29 step 100: [4.347786], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 29 step 200: [4.3915133], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 30 step 0: [4.2809467], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 30 step 100: [4.4884515], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 30 step 200: [4.720532], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 31 step 0: [4.6420197], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 31 step 100: [4.3932524], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 31 step 200: [4.756996], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 32 step 0: [4.538313], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 32 step 100: [4.6553774], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 32 step 200: [4.514392], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 33 step 0: [4.2243733], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 33 step 100: [4.2061176], acc: [0.125]
# [INFO: train.py:  123]: Loss at epoch 33 step 200: [4.8755765], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 34 step 0: [4.9773216], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 34 step 100: [4.738996], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 34 step 200: [4.6106257], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 35 step 0: [4.474992], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 35 step 100: [4.6804514], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 35 step 200: [4.009863], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 36 step 0: [4.450961], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 36 step 100: [4.6509666], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 36 step 200: [4.345744], acc: [0.125]
# [INFO: train.py:  123]: Loss at epoch 37 step 0: [4.588981], acc: [0.125]
# [INFO: train.py:  123]: Loss at epoch 37 step 100: [4.0587773], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 37 step 200: [4.176918], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 38 step 0: [4.415605], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 38 step 100: [4.767333], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 38 step 200: [4.6189976], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 39 step 0: [4.415846], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 39 step 100: [4.4397182], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 39 step 200: [4.3654847], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 40 step 0: [4.6567564], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 40 step 100: [4.2272406], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 40 step 200: [4.25696], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 41 step 0: [4.6229334], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 41 step 100: [4.5427265], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 41 step 200: [4.1254253], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 42 step 0: [4.233179], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 42 step 100: [4.51416], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 42 step 200: [4.168515], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 43 step 0: [4.2614264], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 43 step 100: [4.3235903], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 43 step 200: [3.9711213], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 44 step 0: [4.225455], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 44 step 100: [4.353599], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 44 step 200: [3.965623], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 45 step 0: [4.886133], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 45 step 100: [4.5685453], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 45 step 200: [4.262517], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 46 step 0: [4.1261344], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 46 step 100: [4.109868], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 46 step 200: [3.8954964], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 47 step 0: [4.0771337], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 47 step 100: [4.0736485], acc: [0.125]
# [INFO: train.py:  123]: Loss at epoch 47 step 200: [4.0999002], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 48 step 0: [4.2023067], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 48 step 100: [3.9955363], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 48 step 200: [4.1338673], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 49 step 0: [4.3165913], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 49 step 100: [4.033274], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 49 step 200: [3.8160958], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 50 step 0: [4.020291], acc: [0.125]
# [INFO: train.py:  123]: Loss at epoch 50 step 100: [4.215229], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 50 step 200: [3.8788862], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 51 step 0: [3.9856882], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 51 step 100: [4.198495], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 51 step 200: [4.2547517], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 52 step 0: [4.1667595], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 52 step 100: [4.2539554], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 52 step 200: [4.0170765], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 53 step 0: [4.2324057], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 53 step 100: [4.090042], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 53 step 200: [4.1791134], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 54 step 0: [4.0894413], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 54 step 100: [4.118537], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 54 step 200: [3.9243639], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 55 step 0: [4.1366267], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 55 step 100: [4.210012], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 55 step 200: [4.1504526], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 56 step 0: [4.020673], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 56 step 100: [4.2888484], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 56 step 200: [4.1728153], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 57 step 0: [4.158638], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 57 step 100: [4.1371646], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 57 step 200: [3.9339561], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 58 step 0: [4.062765], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 58 step 100: [4.027998], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 58 step 200: [4.0752025], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 59 step 0: [4.1331863], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 59 step 100: [4.129676], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 59 step 200: [3.967423], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 60 step 0: [4.226557], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 60 step 100: [4.0354404], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 60 step 200: [4.217484], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 61 step 0: [4.0830297], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 61 step 100: [3.9343479], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 61 step 200: [3.9776464], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 62 step 0: [4.042197], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 62 step 100: [3.998217], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 62 step 200: [3.97866], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 63 step 0: [4.1046977], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 63 step 100: [4.0863256], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 63 step 200: [4.017205], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 64 step 0: [3.9876804], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 64 step 100: [4.045273], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 64 step 200: [3.9783657], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 65 step 0: [4.136282], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 65 step 100: [3.7989206], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 65 step 200: [3.8711638], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 66 step 0: [4.000336], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 66 step 100: [4.033142], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 66 step 200: [4.0441175], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 67 step 0: [3.958294], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 67 step 100: [4.1395707], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 67 step 200: [4.0640497], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 68 step 0: [4.0699472], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 68 step 100: [4.1003375], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 68 step 200: [4.2905183], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 69 step 0: [4.0579705], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 69 step 100: [3.9916248], acc: [0.0625]
# [INFO: train.py:  123]: Loss at epoch 69 step 200: [4.2406454], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 70 step 0: [3.9329782], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 70 step 100: [4.0831017], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 70 step 200: [3.9685123], acc: [0.]
# [INFO: train.py:  123]: Loss at epoch 71 step 0: [4.0263124], acc: [0.]

# In[1]:


get_ipython().system("python eval.py --weights 'checkpoints_models/tsn_model.pdparams' --use_gpu True")


# 由于最近一直忙于竞标，没查出来有什么问题
# [INFO: eval.py:  123]: Namespace(batch_size=1, config='configs/tsn.txt', filelist=None, infer_topk=1, log_interval=1, model_name='tsn', save_dir='./output', use_gpu=True, weights='checkpoints_models/tsn_model.pdparams')
# {'MODEL': {'name': 'TSN', 'format': 'pkl', 'num_classes': 51, 'seg_num': 16, 'seglen': 1, 'image_mean': [0.485, 0.456, 0.406], 'image_std': [0.229, 0.224, 0.225], 'num_layers': 50}, 'TRAIN': {'epoch': 45, 'short_size': 240, 'target_size': 56, 'num_reader_threads': 1, 'buf_size': 1024, 'batch_size': 16, 'use_gpu': True, 'num_gpus': 1, 'filelist': './hmdbjpg/train.list', 'learning_rate': 0.01, 'learning_rate_decay': 0.1, 'l2_weight_decay': 0.0001, 'momentum': 0.9, 'total_videos': 80}, 'VALID': {'short_size': 240, 'target_size': 56, 'num_reader_threads': 1, 'buf_size': 1024, 'batch_size': 16, 'filelist': './hmdbjpg/test.list'}, 'TEST': {'seg_num': 7, 'short_size': 240, 'target_size': 224, 'num_reader_threads': 1, 'buf_size': 1024, 'batch_size': 10, 'filelist': './hmdbjpg/test.list'}, 'INFER': {'short_size': 240, 'target_size': 224, 'num_reader_threads': 1, 'buf_size': 1024, 'batch_size': 1, 'filelist': './hmdbjpg/test.list'}} config
# {'MODEL': {'name': 'TSN', 'format': 'pkl', 'num_classes': 51, 'seg_num': 16, 'seglen': 1, 'image_mean': [0.485, 0.456, 0.406], 'image_std': [0.229, 0.224, 0.225], 'num_layers': 50}, 'TRAIN': {'epoch': 45, 'short_size': 240, 'target_size': 56, 'num_reader_threads': 1, 'buf_size': 1024, 'batch_size': 16, 'use_gpu': True, 'num_gpus': 1, 'filelist': './hmdbjpg/train.list', 'learning_rate': 0.01, 'learning_rate_decay': 0.1, 'l2_weight_decay': 0.0001, 'momentum': 0.9, 'total_videos': 80}, 'VALID': {'short_size': 240, 'target_size': 56, 'num_reader_threads': 1, 'buf_size': 1024, 'batch_size': 16, 'filelist': './hmdbjpg/test.list'}, 'TEST': {'seg_num': 7, 'short_size': 240, 'target_size': 224, 'num_reader_threads': 1, 'buf_size': 1024, 'batch_size': 10, 'filelist': './hmdbjpg/test.list'}, 'INFER': {'short_size': 240, 'target_size': 224, 'num_reader_threads': 1, 'buf_size': 1024, 'batch_size': 1, 'filelist': './hmdbjpg/test.list'}}
# W0906 10:50:30.306151   127 device_context.cc:252] Please NOTE: device: 0, CUDA Capability: 70, Driver API Version: 10.1, Runtime API Version: 9.0
# W0906 10:50:30.310778   127 device_context.cc:260] device: 0, cuDNN Version: 7.3.
# 验证集准确率为:0.019607843831181526

# In[ ]:


get_ipython().system("python infer.py --weights 'checkpoints_models/tsn_model50[0.125].pdparams' --use_gpu True")


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

# In[ ]:


get_ipython().system('python testtesr.py')


# In[ ]:


get_ipython().system('python jpg2pkl.py')
get_ipython().system('python data_list_gener.py')


# In[ ]:


get_ipython().system('python train.py --use_gpu True --epoch 100')

