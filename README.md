# 说明
Representation Flow for Action Recognition PaddlePddle

representation flow for action recognition 用PaddlePaddle复现 
论文地址：https://arxiv.org/pdf/1810.01455 
论文Github地址：https://github.com/piergiaj/representation-flow-cvpr19

#论文数据集
HMDB51数据集
数据集地址
https://aistudio.baidu.com/aistudio/datasetdetail/47656



#文件结构
avi2jpg.py	avi视频中提取jpg图像帧
/hmdbjpg/   用于存储将视频变换的jpg文件及split1 train.list（split1） test.list（split1）
jpg2pkl.py	按（split1）划分数据集到train.list（split1） test.list（split1）
train.py	模型训练程序
eval.py	模型测试程序
notebook.py	AI Studio 的notebook程序
model/baseline_2d_resnets_pp.py	ResNet50主干网络结构+flow 主干网络为2D ResNet50
model/rep_flow_layer_pp.py	光流表示层网络结构
configs/tsn.txt 配置文件
#数据集处理
解压：
!unzip -q /home/aistudio/data/data47656/hmdb51_org.rar -d data

##建立第一次解压文件夹
import os
video_src_src_path = "/home/aistudio/data/hmdb"
#批量打印命令代码
import os
import numpy as np
import cv2



##建立第二次解压文件夹
label_name = os.listdir("/home/aistudio/data/hmdb")
video_src_src_path2 = "/home/aistudio/data/hmdb/"
for rarflie in label_name:
    filei =video_src_src_path2+rarflie
    #print(filei)
    
    runcode = '!rar x {} -d hmdbjpg/'.format(filei)
    print(runcode)
    #!rar x  "/home/aistudio/data/sit.rar" -d data/hmdb
if not os.path.exists(video_src_src_path):
    os.mkdir(video_src_src_path)
##建立第二次解压生成的批处理程序，用如下批处理程序在AI Studio中运行，解压所有种类视频文件
!rar x /home/aistudio/data/hmdb/hug.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/draw_sword.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/ride_bike.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/swing_baseball.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/dive.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/kick_ball.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/dribble.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/pour.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/shoot_bow.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/pick.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/smoke.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/clap.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/eat.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/sword_exercise.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/shake_hands.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/pullup.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/flic_flac.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/throw.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/push.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/stand.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/turn.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/cartwheel.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/run.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/smile.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/ride_horse.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/hit.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/sit.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/fall_floor.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/laugh.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/shoot_ball.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/somersault.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/brush_hair.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/golf.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/shoot_gun.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/drink.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/jump.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/sword.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/chew.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/walk.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/kick.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/talk.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/punch.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/climb.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/wave.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/catch.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/kiss.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/handstand.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/pushup.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/fencing.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/situp.rar -d hmdbjpg/
!rar x /home/aistudio/data/hmdb/climb_stairs.rar -d hmdbjpg/

#将文件转成jpg
!python avi2jpg.py

#将文件转换为pkl  
## 注意： 需要train_split1.txt test_split1.txt
!python jpg2pkl.py


#运行训练
python train.py --use_gpu True 

#模型test
!python eval.py --weights 'checkpoints_models/tsn_model.pdparams' --use_gpu True

