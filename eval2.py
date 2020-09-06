import os
import sys
import time
import logging
import argparse
import ast
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
import paddle.fluid as fluid

from model import  baseline_2d_resnets_pp
from reader import KineticsReader
from config import parse_config, merge_configs, print_configs

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)




def eval2(weights,batch_size):
    # parse config

   
   
    use_gpu = True
    model_name = 'tsn'
    config = 'configs/tsn.txt'
    use_gpu = use_gpu
    weights = weights
    batch_size = batch_size
    log_interval = 1
    infer_topk =1
    save_dir = './output'



    config = parse_config(config)
    val_config = config#merge_configs(config, 'valid', vars(args))
    print_configs(val_config, "Valid")
    with fluid.dygraph.guard():
        #val_model = TSN18.TSNResNet('TSN18', val_config['MODEL']['num_layers'],
                                    #val_config['MODEL']['num_classes'],
                                    #val_config['MODEL']['seg_num'], 0.00002)

        val_model = baseline_2d_resnets_pp.ResNet50Flow('ResNet50Flow',val_config['MODEL']['num_layers'],
                                     val_config['MODEL']['num_classes'], 
                                     val_config['MODEL']['seg_num'])

        label_dic = np.load('label_dir.npy', allow_pickle=True).item()
        label_dic = {v: k for k, v in label_dic.items()}

        # get infer reader
        val_reader = KineticsReader(model_name.upper(), 'valid', val_config).create_reader()

        # if no weight files specified, exit()
        if weights:
            weights = weights
        else:
            print("model path must be specified")
            exit()
            
        para_state_dict, _ = fluid.load_dygraph(weights)
        val_model.load_dict(para_state_dict)
        val_model.eval()
        
        acc_list = []
        for batch_id, data in enumerate(val_reader()):
            dy_x_data = np.array([x[0] for x in data]).astype('float32')
            y_data = np.array([[x[1]] for x in data]).astype('int64')
            
            img = fluid.dygraph.to_variable(dy_x_data)
            label = fluid.dygraph.to_variable(y_data)
            label.stop_gradient = True
            
            out, acc = val_model(img, label)
            acc_list.append(acc.numpy()[0])
            testacc = np.mean(acc_list)
        print("验证集准确率为:{}".format(np.mean(acc_list)))
    return testacc        
            
            
if __name__ == "__main__":
    args = parse_args()
    # check whether the installed paddle is compiled with GPU
    logger.info(args)

    testacc = eval(args)
