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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name',
        type=str,
        default='tsn',
        help='name of model to train.')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/tsn.txt',
        help='path to config file of model')
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
    parser.add_argument(
        '--weights',
        type=str,
        default='./checkpoints_models/tsn_model[0.95].pdparams',
        help='weight path, None to use weights from Paddle.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='sample number in a batch for inference.')
    parser.add_argument(
        '--filelist',
        type=str,
        default=None,
        help='path to inferenece data file lists file.')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='mini-batch interval to log.')
    parser.add_argument(
        '--infer_topk',
        type=int,
        default=1,
        help='topk predictions to restore.')
    parser.add_argument(
        '--save_dir', type=str, default='./output', help='directory to store results')
    args = parser.parse_args()
    return args


def eval(args):
    # parse config
    config = parse_config(args.config)
    print(config,'config')
    val_config = merge_configs(config, 'valid', vars(args))
    #print_configs(val_config, "Valid")
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
        val_reader = KineticsReader(args.model_name.upper(), 'valid', val_config).create_reader()

        # if no weight files specified, exit()
        if args.weights:
            weights = args.weights
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
    #return testacc        
            
            
if __name__ == "__main__":
    args = parse_args()
    # check whether the installed paddle is compiled with GPU
    logger.info(args)

    eval(args)
