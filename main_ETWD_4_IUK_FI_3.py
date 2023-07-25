import os
import time
import datetime
import pickle
import yaml
import threading
import logging

import torch.nn as nn
from tensorboardX import SummaryWriter

from src.server_ETWD_4_IUK_FI import Server
from src.utils.utils_inital import launch_tensor_board
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ETWD')
    parser.add_argument('--liver_solo', default=2.0, type=float)
    parser.add_argument('--kidney_solo', default=2.0, type=float)
    parser.add_argument('--pancreas_solo', default=10.0, type=float)
    parser.add_argument('--data_path', default='../data/Fed/', type=str)

    parser.add_argument('--liver_train', default='liver_train_v3.txt', type=str)
    parser.add_argument('--liver_val', default='liver_val_v3.txt', type=str)
    parser.add_argument('--liver_test', default='liver_test_v3.txt', type=str)

    parser.add_argument('--kidney_train', default='kidney_train_v3.txt', type=str)
    parser.add_argument('--kidney_val', default='kidney_val_v3.txt', type=str)
    parser.add_argument('--kidney_test', default='kidney_test_v3.txt', type=str)

    parser.add_argument('--pancreas_train', default='pancreas_train_v3.txt', type=str)
    parser.add_argument('--pancreas_val', default='pancreas_val_v3.txt', type=str)
    parser.add_argument('--pancreas_test', default='pancreas_test_v3.txt', type=str)

    parser.add_argument('--bilinear', default=True, type=bool)
    parser.add_argument('--res', default=True, type=bool)

    parser.add_argument('--E', default=2, type=int)
    parser.add_argument('--R', default=500, type=int)

    parser.add_argument('--en_weight', default=0.9, type=float)
    parser.add_argument('--attn_weight', default=0.1, type=float)
    parser.add_argument('--ss_weight', default=0.2, type=float)

    parser.add_argument('--inter_alpha', default=1000, type=float)
    parser.add_argument('--inter_beta',  default=0.000994, type=float)
    parser.add_argument('--union_alpha', default=1000, type=float)
    parser.add_argument('--union_beta',  default=0.000901, type=float)

    parser.add_argument('--clients_auto_weight_flag', default=False, type=bool)
    parser.add_argument('--scale_square', default=1000000, type=float)

    parser.add_argument('--liver_client_weight', default=0.2797413793103448, type=float)
    parser.add_argument('--kidney_client_weight', default=0.37981974921630096, type=float)
    parser.add_argument('--pancreas_client_weight', default=0.3404388714733542, type=float)

    parser.add_argument('--begin_SS_round', default=15, type=int)
    parser.add_argument('--SSK', default=0.2, type=float)

    parser.add_argument('--begin_FI_round', default=15, type=int)
    parser.add_argument('--FIK', default=0.2, type=float)
    parser.add_argument('--FI_batch_size', default=8, type=int)
    parser.add_argument('--FI_num_batch', default=15, type=int)


    parser.add_argument('--load_trained_model', default=False, type=bool)
    # parser.add_argument('--load_model_path', default='./0.8798637850997348.pth', type=str)

    args = parser.parse_args()

    # read configuration file
    with open('./config.yaml') as c:
        configs = list(yaml.load_all(c, Loader=yaml.FullLoader))
    global_config = configs[0]["global_config"]
    data_config   = configs[1]["data_config"]
    fed_config    = configs[2]["fed_config"]
    optim_config  = configs[3]["optim_config"]
    init_config   = configs[4]["init_config"]
    log_config    = configs[5]["log_config"]
    model_config  = configs[8]["ETWD_config"]

    assert str(model_config["name"]) == 'ETWD', 'model config error'

    # modify E of fed_config
    fed_config["E"] = args.E

    # modify R of fed_config
    fed_config["R"] = args.R

    # modify data_path of data_config
    data_config["data_path"] = args.data_path

    # modify is_mp of global_config
    global_config['is_mp'] = False

    # modify log_path to contain current time and model name
    log_config["log_path"] = os.path.join(log_config["log_path"], str(model_config["name"]), str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))

    # modify model_save_path to contain current time and model name
    log_config["model_save_path"] = os.path.join(log_config["model_save_path"], str(model_config["name"]), str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))
    if not os.path.isdir(log_config["model_save_path"]):
        os.makedirs(log_config["model_save_path"])

    # modify model_tmp_save_path to contain current time and model name
    log_config["model_tmp_save_path"] = os.path.join(log_config["model_tmp_save_path"], str(model_config["name"]), str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))
    if not os.path.isdir(log_config["model_tmp_save_path"]):
        os.makedirs(log_config["model_tmp_save_path"])

    # modify model_tmp_save_path to contain current time and model name
    log_config["model_BCV_save_path"] = os.path.join(log_config["model_BCV_save_path"], str(model_config["name"]), str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))
    if not os.path.isdir(log_config["model_BCV_save_path"]):
        os.makedirs(log_config["model_BCV_save_path"])

    # modify bilinear of model_config
    model_config["bilinear"] = args.bilinear

    # modify res of model_config
    model_config["res"] = args.res

    # initiate TensorBaord for tracking losses and metrics
    writer = SummaryWriter(log_dir=log_config["log_path"], filename_suffix="ETWD")

    # set the configuration of global logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(log_config["log_path"], log_config["log_name"]),
        level=logging.INFO,
        format="[%(levelname)s](%(asctime)s) %(message)s",
        datefmt="%Y/%m/%d/ %I:%M:%S %p")

    # display and log experiment configuration
    message = "\n[WELCOME] Unfolding configurations...!"
    print(message); logging.info(message)

    for config in configs:
        print(config); logging.info(config)


    # initialize federated learning
    central_server = Server(writer, args, model_config, global_config, data_config, init_config, fed_config, optim_config, log_config)
    central_server.setup(args)

    # do federated learning
    central_server.fit()

    # bye!
    message = "...done all learning process!\n...exit program!"
    print(message); logging.info(message)
    time.sleep(3); exit()

