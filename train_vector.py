# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import, division, print_function

import argparse
import os
import pdb
import pprint
import sys
import time

import _init_paths
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from model.da_faster_rcnn_instance_da_weight.resnet_vector import resnet
#from model.da_faster_rcnn_instance_da_weight.vgg16 import vgg16
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import (
    EFocalLoss,
    FocalLoss,
    adjust_learning_rate,
    clip_gradient,
    load_net,
    save_checkpoint,
    save_net,
    weights_normal_init,
)
from roi_da_data_layer.roibatchLoader import roibatchLoader
from roi_da_data_layer.roidb import combined_roidb
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler

print(sys.path)


def parse_args():
    """
  Parse input arguments    
  """
    parser = argparse.ArgumentParser(description="Train a Fast R-CNN network")
    parser.add_argument(
        "--dataset",
        dest="dataset",
        help="training dataset",
        default="dc",
        type=str,
    )
    # parser.add_argument(
    #     "--dataset_t",
    #     dest="dataset_t",
    #     help="test dataset",
    #     default="df",
    #     type=str,
    # )
    parser.add_argument(
        "--net", dest="net", help="vgg16, res101", default="res101", type=str
    )
    parser.add_argument(
        "--pretrained_path",
        dest="pretrained_path",
        help="vgg16, res101",
        default="/home/wam/SW_Faster_ICR_CCR/pretrain/resnet101_caffe.pth",
        type=str,
    )
    parser.add_argument(
        "--checkpoint_interval",
        dest="checkpoint_interval",
        help="number of iterations to save checkpoint",
        default=1,
        type=int,
    )

    parser.add_argument(
        "--save_dir",
        dest="save_dir",
        help="directory to save models",
        default="/data2/lr/models",
        type=str,
    )
    parser.add_argument(
        "--nw",
        dest="num_workers",
        help="number of worker to load data",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--cuda", dest="cuda", help="whether use CUDA", action="store_true"
    )
    parser.add_argument(
        "--ls",
        dest="large_scale",
        help="whether use large imag scale",
        action="store_true",
    )
    parser.add_argument(
        "--bs", dest="batch_size", help="batch_size", default=1, type=int
    )
    parser.add_argument(
        "--cag",
        dest="class_agnostic",
        help="whether perform class_agnostic bbox regression",
        action="store_true",
    )

    # config optimization
    parser.add_argument(
        "--max_iter",
        dest="max_iter",
        help="max iteration for train",
        default=10000,
        type=int,
    )
    parser.add_argument(
        "--o", dest="optimizer", help="training optimizer", default="sgd", type=str
    )
    parser.add_argument(
        "--lr", dest="lr", help="starting learning rate", default=0.001, type=float
    )
    parser.add_argument(
        "--lr_decay_step",
        dest="lr_decay_step",
        help="step to do learning rate decay, unit is iter",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--lr_decay_gamma",
        dest="lr_decay_gamma",
        help="learning rate decay ratio",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--lamda", dest="lamda", help="DA loss param", default=0.1, type=float
    )

    # set training session
    parser.add_argument(
        "--s", dest="session", help="training session", default=1, type=int
    )

    # resume trained model
    parser.add_argument(
        "--r", dest="resume", help="resume checkpoint or not", default=False, type=bool
    )
    parser.add_argument(
        "--resume_name",
        dest="resume_name",
        help="resume checkpoint path",
        default="",
        type=str,
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        help="resume from which model",
        default="",
        type=str,
    )

    # setting display config
    parser.add_argument(
        "--disp_interval",
        dest="disp_interval",
        help="number of iterations to display",
        default=100,
        type=int,
    )

    parser.add_argument(
        "--lc",
        dest="lc",
        help="whether use context vector for pixel level",
        action="store_true",
    )
    parser.add_argument(
        "--gc",
        dest="gc",
        help="whether use context vector for global level",
        action="store_true",
    )
    parser.add_argument(
        "--da_use_contex",
        dest="da_use_contex",
        help="whether use context vector for instance da",
        action="store_true",
    )
    parser.add_argument(
        "--ef",
        dest="ef",
        help="whether use exponential focal loss",
        action="store_true",
    )
    parser.add_argument(
        "--gamma", dest="gamma", help="value of gamma", default=5, type=float
    )
    parser.add_argument(
        "--max_epochs",
        dest="max_epochs",
        help="max epoch for train",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--start_epoch", dest="start_epoch", help="starting epoch", default=1, type=int
    )

    parser.add_argument(
        "--eta",
        dest="eta",
        help="trade-off parameter between detection loss and domain-alignment loss."
        " Used for Car datasets",
        default=0.1,
        type=float,
    )

    parser.add_argument(
        "--instance_da_eta",
        dest="instance_da_eta",
        help="instance_da_eta",
        default=0.1,
        type=float,
    )

    parser.add_argument(
        "--da_weight", dest="da_weight", help="da_weight", default=1.0, type=float
    )

    args = parser.parse_args()
    return args


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(
                self.num_per_batch * batch_size, train_size
            ).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
        self.rand_num = (
            rand_num.expand(self.num_per_batch, self.batch_size) + self.range
        )

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data


if __name__ == "__main__":

    args = parse_args()

    # print("Called with args:")
    # print(args)

    # if args.dataset == "pascal_voc":
    #     print("loading our dataset...........")
    #     args.imdb_name = "voc_2007_train"
    #     args.imdbval_name = "voc_2007_test"
    #     args.set_cfgs = [
    #         "ANCHOR_SCALES",
    #         "[4,8,16,32]",
    #         "ANCHOR_RATIOS",
    #         "[0.5,1,2]",
    #         "MAX_NUM_GT_BOXES",
    #         "50",
    #     ]
    # elif args.dataset == "cityscape":
    #     print("loading our dataset...........")
    #     args.s_imdb_name = "cityscape_2007_train_s"
    #     args.t_imdb_name = "cityscape_2007_train_t"
    #     args.s_imdbtest_name = "cityscape_2007_test_s"
    #     args.t_imdbtest_name = "cityscape_2007_test_t"
    #     args.set_cfgs = [
    #         "ANCHOR_SCALES",
    #         "[8,16,32]",
    #         "ANCHOR_RATIOS",
    #         "[0.5,1,2]",
    #         "MAX_NUM_GT_BOXES",
    #         "30",
    #     ]

    # elif args.dataset == "rpc":
    #     print("loading our dataset...........")
    #     args.s_imdb_name = "rpc_fake_train"
    #     args.t_imdb_name = "rpc_val"
    #     # args.s_imdbtest_name = "cityscape_2007_test_s"
    #     args.t_imdbtest_name = "rpc_test"
    #     args.set_cfgs = [
    #         "ANCHOR_SCALES",
    #         "[8,16,32]",
    #         "ANCHOR_RATIOS",
    #         "[0.5,1,2]",
    #         "MAX_NUM_GT_BOXES",
    #         "30",
    #     ]

    # elif args.dataset == "clipart":
    #     print("loading our dataset...........")
    #     args.s_imdb_name = "voc_2007_trainval+voc_2012_trainval"
    #     args.t_imdb_name = "clipart_trainval"
    #     args.t_imdbtest_name = "clipart_trainval"
    #     args.set_cfgs = [
    #         "ANCHOR_SCALES",
    #         "[8,16,32]",
    #         "ANCHOR_RATIOS",
    #         "[0.5,1,2]",
    #         "MAX_NUM_GT_BOXES",
    #         "20",
    #     ]

    # elif args.dataset == "water":
    #     print("loading our dataset...........")
    #     args.s_imdb_name = "voc_water_2007_trainval+voc_water_2012_trainval"
    #     args.t_imdb_name = "water_train"
    #     args.t_imdbtest_name = "water_test"
    #     args.set_cfgs = [
    #         "ANCHOR_SCALES",
    #         "[8,16,32]",
    #         "ANCHOR_RATIOS",
    #         "[0.5,1,2]",
    #         "MAX_NUM_GT_BOXES",
    #         "20",
    #     ]

    # elif args.dataset == "pascal_voc_0712":
    #     args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
    #     args.imdbval_name = "voc_2007_test"
    #     args.set_cfgs = [
    #         "ANCHOR_SCALES",
    #         "[8, 16, 32]",
    #         "ANCHOR_RATIOS",
    #         "[0.5,1,2]",
    #         "MAX_NUM_GT_BOXES",
    #         "20",
    #     ]
    # elif args.dataset == "sim10k":
    #     print("loading our dataset...........")
    #     args.s_imdb_name = "sim10k_2019_train"
    #     args.t_imdb_name = "cityscapes_car_2019_train"
    #     args.s_imdbtest_name = "sim10k_2019_val"
    #     args.t_imdbtest_name = "cityscapes_car_2019_val"

    if args.dataset == "dc":
        args.imdb_name = "bdd_daytime_clear"
        args.imdbval_name = "bdd_daytime_clear"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    
    # if args.dataset_t == "dc":
    #     args.imdb_name_target = "bdd_daytime_clear"
    #     args.imdbval_name_target = "bdd_daytime_clear"
    #     args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    # elif args.dataset_t == "df":
    #     args.imdb_name_target = "bdd_daytime_foggy"
    #     args.imdbval_name_target = "bdd_daytime_foggy"
    #     args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    # elif args.dataset_t == "dsa":
    #     args.imdb_name_target = "bdd_daytime_sand"
    #     args.imdbval_name_target = "bdd_daytime_sand"
    #     args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    # elif args.dataset_t == "dsn":
    #     args.imdb_name_target = "bdd_daytime_snow"
    #     args.imdbval_name_target = "bdd_daytime_snow"
    #     args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    # elif args.dataset_t == "dur":
    #     args.imdb_name_target = "bdd_dusk_rainy"
    #     args.imdbval_name_target = "bdd_dusk_rainy"
    #     args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    # elif args.dataset_t == "nc":
    #     args.imdb_name_target = "bdd_night_clear"
    #     args.imdbval_name_target = "bdd_night_clear"
    #     args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    # elif args.dataset_t == "nr":
    #     args.imdb_name_target = "bdd_night_rainy"
    #     args.imdbval_name_target = "bdd_night_rainy"
    #     args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    args.cfg_file = (
        "cfgs/{}_ls.yml".format(args.net)
        if args.large_scale
        else "cfgs/{}.yml".format(args.net)
    )

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # print("Using config:")
    # pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda

    s_imdb, s_roidb, s_ratio_list, s_ratio_index = combined_roidb(args.imdb_name)
    s_train_size = len(s_roidb)  # add flipped         image_index*2

    t1_imdb, t1_roidb, t1_ratio_list, t1_ratio_index = combined_roidb('bdd_night_rainy')
    t1_train_size = len(t1_roidb)  # add flipped         image_index*2

    t2_imdb, t2_roidb, t2_ratio_list, t2_ratio_index = combined_roidb('bdd_dusk_rainy')
    t2_train_size = len(t2_roidb)  # add flipped         image_index*2

    print("source {:d} target1_{:d} target2_{:d} roidb entries".format(len(s_roidb), len(t1_roidb), len(t2_roidb)))

    output_dir = args.save_dir +  "/" + "com_nr_dur"
    # output_dir = args.save_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    s_sampler_batch = sampler(s_train_size, args.batch_size)
    t1_sampler_batch = sampler(t1_train_size, args.batch_size)
    t2_sampler_batch = sampler(t2_train_size, args.batch_size)

    dataset_s = roibatchLoader(
        s_roidb,
        s_ratio_list,
        s_ratio_index,
        args.batch_size,
        s_imdb.num_classes,
        training=True,
    )

    dataloader_s = torch.utils.data.DataLoader(
        dataset_s,
        batch_size=args.batch_size,
        sampler=s_sampler_batch,
        num_workers=args.num_workers,
    )

    dataset_t1 = roibatchLoader(
        t1_roidb,
        t1_ratio_list,
        t1_ratio_index,
        args.batch_size,
        t1_imdb.num_classes,
        training=True,
    )
    dataloader_t1 = torch.utils.data.DataLoader(
        dataset_t1,
        batch_size=args.batch_size,
        sampler=t1_sampler_batch,
        num_workers=args.num_workers,
    )

    dataset_t2 = roibatchLoader(
        t2_roidb,
        t2_ratio_list,
        t2_ratio_index,
        args.batch_size,
        t2_imdb.num_classes,
        training=True,
    )
    dataloader_t2 = torch.utils.data.DataLoader(
        dataset_t2,
        batch_size=args.batch_size,
        sampler=t2_sampler_batch,
        num_workers=args.num_workers,
    )

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    im_cls_lb = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    im_data_t = torch.FloatTensor(1)
    im_info_t = torch.FloatTensor(1)
    im_cls_lb_t = torch.FloatTensor(1)
    num_boxes_t = torch.LongTensor(1)
    gt_boxes_t = torch.FloatTensor(1)
    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        im_cls_lb = im_cls_lb.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

        im_data_t = im_data_t.cuda()
        im_info_t = im_info_t.cuda()
        im_cls_lb_t = im_cls_lb_t.cuda()
        num_boxes_t = num_boxes_t.cuda()
        gt_boxes_t = gt_boxes_t.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    im_data_t = Variable(im_data_t)
    im_info_t = Variable(im_info_t)
    num_boxes_t = Variable(num_boxes_t)
    gt_boxes_t = Variable(gt_boxes_t)
    if args.cuda:
        cfg.CUDA = True

    if args.net == "vgg16":
        fasterRCNN = vgg16(
            s_imdb.classes,
            pretrained_path=args.pretrained_path,
            pretrained=True,
            class_agnostic=args.class_agnostic,
            lc=args.lc,
            gc=args.gc,
            da_use_contex=args.da_use_contex,
        )

    elif args.net == "res101":
        fasterRCNN = resnet(
            s_imdb.classes,
            101,
            pretrained=True,
            class_agnostic=args.class_agnostic,
            lc=args.lc,
            gc=args.gc,
        )

    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr

    #params = []
    #for key, value in dict(fasterRCNN.named_parameters()).items():
    #    if value.requires_grad:
    #        if "bias" in key:
    #            params += [
    #                {
    #                    "params": [value],
    #                    "lr": lr * (cfg.TRAIN.DOUBLE_BIAS + 1),
    #                    "weight_decay": cfg.TRAIN.BIAS_DECAY
    #                    and cfg.TRAIN.WEIGHT_DECAY
    #                    or 0,
    #                }
    #            ]
    #        else:
    #            params += [
    #                {
    #                    "params": [value],
    #                    "lr": lr,
    #                    "weight_decay": cfg.TRAIN.WEIGHT_DECAY,
    #                }
    #            ]

    paramtxt = open('update/RCNN_base1.txt', 'r')
    param = paramtxt.readlines()
    RCNN_base1 = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        RCNN_base1.append(name)

    paramtxt = open('update/RCNN_base2.txt', 'r')
    param = paramtxt.readlines()
    RCNN_base2 = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        RCNN_base2.append(name)

    paramtxt = open('update/netD_pixel.txt', 'r')
    param = paramtxt.readlines()
    netD_pixel = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        netD_pixel.append(name)

    paramtxt = open('update/netD_base.txt', 'r')
    param = paramtxt.readlines()
    netD_base = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        netD_base.append(name)

    paramtxt = open('update/RCNN_top_base.txt', 'r')
    param = paramtxt.readlines()
    RCNN_top_base = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        RCNN_top_base.append(name)

    paramtxt = open('update/RCNN_cls_score_base.txt', 'r')
    param = paramtxt.readlines()
    RCNN_cls_score_base = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        RCNN_cls_score_base.append(name)

    paramtxt = open('update/RCNN_bbox_pred_base.txt', 'r')
    param = paramtxt.readlines()
    RCNN_bbox_pred_base = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        RCNN_bbox_pred_base.append(name)

    paramtxt = open('update/RCNN_rpn.txt', 'r')
    param = paramtxt.readlines()
    RCNN_rpn = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        RCNN_rpn.append(name)

    paramtxt = open('update/di.txt', 'r')
    param = paramtxt.readlines()
    di = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        di.append(name)

    paramtxt = open('update/netD_ds.txt', 'r')
    param = paramtxt.readlines()
    netD_ds = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        netD_ds.append(name)

    paramtxt = open('update/RCNN_bbox_pred_di.txt', 'r')
    param = paramtxt.readlines()
    RCNN_bbox_pred_di = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        RCNN_bbox_pred_di.append(name)

    paramtxt = open('update/RCNN_cls_score_di.txt', 'r')
    param = paramtxt.readlines()
    RCNN_cls_score_di = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        RCNN_cls_score_di.append(name)

    paramtxt = open('update/RCNN_top_di.txt', 'r')
    param = paramtxt.readlines()
    RCNN_top_di = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        RCNN_top_di.append(name)

    di_p = []; netD_base_p = []; netD_pixel_p = []; netD_ds_p = []; RCNN_base1_p = []; RCNN_base2_p = [];
    RCNN_bbox_pred_base_p = []; RCNN_bbox_pred_di_p = []; RCNN_cls_score_base_p = [];
    RCNN_cls_score_di_p = []; RCNN_rpn_p = []; RCNN_top_base_p = []; RCNN_top_di_p = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if key in RCNN_top_di:
            if value.requires_grad:
                if 'bias' in key:
                    RCNN_top_di_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    RCNN_top_di_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

        if key in di:
            if value.requires_grad:
                if 'bias' in key:
                    di_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    di_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in netD_pixel:
            if value.requires_grad:
                if 'bias' in key:
                    netD_pixel_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    netD_pixel_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in netD_base:
            if value.requires_grad:
                if 'bias' in key:
                    netD_base_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    netD_base_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in netD_ds:
            if value.requires_grad:
                if 'bias' in key:
                    netD_ds_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    netD_ds_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in RCNN_base1:
            if value.requires_grad:
                if 'bias' in key:
                    RCNN_base1_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    RCNN_base1_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in RCNN_base2:
            if value.requires_grad:
                if 'bias' in key:
                    RCNN_base2_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    RCNN_base2_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in RCNN_bbox_pred_base:
            if value.requires_grad:
                if 'bias' in key:
                    RCNN_bbox_pred_base_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    RCNN_bbox_pred_base_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in RCNN_bbox_pred_di:
            if value.requires_grad:
                if 'bias' in key:
                    RCNN_bbox_pred_di_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    RCNN_bbox_pred_di_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in RCNN_cls_score_base:
            if value.requires_grad:
                if 'bias' in key:
                    RCNN_cls_score_base_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    RCNN_cls_score_base_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in RCNN_cls_score_di:
            if value.requires_grad:
                if 'bias' in key:
                    RCNN_cls_score_di_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    RCNN_cls_score_di_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in RCNN_rpn:
            if value.requires_grad:
                if 'bias' in key:
                    RCNN_rpn_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    RCNN_rpn_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in RCNN_top_base:
            if value.requires_grad:
                if 'bias' in key:
                    RCNN_top_base_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    RCNN_top_base_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    opt_di = torch.optim.SGD(di_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_netD_pixel = torch.optim.SGD(netD_pixel_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_netD_base = torch.optim.SGD(netD_base_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_netD_ds = torch.optim.SGD(netD_ds_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_RCNN_base1 = torch.optim.SGD(RCNN_base1_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_RCNN_base2 = torch.optim.SGD(RCNN_base2_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_RCNN_bbox_pred_base = torch.optim.SGD(RCNN_bbox_pred_base_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_RCNN_bbox_pred_di = torch.optim.SGD(RCNN_bbox_pred_di_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_RCNN_cls_score_base = torch.optim.SGD(RCNN_cls_score_base_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_RCNN_cls_score_di = torch.optim.SGD(RCNN_cls_score_di_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_RCNN_rpn = torch.optim.SGD(RCNN_rpn_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_RCNN_top_base = torch.optim.SGD(RCNN_top_base_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_RCNN_top_di = torch.optim.SGD(RCNN_top_di_p, momentum=cfg.TRAIN.MOMENTUM)

    optimizer = [opt_di, opt_netD_base, opt_netD_ds, opt_RCNN_base1, opt_RCNN_base2, opt_RCNN_bbox_pred_base, opt_RCNN_bbox_pred_di, opt_RCNN_cls_score_base, \
    opt_RCNN_cls_score_di, opt_RCNN_rpn, opt_RCNN_top_base, opt_RCNN_top_di, opt_netD_pixel]

    def reset_grad():
        opt_di.zero_grad()
        opt_netD_base.zero_grad()
        opt_netD_ds.zero_grad()
        opt_RCNN_base1.zero_grad()
        opt_RCNN_base2.zero_grad()
        opt_RCNN_bbox_pred_base.zero_grad()
        opt_RCNN_bbox_pred_di.zero_grad()
        opt_RCNN_cls_score_di.zero_grad()
        opt_RCNN_cls_score_base.zero_grad()
        opt_RCNN_rpn.zero_grad()
        opt_RCNN_top_base.zero_grad()
        opt_RCNN_top_di.zero_grad()
        opt_netD_pixel.zero_grad()

    def group_step(step_list):
        for i in range(len(step_list)):
            step_list[i].step()
        reset_grad()

    if args.cuda:
        fasterRCNN.cuda()

    if args.resume:
        print(args.resume_name)
        load_name = os.path.join(output_dir, args.resume_name)
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint["session"]
        args.start_epoch = checkpoint["epoch"]
        fasterRCNN.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr = optimizer.param_groups[0]["lr"]
        if "pooling_mode" in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint["pooling_mode"]
        print("loaded checkpoint %s" % (load_name))

    iters_per_epoch = int(s_train_size / args.batch_size)
    # iters_per_epoch = 200
    if args.ef:
        FL = EFocalLoss(class_num=2, gamma=args.gamma)
    else:
        FL = FocalLoss(class_num=2, gamma=args.gamma)

    count_iter = 0
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        fasterRCNN.train()
        loss_temp = 0
        start = time.time()

        if epoch % (args.lr_decay_step + 1) == 0:
            lr_decay = args.lr_decay_gamma
            for m in range(len(optimizer)):
                adjust_learning_rate(optimizer[m], lr_decay)
            lr *= args.lr_decay_gamma

        data_iter_s = iter(dataloader_s)
        data_iter_t1 = iter(dataloader_t1)
        data_iter_t2 = iter(dataloader_t2)
        for step in range(iters_per_epoch):
            try:
                data_s = next(data_iter_s)
            except:
                data_iter_s = iter(dataloader_s)
                data_s = next(data_iter_s)
            switch = np.random.randint(1,3)
            if switch == 1:
                try:
                    data_t = next(data_iter_t1)
                except:
                    data_iter_t1 = iter(dataloader_t1)
                    data_t = next(data_iter_t1)
            elif switch == 2:
                try:
                    data_t = next(data_iter_t2)
                except:
                    data_iter_t2 = iter(dataloader_t2)
                    data_t = next(data_iter_t2)
            eta = 1.0
            count_iter += 1
            # put source data into variable
            # im_data.data.resize_(data_s[0].size()).copy_(data_s[0])
            # im_info.data.resize_(data_s[1].size()).copy_(data_s[1])
            # im_cls_lb.data.resize_(data_s[2].size()).copy_(data_s[2])
            # gt_boxes.data.resize_(data_s[3].size()).copy_(data_s[3])
            # num_boxes.data.resize_(data_s[4].size()).copy_(data_s[4])
            im_data.resize_(data_s[0].size()).copy_(data_s[0])
            im_info.resize_(data_s[1].size()).copy_(data_s[1])
            im_cls_lb.resize_(data_s[2].size()).copy_(data_s[2])
            gt_boxes.resize_(data_s[3].size()).copy_(data_s[3])
            num_boxes.resize_(data_s[4].size()).copy_(data_s[4])

            #First Step
            fasterRCNN.zero_grad()
            reset_grad()

            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, out_d_pixel, out_d, out_ds = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, phase=1, eta=eta)

            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                   + (RCNN_loss_cls[0].mean() + RCNN_loss_bbox[0].mean() \
                   + RCNN_loss_cls[1].mean() + RCNN_loss_bbox[1].mean()) * 0.5

            domain_s = Variable(torch.zeros(out_d.size(0)).long().cuda())
            dloss_s = 0.5 * FL(out_d, domain_s)
            domain_s_d = Variable(torch.zeros(out_ds.size(0)).long().cuda())
            dloss_s_d = 0.5 * FL(out_ds, domain_s_d)
            dloss_s_p = 0.5 * torch.mean(out_d_pixel ** 2)

            # put target data into variable
            # im_data.data.resize_(data_t[0].size()).copy_(data_t[0])
            # im_info.data.resize_(data_t[1].size()).copy_(data_t[1])
            # # gt is empty
            # gt_boxes.data.resize_(1, 1, 5).zero_()
            # num_boxes.data.resize_(1).zero_()
            im_data_t.resize_(data_t[0].size()).copy_(data_t[0])
            im_info_t.resize_(data_t[1].size()).copy_(data_t[1])
            # gt is empty
            gt_boxes_t.resize_(args.batch_size, 1, 5).zero_()
            num_boxes_t.resize_(args.batch_size).zero_()

            out_d_pixel, out_d, out_d_ds = fasterRCNN(im_data_t, im_info_t, gt_boxes_t, num_boxes_t, phase=1, target=True, eta=eta)
            domain_t = Variable(torch.ones(out_d.size(0)).long().cuda())
            dloss_t = 0.5 * FL(out_d, domain_t)
            domain_t_d = Variable(torch.ones(out_d_ds.size(0)).long().cuda())
            dloss_t_d = 0.5 * FL(out_d_ds, domain_t_d)
            dloss_t_p = 0.5 * torch.mean((1 - out_d_pixel) ** 2)

            loss += (dloss_s + dloss_t + dloss_s_d + dloss_t_d) * 0.5 + (dloss_s_p + dloss_t_p) * 0.5

            loss.backward()
            group_step([opt_di, opt_netD_base, opt_netD_ds, opt_RCNN_base1, opt_RCNN_base2, opt_RCNN_bbox_pred_base, opt_RCNN_bbox_pred_di, opt_RCNN_cls_score_base, \
                opt_RCNN_cls_score_di, opt_RCNN_rpn, opt_RCNN_top_base, opt_RCNN_top_di, opt_netD_pixel])

            #Second Step
            fasterRCNN.zero_grad()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, out_ds, MI_s = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, phase=2, eta=eta)
            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                   + (RCNN_loss_cls[0].mean() + RCNN_loss_bbox[0].mean())

            loss_temp += (rpn_loss_cls.mean() + rpn_loss_box.mean() + RCNN_loss_cls[0].mean() + RCNN_loss_bbox[0].mean()).item()

            domain_s = Variable(torch.zeros(out_ds.size(0)).long().cuda())
            dloss_s = 0.5 * FL(out_ds, domain_s)

            out_ds, MI_t = fasterRCNN(im_data_t, im_info_t, gt_boxes_t, num_boxes_t, phase=2, target=True, eta=eta)
            domain_t = Variable(torch.ones(out_ds.size(0)).long().cuda())
            dloss_t = 0.5 * FL(out_ds, domain_t)

            loss += (dloss_s + dloss_t) * 0.5 + (MI_s + MI_t) * 0.5

            loss.backward()
            group_step([opt_di, opt_netD_ds, opt_RCNN_bbox_pred_di, opt_RCNN_cls_score_di, opt_RCNN_top_di])

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= args.disp_interval + 1

                loss_rpn_cls = rpn_loss_cls.item()
                loss_rpn_box = rpn_loss_box.item()
                loss_rcnn_cls = RCNN_loss_cls[0].item()
                loss_rcnn_box = RCNN_loss_bbox[0].item()
                dloss_s = dloss_s.item()
                dloss_t = dloss_t.item()
                dloss_s_p = dloss_s_p.item()
                dloss_t_p = dloss_t_p.item()
                MI_s_p = MI_s.item()
                MI_t_p = MI_t.item()
                fg_cnt = torch.sum(rois_label.data.ne(0))
                bg_cnt = rois_label.data.numel() - fg_cnt

                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                      % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                print(
                    "\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f, dloss s: %.4f, dloss t: %.4f, dloss s pixel: %.4f, dloss t pixel: %.4f, MI_s_p: %.4f, MI_t_p: %.4f, eta: %.4f" \
                    % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box, dloss_s, dloss_t, dloss_s_p, dloss_t_p, MI_s_p, MI_t_p,
                       args.eta))

                if switch == 1:
                    print('nr is used!!!')
                elif switch == 2:
                    print('dur is used!!!')

                loss_temp = 0
                start = time.time()
        if epoch % args.checkpoint_interval == 0 or epoch == args.max_epochs:
            save_name = os.path.join(
                output_dir, "{}.pth".format('compound_nr+dur' + "_" + str(epoch) + "_" + str(iters_per_epoch)),
            )
            save_checkpoint(
                {
                    "session": args.session,
                    "epoch": epoch + 1,
                    "model": fasterRCNN.state_dict(),
                    "pooling_mode": cfg.POOLING_MODE,
                    "class_agnostic": args.class_agnostic,
                },
                save_name,
            )
            print("save model: {}".format(save_name))
