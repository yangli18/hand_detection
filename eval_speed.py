"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOCroot
from data import VOC_CLASSES as labelmap
import torch.utils.data as data

from data import VOC_CLASSES

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2
from scipy.io import loadmat

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

DatasetRoot = 'data/VOCdevkit'
parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/4th_training/ssd300_0712_20000.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--dataset_root', default=DatasetRoot, help='Location of VOC root directory')
parser.add_argument('--remove_ignored', default=True, type=str2bool, help='Remove small hands detections')
parser.add_argument('--version', default='ssd_new_mobilenet_FFA', type=str, help='Detection model version')
parser.add_argument('--input_dim', default=300, type=int, help='input dimension')

args = parser.parse_args()


## using MobileNet v1
if args.version == 'ssd_new_mobilenet_FFA':
    from models.ssd_new_mobilenet_FFA import build_ssd
elif args.version == 'ssd_new_mobilenet':
    from models.ssd_new_mobilenet import build_ssd
elif args.version == 'ssd_mobilenet':
    from models.ssd_mobilenet import build_ssd
else:
    raise ValueError('The version of model is not valid!')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

annopath = os.path.join(args.dataset_root, 'VOC2007', 'Annotations', '%s.xml')
imgpath = os.path.join(args.dataset_root, 'VOC2007', 'JPEGImages', '%s.jpg')
imgsetpath = os.path.join(args.dataset_root, 'VOC2007', 'ImageSets', 'Main', '{:s}.txt')


set_type = 'hand_test_big'
devkit_path = 'data/'

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def test_net(net, input_dim=300):
    """Test the detection time on the image database."""
    imgsetfile = imgsetpath.format('hand_test_big')
    
    with open(imgsetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    num_images = len(imagenames)

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}

    all_det_time = 0

    # warm up
    for i in range(100):
        fake_img = np.zeros((input_dim,input_dim,3))
        x = fake_img.astype(np.float32)
        x = torch.from_numpy(x).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        detections = net(x)

    for i in range(num_images):
        img = cv2.imread( imgpath % imagenames[i] )
        #h,w,c = img.shape
        img_rz = cv2.resize(img, (input_dim,input_dim))
        x = (img_rz.astype(np.float32) / 255.0 - 0.5)*2
        x = Variable(torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0), volatile=True)
        if args.cuda:
            x = x.cuda()

        _t['im_detect'].tic()
        detections = net(x)
        detect_time = _t['im_detect'].toc(average=False)
        all_det_time += detect_time
        # print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
        #                                             num_images, detect_time))
    print('im_avg_detect: {:.10f}'.format(all_det_time/num_images))


if __name__ == '__main__':
    # load net
    num_classes = len(VOC_CLASSES) + 1 # +1 background
    net = build_ssd('test', size=args.input_dim, num_classes=num_classes)
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model: ',args.trained_model)
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True

    # evaluation
    test_net(net, args.input_dim)
