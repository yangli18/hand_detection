
from __future__ import division
import os
import torch
import cv2
import numpy as np
import lmdb
import random
import torch.utils.data as data

VOC_CLASSES = ('hand', )


class HandDataset(data.Dataset):
    
    "Oxford hand dataset."
    
    def __init__(self, lmdb_root, shuffle=False, transform=None, input_size = 300):

        self.name = 'Oxford hand dataset'
        self.lmdb_root = lmdb_root
        self.transform = transform
        #self.target_transform = target_transform #
        self.input_size = input_size
        self.env = lmdb.open(lmdb_root, max_readers=1,
                 readonly=True,
                 lock=False,
                 readahead=False,
                 meminit=False)

        self.txn = self.env.begin(write=False)

        #self.nSamples = int(self.txn.get('num-samples'))
        self.nSamples = int(self.txn.get('num-samples'.encode()))
        self.indices = range(self.nSamples)
        
        if shuffle:
            random.shuffle(self.indices)
 
    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        assert index <= len(self), 'index range error'
        #imgkey = 'image-%06d' % (self.indices[index]+1)
        #labkey = 'label-%06d' % (self.indices[index]+1)
        imgkey = ('image-%06d' % (self.indices[index]+1)).encode()
        labkey = ('label-%06d' % (self.indices[index]+1)).encode()

        imageBin = self.txn.get(imgkey)
        imageBuf = np.fromstring(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_COLOR) ## BGR format, same as cv2.imread('...')
        H,W,C = img.shape

        tid = 0
        #truths = self.txn.get(labkey).rstrip().split('\n')
        truths = self.txn.get(labkey).decode().rstrip().split('\n')
        target1 = np.zeros( [len(truths),8] )
        
        for truth in truths:
            truth = truth.split()
            tmp = [float(t) for t in truth]
            #if tmp[3] > 8.0/img.shape[0]:
            target1[tid,:] = np.array(tmp)
            tid = tid + 1

        ''''''

        labels = np.zeros([len(truths),1])
        ## boxes # [[xmin, ymin, xmax, ymax, label_ind], ... ]
        ## 
        if self.transform is not None:
            img, boxes, labels = self.transform(img, target1,labels) ##
            # to rgb
            # img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            #target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            target = np.hstack((boxes, labels))
        
        return torch.from_numpy(img).permute(2, 0, 1), target, H, W # H*W*C -> C*H*W


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets
