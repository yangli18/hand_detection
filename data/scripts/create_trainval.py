import scipy.io as sio
import numpy as np
import os
import random
import cv2
import lmdb


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_COLOR)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def createDataset(outputPath, ann_path, img_path, checkValid=True):
    """
    Create LMDB dataset for training.
    """

    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1

    ann_list = os.listdir(ann_path)
    img_list = os.listdir(img_path)
    random.seed(2018)
    random.shuffle(ann_list)

    ## write each pair of img & labels to the lmdb file
    for ann_file in ann_list:
        if (ann_file[-3:] != 'mat' and ann_file[-3:] != 'xml'):
            continue

        ## label
        assert(ann_file[-3:] == 'mat')
        ann = sio.loadmat(ann_path + ann_file)
        labels = []
        boxes = ann['boxes'][0]
        for i in range(boxes.shape[0]):
            ## x = a[1]，y = a[0]
            a = boxes[i][0][0][0][0]
            b = boxes[i][0][0][1][0]
            c = boxes[i][0][0][2][0]
            d = boxes[i][0][0][3][0]

            ##
            xmin = np.min([a[1], b[1], c[1], d[1]])
            ymin = np.min([a[0], b[0], c[0], d[0]])
            xmax = np.max([a[1], b[1], c[1], d[1]])
            ymax = np.max([a[0], b[0], c[0], d[0]])
            label = "{:.4f} {:.4f} {:.4f} {:.4f} \
                    {:.4f} {:.4f} {:.4f} {:.4f}\n".format(a[1], a[0], b[1], b[0],\
                                                          c[1], c[0], d[1], d[0])
            labels = labels + [label]

        labels = ' '.join(labels).encode()

        ## img
        img_name = ann_file[:-3] + 'jpg'
        with open(img_path + img_name, 'rb') as f:
            imageBin = f.read()

        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % img_name)

        imageKey = ('image-%06d' % cnt).encode()
        labelKey = ('label-%06d' % cnt).encode()
        cache[imageKey] = imageBin
        cache[labelKey] = labels
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d' % cnt)

        cnt += 1

    nSamples = cnt - 1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    ann_path = 'hand_dataset/training_dataset/training_data/annotations/'
    img_path = 'hand_dataset/training_dataset/training_data/images/'
    output_path = 'trainval/'
    createDataset(output_path, ann_path, img_path)