from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable

from data import detection_collate, VOC_CLASSES
from data.hand_dataset import HandDataset

import torch.utils.data as data

from layers.modules import MultiBoxLoss

import numpy as np
import time

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

#os.environ['CUDA_LAUNCH_BLOCKING'] = 1
DatasetRoot = 'data/trainval'
parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--basenet', default='mobilenet_v1_1.0_224.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=5, type=int, help='Number of workers used in dataloading')
parser.add_argument('--iterations', default=90000, type=int, help='Number of training iterations')
parser.add_argument('--start_iter', default=0, type=int, help='Begin counting iterations starting from this value (should be used with resume)')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=0.00005, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--send_images_to_visdom', type=str2bool, default=False, help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--voc_root', default=DatasetRoot, help='Location of VOC root directory')
parser.add_argument('--version', default='ssd_new_mobilenet_FFA', type=str, help='model version')
parser.add_argument('--data_aug_version', default='v1', type=str, help='model version')
parser.add_argument('--input_dim', default=300, type=int, help='input dimension')
args = parser.parse_args()

''' Data Augmentation '''
if args.data_aug_version == 'v1':
    from utils.augmentations import SSDmobilenetAugmentation
elif args.data_aug_version == 'v2':
    from utils.augmentations import SSDmobilenetAugmentation_v2 ## Add image rotation
else:
    raise ValueError('The version of data augmentation is not valid!')

''' Detection Model '''
## using MobileNet v1
if args.version == 'ssd_new_mobilenet_FFA':
    from models.ssd_new_mobilenet_FFA import build_ssd
elif args.version == 'ssd_new_mobilenet':
    from models.ssd_new_mobilenet import build_ssd
elif args.version == 'ssd_mobilenet':
    from models.ssd_mobilenet import build_ssd
else:
    raise ValueError('The version of model is not valid!')


if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if not os.path.exists(args.save_folder + args.version):
    os.mkdir(args.save_folder + args.version)

ssd_dim = args.input_dim  # only support 300 now
num_classes = len(VOC_CLASSES) + 1
batch_size = args.batch_size
accum_batch_size = 32
iter_size = accum_batch_size / batch_size
max_iter = 120000
#weight_decay = 0.0005
stepvalues = (80000, 100000, 120000)
#gamma = 0.1
#momentum = 0.9

if args.visdom:
    import visdom
    viz = visdom.Visdom()

ssd_net = build_ssd('train', size=args.input_dim, num_classes=num_classes)
net = ssd_net

if args.cuda:
    net = torch.nn.DataParallel(ssd_net, device_ids=[0])
    cudnn.benchmark = True

## initialize weights
if args.resume:
    print('Resuming training, loading {}...'.format(args.resume))
    ssd_net.load_weights(args.resume)
else:
    mobilenet_weights = torch.load(args.save_folder + args.basenet)
    # remove unused weights in mobilenet_weights
    mobilenet_weights.pop('classifier.weight')
    mobilenet_weights.pop('classifier.bias')
    # load weights
    print('Loading base network...')
    ssd_net.mobilenet.load_state_dict(mobilenet_weights)

if args.cuda:
    net = net.cuda()


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()


if not args.resume:
    print('Initializing weights...')
    # initialize newly added layers' weights with xavier method
    ssd_net.extras.apply(weights_init)
    ssd_net.loc.apply(weights_init)
    ssd_net.conf.apply(weights_init)

    objlist = dir(ssd_net)
    if 'toplayer' in objlist and 'latlayer' in objlist:
        ssd_net.toplayer.apply(weights_init)
        ssd_net.latlayer.apply(weights_init)

optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)


criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, args.cuda)


def train():
    net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0
    print('Loading Dataset...')

    if args.data_aug_version == 'v1':
        dataset = HandDataset(args.voc_root, shuffle=False, transform=SSDmobilenetAugmentation(size=args.input_dim))
    elif args.data_aug_version == 'v2':
        dataset = HandDataset(args.voc_root, shuffle=False, transform=SSDmobilenetAugmentation_v2(size=args.input_dim))

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD_mobilenet %s on' % args.version, dataset.name, 'with multibox loss')
    print('Data augmentaion: %s' % args.data_aug_version)
    step_index = 0
    if args.visdom:
        # initialize visdom loss plot
        lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 3)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='Loss',
                title='Current SSD Training Loss',
                legend=['Loc Loss', 'Conf Loss', 'Loss']
            )
        )
        epoch_lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 3)).cpu(),
            opts=dict(
                xlabel='Epoch',
                ylabel='Loss',
                title='Epoch SSD Training Loss',
                legend=['Loc Loss', 'Conf Loss', 'Loss']
            )
        )
    batch_iterator = None
    data_loader = data.DataLoader(dataset, batch_size, num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate, pin_memory=True)
    
    for iteration in range(args.start_iter, max_iter):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader)

        if iteration in stepvalues:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)
            if args.visdom:
                viz.line(
                    X=torch.ones((1, 3)).cpu() * epoch,
                    Y=torch.Tensor([loc_loss, conf_loss,
                        loc_loss + conf_loss]).unsqueeze(0).cpu() / epoch_size,
                    win=epoch_lot,
                    update='append'
                )
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        # load train data
        images, targets = next(batch_iterator)
        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno, volatile=True) for anno in targets]

        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.data[0]
        conf_loss += loss_c.data[0]
        if iteration % 100 == 0:
            print('Timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || lr: %f ' % (optimizer.param_groups[0]['lr']) + ' || loc_loss: %.4f ' % (loss_l.data[0]) + ' || cls_loss: %.4f ' % (loss_c.data[0]) , end=' ')
            #print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')
            if args.visdom and args.send_images_to_visdom:
                random_batch_index = np.random.randint(images.size(0))
                viz.image(images.data[random_batch_index].cpu().numpy())
        if args.visdom:
            viz.line(
                X=torch.ones((1, 3)).cpu() * iteration,
                Y=torch.Tensor([loss_l.data[0], loss_c.data[0],
                    loss_l.data[0] + loss_c.data[0]]).unsqueeze(0).cpu(),
                win=lot,
                update='append'
            )
            # hacky fencepost solution for 0th epoch plot
            if iteration == 0:
                viz.line(
                    X=torch.zeros((1, 3)).cpu(),
                    Y=torch.Tensor([loc_loss, conf_loss,
                        loc_loss + conf_loss]).unsqueeze(0).cpu(),
                    win=epoch_lot,
                    update=True
                )
        if iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), args.save_folder + args.version + '/' +
                       repr(iteration) + '.pth')
    torch.save(ssd_net.state_dict(), args.save_folder + '' + args.version + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()
