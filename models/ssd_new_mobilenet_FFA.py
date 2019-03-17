import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import v2
import os

from collections import namedtuple, OrderedDict

Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'stride', 'depth'])

# _CONV_DEFS specifies the MobileNet body
_CONV_DEFS = [
    Conv(kernel=[3, 3], stride=2, depth=32),
    DepthSepConv(kernel=[3, 3], stride=1, depth=64),
    DepthSepConv(kernel=[3, 3], stride=2, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=2, depth=256),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256), # conv5
    DepthSepConv(kernel=[3, 3], stride=1, depth=512), # conv6: s2-> s1
    DepthSepConv(kernel=[3, 3], stride=1, depth=512), 
    DepthSepConv(kernel=[3, 3], stride=1, depth=512), # conv8 (38*38 as feat)
    DepthSepConv(kernel=[3, 3], stride=2, depth=512), # conv9: s1 -> s2
    DepthSepConv(kernel=[3, 3], stride=1, depth=512), 
    DepthSepConv(kernel=[3, 3], stride=1, depth=512), # conv11 (19*19 as feat)
    DepthSepConv(kernel=[3, 3], stride=2, depth=1024),
    DepthSepConv(kernel=[3, 3], stride=1, depth=1024) # conv13 (10*10 as feat)
]

## 
def mobilenet_v1_base(final_endpoint='features.Conv2d_13_pointwise',
                      min_depth=8,
                      depth_multiplier=1.0,
                      conv_defs=_CONV_DEFS,
                      output_stride=None):

    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    end_points = OrderedDict()

    # Used to find thinned depths for each layer.
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')

    if conv_defs is None:
        conv_defs = _CONV_DEFS

    if output_stride is not None and output_stride not in [8, 16, 32]:
        raise ValueError('Only allowed output_stride values are 8, 16, 32.')

    def conv_bn(in_channels, out_channels, kernel_size=3, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001),
            nn.ReLU6(inplace=True)
        )

    def conv_dw(in_channels, kernel_size=3, stride=1, dilation=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, 1,\
                      groups=in_channels, dilation=dilation, bias=False),
            nn.BatchNorm2d(in_channels, eps=0.001),
            nn.ReLU6(inplace=True)
        )

    def conv_pw(in_channels, out_channels, kernel_size=1, stride=1, dilation=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001),
            nn.ReLU6(inplace=True),
        )

    current_stride = 1

    # The atrous convolution rate parameter.
    rate = 1

    in_channels = 3
    for i, conv_def in enumerate(conv_defs):
        end_point_base = 'features.Conv2d_%d' % i

        if output_stride is not None and current_stride == output_stride:
            layer_stride = 1
            layer_rate = rate
            rate *= conv_def.stride
        else:
            layer_stride = conv_def.stride
            layer_rate = 1
            current_stride *= conv_def.stride

        out_channels = depth(conv_def.depth)
        
        # use Conv or DepthSepConv
        if isinstance(conv_def, Conv):
            end_point = end_point_base
            end_points[end_point] = conv_bn(in_channels, out_channels, conv_def.kernel,
                                            stride=conv_def.stride)
            if end_point == final_endpoint:
                return nn.Sequential(end_points)

        elif isinstance(conv_def, DepthSepConv):
            end_points[end_point_base] = nn.Sequential(OrderedDict([
                ('depthwise', conv_dw(in_channels, conv_def.kernel, stride=layer_stride, dilation=layer_rate)),
                ('pointwise', conv_pw(in_channels, out_channels, 1, stride=1))]))

            if end_point_base + '_pointwise' == final_endpoint:
                return nn.Sequential(end_points)

        else:
            raise ValueError('Unknown convolution type %s for layer %d'
                                                % (conv_def.ltype, i))
        in_channels = out_channels
    raise ValueError('Unknown final endpoint %s' % final_endpoint)

class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        # TODO: implement __call__ in PriorBox
        self.priorbox = PriorBox(v2)
        
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = 300
        
        ## 38,19,10,5,3,1
        self.src_names = ['features.Conv2d_8', 'features.Conv2d_11', 'features.Conv2d_13', 'Conv2d_14', 'Conv2d_15', 'Conv2d_16']
        self.src_num = len(self.src_names)
        self.src_channels = [512, 512, 1024, 512, 256, 256]
        self.feat_channels = [256, 256, 256, 256, 256, 256]
        # SSD network
        self.mobilenet = mobilenet_v1_base()

        # extra layers
        self.extras = extra_layers()

        # Lateral layers
        latlayer = list()
        for i in range(self.src_num):
            in_channel = self.src_channels[i]
            latlayer += [nn.Conv2d(in_channel, 256, kernel_size=1, stride=1, padding=0 )]
        self.latlayer = nn.ModuleList(latlayer)
                
        ## top layers
        toplayer = list()
        for i in range(self.src_num):
            if i>= self.src_num-2: # ignore Conv2d_15,16
                toplayer += [None]
            else:
                #toplayer += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)]
                toplayer += [ nn.Sequential(
                                nn.Conv2d(256, 256, 3, 1, 1, groups=256, bias=False),
                                nn.BatchNorm2d(256, eps=0.001),
                                nn.Conv2d(256, 256, 1, 1, 0, bias=False),
                                nn.BatchNorm2d(256, eps=0.001),
                                #nn.ReLU6(inplace=True)
                            )]
        self.toplayer = nn.ModuleList(toplayer)

        # loc head && conf head
        head = self.multibox(mbox['300'], num_classes)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax()
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)


    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # bottom-up
        # all_modules = self.mobilenet._modules.items() + self.extras._modules.items()
        all_modules = self.mobilenet._modules.copy()
        all_modules.update(self.extras._modules)
        all_modules = all_modules.items()
        
        for name, module in all_modules:
            x = module(x)
            if name in self.src_names:
                sources.append(x)

        # top-down
        features = [None for i in range(self.src_num)]
        for i in range(self.src_num-1,-1,-1):
            if i>=self.src_num-2: ## ignore Convd15,Convd16
                features[i] = self.latlayer[i](sources[i])
            else:
                features[i] = self.toplayer[i]( self.upsample_add(features[i+1], self.latlayer[i](sources[i]) ) )

        # apply multibox head to source layers
        for (x, l, c) in zip(features, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 10),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
                self.priors.type(type(x.data)).cuda()                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 10),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def upsample_add(self, x, y):
        _,_,H,W = y.size()
        x = F.upsample(x, scale_factor=2, mode='nearest')
        return x[:,:,:H,:W] + y

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')
    
    def multibox(self, cfg, num_classes):
        loc_layers = []
        conf_layers = []

        for k, in_channels in enumerate(self.feat_channels):
            loc_layers += [nn.Conv2d(in_channels,
                                    cfg[k] * 10, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(in_channels,
                            cfg[k] * num_classes, kernel_size=3, padding=1)]

        return (loc_layers, conf_layers)

def extra_layers():
    
    layers = OrderedDict()
    
    def conv_module1(in_channels, inter_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU6(inplace=True),
            
            nn.Conv2d(inter_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU6(inplace=True)
        )
    
    def conv_module2(in_channels, inter_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU6(inplace=True),
            
            nn.Conv2d(inter_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU6(inplace=True)
        )
    
    layers[ 'Conv2d_14' ] = conv_module1(1024, 256, 512)    # 5*5
    layers[ 'Conv2d_15' ] = conv_module2(512, 128, 256)     # 3*3
    layers[ 'Conv2d_16' ] = conv_module2(256, 128, 256)     # 1*1
    
    return nn.Sequential(layers)



base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    ## other option [3,6,6,6,6,6]
    '512': [],
}


def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 300:
        print("Error: Sorry only SSD300 is supported currently!")
        return

    return SSD(phase, num_classes)
