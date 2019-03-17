import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple, OrderedDict, Iterable

def make_fixed_padding(kernel_size, rate=1):
    """Pads the input along the spatial dimensions independently of input size.

    Pads the input such that if it was used in a convolution with 'VALID' padding,
    the output would have the same dimensions as if the unpadded input was used
    in a convolution with 'SAME' padding.

    Args:
        kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
        rate: An integer, rate for atrous convolution.

    Returns:
        output: A padding module.
    """
    if not isinstance(kernel_size, Iterable):
        kernel_size = (kernel_size, kernel_size)
    kernel_size_effective = [kernel_size[0] + (kernel_size[0] - 1) * (rate - 1),
                            kernel_size[0] + (kernel_size[0] - 1) * (rate - 1)]
    pad_total = [kernel_size_effective[0] - 1, kernel_size_effective[1] - 1]
    pad_beg = [pad_total[0] // 2, pad_total[1] // 2]
    pad_end = [pad_total[0] - pad_beg[0], pad_total[1] - pad_beg[1]]
    padding_module = nn.ZeroPad2d((pad_beg[0], pad_end[0],
                                    pad_beg[1], pad_end[1]))
    return padding_module

class Conv2d_tf(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get('padding', 'SAME')
        kwargs['padding'] = 0
        if not isinstance(self.stride, Iterable):
            self.stride = (self.stride, self.stride)
        if not isinstance(self.dilation, Iterable):
            self.dilation = (self.dilation, self.dilation)

    def forward(self, input):
        # from https://github.com/pytorch/pytorch/issues/3867
        if self.padding == 'VALID':
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            padding=0,
                            dilation=self.dilation, groups=self.groups)
        input_rows = input.size(2)
        filter_rows = self.weight.size(2)
        effective_filter_size_rows = (filter_rows - 1) * self.dilation[0] + 1
        out_rows = (input_rows + self.stride[0] - 1) // self.stride[0]
        padding_rows = max(0, (out_rows - 1) * self.stride[0] + effective_filter_size_rows -
                                input_rows)
        # padding_rows = max(0, (out_rows - 1) * self.stride[0] +
        #                         (filter_rows - 1) * self.dilation[0] + 1 - input_rows)
        rows_odd = (padding_rows % 2 != 0)
        # same for padding_cols
        input_cols = input.size(3)
        filter_cols = self.weight.size(3)
        effective_filter_size_cols = (filter_cols - 1) * self.dilation[1] + 1
        out_cols = (input_cols + self.stride[1] - 1) // self.stride[1]
        padding_cols = max(0, (out_cols - 1) * self.stride[1] + effective_filter_size_cols -
                                input_cols)
        # padding_cols = max(0, (out_cols - 1) * self.stride[1] +
        #                         (filter_cols - 1) * self.dilation[1] + 1 - input_cols)
        cols_odd = (padding_cols % 2 != 0)

        if rows_odd or cols_odd:
            input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd)])

        return F.conv2d(input, self.weight, self.bias, self.stride,
                        padding=(padding_rows // 2, padding_cols // 2),
                        dilation=self.dilation, groups=self.groups)

def expand_input(n, divisible_by=8):
    return lambda num_inputs, **_: _make_divisible(num_inputs * n, divisible_by)

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _split_divisible(num, num_ways, divisible_by=8):
    """Evenly splits num, num_ways so each piece is a multiple of divisible_by."""
    assert num % divisible_by == 0
    assert num / num_ways >= divisible_by
    # Note: want to round down, we adjust each split to match the total.
    base = num // num_ways // divisible_by * divisible_by
    result = []
    accumulated = 0
    for i in range(num_ways):
        r = base
        while accumulated + r < num * (i + 1) / num_ways:
            r += divisible_by
        result.append(r)
        accumulated += r
    assert accumulated == num
    return result


def depth_multiplier(depth,
                     multiplier,
                     divisible_by=8,
                     min_depth=8):
    d = depth
    return _make_divisible(d * multiplier, divisible_by,
                                                    min_depth)

def conv_bn(in_channels, out_channels, kernel_size=3, stride=1, padding='SAME'):
    return nn.Sequential(
        Conv2d_tf(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels, eps=0.001),
        nn.ReLU6(inplace=True)
    )

def conv_dw(in_channels, kernel_size=3, stride=1, padding='SAME', dilation=1):
    return nn.Sequential(
        Conv2d_tf(in_channels, in_channels, kernel_size, stride, padding=padding,\
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

def Conv(in_channels, out_channels, kernel_size=3, stride=1, use_explicit_padding=False, **unused_kwargs):
    tmp = OrderedDict()
    if use_explicit_padding:
        tmp.update({'Pad': make_fixed_padding(kernel_size)})
        padding = 'VALID'
    else:
        padding = 'SAME'
    tmp.update({'conv': conv_bn(in_channels, out_channels, kernel_size,
                        stride=stride, padding=padding)})
    return nn.Sequential(tmp)

def DepthSepConv(in_channels, out_channels, kernel_size=3, stride=1, layer_rate=1, use_explicit_padding=False, **unused_kwargs):
    tmp = OrderedDict()
    if use_explicit_padding:
        tmp.update({'Pad': make_fixed_padding(kernel_size, layer_rate)})
        padding = 'VALID'
    else:
        padding = 'SAME'
    tmp.update(OrderedDict([
            ('depthwise', conv_dw(in_channels, kernel_size, stride=stride, padding=padding, dilation=layer_rate)),
            ('pointwise', conv_pw(in_channels, out_channels, 1, stride=1))]))
    return nn.Sequential(tmp)
            
class ExpandedConv(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_size=expand_input(6), kernel_size=3, stride=1, layer_rate=1, residual=True, use_explicit_padding=False, **unused_kwargs):
        super(ExpandedConv, self).__init__()
        self.residual = residual and stride == 1 and in_channels == out_channels

        inner_size = expansion_size(in_channels)

        tmp = OrderedDict()
        if inner_size > in_channels:
            tmp['expand'] = conv_pw(in_channels, inner_size, 1, stride=1)
        if use_explicit_padding:
            tmp.update({'Pad': make_fixed_padding(kernel_size, layer_rate)})
            padding = 'VALID'
        else:
            padding = 'SAME'
        tmp['depthwise'] = conv_dw(inner_size, kernel_size=kernel_size, stride=stride, padding=padding, dilation=layer_rate)
        tmp['project'] = nn.Sequential(
            nn.Conv2d(inner_size, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels))
        self.module = nn.Sequential(tmp)

    def forward(self, x):
        if self.residual:
            return x + self.module(x)
        else:
            return self.module(x)