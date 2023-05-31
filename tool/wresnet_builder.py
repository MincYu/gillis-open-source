from mxnet.gluon import nn
from mxnet import nd
from mxnet import symbol
import mxnet as mx
from mxnet.contrib import onnx as onnx_mxnet
import numpy as np
import os

class WResidual34(nn.HybridBlock):
    def __init__(self, channels, factor=1, same_shape=True, dim_same = True, **kwargs):
        super(WResidual34, self).__init__(**kwargs)
        self.same_shape = same_shape
        self.dim_same = dim_same
        with self.name_scope():
            strides = 1 if same_shape else 2
            self.conv1 = nn.Conv2D(channels * factor, kernel_size=3, padding=1, strides=strides, use_bias=False)
            self.bn1 = nn.BatchNorm()
            self.relu1 = nn.Activation(activation='relu')
            self.conv2 = nn.Conv2D(channels * factor, kernel_size=3, padding=1, use_bias=False)
            self.bn2 = nn.BatchNorm()

            if self.dim_same:
                self.conv3 = nn.Conv2D(channels * factor, kernel_size=1, padding=0, strides=strides, use_bias=False)
                self.bn3 = nn.BatchNorm()
            
            self.relu = nn.Activation(activation='relu')

    def hybrid_forward(self, F, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.dim_same:
            x = self.bn3(self.conv3(x))
        return self.relu(out + x)

class WResidual50(nn.HybridBlock):
    def __init__(self, channels, factor=1, same_shape=True, dim_same = True, **kwargs):
        super(WResidual50, self).__init__(**kwargs)
        self.same_shape = same_shape
        self.dim_same = dim_same
        with self.name_scope():
            strides = 1 if same_shape else 2
            self.conv1 = nn.Conv2D(channels * factor, kernel_size=1, padding=0, use_bias=False)
            self.bn1 = nn.BatchNorm()
            self.relu1 = nn.Activation(activation='relu')
            self.conv2 = nn.Conv2D(channels * factor, kernel_size=3, padding=1, strides=strides, use_bias=False)
            self.bn2 = nn.BatchNorm()
            self.relu2 = nn.Activation(activation='relu')
            self.conv3 = nn.Conv2D(channels * 4 * factor, kernel_size=1, padding=0, use_bias=False)
            self.bn3 = nn.BatchNorm()
            
            if self.dim_same:
                self.conv4 = nn.Conv2D(channels * 4 * factor, kernel_size=1, padding=0, strides=strides, use_bias=False)
                self.bn4 = nn.BatchNorm()
            
            self.relu = nn.Activation(activation='relu')

    def hybrid_forward(self, F, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.dim_same:
            x = self.bn4(self.conv4(x))
        return self.relu(out + x)

    
class WResNet(nn.HybridBlock):
    def __init__(self, num_classes=1000, layers=50, factor=1, verbose=False, **kwargs):
        super(WResNet, self).__init__(**kwargs)
        self.verbose = verbose
        self.WResidual_blocks = {50: WResidual50, 34: WResidual34}
        with self.name_scope():
            net = self.net = nn.HybridSequential()
            # block 1
            net.add(nn.Conv2D(channels=64, kernel_size=7, strides=2, padding=3, use_bias=False))
            net.add(nn.BatchNorm())
            net.add(nn.Activation(activation='relu'))
            net.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))
            WResidual = self.WResidual_blocks[layers]

            # block 2
            net.add(WResidual(64, factor))
            for _ in range(2):
                net.add(WResidual(64, factor, dim_same = False))
            # block 3
            net.add(WResidual(128, factor, same_shape = False))
            for _ in range(3):
                net.add(WResidual(128, factor, dim_same = False))
            # block 4
            net.add(WResidual(256, factor, same_shape = False))
            for _ in range(5):
                net.add(WResidual(256, factor, dim_same = False))
            # block 5
            net.add(WResidual(512, factor, same_shape = False))
            for _ in range(2):
                net.add(WResidual(512, factor, dim_same = False))

            net.add(nn.GlobalAvgPool2D())
            net.add(nn.Flatten())
            net.add(nn.Dense(num_classes))

    def hybrid_forward(self, F, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s'%(i+1, out.shape))
        return out

def get_net(layers, factor):
    net = WResNet(layers=layers, factor=factor)
    net.initialize(ctx=mx.cpu(), init=mx.init.Xavier())
    return net

def build_wrn(model_dir, layers, factor):
    net = get_net(layers, factor).net

    shape = (1, 3, 224, 224)
    names = []
    # slice_indice = [(0, 14), (14, len(net))]
    slice_indice = [(0, len(net))]
    for i in slice_indice:
        sub_net = net[i[0]:i[1]]
        sub_net.hybridize()
        sub_res = sub_net(mx.nd.random.uniform(shape=shape))

        name = f'net{i[0]}'
        sub_net.export(f'{model_dir}/{name}')
        names.append((name, shape))

        shape = sub_res.shape
    return names

def export_onnx(model_dir, names):
    for n, s in names:
        print(f'exporting model {n} with shape {s} ...')
        sym = f'{model_dir}/{n}-symbol.json'
        params = f'{model_dir}/{n}-0000.params'

        onnx_file = f'{model_dir}/{n}.onnx'
        converted_model_path = onnx_mxnet.export_model(sym, params, [s], np.float32, onnx_file)

if __name__ == "__main__":
    # export_onnx(os.getcwd(), [('resnext101', (1, 3, 224, 224))])
    # exit(0)
    layers = 50
    factor = 5
    model_dir = os.getcwd() + f'/wrn{layers}_{factor}'
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    names = build_wrn(model_dir, layers, factor)
    # export_onnx(model_dir, names)


