import paddle.fluid as fluid
import numpy as np
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear,PRelu
from paddle.fluid.dygraph.base import to_variable
from model.rep_flow_layer_pp import FlowLayer
from collections import OrderedDict
class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None):
        super(ConvBNLayer, self).__init__(name_scope)

        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=False)

        self._batch_norm = BatchNorm(num_filters, act=act)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class BottleneckBlock(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True):
        super(BottleneckBlock, self).__init__(name_scope)

        self.conv0 = ConvBNLayer(
            self.full_name(),
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='relu'
        )
        self.conv1 = ConvBNLayer(
            self.full_name(),
            num_channels=num_filters,  # ?num_channels
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu'
        )
        self.conv2 = ConvBNLayer(
            self.full_name(),
            num_channels=num_filters,  # ?num_channels
            num_filters=num_filters * 4,
            filter_size=1,
            act='relu')

        if not shortcut:
            self.short = ConvBNLayer(
                self.full_name(),
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                stride=stride)

        self.shortcut = shortcut

        self._num_channels_out = num_filters * 4

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = fluid.layers.elementwise_add(x=short, y=conv2)
        layer_helper = LayerHelper(self.full_name(), act='relu')
        return layer_helper.append_activation(y)


class ResNet50Flow(fluid.dygraph.Layer):
    # 定义网络结构，代码补齐
    def __init__(self, name_scope, layers=50, class_dim=51, seg_num=10,n_iter=20,learnable=[1,1,1,1,1], weight_devay=None):
        super(ResNet50Flow, self).__init__(name_scope)

        self.layers = layers
        self.seg_num = seg_num
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "have not this layers {} in supported_layers".format(layers)
        if layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_filters = [64, 128, 256, 512]

        #nn.Conv3d(128 * block.expansion, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.flow_cmp = Conv2D(512, num_filters=32, filter_size=1, stride=1, padding=0, groups=1, act=None,bias_attr=False)
        self.flow_layer = FlowLayer('flow1',channels=32,bottleneck=32, n_iter=n_iter, params=learnable)

        # Flow-of-flow
        self.flow_cmp2 =  Conv2D(32, num_filters=32, filter_size=1, stride=1, padding=0, groups=1, act=None,bias_attr=False)
        self.flow_layer2 = FlowLayer('flow2',channels=32, bottleneck=32,n_iter=n_iter, params=learnable)
        #
        self.unbottleneck = Conv2D(32, 512, filter_size=1, stride=1, padding=0, groups=1, act=None,bias_attr=False)
        self.bnf = BatchNorm(512)
        self.relu = PRelu(mode='all')
        self.conv = ConvBNLayer(
            self.full_name(),
            num_channels=3,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu'
        )
        self.pool2d_max = Pool2D(
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')

        self.bottleneck_block_list = []
        num_channels = 64



        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        self.full_name(),
                        num_channels=num_channels,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        shortcut=shortcut))
                num_channels = bottleneck_block._num_channels_out
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True


        self.pool2d_avg = Pool2D(pool_size=7, pool_type='avg', global_pooling=True)

        import math
        stdv = 1.0 / math.sqrt(2048 * 1.0)
        
        self.Dropout = fluid.dygraph.Dropout(p=0.3)

        self.out = Linear(input_dim=num_channels,
                          output_dim=class_dim,
                          act='softmax',
                          param_attr=fluid.param_attr.ParamAttr(
                              initializer=fluid.initializer.Uniform(-stdv, stdv)))
    def forward(self, inputs, label=None):

        b,t,c,h,w=inputs.shape
        out = fluid.layers.reshape(inputs, [-1, c, h, w])
        y = self.conv(out)
        y = self.pool2d_max(y)


        #[3 4 6 3]
        for bottleneck_block1 in self.bottleneck_block_list[0:3]:
            y = bottleneck_block1(y)

        for bottleneck_block2 in self.bottleneck_block_list[3:7]:
            y = bottleneck_block2(y)
            
        y = self.flow_cmp(y)  # 卷积通道变为32
        #bbb = self.flow_cmp.weight[0]
        #print(bbb)

        _, ci, h, w = y.shape
        res = y
        y = fluid.layers.reshape(y,[-1,t,ci,h,w])
        y = fluid.layers.transpose(y, perm=[0,2,1,3,4])


        #y = self.flow_layer.norm_img(y)
        _, ci,t, h, w = y.shape
        t = t - 1
        
        y = self.flow_layer(y)
        _, ci, t, h, w = y.shape
        
        #print(y.shape)
        y = fluid.layers.transpose(y, perm=[0,2,1,3,4])
        y = fluid.layers.reshape(y,[-1,ci,h,w])
        
        y = self.flow_cmp2(y)  # 卷积通道变为32
        _,ci, h, w = y.shape
        y = fluid.layers.reshape(y,[-1,t,ci,h,w])
        y = fluid.layers.transpose(y, perm=[0,2,1,3,4])

        _, ci, t, h, w = y.shape
        t = t - 1
        y = self.flow_layer2(y)
        
        _, ci, t, h, w = y.shape
        y = fluid.layers.transpose(y, perm=[0,2,1,3,4])
        y = fluid.layers.reshape(y,[-1,ci,h,w])
        #x = self.bnf(x)
        y = self.unbottleneck(y)
        
        y = self.bnf(y)
        y = self.relu(y)


        for bottleneck_block3 in self.bottleneck_block_list[7:13]:
            y = bottleneck_block3(y)


        for bottleneck_block4 in self.bottleneck_block_list[13:16]:
            y = bottleneck_block4(y)
            


      
        
        y = self.pool2d_avg(y)
        out = fluid.layers.reshape(x=y, shape=[-1, self.seg_num-2, y.shape[1]])
        y = self.Dropout(y)
        out = fluid.layers.reduce_mean(out, dim=1)
        y = self.out(out)

        if label is not None:
            acc = fluid.layers.accuracy(input=y, label=label)
            return y, acc
        else:
            return y


if __name__ == '__main__':
    with fluid.dygraph.guard():
        network = ResNet50Flow('resnet', 50)
        img = np.zeros([1, 10, 3, 112, 112]).astype('float32')
        img = fluid.dygraph.to_variable(img)
        paddle_weight = network.state_dict()
        new_weight_dict = OrderedDict()
        for paddle_key in paddle_weight:
            print(paddle_key)
        print("########################################################")
        for paddle_key2 in paddle_weight.keys():
            print(paddle_key2,paddle_weight[paddle_key2].shape)

        fluid.dygraph.Layer.set_dict(stat_dict, include_sublayers=True)
        outs = network(img).numpy()
        print(outs)