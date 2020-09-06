import paddle.fluid as fluid

#from Fconv2d import conv2d_with_filter as fconv2d

import numpy as np
#aaa = torch.max(torch.FloatTensor([[[1,2,3],[3,2,5]],[[1,2,3],[3,2,9]]]))
#aaa = aaa
import math
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D,Conv3D,BatchNorm
import pdb
from paddle.fluid.layers import unstack
from paddle.fluid.layers import stack
#from paddle.fluid.dygraph.base import to_variable

# x 为一个秩为2的张量



#f_grad  = nn.Parameter(torch.FloatTensor([[[[-1], [1]]]]).repeat(32, 32, 1, 1))
#f_grad = f_grad
class FlowLayer(fluid.dygraph.Layer):
    def __init__(self,name_scope,channels=1, bottleneck=32, params=[1, 1, 1, 1, 1], n_iter=10,act=None):
        super(FlowLayer, self).__init__(name_scope)
        self.name_scope = name_scope
        self.n_iter = n_iter
        self._batch_norm = fluid.BatchNorm(channels, act=act)
        self._conv = Conv2D(32, num_filters=32, filter_size=1, stride=1, padding=0, groups=1, act=None, bias_attr=False)
        # self.bottleneck = nn.Conv3d(channels, bottleneck, stride=1, padding=0, bias=False, kernel_size=1)

        self.bottleneck = Conv3D(bottleneck, channels, 1, stride=1, padding=0, bias_attr=None,  act=None)
        # self.unbottleneck = nn.Conv3d(bottleneck*2, channels, stride=1, padding=0, bias=False, kernel_size=1)
        self.unbottleneck = Conv3D(bottleneck * 2, channels, stride=1, padding=0, bias_attr=False, filter_size=1)
        # self.bn = nn.BatchNorm3d(channels)
        self.bn = BatchNorm(channels)
        #print(channels)
        channels = bottleneck

        if params[0]:
            #self.img_grad = nn.Parameter(torch.FloatTensor([[[[-0.5,0,0.5]]]]).repeat(channels,channels,1,1))
            #stop_gradient 属性为true，这意味这反向梯度不会被传递过这个数据变量。如果用户想传递反向梯度，可以设置 var.stop_gradient = False 。
            #self.img_grad2 = nn.Parameter(torch.FloatTensor([[[[-0.5,0,0.5]]]]).transpose(3,2).repeat(channels,channels,1,1))
            img_grad_temp = np.array([[[[-0.5, 0, 0.5]]]]).astype("float32").repeat(channels , 1)
            img_grad_temp2 = np.array([[[[-0.5, 0, 0.5]]]]).astype("float32").transpose(0,1,3,2).repeat(channels, 1)
            img_grad_temp = np.reshape(img_grad_temp, [channels, 1, 1, 3])
            img_grad_temp2 = np.reshape(img_grad_temp2, [channels, 1, 3, 1])
            self.img_grad = self.create_parameter(shape=[channels, 1, 1, 3], dtype="float32")
            self.img_grad2 = self.create_parameter(shape=[channels, 1, 3, 1], dtype="float32")
            self.img_grad.set_value(img_grad_temp)
            self.img_grad2.set_value(img_grad_temp2)
        else:
            #self.img_grad = nn.Parameter(torch.FloatTensor([[[[-0.5,0,0.5]]]]).repeat(channels,channels,1,1), requires_grad=False)
            img_grad_temp = np.array([[[[-0.5, 0, 0.5]]]]).astype("float32").repeat(channels , 1)
            img_grad_temp2 = np.array([[[[-0.5, 0, 0.5]]]]).astype("float32").transpose(0,1,3,2).repeat(channels , 1)
            img_grad_temp = np.reshape(img_grad_temp, [channels, 1, 1, 3])
            img_grad_temp2 = np.reshape(img_grad_temp2, [channels, 1, 3, 1])
            self.img_grad = self.create_parameter(shape=[channels, 1, 1, 3], dtype="float32")
            self.img_grad2 = self.create_parameter(shape=[channels, 1, 3, 1], dtype="float32")
            self.img_grad.set_value(img_grad_temp)
            self.img_grad2.set_value(img_grad_temp2)
            self.img_grad.stop_gradient = True
            self.img_grad2.stop_gradient = True
            
        self.conv2dimg_grad = Conv2D(32, num_filters=1, filter_size=(1,3), stride=1, padding=(0,1), groups=32, act=None, bias_attr=False)#param_attr=self.img_grad,
        self.conv2dimg_grad2 = Conv2D(32, num_filters=1, filter_size=(3,1), stride=1, padding=(1,0), groups=32, act=None, bias_attr=False)#param_attr=self.img_grad2,
        self.conv2dimg_grad.weight = self.img_grad
        self.conv2dimg_grad2.weight = self.img_grad2
        self.prelu = fluid.dygraph.PRelu(mode='all')
        if params[1]:
            #self.f_grad  = nn.Parameter(torch.FloatTensor([[[[-1], [1]]]]).repeat(channels, channels, 1, 1))
            #self.f_grad2 = nn.Parameter(torch.FloatTensor([[[[-1], [1]]]]).repeat(channels, channels, 1, 1))
            #self.div     = nn.Parameter(torch.FloatTensor([[[[-1], [1]]]]).repeat(channels, channels, 1, 1))
            #self.div2    = nn.Parameter(torch.FloatTensor([[[[-1], [1]]]]).repeat(channels, channels, 1, 1))
            img_grad_temp = np.array([[[[-1,1]]]]).astype("float32").repeat(channels , axis= 0)
            img_grad_temp = np.reshape(img_grad_temp, [channels, 1, 1, 2])
            img_grad_temp2 = np.array([[[[-1], [1]]]]).astype("float32").repeat(channels , axis= 0)
            img_grad_temp2 = np.reshape(img_grad_temp, [channels, 1, 2, 1])
            self.f_grad = self.create_parameter(shape=[channels, 1,1,2], dtype="float32")
            
            self.f_grad2 = self.create_parameter(shape=[channels, 1, 2, 1], dtype="float32")
            self.div = self.create_parameter(shape=[channels, 1, 1,2], dtype="float32")
            self.div2 = self.create_parameter(shape=[channels, 1, 2, 1], dtype="float32")
            self.f_grad.set_value(img_grad_temp)
            self.f_grad2.set_value(img_grad_temp2)
            self.div.set_value(img_grad_temp)
            self.div2.set_value(img_grad_temp2)

        else:
            img_grad_temp = np.array([[[[-1], [1]]]]).astype("float32").repeat(channels, axis= 0)
            img_grad_temp = np.reshape(img_grad_temp, [channels, 1, 2, 1])
            self.f_grad = self.create_parameter(shape=[channels, 1,2, 1], dtype="float32")
            self.f_grad2 = self.create_parameter(shape=[channels, 1, 2, 1], dtype="float32")
            self.div = self.create_parameter(shape=[channels, 1, 2, 1], dtype="float32")
            self.div2 = self.create_parameter(shape=[channels, 1, 2, 1], dtype="float32")
            self.f_grad.set_value(img_grad_temp).stop_gradient = True
            self.f_grad2.set_value(img_grad_temp).stop_gradient = True
            self.div.set_value(img_grad_temp).stop_gradient = True
            self.div2.set_value(img_grad_temp).stop_gradient = True
            print('stop_gradient')
        self.conv2df_grad = Conv2D(32, num_filters=1, filter_size=(1,2), stride=1, padding=(0,0), groups=32, act=None, bias_attr=False)#param_attr=self.f_grad,
        self.conv2df_grad2 = Conv2D(32, num_filters=1, filter_size=(2,1), stride=1, padding=(0,0), groups=32, act=None, bias_attr=False)#param_attr=self.f_grad2,
        self.conv2ddiv = Conv2D(32, num_filters=1, filter_size=(1,2), stride=1, padding=(0,0), groups=32, act=None, bias_attr=False)#param_attr=self.div,
        self.conv2ddiv2 = Conv2D(32, num_filters=1, filter_size=(2,1), stride=1, padding=(0,0), groups=32, act=None, bias_attr=False)#param_attr=self.div2,
        self.conv2df_grad.weight = self.f_grad
        self.conv2df_grad2.weight = self.f_grad2
        self.conv2ddiv.weight = self.div
        self.conv2ddiv2.weight = self.div2
        self.channels = channels
        self.t1 = np.array([0.3]).astype("float32")
        self.l1 = np.array([0.15]).astype("float32")
        self.a1 = np.array([0.25]).astype("float32")

        if params[2]:  # XITA
            #self.t = nn.Parameter(torch.FloatTensor([self.t]))
            self.t = self.create_parameter(shape=[1], dtype="float32")
            self.t.set_value(self.t1)
            #print(self.t)
        if params[3]:  # TAU
            #self.l = nn.Parameter(torch.FloatTensor([self.l]))
            self.l = self.create_parameter(shape=[1], dtype="float32")
            self.l.set_value(self.l1)
            #print(self.l)
        if params[4]:  # LABADA
            #self.a = nn.Parameter(torch.FloatTensor([self.a]))
            self.a = self.create_parameter(shape=[1], dtype="float32")
            self.a.set_value(self.a1)
            #print(self.a)
    def norm_img(self, x):
        #mx = torch.max(x)
        mx = fluid.layers.reduce_max(x)
        mn = fluid.layers.reduce_min(x)
        #mn = paddle.tensor.min(x)
        x = 255 * (x - mn) / (mn - mx)
        return x


    def forward_grad(self, x):
        #grad_x = F.conv2d(F.pad(x, (0, 0, 0, 1)), self.f_grad)  # , groups=self.channels)
        #grad_y = F.conv2d(F.pad(x, (0, 0, 0, 1)), self.f_grad2)  # , groups=self.channels)
        x1 = fluid.layers.pad2d(x, paddings=[0,0,0,1])
        #pdb.set_trace()
      
        grad_x = self.conv2df_grad(x1)
        #grad_x[:, :, :, -1] = 0
        
        temp = unstack(grad_x, axis=3)
        
        temp[-1] = temp[-1]*0
        grad_x = stack(temp, axis=3)
       
        x2 = fluid.layers.pad2d(x, paddings=[0,1,0,0])
        grad_y = self.conv2df_grad2(x2)  # , groups=self.channels)
      
        #grad_y[:, :, -1, :] = 0
        temp = unstack(grad_y, axis=2)
        temp[-1] = temp[-1]*0
        grad_y = stack(temp, axis=2)

        bt,c,h,w = grad_x.shape
        
        grad_x = fluid.layers.reshape(grad_x, [-1, c, h, w])
        grad_y = fluid.layers.reshape(grad_y, [-1, c, h, w])
        return grad_x, grad_y


    def divergence(self, x, y):

        #tx = F.pad(x[:,:,:-1,:], (1,0,0,0))
        #ty = F.pad(y[:,:,:-1,:], (0,0,1,0))
        tx = x[:, :, :, :-1]
        ty = y[:, :, :-1, :]
        
        
        tx = fluid.layers.pad2d(tx, paddings=[ 0, 0, 1, 0])
        ty = fluid.layers.pad2d(ty, paddings=[ 1, 0, 0, 0])
        tx = fluid.layers.pad2d(tx, paddings=[ 0, 0, 0, 1])
        ty = fluid.layers.pad2d(ty, paddings=[ 0, 1, 0, 0])
        grad_x = self.conv2ddiv(tx)  # , groups=self.channels)
        
        grad_y = self.conv2ddiv2(ty)  # , groups=self.channels)
        
      
        return grad_x + grad_y


    def forward(self, x):#bcthw
        residual = x[:, :, :-1]
        #print(x.shape)
        x = self.bottleneck(x)
     
        #bbb = self.bottleneck.weight[0]
        #print(bbb)
        inp = self.norm_img(x)
        
        x = inp[:, :, :-1]
        y = inp[:, :, 1:]
        b, c, t, h, w = x.shape
        
        #x = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)
        #y = y.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)
        x = fluid.layers.transpose(x, perm=[0, 2, 1, 3, 4])
        y = fluid.layers.transpose(y, perm=[0, 2, 1, 3, 4])
        x = fluid.layers.reshape(x, [-1, c, h, w])
        y = fluid.layers.reshape(y, [-1, c, h, w])
        #paddle.tensor.zeros_like(input, dtype=None, device=None, stop_gradient=True, name=None)
        u1 = fluid.layers.zeros_like(x)

        u2 = fluid.layers.zeros_like(x)
        l_t = self.l * self.t
        taut = self.a / self.t
     
        #grad2_x = F.conv2d(F.pad(y, (1, 1, 0, 0)), self.img_grad, padding=0, stride=1)  # , groups=self.channels)
        #grad2_x = fluid.layers.conv2d(fluid.layers.pad(y, paddings=[0, 0, 0, 0, 0, 0, 1, 1]), self.img_grad)
        
        #fluid.layers.pad(y, paddings=[0, 0, 0, 0, 0, 0, 1, 1])
        #grad2_x = fconv2d(fluid.layers.pad(y, paddings=[0, 0, 0, 0, 0, 0, 1, 1]), self.img_grad,0,1)
        grad2_x = self.conv2dimg_grad(y)#梯度无变化#梯度无变化#梯度无变化#梯度无变化#梯度无变化#梯度无变化#梯度无变化#梯度无变化#梯度无变化#梯度无变化#梯度无变化
       

        #grad2_x[:, :, :, 0] = 0.5 * (x[:, :, :, 1] - x[:, :, :, 0])
        #grad2_x[:, :, :, -1] = 0.5 * (x[:, :, :, -1] - x[:, :, :, -2])
        temp1 = unstack(x, axis=3)
        temp2 = unstack(x, axis=3)
        temp = unstack(grad2_x, axis=3)
        temp[0] = 0.5 *(temp1[1] - temp2[0])
        temp[-1] = 0.5 *(temp1[-1] - temp2[-2])
        grad2_x = stack(temp, axis=3)
        #查看权重有无变化
        
       # print(self.conv2dimg_grad2[0])
       # print(self.conv2df_grad[0])
       # print(self.conv2df_grad2[0])
        #grad2_y = F.conv2d(F.pad(y, (0, 0, 1, 1)), self.img_grad2, padding=0, stride=1)  # , groups=self.channels)
        #grad2_y = fconv2d(fluid.layers.pad(y, paddings=[0, 0, 0, 0, 1, 1, 0, 0]), self.img_grad2)
        grad2_y = self.conv2dimg_grad2(y)#梯度无变化#梯度无变化#梯度无变化#梯度无变化#梯度无变化#梯度无变化#梯度无变化
        #查看权重有无变化
        
        #grad2_y[:, :, 0, :] = 0.5 * (x[:, :, 1, :] - x[:, :, 0, :])
        #grad2_y[:, :, -1, :] = 0.5 * (x[:, :, -1, :] - x[:, :, -2, :])
        temp1 = unstack(x, axis=2)
        temp2 = unstack(x, axis=2)
        temp = unstack(grad2_x, axis=2)
        temp[0] = 0.5 *(temp1[1] - temp2[0])
        temp[-1] = 0.5 *(temp1[-1] - temp2[-2])
        grad2_x = stack(temp, axis=2)
        
        #p11 = paddle.tensor.zeros_like(x.data)
        #p12 = paddle.tensor.zeros_like(x.data)
        #p21 = paddle.tensor.zeros_like(x.data)
        #p22 = paddle.tensor.zeros_like(x.data)
        p11 = fluid.layers.zeros_like(x)
        p12 = fluid.layers.zeros_like(x)
        p21 = fluid.layers.zeros_like(x)
        p22 = fluid.layers.zeros_like(x)

        gsqx = grad2_x ** 2
        gsqy = grad2_y ** 2
        grad = gsqx + gsqy + 1e-12
        
        rho_c = y - grad2_x * u1 - grad2_y * u2 - x
        



        for i in range(self.n_iter):
            #pdb.set_trace()
            rho = rho_c + grad2_x * u1 + grad2_y * u2 + 1e-12
            
            v1 = fluid.layers.zeros_like(x)
            v2 = fluid.layers.zeros_like(x)
            #mask3 = ((mask1 ^ 1) & (mask2 ^ 1) & (grad > 1e-12)).detach()
            #v1[mask3] = ((-rho / grad) * grad2_x)[mask3]
            #v2[mask3] = ((-rho / grad) * grad2_y)[mask3]
            mask3 = paddle.fluid.layers.where((grad > 1e-12))#((((rho >= -l_t * grad))&((rho <= l_t * grad))&(grad > 1e-12)))
            if mask3.shape[0]:
                v13 = paddle.fluid.layers.gather_nd(v1, mask3)
                v23 = paddle.fluid.layers.gather_nd(v2, mask3)
                v13set = paddle.fluid.layers.gather_nd(((-rho / grad) * grad2_x), mask3)
                v23set = paddle.fluid.layers.gather_nd(((-rho / grad) * grad2_y), mask3)
                v13.set_value(v13set)
                v23.set_value(v23set)
                v1 = paddle.fluid.layers.scatter_nd(mask3,v13,v1.shape)
                v2 = paddle.fluid.layers.scatter_nd(mask3,v23,v2.shape)
            
            #运用gather_nd 和scatter_nd
            #mask1 = (rho < -l_t * grad).detach()
            #v1[mask1] = (l_t * grad2_x)[mask1]
            #v2[mask1] = (l_t * grad2_y)[mask1]
            
            mask1 = paddle.fluid.layers.where(rho < -l_t * grad)
            if mask1.shape[0]:
            
                v11 = paddle.fluid.layers.gather_nd(v1, mask1)
                v21 = paddle.fluid.layers.gather_nd(v2, mask1)
                v11set = paddle.fluid.layers.gather_nd((l_t * grad2_x), mask1)
                v21set = paddle.fluid.layers.gather_nd((l_t * grad2_y), mask1)
                v11.set_value(v11set)
                v1 = paddle.fluid.layers.scatter_nd(mask1,v11,v1.shape)
                v21.set_value(v21set)
                v2 = paddle.fluid.layers.scatter_nd(mask1,v21,v2.shape)

            
            #mask2 = (rho > l_t * grad).detach()
            #v1[mask2] = (-l_t * grad2_x)[mask2]
            #v2[mask2] = (-l_t * grad2_y)[mask2]
            mask2 = paddle.fluid.layers.where(rho > l_t * grad)
            if mask2.shape[0]:
               
                v12 = paddle.fluid.layers.gather_nd(v1, mask2)
                v22 = paddle.fluid.layers.gather_nd(v2, mask2)
                v12set = paddle.fluid.layers.gather_nd((-l_t * grad2_x), mask2)
                v22set = paddle.fluid.layers.gather_nd((-l_t * grad2_y), mask2)
                v12.set_value(v12set)
                v22.set_value(v22set)
                v1 = paddle.fluid.layers.scatter_nd(mask2,v12,v1.shape)         
                v2 = paddle.fluid.layers.scatter_nd(mask2,v22,v2.shape)
            
                #bbb = self.conv2ddiv.weight[0]
                #print(bbb)
            del rho
            del mask1
            del mask2
            del mask3
            """
            del v11
            del v21
            del v11set
            del v21set 
            

            del v12
            del v22
            del v12set
            del v22set

            del v13
            del v23
            del v13set
            del v23set
            """
            
            v1 += u1
            v2 += u2
            
            
            
            u1 = v1 + self.t * self.divergence(p11, p12)#梯度回传没问题 看看前面有没问题ccf
            u2 = v2 + self.t * self.divergence(p21, p22)
            del v1
            del v2
            u1 = u1
            u2 = u2

            u1x, u1y = self.forward_grad(u1)
            u2x, u2y = self.forward_grad(u2)

            p11 = (p11 + taut * u1x) / (1. + taut * fluid.layers.sqrt(u1x ** 2 + u1y ** 2 + 1e-12))
            p12 = (p12 + taut * u1y) / (1. + taut * fluid.layers.sqrt(u1x ** 2 + u1y ** 2 + 1e-12))
            p21 = (p21 + taut * u2x) / (1. + taut * fluid.layers.sqrt(u2x ** 2 + u2y ** 2 + 1e-12))
            p22 = (p22 + taut * u2y) / (1. + taut * fluid.layers.sqrt(u2x ** 2 + u2y ** 2 + 1e-12))
            del u1x
            del u1y
            del u2x
            del u2y

        #flow = torch.cat([u1, u2], dim=1)
        flow = fluid.layers.concat(input=[u1, u2], axis=1)
        #print(u1.shape)
        #print(u2.shape)
        #print(flow.shape)
        #flow = flow.view(b, t, c * 2, h, w).contiguous().permute(0, 2, 1, 3, 4)
        #print(x.shape)
        flow = fluid.layers.reshape(flow, [b, t, c * 2, h, w])
        flow= fluid.layers.transpose(flow, perm=[0, 2, 1, 3, 4])
        flow = self.unbottleneck(flow)
        #print(self.name_scope)
        
        flow = self.bn(flow)
        #print(residual.shape,'xxxx',flow.shape)
        #bbb = residual+flow
        #print(bbb.shape)
        return self.prelu(residual)#+flow

if __name__ == '__main__':

    place = fluid.CUDAPlace(0) if 0 else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        bb = FlowLayer('FlowLayer',channels=32, bottleneck=32, params=[1, 1, 1, 1, 1], n_iter=3)


        x = np.array([[[[2,1],[3,2]]]],dtype="float32")

        x = paddle.fluid.dygraph.base.to_variable(x)
        cc = bb.norm_img(x)
        bb.forward_grad(x)
        print(cc)
