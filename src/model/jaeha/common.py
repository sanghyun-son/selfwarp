import functools

import torch
from torch import nn
from torch.nn import functional as F

def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)

def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += conv3x3(inplanes, outplanes)
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)

def get_norm_layer(layer_type='instance'):
    if layer_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif layer_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif layer_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % layer_type)
    return norm_layer

def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=False)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=False)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=False)
    else:
        raise NotImplementedError('nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer

def conv3x3(in_planes, out_planes):
    return [nn.ReflectionPad2d(1), nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=0, bias=True)]

def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)

####################################################################
#-------------------------- Basic Blocks --------------------------
####################################################################
class MyConv2d(nn.Module):

    def __init__(self, n_in, n_out, kernel_size, stride, padding=0, norm=None, Res=False, sn=False, Leaky=False):
        super().__init__()
        model = [nn.ReflectionPad2d(padding)]

        if sn:
            model += [spectral_norm(nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True))]
        else:
            model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]

        if norm == 'Instance':
            model += [nn.InstanceNorm2d(n_out, affine=False)]
        elif norm == 'Batch':
            model += [nn.BatchNorm2d(n_out)]
        elif norm == 'Layer':
            model += [LayerNorm(n_out)]
        elif norm != 'None':
            raise NotImplementedError('not implemeted norm type')

        if Leaky:
            model += [nn.LeakyReLU(inplace=False)]
        else:
            model += [nn.ReLU(inplace=False)]

        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
        self.Res = Res

    def forward(self, x):
        if self.Res:
            return self.model(x) + x
        else:
            return self.model(x)

######################################################################

class ReLUINSConvTranspose2d(nn.Module):

    def __init__(self, n_in, n_out, kernel_size, stride, padding, output_padding):
        super(ReLUINSConvTranspose2d, self).__init__()
        model = []
        model += [nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=True)]
        model += [LayerNorm(n_out)]
        model += [nn.ReLU(inplace=False)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x)

class INSResBlock(nn.Module):

    def conv3x3(self, inplanes, out_planes, stride=1):
        return [nn.ReflectionPad2d(1), nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride)]

    def __init__(self, inplanes, planes, stride=1, dropout=0.0):
        super().__init__()
        model = []
        model += self.conv3x3(inplanes, planes, stride)
        model += [nn.InstanceNorm2d(planes)]
        model += [nn.ReLU(inplace=False)]
        model += self.conv3x3(planes, planes)
        model += [nn.InstanceNorm2d(planes)]
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class MisINSResBlock(nn.Module):

    def conv3x3(self, dim_in, dim_out, stride=1):
        return nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride))

    def conv1x1(self, dim_in, dim_out):
        return nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0)

    def __init__(self, dim, dim_extra, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Sequential(
                self.conv3x3(dim, dim, stride),
                nn.InstanceNorm2d(dim))
        self.conv2 = nn.Sequential(
                self.conv3x3(dim, dim, stride),
                nn.InstanceNorm2d(dim))
        self.blk1 = nn.Sequential(
                self.conv1x1(dim + dim_extra, dim + dim_extra),
                nn.ReLU(inplace=False),
                self.conv1x1(dim + dim_extra, dim),
                nn.ReLU(inplace=False))
        self.blk2 = nn.Sequential(
                self.conv1x1(dim + dim_extra, dim + dim_extra),
                nn.ReLU(inplace=False),
                self.conv1x1(dim + dim_extra, dim),
                nn.ReLU(inplace=False))
        model = []
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.blk1.apply(gaussian_weights_init)
        self.blk2.apply(gaussian_weights_init)

    def forward(self, x, z):
        residual = x
        z_expand = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
        o1 = self.conv1(x)
        o2 = self.blk1(torch.cat([o1, z_expand], dim=1))
        o3 = self.conv2(o2)
        out = self.blk2(torch.cat([o3, z_expand], dim=1))
        out += residual
        return out

class GaussianNoiseLayer(nn.Module):

    def __init__(self,):
        super(GaussianNoiseLayer, self).__init__()
    def forward(self, x):
        if self.training == False:
            return x
        noise = Variable(torch.randn(x.size()).cuda(x.get_device()))
        return x + noise

class ReLUINSConv2d(nn.Module):

    def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
        super(ReLUINSConv2d, self).__init__()
        model = [nn.ReflectionPad2d(padding)]
        model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
        model += [nn.InstanceNorm2d(n_out, affine=False)]
        model += [nn.ReLU(inplace=False)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
    def forward(self, x):
        return self.model(x)


## The code of LayerNorm is modified from MUNIT (https://github.com/NVlabs/MUNIT)
class LayerNorm(nn.Module):

    def __init__(self, n_out, eps=1e-5, affine=True):
        super().__init__()
        self.n_out = n_out
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
            self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))
        return

    def forward(self, x):
        normalized_shape = x.size()[1:]
        if self.affine:
            return F.layer_norm(x, normalized_shape, self.weight.expand(normalized_shape), self.bias.expand(normalized_shape))
        else:
            return F.layer_norm(x, normalized_shape)


class BasicBlock(nn.Module):

    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super().__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += conv3x3(inplanes, inplanes)
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [convMeanpool(inplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = meanpoolConv(inplanes, outplanes)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out


class ResLeakyReLUConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0, norm='None', sn=False):
        super().__init__()
        model = []
        if sn:
            model += [spectral_norm(nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=1, bias=True))]
        else:
            model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)]
        if 'norm' == 'Instance':
            model += [nn.InstanceNorm2d(n_out, affine=False)]
        model += [nn.LeakyReLU(inplace=False)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x) + x

####################################################################
#--------------------- Spectral Normalization ---------------------
#  This part of code is copied from pytorch master branch (0.5.0)
####################################################################
class SpectralNorm(object):
    def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError(
                'Expected n_power_iterations to be positive, but '
                'got n_power_iterations={}'.format(n_power_iterations)
            )

        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(
                self.dim,
                *[d for d in range(weight_mat.dim()) if d != self.dim],
            )

        height = weight_mat.size(0)
        weight_mat = weight_mat.reshape(height, -1)
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                v = F.normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
                u = F.normalize(torch.matmul(weight_mat, v), dim=0, eps=self.eps)

        sigma = torch.dot(u, torch.matmul(weight_mat, v))
        weight = weight / sigma
        return weight, u
    def remove(self, module):
        weight = getattr(module, self.name)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name, torch.nn.Parameter(weight))

    def __call__(self, module, inputs):
        if module.training:
            weight, u = self.compute_weight(module)
            setattr(module, self.name, weight)
            setattr(module, self.name + '_u', u)
        else:
            r_g = getattr(module, self.name + '_orig').requires_grad
            getattr(module, self.name).detach_().requires_grad_(r_g)

    @staticmethod
    def apply(module, name, n_power_iterations, dim, eps):
        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]
        height = weight.size(dim)
        u = F.normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=fn.eps)
        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        module.register_buffer(fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_forward_pre_hook(fn)
        return fn

def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
    if dim is None:
        if isinstance(
            module,
            (
                torch.nn.ConvTranspose1d,
                torch.nn.ConvTranspose2d,
                torch.nn.ConvTranspose3d,
            )
        ):
            dim = 1
        else:
            dim = 0

    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module

def remove_spectral_norm(module, name='weight'):
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module
    raise ValueError("spectral_norm of '{}' not found in {}".format(name, module))

# x is tensor, scale is pooling scale
def my_pooling(x, scale):
    if x.shape[2] % scale != 0 or x.shape[3] % scale != 0:
        raise NotImplementedError('not suitable input for pooling layer')

    poolinglayer  = nn.AvgPool2d(kernel_size=scale, stride=scale, padding=0)
    c_rep = x.shape[1]  
    x_rep = x.shape[2] // scale
    y_rep = x.shape[3] // scale
    x = poolinglayer(x)

    #import pdb;pdb.set_trace()

    for i in range(x_rep):
        for j in range(y_rep):
            for k in range(c_rep):
                rep = x[:, k, i, j].repeat(1, 1, scale, scale)
                if k == 0:
                    output = rep
                else:
                    output = torch.cat([output,rep], dim=1)

            if j == 0:
                moutput = output 
            else:
                moutput = torch.cat([moutput, output], dim=3)

        if i == 0:
            foutput = moutput
        else:
            foutput = torch.cat([foutput, moutput], dim=2)

    return foutput
