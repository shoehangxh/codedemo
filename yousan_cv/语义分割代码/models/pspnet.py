import torch
import torch.nn as nn
import torch.nn.functional as F

from models.segbase import SegBaseModel
from models.fcn import _FCNHead

__all__ = ['PSPNet', 'get_psp', 'get_psp_resnet50_voc', 'get_psp_resnet50_ade', 'get_psp_resnet101_voc',
           'get_psp_resnet101_ade', 'get_psp_resnet101_citys', 'get_psp_resnet101_coco']


class PSPNet(SegBaseModel):
    # 参数预定义
    def __init__(self, nclass, backbone='resnet50', aux=False, pretrained_base=True, **kwargs):
        super(PSPNet, self).__init__(nclass, aux, backbone, pretrained_base=pretrained_base, **kwargs)
        # PSPHead
        self.head = _PSPHead(nclass, **kwargs)
        # 辅助训练
        if self.aux:
            self.auxlayer = _FCNHead(1024, nclass, **kwargs)
        #
        self.__setattr__('exclusive', ['head', 'auxlayer'] if aux else ['head'])
    # 前向传播
    def forward(self, x):
        # 获取图像尺寸维度
        size = x.size()[2:]
        # 获取resnet中第3、4层特征
        _, _, c3, c4 = self.base_forward(x)
        outputs = []
        # 获取PSPNet主分支的特征
        x = self.head(c4)
        # 上采样
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)
        # 是否辅助训练
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True) # 上采样
            outputs.append(auxout)
        return tuple(outputs)

    # return parameters
    def trainable_parameters(self): #优化器的优化参数时，可以对不同部分的参数使用不同的优化器
        return list(self.head.parameters(),self.pretrained.parameters())


def _PSP1x1Conv(in_channels, out_channels, norm_layer, norm_kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
        nn.ReLU(True)
    )


# 金字塔池化结构
class _PyramidPooling(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(_PyramidPooling, self).__init__()
        out_channels = int(in_channels / 4)
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)
        self.avgpool2 = nn.AdaptiveAvgPool2d(2)
        self.avgpool3 = nn.AdaptiveAvgPool2d(3)
        self.avgpool4 = nn.AdaptiveAvgPool2d(6)
        self.conv1 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
        self.conv2 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
        self.conv3 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
        self.conv4 = _PSP1x1Conv(in_channels, out_channels, **kwargs)

    def forward(self, x):
        size = x.size()[2:] # return w and h
        feat1 = F.interpolate(self.conv1(self.avgpool1(x)), size, mode='bilinear', align_corners=True)
        feat2 = F.interpolate(self.conv2(self.avgpool2(x)), size, mode='bilinear', align_corners=True)
        feat3 = F.interpolate(self.conv3(self.avgpool3(x)), size, mode='bilinear', align_corners=True)
        feat4 = F.interpolate(self.conv4(self.avgpool4(x)), size, mode='bilinear', align_corners=True)
        return torch.cat([x, feat1, feat2, feat3, feat4], dim=1)

# 池化层后经过的卷积分支
class _PSPHead(nn.Module):
    def __init__(self, nclass, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_PSPHead, self).__init__()
        self.psp = _PyramidPooling(2048, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        self.block = nn.Sequential(
            nn.Conv2d(4096, 512, 3, padding=1, bias=False),
            norm_layer(512, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(512, nclass, 1)
        )

    def forward(self, x):
        x = self.psp(x)
        return self.block(x)


def get_psp(dataset='pascal_voc', backbone='resnet50', pretrained=False, root='~/.torch/models',
            pretrained_base=True, **kwargs):
    """
    Example：

    >>> model = get_psp(dataset='pascal_voc', backbone='resnet50', pretrained=False)
    >>> print(model)
    """
    acronyms = {
        'pascal_voc': 'pascal_voc',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    from dataloader import datasets
    model = PSPNet(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=False, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        device = torch.device(kwargs['local_rank'])
        model.load_state_dict(torch.load(get_model_file('psp_%s_%s' % (backbone, acronyms[dataset]), root=root),
                              map_location=device))
    return model


def get_psp_resnet50_voc(**kwargs):
    return get_psp('pascal_voc', 'resnet50', **kwargs)


def get_psp_resnet50_ade(**kwargs):
    return get_psp('ade20k', 'resnet50', **kwargs)


def get_psp_resnet101_voc(**kwargs):
    return get_psp('pascal_voc', 'resnet101', **kwargs)


def get_psp_resnet101_ade(**kwargs):
    return get_psp('ade20k', 'resnet101', **kwargs)


def get_psp_resnet101_citys(**kwargs):
    return get_psp('citys', 'resnet101', **kwargs)


def get_psp_resnet101_coco(**kwargs):
    return get_psp('coco', 'resnet101', **kwargs)


if __name__ == '__main__':
     model = PSPNet(nclass=21,pretrained_base=False)
     model.eval()
     input = torch.randn(2, 3, 40 ,40)
     output = model(input)
     input_names = ['input']
     out_names = ['output']
     torch.onnx.export(model, input, 'E:/dataset/nn/test1.onnx', input_names=input_names, output_names=out_names,
                       verbose=True, opset_version=11)
     predict = torch.argmax(output[0], 1) + 1