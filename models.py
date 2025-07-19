import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
import math


#====================== 注意力机制实现 ======================

class ChannelAttention(nn.Module):
    """Channel Attention Module in CBAM"""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module in CBAM"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAMBlock(nn.Module):
    """CBAM Attention Mechanism: Combining Channel and Spatial Attention"""
    def __init__(self, channel, ratio=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


class SEBlock(nn.Module):
    """SE (Squeeze-and-Excitation) attention mechanism"""
    def __init__(self, channel, ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ECABlock(nn.Module):
    """ECA (Efficient Channel Attention) attention mechanism"""
    def __init__(self, channel, gamma=2, b=1):
        super(ECABlock, self).__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


class OnlySpatialAttention(nn.Module):
    """Module using only spatial attention"""
    def __init__(self, kernel_size=7):
        super(OnlySpatialAttention, self).__init__()
        self.spatial = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        return x * self.spatial(x)


class OnlyChannelAttention(nn.Module):
    """Module using only channel attention"""
    def __init__(self, channel, ratio=16):
        super(OnlyChannelAttention, self).__init__()
        self.channel = ChannelAttention(channel, ratio=ratio)

    def forward(self, x):
        return x * self.channel(x)


class ResNet18WithAttention(nn.Module):
    def __init__(self, num_classes, attention_type='cbam'):
        super(ResNet18WithAttention, self).__init__()
        model = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(model.children())[:-2])  
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        
        self.attention_type = attention_type
        if attention_type == 'cbam':
            self.attention = CBAMBlock(512)
        elif attention_type == 'se':
            self.attention = SEBlock(512)
        elif attention_type == 'eca':
            self.attention = ECABlock(512)
        elif attention_type == 'spatial':
            self.attention = OnlySpatialAttention()
        elif attention_type == 'channel':
            self.attention = OnlyChannelAttention(512)
        else:  
            self.attention = None

    def forward(self, x):
        x = self.backbone(x)
        if self.attention is not None:
            x = self.attention(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet50WithAttention(nn.Module):
    def __init__(self, num_classes, attention_type='cbam'):
        super(ResNet50WithAttention, self).__init__()
        model = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
        
        
        self.attention_type = attention_type
        if attention_type == 'cbam':
            self.attention = CBAMBlock(2048)
        elif attention_type == 'se':
            self.attention = SEBlock(2048)
        elif attention_type == 'eca':
            self.attention = ECABlock(2048)
        elif attention_type == 'spatial':
            self.attention = OnlySpatialAttention()
        elif attention_type == 'channel':
            self.attention = OnlyChannelAttention(2048)
        else:  
            self.attention = None

    def forward(self, x):
        x = self.backbone(x)
        if self.attention is not None:
            x = self.attention(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class SqueezeNetWithAttention(nn.Module):
    def __init__(self, num_classes, attention_type='cbam'):
        super(SqueezeNetWithAttention, self).__init__()
        model = models.squeezenet1_1(pretrained=True)
        self.backbone = model.features
        
        
        self.attention_type = attention_type
        if attention_type == 'cbam':
            self.attention = CBAMBlock(512)
        elif attention_type == 'se':
            self.attention = SEBlock(512)
        elif attention_type == 'eca':
            self.attention = ECABlock(512)
        elif attention_type == 'spatial':
            self.attention = OnlySpatialAttention()
        elif attention_type == 'channel':
            self.attention = OnlyChannelAttention(512)
        else:  
            self.attention = None
            
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        if self.attention is not None:
            x = self.attention(x)
        x = self.classifier(x)
        return x


class ShuffleNetV2WithAttention(nn.Module):
    def __init__(self, num_classes, attention_type='cbam'):
        super(ShuffleNetV2WithAttention, self).__init__()
        model = models.shufflenet_v2_x1_0(pretrained=True)
        self.backbone = model.conv1
        self.maxpool = model.maxpool
        self.stage2 = model.stage2
        self.stage3 = model.stage3
        self.stage4 = model.stage4
        self.conv5 = model.conv5
        
        
        self.attention_type = attention_type
        if attention_type == 'cbam':
            self.attention = CBAMBlock(1024)
        elif attention_type == 'se':
            self.attention = SEBlock(1024)
        elif attention_type == 'eca':
            self.attention = ECABlock(1024)
        elif attention_type == 'spatial':
            self.attention = OnlySpatialAttention()
        elif attention_type == 'channel':
            self.attention = OnlyChannelAttention(1024)
        else:  
            self.attention = None
            
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        if self.attention is not None:
            x = self.attention(x)
        x = self.classifier(x)
        return x


class MobileNetV3WithAttention(nn.Module):
    def __init__(self, num_classes, attention_type='cbam'):
        super(MobileNetV3WithAttention, self).__init__()
        model = models.mobilenet_v3_small(pretrained=True)
        self.backbone = model.features
        
        
        self.attention_type = attention_type
        if attention_type == 'cbam':
            self.attention = CBAMBlock(576)
        elif attention_type == 'se':
            self.attention = SEBlock(576)
        elif attention_type == 'eca':
            self.attention = ECABlock(576)
        elif attention_type == 'spatial':
            self.attention = OnlySpatialAttention()
        elif attention_type == 'channel':
            self.attention = OnlyChannelAttention(576)
        else:  
            self.attention = None
            
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(576, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        if self.attention is not None:
            x = self.attention(x)
        x = self.classifier(x)
        return x


def get_model(model_name, num_classes, attention_type='cbam'):
    """
    Retrieve the specified model
    
    Args:
        model_name: Model name, support 'resnet18', 'resnet50', 'squeezenet', 'shufflenetv2', 'mobilenetv3'
        num_classes: Number of categories
        attention_type: Type of attention mechanism, support 'cbam', 'se', 'eca', 'spatial', 'channel', 'none'
        
    Returns:
        model: Model instance
    """
    if model_name == 'resnet18':
        return ResNet18WithAttention(num_classes, attention_type)
    elif model_name == 'resnet50':
        return ResNet50WithAttention(num_classes, attention_type)
    elif model_name == 'squeezenet':
        return SqueezeNetWithAttention(num_classes, attention_type)
    elif model_name == 'shufflenetv2':
        return ShuffleNetV2WithAttention(num_classes, attention_type)
    elif model_name == 'mobilenetv3':
        return MobileNetV3WithAttention(num_classes, attention_type)
    else:
        raise ValueError(f"Unsupported model: {model_name}") 