import torch.nn as nn
from dependency import DependencyList
# from torchsummary import summary


import torch
from torch import nn
from torch import Tensor
# from .._internally_replaced_utils import load_state_dict_from_url
from typing import Callable, Any, Optional, List


__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                      bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes


# necessary for backwards compatibility
ConvBNReLU = ConvBNActivation


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes_input: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None, new_conv_sizes = None, global_config = None, ch_in = None
    ) -> None:
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        stride_val = 1
        # input_channel = 32
        last_channel = 1280
        if not new_conv_sizes:
            inverted_residual_setting=[
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 16, 1, stride_val], # NOTE: change stride 2 -> 1 for CIFAR10
                [6, 16, 1, 1],
                [6, 16, 1, 2],
                [6, 16, 1, 1],
                [6, 16, 1, 1],
                [6, 16, 1, 2],
                [6, 16, 1, 1],
                [6, 16, 1, 1],
                [6, 16, 1, 1],
                [6, 16, 1, 1],
                [6, 16, 1, 1],
                [6, 16, 1, 1],
                [6, 16, 1, 2],
                [6, 16, 1, 1],
                [6, 16, 1, 1],
                [6, 16, 1, 1]
            ]
            input_channel = _make_divisible(16 * width_mult, round_nearest)
            # inverted_residual_setting = [
            #     # t, c, n, s
            #     [1, 16, 1, 1],
            #     [6, 24, 2, 2],
            #     [6, 32, 3, 2],
            #     [6, 64, 4, 2],
            #     [6, 96, 3, 1],
            #     [6, 160, 3, 2],
            #     [6, 320, 1, 1],
            # ]
        else:
            input_channel = new_conv_sizes[0]
            # self.stem_conv = conv3x3(ch_in, input_channel, stride=2)
            new_conv_sizes = new_conv_sizes[1:]
            input_channel = _make_divisible(input_channel * width_mult, round_nearest)
            inverted_residual_setting=[
                # t, c, n, s
                [1, new_conv_sizes[0], 1, 1],
                [6, new_conv_sizes[1], 1, stride_val], # NOTE: change stride 2 -> 1 for CIFAR10
                [6, new_conv_sizes[2], 1, 1],
                [6, new_conv_sizes[3], 1, 2],
                [6, new_conv_sizes[4], 1, 1],
                [6, new_conv_sizes[5], 1, 1],
                [6, new_conv_sizes[6], 1, 2],
                [6, new_conv_sizes[7], 1, 1],
                [6, new_conv_sizes[8], 1, 1],
                [6, new_conv_sizes[9], 1, 1],
                [6, new_conv_sizes[10], 1, 1],
                [6, new_conv_sizes[11], 1, 1],
                [6, new_conv_sizes[12], 1, 1],
                [6, new_conv_sizes[13], 1, 2],
                [6, new_conv_sizes[14], 1, 1],
                [6, new_conv_sizes[15], 1, 1],
                [6, new_conv_sizes[16], 1, 1]
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [ConvBNReLU(3, input_channel, stride=stride_val, norm_layer=norm_layer)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes_input),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


# def mobilenet_v2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV2:
#     """
#     Constructs a MobileNetV2 architecture from
#     `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     model = MobileNetV2(**kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model

# class MobileNetV2(nn.Module):
#     def __init__(self, num_classes_input=1000, ch_in=3, new_conv_sizes = None, global_config = None):
#         super(MobileNetV2, self).__init__()
#         print(new_conv_sizes)
#         if not new_conv_sizes:
#             print('*'*200)
#             self.configs=[
#                 # t, c, n, s
#                 [1, 16, 1, 1],
#                 [6, 16, 1, 2],
#                 [6, 16, 1, 1],
#                 [6, 16, 1, 2],
#                 [6, 16, 1, 1],
#                 [6, 16, 1, 1],
#                 [6, 16, 1, 2],
#                 [6, 16, 1, 1],
#                 [6, 16, 1, 1],
#                 [6, 16, 1, 1],
#                 [6, 16, 1, 1],
#                 [6, 16, 1, 1],
#                 [6, 16, 1, 1],
#                 [6, 16, 1, 2],
#                 [6, 16, 1, 1],
#                 [6, 16, 1, 1],
#                 [6, 16, 1, 1]
#             ]
#             # self.configs=[
#             #     # t, c, n, s
#             #     [1, 16, 1, 1],
#             #     [6, 24, 1, 2],
#             #     [6, 24, 1, 1],
#             #     [6, 32, 1, 2],
#             #     [6, 32, 1, 1],
#             #     [6, 32, 1, 1],
#             #     [6, 64, 1, 2],
#             #     [6, 64, 1, 1],
#             #     [6, 64, 1, 1],
#             #     [6, 64, 1, 1],
#             #     [6, 96, 1, 1],
#             #     [6, 96, 1, 1],
#             #     [6, 96, 1, 1],
#             #     [6, 160, 1, 2],
#             #     [6, 160, 1, 1],
#             #     [6, 160, 1, 1],
#             #     [6, 320, 1, 1]
#             # ]
#             self.stem_conv = conv3x3(ch_in, 16, stride=2)
#             input_channel = 16
#         else:
#             input_channel = new_conv_sizes[0]
#             self.stem_conv = conv3x3(ch_in, input_channel, stride=2)
#             new_conv_sizes = new_conv_sizes[1:]
#             assert len(new_conv_sizes) == 17, f'incorrect conv sizes list provided, should be 17 but is {len(new_conv_sizes)}'
#             self.configs=[
#                 # t, c, n, s
#                 [1, new_conv_sizes[0], 1, 1],
#                 [6, new_conv_sizes[1], 1, 2],
#                 [6, new_conv_sizes[2], 1, 1],
#                 [6, new_conv_sizes[3], 1, 2],
#                 [6, new_conv_sizes[4], 1, 1],
#                 [6, new_conv_sizes[5], 1, 1],
#                 [6, new_conv_sizes[6], 1, 2],
#                 [6, new_conv_sizes[7], 1, 1],
#                 [6, new_conv_sizes[8], 1, 1],
#                 [6, new_conv_sizes[9], 1, 1],
#                 [6, new_conv_sizes[10], 1, 1],
#                 [6, new_conv_sizes[11], 1, 1],
#                 [6, new_conv_sizes[12], 1, 1],
#                 [6, new_conv_sizes[13], 1, 2],
#                 [6, new_conv_sizes[14], 1, 1],
#                 [6, new_conv_sizes[15], 1, 1],
#                 [6, new_conv_sizes[16], 1, 1]
#             ]


#         # self.configs=[
#         #     # t, c, n, s
#         #     [1, 16, 1, 1],
#         #     [6, 24, 2, 2],
#         #     [6, 32, 3, 2],
#         #     [6, 64, 4, 2],
#         #     [6, 96, 3, 1],
#         #     [6, 160, 3, 2],
#         #     [6, 320, 1, 1]
#         # ]

        

#         layers = []
#         # input_channel = new_conv_sizes[0]
#         for t, c, n, s in self.configs:
#             for i in range(n):
#                 stride = s if i == 0 else 1
#                 layers.append(InvertedBlock(ch_in=input_channel, ch_out=c, expand_ratio=t, stride=stride))
#                 input_channel = c

#         self.layers = nn.Sequential(*layers)

#         self.last_conv = conv1x1(input_channel, 1280)

#         self.classifier = nn.Sequential(
#             nn.Dropout2d(0.2),
#             nn.Linear(1280, num_classes_input)
#         )
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)

#     def forward(self, x):
#         x = self.stem_conv(x)
#         x = self.layers(x)
#         x = self.last_conv(x)
#         x = self.avg_pool(x).view(-1, 1280)
#         x = self.classifier(x)
#         return x

def update_mobilenet(new_channel_sizes_list,new_kernel_sizes, global_config,class_num = None, predefined = False):
    assert class_num is not None, 'class number should not be none'
    # class_num = 0
    # if global_config.CONFIG['dataset'] == 'CIFAR10':
    #     class_num = 10
    # elif global_config.CONFIG['dataset'] == 'CIFAR100':
    #     class_num = 100
    if not predefined:
        new_channel_sizes = [new_channel_sizes_list[0]]
        for index,i in enumerate(new_channel_sizes_list[2:]):
            if (index)%3 == 0:
                new_channel_sizes.append(i)


        # new_channel_sizes = [new_channel_sizes_list[0:7],
        #                     new_channel_sizes_list[7:9] + new_channel_sizes_list[10:16],
        #                     new_channel_sizes_list[16:18] + new_channel_sizes_list[19:29],
        #                     new_channel_sizes_list[29:31] + new_channel_sizes_list[32:]]
        new_network = MobileNetV2(num_classes_input=class_num, ch_in=3, new_conv_sizes = new_channel_sizes)
        # new_network=DASNet34(num_classes_input=class_num,new_output_sizes=new_channel_sizes,new_kernel_sizes=new_kernel_sizes)
        return new_network
    elif predefined:
        new_channel_sizes = [32,16,24,24,32,32,32,64,64,64,64,96,96,96,160,160,160,320]
        new_network = MobileNetV2(num_classes_input=class_num, ch_in=3, new_conv_sizes = new_channel_sizes)
        return new_network
    else:
        raise ValueError('Incorrect model provided')
    # elif global_config.CONFIG['network']=='DASNet50':
    #     new_network=DASNet50(num_classes_input=class_num,new_output_sizes=new_channel_sizes,new_kernel_sizes=new_kernel_sizes)
    

if __name__=="__main__":
    # model check
    model = MobileNetV2(ch_in=3, num_classes_input=1000)
    DependencyList(model_type = 'mobilenetv2', model = model)
    print(model)
    # summary(model, (3, 224, 224), device='cpu')