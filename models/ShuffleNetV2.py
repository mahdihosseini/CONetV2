import torch
from torch import Tensor
import torch.nn as nn
# from .._internally_replaced_utils import load_state_dict_from_url
from typing import Callable, Any, List


__all__ = [
    'ShuffleNetV2', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
    'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0'
]

model_urls = {
    'shufflenetv2_x0.5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
    'shufflenetv2_x1.5': None,
    'shufflenetv2_x2.0': None,
}


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int
    ) -> None:
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1), f'{self.stride}, {inp}, {branch_features << 1}, {oup}, {branch_features} '

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(
        i: int,
        o: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False
    ) -> nn.Conv2d:
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(
        self,
        stages_repeats,
        stages_out_channels,
        num_classes= 100,
        inverted_residual: Callable[..., nn.Module] = InvertedResidual
    ) -> None:
       
        # if global_config.CONFIG['dataset'] == 'CIFAR10':
        #     num_classes = 10
        # elif global_config.CONFIG['dataset'] == 'CIFAR100':
        #     num_classes = 100

        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels
        # [[16], [16,16,6,16], [16,16,6,16,16,16,6,16], [16,16,16,16], [16]]
        stride_val = 1
        input_channels = 3
        output_channels = self._stage_out_channels[0][0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride = stride_val, padding = 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential
        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        stage_num = 2
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
            print(stage_num)
            print(output_channels)
            # print("KDSNDJ")
            print(input_channels, output_channels[0])
            seq = [inverted_residual(input_channels, output_channels[0], 2)]
            i = 0
            for i in range(repeats - 1):
                # print("KDSNDJ")
                print(output_channels[i], output_channels[i+1])
                print('layer', i)
                seq.append(inverted_residual(output_channels[i], output_channels[i+1], 1))
            
            setattr(self, name, nn.Sequential(*seq))
            
            input_channels = output_channels[-1]
            stage_num+= 1 
        
        output_channels = 1024
        # output_channels = self._stage_out_channels[-1][0]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(output_channels, num_classes)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # globalpool
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


# def _shufflenetv2(arch: str, *args: Any, **kwargs: Any) -> ShuffleNetV2:
#     model = ShuffleNetV2(*args)

#     # if pretrained:
#     #     model_url = model_urls[arch]
#     #     if model_url is None:
#     #         raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
#     #     else:
#     #         state_dict = load_state_dict_from_url(model_url, progress=progress)
#     #         model.load_state_dict(state_dict)

#     return model



def shufflenetv2(num_classes_input = 10,new_output_sizes=None,new_kernel_sizes=None, global_config = None):

# def shufflenetv2(default = False, pretrained: bool = False, progress: bool = True, channelSize = None, class_num = None, **kwargs: Any) -> ShuffleNetV2:
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    #Blocks 4, 8, 4 = 16
    if(new_output_sizes == None):
        return  ShuffleNetV2(stages_repeats=[4, 8, 4], stages_out_channels=[[16], [16,16,16,16], [16,16,16,16,16,16,16,16], [16,16,16,16], [16]])
        # new_channel_sizes = [[16], [16,16,16,16], [16,16,16,16,16,16,16,16], [16,16,16,16], [16]]
        # ShuffleNetV2(stages_repeats=[4, 8, 4], stages_out_channels=[[24], [116,116,116,116], [232,232,232,232,232,232,232,232], [464,464,464,464], [1024]])
    else:
        raise ValueError('UNKNOWN')
    # else:
    #     return ShuffleNetV2(stages_repeats=[4, 8, 4], stages_out_channels=new_output_sizes,num_classes=num_classes_input)



def update_shuffle_network(new_channel_sizes_list,new_kernel_sizes=None, global_config=None, class_num = 10, predefined = False):

    # class_num = 100
    # if global_config.CONFIG['dataset'] == 'CIFAR10':
    #     class_num = 10
    # elif global_config.CONFIG['dataset'] == 'CIFAR100':
    #     class_num = 100
    if not predefined:
        # if global_config.CONFIG['network']=='shufflenetv2':
        # if True:
        print(new_channel_sizes_list)

        # {0: 0, 3: 1, 6: 2, 9: 3, 12: 4, 15: 5, 18: 6, 21: 7, 24: 8, 27: 9, 30: 10, 33: 11, 36: 12, 39: 13, 42: 14, 
        # 45: 15, 48: 16, 51: 17, 54: 18, 57: 19, 60: 20, 63: 21, 66: 22, 69: 23, 72: 24, 75: 25, 78: 26, 81: 27, 
        # 84: 28, 87: 29, 90: 30, 93: 31, 96: 32, 99: 33, 102: 34, 105: 35, 108: 36, 111: 37, 114: 38, 117: 39, 
        # 120: 40, 123: 41, 126: 42, 129: 43, 132: 44, 135: 45, 138: 46, 141: 47, 144: 48, 
        # 147: 49, 150: 50, 153: 51, 156: 52, 159: 53, 162: 54, 165: 55}
        # # Block 1
        # for c_size in enumerate(new_channel_sizes_list[2:15])
        # [c_size for c_size in enumerate(new_channel_sizes_list[2:15]) % 3]


        new_channel_sizes = [[new_channel_sizes_list[0]], [new_channel_sizes_list[5],new_channel_sizes_list[8],new_channel_sizes_list[11],new_channel_sizes_list[14]], 
        [new_channel_sizes_list[19],new_channel_sizes_list[22],new_channel_sizes_list[25],new_channel_sizes_list[28],new_channel_sizes_list[31],new_channel_sizes_list[34],new_channel_sizes_list[37],new_channel_sizes_list[40]], 
        [new_channel_sizes_list[45],new_channel_sizes_list[48],new_channel_sizes_list[51],new_channel_sizes_list[54]], [new_channel_sizes_list[55]]]
    
        new_network=ShuffleNetV2(stages_repeats=[4, 8, 4], stages_out_channels=new_channel_sizes,num_classes=class_num)
        # else:
        #     raise ValueError('Incorrect model provided')
        return new_network

    if predefined:
        # new_channel_sizes = [(116, 4), (232, 8), (464, 4), (1024, 1)]
        new_network=shufflenetv2()
        return new_network

# new_channel_sizes = [[16], [16,16,16,16], [16,16,16,16,16,16,16,16], [16,16,16,16], [16]]
# net = shufflenetv2(new_output_sizes=new_channel_sizes, num_classes_input=10)
# print(net)


# new_channel_sizes = [[16], [16,16,16,16], [16,16,16,16,16,16,16,16], [16,16,16,16], [16]]
# net = update_shuffle_network(new_channel_sizes_list=new_channel_sizes)

# counter = 0
# print("ASTRWNFS")
# for name, param in net.named_parameters():
#     if(len(param.shape) == 4):
#         print(counter, "     ",name, (param).shape)
#     counter += 1

# adjList = {0:[3,9], 3:[6], 6:[18], 9:[12], 12:[15], 15:[18], 18:[21], 21:[24]
#           ,24:[27],27:[30], 30:[33], 33:[36], 36:[39], 39:[42],
		  
		  
# 		  42:[45,51], 45:[48],48:[60], 51:[54], 54:[57], 57:[60], 60:[63], 63:[66], 66:[69]
# 		  , 69:[72], 72:[75], 75:[78], 78:[81], 81:[84], 84:[87], 87:[90], 90:[93], 93:[96]
# 		  , 96:[99], 99:[102], 102:[105], 105:[108], 108:[111], 111:[114], 114:[117], 117:[120],
		  
		  
# 		  120:[123,129], 123:[126], 126:[138], 129:[132], 132:[135], 135:[138]
# 		  , 138:[141], 141:[144], 144:[147], 147:[150], 150:[153], 153:[156], 156:[159], 159:[162],
		  
# 		  162:[165],
# 		  #FC Next
# 		  165:[]}

'''
{0: ['Conv2d', [24, 3, 3, 3]], 3: ['Conv2d', [24, 1, 3, 3]], 6: ['Conv2d', [58, 24, 1, 1]], 9: ['Conv2d', [58, 24, 1, 1]], 12: ['Conv2d', [58, 1, 3, 3]], 15: ['Conv2d', [58, 58, 1, 1]], 18: ['Conv2d', [58, 58, 1, 1]], 21: ['Conv2d', [58, 1, 3, 3]], 24: ['Conv2d', [58, 58, 1, 1]], 27: ['Conv2d', [58, 58, 1, 1]], 30: ['Conv2d', [58, 1, 3, 3]], 33: ['Conv2d', [58, 58, 1, 1]], 36: ['Conv2d', [58, 58, 1, 1]], 39: ['Conv2d', [58, 1, 3, 3]], 42: ['Conv2d', [58, 58, 1, 1]], 45: ['Conv2d', [116, 1, 3, 3]], 48: ['Conv2d', [116, 116, 1, 1]], 51: ['Conv2d', [116, 116, 1, 1]], 54: ['Conv2d', [116, 1, 3, 3]], 57: ['Conv2d', [116, 116, 1, 1]], 60: ['Conv2d', [116, 116, 1, 1]], 63: ['Conv2d', [116, 1, 3, 3]], 66: ['Conv2d', [116, 116, 1, 1]], 69: ['Conv2d', [116, 116, 1, 1]], 72: ['Conv2d', [116, 1, 3, 3]], 75: ['Conv2d', [116, 116, 1, 1]], 78: ['Conv2d', [116, 116, 1, 1]], 81: ['Conv2d', [116, 1, 3, 3]], 84: ['Conv2d', [116, 116, 1, 1]], 87: ['Conv2d', [116, 116, 1, 1]], 90: ['Conv2d', [116, 1, 3, 3]], 93: ['Conv2d', [116, 116, 1, 1]], 96: ['Conv2d', [116, 116, 1, 1]], 99: ['Conv2d', [116, 1, 3, 3]], 102: ['Conv2d', [116, 116, 1, 1]], 105: ['Conv2d', [116, 116, 1, 1]], 108: ['Conv2d', [116, 1, 3, 3]], 111: ['Conv2d', [116, 116, 1, 1]], 114: ['Conv2d', [116, 116, 1, 1]], 117: ['Conv2d', [116, 1, 3, 3]], 120: ['Conv2d', [116, 116, 1, 1]], 123: ['Conv2d', [232, 1, 3, 3]], 126: ['Conv2d', [232, 232, 1, 1]], 129: ['Conv2d', [232, 232, 1, 1]], 132: ['Conv2d', [232, 1, 3, 3]], 135: ['Conv2d', [232, 232, 1, 1]], 138: ['Conv2d', [232, 232, 1, 1]], 141: ['Conv2d', [232, 1, 3, 3]], 144: ['Conv2d', [232, 232, 1, 1]], 147: ['Conv2d', [232, 232, 1, 1]], 150: ['Conv2d', [232, 1, 3, 3]], 153: ['Conv2d', [232, 232, 1, 1]], 156: ['Conv2d', [232, 232, 1, 1]], 159: ['Conv2d', [232, 1, 3, 3]], 162: ['Conv2d', [232, 232, 1, 1]], 165: ['Conv2d', [1024, 464, 1, 1]]}
'''