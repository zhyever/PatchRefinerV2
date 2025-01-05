

import torch
import torch.nn as nn

from depth_anything.blocks import FeatureFusionBlock, _make_scratch
import torch.nn.functional as F
from estimator.registry import MODELS
import timm
from torch.nn.parameter import Parameter
from estimator.models.blocks.convs import SingleConvCNNLN, DoubleConv, SingleConv
from timm.layers import Conv2dSame

def _make_scratch_simple(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    out_shape5 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape*2
        out_shape3 = out_shape*4
        out_shape4 = out_shape*8
        out_shape5 = out_shape*16

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer5_rn = nn.Conv2d(
        in_shape[4], out_shape5, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    
    return scratch

def _make_fusion_block(features, use_bn, size = None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class SimpleDPTHead(nn.Module):
    def __init__(self, in_channels, features=256, use_bn=False, out_channels=[256, 512, 1024, 1024]):
        super(SimpleDPTHead, self).__init__()
        
        # out_channels = [in_channels // 8, in_channels // 4, in_channels // 2, in_channels]
        # out_channels = [in_channels // 4, in_channels // 2, in_channels, in_channels]
        # out_channels = [in_channels, in_channels, in_channels, in_channels]
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.scratch = _make_scratch_simple(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet5 = _make_fusion_block(features, use_bn)
        

        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )
        
        self.scratch.output_conv3 = nn.Sequential(
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            # nn.ReLU(True),
            # nn.Identity(),
        )
        
        nn.init.normal_(self.scratch.output_conv3[0].weight, mean=1.0)
        nn.init.constant_(self.scratch.output_conv3[0].bias, 0)
    
    def forward(self, out_features):
        
        # up to bottom
        layer_1, layer_2, layer_3, layer_4, layer_5 = out_features
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        layer_5_rn = self.scratch.layer5_rn(layer_5)
        
        path_5 = self.scratch.refinenet5(layer_5_rn, size=layer_4_rn.shape[2:])
        path_4 = self.scratch.refinenet4(path_5, layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        last_feat = self.scratch.output_conv2(out)
        out = self.scratch.output_conv3(last_feat)
        # print(torch.min(last_feat), torch.max(last_feat), torch.min(out), torch.max(out))
        
        feats = [layer_5_rn, path_5, path_4, path_3, path_2, last_feat]
        return feats, out

# from zoedepth.models.zoedepth import ZoeDepth
@MODELS.register_module()
class LightWeightRefinerPG(nn.Module):
    def __init__(
        self, 
        encoder_name,
        coarse_condition,
        encoder_channels=[16, 24, 40, 112, 960],):
        
        super(LightWeightRefinerPG, self).__init__()
        # process fine model
        
        self.refiner_encoder = timm.create_model(encoder_name, pretrained=True, features_only=True)
        self.register_buffer("refiner_pixel_mean", torch.Tensor(self.refiner_encoder.default_cfg['mean']).view(-1, 1, 1), False)
        self.register_buffer("refiner_pixel_std", torch.Tensor(self.refiner_encoder.default_cfg['std']).view(-1, 1, 1), False)
        self.decoder = SimpleDPTHead(in_channels=32, features=256, use_bn=False, out_channels=encoder_channels)
        
        # self.zoe = ZoeDepth.build(**zoe)
        self.coarse_condition = coarse_condition
        
    def forward(self,
                crop_image,
                coarse_depth=None,):
        
        # print(torch.min(crop_image), torch.max(crop_image), self.refiner_pixel_mean, self.refiner_pixel_std)
        crop_image = (crop_image - self.refiner_pixel_mean) / self.refiner_pixel_std
        
        if self.coarse_condition:
            refiner_features = self.refiner_encoder(torch.cat([crop_image, coarse_depth], dim=1))
        else:
            refiner_features = self.refiner_encoder(crop_image)
        
        # refiner_features = self.refiner_encoder(crop_image) # 192, 96, 48, 24, 12
        # deep_model_output_dict = self.zoe(crop_image, return_final_centers=True)
        # deep_features = deep_model_output_dict['temp_features'] # x_d0 1/128, x_blocks_feat_0 1/64, x_blocks_feat_1 1/32, x_blocks_feat_2 1/16, x_blocks_feat_3 1/8, midas_final_feat 1/4 [based on 384x4, 512x4]
        # coarse_prediction = deep_model_output_dict['metric_depth']
        # refiner_features = [
        #     deep_features['x_d0'],
        #     deep_features['x_blocks_feat_0'],
        #     deep_features['x_blocks_feat_1'],
        #     deep_features['x_blocks_feat_2'],
        #     deep_features['x_blocks_feat_3']] # bs, c, h, w
        # for i in refiner_features:
        #     print(i.shape)
        
        feats, out_depth = self.decoder(refiner_features)
        return feats, out_depth