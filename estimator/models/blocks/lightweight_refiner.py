

import torch
import torch.nn as nn

import torch.nn.functional as F
from estimator.registry import MODELS
import timm
from torch.nn.parameter import Parameter
from estimator.models.blocks.convs import SingleConvCNNLN, DoubleConv, DoubleResConv
from timm.layers import Conv2dSame
from estimator.models.blocks.transformers import TwoWayTransformer
from external.depth_anything.blocks import FeatureFusionBlock, _make_scratch

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
        
        # self.projects = nn.ModuleList([
        #     nn.Conv2d(
        #         in_channels=in_channels,
        #         out_channels=out_channel,
        #         kernel_size=1,
        #         stride=1,
        #         padding=0,
        #     ) for out_channel in out_channels
        # ])
        
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

class UpSample(nn.Module):
    """Upscaling then cat and DoubleConv"""

    def __init__(self, skip, cur, dec_c):
        super().__init__()
        self.skip_conv = DoubleResConv(skip, activation=nn.GELU())
        self.cat_reduce_conv = nn.Sequential(
            nn.Conv2d(skip + cur, dec_c, kernel_size=3, padding=1, bias=False),
            nn.GELU())
        self.fusion_conv = DoubleResConv(dec_c, activation=nn.GELU())
        
    
    def forward(self, cur_x, skip_x):
        ''' Args:
            skip_x: the feature map from the skip connection
            cur_x: the feature map list from the encoder and everything you want to concate with current feat maps
        '''
        
        skip_x = self.skip_conv(skip_x)
        
        cur_x = F.interpolate(cur_x, skip_x.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([cur_x, skip_x], dim=1)
        x = self.cat_reduce_conv(x)
        
        x = self.fusion_conv(x)
        
        return x
    
class DepthResDecoder(nn.Module):
    def __init__(self, in_channels, proj_channels, out_channels):
        super(DepthResDecoder, self).__init__()
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=chl,
                out_channels=out_chl,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for chl, out_chl in zip(in_channels, proj_channels)])
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.proj_channels = proj_channels
        
        self.btlnk = nn.Sequential(
            nn.Conv2d(proj_channels[-1], proj_channels[-1], kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),)
        
        self.decoder_layers = nn.ModuleList()
        _chl = self.proj_channels[::-1][0]
        temp_chl = self.proj_channels[::-1][1:]
        for tmp_c, dec_c in zip(temp_chl, out_channels[::-1][1:]):
            # skip, cur, dec_c      
            layer = UpSample(tmp_c, _chl, dec_c)
            _chl = dec_c
            self.decoder_layers.append(layer)

        head_features_1 = self.out_channels[0]
        head_features_2 = self.out_channels[0]
        
        self.output_conv1 = nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1)
        
        self.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True))
        
        self.output_conv3 = nn.Sequential(
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True))
        
        # nn.init.normal_(self.output_conv3[0].weight, mean=1.0)
        # nn.init.constant_(self.output_conv3[0].bias, 0)
    
    def forward(self, out_features):
        
        # up to bottom
        layer_1, layer_2, layer_3, layer_4, layer_5 = out_features
        
        layer_1_rn = self.projects[0](layer_1)
        layer_2_rn = self.projects[1](layer_2)
        layer_3_rn = self.projects[2](layer_3)
        layer_4_rn = self.projects[3](layer_4)
        layer_5_rn = self.projects[4](layer_5)
        
        layer_5_rn = self.btlnk(layer_5_rn)
        
        path_5 = self.decoder_layers[0](layer_5_rn, layer_4_rn)
        path_4 = self.decoder_layers[1](path_5, layer_3_rn)
        path_3 = self.decoder_layers[2](path_4, layer_2_rn)
        path_2 = self.decoder_layers[3](path_3, layer_1_rn)
        path_1 = F.interpolate(path_2, scale_factor=2, mode='bilinear', align_corners=True)
        
        out = self.output_conv1(path_1)
        last_feat = self.output_conv2(out)
        out = self.output_conv3(last_feat)
        # print(torch.min(last_feat), torch.max(last_feat), torch.min(out), torch.max(out))
        
        feats = [layer_5_rn, path_5, path_4, path_3, path_2, last_feat]
        return feats, out
    
# from zoedepth.models.zoedepth import ZoeDepth
@MODELS.register_module()
class LightWeightRefiner(nn.Module):
    def __init__(
        self, 
        encoder_name,
        coarse_condition,
        encoder_channels=[16, 24, 40, 112, 960],
        proj_channels=[16, 24, 40, 112, 960],
        decoder_channels=[32, 64, 128, 256, 512],
        coarse_feat_chl=[256, 256, 256, 256, 256, 32],
        with_decoder=False,
        cls_pretrain=True,):
        
        super(LightWeightRefiner, self).__init__()
        # process fine model
        
        self.encoder_name = encoder_name
        self.cls_pretrain = cls_pretrain
        if self.cls_pretrain:
            self.refiner_encoder = timm.create_model(encoder_name, pretrained=True, features_only=True)
        else:
            self.refiner_encoder = timm.create_model(encoder_name, pretrained=False, features_only=True)
            
        self.register_buffer("refiner_pixel_mean", torch.Tensor(self.refiner_encoder.default_cfg['mean']).view(-1, 1, 1), False)
        self.register_buffer("refiner_pixel_std", torch.Tensor(self.refiner_encoder.default_cfg['std']).view(-1, 1, 1), False)
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.proj_channels = proj_channels
        self.coarse_feat_chl = coarse_feat_chl
        self.coarse_condition = coarse_condition
        self.with_decoder = with_decoder
        
        if self.with_decoder:
            self.decoder = SimpleDPTHead(in_channels=32, features=256, use_bn=False, out_channels=encoder_channels)
            # self.decoder = DepthResDecoder(in_channels=encoder_channels, proj_channels=proj_channels, out_channels=decoder_channels)
        
        if 'convnext' in self.encoder_name:
            self.upsample_convx = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=encoder_channels[1],
                    out_channels=encoder_channels[0],
                    kernel_size=2, stride=2),
                nn.ReLU())
            
    def forward(self,
                crop_image,
                coarse_depth=None,
                coarse_feats=None,
                pe_list=None,
                pe_patch_list=None,):
        
        # print(torch.min(crop_image), torch.max(crop_image), self.refiner_pixel_mean, self.refiner_pixel_std)
        crop_image = (crop_image - self.refiner_pixel_mean) / self.refiner_pixel_std
        
        if self.coarse_condition:
            refiner_features = self.refiner_encoder(torch.cat([crop_image, coarse_depth], dim=1))
        else:
            refiner_features = self.refiner_encoder(crop_image)
        
        if self.with_decoder:
            if 'convnext' in self.encoder_name:
                raise NotImplementedError
                # refiner_features.insert(0, F.interpolate(refiner_features[0], scale_factor=2, mode='bilinear', align_corners=True))
            feats, out_depth = self.decoder(refiner_features)
        else:
            # feats, out_depth = refiner_features[::-1], torch.zeros_like(crop_image[:, :1, :, :])
            if 'convnext' in self.encoder_name:
                high_level_feat = refiner_features[0]
                # high_level_feat_upsample = F.interpolate(high_level_feat, scale_factor=2, mode='bilinear', align_corners=True)
                high_level_feat_upsample = self.upsample_convx(high_level_feat)
                refiner_features.insert(0, high_level_feat_upsample)
                high_level_feat_upsample = F.interpolate(high_level_feat_upsample, scale_factor=2, mode='bilinear', align_corners=True)
                refiner_features.insert(0, high_level_feat_upsample)
            else:
                high_level_feat = refiner_features[0]
                high_level_feat_upsample = F.interpolate(high_level_feat, scale_factor=2, mode='bilinear', align_corners=True)
                refiner_features.insert(0, high_level_feat_upsample)
                
            feats = refiner_features[::-1]
            out_depth = torch.zeros_like(crop_image[:, :1, :, :])

        return feats, out_depth