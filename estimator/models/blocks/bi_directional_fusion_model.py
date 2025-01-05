
import torch
import torch.nn as nn
import torch.nn.functional as F
from estimator.registry import MODELS
from estimator.models.blocks.convs import SingleConvCNNLN, LayerNorm
from estimator.models.blocks.lightweight_refiner import SimpleDPTHead, _make_scratch_simple
from estimator.models.blocks.fusion_model import UpSample
from external.depth_anything.blocks import ResidualConvUnit
from estimator.models.blocks.transformers import TwoWayTransformer

def _make_fusion_block(features, use_bn, size=None, gate=True, fusion=True):
    return GatedFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
        gate=gate,
        fusion=fusion,
    )


class GatedConvUnit(nn.Module):
    """Gated convolution module.
    """

    def __init__(self, features, activation, gate=True, fusion=True):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.groups=1

        self.conv = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        
        self.fusion = fusion
        
        if self.fusion:
            self.gate = gate
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(features*2, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups),
                LayerNorm([features]),
                activation,
                nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0, bias=False, groups=self.groups))
            
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()
        
    def forward(self, x, c_feat):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        out = self.activation(x)
        out = self.conv(out)
        
        if self.groups > 1:
            out = self.conv_merge(out)
        out = self.skip_add.add(out, x)
        
        if self.fusion:
            fused_feat = torch.cat([out, c_feat], dim=1)
            fused_feat = self.fusion_conv(fused_feat)
            
            if self.gate:
                gate_att = F.sigmoid(fused_feat)
                out = self.skip_add.mul(out, gate_att)
            else:
                out = fused_feat
        
        return out


class GatedFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None, gate=True, fusion=True):
        """Init.

        Args:
            features (int): number of features
        """
        super(GatedFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        out_features = features
        if self.expand==True:
            out_features = features//2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.GateresConfUnit1 = GatedConvUnit(features, activation, gate=gate, fusion=fusion)
        self.GateresConfUnit2 = GatedConvUnit(features, activation, gate=gate, fusion=fusion)
        
        self.skip_add = nn.quantized.FloatFunctional()

        self.size=size

    def forward(self, *xs, size=None, coarse_feat=None, upscale=True):
        """Forward pass.

        Returns:
            tensor: output
        """
        
        output = xs[0]

        if len(xs) == 2:
            res = self.GateresConfUnit1(xs[1], c_feat=coarse_feat)
            output = self.skip_add.add(output, res)
            
        output = self.GateresConfUnit2(output, c_feat=coarse_feat)

        if upscale:
            if (size is None) and (self.size is None):
                modifier = {"scale_factor": 2}
            elif size is None:
                modifier = {"size": self.size}
            else:
                modifier = {"size": size}

            output = nn.functional.interpolate(
                output, **modifier, mode="bilinear", align_corners=self.align_corners)

            output = self.out_conv(output)
        else:
            output = self.out_conv(output)

        return output
    
class C2FModule(nn.Module):
    def __init__(self, coarse_chl, fine_chl, features=256, use_bn=False, fusion=True, gate=True,):
        super(C2FModule, self).__init__()
        
        self.scratch = _make_scratch_simple(
            fine_chl,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn=use_bn, gate=gate, fusion=fusion)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn=use_bn, gate=gate, fusion=fusion)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn=use_bn, gate=gate, fusion=fusion)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn=use_bn, gate=gate, fusion=fusion)
        self.scratch.refinenet5 = _make_fusion_block(features, use_bn=use_bn, gate=gate, fusion=fusion)
        
        head_features_1 = features
        head_features_2 = coarse_chl[0]
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True))
        
        self.scratch.output_conv2_fusion = _make_fusion_block(head_features_2, use_bn=use_bn, gate=gate, fusion=fusion)
        
        self.scratch.output_conv3 = nn.Sequential(
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0))
        
        nn.init.normal_(self.scratch.output_conv3[0].weight, mean=1.0)
        nn.init.constant_(self.scratch.output_conv3[0].bias, 0)
    
    def forward(self, fine_features, coarse_features):
        
        # up to bottom
        layer_1, layer_2, layer_3, layer_4, layer_5 = fine_features
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        layer_5_rn = self.scratch.layer5_rn(layer_5)
        
        path_5 = self.scratch.refinenet5(layer_5_rn, size=layer_4_rn.shape[2:], coarse_feat=coarse_features[5])
        path_4 = self.scratch.refinenet4(path_5, layer_4_rn, size=layer_3_rn.shape[2:], coarse_feat=coarse_features[4])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:], coarse_feat=coarse_features[3])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:], coarse_feat=coarse_features[2])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn, coarse_feat=coarse_features[1])
        
        out = self.scratch.output_conv1(path_1)
        last_feat = self.scratch.output_conv2(out)
        last_feat = self.scratch.output_conv2_fusion(last_feat, coarse_feat=coarse_features[0], upscale=False)
        out = self.scratch.output_conv3(last_feat)
        # print(torch.min(last_feat), torch.max(last_feat), torch.min(out), torch.max(out))
        
        feats = [layer_5_rn, path_5, path_4, path_3, path_2, last_feat]
        return feats, out


class C2FNOENCModule(nn.Module):
    def __init__(self, coarse_chl, fine_chl, features=256, use_bn=False, fusion=True, gate=True,):
        super(C2FNOENCModule, self).__init__()
        
        self.scratch = _make_scratch_simple(
            fine_chl,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None
        
        self.scratch.layer1_gate1 = GatedConvUnit(features, nn.ReLU(False), gate=gate, fusion=fusion)
        self.scratch.layer1_gate2 = GatedConvUnit(features, nn.ReLU(False), gate=gate, fusion=fusion)
        
        self.scratch.layer2_gate1 = GatedConvUnit(features, nn.ReLU(False), gate=gate, fusion=fusion)
        self.scratch.layer2_gate2 = GatedConvUnit(features, nn.ReLU(False), gate=gate, fusion=fusion)
        
        self.scratch.layer3_gate1 = GatedConvUnit(features, nn.ReLU(False), gate=gate, fusion=fusion)
        self.scratch.layer3_gate2 = GatedConvUnit(features, nn.ReLU(False), gate=gate, fusion=fusion)
        
        self.scratch.layer4_gate1 = GatedConvUnit(features, nn.ReLU(False), gate=gate, fusion=fusion)
        self.scratch.layer4_gate2 = GatedConvUnit(features, nn.ReLU(False), gate=gate, fusion=fusion)
        
        self.scratch.layer5_gate1 = GatedConvUnit(features, nn.ReLU(False), gate=gate, fusion=fusion)
        self.scratch.layer5_gate2 = GatedConvUnit(features, nn.ReLU(False), gate=gate, fusion=fusion)
        
        self.scratch.upsample_conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=fine_chl[0],
                out_channels=32,
                kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(
                32, 32, kernel_size=3, stride=1, padding=1, bias=False))
        
        self.scratch.layer6_gate1 = GatedConvUnit(32, nn.ReLU(False), gate=gate, fusion=fusion)
        self.scratch.layer6_gate2 = GatedConvUnit(32, nn.ReLU(False), gate=gate, fusion=fusion)
        
        self.scratch.output_conv = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
    
    def forward(self, fine_features, coarse_features):
        
        # up to bottom
        layer_1, layer_2, layer_3, layer_4, layer_5 = fine_features
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        layer_5_rn = self.scratch.layer5_rn(layer_5)
        layer_0_rn = self.scratch.upsample_conv(layer_1)
        
        path_5 = self.scratch.layer1_gate1(layer_5_rn, coarse_features[5])
        path_5 = self.scratch.layer1_gate2(path_5, coarse_features[5])
        
        path_4 = self.scratch.layer2_gate1(layer_4_rn, coarse_features[4])
        path_4 = self.scratch.layer2_gate2(path_4, coarse_features[4])
        
        path_3 = self.scratch.layer3_gate1(layer_3_rn, coarse_features[3])
        path_3 = self.scratch.layer3_gate2(path_3, coarse_features[3])
        
        path_2 = self.scratch.layer4_gate1(layer_2_rn, coarse_features[2])
        path_2 = self.scratch.layer4_gate2(path_2, coarse_features[2])
        
        path_1 = self.scratch.layer5_gate1(layer_1_rn, coarse_features[1])
        path_1 = self.scratch.layer5_gate2(path_1, coarse_features[1])
        
        path_0 = self.scratch.layer6_gate1(layer_0_rn, coarse_features[0])
        path_0 = self.scratch.layer6_gate2(path_0, coarse_features[0])
        
        feats = [path_5, path_4, path_3, path_2, path_1, path_0]
        
        out = self.scratch.output_conv(path_0)
        return feats, out


@MODELS.register_module()
class BiDirectionalFusion(nn.Module):
    def __init__(
        self,
        encoder_name='',
        coarse2fine=True, # mid module
        coarse2fine_type='self-agg',
        fine2coarse=True, # final module
        coarse_chl=[32, 256, 256, 256, 256, 256], 
        fine_chl=[32, 32, 64, 96, 960],
        fine_chl_after_coarse2fine=[32, 256, 256, 256, 256, 256],
        temp_chl=[32, 64, 64, 128, 256, 512],
        dec_chl=[512, 256, 128, 64, 32],
        glb_att=False,
        att_dim=256,
        select_feat_index=[-1],
        pe_type='none'):
        
        super(BiDirectionalFusion, self).__init__()
        
        self.encoder_name = encoder_name
        
        # fine2coarse part
        # self.encoder_layers = nn.ModuleList()
        self.fusion_layers_1 = nn.ModuleList()
        self.fusion_layers_2 = nn.ModuleList()
        
        for idx, (coarse_c, fine_c, tmp_c) in enumerate(zip(coarse_chl, fine_chl_after_coarse2fine, temp_chl)):
            layer = SingleConvCNNLN(coarse_c + fine_c, tmp_c)
            self.fusion_layers_1.append(layer)
            layer = SingleConvCNNLN(tmp_c + 2, tmp_c)
            self.fusion_layers_2.append(layer)

        self.f2r_agg = nn.ModuleList()
        temp_chl = temp_chl[::-1]
        _chl = temp_chl[0]
        temp_chl = temp_chl[1:]
        for tmp_c, dec_c in zip(temp_chl, dec_chl):
            layer = UpSample(tmp_c + _chl + 2, dec_c)
            _chl = dec_c
            self.f2r_agg.append(layer)

        # prediction head
        if len(dec_chl) != 0:
            self.final_conv = nn.Conv2d(dec_chl[-1], 1, 3, 1, 1, bias=False)
        else:
            self.final_conv = nn.Conv2d(_chl, 1, 3, 1, 1, bias=False)
        
        self.glb_att = glb_att
        if self.glb_att:
            self.pe_type = pe_type
            self.att_dim = att_dim
            # self.feat_proj = nn.Conv2d(256, encoder_channels[-1], kernel_size=1, stride=1, padding=0)
            self.select_feat_index = select_feat_index
            self.feat_proj_coarse = nn.ModuleList()
            self.feat_proj_fine = nn.ModuleList()
            
            self.att_block = nn.ModuleList()
            
            for i in self.select_feat_index:
                feat_proj = SingleConvCNNLN(coarse_chl[i], att_dim, kernel_size=1, padding=0)
                self.feat_proj_coarse.append(feat_proj)
                feat_proj = SingleConvCNNLN(fine_chl[i], att_dim, kernel_size=1, padding=0)
                self.feat_proj_fine.append(feat_proj)
                
                att_block = TwoWayTransformer(
                    depth=2,
                    embedding_dim=att_dim, 
                    num_heads=8, 
                    mlp_dim=1024, 
                    activation=nn.ReLU, 
                    attention_downsample_rate=2)
                self.att_block.append(att_block)

            fine_chl[i] = fine_chl[i] + att_dim
            
        # coarse2fine part
        self.coarse2fine = coarse2fine
        self.coarse2fine_type = coarse2fine_type
        if self.coarse2fine:
            if self.coarse2fine_type == 'self-agg':
                # self.c2f = SimpleDPTHead(in_channels=32, features=256, use_bn=False, out_channels=fine_chl)
                self.c2f = C2FModule(coarse_chl=coarse_chl, fine_chl=fine_chl, fusion=False, gate=False,)
            elif self.coarse2fine_type == 'coarse-gated':
                self.c2f = C2FModule(coarse_chl=coarse_chl, fine_chl=fine_chl, fusion=True, gate=True,)
            elif self.coarse2fine_type == 'coarse-fusion':
                self.c2f = C2FModule(coarse_chl=coarse_chl, fine_chl=fine_chl, fusion=True, gate=False,)
            elif self.coarse2fine_type == 'only-gate':
                self.c2f = C2FNOENCModule(coarse_chl=coarse_chl, fine_chl=fine_chl, fusion=True, gate=False,)
            
    def forward(
        self, 
        c_feat,
        f_feat,
        pred1, # coarse_predicton_patch
        pred2, # refiner_prediction
        update_base=None,
        pe_list=None,
        pe_patch_list=None,):
        
        _, _, h1, w1 = c_feat[-1].shape
        _, _, h2, w2 = f_feat[-1].shape
        if h1 != h2 or w1 != w2:
            for i in range(len(c_feat)):
                c_feat[i] = F.interpolate(c_feat[i], f_feat[i].shape[-2:], mode='bilinear', align_corners=True)
        
        if self.glb_att:
            for idx, i in enumerate(self.select_feat_index):
                select_coarse = c_feat[i]
                select_fine = f_feat[i]
                select_coarse_proj = self.feat_proj_coarse[idx](select_coarse)
                select_fine_proj = self.feat_proj_fine[idx](select_fine)
                pe, pe_patch = pe_list[::-1][i], pe_patch_list[::-1][i]
                # print(select_fine.shape, pe_patch.shape, select_coarse_proj.shape, pe.shape)
                select_fine_update, coarse_as_key = self.att_block[idx](select_fine_proj, pe_patch, select_coarse_proj, pe)
                f_feat[i] = torch.cat([f_feat[i], select_fine_update], dim=1)
                
        # coarse2fine part
        if self.coarse2fine:
            f_feat = f_feat[1:] # decoder features: high-resolution to low-resolution
            if self.coarse2fine_type == 'self-agg':
                f_feat, out_depth = self.c2f(f_feat, c_feat)
                f_feat, pred2 = f_feat[::-1], out_depth
            elif self.coarse2fine_type == 'coarse-gated' or self.coarse2fine_type == 'coarse-fusion' or self.coarse2fine_type == 'only-gate':
                f_feat, out_depth = self.c2f(f_feat, c_feat)
                f_feat, pred2 = f_feat[::-1], out_depth
        
        # fine2coarse part
        temp_feat_list = []
        for idx, (c, f) in enumerate(zip(c_feat, f_feat)):
            feat = torch.cat([c, f], dim=1)
            f = self.fusion_layers_1[idx](feat)
            pred1_lvl = F.interpolate(pred1, f.shape[-2:], mode='bilinear', align_corners=True)
            pred2_lvl = F.interpolate(pred2, f.shape[-2:], mode='bilinear', align_corners=True)
            f = torch.cat([f, pred1_lvl, pred2_lvl], dim=1)
            f = self.fusion_layers_2[idx](f)
            temp_feat_list.append(f)
            
        
        dec_feat = temp_feat_list[0]
        temp_feat_list = temp_feat_list[::-1]
        _feat = temp_feat_list[0]
        temp_feat_list = temp_feat_list[1:]
        
        for idx, (feat, dec_layer) in enumerate(zip(temp_feat_list, self.f2r_agg)):
            dec_feat = dec_layer.forward_hardcode(_feat, feat, pred1, pred2)
            _feat = dec_feat
        
        final_feat = dec_feat
        final_offset = self.final_conv(final_feat)

        if update_base is not None:
            depth_prediction = update_base + final_offset
            depth_prediction = torch.clamp(depth_prediction, min=0)
        else:
            depth_prediction = final_offset
        
        return depth_prediction
    

class SingleConvCNNLNHeavy(nn.Module):
    """(convolution => [LayerNorm] => GELU) * 1 # used in encoder (for input features)"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            LayerNorm([out_channels]),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            LayerNorm([out_channels]),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GELU())

    def forward(self, x):
        return self.single_conv(x)

class DoubleConvHeavy(nn.Module):
    """(convolution => [GELU]) * 2 # used in decoder"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GELU())

    def forward(self, x):
        return self.double_conv(x)


class UpSampleHeavy(nn.Module):
    """Upscaling then cat and DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConvHeavy(in_channels, out_channels, in_channels)

    def forward_hardcode(self, x1, x2, pred1, pred2, update_depth=None):
        x1 = F.interpolate(x1, x2.shape[-2:], mode='bilinear', align_corners=True)
        pred1 = F.interpolate(pred1, x2.shape[-2:], mode='bilinear', align_corners=True)
        pred2 = F.interpolate(pred2, x2.shape[-2:], mode='bilinear', align_corners=True)
        if update_depth is not None:
            update_depth = F.interpolate(update_depth, x2.shape[-2:], mode='bilinear', align_corners=True)
            x = torch.cat([x1, x2, pred1, pred2, update_depth], dim=1)
        else:
            x = torch.cat([x1, x2, pred1, pred2], dim=1)
        return self.conv(x)
    
    def forward(self, x1, feat_list):
        ''' Args:
            x1: the feature map from the skip connection
            feat_list: the feature map list from the encoder and everything you want to concate with current feat maps
        '''
        upscale_feat_list = [x1]
        for feat in feat_list:
            upscale_feat_list.append(F.interpolate(feat, x1.shape[-2:], mode='bilinear', align_corners=True))
            
        x = torch.cat(upscale_feat_list, dim=1)
        return self.conv(x)
    
@MODELS.register_module()
class BiDirectionalFusionHeavy(nn.Module):
    def __init__(
        self,
        encoder_name='',
        coarse2fine=True, # mid module
        coarse2fine_type='self-agg',
        fine2coarse=True, # final module
        coarse_chl=[32, 256, 256, 256, 256, 256], 
        fine_chl=[32, 32, 64, 96, 960],
        fine_chl_after_coarse2fine=[32, 256, 256, 256, 256, 256],
        temp_chl=[32, 64, 64, 128, 256, 512],
        dec_chl=[512, 256, 128, 64, 32],
        glb_att=False,
        att_dim=256,
        select_feat_index=[-1],
        pe_type='none'):
        
        super(BiDirectionalFusionHeavy, self).__init__()
        
        self.encoder_name = encoder_name
        
        # fine2coarse part
        # self.encoder_layers = nn.ModuleList()
        self.fusion_layers_1 = nn.ModuleList()
        self.fusion_layers_2 = nn.ModuleList()
        
        for idx, (coarse_c, fine_c, tmp_c) in enumerate(zip(coarse_chl, fine_chl_after_coarse2fine, temp_chl)):
            layer = SingleConvCNNLNHeavy(coarse_c + fine_c, tmp_c)
            self.fusion_layers_1.append(layer)
            layer = SingleConvCNNLNHeavy(tmp_c + 2, tmp_c)
            self.fusion_layers_2.append(layer)

        self.f2r_agg = nn.ModuleList()
        temp_chl = temp_chl[::-1]
        _chl = temp_chl[0]
        temp_chl = temp_chl[1:]
        for tmp_c, dec_c in zip(temp_chl, dec_chl):
            layer = UpSampleHeavy(tmp_c + _chl + 2, dec_c)
            _chl = dec_c
            self.f2r_agg.append(layer)

        # prediction head
        if len(dec_chl) != 0:
            self.final_conv = nn.Conv2d(dec_chl[-1], 1, 3, 1, 1, bias=False)
        else:
            self.final_conv = nn.Conv2d(_chl, 1, 3, 1, 1, bias=False)
        
        self.glb_att = glb_att
        if self.glb_att:
            self.pe_type = pe_type
            self.att_dim = att_dim
            # self.feat_proj = nn.Conv2d(256, encoder_channels[-1], kernel_size=1, stride=1, padding=0)
            self.select_feat_index = select_feat_index
            self.feat_proj_coarse = nn.ModuleList()
            self.feat_proj_fine = nn.ModuleList()
            
            self.att_block = nn.ModuleList()
            
            for i in self.select_feat_index:
                feat_proj = SingleConvCNNLN(coarse_chl[i], att_dim, kernel_size=1, padding=0)
                self.feat_proj_coarse.append(feat_proj)
                feat_proj = SingleConvCNNLN(fine_chl[i], att_dim, kernel_size=1, padding=0)
                self.feat_proj_fine.append(feat_proj)
                
                att_block = TwoWayTransformer(
                    depth=2,
                    embedding_dim=att_dim, 
                    num_heads=8, 
                    mlp_dim=1024, 
                    activation=nn.ReLU, 
                    attention_downsample_rate=2)
                self.att_block.append(att_block)

            fine_chl[i] = fine_chl[i] + att_dim
            
        # coarse2fine part
        self.coarse2fine = coarse2fine
        self.coarse2fine_type = coarse2fine_type
        if self.coarse2fine:
            if self.coarse2fine_type == 'self-agg':
                # self.c2f = SimpleDPTHead(in_channels=32, features=256, use_bn=False, out_channels=fine_chl)
                self.c2f = C2FModule(coarse_chl=coarse_chl, fine_chl=fine_chl, fusion=False, gate=False,)
            elif self.coarse2fine_type == 'coarse-gated':
                self.c2f = C2FModule(coarse_chl=coarse_chl, fine_chl=fine_chl, fusion=True, gate=True,)
            elif self.coarse2fine_type == 'coarse-fusion':
                self.c2f = C2FModule(coarse_chl=coarse_chl, fine_chl=fine_chl, fusion=True, gate=False,)
            elif self.coarse2fine_type == 'only-gate':
                self.c2f = C2FNOENCModule(coarse_chl=coarse_chl, fine_chl=fine_chl, fusion=True, gate=False,)
            
    def forward(
        self, 
        c_feat,
        f_feat,
        pred1, # coarse_predicton_patch
        pred2, # refiner_prediction
        update_base=None,
        pe_list=None,
        pe_patch_list=None,):
        
        _, _, h1, w1 = c_feat[-1].shape
        _, _, h2, w2 = f_feat[-1].shape
        if h1 != h2 or w1 != w2:
            for i in range(len(c_feat)):
                c_feat[i] = F.interpolate(c_feat[i], f_feat[i].shape[-2:], mode='bilinear', align_corners=True)
        
        if self.glb_att:
            for idx, i in enumerate(self.select_feat_index):
                select_coarse = c_feat[i]
                select_fine = f_feat[i]
                select_coarse_proj = self.feat_proj_coarse[idx](select_coarse)
                select_fine_proj = self.feat_proj_fine[idx](select_fine)
                pe, pe_patch = pe_list[::-1][i], pe_patch_list[::-1][i]
                # print(select_fine.shape, pe_patch.shape, select_coarse_proj.shape, pe.shape)
                select_fine_update, coarse_as_key = self.att_block[idx](select_fine_proj, pe_patch, select_coarse_proj, pe)
                f_feat[i] = torch.cat([f_feat[i], select_fine_update], dim=1)
                
        # coarse2fine part
        if self.coarse2fine:
            f_feat = f_feat[1:] # decoder features: high-resolution to low-resolution
            if self.coarse2fine_type == 'self-agg':
                f_feat, out_depth = self.c2f(f_feat, c_feat)
                f_feat, pred2 = f_feat[::-1], out_depth
            elif self.coarse2fine_type == 'coarse-gated' or self.coarse2fine_type == 'coarse-fusion' or self.coarse2fine_type == 'only-gate':
                f_feat, out_depth = self.c2f(f_feat, c_feat)
                f_feat, pred2 = f_feat[::-1], out_depth
                
        # fine2coarse part
        temp_feat_list = []
        for idx, (c, f) in enumerate(zip(c_feat, f_feat)):
            feat = torch.cat([c, f], dim=1)
            f = self.fusion_layers_1[idx](feat)
            pred1_lvl = F.interpolate(pred1, f.shape[-2:], mode='bilinear', align_corners=True)
            pred2_lvl = F.interpolate(pred2, f.shape[-2:], mode='bilinear', align_corners=True)
            f = torch.cat([f, pred1_lvl, pred2_lvl], dim=1)
            f = self.fusion_layers_2[idx](f)
            temp_feat_list.append(f)
            
        
        dec_feat = temp_feat_list[0]
        temp_feat_list = temp_feat_list[::-1]
        _feat = temp_feat_list[0]
        temp_feat_list = temp_feat_list[1:]
        
        for idx, (feat, dec_layer) in enumerate(zip(temp_feat_list, self.f2r_agg)):
            dec_feat = dec_layer.forward_hardcode(_feat, feat, pred1, pred2)
            _feat = dec_feat
        
        final_feat = dec_feat
        final_offset = self.final_conv(final_feat)

        if update_base is not None:
            depth_prediction = update_base + final_offset
            depth_prediction = torch.clamp(depth_prediction, min=0)
        else:
            depth_prediction = final_offset
        
        return depth_prediction
    

