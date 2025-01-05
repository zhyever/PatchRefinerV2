
# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Zhenyu Li

import itertools

import timm
import math
import copy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mmengine import print_log
from torchvision.ops import roi_align as torch_roi_align
from huggingface_hub import PyTorchModelHubMixin
from mmengine.config import ConfigDict
from transformers import PretrainedConfig
from torch.nn.parameter import Parameter
from timm.layers import Conv2dSame

from estimator.registry import MODELS
from estimator.models import build_model
from estimator.models.utils import get_activation, generatemask, RunningAverageMap
from estimator.models.baseline_pretrain import BaselinePretrain
from estimator.models.utils import HookTool
from estimator.models.blocks import PositionEmbeddingRandom

from external.zoedepth.models.zoedepth import ZoeDepth
from external.zoedepth.models.base_models.midas import Resize as ResizeZoe
from external.depth_anything.transform import Resize as ResizeDA
from external.depth_anything_v2.dpt import DepthAnythingV2

from estimator.models.utils import HookTool

@MODELS.register_module()
class PatchRefinerPlus(BaselinePretrain, PyTorchModelHubMixin):
    def __init__(
        self,
        config):
        """ZoeDepth model
        """
        nn.Module.__init__(self)
        
        if isinstance(config, ConfigDict):
            # convert a ConfigDict to a PretrainedConfig for hf saving
            config = PretrainedConfig.from_dict(config.to_dict())
        else:
            # used when loading patchfusion from hf model space
            config = PretrainedConfig.from_dict(ConfigDict(**config).to_dict())
            config.base_depth_pretrain_model = None
        
        self.config = config
        self.min_depth = config.min_depth
        self.max_depth = config.max_depth

        self.patch_process_shape = config.patch_process_shape
        self.tile_cfg = self.prepare_tile_cfg(config.image_raw_shape, config.patch_split_num)
        
        self.sigloss = build_model(config.sigloss) # here
        self.gmloss = build_model(config.gmloss)
        self.sigweight = config.sigweight
        
        # pre-norm bbox: if setting it as True, no need to norm the bbox in roi_align during the forward pass in training stage
        # Reason: we may merge datasets to train the model, and we norm the bbox in the dataloader
        self.pre_norm_bbox = config.pre_norm_bbox # here
        self.pretrain_stage = config.pretrain_stage
        
        if config.pretrain_stage is True:
            self.refiner_fine_branch = build_model(config.refiner.fine_branch)
            self.fusion_feat_level = config.fusion_feat_level
            self.refiner_fusion_model = build_model(config.refiner.fusion_model) # FusionUnet(input_chl=self.update_feat_chl, temp_chl=[32, 256, 256], dec_chl=[256, 32])
            self.strategy_refiner_target = config.strategy_refiner_target
            self.hack_strategy = config.hack_strategy
            nn.init.normal_(self.refiner_fusion_model.final_conv.weight, mean=1.0)
            
        else:
            # process coarse model
            if config.coarse_branch.type == 'ZoeDepth':
                self.coarse_branch = ZoeDepth.build(**config.coarse_branch)
                print_log("Current zoedepth.core.prep.resizer is {}".format(type(self.coarse_branch.core.prep.resizer)), logger='current')
                if config.pretrain_coarse_model is not None:
                    print_log("Loading coarse_branch from {}".format(config.pretrain_coarse_model), logger='current')
                    print_log(self.coarse_branch.load_state_dict(torch.load(config.pretrain_coarse_model, map_location='cpu')['model_state_dict'], strict=False), logger='current') # coarse ckp strict=False for timm
                self.resizer = ResizeZoe(self.patch_process_shape[1], self.patch_process_shape[0], keep_aspect_ratio=False, ensure_multiple_of=32, resize_method="minimal")
                
            elif config.coarse_branch.type == 'DA-ZoeDepth':
                self.coarse_branch = ZoeDepth.build(**config.coarse_branch)
                print_log("Current zoedepth.core.prep.resizer is {}".format(type(self.coarse_branch.core.prep.resizer)), logger='current')
                if config.pretrain_coarse_model is not None:
                    print_log("Loading coarse_branch from {}".format(config.pretrain_coarse_model), logger='current')
                    print_log(self.coarse_branch.load_state_dict(torch.load(config.pretrain_coarse_model, map_location='cpu')['model_state_dict'], strict=True), logger='current') # coarse ckp
                self.resizer = ResizeDA(self.patch_process_shape[1], self.patch_process_shape[0], keep_aspect_ratio=False, ensure_multiple_of=14, resize_method="minimal")
            
            elif config.coarse_branch.type == 'DA2':
                self.coarse_branch = DepthAnythingV2(**{**config.coarse_branch['model_cfg'], 'max_depth': config.max_depth})
                self.coarse_branch.load_state_dict(torch.load(config.coarse_branch['pretrained'], map_location='cpu'))
                self.resizer = ResizeDA(self.patch_process_shape[1], self.patch_process_shape[0], keep_aspect_ratio=False, ensure_multiple_of=14, resize_method="minimal")
                if config.pretrain_coarse_model is not None:
                    print_log("Loading coarse_branch from {}".format(config.pretrain_coarse_model), logger='current')
                    print_log(self.coarse_branch.load_state_dict(torch.load(config.pretrain_coarse_model, map_location='cpu')['model_state_dict'], strict=True), logger='current') # coarse ckp
                    
            self.e2e_training = config.e2e_training
            if self.e2e_training is False:
                for param in self.coarse_branch.parameters():
                    param.requires_grad = False
            
            # process fine model
            self.refiner_fine_branch = build_model(config.refiner.fine_branch)
            
            self.fusion_feat_level = config.fusion_feat_level
            self.refiner_fusion_model = build_model(config.refiner.fusion_model) # FusionUnet(input_chl=self.update_feat_chl, temp_chl=[32, 256, 256], dec_chl=[256, 32])
            self.strategy_refiner_target = config.strategy_refiner_target

            if config.pretrained is not None:
                pretrained_dict = torch.load(config.pretrained, map_location='cpu')['model_state_dict']
                print_log("Loading the refiner part in patchrefiner from {}".format(config.pretrained), logger='current')
                print_log(self.load_state_dict(pretrained_dict, strict=False), logger='current') # coarse ckp
        
            # changing the input channel of the refiner encoder
            if config.refiner.fine_branch.coarse_condition:
                if config.refiner.fine_branch.encoder_name == 'mobilenetv3_large_100':
                    _weight = self.refiner_fine_branch.refiner_encoder.conv_stem.weight.clone() 
                    __weight = torch.zeros((16, 4, 3, 3))
                    __weight[:, :3, :, :] = _weight 
                    _new_conv_in = nn.Conv2d(4, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                    _new_conv_in.weight = Parameter(__weight)
                    self.refiner_fine_branch.refiner_encoder.conv_stem = _new_conv_in
                elif config.refiner.fine_branch.encoder_name == 'tf_efficientnet_b5_ap':
                    _weight = self.refiner_fine_branch.refiner_encoder.conv_stem.weight.clone()  
                    __weight = torch.zeros((48, 4, 3, 3))
                    __weight[:, :3, :, :] = _weight
                    _new_conv_in = Conv2dSame(4, 48, kernel_size=(3, 3), stride=(2, 2), bias=False)
                    _new_conv_in.weight = Parameter(__weight)
                    self.refiner_fine_branch.refiner_encoder.conv_stem = _new_conv_in
                elif config.refiner.fine_branch.encoder_name == 'mobilenetv4_conv_small.e2400_r224_in1k':
                    _weight = self.refiner_fine_branch.refiner_encoder.conv_stem.weight.clone()  
                    __weight = torch.zeros((32, 4, 3, 3))
                    __weight[:, :3, :, :] = _weight
                    _new_conv_in = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                    _new_conv_in.weight = Parameter(__weight)
                    self.refiner_fine_branch.refiner_encoder.conv_stem = _new_conv_in
                elif config.refiner.fine_branch.encoder_name == 'mobilenetv4_conv_medium.e500_r256_in1k':
                    _weight = self.refiner_fine_branch.refiner_encoder.conv_stem.weight.clone()  
                    __weight = torch.zeros((32, 4, 3, 3)) 
                    __weight[:, :3, :, :] = _weight
                    _new_conv_in = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                    _new_conv_in.weight = Parameter(__weight)
                    self.refiner_fine_branch.refiner_encoder.conv_stem = _new_conv_in
                elif config.refiner.fine_branch.encoder_name == 'mobilenetv4_conv_large.e600_r384_in1k':
                    _weight = self.refiner_fine_branch.refiner_encoder.conv_stem.weight.clone()  
                    __weight = torch.zeros((24, 4, 3, 3)) 
                    __weight[:, :3, :, :] = _weight
                    _new_conv_in = nn.Conv2d(3, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                    _new_conv_in.weight = Parameter(__weight)
                    self.refiner_fine_branch.refiner_encoder.conv_stem = _new_conv_in
                elif config.refiner.fine_branch.encoder_name == 'convnextv2_large.fcmae_ft_in22k_in1k_384':
                    _weight = self.refiner_fine_branch.refiner_encoder.stem_0.weight.clone()  
                    __weight = torch.zeros((192, 4, 4, 4))
                    __weight[:, :3, :, :] = _weight
                    _new_conv_in = nn.Conv2d(4, 192, kernel_size=(4, 4), stride=(4, 4))
                    _new_conv_in.weight = Parameter(__weight)
                    self.refiner_fine_branch.refiner_encoder.stem_0 = _new_conv_in   
                elif config.refiner.fine_branch.encoder_name == 'convnextv2_large':
                    _weight = self.refiner_fine_branch.refiner_encoder.stem_0.weight.clone()  
                    __weight = torch.zeros((192, 4, 4, 4))
                    __weight[:, :3, :, :] = _weight
                    _new_conv_in = nn.Conv2d(4, 192, kernel_size=(4, 4), stride=(4, 4))
                    _new_conv_in.weight = Parameter(__weight)
                    self.refiner_fine_branch.refiner_encoder.stem_0 = _new_conv_in   
                elif config.refiner.fine_branch.encoder_name == 'convnext_large':
                    _weight = self.refiner_fine_branch.refiner_encoder.stem_0.weight.clone()  
                    __weight = torch.zeros((192, 4, 4, 4))
                    __weight[:, :3, :, :] = _weight
                    _new_conv_in = nn.Conv2d(4, 192, kernel_size=(4, 4), stride=(4, 4))
                    _new_conv_in.weight = Parameter(__weight)
                    self.refiner_fine_branch.refiner_encoder.stem_0 = _new_conv_in   
            
            if config.whole_pretrained is not None:
                pretrained_dict = torch.load(config.whole_pretrained, map_location='cpu')['model_state_dict']
                print_log("Loading everything from {}".format(config.whole_pretrained), logger='current')
                print_log(self.load_state_dict(pretrained_dict, strict=False), logger='current') # coarse ckp
            
        if self.refiner_fusion_model.glb_att is True:
            self.pe = nn.ModuleList()
            for i in range(5):
                self.pe.append(PositionEmbeddingRandom(num_pos_feats=self.refiner_fusion_model.att_dim//2, pe_type=self.refiner_fusion_model.pe_type))
            
    def load_dict(self, dict):
        return self.load_state_dict(dict, strict=False) # strict=False for timm
        
    def get_save_dict(self):
        return self.state_dict()
    
    def coarse_forward(self, image_lr):
        
        if self.e2e_training is False:
            with torch.no_grad():
                if self.coarse_branch.training:
                    self.coarse_branch.eval()
    
        deep_model_output_dict = self.coarse_branch(image_lr, return_final_centers=True)
        deep_features = deep_model_output_dict['temp_features'] # x_d0 1/128, x_blocks_feat_0 1/64, x_blocks_feat_1 1/32, x_blocks_feat_2 1/16, x_blocks_feat_3 1/8, midas_final_feat 1/4 [based on 384x4, 512x4]
        coarse_prediction = deep_model_output_dict['metric_depth']
        
        coarse_features = [
            deep_features['x_d0'],
            deep_features['x_blocks_feat_0'],
            deep_features['x_blocks_feat_1'],
            deep_features['x_blocks_feat_2'],
            deep_features['x_blocks_feat_3'],
            deep_features['midas_final_feat']] # bs, c, h, w

        return coarse_features, coarse_prediction
        
    def coarse_postprocess_train(self, coarse_prediction, coarse_features, bboxs, bboxs_feat):
        coarse_features_patch_area = []
        for idx, feat in enumerate(coarse_features):
            bs, _, h, w = feat.shape
            cur_lvl_feat = torch_roi_align(feat, bboxs_feat, (h, w), h/self.patch_process_shape[0], aligned=True)
            coarse_features_patch_area.append(cur_lvl_feat)
        
        coarse_prediction_roi = torch_roi_align(coarse_prediction, bboxs_feat, coarse_prediction.shape[-2:], coarse_prediction.shape[-2]/self.patch_process_shape[0], aligned=True)
        
        if self.refiner_fusion_model.glb_att is True:
            pe_list = []
            pe_patch_list = []
            for idx, feat in enumerate(coarse_features[:-1]):
                bs, _, h, w = feat.shape
                pe = self.pe[idx]((h, w)).unsqueeze(dim=0).repeat(bs, 1, 1, 1)
                pe_patch = torch_roi_align(pe, bboxs_feat, (h, w), h/self.patch_process_shape[0], aligned=True)
                pe_list.append(pe)
                pe_patch_list.append(pe_patch)
        else:
            pe_list, pe_patch_list = None, None
        
        # lvl, -> bs, c, h, w; bs, 1, h, w
        return coarse_features_patch_area, coarse_prediction_roi, pe_list, pe_patch_list
    
    def coarse_postprocess_test(self, coarse_prediction, coarse_features, bboxs, bboxs_feat):
        patch_num = bboxs_feat.shape[0]

        coarse_features_patch_area = []
        coarse_features_tempsave = []
        for idx, feat in enumerate(coarse_features):
            bs, _, h, w = feat.shape
            feat_extend = feat.repeat(patch_num, 1, 1, 1)
            cur_lvl_feat = torch_roi_align(feat_extend, bboxs_feat, (h, w), h/self.patch_process_shape[0], aligned=True)
            coarse_features_patch_area.append(cur_lvl_feat)
            coarse_features_tempsave.append(feat_extend)
        
        coarse_prediction = coarse_prediction.repeat(patch_num, 1, 1, 1)
        coarse_prediction_roi = torch_roi_align(coarse_prediction, bboxs_feat, coarse_prediction.shape[-2:], coarse_prediction.shape[-2]/self.patch_process_shape[0], aligned=True)
        # coarse_prediction_roi = None
        
        return_dict = {
            'coarse_depth_roi': coarse_prediction_roi,
            'coarse_feats_roi': coarse_features_patch_area,
            'coarse_feats': coarse_features_tempsave,}
        
        if self.refiner_fusion_model.glb_att is True:
            pe_list = []
            pe_patch_list = []
            for idx, feat in enumerate(coarse_features[:-1]):
                bs, _, h, w = feat.shape
                pe = self.pe[idx]((h, w)).unsqueeze(dim=0).repeat(patch_num, 1, 1, 1)
                pe_patch = torch_roi_align(pe, bboxs_feat, (h, w), h/self.patch_process_shape[0], aligned=True)
                pe_list.append(pe)
                pe_patch_list.append(pe_patch)
            return_dict['pe_list'] = pe_list
            return_dict['pe_patch_list'] = pe_patch_list
            
        return return_dict
        
    def refiner_fine_forward(self, image_hr, coarse_depth_roi, coarse_feats=None, pe_list=None, pe_patch_list=None):
        refiner_features, refiner_continous_depth = self.refiner_fine_branch(image_hr, coarse_depth_roi, coarse_feats, pe_list, pe_patch_list)
        return refiner_features, refiner_continous_depth
    
    def refiner_fusion_forward(
        self, 
        coarse_features_patch, 
        coarse_predicton_patch, 
        refiner_features, 
        refiner_prediction, 
        update_base=None,
        pe_list=None, 
        pe_patch_list=None):

        c_feat_list = []
        r_feat_list = []

        for idx, (c_feat, r_feat) in enumerate(zip(coarse_features_patch[-self.fusion_feat_level:], refiner_features[-self.fusion_feat_level:])):
            c_feat_list.append(c_feat)
            r_feat_list.append(r_feat)
        
        offset_pred = self.refiner_fusion_model(
            c_feat=c_feat_list[::-1], 
            f_feat=r_feat_list[::-1], 
            pred1=coarse_predicton_patch,
            pred2=refiner_prediction,
            update_base=update_base,
            pe_list=pe_list,
            pe_patch_list=pe_patch_list,)
        
        return offset_pred
    
    def infer_forward(self, imgs_crop, bbox_feat_forward, tile_temp, coarse_temp_dict):

        # if self.refiner_fine_branch.glb_att is True:
        #     refiner_features, refiner_continous_depth = \
        #         self.refiner_fine_forward(
        #             imgs_crop, 
        #             coarse_temp_dict['coarse_depth_roi'], 
        #             coarse_temp_dict['coarse_feats'], 
        #             coarse_temp_dict['pe_list'], 
        #             coarse_temp_dict['pe_patch_list'])
        # else:
        #     refiner_features, refiner_continous_depth = \
        #         self.refiner_fine_forward(
        #             imgs_crop, 
        #             coarse_temp_dict['coarse_depth_roi'])
        
        refiner_features, refiner_continous_depth = \
                self.refiner_fine_forward(
                    imgs_crop, 
                    coarse_temp_dict['coarse_depth_roi'])

        # update
        if self.strategy_refiner_target == 'offset_fine':
            update_base = refiner_continous_depth
        elif self.strategy_refiner_target == 'offset_coarse':
            update_base = coarse_temp_dict['coarse_depth_roi']
        else:
            update_base = None
            
        depth_prediction = self.refiner_fusion_forward(coarse_temp_dict['coarse_feats_roi'], coarse_temp_dict['coarse_depth_roi'], refiner_features, refiner_continous_depth, update_base=update_base,
                                                       pe_list=coarse_temp_dict.get('pe_list', None), pe_patch_list=coarse_temp_dict.get('pe_patch_list', None))
        
        if self.strategy_refiner_target == 'direct':
            depth_prediction = F.sigmoid(depth_prediction) * self.max_depth
  
        return depth_prediction
        
    def forward(
        self,
        mode=None,
        image_lr=None,
        image_hr=None,
        crops_image_hr=None,
        depth_gt=None,
        crop_depths=None,
        bboxs=None,
        tile_cfg=None,
        cai_mode='m1',
        process_num=4,
        select_patch=-1,
        **kwargs):
        
        if self.pretrain_stage is True:
            feats, depth_prediction = self.refiner_fine_branch(image_lr)
            
            # c_feat_list # 384512, 192256, 96128, 4864, 2432, 1216
            c_feat_list = []
            for idx, f in enumerate(feats):
                bs, c, h, w = f.shape
                if idx == 5:
                    if self.config.coarse_branch.type == 'DA2':
                        c = 128
                    else:
                        c = 32
                else:
                    c = 256
                if self.hack_strategy == 'mean_0_std_1':
                    c_feat_list.append(torch.normal(0, 1, size=(bs, c, h, w)).to(f.device))
                elif self.hack_strategy == 'constant':
                    c_feat_list.append(torch.ones((bs, c, h, w)).to(f.device))
                else:
                    raise NotImplementedError("hack_strategy not implemented")

            coarse_predicton_patch = torch.zeros_like(depth_prediction)
            
            depth_prediction = self.refiner_fusion_model(
                c_feat=c_feat_list[::-1], 
                f_feat=feats[::-1], 
                pred1=coarse_predicton_patch,
                pred2=depth_prediction,
                update_base=None,
                pe_list=None,
                pe_patch_list=None,)
            
            depth_prediction = F.relu(depth_prediction)
            
            if mode == 'train':
                loss_dict = {}
                sig_loss_fine = self.sigloss(depth_prediction, depth_gt, self.min_depth, self.max_depth)
                loss_dict['sig_fine_loss'] = sig_loss_fine
                loss_dict['total_loss'] = loss_dict['sig_fine_loss']
                
                return loss_dict, {'rgb': image_lr, 'depth_pred': depth_prediction, 'depth_gt': crop_depths}
            else:
                return depth_prediction, {'rgb': image_lr, 'depth_pred': depth_prediction, 'depth_gt': depth_gt}
            
        else:
            if mode == 'train':
                if self.pre_norm_bbox:
                    bboxs_feat = bboxs
                else:
                    bboxs_feat_factor = torch.tensor([
                        1 / self.tile_cfg['image_raw_shape'][1] * self.patch_process_shape[1], 
                        1 / self.tile_cfg['image_raw_shape'][0] * self.patch_process_shape[0], 
                        1 / self.tile_cfg['image_raw_shape'][1] * self.patch_process_shape[1], 
                        1 / self.tile_cfg['image_raw_shape'][0] * self.patch_process_shape[0]], device=bboxs.device).unsqueeze(dim=0)
                    bboxs_feat = bboxs * bboxs_feat_factor
                inds = torch.arange(bboxs.shape[0]).to(bboxs.device).unsqueeze(dim=-1)
                bboxs_feat = torch.cat((inds, bboxs_feat), dim=-1)
                
                # all of them are at whole-image level
                coarse_features, coarse_prediction = self.coarse_forward(image_lr) 
                coarse_features_patch, coarse_prediction_roi, pe_list, pe_patch_list = self.coarse_postprocess_train(coarse_prediction, coarse_features, bboxs, bboxs_feat)
                # patch refiner features 
                refiner_features, refiner_continous_depth = self.refiner_fine_forward(crops_image_hr, coarse_prediction_roi, coarse_features, pe_list, pe_patch_list)

                # update
                if self.strategy_refiner_target == 'offset_fine':
                    update_base = refiner_continous_depth
                elif self.strategy_refiner_target == 'offset_coarse':
                    update_base = coarse_prediction_roi
                else:
                    update_base = None
                    
                depth_prediction = self.refiner_fusion_forward(coarse_features_patch, coarse_prediction_roi, refiner_features, refiner_continous_depth, update_base=update_base, pe_list=pe_list, pe_patch_list=pe_patch_list)
                
                if self.strategy_refiner_target == 'direct':
                    depth_prediction = F.sigmoid(depth_prediction) * self.max_depth

                loss_dict = {}
                
                sig_loss_fine = self.sigloss(depth_prediction, crop_depths, self.min_depth, self.max_depth)
                loss_dict['sig_fine_loss'] = sig_loss_fine
                gm_loss = self.gmloss(depth_prediction, crop_depths, self.min_depth, self.max_depth)
                loss_dict['gm_loss'] = gm_loss
                loss_dict['total_loss'] = self.sigweight * loss_dict['sig_fine_loss'] + (1 - self.sigweight) * loss_dict['gm_loss']
                
                return loss_dict, {'rgb': crops_image_hr, 'depth_pred': depth_prediction, 'depth_gt': crop_depths}
                

            else:
                
                if tile_cfg is None:
                    tile_cfg = self.tile_cfg
                else:
                    tile_cfg = self.prepare_tile_cfg(tile_cfg['image_raw_shape'], tile_cfg['patch_split_num'])
                
                assert image_hr.shape[0] == 1
                
                coarse_features, coarse_prediction = self.coarse_forward(image_lr) 
                
                tile_temp = {
                    'coarse_prediction': coarse_prediction,
                    'coarse_features': coarse_features,}

                blur_mask = generatemask((self.patch_process_shape[0], self.patch_process_shape[1]), border=0.15)
                blur_mask = torch.tensor(blur_mask, device=image_hr.device)
                
                avg_depth_map = self.regular_tile(
                    offset=[0, 0], 
                    offset_process=[0, 0], 
                    image_hr=image_hr[0], 
                    init_flag=True, 
                    tile_temp=tile_temp, 
                    blur_mask=blur_mask,
                    tile_cfg=tile_cfg,
                    process_num=process_num,
                    select_patch=select_patch)
                
                if cai_mode == 'm2' or cai_mode[0] == 'r':
                    avg_depth_map = self.regular_tile(
                        offset=[0, tile_cfg['patch_raw_shape'][1]//2], 
                        offset_process=[0, self.patch_process_shape[1]//2], 
                        image_hr=image_hr[0], init_flag=False, tile_temp=tile_temp, blur_mask=blur_mask, avg_depth_map=avg_depth_map, tile_cfg=tile_cfg, process_num=process_num)
                    avg_depth_map = self.regular_tile(
                        offset=[tile_cfg['patch_raw_shape'][0]//2, 0],
                        offset_process=[self.patch_process_shape[0]//2, 0], 
                        image_hr=image_hr[0], init_flag=False, tile_temp=tile_temp, blur_mask=blur_mask, avg_depth_map=avg_depth_map, tile_cfg=tile_cfg, process_num=process_num)
                    avg_depth_map = self.regular_tile(
                        offset=[tile_cfg['patch_raw_shape'][0]//2, tile_cfg['patch_raw_shape'][1]//2],
                        offset_process=[self.patch_process_shape[0]//2, self.patch_process_shape[1]//2], 
                        init_flag=False, image_hr=image_hr[0], tile_temp=tile_temp, blur_mask=blur_mask, avg_depth_map=avg_depth_map, tile_cfg=tile_cfg, process_num=process_num)
                
                if cai_mode[0] == 'r':
                    blur_mask = generatemask((tile_cfg['patch_raw_shape'][0], tile_cfg['patch_raw_shape'][1]), border=0.15) + 1e-3
                    blur_mask = torch.tensor(blur_mask, device=image_hr.device)
                    avg_depth_map.resize(tile_cfg['image_raw_shape'])
                    patch_num = int(cai_mode[1:]) // process_num
                    for i in range(patch_num):
                        avg_depth_map = self.random_tile(
                            image_hr=image_hr[0], tile_temp=tile_temp, blur_mask=blur_mask, avg_depth_map=avg_depth_map, tile_cfg=tile_cfg, process_num=process_num)

                # depth = avg_depth_map.average_map
                depth = avg_depth_map.get_avg_map()
                depth = depth.unsqueeze(dim=0).unsqueeze(dim=0)

                return depth, \
                    {'rgb': image_lr, 
                     'depth_pred': depth, 
                     'depth_gt': depth_gt,
                     'coarse_prediction': coarse_prediction}
                    #  'uncertainty': avg_depth_map.uncertainty_map.unsqueeze(dim=0).unsqueeze(dim=0),
                    #  'count_map': avg_depth_map.count_map_raw.unsqueeze(dim=0).unsqueeze(dim=0)
                # return coarse_prediction, {'rgb': image_lr, 'depth_pred': coarse_prediction, 'depth_gt': depth_gt}
            