import copy
import kornia

import torch
import torch.nn as nn
from mmengine import print_log
import torch.nn.functional as F
import random
import math

from estimator.registry import MODELS
from kornia.losses import dice_loss, focal_loss

import numpy as np
from estimator.utils import RandomBBoxQueries
import kornia
from estimator.utils import get_boundaries, compute_metrics, compute_boundary_metrics, extract_edges
# from pytorch3d.loss import chamfer_distance
from estimator.utils import RandomBBoxQueries

@MODELS.register_module()
class SILogLoss(nn.Module):
    """SILog loss (pixel-wise)"""
    def __init__(self, beta=0.15, **kwargs):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'
        self.beta = beta

    def forward(self, input, target, min_depth, max_depth, additional_mask=None):
        _, _, h_i, w_i = input.shape
        _, _, h_t, w_t = target.shape
        
        if h_i != h_t or w_i != w_t:
            input = F.interpolate(input, (h_t, w_t), mode='bilinear', align_corners=True)
        
        mask = torch.logical_and(target>min_depth, target<max_depth)
        
        if additional_mask is not None:
            mask_merge = torch.logical_and(mask, additional_mask)
            if torch.sum(mask_merge) >= h_i * w_i * 0.001:
                mask = mask_merge
            else:
                print_log("torch.sum(mask_merge) < h_i * w_i * 0.001, reduce to previous mask for stable training", logger='current')
        
        if torch.sum(mask) <= 1:
            print_log("torch.sum(mask) <= 1, hack to skip avoiding nan", logger='current')
            return input * 0.0
        
        input = input[mask]
        target = target[mask]
        alpha = 1e-7
        g = torch.log(input + alpha) - torch.log(target + alpha)
        Dg = torch.var(g) + self.beta * torch.pow(torch.mean(g), 2)
        loss = 10 * torch.sqrt(Dg)

        if torch.isnan(loss):
            print_log("Nan SILog loss", logger='current')
            print_log("input: {}".format(input.shape), logger='current')
            print_log("target: {}".format(target.shape), logger='current')
            
            print_log("G: {}".format(torch.sum(torch.isnan(g))), logger='current')
            print_log("Input min: {} max: {}".format(torch.min(input), torch.max(input)), logger='current')
            print_log("Target min: {} max: {}".format(torch.min(target), torch.max(target)), logger='current')
            print_log("Dg: {}".format(torch.isnan(Dg)), logger='current')
            print_log("loss: {}".format(torch.isnan(loss)), logger='current')

        return loss


def get_grad_map(value):
    grad = kornia.filters.spatial_gradient(value)
    grad_xy = (grad[:,:,0,:,:] ** 2 + grad[:,:,1,:,:] ** 2) ** (1/2)
    return grad_xy

def get_grad_error_mask(gt, coarse_prediction, shape=(384, 512), min_depth=1e-3, max_depth=80):
    invalid_mask = torch.logical_or(gt<=min_depth, gt>=max_depth)
    gt_grad = get_grad_map(gt)
    coarse_prediction_grad = get_grad_map(coarse_prediction)
    grad_error = ((gt_grad - coarse_prediction_grad) / gt).abs()
    grad_error[grad_error>0.001] = 1.
    grad_error[invalid_mask] = 2. # filter invalid area
    grad_error[gt>10000] = 3.
    return grad_error.long().squeeze(dim=1)

def get_grad_value_error_mask(gt, coarse_prediction, shape=(384, 512), min_depth=1e-3, max_depth=80):
    invalid_mask = torch.logical_or(gt<=min_depth, gt>=max_depth)
    error = ((gt - coarse_prediction) / gt).abs() 
    error[error>0.1] = 1.
    gt_grad = get_grad_map(gt)
    coarse_prediction_grad = get_grad_map(coarse_prediction)
    grad_error = ((gt_grad - coarse_prediction_grad) / gt).abs()
    error[grad_error>0.001] = 1.
    error[invalid_mask] = 2. # filter invalid area
    error[gt>10000] = 3.
    return error.long().squeeze(dim=1)

def get_incoherent_mask(gt, shape=(384, 512), min_depth=1e-3, max_depth=80):
    # incoherent
    ori_shpae = gt.shape[-2:]
    gt_lr = F.interpolate(gt, shape, mode='bilinear', align_corners=True)
    invalid_mask = torch.logical_or(gt<=min_depth, gt>=max_depth)
    gt_recover = F.interpolate(gt_lr, ori_shpae, mode='bilinear', align_corners=True)
    residue = (gt - gt_recover).abs()
    
    gt_label = torch.zeros_like(gt)
    gt_label[residue >= 0.01] = 1. # set incoherent area as 1
    gt_label[invalid_mask] = 2. # filter invalid area
    gt_label[gt>10000] = 3.
    return gt_label.long().squeeze(dim=1)

def get_incoherent_grad_error_mask(gt, coarse_prediction, shape=(384, 512), min_depth=1e-3, max_depth=80):
    # incoherent
    ori_shpae = gt.shape[-2:]
    gt_lr = F.interpolate(gt, shape, mode='bilinear', align_corners=True)
    invalid_mask = torch.logical_or(gt<=min_depth, gt>=max_depth)
    gt_recover = F.interpolate(gt_lr, ori_shpae, mode='bilinear', align_corners=True)
    residue = (gt - gt_recover).abs()
    # coarse_prediction = F.interpolate(coarse_prediction, gt.shape[-2:], mode='bilinear', align_corners=True)
    # error = (gt - coarse_prediction).abs()
    
    # grad error
    gt_grad = get_grad_map(gt)
    coarse_prediction_grad = get_grad_map(coarse_prediction)
    grad_error = ((gt_grad - coarse_prediction_grad) / gt).abs()
    
    bad_area_mask = torch.logical_or(residue>0.01, grad_error>0.001)
    gt_label = torch.zeros_like(gt)
    gt_label[bad_area_mask] = 1.
    gt_label[invalid_mask] = 2. # filter invalid area
    gt_label[gt>10000] = 3.
    return gt_label.long().squeeze(dim=1)

def get_incoherent_grad_value_error_mask(gt, coarse_prediction, shape=(384, 512), min_depth=1e-3, max_depth=80):
    # incoherent
    ori_shpae = gt.shape[-2:]
    gt_lr = F.interpolate(gt, shape, mode='bilinear', align_corners=True)
    invalid_mask = torch.logical_or(gt<=min_depth, gt>=max_depth)
    gt_recover = F.interpolate(gt_lr, ori_shpae, mode='bilinear', align_corners=True)
    residue = (gt - gt_recover).abs()
    
    # value error
    coarse_prediction = F.interpolate(coarse_prediction, gt.shape[-2:], mode='bilinear', align_corners=True)
    error = (gt - coarse_prediction).abs()
    bad_area_mask = torch.logical_or(residue>0.01, error>0.5)
    
    # grad error
    gt_grad = get_grad_map(gt)
    coarse_prediction_grad = get_grad_map(coarse_prediction)
    grad_error = ((gt_grad - coarse_prediction_grad) / gt).abs()
    bad_area_mask = torch.logical_or(grad_error, grad_error>0.001)
    
    gt_label = torch.zeros_like(gt)
    gt_label[bad_area_mask] = 1.
    gt_label[invalid_mask] = 2. # filter invalid area
    gt_label[gt>10000] = 3.
    return gt_label.long().squeeze(dim=1)

class GeneralizedSoftDiceLoss(nn.Module):
    def __init__(self,
                 p=1,
                 smooth=1,
                 reduction='mean'):
        super(GeneralizedSoftDiceLoss, self).__init__()
        self.p = p
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, probs, label):
        '''
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        '''
        # compute loss
        numer = torch.sum((probs*label), dim=(2, 3))
        denom = torch.sum(probs.pow(self.p) + label.pow(self.p), dim=(2, 3))
        numer = torch.sum(numer, dim=1)
        denom = torch.sum(denom, dim=1)
        loss = 1 - (2*numer+self.smooth)/(denom+self.smooth)
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss

@MODELS.register_module()
class EdgeClsLoss(nn.Module):
    """Error loss (pixel-wise)"""
    def __init__(self, focal_weight=0.5):
        super(EdgeClsLoss, self).__init__()
        self.name = 'Error'
        self.criterion_dice = GeneralizedSoftDiceLoss()
        self.criterion_bce = nn.BCELoss()
        self.focal_weight = focal_weight

    def forward(self, input, target):
        _, _, h_i, w_i = input.shape
        _, h_t, w_t = target.shape
        
        if h_i != h_t or w_i != w_t:
            input = F.interpolate(input, (h_t, w_t), mode='bilinear', align_corners=True)

        target = target.long()
        dice = dice_loss(input, target)
        focal = focal_loss(input, target, alpha=self.focal_weight, reduction='mean')
        
        return dice, focal


@MODELS.register_module()
class ErrorLoss(nn.Module):
    """Error loss (pixel-wise)"""
    def __init__(self, loss_type, focal_weight):
        super(ErrorLoss, self).__init__()
        self.name = 'Error'
        self.criterion_dice = GeneralizedSoftDiceLoss()
        self.criterion_bce = nn.BCELoss()
        self.loss_type = loss_type
        self.focal_weight = focal_weight

    def forward(self, input, target, coarse_prediction, min_depth, max_depth):
        _, _, h_i, w_i = input.shape
        _, _, h_c, w_c = coarse_prediction.shape
        _, _, h_t, w_t = target.shape
        
        if h_i != h_t or w_i != w_t:
            input = F.interpolate(input, (h_t, w_t), mode='bilinear', align_corners=True)
        if h_c != h_t or w_c != w_t:
            coarse_prediction = F.interpolate(coarse_prediction, (h_t, w_t), mode='bilinear')
            
        
        if self.loss_type == 'incoh':
            gt_mask = get_incoherent_mask(target, shape=(h_i, w_i), min_depth=min_depth, max_depth=max_depth)
        elif self.loss_type == 'incoh+grad':
            gt_mask = get_incoherent_grad_error_mask(target, coarse_prediction, shape=(h_i, w_i), min_depth=min_depth, max_depth=max_depth)
        elif self.loss_type == 'incoh+grad+depth':
            gt_mask = get_incoherent_grad_value_error_mask(target, coarse_prediction, shape=(h_i, w_i), min_depth=min_depth, max_depth=max_depth)
        else:
            raise NotImplementedError
        

        dice = dice_loss(input, gt_mask)
        # focal = focal_loss(input, gt_mask, alpha=0.5, reduction='mean')
        focal = focal_loss(input, gt_mask, alpha=self.focal_weight, reduction='mean')
        
        return dice, focal, gt_mask


def ind2sub(idx, w):
    row = idx // w
    col = idx % w
    return row, col

def sub2ind(r, c, cols):
    idx = r * cols + c
    return idx
import matplotlib.pyplot as plt

@MODELS.register_module()
class EdgeguidedRankingLoss(nn.Module):
    def __init__(
        self, 
        point_pairs=10000, 
        sigma=0.03, 
        alpha=1.0, 
        reweight_target=False, 
        only_missing_area=False,
        min_depth=1e-3, 
        max_depth=80,
        missing_value=0,
        random_direct=True):
        super(EdgeguidedRankingLoss, self).__init__()
        self.point_pairs = point_pairs # number of point pairs
        self.sigma = sigma # used for determining the ordinal relationship between a selected pair
        self.alpha = alpha # used for balancing the effect of = and (<,>)
        #self.regularization_loss = GradientLoss(scales=4)
        self.reweight_target = reweight_target
        self.only_missing_area = only_missing_area
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.missing_value = missing_value
        self.random_direct = random_direct
        
        self.idx = 0
        self.idx_inner = 0
        
    def getEdge(self, images):
        n,c,h,w = images.size()
        a = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).cuda().view((1,1,3,3)).repeat(1, 1, 1, 1)
        b = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).cuda().view((1,1,3,3)).repeat(1, 1, 1, 1)
        if c == 3:
            gradient_x = F.conv2d(images[:,0,:,:].unsqueeze(1), a)
            gradient_y = F.conv2d(images[:,0,:,:].unsqueeze(1), b)
        else:
            gradient_x = F.conv2d(images, a)
            gradient_y = F.conv2d(images, b)
        edges = torch.sqrt(torch.pow(gradient_x,2)+ torch.pow(gradient_y,2))
        edges = F.pad(edges, (1,1,1,1), "constant", 0)
        thetas = torch.atan2(gradient_y, gradient_x)
        thetas = F.pad(thetas, (1,1,1,1), "constant", 0)

        return edges, thetas

    def edgeGuidedSampling(self, inputs, targets, edges_img, thetas_img, missing_mask, depth_gt, strict_mask):
        
        if self.only_missing_area:
            edges_mask = torch.logical_and(edges_img, missing_mask) # 1 edge, 2 missing mask (valid range)
            # edges_mask = missing_mask
        else:
            edges_mask = torch.logical_and(edges_img, strict_mask) # 1 edge, 2 strict mask (valid range)
        
        # plt.figure()
        # plt.imshow(edges_mask.squeeze().cpu().numpy())
        # plt.savefig('/ibex/ai/home/liz0l/codes/PatchRefiner/work_dir/debug/debug_{}_{}_mask.png'.format(self.idx, self.idx_inner))
        
        edges_loc = edges_mask.nonzero()
        minlen = edges_loc.shape[0]
        
        if minlen == 0:
            return torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), 0

        # find anchor points (i.e, edge points)
        sample_num = self.point_pairs
        sample_index = torch.randint(0, minlen, (sample_num,), dtype=torch.long).cuda()
        sample_h, sample_w = edges_loc[sample_index, 0], edges_loc[sample_index, 1]
        theta_anchors = thetas_img[sample_h, sample_w] 
        
        sidx = edges_loc.shape[0] // 2
        ## compute the coordinates of 4-points,  distances are from [-30, 30]
        distance_matrix = torch.randint(2, 31, (4,sample_num)).cuda()
        pos_or_neg = torch.ones(4, sample_num).cuda()
        pos_or_neg[:2,:] = -pos_or_neg[:2,:]
            
        distance_matrix = distance_matrix.float() * pos_or_neg
        p = random.random()

        if self.random_direct:
            if p < 0.5:
                col = sample_w.unsqueeze(0).expand(4, sample_num).long() + torch.round(distance_matrix.double() * torch.cos(theta_anchors).unsqueeze(0)).long()
                row = sample_h.unsqueeze(0).expand(4, sample_num).long() + torch.round(distance_matrix.double() * torch.sin(theta_anchors).unsqueeze(0)).long()
            else:
                theta_anchors = theta_anchors + math.pi / 2
                theta_anchors = (theta_anchors + math.pi) % (2 * math.pi) - math.pi
                col = sample_w.unsqueeze(0).expand(4, sample_num).long() + torch.round(distance_matrix.double() * torch.sin(theta_anchors).unsqueeze(0)).long()
                row = sample_h.unsqueeze(0).expand(4, sample_num).long() + torch.round(distance_matrix.double() * torch.cos(theta_anchors).unsqueeze(0)).long()
        else:
            col = sample_w.unsqueeze(0).expand(4, sample_num).long() + torch.round(distance_matrix.double() * torch.cos(theta_anchors).unsqueeze(0)).long()
            row = sample_h.unsqueeze(0).expand(4, sample_num).long() + torch.round(distance_matrix.double() * torch.sin(theta_anchors).unsqueeze(0)).long()

        # constrain 0=<c<=w, 0<=r<=h
        # Note: index should minus 1
        w, h = depth_gt.shape[-1], depth_gt.shape[-2]
        invalid_mask = (col<0) + (col>w-1) + (row<0) + (row>h-1)
        invalid_mask = torch.sum(invalid_mask, dim=0) > 0
        col = col[:, torch.logical_not(invalid_mask)]
        row = row[:, torch.logical_not(invalid_mask)]
    
        a = torch.stack([row[0, :], col[0, :]])
        b = torch.stack([row[1, :], col[1, :]])
        c = torch.stack([row[2, :], col[2, :]])
        d = torch.stack([row[3, :], col[3, :]])
        
        # if self.only_missing_area:
        #     valid_check_a_strict = strict_mask[a[0, :], a[1, :]] == True
        #     valid_check_b_strict = strict_mask[b[0, :], b[1, :]] == True
        #     valid_check_c_strict = strict_mask[c[0, :], c[1, :]] == True
        #     valid_check_d_strict = strict_mask[d[0, :], d[1, :]] == True
            
        #     valid_mask_ab = torch.logical_not(torch.logical_and(valid_check_a_strict, valid_check_b_strict))
        #     valid_mask_bc = torch.logical_not(torch.logical_and(valid_check_b_strict, valid_check_c_strict))
        #     valid_mask_cd = torch.logical_not(torch.logical_and(valid_check_c_strict, valid_check_d_strict))

        #     valid_mask = torch.logical_and(valid_mask_ab, valid_mask_bc)
        #     valid_mask = torch.logical_and(valid_mask, valid_mask_cd)
            
        #     a = a[:, valid_mask]
        #     b = b[:, valid_mask]   
        #     c = c[:, valid_mask]
        #     d = d[:, valid_mask]   
        
        if a.numel() == 0 or b.numel() == 0 or c.numel() == 0 or d.numel() == 0:
            return torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), 0
        
        # sidx = 0
        # plt.figure()
        # plt.imshow(depth_gt.squeeze().cpu().numpy())
        # circle = plt.Circle((a[1][sidx], a[0][sidx]), 3, color='r')
        # plt.gca().add_patch(circle)
        # circle = plt.Circle((b[1][sidx], b[0][sidx]), 3, color='b')
        # plt.gca().add_patch(circle)
        # circle = plt.Circle((c[1][sidx], c[0][sidx]), 3, color='g')
        # plt.gca().add_patch(circle)
        # circle = plt.Circle((d[1][sidx], d[0][sidx]), 3, color='yellow')
        # plt.gca().add_patch(circle)
        # plt.savefig('/ibex/ai/home/liz0l/codes/PatchRefiner/work_dir/debug/debug_{}_{}.png'.format(self.idx, self.idx_inner))
        # self.idx_inner += 1
        
        A = torch.cat((a,b,c), 1)
        B = torch.cat((b,c,d), 1)
        sumple_num = A.shape[1]
        
        inputs_A = inputs[A[0, :], A[1, :]]
        inputs_B = inputs[B[0, :], B[1, :]]
        targets_A = targets[A[0, :], A[1, :]]
        targets_B = targets[B[0, :], B[1, :]]

        return inputs_A, inputs_B, targets_A, targets_B, sumple_num


    def randomSampling(self, inputs, targets, valid_part, sample_num):
        # Apply masks to get the valid indices for missing and valid parts
        valid_indices = torch.nonzero(valid_part.float()).squeeze()

        # Ensure that we have enough points to sample from
        sample_num = min(sample_num, len(valid_indices))

        # Shuffle and sample indices from the missing and valid parts
        shuffle_valid_indices_1 = torch.randperm(len(valid_indices))[:sample_num].cuda()
        shuffle_valid_indices_2 = torch.randperm(len(valid_indices))[:sample_num].cuda()
        
        # Select the sampled points for inputs and targets based on the shuffled indices
        inputs_A = inputs[valid_indices[shuffle_valid_indices_1]]
        inputs_B = inputs[valid_indices[shuffle_valid_indices_2]]

        targets_A = targets[valid_indices[shuffle_valid_indices_1]]
        targets_B = targets[valid_indices[shuffle_valid_indices_2]]
        return inputs_A, inputs_B, targets_A, targets_B, sample_num

    def forward(self, inputs, targets, images, depth_gt=None, interpolate=True):
        if interpolate:
            targets = F.interpolate(targets, inputs.shape[-2:], mode='bilinear', align_corners=True)
            images = F.interpolate(images, inputs.shape[-2:], mode='bilinear', align_corners=True)
            depth_gt = F.interpolate(depth_gt, inputs.shape[-2:], mode='bilinear', align_corners=True)
            
        n, _, _, _= inputs.size()
        
        # strict_mask is a range mask
        strict_mask = torch.logical_and(depth_gt>self.min_depth, depth_gt<self.max_depth)
        
        if self.only_missing_area:
            masks = depth_gt == self.missing_value # only consider missing values in semi loss
        else:
            masks = torch.ones_like(strict_mask).bool()

        # edges_img, thetas_img = self.getEdge(images)

        edges_imgs = []
        for i in range(n):
            edges_img = extract_edges(targets[i, 0].detach().cpu(), use_canny=True, preprocess='log')
            edges_img = edges_img > 0
            # plt.figure()
            # plt.imshow(edges_img)
            # plt.savefig('/ibex/ai/home/liz0l/codes/PatchRefiner/work_dir/debug/debug_{}_{}_mask_ori.png'.format(self.idx, i))
            edges_img = torch.from_numpy(edges_img).cuda()
            edges_imgs.append(edges_img)
        edges_img = torch.stack(edges_imgs, dim=0).unsqueeze(dim=1)
        thetas_img = kornia.filters.sobel(targets, normalized=True, eps=1e-6)
        
        # initialization
        loss = torch.DoubleTensor([0.0]).cuda()
        sample_num_sum = torch.tensor([0.0])
        for i in range(n):
            # Edge-Guided sampling
            inputs_A, inputs_B, targets_A, targets_B, sample_num_e = self.edgeGuidedSampling(
                inputs[i].squeeze(), 
                targets[i].squeeze(), 
                edges_img[i].squeeze(), 
                thetas_img[i].squeeze(), 
                masks[i].squeeze(), 
                depth_gt[i].squeeze(),
                strict_mask[i].squeeze())
            sample_num_sum += sample_num_e
            
            if sample_num_e == 0:
                continue
            if len(inputs_A) == 0 or len(inputs_B) == 0 or len(targets_A) == 0 or len(targets_B) == 0:
                continue
            
            try:
                inputs_A_r, inputs_B_r, targets_A_r, targets_B_r, sample_num_r = self.randomSampling(
                    inputs[i].squeeze().view(-1), 
                    targets[i].squeeze().view(-1), 
                    strict_mask[i].squeeze().view(-1), 
                    sample_num_e)
                sample_num_sum += sample_num_r
            
                # Combine EGS + RS
                inputs_A = torch.cat((inputs_A, inputs_A_r), 0)
                inputs_B = torch.cat((inputs_B, inputs_B_r), 0)
                targets_A = torch.cat((targets_A, targets_A_r), 0)
                targets_B = torch.cat((targets_B, targets_B_r), 0)
                
            except TypeError as e:
                print_log(e, logger='current')
            
            inputs_A = inputs_A / (250 / 80)
            inputs_B = inputs_B / (250 / 80)
            
            # GT ordinal relationship
            target_ratio = torch.div(targets_A+1e-6, targets_B+1e-6)
            target_weight = torch.abs(targets_A - targets_B) / (torch.max(torch.abs(targets_A - targets_B)) + 1e-6) # avoid nan
            target_weight = torch.exp(target_weight)
            
            mask_eq = target_ratio.lt(1.0 + self.sigma) * target_ratio.gt(1.0/(1.0+self.sigma))
            labels = torch.zeros_like(target_ratio)
            labels[target_ratio.ge(1.0 + self.sigma)] = 1
            labels[target_ratio.le(1.0/(1.0+self.sigma))] = -1

            if self.reweight_target:
                equal_loss = (inputs_A - inputs_B).pow(2) / target_weight * mask_eq.double() # can also use the weight
            else:
                equal_loss = (inputs_A - inputs_B).pow(2) * mask_eq.double()
                
            if self.reweight_target:
                unequal_loss = torch.log(1 + torch.exp(((-inputs_A + inputs_B) / target_weight) * labels)) * (~mask_eq).double()
            else:
                unequal_loss = torch.log(1 + torch.exp((-inputs_A + inputs_B) * labels)) * (~mask_eq).double()
            
            # Please comment the regularization term if you don't want to use the multi-scale gradient matching loss !!!
            loss = loss + self.alpha * equal_loss.mean() + 1.0 * unequal_loss.mean() #+  0.2 * regularization_loss.double()
            
        self.idx += 1
        return loss[0].float()/n, float(sample_num_sum/n)
    

def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


@MODELS.register_module()
class ScaleAndShiftInvariantDALoss(nn.Module):
    def __init__(self, grad_matching, **kargs):
        super().__init__()
        self.grad_matching = grad_matching
        self.name = "SSILoss"

    def forward(self, prediction, target, gt_depth, mask, min_depth=None, max_depth=None, **kwargs):
        
        _, _, h_i, w_i = prediction.shape
        _, _, h_t, w_t = target.shape
    
        if h_i != h_t or w_i != w_t:
            prediction = F.interpolate(prediction, (h_t, w_t), mode='bilinear', align_corners=True)

        prediction, target, mask = prediction.squeeze(), target.squeeze(), mask.squeeze()
        
        if torch.sum(mask) <= 1:
            print_log("torch.sum(mask) <= 1, hack to skip avoiding bugs", logger='current')
            return input * 0.0
        
        assert prediction.shape == target.shape, f"Shape mismatch: Expected same shape but got {prediction.shape} and {target.shape}."

        shift_pred = torch.mean(prediction[mask])
        shift_gt = torch.mean(target[mask])
        scale_pred = torch.std(prediction[mask])
        scale_gt = torch.std(target[mask])
        
        scaled_prediction = (prediction - shift_pred) / scale_pred
        scale_target = (target - shift_gt) / scale_gt
        
        sampling_mask = mask
        if self.grad_matching:
            N = torch.sum(sampling_mask)
            d_diff = scaled_prediction - scale_target
            d_diff = torch.mul(d_diff, sampling_mask)

            v_gradient = torch.abs(d_diff[:, 0:-2, :] - d_diff[:, 2:, :])
            v_mask = torch.mul(sampling_mask[:, 0:-2, :], sampling_mask[:, 2:, :])
            v_gradient = torch.mul(v_gradient, v_mask)

            h_gradient = torch.abs(d_diff[:, :, 0:-2] - d_diff[:, :, 2:])
            h_mask = torch.mul(sampling_mask[:, :, 0:-2], sampling_mask[:, :, 2:])
            h_gradient = torch.mul(h_gradient, h_mask)

            gradient_loss = torch.sum(h_gradient) + torch.sum(v_gradient)
            loss = gradient_loss / N
        else:
            # loss = nn.functional.l1_loss(scaled_prediction[sampling_mask], pseudo_label[sampling_mask])
            loss = nn.functional.l1_loss(scaled_prediction[mask], scale_target[mask])
            
        return loss
    
@MODELS.register_module()
class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, ssi=True, only_missing_area=False, grad_matching=False, inverse=False, **kargs):
        super().__init__()
        self.name = "SSILoss"
        self.ssi = ssi
        self.only_missing_area = only_missing_area
        self.grad_matching = grad_matching
        self.inverse = inverse

    # def forward(self, prediction, target, mask, interpolate=True, return_interpolated=False):
    def forward(self, prediction, pseudo_label, gt_depth, mask, min_depth, max_depth):
        
        bs, _, h_i, w_i = prediction.shape
        _, _, h_t, w_t = pseudo_label.shape
        if h_i != h_t or w_i != w_t:
            prediction = F.interpolate(prediction, (h_t, w_t), mode='bilinear', align_corners=True)
        prediction, pseudo_label, mask = prediction.squeeze(), pseudo_label.squeeze(), mask.squeeze()
        if torch.sum(mask) <= 1:
            print_log("torch.sum(mask) <= 1, hack to skip avoiding bugs", logger='current')
            return prediction * 0.0
        assert prediction.shape == pseudo_label.shape, f"Shape mismatch: Expected same shape but got {prediction.shape} and {pseudo_label.shape}."

        if self.inverse:
            sampling_mask = mask # all values
            N = torch.sum(sampling_mask)

            v_pred = prediction[:, 0:-2, :] - prediction[:, 2:, :]
            v_target = pseudo_label[:, 0:-2, :] - pseudo_label[:, 2:, :]
            v_mask = torch.mul(sampling_mask[:, 0:-2, :], sampling_mask[:, 2:, :])
            
            h_pred = prediction[:, :, 0:-2] - prediction[:, :, 2:]
            h_target = pseudo_label[:, :, 0:-2] - pseudo_label[:, :, 2:]
            h_mask = torch.mul(sampling_mask[:, :, 0:-2], sampling_mask[:, :, 2:])
            
            scale, shift = compute_scale_and_shift(v_pred, v_target, v_mask)
            scaled_v_pred = scale.view(-1, 1, 1) * v_pred + shift.view(-1, 1, 1)

            scale, shift = compute_scale_and_shift(h_pred, h_target, h_mask)
            scaled_h_pred = scale.view(-1, 1, 1) * h_pred + shift.view(-1, 1, 1)
            
            loss_v = torch.abs(scaled_v_pred - v_target) * v_mask
            loss_h = torch.abs(scaled_h_pred - h_target) * h_mask
            
            gradient_loss = torch.sum(loss_v) + torch.sum(loss_h)
            loss = gradient_loss / N
            
            return loss
        
        else:
        
            if self.ssi:
                scale, shift = compute_scale_and_shift(prediction, pseudo_label, mask)
                scaled_prediction = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)
            else:
                scaled_prediction = prediction

            if self.only_missing_area:
                missing_mask = gt_depth == 0.
                missing_mask_extend = kornia.filters.gaussian_blur2d(missing_mask.float(), kernel_size=(7, 7), sigma=(5., 5.), border_type='reflect', separable=True)
                missing_mask_extend = missing_mask_extend > 0
                missing_mask_extend = missing_mask_extend.squeeze()
                
                prediction, pseudo_label, gt_depth = prediction.squeeze(), pseudo_label.squeeze(), gt_depth.squeeze()
                
                # compute mask, edges
                valid_mask = torch.logical_and(gt_depth>min_depth, gt_depth<max_depth)
                missing_value_mask = torch.logical_and(valid_mask, missing_mask_extend)
                
                # get edge
                pesudo_edge_list = []
                for idx in range(bs):
                    pesudo_edge = torch.from_numpy(extract_edges(pseudo_label[idx].detach().cpu(), use_canny=True, preprocess='log')).cuda()
                    pesudo_edge_list.append(pesudo_edge)
                pesudo_edges = torch.stack(pesudo_edge_list, dim=0).unsqueeze(dim=1)
                pesudo_edges_extend = kornia.filters.gaussian_blur2d(pesudo_edges.float(), kernel_size=(7, 7), sigma=(5., 5.), border_type='reflect', separable=True)
                pesudo_edges = pesudo_edges_extend > 0 # edge mask
                pesudo_edges = pesudo_edges.squeeze()
                sampling_mask = torch.logical_and(missing_value_mask, pesudo_edges)
            else:
                sampling_mask = mask # all values
                    
            if self.grad_matching:
                N = torch.sum(sampling_mask)
                d_diff = scaled_prediction - pseudo_label
                d_diff = torch.mul(d_diff, sampling_mask)

                v_gradient = torch.abs(d_diff[:, 0:-2, :] - d_diff[:, 2:, :])
                v_mask = torch.mul(sampling_mask[:, 0:-2, :], sampling_mask[:, 2:, :])
                v_gradient = torch.mul(v_gradient, v_mask)

                h_gradient = torch.abs(d_diff[:, :, 0:-2] - d_diff[:, :, 2:])
                h_mask = torch.mul(sampling_mask[:, :, 0:-2], sampling_mask[:, :, 2:])
                h_gradient = torch.mul(h_gradient, h_mask)

                gradient_loss = torch.sum(h_gradient) + torch.sum(v_gradient)
                loss = gradient_loss / N
            else:
                loss = nn.functional.l1_loss(scaled_prediction[sampling_mask], pseudo_label[sampling_mask])
            
            return loss


@MODELS.register_module()
class ScaleAndShiftInvariantUncertLoss(nn.Module):
    def __init__(self, only_missing_area=False, grad_matching=False, **kargs):
        super().__init__()
        self.name = "SSILoss"
        self.only_missing_area = only_missing_area
        self.grad_matching = grad_matching

    def forward(self, prediction, pseudo_label, gt_depth, mask, min_depth, max_depth, uncert):
        bs, _, h_i, w_i = prediction.shape
        _, _, h_t, w_t = pseudo_label.shape
        _, _, h_u, w_u = uncert.shape
        
        if h_i != h_t or w_i != w_t:
            prediction = F.interpolate(prediction, (h_t, w_t), mode='bilinear', align_corners=True)
        if h_u != h_t or w_u != w_t:
            uncert = F.interpolate(uncert, (h_t, w_t), mode='bilinear', align_corners=True)

        prediction, pseudo_label, mask, uncert = prediction.squeeze(), pseudo_label.squeeze(), mask.squeeze(), uncert.squeeze()
        
        if torch.sum(mask) <= 1:
            print_log("torch.sum(mask) <= 1, hack to skip avoiding bugs", logger='current')
            return input * 0.0
        
        assert prediction.shape == pseudo_label.shape, f"Shape mismatch: Expected same shape but got {prediction.shape} and {pseudo_label.shape}."

        scale, shift = compute_scale_and_shift(prediction, pseudo_label, mask)

        scaled_prediction = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        if self.only_missing_area:
            missing_mask = gt_depth == 0.
            missing_mask_extend = kornia.filters.gaussian_blur2d(missing_mask.float(), kernel_size=(7, 7), sigma=(5., 5.), border_type='reflect', separable=True)
            missing_mask_extend = missing_mask_extend > 0
            missing_mask_extend = missing_mask_extend.squeeze()
            
            prediction, pseudo_label, gt_depth = prediction.squeeze(), pseudo_label.squeeze(), gt_depth.squeeze()
            
            # compute mask, edges
            valid_mask = torch.logical_and(gt_depth>min_depth, gt_depth<max_depth)
            missing_value_mask = torch.logical_and(valid_mask, missing_mask_extend)
            
            # get edge
            pesudo_edge_list = []
            for idx in range(bs):
                pesudo_edge = torch.from_numpy(extract_edges(pseudo_label[idx].detach().cpu(), use_canny=True, preprocess='log')).cuda()
                pesudo_edge_list.append(pesudo_edge)
            pesudo_edges = torch.stack(pesudo_edge_list, dim=0).unsqueeze(dim=1)
            pesudo_edges_extend = kornia.filters.gaussian_blur2d(pesudo_edges.float(), kernel_size=(7, 7), sigma=(5., 5.), border_type='reflect', separable=True)
            pesudo_edges = pesudo_edges_extend > 0 # edge mask
            pesudo_edges = pesudo_edges.squeeze()
            sampling_mask = torch.logical_and(missing_value_mask, pesudo_edges)
        else:
            sampling_mask = mask # all values
        
        conf = 1 - uncert
        if self.grad_matching:
            N = torch.sum(sampling_mask)
            d_diff = scaled_prediction - pseudo_label
            d_diff = torch.mul(d_diff, sampling_mask)

            v_gradient = torch.abs(d_diff[:, 0:-2, :] - d_diff[:, 2:, :])
            v_mask = torch.mul(sampling_mask[:, 0:-2, :], sampling_mask[:, 2:, :])
            v_conf = (conf[:, 0:-2, :] + conf[:, 2:, :]) / 2
            v_gradient = torch.mul(v_gradient, v_mask) * v_conf

            h_gradient = torch.abs(d_diff[:, :, 0:-2] - d_diff[:, :, 2:])
            h_mask = torch.mul(sampling_mask[:, :, 0:-2], sampling_mask[:, :, 2:])
            h_conf = (conf[:, :, 0:-2] + conf[:, :, 2:]) / 2
            h_gradient = torch.mul(h_gradient, h_mask) * h_conf

            gradient_loss = torch.sum(h_gradient) + torch.sum(v_gradient)
            loss = gradient_loss / N
        else:
            loss = nn.functional.l1_loss(scaled_prediction[sampling_mask], pseudo_label[sampling_mask])
        
        return loss


@MODELS.register_module()
class BaseDistillLoss(nn.Module):
    def __init__(self, student_trans='conv', teacher_trans='raw', embed_dims=256, ssi_feat=False):
        super().__init__()
        
        connecter = []
        if 'conv' in student_trans:
            s_op = nn.Conv2d(embed_dims, embed_dims, kernel_size=1, stride=1, padding=0)
            connecter.append(s_op)
        self.connector = nn.Sequential(*connecter)

        self.ssi_feat = ssi_feat
        
    def forward(self, student_feat, teacher_feat, depth_gt, min_depth, max_depth):
        student_feat = self.connector(student_feat)
        
        bs, c, h_i, w_i = student_feat.shape
        _, _, h_t, w_t = teacher_feat.shape
        _, _, h_g, w_g = depth_gt.shape

        # align depth gt shape to the features
        if h_i != h_g or w_i != w_g:
            depth_gt = F.interpolate(depth_gt, (h_t, w_t))
        
        valid_mask = torch.logical_and(depth_gt > min_depth, depth_gt < max_depth)
        valid_mask = valid_mask.repeat(1, c, 1, 1)
        
        if self.ssi_feat:
            student_feat_reshape = student_feat.reshape(bs * c, h_t, w_t)
            teacher_feat_reshape = teacher_feat.reshape(bs * c, h_t, w_t)
            valid_mask_reshape = valid_mask.reshape(bs * c, h_t, w_t)
            scale, shift = compute_scale_and_shift(student_feat_reshape, teacher_feat_reshape, valid_mask_reshape)
            student_feat_reshape = scale.view(-1, 1, 1) * student_feat_reshape + shift.view(-1, 1, 1)
            # print(student_feat_reshape.shape, teacher_feat_reshape.shape, valid_mask_reshape.shape, scale.shape, shift.shape)
            student_feat = student_feat_reshape.reshape(bs, c, h_t, w_t)
        
        loss = F.mse_loss(student_feat[valid_mask], teacher_feat[valid_mask])
        return loss
        
# # https://github.com/irfanICMLL/structure_knowledge_distillation/blob/master/utils/utils.py
def L2(f_):
    return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0],1,f_.shape[2],f_.shape[3]) + 1e-8

def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat/tmp
    feat = feat.reshape(feat.shape[0],feat.shape[1],-1)
    return torch.einsum('icm,icn->imn', [feat, feat])

def sim_dis_compute(f_S, f_T, mask):
    valid_mask = torch.einsum('icm,icn->imn', [mask.reshape(mask.shape[0], mask.shape[1], -1), mask.reshape(mask.shape[0], mask.shape[1], -1)])
    similarity_f_T = similarity(f_T)
    similarity_f_S = similarity(f_S)
    sim_err = (valid_mask*((similarity_f_T - similarity_f_S)**2))/(torch.sum(valid_mask))
    sim_dis = sim_err.sum()
    return sim_dis

# def sim_dis_compute(f_S, f_T):
#     sim_err = (((similarity(f_T) - similarity(f_S))**2))/((f_T.shape[-1]*f_T.shape[-2]))/f_T.shape[0]
#     sim_dis = sim_err.sum()
#     return sim_dis

@MODELS.register_module()
class StructureDistillLoss(nn.Module):
    def __init__(
        self, 
        student_trans='conv', 
        teacher_trans='raw', 
        embed_dims=256,
        # window_size=[3, 7, 15, 31, 63],
        # window_size=[3, 7, 15, 31],
        # window_size=[63],
        window_size=[31],
        gamma_window=0.3,
        process_h=384,
        process_w=512,
        region_num=100,):
        
        super().__init__()
        
        self.embed_dims = embed_dims
        connecter = []
        if 'conv' in student_trans:
            s_op = nn.Conv2d(embed_dims, embed_dims, kernel_size=1, stride=1, padding=0)
            connecter.append(s_op)
        self.connector = nn.Sequential(*connecter)

        self.window_size = window_size 
        self.gamma_window = gamma_window
        self.anchor_generator = RandomBBoxQueries(4, process_h, process_w, self.window_size, N=region_num)
        self.region_num = region_num
        
    def forward(self, student_feat, teacher_feat, depth_gt, min_depth, max_depth):
        
        self.anchor_generator.to(student_feat.device)
        student_feat = self.connector(student_feat)
        
        bs, c, h_i, w_i = student_feat.shape
        _, _, h_t, w_t = teacher_feat.shape
        _, _, h_g, w_g = depth_gt.shape

        # align depth gt shape to the features
        if h_i != h_g or w_i != w_g:
            depth_gt = F.interpolate(depth_gt, (h_t, w_t))
        
        valid_mask = torch.logical_and(depth_gt > min_depth, depth_gt < max_depth)
        valid_mask = valid_mask.repeat(1, c, 1, 1)
        student_feat[valid_mask == 0] = 0
        teacher_feat[valid_mask == 0] = 0
        
        student_feat = student_feat.unsqueeze(dim=1).expand(-1, self.region_num, -1, -1, -1).flatten(-2)
        teacher_feat = teacher_feat.unsqueeze(dim=1).expand(-1, self.region_num, -1, -1, -1).flatten(-2) # B, num_region, C, HxW
        sampling_mask = valid_mask.unsqueeze(dim=1).expand(-1, self.region_num, -1, -1, -1).flatten(-2) # B, num_region, C, HxW
        # print(student_feat.shape, teacher_feat.shape, sampling_mask.shape)

        # start calculating loss
        loss = 0
        w_window = 1.0
        w_window_sum = 0.0
        
        for idx, win_size in enumerate(self.window_size):
            if idx > 0 :
                w_window = w_window * self.gamma_window
            
            abs_coords = self.anchor_generator.absolute[win_size]  # B, N, 2; babs[b,n] = [x,y]
            B, N, _two = abs_coords.shape
            k = win_size // 2
            x = torch.arange(-k, k+1)
            y = torch.arange(-k, k+1)
            Y, X = torch.meshgrid(y, x)
            base_coords = torch.stack((X, Y), dim=0)[None, None,...].to(student_feat.device)  # .shape 1, 1, 2, k, k
            
            coords = abs_coords[...,None,None] + base_coords  # shape B, N, 2, k, k
            
            x = coords[:,:,0,:,:]
            y = coords[:,:,1,:,:]
            flatten_indices = y * w_t + x  # .shape B, N, k, k
            
            flatten_flatten_indices = flatten_indices.flatten(2)  # .shape B, N, kxk (batch num_region, kxk)
            flatten_flatten_indices = flatten_flatten_indices.unsqueeze(dim=2).repeat(1, 1, self.embed_dims, 1) # .shape B, N, C, kxk

            # B, N, C, kxk
            sampling_mask_sample = torch.gather(sampling_mask, dim=-1, index=flatten_flatten_indices.long())
            # pesudo_edges_sample = torch.gather(pesudo_edges, dim=-1, index=flatten_flatten_indices.long())
            student_feat_sample = torch.gather(student_feat, dim=-1, index=flatten_flatten_indices.long())
            teacher_feat_sample = torch.gather(teacher_feat, dim=-1, index=flatten_flatten_indices.long())
            
            for mask, stu_f, tea_f in zip(sampling_mask_sample, student_feat_sample, teacher_feat_sample):
                
                stu_f = stu_f.unsqueeze(-1)
                tea_f = tea_f.unsqueeze(-1)
                hack_mask = mask.unsqueeze(-1)
                loss_win = sim_dis_compute(stu_f, tea_f, hack_mask[:, 0:1, :, :])
                # loss_win = sim_dis_compute(stu_f, tea_f)
                loss += loss_win * w_window / bs # window weight and batch size

            w_window_sum += w_window # num of windows
        loss = loss / w_window_sum
        return loss

@MODELS.register_module()
class GradMatchLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "GMLoss"

    def forward(self, input, target, min_depth, max_depth, additional_mask=None):
        _, _, h_i, w_i = input.shape
        _, _, h_t, w_t = target.shape
        
        if h_i != h_t or w_i != w_t:
            input = F.interpolate(input, (h_t, w_t), mode='bilinear', align_corners=True)
        
        mask = torch.logical_and(target>min_depth, target<max_depth)
        
        if additional_mask is not None:
            mask_merge = torch.logical_and(mask, additional_mask)
            if torch.sum(mask_merge) >= h_i * w_i * 0.001:
                mask = mask_merge
            else:
                print_log("torch.sum(mask_merge) < h_i * w_i * 0.001, reduce to previous mask for stable training", logger='current')
        
        if torch.sum(mask) <= 1:
            print_log("torch.sum(mask) <= 1, hack to skip avoiding nan", logger='current')
            return input * 0.0
        
        N = torch.sum(mask)
        d_diff = input - target
        d_diff = torch.mul(d_diff, mask)

        v_gradient = torch.abs(d_diff[:, 0:-2, :] - d_diff[:, 2:, :])
        v_mask = torch.mul(mask[:, 0:-2, :], mask[:, 2:, :])
        v_gradient = torch.mul(v_gradient, v_mask)

        h_gradient = torch.abs(d_diff[:, :, 0:-2] - d_diff[:, :, 2:])
        h_mask = torch.mul(mask[:, :, 0:-2], mask[:, :, 2:])
        h_gradient = torch.mul(h_gradient, h_mask)

        gradient_loss = torch.sum(h_gradient) + torch.sum(v_gradient)
        loss = gradient_loss / N
        
        return loss


@MODELS.register_module()
class EALoss(nn.Module):
    """SILog loss (pixel-wise)"""
    def __init__(self, beta=0.15, **kwargs):
        super(EALoss, self).__init__()
        self.name = 'SILog'
        self.beta = beta

    def forward(self, input, target, coarse, min_depth, max_depth, additional_mask=None):
        _, _, h_i, w_i = input.shape
        _, _, h_t, w_t = target.shape
        _, _, h_c, w_c = coarse.shape
        
        if h_i != h_t or w_i != w_t:
            input = F.interpolate(input, (h_t, w_t), mode='bilinear', align_corners=True)
        if h_c != h_t or w_c != w_t:
            coarse = F.interpolate(coarse, (h_t, w_t), mode='bilinear', align_corners=True)
        
        mask = torch.logical_and(target>min_depth, target<max_depth)
        
        if additional_mask is not None:
            mask_merge = torch.logical_and(mask, additional_mask)
            if torch.sum(mask_merge) >= h_i * w_i * 0.001:
                mask = mask_merge
            else:
                print_log("torch.sum(mask_merge) < h_i * w_i * 0.001, reduce to previous mask for stable training", logger='current')
        
        if torch.sum(mask) <= 1:
            print_log("torch.sum(mask) <= 1, hack to skip avoiding nan", logger='current')
            return input * 0.0
        
        input = input[mask]
        target = target[mask]
        coarse = coarse[mask]
        
        alpha = 1e-7
        
        g_c = torch.log(coarse + alpha) - torch.log(target + alpha)
        coarse_error = torch.pow(g_c, 2)
        
        g_f = (torch.log(input + alpha) - torch.log(target + alpha)) * coarse_error
        Dg = torch.var(g_f) + self.beta * torch.pow(torch.mean(g_f), 2)
        loss = 10 * torch.sqrt(Dg)

        if torch.isnan(loss):
            print_log("Nan SILog loss", logger='current')
            print_log("input: {}".format(input.shape), logger='current')
            print_log("target: {}".format(target.shape), logger='current')
            
            print_log("G: {}".format(torch.sum(torch.isnan(g))), logger='current')
            print_log("Input min: {} max: {}".format(torch.min(input), torch.max(input)), logger='current')
            print_log("Target min: {} max: {}".format(torch.min(target), torch.max(target)), logger='current')
            print_log("Dg: {}".format(torch.isnan(Dg)), logger='current')
            print_log("loss: {}".format(torch.isnan(loss)), logger='current')

        return loss
