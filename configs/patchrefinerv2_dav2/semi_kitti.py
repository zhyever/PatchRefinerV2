_base_ = [
    '../_base_/datasets/cityscapes.py',
    # '../_base_/datasets/u4k.py',
    '../_base_/datasets/general_dataset.py',
    '../_base_/run_time.py',
    # './base_pr_s2r_optim.py',
]

max_depth=80
min_depth = 1e-3

student_model=dict(
    type='PatchRefinerPlus',
    config=dict(
        # e2e_training=False,
        e2e_training=True,
        pretrain_stage=False,
        
        image_raw_shape=[352, 1216],
        patch_process_shape=[448, 448],
        patch_raw_shape=[176, 304], 
        patch_split_num=[2, 4],

        fusion_feat_level=6,
        min_depth=1e-3,
        max_depth=80,

        pretrain_coarse_model='./work_dir/project_folder/plus/dav2/kitti/coarse_pretrain_kitti_448/checkpoint_12.pth', # will load the coarse model     
        strategy_refiner_target='offset_coarse',
        # strategy_refiner_target='direct',
        
        coarse_branch=dict(
            type='DA2',
            pretrained='work_dir/project_folder/depthanythingv2/depth_anything_v2_metric_hypersim_vitl.pth',
            model_cfg=dict(
                encoder='vitl', 
                features=256, 
                out_channels=[256, 512, 1024, 1024])),
        
        refiner=dict(
            fine_branch=dict(
                type='LightWeightRefiner',
                coarse_condition=True,
                with_decoder=False,
                encoder_name='tf_efficientnet_b5_ap',),
            fusion_model=dict(
                type='BiDirectionalFusion',
                encoder_name='tf_efficientnet_b5_ap',
                coarse2fine=True, # mid module
                coarse2fine_type='coarse-gated',
                coarse_chl=[128, 256, 256, 256, 256, 256], 
                fine_chl=[24, 40, 64, 176, 512],
                fine_chl_after_coarse2fine=[128, 256, 256, 256, 256, 256],
                temp_chl=[32, 64, 64, 128, 256, 512],
                dec_chl=[512, 256, 128, 64, 32])),
        
        sigloss=dict(type='SILogLoss'),
        gmloss=dict(type='GradMatchLoss'),
        
        sigweight=1,
        pre_norm_bbox=True,
        
        # pretrained=None,
        whole_pretrained='work_dir/project_folder/plus/dav2/kitti/plus_eff_kitti/checkpoint_12.pth',
))


model=dict(
    type='PatchRefinerSemi',
    model_cfg_student=student_model,
    model_cfg_teacher=None,
    mix_loss=False,
    edge_loss_weight=0.5,
    edgeloss=dict(
        type='ScaleAndShiftInvariantLoss',
        only_missing_area=False,
        grad_matching=True),
    sigloss=dict(type='SILogLoss'),
    min_depth=min_depth,
    max_depth=max_depth,)

collect_input_args=['image_lr', 'image_hr', 'crops_image_hr', 'depth_gt', 'crop_depths', 'bboxs', 'center_mask', 'pseudo_label', 'seg_image']

project='patchrefinerplus'
# train_cfg=dict(max_epochs=2, val_interval=50, save_checkpoint_interval=2, log_interval=100, train_log_img_interval=1000, val_log_img_interval=50, val_type='iter_base', eval_start=0)
# train_cfg=dict(max_epochs=2, val_interval=1, save_checkpoint_interval=2, log_interval=100, train_log_img_interval=1000, val_log_img_interval=50, val_type='epoch_base', eval_start=0)
train_cfg=dict(max_epochs=3, val_interval=1, save_checkpoint_interval=3, log_interval=100, train_log_img_interval=1000, val_log_img_interval=50, val_type='epoch_base', eval_start=0)

convert_syncbn=False
find_unused_parameters=True

optim_wrapper=dict(
    # optimizer=dict(type='AdamW', lr=0.000161, weight_decay=0.01),
    optimizer=dict(type='AdamW', lr=0.00012, weight_decay=0.01),
    clip_grad=dict(type='norm', max_norm=35, norm_type=2), # norm clip
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys={
            'student_model.refiner_fine_branch.refiner_encoder': dict(lr_mult=0.1, decay_mult=1.0),
            'student_model.coarse_branch': dict(lr_mult=1/30, decay_mult=1.0),
        }))

param_scheduler=dict(
    cycle_momentum=True,
    base_momentum=0.85,
    max_momentum=0.95,
    div_factor=1,
    final_div_factor=100,
    pct_start=0.3,
    three_phase=False,)

convert_syncbn=False
find_unused_parameters=True

train_dataloader=dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        resize_mode='depth-anything',
        pseudo_label_path='???',
        transform_cfg=dict(
            image_raw_shape=[352, 1216],
            network_process_size=[448, 448])))

val_dataloader=dict(
    dataset=dict(
        resize_mode='depth-anything',
        transform_cfg=dict(
            network_process_size=[448, 448])))

general_dataloader=dict(
    dataset=dict(
        network_process_size=(448, 448),
        resize_mode='depth-anything'))