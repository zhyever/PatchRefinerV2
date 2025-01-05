_base_ = [
    '../_base_/datasets/u4k.py',
    '../_base_/datasets/general_dataset.py',
    '../_base_/run_time.py',
]

min_depth=1e-3
max_depth=80

model=dict(
    type='PatchRefinerPlus',
    config=dict(
        # e2e_training=False,
        e2e_training=True,
        
        pretrain_stage=False,
        
        image_raw_shape=[2160, 3840],
        patch_process_shape=[448, 448],
        patch_raw_shape=[540, 960], 
        patch_split_num=[4, 4],

        fusion_feat_level=6,
        min_depth=1e-3,
        max_depth=80,

        pretrain_coarse_model='./work_dir/project_folder/plus/dav2/u4k/dav2_pretrain_u4k_pfsetting_ft_448/checkpoint_24.pth', # will load the coarse model     
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
                # encoder_channels=[32, 32, 64, 96, 960],
                encoder_name='mobilenetv4_conv_small.e2400_r224_in1k',),
            fusion_model=dict(
                type='BiDirectionalFusion',
                encoder_name='mobilenetv4_conv_small.e2400_r224_in1k',
                coarse2fine=True, # mid module
                coarse2fine_type='coarse-gated',
                coarse_chl=[128, 256, 256, 256, 256, 256], 
                fine_chl=[32, 32, 64, 96, 960],
                fine_chl_after_coarse2fine=[128, 256, 256, 256, 256, 256],
                temp_chl=[32, 64, 64, 128, 256, 512],
                dec_chl=[512, 256, 128, 64, 32])),
                # input_chl=[32+32, 256+32, 256+32, 256+64, 256+96, 256+960],
                # input_chl=[32+32, 256+256, 256+256, 256+256, 256+256, 256+256],
                # temp_chl=[32, 64, 64, 128, 256, 512],
                # dec_chl=[512, 256, 128, 64, 32]
        
        sigloss=dict(type='SILogLoss'),
        gmloss=dict(type='GradMatchLoss'),
        
        sigweight=1,
        pre_norm_bbox=True,
        
        # pretrained=None,
        pretrained='work_dir/project_folder/plus/dav2/u4k/dav2_pretrain_mobile/checkpoint_96.pth',
        whole_pretrained=None,
))

collect_input_args=['image_lr', 'image_hr', 'crops_image_hr', 'depth_gt', 'crop_depths', 'bboxs',]

project='patchrefinerplus'

# train_cfg=dict(max_epochs=24, val_interval=2, save_checkpoint_interval=24, log_interval=100, train_log_img_interval=500, val_log_img_interval=6, val_type='epoch_base', eval_start=0)
train_cfg=dict(max_epochs=48, val_interval=4, save_checkpoint_interval=24, log_interval=100, train_log_img_interval=500, val_log_img_interval=6, val_type='epoch_base', eval_start=0)

optim_wrapper=dict(
    # optimizer=dict(type='AdamW', lr=0.000161, weight_decay=0.01),
    optimizer=dict(type='AdamW', lr=0.00012, weight_decay=0.00001),
    clip_grad=dict(type='norm', max_norm=35, norm_type=2), # norm clip
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys={
            'refiner_fine_branch.refiner_encoder': dict(lr_mult=0.1, decay_mult=1.0),
            'coarse_branch': dict(lr_mult=1/30, decay_mult=1000),
        }))

param_scheduler=dict(
    cycle_momentum=True,
    base_momentum=0.85,
    max_momentum=0.95,
    div_factor=2,
    # div_factor=10,
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
        transform_cfg=dict(
            image_raw_shape=[2160, 3840],
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