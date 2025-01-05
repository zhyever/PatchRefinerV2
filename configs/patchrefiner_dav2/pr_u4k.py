_base_ = [
    '../_base_/datasets/u4k.py',
    '../_base_/datasets/general_dataset.py',
    '../_base_/run_time.py',
]

min_depth=1e-3
max_depth=80

model=dict(
    type='PatchRefiner',
    config=dict(
        image_raw_shape=[2160, 3840],
        patch_process_shape=[448, 448],
        patch_raw_shape=[540, 960], 
        patch_split_num=[4, 4],

        fusion_feat_level=6,
        min_depth=1e-3,
        max_depth=80,

        # pretrain_coarse_model='./work_dir/zoedepth/u4k/coarse_pretrain/checkpoint_24.pth', # will load the coarse model     
        pretrain_coarse_model='./work_dir/project_folder/plus/dav2/u4k/dav2_pretrain_u4k_pfsetting_ft_448/checkpoint_24.pth', # will load the coarse model     
        pretrain_fine_model='./work_dir/project_folder/plus/dav2/u4k/dav2_pretrain_u4k_pfsetting_ft_448/checkpoint_24.pth', # will load the coarse model  
        
        strategy_refiner_target='offset_coarse',
        
        coarse_branch=dict(
            type='DA2',
            pretrained='work_dir/project_folder/depthanythingv2/depth_anything_v2_metric_hypersim_vitl.pth',
            model_cfg=dict(
                encoder='vitl', 
                features=256, 
                out_channels=[256, 512, 1024, 1024])),
        
        refiner=dict(
            fine_branch=dict(
                type='DA2',
                pretrained='work_dir/project_folder/depthanythingv2/depth_anything_v2_metric_hypersim_vitl.pth',
                model_cfg=dict(
                    encoder='vitl', 
                    features=256, 
                    out_channels=[256, 512, 1024, 1024])),
            fusion_model=dict(
                type='FusionUnet',
                input_chl=[128*2, 256*2, 256*2, 256*2, 256*2, 256*2],
                temp_chl=[128, 256, 256, 256, 256, 256],
                dec_chl=[256, 256, 256, 256, 128])),
        
        sigloss=dict(type='SILogLoss'),
        pretrained=None,
        pre_norm_bbox=True,
))

collect_input_args=['image_lr', 'image_hr', 'crops_image_hr', 'depth_gt', 'crop_depths', 'bboxs',]

project='patchrefiner'

train_cfg=dict(max_epochs=24, val_interval=2, save_checkpoint_interval=24, log_interval=100, train_log_img_interval=500, val_log_img_interval=30, val_type='epoch_base', eval_start=0)

optim_wrapper=dict(
    # optimizer=dict(type='AdamW', lr=0.000161, weight_decay=0.01),
    optimizer=dict(type='AdamW', lr=0.00012, weight_decay=0.01),
    clip_grad=dict(type='norm', max_norm=35, norm_type=2), # norm clip
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys={
            'refiner_fine_branch.pretrained': dict(lr_mult=1/30, decay_mult=1.0), ## change
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

convert_syncbn=True
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