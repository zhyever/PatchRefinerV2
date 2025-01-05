_base_ = [
    '../_base_/datasets/kitti.py', '../_base_/datasets/general_dataset.py'
]

min_depth=1e-3
max_depth=80
    
model=dict(
    type='BaselinePretrain',
    min_depth=min_depth,
    max_depth=max_depth,
    target='coarse',
    coarse_branch=dict(
        type='DA2',
        pretrained='work_dir/project_folder/depthanythingv2/depth_anything_v2_metric_hypersim_vitl.pth',
        model_cfg=dict(
            encoder='vitl', 
            features=256, 
            out_channels=[256, 512, 1024, 1024])),
    fine_branch=None,
    sigloss=dict(type='SILogLoss'))

collect_input_args=['image_lr', 'crops_image_hr', 'depth_gt', 'crop_depths', 'bboxs', 'image_hr']

project='patchrefinerplus'

# train_cfg=dict(max_epochs=24, val_interval=2, save_checkpoint_interval=24, log_interval=100, train_log_img_interval=500, val_log_img_interval=50, val_type='epoch_base', eval_start=0)
train_cfg=dict(max_epochs=12, val_interval=2, save_checkpoint_interval=12, log_interval=100, train_log_img_interval=500, val_log_img_interval=50, val_type='epoch_base', eval_start=0)

# optim_wrapper=dict(
#     optimizer=dict(type='Adam', lr=1e-4),
#     clip_grad=dict(type='norm', max_norm=0.1, norm_type=2), # norm clip
#     paramwise_cfg=dict(
#         bypass_duplicate=True,
#         custom_keys={
#             'coarse_branch.pretrained': dict(lr_mult=0.1),
#         }))

optim_wrapper=dict(
    # optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.01),
    optimizer=dict(type='AdamW', lr=0.0002/50, weight_decay=0.01),
    clip_grad=dict(type='norm', max_norm=0.1, norm_type=2), # norm clip
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys={
        }))

param_scheduler=dict(
    cycle_momentum=True,
    base_momentum=0.85,
    max_momentum=0.95,
    div_factor=1,
    final_div_factor=10000,
    pct_start=0.5,
    three_phase=False,)

env_cfg=dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='forkserver'),
    dist_cfg=dict(backend='nccl'))

convert_syncbn=False
find_unused_parameters=True

train_dataloader=dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        resize_mode='depth-anything',
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