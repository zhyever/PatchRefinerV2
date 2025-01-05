_base_ = [
    '../_base_/datasets/cityscapes.py',
    # '../_base_/datasets/u4k.py',
    '../_base_/datasets/general_dataset.py',
    '../_base_/run_time.py',
    # './base_pr_s2r_optim.py',
]

max_depth = 250
min_depth = 1e-3

zoe_depth_config=dict(
    type='ZoeDepth',
    
    min_depth=min_depth,
    max_depth=max_depth,
    
    # some important params
    midas_model_type='DPT_BEiT_L_384',
    pretrained_resource=None,
    use_pretrained_midas=True,
    train_midas=True,
    freeze_midas_bn=True,
    do_resize=False, # do not resize image in midas

    # default settings
    attractor_alpha=1000,
    attractor_gamma=2,
    attractor_kind='mean',
    attractor_type='inv',
    aug=True,
    bin_centers_type='softplus',
    bin_embedding_dim=128,
    clip_grad=0.1,
    dataset='nyu',
    distributed=True,
    force_keep_ar=True,
    gpu='NULL',
    img_size=[384, 512],
    inverse_midas=False,
    log_images_every=0.1,
    max_temp=50.0,
    max_translation=100,
    memory_efficient=True,
    min_temp=0.0212,
    model='zoedepth',
    n_attractors=[16, 8, 4, 1],
    n_bins=64,
    name='ZoeDepth',
    notes='',
    output_distribution='logbinomial',
    prefetch=False,
    print_losses=False,
    project='ZoeDepth',
    random_crop=False,
    random_translate=False,
    root='.',
    save_dir='',
    shared_dict='NULL',
    tags='',
    translate_prob=0.2,
    uid='NULL',
    use_amp=False,
    use_shared_dict=False,
    validate_every=0.25,
    version_name='v1',
    workers=16,
)

student_model=dict(
    type='PatchRefinerPlus',
    config=dict(
        e2e_training=True,
        pretrain_stage=False,
        
        image_raw_shape=[1024, 2048],
        patch_process_shape=[384, 512],
        patch_raw_shape=[256, 512], 
        patch_split_num=[4, 4],

        fusion_feat_level=6,
        min_depth=1e-3,
        max_depth=max_depth,

        pretrain_coarse_model='./work_dir/pr_zoedepth/cityscapes/coarse_pretrain/checkpoint_05.pth', # will load the coarse model     
        strategy_refiner_target='offset_coarse',
        # strategy_refiner_target='direct',
        
        coarse_branch=zoe_depth_config,
        refiner=dict(
            fine_branch=dict(
                type='LightWeightRefiner',
                coarse_condition=True,
                with_decoder=False,
                # encoder_channels=[32, 32, 64, 96, 960],
                encoder_name='tf_efficientnet_b5_ap',),
            fusion_model=dict(
                type='BiDirectionalFusion',
                encoder_name='tf_efficientnet_b5_ap',
                coarse2fine=True, # mid module
                coarse2fine_type='coarse-gated',
                coarse_chl=[32, 256, 256, 256, 256, 256], 
                fine_chl=[24, 40, 64, 176, 512],
                fine_chl_after_coarse2fine=[32, 256, 256, 256, 256, 256],
                temp_chl=[32, 64, 64, 128, 256, 512],
                dec_chl=[512, 256, 128, 64, 32])),
        
        sigloss=dict(type='SILogLoss'),
        gmloss=dict(type='GradMatchLoss'),
        
        sigweight=1,
        pre_norm_bbox=True,
        
        whole_pretrained='work_dir/project_folder/plus/zoedepth/cs/eff_onlyrealdomain_cspretrain_24e/checkpoint_24.pth',
        pretrained=None,
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
            'student_model.coarse_branch': dict(lr_mult=0.1, decay_mult=1.0),
        }))


param_scheduler=dict(
    cycle_momentum=True,
    base_momentum=0.85,
    max_momentum=0.95,
    div_factor=1,
    final_div_factor=100,
    pct_start=0.3,
    three_phase=False,)

train_dataloader=dict(
    dataset=dict(
        # pseudo_label_path='./work_dir/zoedepth/u4k/patchrefiner/generate_pls_cs',
        pseudo_label_path='./work_dir/project_folder/zoedepth/u4k/patchrefiner/generate_pls_cs_ctnum',
        with_pseudo_label=True,
        transform_cfg=dict(
            image_raw_shape=[1024, 2048])))

