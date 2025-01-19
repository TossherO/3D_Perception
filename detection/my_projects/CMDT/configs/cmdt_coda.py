_base_ = ['default_runtime.py']
custom_imports = dict(
    imports=['my_projects.CMDT.cmdt', 'my_projects.datasets'], allow_failed_imports=False)

# optimizer
lr = 0.00014 / 4
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.01, decay_mult=5),
            'img_neck': dict(lr_mult=0.1)
        }),
    clip_grad=dict(max_norm=35, norm_type=2))
param_scheduler = [
    # learning rate scheduler
    # During the first 8 epochs, learning rate increases from 0 to lr * 10
    # during the next 12 epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
    dict(
        type='CosineAnnealingLR',
        T_max=4,
        eta_min=lr * 6,
        begin=0,
        end=4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=26,
        eta_min=lr * 1e-3,
        begin=4,
        end=30,
        by_epoch=True,
        convert_to_iter_based=True),
    # momentum scheduler
    # During the first 8 epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next 12 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type='CosineAnnealingMomentum',
        T_max=4,
        eta_min=0.85 / 0.95,
        begin=0,
        end=4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=26,
        eta_min=1,
        begin=4,
        end=30,
        by_epoch=True,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=30, val_interval=1)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(enable=False, base_batch_size=16)

# dataset settings
point_cloud_range = [-21.0, -21.0, -2.0, 21.0, 21.0, 6.0]
# class_names=[
#     #DynamicClasses
#     "Car", "Pedestrian", "Bike", "Motorcycle", "Golf Cart", #Unused
#     "Truck", #Unused
#     "Scooter",
#     #StaticClasses
#     "Tree", "Traffic Sign", "Canopy", "Traffic Light", "Bike Rack", "Bollard", "Construction Barrier", #Unused
#     "Parking Kiosk", "Mailbox", "Fire Hydrant",
#     #StaticClassMixed
#     "Freestanding Plant", "Pole", "Informational Sign", "Door", "Fence", "Railing", "Cone",
#     "Chair", "Bench", "Table", "Trash Can", "Newspaper Dispenser",
#     #StaticClassesIndoor
#     "Room Label", "Stanchion", "Sanitizer Dispenser", "Condiment Dispenser", "Vending Machine",
#     "Emergency Aid Kit", "Fire Extinguisher", "Computer", "Television", #unused
#     "Other", "Horse",
#     #NewClasses
#     "Pickup Truck", "Delivery Truck", "Service Vehicle", "Utility Vehicle", "Fire Alarm", "ATM", "Cart", "Couch", 
#     "Traffic Arm", "Wall Sign", "Floor Sign", "Door Switch", "Emergency Phone", "Dumpster", "Vacuum Cleaner", #unused
#     "Segway", "Bus", "Skateboard", "Water Fountain"
# ]
class_names = ['Car', 'Pedestrian', 'Cyclist']
metainfo = dict(classes=class_names)
dataset_type = 'CodaDataset'
data_root = 'data/CODA/'
input_modality = dict(use_lidar=True, use_camera=True)
data_prefix = dict(
    pts='',
    img='',
    sweeps='')
backend_args = None

# pipeline settings
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)
ida_aug_conf = {
        "resize_lim": (0.5, 0.625),
        "final_dim": (640, 800),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 1024,
        "W": 1224,
        "rand_flip": True,
    }
db_sampler=dict(
    type='UnifiedDataBaseSampler',
    data_root=data_root,
    info_path=data_root + 'coda_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            Car=5,
            Pedestrian=5,
            Cyclist=5)),
    classes=class_names,
    sample_groups=dict(
            Car=3,
            Pedestrian=2,
            Cyclist=4),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=[0, 1, 2, 3]))

# pipeline
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=[0, 1, 2, 3],
    ),
    dict(
        type='LoadMultiViewImageFromFilesNus',
        to_float32=True,
        color_type='color',
        num_views=2,
        backend_args=backend_args
    ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # dict(
    #     type='UnifiedObjectSample',
    #     sample_2d=True,
    #     mixup_rate=0.5,
    #     db_sampler=db_sampler
    # ),
    # dict(type='ModalMask3D', mode='train'),
    dict(
        type='GlobalRotScaleTransAll',
        scale_ratio_range=[0.9, 1.1],
        rot_range=[-0.78539816, 0.78539816],
        translation_std=[0.5, 0.5, 0.5]),
    dict(
        type='CustomRandomFlip3D',
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='Pack3DDetInputs', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'],
         meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                    'depth2img', 'cam2img', 'pad_shape',
                    'scale_factor', 'flip', 'pcd_horizontal_flip',
                    'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                    'img_norm_cfg', 'pcd_trans', 'sample_idx',
                    'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                    'transformation_3d_flow', 'rot_degree',
                    'gt_bboxes_3d', 'gt_labels_3d'))
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=[0, 1, 2, 3],
    ),
    dict(
        type='LoadMultiViewImageFromFilesNus',
        to_float32=True,
        color_type='color',
        num_views=2,
        backend_args=backend_args
    ),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=False),
            dict(type='NormalizeMultiviewImage', **img_norm_cfg),
            dict(type='PadMultiViewImage', size_divisor=32)
        ]),
    dict(type='Pack3DDetInputs', keys=['points', 'img'])
]

# dataloader
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='coda_infos_train.pkl',
            pipeline=train_pipeline,
            metainfo=metainfo,
            modality=input_modality,
            test_mode=False,
            data_prefix=data_prefix,
            use_valid_flag=True,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR')))
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='coda_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=True,
        data_prefix=data_prefix,
        box_type_3d='LiDAR',
        backend_args=backend_args))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='coda_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        data_prefix=data_prefix,
        test_mode=True,
        box_type_3d='LiDAR',
        backend_args=backend_args))

# evaluator
val_evaluator = dict(
    type='CodaMetric',
    ann_file=data_root + 'coda_infos_val.pkl',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator

# visualization settings
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# model
voxel_size = [0.075, 0.075, 0.2]
out_size_factor = 8
model = dict(
    type='CmdtDetector',
    use_grid_mask=True,
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        bgr_to_rgb=False),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CPFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=2),
    pts_voxel_layer=dict(
        num_point_features=4,
        max_num_points=10,
        voxel_size=voxel_size,
        max_voxels=(120000, 160000),
        point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(
        type='HardSimpleVFE',
        num_features=4),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,
        sparse_shape=[41, 560, 560],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='CmdtHead',
        num_query=500,
        in_channels=512,
        hidden_dim=256,
        downsample_scale=8,
        common_heads=dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        task=dict(num_class=len(class_names), class_names=class_names),
        bbox_coder=dict(
            type='NMSFreeBBoxCoder',
            post_center_range=[-30.0, -30.0, -5.0, 30.0, 30.0, 10.0],
            pc_range=point_cloud_range,
            max_num=100,
            nms_radius=None,
            score_threshold=0.1,
            voxel_size=voxel_size,
            num_classes=len(class_names)), 
        separate_head=dict(
            type='SeparateTaskHead', init_bias=-2.19, final_kernel=1),
        transformer=dict(
            type='CmdtTransformer',
            decoder=dict(
                type='CmdtTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='CmdtTransformerDecoderLayer',
                    with_cp=False,
                    batch_first=True,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='DeformableAttention2MultiModality',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        loss_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25, reduction='mean', loss_weight=2.0),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        loss_heatmap=dict(type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
    ),
    train_cfg=dict(
        pts=dict(
            assigner=dict(
                type='HungarianAssigner3D',
                # cls_cost=dict(type='ClassificationCost', weight=2.0),
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
                pc_range=point_cloud_range,
                code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=[560, 560, 40],  # [x_len, y_len, 1]
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            point_cloud_range=point_cloud_range)))

# hooks
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1),
    changestrategy=dict(
        type='ChangeStrategyHook',
        change_epoch=[16, 26, -1],
        change_strategy=['remove_GTSample', 'remove_DN', 'change_layers_loss_weight'],
        change_args=[None, None, None])
    )

load_from='ckpts/pretrain/nuim_r50.pth'

# resume = True