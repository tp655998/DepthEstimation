# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='DepthEncoderDecoder',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3, 4),
        style='pytorch',
        norm_cfg=norm_cfg),
    decode_head=dict(
        type='BTSHead',
        scale_up=True,
        in_channels=[64, 256, 512, 1024, 2048],
        channels=32, # last one
        loss_decode=dict(
            type='SigLoss', valid_mask=True, loss_weight=10, max_depth=10.0),
        loss_vb=dict(
            type='VisualBiasLoss', fx=518.85, fy=519.46, valid_mask=True, loss_weight=0.2, max_depth=10.0),
        loss_vnl=dict(
            type='VNL_Loss', fx=518.85, fy=519.46, valid_mask=True, loss_weight=0.05)
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
