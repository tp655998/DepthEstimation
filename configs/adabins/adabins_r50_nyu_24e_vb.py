_base_ = [
    '../_base_/models/adabins.py', '../_base_/datasets/nyu.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_24x.py'
]

norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    backbone=dict(
        _delete_=True,
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3, 4),
        style='pytorch',
        norm_cfg=norm_cfg,
        init_cfg=dict(
            type='Pretrained', checkpoint='torchvision://resnet50'),),
    decode_head=dict(
        in_channels=[64, 256, 512, 1024, 2048],
        up_sample_channels=[128, 256, 512, 1024, 2048],
        channels=128, # last one
        min_depth=1e-3,
        max_depth=10,
        norm_cfg=norm_cfg),
    )

# batch size==================================
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=4,
)


# runtime settings
#runner = dict(type='IterBasedRunner', max_iters=1600 * 24) #epoch=======================================
runner = dict(type='EpochBasedRunner', max_epochs=150) #epoch=======================================

# runtime
evaluation = dict(interval=5,                  
                  rule='less', 
                #   rule='greater', 
                #   save_best='rmse',
                #   save_best='a1',
                  save_best='abs_rel',
                  greater_keys=("a1", "a2", "a3"), 
                  less_keys=("abs_rel", "rmse", "log_10", "rmse_log", "silog", "sq_rel"))


checkpoint_config = dict(by_epoch=True, max_keep_ckpts=1, interval=5)