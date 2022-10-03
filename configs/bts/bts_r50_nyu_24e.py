_base_ = [
    '../_base_/models/bts.py', '../_base_/datasets/nyu.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_24x.py'
]

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    decode_head=dict(
        final_norm=False,
        min_depth=1e-3,
        max_depth=10,
    ))
    
# batch size==================================
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=4,
)

    
runner = dict(type='EpochBasedRunner', max_epochs=150) #epoch=======================================

# runtime
evaluation = dict(interval=5)

checkpoint_config = dict(by_epoch=True, max_keep_ckpts=1, interval=10)