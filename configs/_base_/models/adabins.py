# model settings
model = dict(
    type='DepthEncoderDecoder',
    backbone=dict(
        type='EfficientNet'),
    decode_head=dict(
        type='AdabinsHead',
        in_channels=[24, 40, 64, 176, 2048],
        up_sample_channels=[128, 256, 512, 1024, 2048],
        channels=128, # last one
        align_corners=True, # for upsample
        loss_decode=dict(
            # type='SigLoss', valid_mask=True, loss_weight=10.0, max_depth=10.0),
            type='SigLoss', valid_mask=True, loss_weight=10.0, max_depth=80.0),
        loss_vb=dict(
            # type='VisualBiasLoss', fx=518.85, fy=519.46, valid_mask=True, loss_weight=0.4, max_depth=10.0),
            type='VisualBiasLoss', fx=413.1615, fy=718.335, input_size=(352, 704), valid_mask=True, loss_weight=0.4, max_depth=80.0),
        loss_vnl=dict(
            # type='VNL_Loss', fx=518.85, fy=519.46, loss_weight=2.0)),
            type='VNL_Loss', fx=413.1615, fy=718.335, input_size=(352, 704), loss_weight=2.0, max_depth=80.0)),
            # type='VNL_Loss', fx=413.1615, loss_weight=4.0, max_depth=80.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# input_size=(352, 704),
        # results['cam_intrinsic'] = \
        #     [[5.1885790117450188e+02, 0, 3.2558244941119034e+02],
        #      [5.1946961112127485e+02, 0, 2.5373616633400465e+02],
        #      [0                     , 0, 1                    ]]

            # '2011_09_29' : [[7.183351e+02, 0.000000e+00, 6.003891e+02, 4.450382e+01], 
            #                 [0.000000e+00, 7.183351e+02, 1.815122e+02, -5.951107e-01],
            #                 [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.616315e-03]],