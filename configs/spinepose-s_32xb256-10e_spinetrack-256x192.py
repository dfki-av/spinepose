_base_ = ["./data_config.py"]

# common setting
num_keypoints = 37
input_size = (192, 256)

# codec settings
codec = dict(
    type="SimCCLabel",
    input_size=input_size,
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False,
)

# model settings
model = dict(
    type="TopdownPoseEstimator",
    data_preprocessor=dict(
        type="PoseDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        _scope_="mmdet",
        type="CSPNeXt",
        arch="P5",
        expand_ratio=0.5,
        deepen_factor=0.33,
        widen_factor=0.5,
        out_indices=(4,),
        channel_attention=True,
        norm_cfg=dict(type="SyncBN"),
        act_cfg=dict(type="SiLU"),
    ),
    head=dict(
        type="RTMCCHead",
        in_channels=512,
        out_channels=num_keypoints,
        input_size=input_size,
        in_featuremap_size=tuple([s // 32 for s in input_size]),
        simcc_split_ratio=codec["simcc_split_ratio"],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.0,
            drop_path=0.0,
            act_fn="SiLU",
            use_rel_bias=False,
            pos_enc=False,
        ),
        loss=dict(
            type="KLDiscretLoss", use_target_weight=True, beta=10.0, label_softmax=True
        ),
        decoder=codec,
    ),
    test_cfg=dict(flip_test=True),
)

# pipelines
backend_args = dict(backend="local")
val_pipeline = [
    dict(type="LoadImage", backend_args=backend_args),
    dict(type="GetBBoxCenterScale"),
    dict(type="TopdownAffine", input_size=input_size),
    dict(type="PackPoseInputs"),
]

val_dataloader = dict(dataset=dict(pipeline=val_pipeline))
test_dataloader = val_dataloader
