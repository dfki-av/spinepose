_base_ = ["mmpose::_base_/default_runtime.py"]

# runtime
train_cfg = None
randomness = dict(seed=21)
include_coco = False
include_halpe = False

# val datasets
data_mode = "topdown"
data_root = "data/"
datasets=[
    # SpineTrack
    dict(
        type="SpineTrackDataset",
        data_root=data_root,
        data_mode=data_mode,
        ann_file="spinetrack/annotations/person_keypoints_val2017.json",
        data_prefix=dict(img="spinetrack/images/val2017/"),
        test_mode=True,
    ),
]
if include_coco:
    datasets.append(
        # COCO
        dict(
            type="CocoDataset",
            data_root=data_root,
            data_mode=data_mode,
            ann_file="coco/annotations/person_keypoints_val2017.json",
            bbox_file=f"{data_root}coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json",
            data_prefix=dict(img="coco/val2017/"),
            test_mode=True,
        )
    )
if include_halpe:
    datasets.append(
        # Halpe26
        dict(
            type="HalpeDataset",
            data_root=data_root,
            data_mode=data_mode,
            ann_file="halpe/annotations/halpe_val_v1.json",
            bbox_file=f"{data_root}coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json",
            data_prefix=dict(img="coco/val2017/"),
            test_mode=True,
        )
    )
val_dataset = dict(
    type="CombinedDataset",
    metainfo=dict(from_file="configs/_base_/datasets/spinetrack.py"),
    datasets=datasets,
    test_mode=True,
)

# data loaders
batch_size = 64
num_workers = 32
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=val_dataset,
)
test_dataloader = val_dataloader

# evaluators
num_keypoints = 37
metrics = [
    dict(
        type="SpineTrackMetric",
        prefix="spinetrack",
        ann_file=f"{data_root}spinetrack/annotations/person_keypoints_val2017.json",
    )
]
if include_coco:
    metrics.append(
        dict(
            type="CocoMetric",
            prefix="coco",
            ann_file=f"{data_root}coco/annotations/person_keypoints_val2017.json",
            pred_converter=dict(
                type="KeypointConverter",
                num_keypoints=17,
                mapping=[(i, i) for i in range(17)],
            ),
        )
    )
if include_halpe:
    metrics.append(
        dict(
            type="CocoMetric",
            prefix="halpe",
            ann_file=f"{data_root}halpe/annotations/halpe_val_v1.json",
            gt_converter=dict(
                type="KeypointConverter",
                num_keypoints=26,
                mapping=[(i, i) for i in range(26)],
            ),
            pred_converter=dict(
                type="KeypointConverter",
                num_keypoints=26,
                mapping=[(i, i) for i in range(26)],
            ),
        )
    )
val_evaluator = dict(
    type="MultiDatasetEvaluator",
    metrics=metrics,
    datasets=datasets,
)
test_evaluator = val_evaluator
