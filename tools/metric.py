import datetime
from typing import Dict, Optional, Sequence

import numpy as np
from mmengine.fileio import dump, load
from xtcocotools.cocoeval import COCOeval

from mmpose.registry import METRICS
from mmpose.evaluation.metrics.coco_metric import CocoMetric


@METRICS.register_module()
class SpineTrackMetric(CocoMetric):
    default_prefix: Optional[str] = "spinetrack"
    
    body_indices = list(range(17))
    feet_indices = list(range(20, 26))
    spine_indices = [36, 35, 18, 30, 29, 28, 27, 26, 19]

    def gt_to_coco_json(self, gt_dicts: Sequence[dict], outfile_prefix: str) -> str:
        images, annotations, img_ids = [], [], set()

        for gt_dict in gt_dicts:
            if gt_dict["img_id"] not in img_ids:
                images.append(dict(
                    id=gt_dict["img_id"],
                    width=gt_dict["width"],
                    height=gt_dict["height"]
                ))
                img_ids.add(gt_dict["img_id"])

            for ann in gt_dict["raw_ann_info"]:
                annotations.append(dict(
                    id=ann["id"],
                    image_id=ann["image_id"],
                    category_id=ann["category_id"],
                    bbox=ann["bbox"],
                    keypoints=ann["keypoints"],
                    iscrowd=ann["iscrowd"],
                    area=ann.get("area", 1.0)
                ))

        coco_json = dict(
            info=dict(
                date_created=str(datetime.datetime.now()),
                description="SpineTrack COCO-format JSON."
            ),
            images=images,
            annotations=annotations,
            categories=[{"supercategory": "person", "id": 1, "name": "person"}],
        )

        path = f"{outfile_prefix}.gt.json"
        dump(coco_json, path)
        return path

    def results2json(self, keypoints: Dict[int, list], outfile_prefix: str) -> str:
        results = []

        for img_kpts in keypoints.values():
            for instance in img_kpts:
                results.append(dict(
                    image_id=instance["img_id"],
                    category_id=instance["category_id"],
                    keypoints=instance["keypoints"].reshape(-1).tolist(),
                    score=float(instance["score"])
                ))

        path = f"{outfile_prefix}.keypoints.json"
        dump(results, path)
        return path

    def _evaluate_subset(self, indices, outfile_prefix, subset_name):
        gt_subset = self._subset_coco_annotations(indices)
        pred_subset = self._subset_coco_predictions(indices, outfile_prefix)

        coco_det = gt_subset.loadRes(pred_subset)
        sigmas = self.dataset_meta["sigmas"][indices]

        coco_eval = COCOeval(gt_subset, coco_det, "keypoints", sigmas, self.use_area)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        metrics_names = [
            "AP", "AP .5", "AP .75", "AP (M)", "AP (L)",
            "AR", "AR .5", "AR .75", "AR (M)", "AR (L)"
        ]

        return {f"{subset_name}_{name}": value for name, value in zip(metrics_names, coco_eval.stats)}

    def _subset_coco_annotations(self, indices):
        from xtcocotools.coco import COCO
        import copy

        subset_coco = copy.deepcopy(self.coco)
        for ann in subset_coco.dataset["annotations"]:
            kpts = np.array(ann["keypoints"]).reshape(-1, 3)[indices].flatten()
            ann["keypoints"] = kpts.tolist()
            ann["num_keypoints"] = int((kpts[2::3] > 0).sum())
        subset_coco.createIndex()
        return subset_coco

    def _subset_coco_predictions(self, indices, outfile_prefix):
        preds = load(f"{outfile_prefix}.keypoints.json")
        subset_preds = []

        for pred in preds:
            kpts = np.array(pred["keypoints"]).reshape(-1, 3)[indices].flatten()
            subset_preds.append(dict(
                image_id=pred["image_id"],
                category_id=pred["category_id"],
                keypoints=kpts.tolist(),
                score=pred["score"]
            ))

        subset_pred_path = f"{outfile_prefix}_{len(indices)}kpts.json"
        dump(subset_preds, subset_pred_path)
        return subset_pred_path

    def _do_python_keypoint_eval(self, outfile_prefix: str) -> list:
        eval_results = {}

        subsets = {
            "body": self.body_indices,
            "feet": self.feet_indices,
            "spine": self.spine_indices,
            "all": list(range(self.dataset_meta["num_keypoints"]))
        }

        for subset_name, indices in subsets.items():
            subset_result = self._evaluate_subset(indices, outfile_prefix, subset_name)
            eval_results.update(subset_result)

        return list(eval_results.items())
