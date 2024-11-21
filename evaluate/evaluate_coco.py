import json
import torch
import re
import tqdm
import ipdb
import os
import ast
import pandas as pd
from mmdet.evaluation import CocoMetric
from mmengine.fileio import dump
# import warnings
# warnings.simplefilter("error", SyntaxWarning)

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_eval_results(eval_results, output_path):
    csv_output_path = f"{output_path}.csv"
    df = pd.DataFrame(list(eval_results.items()), columns=['Metric', 'Value'])
    df.to_csv(csv_output_path, index=False)
    print(f"评估结果已保存到: {csv_output_path}")

def check_bbox_format(bbox):
    return all(isinstance(coord, int) for coord in bbox)

def parse_predictions(json_data, categories):
    results = []
    for entry in tqdm.tqdm(json_data):
        image_id = int(entry['image'].split('.')[0])
        width, height = entry['width'], entry['height']
        preds = []
        for result in entry['result']:
            label, bboxes = extract_data(result)
            if label is not None and bboxes is not None:
                try:
                    label_id = categories.index(label)
                except:
                    # print(f"fail to parse label: {label}")
                    continue
                for bbox in bboxes:
                    try:
                        x1, y1, x2, y2 = bbox
                    except Exception as e:
                        # print(f"An error occurred while parsing bbox: {e}, original bbox info: {bbox}")
                        continue
                    if not check_bbox_format(bbox):
                        continue
                    x1 = x1 / 1000 * width
                    y1 = y1 / 1000 * height
                    x2 = x2 / 1000 * width
                    y2 = y2 / 1000 * height
                    preds.append([x1, y1, x2, y2, 1.0, label_id])
        pred_dict = {
            'bboxes': torch.tensor([p[:4] for p in preds]),
            'scores': torch.tensor([p[4] for p in preds]),
            'labels': torch.tensor([p[5] for p in preds])
        }
        results.append({
            'pred_instances': pred_dict,
            'img_id': image_id,
            'ori_shape': (height, width)
        })
    return results

def extract_data(sample_string):
    ref_pattern = r'<ref>(.*?)</ref>'
    bbox_pattern = r'<box>(.*?)</box>'
    ref_match = re.search(ref_pattern, sample_string)
    bbox_match = re.search(bbox_pattern, sample_string)

    if ref_match and bbox_match:
        label = ref_match.group(1).strip()
        try:
            bbox = ast.literal_eval(bbox_match.group(1).strip())
        except Exception as e:
            # print(f"An unexpected error occurred while evaluating bbox: {e}")
            return None, None
        return label, bbox

    return None, None

def main(infer_file, ann_file, output_file):
    json_data = load_json(infer_file)

    coco_annotations = load_json(ann_file)
    categories = [cat['name'] for cat in coco_annotations['categories']]

    predictions = parse_predictions(json_data, categories)

    coco_metric = CocoMetric(
        ann_file=ann_file,
        metric='bbox',
        classwise=True,
        outfile_prefix='eval_output'
    )
    coco_metric.dataset_meta = dict(classes=categories)

    coco_metric.process({}, predictions)

    eval_results = coco_metric.evaluate(size=len(predictions))
    print(eval_results)

    save_eval_results(eval_results, output_file)

if __name__ == "__main__":
    infer_file = '/mnt/afs/dailinjun/workdir/grounding/api_infer/test_result/d1119_qwen_0.5b_coco_1000hf.json'
    ann_file = '/mnt/afs2/kongxiangli/visonllm-v2/vllm_data/coco/annotations/instances_val2017.json'
    output_root = '/mnt/afs/dailinjun/workdir/grounding/api_infer/evaluate_result/'
    output_file = os.path.join(output_root, os.path.splitext(os.path.basename(infer_file))[0])

    main(infer_file, ann_file, output_file)
