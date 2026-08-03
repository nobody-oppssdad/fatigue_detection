import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
# 新增：导入NMS工具
import torchvision.ops as ops

from model.faster_rcnn import get_faster_rcnn
from utils.dataset import VOCDataset
from utils.transforms import get_val_transform
from utils.utils import collate_fn, get_classes


def get_val_image_paths(data_root):
    """从VOC的ImageSets中读取val.txt，获取真实图像路径列表"""
    img_sets_path = os.path.join(data_root, "ImageSets", "Main", "val.txt")
    if not os.path.exists(img_sets_path):
        raise FileNotFoundError(f"未找到val.txt：{img_sets_path}，请检查数据集结构")

    # 读取val.txt中的图像ID
    with open(img_sets_path, "r") as f:
        img_ids = [line.strip().split()[0] for line in f if line.strip()]

    # 拼接图像路径（JPEGImages目录下）
    img_paths = []
    for img_id in img_ids:
        img_path = os.path.join(data_root, "JPEGImages", f"{img_id}.jpg")
        # 兼容png格式
        if not os.path.exists(img_path):
            img_path = os.path.join(data_root, "JPEGImages", f"{img_id}.png")
        img_paths.append(img_path)

    return img_paths


def main(args):
    # ====================== 新增：手动排查单张图像预测 ======================
    print("🔍 开始排查单张图像预测结果...")
    # 1. 加载类别和模型（复用评估脚本的配置，避免参数不一致）
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    classes = [cls.strip() for cls in get_classes(args.data_root)]
    num_classes = len(classes) + 1  # 背景+目标类别

    # 2. 加载模型（和评估用同一个权重）
    model = get_faster_rcnn(num_classes).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 3. 加载单张验证集图像（取第一个样本）
    val_dataset = VOCDataset(
        root=args.data_root,
        split="val",
        transforms=get_val_transform()
    )
    # 取验证集第一张图
    img, target = val_dataset[0]
    img_batch = img.unsqueeze(0).to(device)  # 转成batch维度

    # 4. 模型预测并打印关键信息
    with torch.no_grad():
        pred = model(img_batch)[0]  # 取batch中第一个结果

        # ========== 仅新增：单张图像预测后添加NMS+置信度过滤 ==========
        # 过滤低置信度框
        high_conf_idx = pred['scores'] >= args.conf_thresh
        boxes = pred['boxes'][high_conf_idx]
        scores = pred['scores'][high_conf_idx]
        labels = pred['labels'][high_conf_idx]

        # 应用NMS（非极大值抑制）合并重复框
        if len(boxes) > 0:
            keep_idx = ops.nms(boxes, scores, iou_threshold=0.3)
            pred['boxes'] = boxes[keep_idx]
            pred['scores'] = scores[keep_idx]
            pred['labels'] = labels[keep_idx]
        # ========== NMS处理结束 ==========

    # 打印排查信息
    print("\n📌 单张图像预测排查结果：")
    print(f"   类别列表（训练/评估用）：{classes}")
    print(f"   类别ID映射：{dict(enumerate(classes, start=1))}")  # 标注/模型的类别ID从1开始
    print(f"   预测框数量：{len(pred['boxes'])}")
    if len(pred['boxes']) > 0:
        print(f"   预测置信度范围：{pred['scores'].min().cpu().item():.6f} ~ {pred['scores'].max().cpu().item():.6f}")
        print(f"   预测类别ID：{pred['labels'].cpu().numpy()}")
        print(f"   第一个预测框坐标：{pred['boxes'][0].cpu().numpy()}")
    else:
        print("   ❌ 模型未输出任何预测框！")
    print(f"   标注类别ID：{target['labels'].cpu().numpy()}")
    print(f"   标注框坐标：{target['boxes'].cpu().numpy()}")
    print("=" * 60 + "\n")
    # ====================== 排查代码结束 ======================

    # 1. 设备和类别配置（原代码保留）
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    classes = [cls.strip() for cls in get_classes(args.data_root)]
    num_classes = len(classes) + 1

    # 2. 获取验证集真实图像路径（从ImageSets中读取，避免硬编码）
    val_img_paths = get_val_image_paths(args.data_root)
    # 过滤不存在的图像（自动跳过）
    valid_img_paths = [p for p in val_img_paths if os.path.exists(p)]
    if len(valid_img_paths) < len(val_img_paths):
        print(f"⚠️  跳过 {len(val_img_paths) - len(valid_img_paths)} 张不存在的图像")

    # 3. 加载验证集（只加载存在的图像）
    # 注意：这里假设VOCDataset支持通过img_ids初始化，若不支持可修改为自定义数据集
    val_dataset = VOCDataset(
        root=args.data_root,
        split="val",
        transforms=get_val_transform()
    )
    # 过滤数据集（只保留存在的图像）
    valid_indices = [i for i, p in enumerate(val_img_paths) if os.path.exists(p)]
    val_dataset = torch.utils.data.Subset(val_dataset, valid_indices)

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # 替换原代码中“4. 生成COCO真实标注（基于真实图像路径）”的全部内容
    # 4. 生成COCO真实标注（包含标注框）
    coco_gt = {"images": [], "annotations": [], "categories": []}
    # 类别
    for idx, cls in enumerate(classes):
        coco_gt["categories"].append({"id": idx + 1, "name": cls, "supercategory": "none"})

    # 图像和标注（关键：从VOCDataset中读取真实标注框）
    anno_id = 0  # 标注ID自增
    for img_idx, (img_path, dataset_idx) in enumerate(zip(valid_img_paths, valid_indices)):
        # 读取图像信息
        img_name = os.path.basename(img_path)
        with Image.open(img_path) as img:
            w, h = img.size
        # 添加图像信息
        coco_gt["images"].append({
            "id": img_idx,
            "file_name": img_name,
            "width": w,
            "height": h
        })

        # 从数据集获取该图像的标注框（核心补全部分）
        _, target = val_dataset.dataset[dataset_idx]  # val_dataset是Subset，需通过dataset获取原数据
        boxes = target["boxes"].cpu().numpy()  # 标注框坐标 (x1,y1,x2,y2)
        labels = target["labels"].cpu().numpy()  # 标注类别ID

        # 转换为COCO格式的标注（x1,y1,w,h）
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            # 过滤无效框（宽/高<=0）
            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                continue
            coco_gt["annotations"].append({
                "id": anno_id,
                "image_id": img_idx,
                "category_id": int(label),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],  # COCO格式：x,y,w,h
                "area": float((x2 - x1) * (y2 - y1)),
                "iscrowd": 0
            })
            anno_id += 1

    # 保存真实标注
    gt_file = "gt.json"
    with open(gt_file, "w", encoding="utf-8") as f:
        json.dump(coco_gt, f)
    coco_gt_api = COCO(gt_file)

    # 5. 模型预测
    model = get_faster_rcnn(num_classes).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    coco_preds = []
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(val_loader, desc="评估中")):
            current_img_id = batch_idx  # 与真实标注ID对应
            images = list(img.to(device) for img in images)
            preds = model(images)[0]

            # ========== 仅新增：批量预测后添加NMS+置信度过滤 ==========
            # 过滤低置信度
            mask = preds["scores"] >= args.conf_thresh
            boxes = preds["boxes"][mask]
            scores = preds["scores"][mask]
            labels = preds["labels"][mask]

            # 应用NMS合并重复框
            if len(boxes) > 0:
                keep_idx = ops.nms(boxes, scores, iou_threshold=0.3)
                boxes = boxes[keep_idx]
                scores = scores[keep_idx]
                labels = labels[keep_idx]

            # 转回numpy（保持原代码逻辑）
            boxes = boxes.cpu().numpy() if len(boxes) > 0 else np.array([])
            scores = scores.cpu().numpy() if len(scores) > 0 else np.array([])
            labels = labels.cpu().numpy() if len(labels) > 0 else np.array([])
            # ========== NMS处理结束 ==========

            # 处理坐标（原代码逻辑保留）
            img_w = coco_gt["images"][current_img_id]["width"]
            img_h = coco_gt["images"][current_img_id]["height"]
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = np.clip(box, 0, max(img_w, img_h))
                w, h = x2 - x1, y2 - y1
                if w > 1 and h > 1:
                    coco_preds.append({
                        "image_id": current_img_id,
                        "category_id": int(label),
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "score": float(score)
                    })

    # 6. 保存并评估
    pred_file = "pred.json"
    with open(pred_file, "w", encoding="utf-8") as f:
        json.dump(coco_preds, f)

    if not coco_preds:
        print("❌ 无有效预测结果！建议按默认配置训练30轮")
        return

    coco_dt_api = coco_gt_api.loadRes(pred_file)
    coco_eval = COCOeval(coco_gt_api, coco_dt_api, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    print("\n" + "=" * 50)
    print("评估结果（COCO标准指标）")
    print("=" * 50)
    coco_eval.summarize()

    # 清理临时文件
    os.remove(gt_file)
    os.remove(pred_file)
    print("\n✅ 评估完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动适配图像路径的评估脚本")
    parser.add_argument("--checkpoint", required=True, help="模型权重路径")
    parser.add_argument("--data_root", default="data", help="数据集根目录")
    parser.add_argument("--device", default="cuda", help="设备（cuda/cpu）")
    parser.add_argument("--conf_thresh", type=float, default=0.5, help="置信度阈值")
    args = parser.parse_args()
    main(args)