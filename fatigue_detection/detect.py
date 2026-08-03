import os
import json
import torch
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from model.faster_rcnn import get_faster_rcnn
from utils.dataset import VOCDataset
from utils.transforms import get_val_transform
from utils.utils import collate_fn, get_classes


# -------------------------- 核心修复：自定义COCO类（全UTF-8支持） --------------------------
class MyCOCO(COCO):
    """重写COCO类，强制所有JSON操作使用UTF-8编码"""

    def __init__(self, annotation_file=None):
        self.dataset, self.anns, self.cats, self.imgs = {}, {}, {}, {}
        self.imgToAnns, self.catToImgs = {}, {}
        self.dataset_name = None
        if annotation_file is not None:
            # 强制用UTF-8读取标注文件
            with open(annotation_file, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
            self.dataset = dataset
            self.createIndex()

    def loadRes(self, resFile):
        """重写loadRes方法，强制UTF-8读取预测结果文件"""
        res = COCO()
        res.dataset['images'] = [img for img in self.dataset['images']]

        # 强制用UTF-8读取预测结果
        with open(resFile, 'r', encoding='utf-8') as f:
            anns = json.load(f)

        assert type(anns) == list, 'results in not an array of objects'
        annsImgIds = [ann['image_id'] for ann in anns]
        assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
            'Results do not correspond to current coco set'

        if 'caption' in anns[0]:
            imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
            res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
            for id, ann in enumerate(anns):
                ann['id'] = id + 1
        elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
            res.dataset['categories'] = self.dataset['categories']
            for id, ann in enumerate(anns):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
                ann['area'] = bb[2] * bb[3]
                ann['id'] = id + 1
                ann['iscrowd'] = 0
        elif 'segmentation' in anns[0]:
            res.dataset['categories'] = self.dataset['categories']
            for id, ann in enumerate(anns):
                ann['area'] = self.annArea(ann)
                ann['id'] = id + 1
                ann['iscrowd'] = 0
        res.dataset['annotations'] = anns
        res.createIndex()
        return res


# -------------------------- 疲劳等级判定函数（全中文） --------------------------
def judge_fatigue_level(eye_status, mouth_status):
    """
    根据眼睛和嘴巴状态判定疲劳等级（全中文）
    :param eye_status: 眼睛状态（闭眼/睁眼）
    :param mouth_status: 嘴巴状态（张嘴/闭嘴）
    :return: 疲劳等级名称、框颜色（RGB）
    """
    # 极度疲劳：闭眼+张嘴 → 红色框
    if eye_status == "闭眼" and mouth_status == "张嘴":
        return "极度疲劳", (255, 0, 0)
    # 疲劳：闭眼+闭嘴 → 橙色框
    elif eye_status == "闭眼" and mouth_status == "闭嘴":
        return "疲劳", (255, 165, 0)
    # 轻度疲劳：睁眼+张嘴 → 蓝色框
    elif eye_status == "睁眼" and mouth_status == "张嘴":
        return "轻度疲劳", (0, 0, 255)
    # 正常：睁眼+闭嘴 → 绿色框
    elif eye_status == "睁眼" and mouth_status == "闭嘴":
        return "正常", (0, 255, 0)
    # 未知状态 → 灰色框
    else:
        return "未知", (128, 128, 128)


def get_head_bbox(preds):
    """
    根据单张图像的所有预测框（眼睛/嘴巴）计算人头外接框
    :param preds: 单张图像的预测框列表 [{"bbox": [x1,y1,w,h], "category_id": xxx}, ...]
    :return: 人头框坐标 (x1, y1, x2, y2)，无有效框返回None
    """
    all_boxes = []
    for pred in preds:
        x, y, w, h = pred["bbox"]
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        all_boxes.append([x1, y1, x2, y2])

    if len(all_boxes) == 0:
        return None

    # 计算所有框的外接矩形
    all_x1 = min([box[0] for box in all_boxes])
    all_y1 = min([box[1] for box in all_boxes])
    all_x2 = max([box[2] for box in all_boxes])
    all_y2 = max([box[3] for box in all_boxes])

    # 扩大1.5倍（更贴合人头范围）
    expand_ratio = 1.5
    w = all_x2 - all_x1
    h = all_y2 - all_y1
    center_x = (all_x1 + all_x2) / 2
    center_y = (all_y1 + all_y2) / 2
    new_w = w * expand_ratio
    new_h = h * expand_ratio

    head_x1 = max(0, center_x - new_w / 2)
    head_y1 = max(0, center_y - new_h / 2)
    head_x2 = center_x + new_w / 2
    head_y2 = center_y + new_h / 2

    return (head_x1, head_y1, head_x2, head_y2)


def create_coco_gt(voc_dataset, save_path):
    """生成COCO格式真实标注（修复索引错误+UTF-8编码）"""
    coco_gt = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 加载类别（全中文映射）
    classes = get_classes(voc_dataset.root)
    # 将英文类别映射为中文（需与训练时的类别名称对应）
    cls_en_to_cn = {
        "closed_eye": "闭眼",
        "open_eye": "睁眼",
        "closed_mouth": "闭嘴",
        "open_mouth": "张嘴"
    }
    classes_cn = [cls_en_to_cn[cls.strip()] for cls in classes]

    for idx, cls in enumerate(classes_cn):
        coco_gt["categories"].append({
            "id": idx + 1,  # 1-based ID，与训练标签一致
            "name": cls,
            "supercategory": "无"
        })

    # 图像信息：按数据集顺序分配连续ID（0开始）
    for img_idx, img_path in enumerate(voc_dataset.images):
        img = Image.open(img_path)
        width, height = img.size
        coco_gt["images"].append({
            "id": img_idx,
            "file_name": os.path.basename(img_path),
            "width": width,
            "height": height
        })

    # 标注信息：修复索引错误（核心！）
    ann_id = 0
    for img_idx, target in enumerate(voc_dataset.targets):
        # 遍历当前图像的所有标注框（原代码错误地取了[0]）
        boxes = target["boxes"].cpu().numpy()  # 所有框的数组 (N,4)
        labels = target["labels"].cpu().numpy()  # 所有类别的数组 (N,)

        for box, cls_id in zip(boxes, labels):
            x1, y1, x2, y2 = box  # 单框坐标
            area = (x2 - x1) * (y2 - y1)

            coco_gt["annotations"].append({
                "id": ann_id,
                "image_id": img_idx,
                "category_id": int(cls_id),  # 1-based类别ID
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],  # COCO格式：x,y,w,h
                "area": float(area),
                "iscrowd": 0,
                "segmentation": []
            })
            ann_id += 1

    # 保存真实标注（强制UTF-8编码）
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(coco_gt, f, ensure_ascii=False, indent=2)
    print(f"✅ 真实标注已保存：{save_path}（UTF-8编码）")
    return coco_gt


def generate_coco_preds(model, val_loader, val_dataset, conf_thresh, device, save_path):
    model.eval()
    coco_preds = []

    # 英文类别转中文
    classes = get_classes(val_dataset.root)
    cls_en_to_cn = {
        "closed_eye": "闭眼",
        "open_eye": "睁眼",
        "closed_mouth": "闭嘴",
        "open_mouth": "张嘴"
    }
    classes_cn = [cls_en_to_cn[cls.strip()] for cls in classes]
    cls_id_to_cn = {idx + 1: cls for idx, cls in enumerate(classes_cn)}  # 1-based ID映射

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(val_loader, desc="生成预测结果")):
            current_img_id = batch_idx
            images = list(img.to(device) for img in images)

            # 模型预测
            predictions = model(images)[0]
            if len(predictions["scores"]) == 0:
                continue

            mask = predictions["scores"] >= conf_thresh
            boxes = predictions["boxes"][mask].cpu().numpy()
            scores = predictions["scores"][mask].cpu().numpy()
            labels = predictions["labels"][mask].cpu().numpy()

            # 获取原始图像尺寸（计算反缩放比例）
            img_path = val_dataset.images[batch_idx]
            img_ori = Image.open(img_path)
            ori_w, ori_h = img_ori.size  # 原始尺寸 (宽, 高)

            # ====================== 【唯一修改：正确的坐标反缩放逻辑】 ======================
            # 动态获取模型输入尺寸，替代硬编码
            _, input_h, input_w = images[0].shape
            # 计算等比例缩放比例 + padding
            scale = min(input_w / ori_w, input_h / ori_h)
            pad_w = (input_w - ori_w * scale) / 2
            pad_h = (input_h - ori_h * scale) / 2

            # 反缩放预测框到原始尺寸（修正原错误的直接乘比例）
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                # 先减去padding，再除以缩放比例，还原到原图坐标
                x1 = (x1 - pad_w) / scale
                y1 = (y1 - pad_h) / scale
                x2 = (x2 - pad_w) / scale
                y2 = (y2 - pad_h) / scale

                # 限制坐标在原始图像范围内
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(ori_w, x2)
                y2 = min(ori_h, y2)
                w = x2 - x1
                h = y2 - y1

                if w > 1 and h > 1:
                    coco_preds.append({
                        "image_id": current_img_id,
                        "category_id": int(label),
                        "category_name": cls_id_to_cn.get(int(label), "未知"),  # 新增中文类别名
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "score": float(score)
                    })

    # 保存预测结果（强制UTF-8编码）
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(coco_preds, f, ensure_ascii=False, indent=2)
    print(f"✅ 预测结果已保存：{save_path}（UTF-8编码）")
    return coco_preds


# -------------------------- 可视化函数（全中文标签） --------------------------
def visualize_predictions(val_dataset, coco_preds, conf_thresh, save_dir):
    """可视化预测结果（全中文标签+疲劳等级，标签放识别框下方）"""
    os.makedirs(save_dir, exist_ok=True)
    # 英文类别转中文
    classes = get_classes(val_dataset.root)
    cls_en_to_cn = {
        "closed_eye": "闭眼",
        "open_eye": "睁眼",
        "closed_mouth": "闭嘴",
        "open_mouth": "张嘴"
    }
    classes_cn = [cls_en_to_cn[cls.strip()] for cls in classes]
    cls_id_to_cn = {idx + 1: cls for idx, cls in enumerate(classes_cn)}  # 1-based ID映射

    # 建立image_id到预测框的映射
    img_id_to_preds = {}
    for pred in coco_preds:
        img_id = pred["image_id"]
        if img_id not in img_id_to_preds:
            img_id_to_preds[img_id] = []
        img_id_to_preds[img_id].append(pred)

    # 逐图绘制（全中文）
    for img_idx, img_path in enumerate(tqdm(val_dataset.images, desc="可视化预测结果（全中文）")):
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        try:
            # 优先使用中文字体（如系统自带的微软雅黑）
            font = ImageFont.truetype("msyh.ttc", 14)
        except:
            #  fallback到默认字体
            font = ImageFont.load_default()

        # 获取当前图像的预测框
        preds = img_id_to_preds.get(img_idx, [])

        # 1. 判定眼睛/嘴巴状态（中文）
        eye_status = ""
        mouth_status = ""
        for pred in preds:
            cls_name = pred.get("category_name", cls_id_to_cn.get(pred["category_id"], ""))
            # 优先判定眼睛状态（有闭眼则为闭眼，否则睁眼）
            if cls_name == "闭眼":
                eye_status = "闭眼"
            elif cls_name == "睁眼" and eye_status == "":
                eye_status = "睁眼"
            # 优先判定嘴巴状态（有张嘴则为张嘴，否则闭嘴）
            if cls_name == "张嘴":
                mouth_status = "张嘴"
            elif cls_name == "闭嘴" and mouth_status == "":
                mouth_status = "闭嘴"

        # 2. 绘制原始目标框（眼睛/嘴巴，白色细框+中文标签，标签放框下方）
        if not preds:
            draw.text((10, 10), f"无预测结果（置信度≥{conf_thresh}）", fill="red", font=font)
        else:
            for pred in preds:
                x, y, w, h = pred["bbox"]
                x1, y1, x2, y2 = x, y, x + w, y + h
                cls_name = pred.get("category_name", cls_id_to_cn.get(pred["category_id"], "未知"))
                conf = pred["score"]

                # 绘制小目标框（白色，细框）
                draw.rectangle([(x1, y1), (x2, y2)], outline="white", width=1)
                # 绘制中文小目标标签（放到框的下方）
                label = f"{cls_name}：{conf:.3f}"
                text_bbox = draw.textbbox((x1, y2), label, font=font)  # 标签起点设为框的右下角y2
                draw.rectangle(text_bbox, fill="black")
                draw.text((x1, y2), label, fill="white", font=font)  # 文字显示在框的下方

        # 3. 计算人头框并绘制（按疲劳等级上色+中文标签，标签放框下方）
        head_bbox = get_head_bbox(preds)
        if head_bbox and eye_status and mouth_status:
            # 判定疲劳等级（中文）
            level_name, color = judge_fatigue_level(eye_status, mouth_status)
            x1, y1, x2, y2 = head_bbox

            # 绘制人头框（粗框，醒目）
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=4)
            # 绘制中文疲劳等级标签（放到框的下方）
            level_label = f"{level_name}（{eye_status}+{mouth_status}）"
            label_x = x1
            label_y = y2  # 标签起点设为框的下方y2
            # 避免标签超出图像底部
            if label_y + 20 > img.size[1]:
                label_y = y1 - 25
            text_bbox = draw.textbbox((label_x, label_y), level_label, font=font)
            draw.rectangle(text_bbox, fill="black")
            draw.text((label_x, label_y), level_label, fill=color, font=font)

        # 保存可视化结果
        save_path = os.path.join(save_dir, os.path.basename(img_path))
        img.save(save_path)

    print(f"✅ 可视化结果（全中文，标签放框下方）已保存：{save_dir}")


def evaluate(args):
    # 1. 设备配置（与训练一致）
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"使用设备: {device}")

    # 2. 加载数据集和类别（与训练完全一致）
    val_dataset = VOCDataset(
        root=args.data_root,
        split="val",
        transforms=get_val_transform()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # 单批次单图，确保ID对应
        shuffle=False,  # 不打乱顺序，保证image_id一致
        num_workers=0,  # 禁用多线程，避免顺序混乱
        collate_fn=collate_fn
    )
    classes = get_classes(val_dataset.root)
    # 打印中文类别映射
    cls_en_to_cn = {
        "closed_eye": "闭眼",
        "open_eye": "睁眼",
        "closed_mouth": "闭嘴",
        "open_mouth": "张嘴"
    }
    classes_cn = [cls_en_to_cn[cls.strip()] for cls in classes]
    num_classes = len(classes) + 1
    print(f"评估类别：{classes_cn}（含背景共{num_classes}类）")

    # 3. 生成COCO格式真实标注
    coco_gt_path = "coco_gt.json"
    create_coco_gt(val_dataset, coco_gt_path)

    # 4. 加载COCO标注（使用自定义类，修复编码问题）
    coco_gt = MyCOCO(coco_gt_path)

    # 5. 加载模型（与训练用同一结构）
    model = get_faster_rcnn(num_classes).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"✅ 已加载模型权重：{args.checkpoint}")

    # 6. 生成COCO格式预测结果
    coco_pred_path = "coco_pred.json"
    coco_preds = generate_coco_preds(
        model=model,
        val_loader=val_loader,
        val_dataset=val_dataset,
        conf_thresh=args.conf_thresh,
        device=device,
        save_path=coco_pred_path
    )

    # 7. 检查预测结果有效性
    if not coco_preds:
        print("❌ 无有效预测结果！")
        visualize_predictions(val_dataset, coco_preds, args.conf_thresh, "prediction_vis")
        return

    # 8. 执行COCO评估（使用自定义类的loadRes方法）
    coco_dt = coco_gt.loadRes(coco_pred_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    print("\n" + "=" * 50)
    print("评估结果（COCO标准指标）")
    print("=" * 50)
    coco_eval.summarize()

    # 9. 可视化预测结果（全中文）
    visualize_predictions(val_dataset, coco_preds, args.conf_thresh, "prediction_vis")
    print(f"\n✅ 评估完成！可视化路径：prediction_vis")
    print("\n📋 疲劳等级标注规则：")
    print("   - 极度疲劳：闭眼+张嘴 → 红色粗框")
    print("   - 疲劳：闭眼+闭嘴 → 橙色粗框")
    print("   - 轻度疲劳：睁眼+张嘴 → 蓝色粗框")
    print("   - 正常：睁眼+闭嘴 → 绿色粗框")
    print("   - 未知：状态不全 → 灰色粗框")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="疲劳驾驶检测模型评估（全中文可视化）")
    parser.add_argument("--checkpoint", required=True, help="模型权重路径（如：checkpoints/best.pth）")
    parser.add_argument("--data_root", default="data", help="数据集根目录（与训练一致）")
    parser.add_argument("--device", default="cuda", help="评估设备（cuda/cpu）")
    parser.add_argument("--conf_thresh", type=float, default=0.001, help="置信度阈值（极低值捕捉所有预测）")
    args = parser.parse_args()
    evaluate(args)