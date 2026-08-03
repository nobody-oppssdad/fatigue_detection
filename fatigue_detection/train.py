import torch
import os
import torchvision
import argparse
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
from model.faster_rcnn import get_faster_rcnn
from utils.dataset import VOCDataset
from utils.transforms import get_train_transform, get_val_transform
from utils.utils import collate_fn, save_checkpoint, get_classes
import json
import numpy as np
from collections import defaultdict

# 确保checkpoints目录存在
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('loss_logs', exist_ok=True)
os.makedirs('metric_logs', exist_ok=True)  # 新增：保存指标


def calculate_metrics(model, val_loader, device, num_classes):
    """
    新增：计算mAP@0.5 + 每类Precision
    """
    model.eval()
    all_preds = []
    all_gts = []

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="计算mAP@0.5 & Precision"):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                pred_boxes = output['boxes'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()
                pred_labels = output['labels'].cpu().numpy()

                gt_boxes = targets[i]['boxes'].cpu().numpy()
                gt_labels = targets[i]['labels'].cpu().numpy()

                all_preds.append((pred_boxes, pred_scores, pred_labels))
                all_gts.append((gt_boxes, gt_labels))

    # 初始化指标
    gt_classes = defaultdict(int)
    tp_classes = defaultdict(int)
    fp_classes = defaultdict(int)
    iou_thresh = 0.5

    # 逐类统计
    for cls in range(1, num_classes):
        gt_num = 0
        tp = 0
        fp = 0

        for (pred_box, pred_score, pred_label), (gt_box, gt_label) in zip(all_preds, all_gts):
            cls_gt_idx = np.where(gt_label == cls)[0]
            cls_gt_box = gt_box[cls_gt_idx]
            gt_num += len(cls_gt_box)

            cls_pred_idx = np.where(pred_label == cls)[0]
            cls_pred_box = pred_box[cls_pred_idx]

            used_gt = np.zeros(len(cls_gt_box), dtype=bool)
            for p_box in cls_pred_box:
                max_iou = -1
                max_idx = -1
                for i, g_box in enumerate(cls_gt_box):
                    if used_gt[i]:
                        continue
                    iou = compute_iou(p_box, g_box)
                    if iou > max_iou:
                        max_iou = iou
                        max_idx = i
                if max_iou >= iou_thresh and max_idx != -1:
                    tp += 1
                    used_gt[max_idx] = True
                else:
                    fp += 1

        gt_classes[cls] = gt_num
        tp_classes[cls] = tp
        fp_classes[cls] = fp

    # 计算每类Precision
    precision_dict = {}
    for cls in range(1, num_classes):
        if tp_classes[cls] + fp_classes[cls] == 0:
            precision = 0.0
        else:
            precision = tp_classes[cls] / (tp_classes[cls] + fp_classes[cls])
        precision_dict[cls] = round(precision, 4)

    # 计算mAP@0.5
    total_gt = sum(gt_classes.values())
    total_tp = sum(tp_classes.values())
    total_fp = sum(fp_classes.values())
    if total_gt == 0:
        mAP05 = 0.0
    else:
        mAP05 = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    mAP05 = round(mAP05, 4)

    return mAP05, precision_dict


def compute_iou(box1, box2):
    """计算IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    inter = w * h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def train(args):
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    torch.set_default_dtype(torch.float32)
    print(f"使用设备: {device}，精度: float32")

    # 加载数据集与类别
    classes = get_classes(args.data_root)
    num_classes = len(classes) + 1
    print(f"加载类别：{classes}，总类别数（含背景）：{num_classes}")
    class_id_map = dict(enumerate(classes, start=1))
    print(f"类别ID映射：{class_id_map}")

    train_dataset = VOCDataset(
        root=args.data_root,
        split='train',
        transforms=get_train_transform()
    )
    val_dataset = VOCDataset(
        root=args.data_root,
        split='val',
        transforms=get_val_transform()
    )

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # 初始化模型
    model = get_faster_rcnn(num_classes).to(device)

    # 优化器改为AdamW
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        params,
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.0005,
        eps=1e-8
    )

    # 学习率调度器
    warmup_epochs = 3
    warmup_steps = warmup_epochs * len(train_loader)
    total_train_steps = args.epochs * len(train_loader)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_train_steps - warmup_steps,
        eta_min=args.lr * 0.001
    )
    lr_scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )

    # 加载checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    best_map = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        best_map = checkpoint.get('best_map', 0.0)
        print(f"加载checkpoint成功，从epoch {start_epoch} 开始训练")
        print(f"当前最佳验证损失: {best_val_loss:.4f}, 历史最佳mAP: {best_map:.4f}")

    # 早停初始化
    early_stop_patience = 10
    early_stop_counter = 0

    # 损失日志
    loss_log = {
        'train_loss': [],
        'val_loss': [],
        'epoch': []
    }

    # ===================== 新增：指标日志 =====================
    metric_log = {
        'epoch': [],
        'mAP05': [],
        'class_1_precision': [],  # closed_eye
        'class_2_precision': [],  # open_eye
        'class_3_precision': [],  # closed_mouth
        'class_4_precision': []   # open_mouth
    }

    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        nan_count = 0

        for batch_idx, (images, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            try:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                if torch.isnan(losses):
                    nan_count += 1
                    tqdm.write(f"⚠️  Batch {batch_idx} 出现NaN损失，跳过本轮更新！")
                    continue

                losses.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()

                train_loss += losses.item()

                if batch_idx % 100 == 0 and batch_idx > 0:
                    tqdm.write(f"  Batch {batch_idx}: total_loss={losses.item():.4f}, "
                               f"rpn_box_loss={loss_dict.get('loss_rpn_box_reg', 0):.4f}, "
                               f"roi_box_loss={loss_dict.get('loss_box_reg', 0):.4f}, "
                               f"obj_loss={loss_dict.get('loss_objectness', 0):.4f}, "
                               f"cls_loss={loss_dict.get('loss_classifier', 0):.4f}")

            except Exception as e:
                tqdm.write(f"❌  Batch {batch_idx} 训练出错: {str(e)}，跳过本轮更新！")
                continue

        if nan_count > 0:
            print(f"\n⚠️  Epoch {epoch + 1} 共出现 {nan_count} 个NaN批次，请检查数据和学习率！")

        avg_train_loss = train_loss / (len(train_loader) - nan_count) if (len(train_loader) - nan_count) > 0 else 0.0
        print(f"\nEpoch {epoch + 1} | 训练平均损失: {avg_train_loss:.4f}")

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_nan_count = 0

        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="验证阶段"):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                try:
                    model.train()
                    loss_dict = model(images, targets)
                    model.eval()

                    batch_loss = sum(loss for loss in loss_dict.values()).item()

                    if torch.isnan(torch.tensor(batch_loss)):
                        val_nan_count += 1
                        continue

                    val_loss += batch_loss
                except Exception as e:
                    val_nan_count += 1
                    continue

        val_total = len(val_loader) - val_nan_count
        avg_val_loss = val_loss / val_total if val_total > 0 else 0.0
        print(f"Epoch {epoch + 1} | 验证平均损失: {avg_val_loss:.4f}")
        if val_nan_count > 0:
            print(f"⚠️  验证阶段出现 {val_nan_count} 个NaN批次！")

        # 保存损失
        loss_log['epoch'].append(epoch + 1)
        loss_log['train_loss'].append(avg_train_loss)
        loss_log['val_loss'].append(avg_val_loss)
        with open('loss_logs/train_val_loss.json', 'w', encoding='utf-8') as f:
            json.dump(loss_log, f, ensure_ascii=False, indent=4)

        # ===================== 新增：每轮计算并保存mAP@0.5 + 4类Precision =====================
        mAP05, precision_dict = calculate_metrics(model, val_loader, device, num_classes)
        best_map = max(best_map, mAP05)

        metric_log['epoch'].append(epoch + 1)
        metric_log['mAP05'].append(mAP05)
        metric_log['class_1_precision'].append(precision_dict.get(1, 0.0))
        metric_log['class_2_precision'].append(precision_dict.get(2, 0.0))
        metric_log['class_3_precision'].append(precision_dict.get(3, 0.0))
        metric_log['class_4_precision'].append(precision_dict.get(4, 0.0))

        with open('metric_logs/train_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metric_log, f, ensure_ascii=False, indent=4)

        print(f"✅ Epoch {epoch + 1} | mAP@0.5: {mAP05:.4f} | 最佳mAP: {best_map:.4f}")
        print(f"📊 类别精度: {class_id_map[1]}={precision_dict.get(1,0):.2f} | {class_id_map[2]}={precision_dict.get(2,0):.2f} | {class_id_map[3]}={precision_dict.get(3,0):.2f} | {class_id_map[4]}={precision_dict.get(4,0):.2f}")

        # 早停逻辑
        if avg_val_loss < best_val_loss and not torch.isnan(torch.tensor(avg_val_loss)):
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            print(f"✅ 发现更优模型！最佳验证损失更新为: {best_val_loss:.4f}")
        else:
            early_stop_counter += 1
            print(f"⚠️  早停计数器: {early_stop_counter}/{early_stop_patience}")
            if early_stop_counter >= early_stop_patience:
                print(f"\n🛑 验证损失连续{early_stop_patience}轮未下降，触发早停！")
                break

        # 预测示例检查
        if (epoch + 1) % 5 == 0:
            print("\n📊 预测示例检查：")
            model.eval()
            with torch.no_grad():
                try:
                    sample_images, _ = next(iter(val_loader))
                    sample_images = list(img.to(device) for img in sample_images[:1])
                    predictions = model(sample_images)

                    pred = predictions[0]
                    print(f"  预测框数量: {len(pred['boxes'])}")
                    if len(pred['boxes']) > 0:
                        print(f"  置信度范围: {pred['scores'].min():.4f} ~ {pred['scores'].max():.4f}")
                        print(f"  预测类别: {pred['labels'].cpu().numpy()}")
                    else:
                        print("  ❌ 警告：未预测到任何目标！")
                except Exception as e:
                    print(f"  ❌ 预测示例检查出错: {str(e)}")

        # 损失监控提示
        if avg_train_loss > 1.0 and epoch > 5:
            print("⚠️  警告：训练损失持续偏高，可能存在欠拟合，建议：1.增加训练轮次 2.调小学习率 3.检查数据标注")
        if avg_val_loss > avg_train_loss * 1.5 and avg_train_loss > 0:
            print("⚠️  警告：验证损失远高于训练损失，可能存在过拟合，建议：1.增加数据增强 2.提高weight_decay")

        # 保存Checkpoint
        is_best = avg_val_loss == best_val_loss and not torch.isnan(torch.tensor(avg_val_loss))
        if not torch.isnan(torch.tensor(avg_train_loss)):
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_map': best_map,
                'classes': classes
            }, is_best)

        print(f"Epoch {epoch + 1} 结束 | 最佳mAP@0.5: {best_map:.4f}\n")

    print("🎉 训练完成!")
    print(f"📌 最终最佳验证损失: {best_val_loss:.4f}")
    print(f"📌 最终最佳mAP@0.5: {best_map:.4f}")
    print(f"📌 模型权重保存路径: checkpoints/")
    print(f"📌 损失数据保存路径: loss_logs/train_val_loss.json")
    print(f"📌 指标数据保存路径: metric_logs/train_metrics.json")
    print("📌 下一步：使用 evaluate.py 评估模型，结合预测框可视化检查结果")


# 数据校验工具函数
def validate_targets(targets, img_size):
    """校验并修复标注数据"""
    h, w = img_size
    boxes = targets["boxes"]

    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid]
    targets["boxes"] = boxes
    targets["labels"] = targets["labels"][valid]

    boxes[:, 0] = torch.clamp(boxes[:, 0], 0, w)
    boxes[:, 1] = torch.clamp(boxes[:, 1], 0, h)
    boxes[:, 2] = torch.clamp(boxes[:, 2], 0, w)
    boxes[:, 3] = torch.clamp(boxes[:, 3], 0, h)

    if len(boxes) == 0:
        targets["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
        targets["labels"] = torch.zeros(0, dtype=torch.int64)

    return targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="疲劳驾驶检测模型训练（修复NaN版）")
    parser.add_argument('--data_root', default='data', help='数据集根目录')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮次')
    parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--device', default='cuda', help='训练设备')
    parser.add_argument('--resume', default='', help='续训checkpoint路径')
    args = parser.parse_args()

    print("=" * 50)
    print("训练参数确认：")
    print(f"学习率: {args.lr} | 批次大小: {args.batch_size} | 训练轮次: {args.epochs}")
    print("优化配置：AdamW + CosineAnnealingLR + Warmup")
    print(f"早停配置：验证损失连续10轮未下降则终止训练")
    print("新增功能：每轮保存mAP@0.5 + 4类特征Precision")
    print("=" * 50)

    train(args)