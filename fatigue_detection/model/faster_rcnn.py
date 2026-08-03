import torch  # <--- 确保在文件开头导入 torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import ResNet18_Weights


# --- 替换为 SIoU Loss 定义 ---
# SIoU Loss：考虑角度、距离、形状、重叠损失，对小目标/多尺度目标更友好
class SIoULoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(SIoULoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred_boxes, target_boxes):
        """
        计算SIoU损失（SIoU: Superior IoU）
        :param pred_boxes: 预测的边界框, 格式为 (x1, y1, x2, y2), 形状为 [N, 4]
        :param target_boxes: 真实的边界框, 格式同上, 形状为 [N, 4]
        :return: SIoU损失值
        """
        # 确保输入是浮点数
        pred_boxes = pred_boxes.to(torch.float32)
        target_boxes = target_boxes.to(torch.float32)

        # 1. 计算基本坐标信息
        # 预测框宽高
        pred_w = pred_boxes[:, 2] - pred_boxes[:, 0]
        pred_h = pred_boxes[:, 3] - pred_boxes[:, 1]
        # 真实框宽高
        target_w = target_boxes[:, 2] - target_boxes[:, 0]
        target_h = target_boxes[:, 3] - target_boxes[:, 1]

        # 预测框/真实框中心点
        pred_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
        pred_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
        target_cx = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
        target_cy = (target_boxes[:, 1] + target_boxes[:, 3]) / 2

        # 2. 计算交集和IoU
        inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        pred_area = pred_w * pred_h
        target_area = target_w * target_h
        union_area = pred_area + target_area - inter_area
        iou = inter_area / (union_area + 1e-6)  # 避免除零

        # 3. 计算角度损失 (Angle Loss)
        # 计算目标框与预测框的角度差
        dx = target_cx - pred_cx
        dy = target_cy - pred_cy
        # 避免除零
        pred_w = torch.clamp(pred_w, min=1e-6)
        pred_h = torch.clamp(pred_h, min=1e-6)
        target_w = torch.clamp(target_w, min=1e-6)
        target_h = torch.clamp(target_h, min=1e-6)

        # 角度正切值
        tan_alpha = (dy * target_w) / (dx * target_h + 1e-6)
        alpha = torch.atan(tan_alpha)
        # 角度损失（SIoU核心：考虑框的方向对齐）
        angle_loss = 1 - 2 * torch.pow(torch.sin(alpha), 2)

        # 4. 计算距离损失 (Distance Loss)
        # 最小包围盒
        enclose_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        enclose_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        enclose_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        enclose_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
        enclose_w = enclose_x2 - enclose_x1
        enclose_h = enclose_y2 - enclose_y1

        # 归一化距离
        rho_x = (dx / enclose_w) ** 2
        rho_y = (dy / enclose_h) ** 2
        distance_loss = 2 - torch.exp(-rho_x) - torch.exp(-rho_y)

        # 5. 计算形状损失 (Shape Loss)
        # 宽/高的归一化差异
        omega_w = torch.pow((pred_w / target_w) - 1, 2)
        omega_h = torch.pow((pred_h / target_h) - 1, 2)
        shape_loss = torch.pow(1 - torch.exp(-omega_w), 2) + torch.pow(1 - torch.exp(-omega_h), 2)

        # 6. 计算重叠损失 (Overlap Loss)
        overlap_loss = 1 - iou

        # 7. 总SIoU损失（四个部分加权，SIoU原始论文权重）
        loss = overlap_loss + (distance_loss + shape_loss + angle_loss) * (1 - iou)

        # 损失归约
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# --- 保持原有结构的 get_faster_rcnn 函数 ---
def get_faster_rcnn(num_classes):
    # 1. 加载backbone（ResNet18，适配你的小数据集）
    backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
        backbone_name='resnet18',
        weights=ResNet18_Weights.IMAGENET1K_V1
    )

    # 2. 调整锚框：匹配你的标注框尺寸+比例
    # anchor_sizes：对应目标宽度（32/48/64适配30~50px的目标）
    anchor_sizes = ((32,), (48,), (64,), (96,), (128,))
    # aspect_ratios：核心！设置为2.0/2.5，匹配眼睛/嘴巴的宽高比
    aspect_ratios = ((2.0, 2.5),) * len(anchor_sizes)

    # 3. 生成锚框
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios
    )

    # 4. 构建Faster R-CNN（其他参数适配你的场景）
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_score_thresh=0.1,  # 降低阈值，让模型更易预测
        box_detections_per_img=100,
        # 注释掉原来的 smooth_l1 loss 设置
        # box_reg_loss_type='smooth_l1',
        # box_reg_loss_weight=2.0
    )

    # --- 核心修改：替换为SIoU Loss ---
    model.roi_heads.box_reg_loss_fn = SIoULoss(reduction='mean')

    return model


# 测试代码（可选，验证模型构建是否成功）
if __name__ == "__main__":
    # 构建模型（5类：4个目标类别 + 背景）
    model = get_faster_rcnn(num_classes=5)
    # 打印模型结构，验证损失函数替换成功
    print("模型构建完成，边界框回归损失已替换为SIoU Loss")
    print(f"当前RoI Heads使用的回归损失：{model.roi_heads.box_reg_loss_fn}")